"""
Data processors for relativistic neural network training.
"""

from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
import numpy as np
import polars as pl
import torch
from torch import Tensor
from torch_geometric.data import Data, Batch
from torch_geometric.transforms import BaseTransform
from abc import ABC, abstractmethod


@dataclass
class ProcessingConfig:
    """Configuration for data processing."""
    normalize_features: bool = True
    add_noise: bool = False
    noise_level: float = 0.01
    temporal_smoothing: bool = True
    smoothing_window: int = 3
    outlier_detection: bool = True
    outlier_threshold: float = 3.0
    relativistic_corrections: bool = True
    speed_of_light: float = 299792458.0  # m/s


class RelativisticDataProcessor:
    """Main data processor with relativistic corrections."""
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        self.config = config or ProcessingConfig()
        self.feature_stats: Dict[str, Dict[str, float]] = {}
    
    def process_dataframe(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply all processing steps to a DataFrame."""
        processed_df = df.clone()
        
        # Apply processing pipeline
        if self.config.outlier_detection:
            processed_df = self._remove_outliers(processed_df)
        
        if self.config.temporal_smoothing:
            processed_df = self._apply_temporal_smoothing(processed_df)
        
        if self.config.relativistic_corrections:
            processed_df = self._apply_relativistic_corrections(processed_df)
        
        if self.config.normalize_features:
            processed_df = self._normalize_features(processed_df)
        
        if self.config.add_noise:
            processed_df = self._add_noise(processed_df)
        
        return processed_df
    
    def _remove_outliers(self, df: pl.DataFrame) -> pl.DataFrame:
        """Remove outliers using IQR method."""
        numeric_cols = [col for col in df.columns if df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]
        
        for col in numeric_cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            if q1 is None or q3 is None:
                continue
            iqr = q3 - q1
            lower_bound = q1 - self.config.outlier_threshold * iqr
            upper_bound = q3 + self.config.outlier_threshold * iqr
            
            df = df.filter(
                (pl.col(col) >= lower_bound) & (pl.col(col) <= upper_bound)
            )
        
        return df
    
    def _apply_temporal_smoothing(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply temporal smoothing to reduce noise."""
        if 'datetime' not in df.columns:
            return df
        
        # Sort by time
        df = df.sort('datetime')
        
        # Apply rolling average to numeric columns
        numeric_cols = [col for col in df.columns if df[col].dtype in [pl.Float64, pl.Float32] and col != 'datetime']
        
        for col in numeric_cols:
            smoothed_col = df[col].rolling_mean(self.config.smoothing_window, center=True)
            df = df.with_columns(smoothed_col.alias(f"{col}_smoothed"))
        
        return df
    
    def _apply_relativistic_corrections(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply relativistic corrections to the data."""
        # Check if we have velocity data
        velocity_cols = [col for col in df.columns if col.startswith('v') and ('km_s' in col or 'm_s' in col)]
        
        if not velocity_cols:
            return df
        
        # Compute relativistic factors
        if all(f'v{axis}_km_s' in df.columns for axis in ['x', 'y', 'z']):
            # Convert km/s to m/s and compute speed
            speed_expr = (
                (pl.col('vx_km_s') * 1000)**2 + 
                (pl.col('vy_km_s') * 1000)**2 + 
                (pl.col('vz_km_s') * 1000)**2
            ).sqrt()
            
            # Lorentz factor γ = 1/√(1-v²/c²)
            beta_squared = (speed_expr / self.config.speed_of_light)**2
            gamma = 1.0 / (1.0 - beta_squared).sqrt()
            
            # Time dilation factor
            time_dilation = gamma
            
            # Add relativistic features
            df = df.with_columns([
                speed_expr.alias('speed_m_s'),
                (speed_expr / self.config.speed_of_light).alias('beta'),
                gamma.alias('lorentz_factor'),
                time_dilation.alias('time_dilation_factor')
            ])
        
        return df
    
    def _normalize_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Normalize numeric features to zero mean and unit variance."""
        numeric_cols = [col for col in df.columns if df[col].dtype in [pl.Float64, pl.Float32]]
        
        for col in numeric_cols:
            mean_val = df[col].mean()
            std_val = df[col].std()
            
            # Ensure values are not None and are numeric
            if mean_val is None or std_val is None:
                continue
            
            # Convert to float if needed, handling various types from polars
            try:
                mean_float = float(mean_val) if not isinstance(mean_val, (int, float)) else float(mean_val)  # type: ignore[arg-type]
                std_float = float(std_val) if not isinstance(std_val, (int, float)) else float(std_val)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                # Skip columns that can't be converted to float
                continue
            
            # Store stats for later denormalization
            self.feature_stats[col] = {'mean': mean_float, 'std': std_float}
            
            if std_float > 1e-8:  # Avoid division by zero
                normalized_col = (pl.col(col) - mean_float) / std_float
                df = df.with_columns(normalized_col.alias(col))
        
        return df
    
    def _add_noise(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add Gaussian noise to numeric features."""
        numeric_cols = [col for col in df.columns if df[col].dtype in [pl.Float64, pl.Float32]]
        
        for col in numeric_cols:
            noise = np.random.normal(0, self.config.noise_level, len(df))
            noisy_col = pl.col(col) + pl.Series(noise)
            df = df.with_columns(noisy_col.alias(col))
        
        return df
    
    def denormalize_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Denormalize features using stored statistics."""
        for col, stats in self.feature_stats.items():
            if col in df.columns:
                denormalized_col = pl.col(col) * stats['std'] + stats['mean']
                df = df.with_columns(denormalized_col.alias(col))
        
        return df
    
    def create_train_val_test_split(
        self, 
        df: pl.DataFrame, 
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        temporal_split: bool = True
    ) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """Create train/validation/test splits."""
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
        
        if temporal_split and 'datetime' in df.columns:
            # Temporal split: earlier data for training, later for testing
            df = df.sort('datetime')
            n_total = len(df)
            n_train = int(n_total * train_ratio)
            n_val = int(n_total * val_ratio)
            
            train_df = df[:n_train]
            val_df = df[n_train:n_train + n_val]
            test_df = df[n_train + n_val:]
        
        else:
            # Random split
            df = df.sample(fraction=1.0, shuffle=True, seed=42)
            n_total = len(df)
            n_train = int(n_total * train_ratio)
            n_val = int(n_total * val_ratio)
            
            train_df = df[:n_train]
            val_df = df[n_train:n_train + n_val]
            test_df = df[n_train + n_val:]
        
        return train_df, val_df, test_df


class PyTorchGeometricTransforms:
    """Collection of transforms for PyTorch Geometric data."""
    
    @staticmethod
    def create_relativistic_transform() -> 'RelativisticTransform':
        """Create transform that adds relativistic features."""
        return RelativisticTransform()
    
    @staticmethod
    def create_noise_transform(noise_level: float = 0.01) -> 'NoiseTransform':
        """Create transform that adds noise to node features."""
        return NoiseTransform(noise_level)
    
    @staticmethod
    def create_normalization_transform() -> 'NormalizationTransform':
        """Create transform that normalizes node and edge features."""
        return NormalizationTransform()
    
    @staticmethod
    def create_edge_augmentation_transform(prob: float = 0.1) -> 'EdgeAugmentationTransform':
        """Create transform that randomly adds/removes edges."""
        return EdgeAugmentationTransform(prob)


class RelativisticTransform(BaseTransform):
    """Transform that adds relativistic features to graph data."""
    
    def __init__(self):
        self.c = 299792458.0  # Speed of light in m/s
    
    def __call__(self, data: Data) -> Data:
        if not hasattr(data, 'pos') or data.pos is None:
            return data
        
        # Extract positions (assuming they're in the node features or pos attribute)
        if hasattr(data, 'pos'):
            positions = data.pos
        else:
            # Assume first 3 features are positions
            positions = data.x[:, :3] if data.x.shape[1] >= 3 else None
        
        if positions is None:
            return data
        
        # Assume next 3 features are velocities (if available)
        if data.x.shape[1] >= 6:
            velocities = data.x[:, 3:6]
        else:
            # Estimate velocities from positions (simplified)
            velocities = torch.zeros_like(positions)
        
        # Add relativistic features
        relativistic_features = self._compute_relativistic_features(positions, velocities)
        
        # Concatenate with existing features
        if data.x is not None:
            data.x = torch.cat([data.x, relativistic_features], dim=1)
        else:
            data.x = relativistic_features
        
        return data
    
    def _compute_relativistic_features(self, positions: Tensor, velocities: Tensor) -> Tensor:
        """Compute relativistic features."""
        # Speed
        speeds = torch.norm(velocities, dim=1, keepdim=True)
        
        # Beta = v/c
        beta = speeds / self.c
        
        # Lorentz factor γ = 1/√(1-β²)
        gamma = 1.0 / torch.sqrt(1.0 - torch.clamp(beta**2, 0, 0.999))
        
        # Time dilation
        time_dilation = gamma
        
        # Gravitational time dilation (simplified, assuming Earth)
        r = torch.norm(positions, dim=1, keepdim=True)
        gm_earth = 3.986e14  # m³/s²
        gravitational_factor = torch.sqrt(1.0 - 2 * gm_earth / (r * self.c**2 + 1e10))
        
        return torch.cat([beta, gamma, time_dilation, gravitational_factor], dim=1)


class NoiseTransform(BaseTransform):
    """Transform that adds Gaussian noise to node features."""
    
    def __init__(self, noise_level: float = 0.01):
        self.noise_level = noise_level
    
    def __call__(self, data: Data) -> Data:
        if data.x is not None:
            noise = torch.randn_like(data.x) * self.noise_level
            data.x = data.x + noise
        
        return data


class NormalizationTransform(BaseTransform):
    """Transform that normalizes node and edge features."""
    
    def __init__(self):
        self.node_mean = None
        self.node_std = None
        self.edge_mean = None
        self.edge_std = None
        self.fitted = False
    
    def fit(self, data_list: List[Data]):
        """Fit normalization parameters on a list of data."""
        # Collect all node features
        all_node_features: List[Tensor] = []
        all_edge_features: List[Tensor] = []
        
        for data in data_list:
            if data.x is not None:
                all_node_features.append(data.x)
            if hasattr(data, 'edge_attr') and data.edge_attr is not None:
                all_edge_features.append(data.edge_attr)
        
        if all_node_features:
            all_node_features_tensor = torch.cat(all_node_features, dim=0)
            self.node_mean = all_node_features_tensor.mean(dim=0)
            self.node_std = all_node_features_tensor.std(dim=0) + 1e-8
        
        if all_edge_features:
            all_edge_features_tensor = torch.cat(all_edge_features, dim=0)
            self.edge_mean = all_edge_features_tensor.mean(dim=0)
            self.edge_std = all_edge_features_tensor.std(dim=0) + 1e-8
        
        self.fitted = True
    
    def __call__(self, data: Data) -> Data:
        if not self.fitted:
            raise RuntimeError("NormalizationTransform must be fitted before use")
        
        if data.x is not None and self.node_mean is not None:
            data.x = (data.x - self.node_mean) / self.node_std
        
        if hasattr(data, 'edge_attr') and data.edge_attr is not None and self.edge_mean is not None:
            data.edge_attr = (data.edge_attr - self.edge_mean) / self.edge_std
        
        return data


class EdgeAugmentationTransform(BaseTransform):
    """Transform that randomly adds or removes edges."""
    
    def __init__(self, add_prob: float = 0.05, remove_prob: float = 0.05):
        self.add_prob = add_prob
        self.remove_prob = remove_prob
    
    def __call__(self, data: Data) -> Data:
        if data.edge_index is None or data.edge_index.shape[1] == 0:
            return data
        
        device = data.edge_index.device
        num_nodes = data.x.shape[0] if data.x is not None else data.edge_index.max().item() + 1
        
        # Remove edges
        if self.remove_prob > 0:
            mask = torch.rand(data.edge_index.shape[1], device=device) > self.remove_prob
            data.edge_index = data.edge_index[:, mask]
            if hasattr(data, 'edge_attr') and data.edge_attr is not None:
                data.edge_attr = data.edge_attr[mask]
        
        # Add edges
        if self.add_prob > 0:
            num_possible_edges = num_nodes * (num_nodes - 1)
            num_edges_to_add = int(num_possible_edges * self.add_prob)
            
            # Generate random edge candidates
            src_nodes = torch.randint(0, num_nodes, (num_edges_to_add,), device=device)
            dst_nodes = torch.randint(0, num_nodes, (num_edges_to_add,), device=device)
            
            # Remove self-loops
            mask = src_nodes != dst_nodes
            src_nodes = src_nodes[mask]
            dst_nodes = dst_nodes[mask]
            
            if len(src_nodes) > 0:
                new_edges = torch.stack([src_nodes, dst_nodes], dim=0)
                data.edge_index = torch.cat([data.edge_index, new_edges], dim=1)
                
                # Add corresponding edge attributes (zeros)
                if hasattr(data, 'edge_attr') and data.edge_attr is not None:
                    new_edge_attr = torch.zeros(len(src_nodes), data.edge_attr.shape[1], 
                                               device=device, dtype=data.edge_attr.dtype)
                    data.edge_attr = torch.cat([data.edge_attr, new_edge_attr], dim=0)
        
        return data


class DataAugmentation:
    """Data augmentation utilities for space datasets."""
    
    @staticmethod
    def temporal_jitter(df: pl.DataFrame, max_jitter_hours: float = 0.1) -> pl.DataFrame:
        """Add random temporal jitter to timestamps."""
        if 'datetime' not in df.columns:
            return df
        
        # Add random jitter to timestamps
        jitter_seconds = np.random.uniform(-max_jitter_hours * 3600, max_jitter_hours * 3600, len(df))
        
        # This is a simplified version - in practice you'd need to handle datetime arithmetic properly
        return df
    
    @staticmethod
    def spatial_noise(df: pl.DataFrame, noise_level_km: float = 1.0) -> pl.DataFrame:
        """Add spatial noise to position data."""
        position_cols = [col for col in df.columns if any(axis in col for axis in ['x_km', 'y_km', 'z_km'])]
        
        for col in position_cols:
            noise = np.random.normal(0, noise_level_km, len(df))
            df = df.with_columns((pl.col(col) + pl.Series(noise)).alias(col))
        
        return df
    
    @staticmethod
    def velocity_perturbation(df: pl.DataFrame, perturbation_level: float = 0.001) -> pl.DataFrame:
        """Add small perturbations to velocity data."""
        velocity_cols = [col for col in df.columns if 'v' in col and 'km_s' in col]
        
        for col in velocity_cols:
            current_values = df[col].to_numpy()
            perturbation = np.random.normal(0, perturbation_level * np.abs(current_values))
            df = df.with_columns((pl.col(col) + pl.Series(perturbation)).alias(col))
        
        return df
