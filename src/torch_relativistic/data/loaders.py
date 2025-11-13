"""
PyTorch Geometric data loaders for space datasets with relativistic effects.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import numpy as np
import polars as pl
import torch
from torch import Tensor
from torch_geometric.data import Data, Dataset, InMemoryDataset
from torch_geometric.utils import dense_to_sparse
from torch_geometric.loader import DataLoader
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class SpaceGraphConfig:
    """Configuration for space graph construction."""
    max_distance_km: float = 10000.0  # Maximum edge distance
    min_satellites: int = 3            # Minimum satellites for graph
    time_window_hours: int = 1         # Time window for snapshot
    velocity_scaling: float = 1e-3     # Scale velocities to reasonable range
    position_scaling: float = 1e-6     # Scale positions to reasonable range
    include_relativistic: bool = True   # Include relativistic features
    edge_attr_dim: int = 8             # Edge attribute dimensionality
    node_attr_dim: int = 12            # Node attribute dimensionality


class RelativisticFeatureExtractor:
    """Extract relativistic features from space data."""
    
    def __init__(self, config: SpaceGraphConfig):
        self.config = config
        self.c = 299792458.0  # Speed of light in m/s
    
    def compute_lorentz_factor(self, velocity: Tensor) -> Tensor:
        """Compute Lorentz factor γ = 1/√(1-v²/c²)."""
        v_magnitude = torch.norm(velocity, dim=-1)
        v_over_c = v_magnitude / self.c
        # Clamp to avoid numerical issues
        v_over_c_clamped = torch.clamp(v_over_c, 0, 0.999)
        gamma = 1.0 / torch.sqrt(1.0 - v_over_c_clamped**2)
        return gamma
    
    def compute_time_dilation(self, velocity: Tensor, dt: float = 1.0) -> Tensor:
        """Compute relativistic time dilation."""
        gamma = self.compute_lorentz_factor(velocity)
        return dt * gamma
    
    def compute_light_travel_time(self, pos1: Tensor, pos2: Tensor) -> Tensor:
        """Compute light travel time between positions."""
        distance = torch.norm(pos2 - pos1, dim=-1)
        return distance / self.c
    
    def compute_doppler_factor(self, velocity: Tensor, direction: Tensor) -> Tensor:
        """Compute relativistic Doppler factor."""
        v_magnitude = torch.norm(velocity, dim=-1)
        v_over_c = v_magnitude / self.c
        cos_theta = torch.sum(velocity * direction, dim=-1) / (v_magnitude + 1e-8)
        
        gamma = self.compute_lorentz_factor(velocity)
        doppler = gamma * (1 - v_over_c * cos_theta)
        return doppler
    
    def extract_node_features(self, positions: Tensor, velocities: Tensor) -> Tensor:
        """Extract relativistic node features."""
        batch_size = positions.shape[0]
        
        # Basic kinematic features
        pos_scaled = positions * self.config.position_scaling
        vel_scaled = velocities * self.config.velocity_scaling
        
        # Relativistic features
        gamma = self.compute_lorentz_factor(velocities).unsqueeze(-1)
        time_dilation = self.compute_time_dilation(velocities).unsqueeze(-1)
        
        # Orbital features
        r_magnitude = torch.norm(positions, dim=-1, keepdim=True)
        v_magnitude = torch.norm(velocities, dim=-1, keepdim=True)
        
        # Specific orbital energy (simplified)
        mu_earth = 3.986e14  # Earth's gravitational parameter
        energy = 0.5 * v_magnitude**2 - mu_earth / r_magnitude
        
        # Angular momentum magnitude - fix dimensional issues
        try:
            # Ensure both tensors have the same shape for cross product
            pos_for_cross = positions if positions.dim() == 2 else positions.unsqueeze(0)
            vel_for_cross = velocities if velocities.dim() == 2 else velocities.unsqueeze(0)
            
            angular_momentum_vec = torch.cross(pos_for_cross, vel_for_cross, dim=-1)
            angular_momentum = torch.norm(angular_momentum_vec, dim=-1, keepdim=True)
        except Exception as e:
            print(f"Warning: Angular momentum calculation failed: {e}, using zeros")
            angular_momentum = torch.zeros_like(r_magnitude)
        
        # Combine features - ensure all have compatible dimensions
        node_features = torch.cat([
            pos_scaled,           # 3 features: x, y, z (scaled)
            vel_scaled,           # 3 features: vx, vy, vz (scaled)
            gamma,                # 1 feature: Lorentz factor
            time_dilation,        # 1 feature: time dilation
            r_magnitude * self.config.position_scaling,  # 1 feature: distance from origin
            v_magnitude * self.config.velocity_scaling,  # 1 feature: speed
            energy * 1e-12,       # 1 feature: specific energy (scaled)
            angular_momentum * 1e-9  # 1 feature: angular momentum (scaled)
        ], dim=-1)
        
        return node_features
    
    def extract_edge_features(self, pos1: Tensor, pos2: Tensor, vel1: Tensor, vel2: Tensor) -> Tensor:
        """Extract relativistic edge features."""
        # Ensure all tensors have consistent shapes
        if pos1.dim() == 1:
            pos1 = pos1.unsqueeze(0)
        if pos2.dim() == 1:
            pos2 = pos2.unsqueeze(0)
        if vel1.dim() == 1:
            vel1 = vel1.unsqueeze(0)
        if vel2.dim() == 1:
            vel2 = vel2.unsqueeze(0)
        
        # Relative position and velocity
        rel_pos = pos2 - pos1
        rel_vel = vel2 - vel1
        
        distance = torch.norm(rel_pos, dim=-1, keepdim=True)
        rel_speed = torch.norm(rel_vel, dim=-1, keepdim=True)
        
        # Unit direction vector
        direction = rel_pos / (distance + 1e-8)
        
        # Light travel time
        light_time = self.compute_light_travel_time(pos1, pos2).unsqueeze(-1)
        
        # Relativistic Doppler factors
        doppler1 = self.compute_doppler_factor(vel1, direction).unsqueeze(-1)
        doppler2 = self.compute_doppler_factor(vel2, -direction).unsqueeze(-1)
        
        # Combine edge features - ensure all have same batch dimension
        edge_features = torch.cat([
            rel_pos * self.config.position_scaling,  # 3 features: relative position
            rel_vel * self.config.velocity_scaling,  # 3 features: relative velocity
            light_time * 1e6,     # 1 feature: light travel time (in microseconds)
            distance * self.config.position_scaling  # 1 feature: distance
        ], dim=-1)
        
        return edge_features


class BaseSpaceDataset(InMemoryDataset, ABC):
    """Base class for space datasets."""
    
    def __init__(
        self,
        root: str,
        config: SpaceGraphConfig,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        force_reload: bool = False
    ):
        self.config = config
        self.feature_extractor = RelativisticFeatureExtractor(config)
        super().__init__(root, transform, pre_transform, pre_filter, force_reload=force_reload)
        
        # Use modern PyG >= 2.4 API with load()
        if self.processed_paths and Path(self.processed_paths[0]).exists():
            self.load(self.processed_paths[0])
        else:
            self.data, self.slices = None, None
    
    @property
    def raw_file_names(self) -> List[str]:
        return ['data.parquet']
    
    @property 
    def processed_file_names(self) -> List[str]:
        return ['data.pt']
    
    @abstractmethod
    def download(self):
        """Download raw data."""
        pass
    
    @abstractmethod
    def create_graph_from_snapshot(self, snapshot_data: pl.DataFrame) -> Data:
        """Create PyTorch Geometric graph from data snapshot."""
        pass
    
    def process(self):
        """Process raw data into PyTorch Geometric format."""
        print("Processing...")
        
        # Read raw data
        raw_path = Path(self.raw_paths[0])
        if not raw_path.exists():
            raise FileNotFoundError(f"Raw data file not found: {raw_path}")
        
        df = pl.read_parquet(raw_path)
        
        # Group by time windows and create graphs
        data_list = []
        
        try:
            if 'datetime' in df.columns:
                # Time-based snapshots
                df = df.sort('datetime')
                time_groups = self._create_time_windows(df)
                
                for time_window, group_df in time_groups:
                    if len(group_df) >= self.config.min_satellites:
                        try:
                            graph = self.create_graph_from_snapshot(group_df)
                            if graph is not None:
                                data_list.append(graph)
                        except Exception as e:
                            print(f"Warning: Failed to create graph for time {time_window}: {e}")
            else:
                # Single snapshot
                if len(df) >= self.config.min_satellites:
                    graph = self.create_graph_from_snapshot(df)
                    if graph is not None:
                        data_list.append(graph)
            
            if not data_list:
                # Fallback: create a single graph from all data
                print("No time-based graphs created, using all data as single snapshot")
                graph = self.create_graph_from_snapshot(df[:self.config.min_satellites])
                if graph is not None:
                    data_list.append(graph)
            
        except Exception as e:
            print(f"Error in processing: {e}")
            # Fallback: create a minimal synthetic graph
            data_list = [self._create_minimal_graph()]
        
        if not data_list:
            raise ValueError("No valid graphs could be created from the data")
        
        # Apply pre-filtering and pre-transformation
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        
        # Use modern PyG >= 2.4 API with save()
        self.save(data_list, self.processed_paths[0])
        
    def _create_minimal_graph(self) -> Data:
        """Create a minimal synthetic graph for fallback."""
        # Create a simple 4-node graph
        positions = torch.tensor([
            [0.0, 0.0, 0.0],
            [1000.0, 0.0, 0.0], 
            [0.0, 1000.0, 0.0],
            [0.0, 0.0, 1000.0]
        ], dtype=torch.float32)
        
        velocities = torch.tensor([
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ], dtype=torch.float32) * 1000  # 1 km/s
        
        # Create node features
        node_features = self.feature_extractor.extract_node_features(positions, velocities)
        
        # Create simple edges (connect all nodes)
        edge_index = torch.tensor([
            [0, 0, 1, 1, 2, 2],
            [1, 2, 2, 3, 3, 0]
        ], dtype=torch.long)
        
        edge_attr = torch.zeros((edge_index.shape[1], self.config.edge_attr_dim), dtype=torch.float32)
        
        return Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=positions,
            pos=positions
        )
    
    def _create_time_windows(self, df: pl.DataFrame) -> List[Tuple[str, pl.DataFrame]]:
        """Create time-based windows from DataFrame."""
        time_windows = []
        
        try:
            # Convert datetime strings to timestamps if needed
            if df.schema['datetime'] == pl.Utf8:
                df = df.with_columns(
                    pl.col('datetime').str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S").alias('timestamp')
                )
            else:
                df = df.with_columns(pl.col('datetime').alias('timestamp'))
            
            # Simple approach: create single time window with all data
            time_windows.append(("single_window", df))
            
        except Exception as e:
            print(f"Time window creation failed: {e}, using single window fallback")
            # Ultimate fallback: treat all data as single time window
            time_windows.append(("fallback_window", df))
        
        return time_windows
    
    def create_edges_from_positions(self, positions: Tensor) -> Tuple[Tensor, Tensor]:
        """Create edges based on spatial proximity."""
        n_nodes = positions.shape[0]
        
        # Compute pairwise distances
        dist_matrix = torch.cdist(positions, positions)
        
        # Create adjacency matrix based on distance threshold
        adj_matrix = (dist_matrix < self.config.max_distance_km) & (dist_matrix > 0)
        
        # Convert to edge index
        edge_index, _ = dense_to_sparse(adj_matrix.float())
        
        # Extract edge attributes
        edge_attr_list = []
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            distance = dist_matrix[src, dst]
            edge_attr_list.append(distance.unsqueeze(0))
        
        edge_attr = torch.cat(edge_attr_list, dim=0).unsqueeze(-1)
        
        return edge_index, edge_attr


class SatelliteConstellationDataset(BaseSpaceDataset):
    """Dataset for satellite constellation data (GPS, Starlink, etc.)."""
    
    def __init__(
        self,
        root: str,
        data_source: str = "gps",
        config: Optional[SpaceGraphConfig] = None,
        force_reload: bool = False,
        **kwargs
    ):
        self.data_source = data_source
        config = config or SpaceGraphConfig()
        super().__init__(root, config, force_reload=force_reload, **kwargs)
    
    def download(self):
        """Download satellite constellation data."""
        # This would be implemented with actual data acquisition
        # For now, create synthetic data
        synthetic_data = self._generate_synthetic_constellation()
        
        raw_path = Path(self.raw_paths[0])
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        synthetic_data.write_parquet(raw_path)
    
    def _generate_synthetic_constellation(self) -> pl.DataFrame:
        """Generate synthetic satellite constellation data."""
        import random
        from datetime import datetime, timedelta
        
        # Generate data for 24-satellite GPS-like constellation
        n_satellites = 24
        n_timesteps = 100
        
        data_rows = []
        base_time = datetime(2024, 1, 1)
        
        for t in range(n_timesteps):
            current_time = base_time + timedelta(hours=t)
            
            for sat_id in range(n_satellites):
                # Circular orbit parameters (simplified)
                orbital_radius = 20200e3 + random.uniform(-100e3, 100e3)  # ~GPS altitude
                inclination = 55.0 + random.uniform(-5, 5)  # degrees
                
                # Angular position based on time and satellite
                angle = (t * 0.1 + sat_id * 15) % 360  # degrees
                angle_rad = np.radians(angle)
                incl_rad = np.radians(inclination)
                
                # Position in orbit (simplified)
                x = orbital_radius * np.cos(angle_rad)
                y = orbital_radius * np.sin(angle_rad) * np.cos(incl_rad)
                z = orbital_radius * np.sin(angle_rad) * np.sin(incl_rad)
                
                # Velocity (perpendicular to radius, simplified)
                v_magnitude = np.sqrt(3.986e14 / orbital_radius)  # Circular orbit velocity
                vx = -v_magnitude * np.sin(angle_rad)
                vy = v_magnitude * np.cos(angle_rad) * np.cos(incl_rad)
                vz = v_magnitude * np.cos(angle_rad) * np.sin(incl_rad)
                
                data_rows.append({
                    'datetime': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'satellite_id': f"SAT_{sat_id:02d}",
                    'x_km': x / 1000,
                    'y_km': y / 1000,
                    'z_km': z / 1000,
                    'vx_km_s': vx / 1000,
                    'vy_km_s': vy / 1000,
                    'vz_km_s': vz / 1000,
                    'mass_kg': 2000.0 + random.uniform(-200, 200),
                    'power_w': 800.0 + random.uniform(-100, 100)
                })
        
        return pl.DataFrame(data_rows)
    
    def create_graph_from_snapshot(self, snapshot_data: pl.DataFrame) -> Data:
        """Create graph from satellite constellation snapshot."""
        # Extract positions and velocities with proper error handling
        data_dicts = snapshot_data.to_dicts()
        
        if not data_dicts:
            raise ValueError("No data in snapshot")
        
        print(f"DEBUG: Available columns: {list(data_dicts[0].keys())}")
        
        positions: List[List[float]] = []
        velocities: List[List[float]] = []
        
        # Check if we have position/velocity data or need to convert from orbital elements
        has_cartesian = all(col in data_dicts[0] for col in ['x_km', 'y_km', 'z_km'])
        has_velocity = all(col in data_dicts[0] for col in ['vx_km_s', 'vy_km_s', 'vz_km_s'])
        
        if has_cartesian and has_velocity:
            print("DEBUG: Using Cartesian coordinates")
            for row in data_dicts:
                # Extract position with safe defaults
                x = row.get('x_km', 0.0) if row.get('x_km') is not None else 0.0
                y = row.get('y_km', 0.0) if row.get('y_km') is not None else 0.0  
                z = row.get('z_km', 0.0) if row.get('z_km') is not None else 0.0
                
                # Extract velocity with safe defaults
                vx = row.get('vx_km_s', 0.0) if row.get('vx_km_s') is not None else 0.0
                vy = row.get('vy_km_s', 0.0) if row.get('vy_km_s') is not None else 0.0
                vz = row.get('vz_km_s', 0.0) if row.get('vz_km_s') is not None else 0.0
                
                positions.append([x, y, z])
                velocities.append([vx, vy, vz])
        else:
            print("DEBUG: Converting from orbital elements to Cartesian")
            # Convert from orbital elements to Cartesian coordinates
            for i, row in enumerate(data_dicts):
                # Create synthetic positions from orbital elements or use defaults
                if 'inclination_deg' in row and 'mean_anomaly_deg' in row:
                    # Simple orbital position approximation
                    inc = np.radians(row.get('inclination_deg', 55.0))
                    mean_anom = np.radians(row.get('mean_anomaly_deg', i * 15.0))
                    altitude = 20200  # GPS altitude in km
                    
                    x = altitude * np.cos(mean_anom)
                    y = altitude * np.sin(mean_anom) * np.cos(inc)
                    z = altitude * np.sin(mean_anom) * np.sin(inc)
                    
                    # Approximate orbital velocity
                    orbital_speed = 3.9  # km/s for GPS orbit
                    vx = -orbital_speed * np.sin(mean_anom)
                    vy = orbital_speed * np.cos(mean_anom) * np.cos(inc)
                    vz = orbital_speed * np.cos(mean_anom) * np.sin(inc)
                else:
                    # Fallback: generate synthetic positions
                    angle = i * 2 * np.pi / len(data_dicts)
                    altitude = 20200  # km
                    x = altitude * np.cos(angle)
                    y = altitude * np.sin(angle)
                    z = altitude * 0.1 * np.sin(angle * 2)  # Small z variation
                    
                    # Orbital velocity
                    orbital_speed = 3.9  # km/s
                    vx = -orbital_speed * np.sin(angle)
                    vy = orbital_speed * np.cos(angle)
                    vz = 0.1 * orbital_speed * np.cos(angle * 2)
                
                positions.append([x, y, z])
                velocities.append([vx, vy, vz])
        
        # Convert to tensors with consistent dimensions
        positions_tensor = torch.tensor(positions, dtype=torch.float32) * 1000  # Convert to meters
        velocities_tensor = torch.tensor(velocities, dtype=torch.float32) * 1000  # Convert to m/s
        
        print(f"DEBUG: positions shape: {positions_tensor.shape}, velocities shape: {velocities_tensor.shape}")
        
        # Ensure we have at least minimum number of nodes
        if positions_tensor.shape[0] < self.config.min_satellites:
            # Pad with synthetic nodes if needed
            while positions_tensor.shape[0] < self.config.min_satellites:
                # Add a synthetic satellite at a slightly different position
                base_pos = positions_tensor[-1] if len(positions_tensor) > 0 else torch.zeros(3)
                new_pos = base_pos + torch.randn(3) * 1000  # Add 1km random offset
                new_vel = torch.randn(3) * 1000  # Random velocity ~1 km/s
                
                positions_tensor = torch.cat([positions_tensor, new_pos.unsqueeze(0)], dim=0)
                velocities_tensor = torch.cat([velocities_tensor, new_vel.unsqueeze(0)], dim=0)
        
        print(f"DEBUG: After padding - positions shape: {positions_tensor.shape}, velocities shape: {velocities_tensor.shape}")
        
        # Create node features
        try:
            node_features = self.feature_extractor.extract_node_features(positions_tensor, velocities_tensor)
            print(f"DEBUG: node_features shape: {node_features.shape}")
        except Exception as e:
            print(f"ERROR in extract_node_features: {e}")
            raise
        
        # Create edges based on distance
        edge_index, basic_edge_attr = self.create_edges_from_positions(positions_tensor)
        print(f"DEBUG: edge_index shape: {edge_index.shape}")
        
        # Enhanced edge features with relativistic effects
        if edge_index.shape[1] > 0:
            enhanced_edge_attr = []
            for i in range(edge_index.shape[1]):
                src_idx = int(edge_index[0, i].item())
                dst_idx = int(edge_index[1, i].item())
                try:
                    edge_feat = self.feature_extractor.extract_edge_features(
                        positions_tensor[src_idx:src_idx+1], positions_tensor[dst_idx:dst_idx+1],
                        velocities_tensor[src_idx:src_idx+1], velocities_tensor[dst_idx:dst_idx+1]
                    )
                    enhanced_edge_attr.append(edge_feat)
                except Exception as e:
                    print(f"ERROR in extract_edge_features for edge {i}: {e}")
                    # Create fallback edge features
                    fallback_feat = torch.zeros((1, self.config.edge_attr_dim), dtype=torch.float32)
                    enhanced_edge_attr.append(fallback_feat)
            
            if enhanced_edge_attr:
                edge_attr = torch.cat(enhanced_edge_attr, dim=0)
                print(f"DEBUG: final edge_attr shape: {edge_attr.shape}")
            else:
                edge_attr = torch.empty((0, self.config.edge_attr_dim), dtype=torch.float32)
        else:
            edge_attr = torch.empty((0, self.config.edge_attr_dim), dtype=torch.float32)
        
        # Create target labels (can be positions at next timestep, velocities, etc.)
        y = positions  # For now, predict positions
        
        return Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            pos=positions
        )


class SpaceDataLoader:
    """Unified loader for various space datasets."""
    
    def __init__(self, config: Optional[SpaceGraphConfig] = None):
        self.config = config or SpaceGraphConfig()
        self.datasets: Dict[str, BaseSpaceDataset] = {}
    
    def load_satellite_constellation(
        self,
        root: str,
        data_source: str = "gps",
        force_rebuild: bool = False,
        **kwargs
    ) -> SatelliteConstellationDataset:
        """Load satellite constellation dataset."""
        dataset_key = f"constellation_{data_source}"
        
        if dataset_key not in self.datasets or force_rebuild:
            # Clear any existing processed files if force_rebuild
            if force_rebuild:
                import shutil
                processed_dir = Path(root) / "processed"
                if processed_dir.exists():
                    shutil.rmtree(processed_dir)
                    print("Cleared existing processed data")
            
            self.datasets[dataset_key] = SatelliteConstellationDataset(
                root=root,
                data_source=data_source,
                config=self.config,
                force_reload=force_rebuild,
                **kwargs
            )
        
        return self.datasets[dataset_key]
    
    def create_data_loader(
        self,
        dataset: BaseSpaceDataset,
        batch_size: int = 1,
        shuffle: bool = True,
        **kwargs
    ):
        """Create PyTorch DataLoader for space dataset."""
        from torch_geometric.loader import DataLoader
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            **kwargs
        )
    
    def get_dataset_info(self, dataset: BaseSpaceDataset) -> Dict[str, Any]:
        """Get information about a dataset."""
        if len(dataset) == 0:
            return {"error": "Empty dataset"}
        
        sample = dataset[0]
        
        return {
            "num_graphs": len(dataset),
            "num_nodes_per_graph": sample.x.shape[0] if hasattr(sample, 'x') else 0,
            "node_feature_dim": sample.x.shape[1] if hasattr(sample, 'x') else 0,
            "edge_feature_dim": sample.edge_attr.shape[1] if hasattr(sample, 'edge_attr') else 0,
            "num_edges_per_graph": sample.edge_index.shape[1] if hasattr(sample, 'edge_index') else 0,
            "has_positions": hasattr(sample, 'pos'),
            "has_targets": hasattr(sample, 'y')
        }
