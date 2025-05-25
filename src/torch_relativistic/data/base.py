"""
Base classes for relativistic datasets.
"""

import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import add_self_loops, degree
from typing import Optional, Tuple, Dict, List, Any, Union
import numpy as np
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class RelativisticData(Data):
    """
    Extended PyTorch Geometric Data class for relativistic information.
    
    Adds spacetime coordinates, velocities, and relativistic parameters
    to the standard graph data structure.
    """
    
    def __init__(self, 
                 x: Optional[torch.Tensor] = None,
                 edge_index: Optional[torch.Tensor] = None,
                 edge_attr: Optional[torch.Tensor] = None,
                 y: Optional[torch.Tensor] = None,
                 pos: Optional[torch.Tensor] = None,
                 # Relativistic extensions
                 spacetime_pos: Optional[torch.Tensor] = None,
                 velocity: Optional[torch.Tensor] = None,
                 proper_time: Optional[torch.Tensor] = None,
                 mass: Optional[torch.Tensor] = None,
                 **kwargs):
        """
        Initialize relativistic graph data.
        
        Args:
            x: Node feature matrix [num_nodes, num_features]
            edge_index: Graph connectivity [2, num_edges]
            edge_attr: Edge feature matrix [num_edges, num_edge_features]
            y: Target values
            pos: Node positions [num_nodes, 3] (spatial only)
            spacetime_pos: Spacetime coordinates [num_nodes, 4] (t, x, y, z)
            velocity: Node velocities [num_nodes, 3] (as fraction of c)
            proper_time: Proper time for each node [num_nodes]
            mass: Node masses [num_nodes] (in natural units)
        """
        super().__init__(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, pos=pos, **kwargs)
        
        self.spacetime_pos = spacetime_pos
        self.velocity = velocity
        self.proper_time = proper_time
        self.mass = mass
        
    def calculate_spacetime_intervals(self) -> torch.Tensor:
        """
        Calculate spacetime intervals between all pairs of nodes.
        
        Returns:
            Tensor: Spacetime intervals [num_nodes, num_nodes]
        """
        if self.spacetime_pos is None:
            raise ValueError("Spacetime positions required for interval calculation")
            
        # Minkowski metric signature (-,+,+,+)
        metric = torch.diag(torch.tensor([-1., 1., 1., 1.], device=self.spacetime_pos.device))
        
        # Calculate all pairwise differences
        diff = self.spacetime_pos.unsqueeze(1) - self.spacetime_pos.unsqueeze(0)  # [N, N, 4]
        
        # Apply Minkowski metric: ds² = η_μν dx^μ dx^ν
        intervals = torch.einsum('ijk,kl,ijl->ij', diff, metric, diff)
        
        return intervals
    
    def calculate_light_cones(self, node_idx: int) -> Dict[str, torch.Tensor]:
        """
        Calculate light cone structure for a specific node.
        
        Args:
            node_idx: Index of the central node
            
        Returns:
            Dict with 'timelike', 'spacelike', 'lightlike' node indices
        """
        intervals = self.calculate_spacetime_intervals()
        node_intervals = intervals[node_idx]
        
        # Classify intervals
        timelike = (node_intervals < -1e-10).nonzero(as_tuple=True)[0]
        spacelike = (node_intervals > 1e-10).nonzero(as_tuple=True)[0]  
        lightlike = (torch.abs(node_intervals) <= 1e-10).nonzero(as_tuple=True)[0]
        
        return {
            'timelike': timelike,
            'spacelike': spacelike, 
            'lightlike': lightlike
        }
    
    def add_relativistic_edge_features(self) -> torch.Tensor:
        """
        Add relativistic features to edges based on spacetime geometry.
        
        Returns:
            Tensor: Enhanced edge features [num_edges, original_features + relativistic_features]
        """
        if self.edge_index is None:
            raise ValueError("Edge index required")
            
        src, dst = self.edge_index
        num_edges = src.size(0)
        
        relativistic_features = []
        
        # 1. Spacetime intervals along edges
        if self.spacetime_pos is not None:
            intervals = self.calculate_spacetime_intervals()
            edge_intervals = intervals[src, dst].unsqueeze(1)
            relativistic_features.append(edge_intervals)
        
        # 2. Relative velocities
        if self.velocity is not None:
            rel_velocity = self.velocity[dst] - self.velocity[src]  # [num_edges, 3]
            rel_speed = torch.norm(rel_velocity, dim=1, keepdim=True)  # [num_edges, 1]
            relativistic_features.extend([rel_velocity, rel_speed])
            
            # Lorentz factor
            gamma = 1.0 / torch.sqrt(1.0 - rel_speed**2 + 1e-8)
            relativistic_features.append(gamma)
        
        # 3. Proper time differences
        if self.proper_time is not None:
            time_diff = (self.proper_time[dst] - self.proper_time[src]).unsqueeze(1)
            relativistic_features.append(time_diff)
        
        # 4. Mass ratios
        if self.mass is not None:
            mass_ratio = (self.mass[dst] / (self.mass[src] + 1e-8)).unsqueeze(1)
            relativistic_features.append(mass_ratio)
        
        if not relativistic_features:
            logger.warning("No relativistic features could be computed")
            return self.edge_attr if self.edge_attr is not None else torch.zeros(num_edges, 1)
        
        # Concatenate all features
        rel_features = torch.cat(relativistic_features, dim=1)
        
        if self.edge_attr is not None:
            return torch.cat([self.edge_attr, rel_features], dim=1)
        else:
            return rel_features


class BaseRelativisticDataset(Dataset, ABC):
    """
    Abstract base class for relativistic datasets.
    """
    
    def __init__(self, 
                 root: str,
                 transform: Optional[callable] = None,
                 pre_transform: Optional[callable] = None,
                 pre_filter: Optional[callable] = None):
        """
        Initialize base relativistic dataset.
        
        Args:
            root: Root directory for data storage
            transform: Transform to apply to each data sample
            pre_transform: Transform to apply before saving processed data
            pre_filter: Filter to apply before processing
        """
        self.c = 299792458.0  # Speed of light in m/s
        self.G = 6.67430e-11  # Gravitational constant in m³/kg/s²
        
        super().__init__(root, transform, pre_transform, pre_filter)
    
    @abstractmethod
    def download(self):
        """Download raw data files."""
        pass
    
    @abstractmethod
    def process(self):
        """Process raw data into RelativisticData objects."""
        pass
    
    def calculate_lorentz_factor(self, velocity: torch.Tensor) -> torch.Tensor:
        """
        Calculate Lorentz factor γ = 1/√(1-v²/c²).
        
        Args:
            velocity: Velocity tensor [batch, 3] in m/s
            
        Returns:
            Lorentz factors [batch]
        """
        v_squared = torch.sum(velocity**2, dim=-1)
        beta_squared = v_squared / (self.c**2)
        
        # Clamp to prevent numerical issues
        beta_squared = torch.clamp(beta_squared, 0.0, 0.999)
        
        gamma = 1.0 / torch.sqrt(1.0 - beta_squared)
        return gamma
    
    def time_dilation(self, proper_time: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
        """
        Calculate time dilation: Δt = γ * Δτ.
        
        Args:
            proper_time: Proper time intervals [batch]
            velocity: Velocity tensor [batch, 3] in m/s
            
        Returns:
            Coordinate time intervals [batch]
        """
        gamma = self.calculate_lorentz_factor(velocity)
        return gamma * proper_time
    
    def gravitational_time_dilation(self, 
                                  proper_time: torch.Tensor,
                                  gravitational_potential: torch.Tensor) -> torch.Tensor:
        """
        Calculate gravitational time dilation: Δt = Δτ / √(1 + 2Φ/c²).
        
        Args:
            proper_time: Proper time intervals [batch]
            gravitational_potential: Gravitational potential [batch] in J/kg
            
        Returns:
            Coordinate time intervals [batch]
        """
        phi_normalized = gravitational_potential / (self.c**2)
        
        # Weak field approximation
        factor = torch.sqrt(1.0 + 2.0 * phi_normalized)
        return proper_time / factor
    
    def add_causal_constraints(self, data: RelativisticData) -> RelativisticData:
        """
        Add causal constraints by removing spacelike connections.
        
        Args:
            data: Input relativistic data
            
        Returns:
            Data with causally consistent edge structure
        """
        if data.spacetime_pos is None:
            logger.warning("Cannot apply causal constraints without spacetime positions")
            return data
        
        intervals = data.calculate_spacetime_intervals()
        
        # Only keep timelike and lightlike connections
        src, dst = data.edge_index
        edge_intervals = intervals[src, dst]
        
        # Keep edges where interval ≤ 0 (timelike or lightlike)
        causal_mask = edge_intervals <= 1e-10
        
        if causal_mask.sum() == 0:
            logger.warning("No causal edges found - adding self-loops")
            data.edge_index, data.edge_attr = add_self_loops(
                data.edge_index, data.edge_attr, num_nodes=data.num_nodes
            )
        else:
            data.edge_index = data.edge_index[:, causal_mask]
            if data.edge_attr is not None:
                data.edge_attr = data.edge_attr[causal_mask]
        
        return data
    
    def normalize_to_natural_units(self, 
                                 length_m: Optional[torch.Tensor] = None,
                                 time_s: Optional[torch.Tensor] = None,
                                 mass_kg: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Convert physical quantities to natural units (c = G = 1).
        
        Args:
            length_m: Lengths in meters
            time_s: Times in seconds  
            mass_kg: Masses in kilograms
            
        Returns:
            Dictionary with normalized quantities
        """
        result = {}
        
        if length_m is not None:
            # Length scale: 1 meter = 1/(3×10⁸) natural units
            result['length'] = length_m / self.c
            
        if time_s is not None:
            # Time scale: 1 second = c natural units
            result['time'] = time_s * self.c
            
        if mass_kg is not None:
            # Mass scale: 1 kg = G/c² natural units  
            result['mass'] = mass_kg * self.G / (self.c**2)
            
        return result
