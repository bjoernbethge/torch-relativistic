"""
Relativistic Graph Neural Network modules inspired by the Terrell-Penrose effect.

This module provides GNN layers that incorporate relativistic concepts, enabling
graph neural networks to process information with inspiration from special relativity.
The key insight is modeling node-to-node communication as if affected by relativistic
effects like time dilation and apparent rotation seen in the Terrell-Penrose effect.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple, List, Union, Dict, Any

# For PyTorch Geometric compatibility - gracefully handle if not available
try:
    from torch_geometric.nn import MessagePassing
    from torch_scatter import scatter_add
    HAS_PYGEOMETRIC = True
except ImportError:
    # Create stub class to allow the module to load without PyTorch Geometric
    class MessagePassing:
        def __init__(self, aggr="add"):
            self.aggr = aggr
    HAS_PYGEOMETRIC = False


class RelativisticGraphConv(MessagePassing if HAS_PYGEOMETRIC else nn.Module):
    """
    A graph convolutional layer that incorporates relativistic effects inspired by the Terrell-Penrose effect.
    
    This layer models information propagation between nodes as if affected by relativistic phenomena.
    Graph nodes are treated as if they exist in different reference frames, with message passing
    being affected by relativistic transformations like time dilation and apparent rotation.
    
    In the Terrell-Penrose effect, rapidly moving objects appear rotated rather than contracted.
    Similarly, this layer applies transformations to node features during message passing,
    where the "speed" parameter controls the strength of these relativistic effects.
    
    Args:
        in_channels (int): Size of input node features
        out_channels (int): Size of output node features
        max_relative_velocity (float, optional): Maximum "speed" as fraction of c. Defaults to 0.9.
        bias (bool, optional): Whether to include a bias term. Defaults to True.
        aggr (str, optional): Aggregation method ('add', 'mean', 'max'). Defaults to "add".
        
    Note:
        Requires PyTorch Geometric for full functionality. Falls back to a simplified version
        if PyTorch Geometric is not available.
    """
    
    def __init__(self, in_channels: int, out_channels: int, max_relative_velocity: float = 0.9,
                 bias: bool = True, aggr: str = "add"):
        if HAS_PYGEOMETRIC:
            super().__init__(aggr=aggr)
        else:
            super().__init__()
            print("Warning: PyTorch Geometric not available. RelativisticGraphConv will use simplified implementation.")
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.max_relative_velocity = max_relative_velocity
        
        # Learnable parameters
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)
        self.velocity_factor = nn.Parameter(torch.Tensor([0.5]))  # Initialize at 0.5c
        
        # Reset parameters
        self.reset_parameters()
    
    def reset_parameters(self):
        """Reset learnable parameters to initial values."""
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)
        with torch.no_grad():
            self.velocity_factor.data.fill_(0.5)
    
    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Optional[Tensor] = None, 
                position: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass of relativistic graph convolution.
        
        Args:
            x (Tensor): Node features of shape [N, in_channels]
            edge_index (Tensor): Graph connectivity of shape [2, E]
            edge_attr (Tensor, optional): Edge features of shape [E, edge_features]. Defaults to None.
            position (Tensor, optional): Node positions of shape [N, D]. Defaults to None.
            
        Returns:
            Tensor: Updated node features of shape [N, out_channels]
        """
        # If positions not provided, use first dimensions of features as abstract positions
        if position is None:
            position = x[:, :min(3, x.size(1))]
        
        if HAS_PYGEOMETRIC:
            # Use PyTorch Geometric's MessagePassing machinery
            return self.propagate(edge_index, x=x, position=position, edge_attr=edge_attr)
        else:
            # Simplified implementation without PyTorch Geometric
            return self._simplified_forward(x, edge_index, position, edge_attr)
    
    def message(self, x_j: Tensor, position_i: Tensor, position_j: Tensor,
                edge_attr: Optional[Tensor] = None) -> Tensor:
        """
        Compute messages between nodes in a relativistic framework.
        
        Args:
            x_j: Features of the source nodes
            position_i: Positions of the target nodes
            position_j: Positions of the source nodes
            edge_attr: Optional edge features
            
        Returns:
            Tensor: Relativistically transformed messages
        """
        # Calculate "relative velocity" between connected nodes
        delta_position = position_j - position_i
        distance = torch.norm(delta_position, dim=1, keepdim=True)
        
        # Scale velocity by distance and learnable factor, clamped to max relative velocity
        v_rel = self.velocity_factor * torch.clamp(distance / distance.max(), 0, self.max_relative_velocity)
        gamma = 1.0 / torch.sqrt(1.0 - v_rel**2 + 1e-8)  # Add epsilon for numerical stability
        
        # Transform features using node linear transformation
        transformed_features = self.linear(x_j)
        
        # Apply relativistic transformation factor
        messages = transformed_features * self._terrell_penrose_factor(delta_position, v_rel, gamma)
        
        # Incorporate edge features if provided
        if edge_attr is not None:
            # Simple multiplication as one way to include edge information
            messages = messages * edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else messages * edge_attr
            
        return messages
    
    def _terrell_penrose_factor(self, delta_pos: Tensor, velocity: Tensor, gamma: Tensor) -> Tensor:
        """
        Calculate a factor inspired by the Terrell-Penrose effect to modify message passing.
        
        Args:
            delta_pos: Positional difference between nodes
            velocity: Relative velocity between nodes
            gamma: Lorentz factor
            
        Returns:
            Tensor: Transformation factor to apply to messages
        """
        # Normalize direction vector
        direction = delta_pos / (torch.norm(delta_pos, dim=1, keepdim=True) + 1e-8)
        
        # Factor inspired by relativistic aberration and apparent rotation
        # This simulates how information from different nodes appears "rotated" due to
        # relative motion, similar to the Terrell-Penrose effect
        aberration_factor = 1.0 / (gamma * (1.0 + velocity * torch.sum(direction, dim=1, keepdim=True)))
        
        return aberration_factor
    
    def _simplified_forward(self, x: Tensor, edge_index: Tensor, position: Tensor,
                           edge_attr: Optional[Tensor] = None) -> Tensor:
        """
        Simplified implementation when PyTorch Geometric is not available.
        
        Args:
            x: Node features
            edge_index: Graph connectivity
            position: Node positions
            edge_attr: Edge features
            
        Returns:
            Tensor: Updated node features
        """
        num_nodes = x.size(0)
        src, dst = edge_index[0], edge_index[1]
        
        # Create messages
        x_j = x[src]
        position_i = position[dst]
        position_j = position[src]
        
        # Calculate relativistic message
        delta_position = position_j - position_i
        distance = torch.norm(delta_position, dim=1, keepdim=True)
        
        v_rel = self.velocity_factor * torch.clamp(distance / distance.max(), 0, self.max_relative_velocity)
        gamma = 1.0 / torch.sqrt(1.0 - v_rel**2 + 1e-8)
        
        transformed_features = self.linear(x_j)
        direction = delta_position / (torch.norm(delta_position, dim=1, keepdim=True) + 1e-8)
        aberration_factor = 1.0 / (gamma * (1.0 + v_rel * torch.sum(direction, dim=1, keepdim=True)))
        
        messages = transformed_features * aberration_factor
        
        if edge_attr is not None:
            if edge_attr.dim() == 1:
                messages = messages * edge_attr.unsqueeze(-1)
            else:
                messages = messages * edge_attr
        
        # Manual aggregation (simplified version of scatter_add)
        output = torch.zeros(num_nodes, self.out_channels, device=x.device)
        
        # For each target node, sum the incoming messages
        for i in range(len(dst)):
            output[dst[i]] += messages[i]
            
        return output


class MultiObserverGNN(nn.Module):
    """
    A Graph Neural Network that processes graphs from multiple relativistic "observer" perspectives.
    
    Inspired by the way the Terrell-Penrose effect shows how appearance changes based on
    different reference frames, this module processes a graph through multiple different
    relativistic reference frames using parameter sharing and learnable velocity configurations.
    
    The multi-view representations are integrated using attention mechanisms to form a
    more robust final representation that incorporates information from different "perspectives".
    
    Args:
        feature_dim (int): Dimension of input node features
        hidden_dim (int): Dimension of hidden node features
        output_dim (int): Dimension of output node features
        num_observers (int, optional): Number of different "observer" perspectives. Defaults to 4.
        velocity_max (float, optional): Maximum velocity for observers (as fraction of c). Defaults to 0.9.
        dropout (float, optional): Dropout probability. Defaults to 0.1.
        
    Note:
        This model can work even without PyTorch Geometric, but full functionality
        requires PyTorch Geometric to be installed.
    """
    
    def __init__(self, feature_dim: int, hidden_dim: int, output_dim: int,
                 num_observers: int = 4, velocity_max: float = 0.9, dropout: float = 0.1):
        super().__init__()
        self.num_observers = num_observers
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.velocity_max = velocity_max
        
        # Feature preprocessing (shared)
        self.feature_transform = nn.Linear(feature_dim, hidden_dim)
        
        # Gemeinsame Basisschicht für alle Beobachter (Parameter-Sharing)
        self.base_layer = RelativisticGraphConv(hidden_dim, hidden_dim)
        
        # Parameter für die relativistischen Geschwindigkeiten (lernbar)
        velocities = torch.linspace(velocity_max/num_observers, velocity_max, num_observers)
        self.velocity_params = nn.Parameter(velocities)
        
        # Batch-Normalisierung für jede Beobachterperspektive
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_observers)])
        
        # Attention-Mechanismus für Perspektiven-Gewichtung
        self.attention = nn.Parameter(torch.ones(num_observers, hidden_dim) / num_observers)
        
        # Integration der verschiedenen Perspektiven
        self.integration = nn.Sequential(
            nn.Linear(num_observers * hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, output_dim)
        )
    
    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Optional[Tensor] = None, 
                position: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass of the multi-observer GNN.
        
        Args:
            x (Tensor): Node features
            edge_index (Tensor): Graph connectivity
            edge_attr (Tensor, optional): Edge features. Defaults to None.
            position (Tensor, optional): Node positions. Defaults to None.
            
        Returns:
            Tensor: Node features viewed from integrated multiple perspectives
        """
        # Transform input features
        h = F.relu(self.feature_transform(x))
        
        # Collect observations from different relativistic reference frames
        multi_view_features = []
        
        for i in range(self.num_observers):
            # Temporär die Geschwindigkeitsparameter anpassen
            with torch.no_grad():
                original_velocity = self.base_layer.max_relative_velocity
                self.base_layer.max_relative_velocity = self.velocity_params[i].item()
            
            # Beobachtung sammeln
            view = self.base_layer(h, edge_index, edge_attr, position)
            
            # Auf Original zurücksetzen
            with torch.no_grad():
                self.base_layer.max_relative_velocity = original_velocity
            
            # Anwendung von Batch-Normalisierung und Aktivierung
            view = self.batch_norms[i](view)
            view = F.relu(view)
            
            # Gewichtung mit Attention
            view = view * F.softmax(self.attention[i], dim=0).unsqueeze(0)
            
            multi_view_features.append(view)
        
        # Concatenate all views
        combined = torch.cat(multi_view_features, dim=1)
        
        # Integrate the different perspectives
        output = self.integration(combined)
        
        return output
