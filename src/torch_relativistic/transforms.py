"""
Relativistic transformations inspired by the Terrell-Penrose effect.

This module provides transformation functions that apply relativistic effects
to tensors, simulating how information would appear if subject to effects
similar to those in the Terrell-Penrose effect. These transformations can be
used as components in neural networks to enable relativistic information processing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple, List, Union, Dict, Any
import math


class TerrellPenroseTransform(nn.Module):
    """
    Applies a transformation inspired by the Terrell-Penrose effect to feature vectors.
    
    The Terrell-Penrose effect describes how fast-moving objects appear rotated rather
    than contracted. This module applies a similar principle to feature vectors,
    where different feature dimensions are transformed based on their "observational distance"
    and a velocity parameter that controls the strength of the effect.
    
    Args:
        feature_dim (int): Dimension of input features
        max_velocity (float, optional): Maximum relativistic velocity (0-1). Defaults to 0.9.
        learnable (bool, optional): Whether velocity parameter is learnable. Defaults to True.
        mode (str, optional): Transformation mode ('rotation' or 'full'). Defaults to "rotation".
        
    Attributes:
        velocity (Parameter): Relativistic velocity parameter (fraction of c)
    """
    
    def __init__(self, feature_dim: int, max_velocity: float = 0.9, 
                 learnable: bool = True, mode: str = "rotation"):
        super().__init__()
        self.feature_dim = feature_dim
        self.max_velocity = max_velocity
        self.mode = mode
        
        # Relativistic velocity parameter (learnable or fixed)
        self.velocity = nn.Parameter(
            torch.Tensor([0.5 * max_velocity]),
            requires_grad=learnable
        )
        
        # Feature distances - determine how each dimension is transformed
        # In the Terrell-Penrose effect, the "distance" affects how objects appear
        # Here each feature dimension has a distance that affects its transformation
        self.distances = nn.Parameter(
            torch.linspace(0, 1, feature_dim).unsqueeze(0),
            requires_grad=learnable
        )
        
        # For rotation mode, we need rotation matrix parameters
        if mode == "rotation":
            # Initialize with identity-like rotations
            angles = torch.zeros(feature_dim // 2)
            self.rotation_angles = nn.Parameter(angles, requires_grad=learnable)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Apply relativistic transformation to input features.
        
        Args:
            x (Tensor): Input tensor [batch_size, ..., feature_dim]
            
        Returns:
            Tensor: Transformed tensor [batch_size, ..., feature_dim]
        """
        # Save original shape
        original_shape = x.shape
        
        # Reshape to [batch * ...rest, feature_dim]
        batch_size = x.size(0)
        reshaped_x = x.reshape(-1, self.feature_dim)
        
        # Calculate relativistic factor
        v = torch.clamp(self.velocity, -self.max_velocity, self.max_velocity)
        gamma = 1.0 / torch.sqrt(1.0 - v**2)
        
        if self.mode == "rotation":
            # Apply Terrell-Penrose-inspired rotation
            transformed_x = self._apply_rotation(reshaped_x, gamma)
        else:
            # Apply full relativistic transformation
            transformed_x = self._apply_full_transform(reshaped_x, gamma)
        
        # Reshape back to original shape
        return transformed_x.reshape(original_shape)
    
    def _apply_rotation(self, x: Tensor, gamma: Tensor) -> Tensor:
        """
        Apply rotation transformation inspired by Terrell-Penrose effect.
        
        Args:
            x (Tensor): Flattened input tensor [batch*..., feature_dim]
            gamma (Tensor): Relativistic gamma factor
            
        Returns:
            Tensor: Rotated tensor [batch*..., feature_dim]
        """
        # Ensure even feature dimension for rotation
        feature_dim = x.size(-1)
        if feature_dim % 2 != 0:
            # If odd, pad with one extra feature
            x = F.pad(x, (0, 1))
            feature_dim += 1
        
        # Reshape to pairs of features for rotation
        pairs = x.view(-1, feature_dim // 2, 2)
        
        # Apply variable rotation based on relativistic effects
        # In Terrell-Penrose effect, apparent rotation depends on velocity and viewing angle
        rotated_pairs = torch.zeros_like(pairs)
        
        # Get rotation angles modulated by relativistic factor
        angles = self.rotation_angles * gamma
        
        # Apply rotation to each pair
        cos_angles = torch.cos(angles).unsqueeze(0)  # [1, feature_dim//2]
        sin_angles = torch.sin(angles).unsqueeze(0)  # [1, feature_dim//2]
        
        # Rotation matrix application
        # [x']   [cos(θ) -sin(θ)] [x]
        # [y'] = [sin(θ)  cos(θ)] [y]
        rotated_pairs[..., 0] = pairs[..., 0] * cos_angles - pairs[..., 1] * sin_angles
        rotated_pairs[..., 1] = pairs[..., 0] * sin_angles + pairs[..., 1] * cos_angles
        
        # Reshape back
        rotated = rotated_pairs.reshape(-1, feature_dim)
        
        # Remove padding if added
        if x.size(-1) != self.feature_dim:
            rotated = rotated[..., :self.feature_dim]
        
        return rotated
    
    def _apply_full_transform(self, x: Tensor, gamma: Tensor) -> Tensor:
        """
        Apply full relativistic transformation with distance-dependent effects.
        
        Args:
            x (Tensor): Flattened input tensor [batch*..., feature_dim]
            gamma (Tensor): Relativistic gamma factor
            
        Returns:
            Tensor: Transformed tensor [batch*..., feature_dim]
        """
        batch_size = x.size(0)
        
        # Expand distances to match batch size
        distances = self.distances.expand(batch_size, -1)  # [batch, feature_dim]
        
        # Apply relativistic effects based on feature "distances"
        # 1. Lorentz contraction along direction of motion
        contraction_factor = 1.0 / gamma
        
        # 2. Time dilation effect on features
        # (in Terrell-Penrose, different parts of object appear as at different times)
        time_dilation = gamma * distances
        
        # 3. Calculate relativistic aberration factor
        # (in Terrell-Penrose, light from different points arrives at different angles)
        v = self.velocity.abs()
        aberration_factor = 1.0 / (gamma * (1.0 + v * distances))
        
        # Combined relativistic factor
        rel_factor = contraction_factor * aberration_factor
        
        # Apply transformation
        transformed = x * rel_factor
        
        return transformed


class LorentzBoost(nn.Module):
    """
    Applies a Lorentz boost transformation to feature vectors.
    
    In special relativity, a Lorentz boost describes the transformation between
    two reference frames in relative motion. This module applies an analogous
    transformation to feature vectors, allowing neural networks to process
    information as if from different "reference frames".
    
    Args:
        feature_dim (int): Dimension of feature vectors
        time_dim (int, optional): Dimension to use as time component. Defaults to 0.
        max_velocity (float, optional): Maximum velocity parameter (0-1). Defaults to 0.9.
        learnable (bool, optional): Whether velocity parameters are learnable. Defaults to True.
        
    Attributes:
        velocity (Parameter): 3D velocity vector (fractions of c)
    """
    
    def __init__(self, feature_dim: int, time_dim: int = 0, max_velocity: float = 0.9,
                 learnable: bool = True):
        super().__init__()
        self.feature_dim = feature_dim
        self.time_dim = time_dim
        self.max_velocity = max_velocity
        
        # At minimum, we need 4D features (time + 3D space)
        assert feature_dim >= 4, "Feature dimension must be at least 4 for spacetime representation"
        
        # Velocity vector (3D)
        self.velocity = nn.Parameter(
            torch.zeros(3) + 0.1,  # Small initial values
            requires_grad=learnable
        )
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Apply Lorentz boost to features.
        
        Args:
            x (Tensor): Input tensor [batch_size, ..., feature_dim]
            
        Returns:
            Tensor: Boosted tensor [batch_size, ..., feature_dim]
        """
        # Save original shape
        original_shape = x.shape
        
        # Reshape to [batch * ...rest, feature_dim]
        batch_size = x.size(0)
        reshaped_x = x.reshape(-1, self.feature_dim)
        
        # Treat first 4 dimensions as spacetime coordinates
        # Extract time component and space components
        spacetime = reshaped_x[:, :4].clone()
        
        # Identify time and space components
        time_indices = [self.time_dim]
        space_indices = [i for i in range(4) if i != self.time_dim]
        
        t = spacetime[:, self.time_dim].unsqueeze(1)  # [batch, 1]
        r = spacetime[:, space_indices]  # [batch, 3]
        
        # Get velocity vector and calculate relativistic gamma
        v = torch.clamp(self.velocity, -self.max_velocity, self.max_velocity)
        v_magnitude = torch.norm(v, dim=0)
        v_normalized = v / (v_magnitude + 1e-8)
        
        gamma = 1.0 / torch.sqrt(1.0 - v_magnitude**2)
        
        # Apply Lorentz boost transformation
        # t' = γ(t - v·r/c²)
        # r' = r + γv[(γ-1)(v·r)/v² - t]
        
        # Transform time component
        v_dot_r = torch.sum(r * v_normalized, dim=1, keepdim=True)
        t_prime = gamma * (t - v_magnitude * v_dot_r)
        
        # Transform space components
        space_transform = r + torch.outer(
            gamma * (gamma - 1) * v_dot_r.squeeze() / (v_magnitude**2 + 1e-8) - gamma * t.squeeze(),
            v
        )
        
        # Combine transformed components back into spacetime
        boosted_spacetime = spacetime.clone()
        boosted_spacetime[:, self.time_dim] = t_prime.squeeze()
        
        # Assign space components to their original positions
        for i, idx in enumerate(space_indices):
            boosted_spacetime[:, idx] = space_transform[:, i]
        
        # Combine with any additional dimensions beyond spacetime
        if self.feature_dim > 4:
            # Copy non-spacetime features unchanged
            boosted = torch.cat([boosted_spacetime, reshaped_x[:, 4:]], dim=1)
        else:
            boosted = boosted_spacetime
        
        # Reshape back to original dimensions
        return boosted.reshape(original_shape)


class RelativisticPooling(nn.Module):
    """
    Pooling operation with relativistic weighting of spatial regions.
    
    Inspired by how the Terrell-Penrose effect causes different spatial regions
    to contribute differently to the observed image, this pooling layer applies
    a similar principle where pooling weights are modulated by relativistic effects.
    
    Args:
        kernel_size (Union[int, Tuple[int, ...]]): Size of the pooling kernel
        stride (Optional[Union[int, Tuple[int, ...]]]): Stride of the pooling operation.
                                                        Defaults to None (same as kernel_size).
        padding (Union[int, Tuple[int, ...]]): Padding to be added to input. Defaults to 0.
        max_velocity (float): Maximum velocity parameter (0-1). Defaults to 0.9.
        mode (str): Pooling mode ('max', 'avg', 'relativistic'). Defaults to "relativistic".
        
    Note:
        When mode='relativistic', pooling weights are non-uniform and depend on
        a learned "velocity" parameter and the position in the kernel.
    """
    
    def __init__(self, kernel_size: Union[int, Tuple[int, ...]], 
                 stride: Optional[Union[int, Tuple[int, ...]]] = None,
                 padding: Union[int, Tuple[int, ...]] = 0,
                 max_velocity: float = 0.9,
                 mode: str = "relativistic"):
        super().__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if stride is not None else self.kernel_size
        self.stride = self.stride if isinstance(self.stride, tuple) else (self.stride, self.stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.max_velocity = max_velocity
        self.mode = mode
        
        # Create relativistic parameters
        if self.mode == "relativistic":
            # Direction of "motion" (normalized in forward pass)
            self.velocity_vector = nn.Parameter(torch.Tensor([0.7, 0.3]))
            
            # Initialize velocity magnitude
            self.velocity_magnitude = nn.Parameter(torch.Tensor([0.5 * max_velocity]))
            
            # Create learnable reference point for relativistic weighting
            # This represents the "observer" position relative to kernel
            self.reference_point = nn.Parameter(torch.Tensor([0.5, 0.5]))
            
            # Initialize weights
            self._init_weights()
    
    def _init_weights(self):
        """Initialize learnable parameters."""
        nn.init.uniform_(self.velocity_vector, -1.0, 1.0)
        nn.init.uniform_(self.velocity_magnitude, 0.1, 0.5 * self.max_velocity)
        nn.init.uniform_(self.reference_point, 0.3, 0.7)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Apply relativistic pooling to input tensor.
        
        Args:
            x (Tensor): Input tensor [batch_size, channels, height, width]
            
        Returns:
            Tensor: Pooled tensor
        """
        if self.mode == "max":
            return F.max_pool2d(x, self.kernel_size, self.stride, self.padding)
        elif self.mode == "avg":
            return F.avg_pool2d(x, self.kernel_size, self.stride, self.padding)
        elif self.mode == "relativistic":
            return self._relativistic_pool(x)
        else:
            raise ValueError(f"Unknown pooling mode: {self.mode}")
    
    def _relativistic_pool(self, x: Tensor) -> Tensor:
        """
        Apply pooling with relativistic weighting.
        
        Args:
            x (Tensor): Input tensor [batch_size, channels, height, width]
            
        Returns:
            Tensor: Pooled tensor
        """
        batch_size, channels, height, width = x.shape
        
        # Prepare output dimensions
        out_height = (height + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
        out_width = (width + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1
        
        # Apply padding if needed
        if self.padding[0] > 0 or self.padding[1] > 0:
            x = F.pad(x, (self.padding[1], self.padding[1], self.padding[0], self.padding[0]))
        
        # Calculate relativistic weighting for each position in kernel
        weights = self._calculate_relativistic_weights()
        
        # Create output tensor
        output = torch.zeros((batch_size, channels, out_height, out_width), device=x.device)
        
        # Unfold input tensor to extract patches
        patches = F.unfold(
            x, 
            kernel_size=self.kernel_size, 
            stride=self.stride
        )
        
        # Reshape patches for weighted combination
        patches = patches.view(batch_size, channels, self.kernel_size[0] * self.kernel_size[1], -1)
        
        # Apply relativistic weighting to each patch
        weights = weights.view(1, 1, -1, 1)  # [1, 1, kernel_size*kernel_size, 1]
        weighted_patches = patches * weights
        
        # Sum weighted values (similar to avg pooling but with learned weights)
        pooled = weighted_patches.sum(dim=2)
        
        # Reshape back to spatial dimensions
        output = pooled.view(batch_size, channels, out_height, out_width)
        
        return output
    
    def _calculate_relativistic_weights(self) -> Tensor:
        """
        Calculate weights based on relativistic effects.
        
        Returns:
            Tensor: Relativistic weights for each position in kernel
        """
        # Create position grid for kernel
        y_indices = torch.arange(self.kernel_size[0], device=self.velocity_vector.device)
        x_indices = torch.arange(self.kernel_size[1], device=self.velocity_vector.device)
        
        grid_y, grid_x = torch.meshgrid(y_indices, x_indices, indexing='ij')
        
        # Normalize positions to [0, 1] range
        pos_y = grid_y.float() / (self.kernel_size[0] - 1)
        pos_x = grid_x.float() / (self.kernel_size[1] - 1)
        
        # Stack positions
        positions = torch.stack([pos_y.flatten(), pos_x.flatten()], dim=1)  # [kernel_size*kernel_size, 2]
        
        # Calculate vector from reference point to each position
        rel_positions = positions - self.reference_point
        
        # Normalize velocity vector
        v_norm = self.velocity_vector / (torch.norm(self.velocity_vector) + 1e-8)
        
        # Calculate relativistic parameters
        v_magnitude = torch.clamp(self.velocity_magnitude, 0.0, self.max_velocity)
        gamma = 1.0 / torch.sqrt(1.0 - v_magnitude**2)
        
        # Project positions onto velocity direction
        proj = torch.sum(rel_positions * v_norm, dim=1)
        
        # Calculate relativistic aberration factor
        # In the Terrell-Penrose effect, light from different parts arrives
        # at different angles, causing the apparent rotation
        aberration_factor = 1.0 / (gamma * (1.0 + v_magnitude * proj))
        
        # Normalize weights to sum to 1 (like avg pooling)
        weights = F.softmax(aberration_factor, dim=0)
        
        return weights


class SpacetimeConvolution(nn.Module):
    """
    Convolution with relativistic spacetime geometry considerations.
    
    This module extends standard convolution to incorporate relativistic effects
    inspired by the Terrell-Penrose effect. The key insight is that information
    from different spatial locations and temporal offsets is processed as if
    affected by relativistic distortions due to relative motion.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (Union[int, Tuple[int, ...]]): Size of the convolution kernel
        stride (Union[int, Tuple[int, ...]], optional): Stride of the convolution. Defaults to 1.
        padding (Union[int, Tuple[int, ...]], optional): Padding added to input. Defaults to 0.
        dilation (Union[int, Tuple[int, ...]], optional): Spacing between kernel elements. Defaults to 1.
        groups (int, optional): Number of blocked connections. Defaults to 1.
        bias (bool, optional): If True, adds a learnable bias. Defaults to True.
        max_velocity (float, optional): Maximum velocity parameter. Defaults to 0.9.
        
    Note:
        For temporal sequences, the first input dimension is treated as time.
        For spatial inputs, relativistic effects are applied to the spatial dimensions.
    """
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: Union[int, Tuple[int, ...]],
                 stride: Union[int, Tuple[int, ...]] = 1,
                 padding: Union[int, Tuple[int, ...]] = 0,
                 dilation: Union[int, Tuple[int, ...]] = 1,
                 groups: int = 1,
                 bias: bool = True,
                 max_velocity: float = 0.9):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Process kernel_size
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,) * 3
        self.stride = stride if isinstance(stride, tuple) else (stride,) * 3
        self.padding = padding if isinstance(padding, tuple) else (padding,) * 3
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation,) * 3
        self.groups = groups
        self.max_velocity = max_velocity
        
        # Standard convolution layer
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=groups,
            bias=bias
        )
        
        # Relativistic parameters
        # Velocity vector represents direction of "motion" in spacetime
        self.velocity = nn.Parameter(torch.Tensor([0.0, 0.3, 0.3]))
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for relativistic parameters."""
        nn.init.xavier_uniform_(self.conv.weight)
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)
            
        # Initialize velocity to a small random value
        nn.init.uniform_(self.velocity, -0.3, 0.3)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Apply spacetime convolution with relativistic effects.
        
        Args:
            x (Tensor): Input tensor [batch_size, channels, time, height, width]
                        or [batch_size, channels, height, width] (will be unsqueezed)
            
        Returns:
            Tensor: Convolved tensor with relativistic effects
        """
        # Handle 4D input (add time dimension)
        if x.dim() == 4:
            x = x.unsqueeze(2)  # Add time dimension
        
        # Apply relativistic transformation to input
        x_transformed = self._apply_spacetime_transform(x)
        
        # Apply convolution
        output = self.conv(x_transformed)
        
        return output
    
    def _apply_spacetime_transform(self, x: Tensor) -> Tensor:
        """
        Apply relativistic spacetime transformation to input.
        
        Args:
            x (Tensor): Input tensor [batch_size, channels, time, height, width]
            
        Returns:
            Tensor: Transformed tensor
        """
        batch_size, channels, time, height, width = x.shape
        
        # Normalize and clamp velocity for stability
        v = torch.clamp(self.velocity, -self.max_velocity, self.max_velocity)
        v_magnitude = torch.norm(v)
        
        # If velocity is very small, skip transformation
        if v_magnitude < 1e-6:
            return x
        
        gamma = 1.0 / torch.sqrt(1.0 - v_magnitude**2)
        
        # Create coordinate grids
        t_coords = torch.linspace(0, 1, time, device=x.device)
        y_coords = torch.linspace(0, 1, height, device=x.device)
        x_coords = torch.linspace(0, 1, width, device=x.device)
        
        grid_t, grid_y, grid_x = torch.meshgrid(t_coords, y_coords, x_coords, indexing='ij')
        
        # Create sampling coordinates with relativistic transformation
        # These represent how coordinates are transformed due to relativistic effects
        
        # Lorentz transformation of spacetime coordinates
        # t' = γ(t - v·x)
        # x' = x + (γ-1)v(v·x)/v² - γvt
        
        # Stack spatial coordinates
        coords = torch.stack([grid_t, grid_y, grid_x], dim=-1)  # [time, height, width, 3]
        
        # Calculate the dot product v·x
        v_dot_x = (coords * v).sum(dim=-1)  # [time, height, width]
        
        # Transform time coordinate
        t_prime = gamma * (grid_t - v[0] * v_dot_x)
        
        # Transform spatial coordinates
        y_prime = grid_y + (gamma - 1) * v[1] * v_dot_x / (v_magnitude**2 + 1e-8) - gamma * v[1] * grid_t
        x_prime = grid_x + (gamma - 1) * v[2] * v_dot_x / (v_magnitude**2 + 1e-8) - gamma * v[2] * grid_t
        
        # Normalize transformed coordinates to [-1, 1] range for grid_sample
        t_norm = 2.0 * t_prime - 1.0
        y_norm = 2.0 * y_prime - 1.0
        x_norm = 2.0 * x_prime - 1.0
        
        # Stack normalized coordinates
        grid = torch.stack([x_norm, y_norm, t_norm], dim=-1)  # [time, height, width, 3]
        
        # Reshape grid for grid_sample
        grid = grid.view(1, time, height, width, 3).expand(batch_size, -1, -1, -1, -1)
        
        # Reshape x for grid_sample (N, C, D, H, W)
        # Use grid_sample for differentiable interpolation
        # This is a 3D analog of the 2D grid_sample
        x_transformed = F.grid_sample(
            x, 
            grid, 
            mode='bilinear', 
            padding_mode='zeros', 
            align_corners=True
        )
        
        return x_transformed
