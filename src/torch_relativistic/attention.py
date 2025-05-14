"""
Relativistic attention mechanisms inspired by the Terrell-Penrose effect.

This module provides attention mechanisms that incorporate concepts from
special relativity, particularly the Terrell-Penrose effect, into neural 
network attention. The key insight is modeling attention between tokens or
nodes as if the information exchange is affected by relativistic effects.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple, List, Union, Dict, Any, Callable
import math


class RelativisticSelfAttention(nn.Module):
    """
    Optimized self-attention mechanism incorporating relativistic time dilation and distortion.
    
    This attention mechanism is inspired by the Terrell-Penrose effect, where
    rapidly moving objects appear rotated rather than contracted. This implementation
    uses parameter sharing, vectorized operations, and rotary position embeddings
    for better efficiency and performance.
    
    Each attention head operates in a different "reference frame" with its own
    relativistic velocity parameter, but shares transformation parameters.
    
    Args:
        hidden_dim (int): Dimension of input features
        num_heads (int, optional): Number of attention heads. Defaults to 8.
        dropout (float, optional): Dropout probability. Defaults to 0.1.
        max_velocity (float, optional): Maximum velocity parameter (0-1). Defaults to 0.9.
        bias (bool, optional): Whether to include bias terms. Defaults to True.
        pre_norm (bool, optional): Whether to use pre-normalization. Defaults to True.
    """
    
    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1,
                 max_velocity: float = 0.9, bias: bool = True, pre_norm: bool = True):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.pre_norm = pre_norm
        self.scale = self.head_dim ** -0.5
        
        assert self.head_dim * num_heads == hidden_dim, "hidden_dim must be divisible by num_heads"
        
        # Shared projection layers (parameter sharing)
        self.qkv_proj = nn.Linear(hidden_dim, hidden_dim * 3, bias=bias)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        
        # Pre-normalization layer for stability
        if pre_norm:
            self.norm = nn.LayerNorm(hidden_dim)
        
        # Dropout layers
        self.attn_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        
        # Relativistic parameters with better initialization
        # Now parameterized as a 2D tensor for more expressive velocity profiles
        velocity_base = torch.linspace(0.1, max_velocity, num_heads)
        self.velocity_scale = nn.Parameter(torch.ones(num_heads, 1))
        self.velocity_bias = nn.Parameter(velocity_base.unsqueeze(1))
        
        # Rotary Position Embedding parameters for relativistic position modeling
        self.rope_freqs = self._get_rope_frequencies(self.head_dim)
        
        # Initialize parameters
        self._reset_parameters()
    
    def _get_rope_frequencies(self, dim: int, base: int = 10000) -> torch.Tensor:
        """Generate frequency bands for rotary embeddings."""
        freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        return freqs
    
    def _apply_rotary_pos_emb(self, q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary position embeddings to queries and keys."""
        device = q.device
        t = positions.float().to(device)
        
        # Get frequencies for each position
        freqs = self.rope_freqs.to(device)
        freqs = freqs.view(1, 1, -1)  # [1, 1, dim/2]
        
        # Create sinusoidal pattern
        theta = torch.outer(t, freqs)  # [seq_len, dim/2]
        sin = torch.sin(theta).unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, dim/2]
        cos = torch.cos(theta).unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, dim/2]
        
        # Reshape for broadcasting
        dim_half = self.head_dim // 2
        reshape_dim = (q.shape[0], q.shape[1], q.shape[2], dim_half)
        
        # For complex number multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        # Extract even and odd dimensions
        q_even = q[:, :, :, 0::2].reshape(reshape_dim)
        q_odd = q[:, :, :, 1::2].reshape(reshape_dim)
        k_even = k[:, :, :, 0::2].reshape(reshape_dim)
        k_odd = k[:, :, :, 1::2].reshape(reshape_dim)
        
        # Apply rotation using complex number multiplication
        q_out_even = q_even * cos - q_odd * sin
        q_out_odd = q_odd * cos + q_even * sin
        k_out_even = k_even * cos - k_odd * sin
        k_out_odd = k_odd * cos + k_even * sin
        
        # Interleave outputs
        q_out = torch.zeros_like(q)
        k_out = torch.zeros_like(k)
        
        q_out[:, :, :, 0::2] = q_out_even
        q_out[:, :, :, 1::2] = q_out_odd
        k_out[:, :, :, 0::2] = k_out_even
        k_out[:, :, :, 1::2] = k_out_odd
        
        return q_out, k_out
    
    def _reset_parameters(self):
        """Initialize attention parameters with optimal distributions."""
        # Xavier initialization for QKV projection
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        if self.qkv_proj.bias is not None:
            nn.init.zeros_(self.qkv_proj.bias)
        
        # Output projection with slightly smaller weights
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.9)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)
    
    def forward(self, x: Tensor, attention_mask: Optional[Tensor] = None, 
                positions: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass of optimized relativistic self-attention.
        
        Args:
            x (Tensor): Input tensor of shape [batch_size, seq_len, hidden_dim]
            attention_mask (Tensor, optional): Attention mask of shape [batch_size, seq_len].
                                              1 indicates value token, 0 indicates padding.
            positions (Tensor, optional): Position tensor for tokens [batch_size, seq_len, dim].
                                         Used to compute "spacetime" distances between tokens.
            
        Returns:
            Tensor: Output tensor of shape [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len, _ = x.size()
        
        # Pre-normalization if enabled
        if self.pre_norm:
            x = self.norm(x)
        
        # Generate position indices if not provided
        if positions is None:
            positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        
        # Project inputs to queries, keys, values in one go (efficient)
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, heads, seq_len, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Apply rotary position embeddings
        if positions is not None:
            flat_positions = positions.reshape(batch_size * seq_len)
            pos_indices = flat_positions.reshape(batch_size, seq_len)
            q, k = self._apply_rotary_pos_emb(q, k, pos_indices)
        
        # Calculate attention scores with optimized scaling
        attn_weights = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        
        # Apply relativistic effects using vectorized operations
        if positions is not None:
            attn_weights = self._apply_relativistic_effects_vectorized(attn_weights, positions)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Convert mask of shape [batch_size, seq_len] to [batch_size, 1, 1, seq_len]
            mask = attention_mask.unsqueeze(1).unsqueeze(2).to(dtype=torch.bool)
            
            # Replace masked positions with large negative value
            attn_weights = attn_weights.masked_fill(~mask, float("-inf"))
        
        # Normalized attention weights with stable softmax
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)  # [batch, heads, seq_len, head_dim]
        
        # Reshape and project to output dimension
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_dim)
        output = self.out_proj(attn_output)
        output = self.output_dropout(output)
        
        return output
    
    def _apply_relativistic_effects_vectorized(self, attn_weights: Tensor, positions: Tensor) -> Tensor:
        """
        Apply relativistic effects to attention weights using vectorized operations.
        
        Args:
            attn_weights (Tensor): Raw attention weights [batch, heads, seq_len, seq_len]
            positions (Tensor): Token positions [batch, seq_len, dim]
            
        Returns:
            Tensor: Modified attention weights with relativistic effects
        """
        batch_size, num_heads, seq_len, _ = attn_weights.size()
        
        # Compute pairwise distances between all tokens more efficiently
        if positions.dim() > 2:
            # Use multi-dimensional positions if available
            pos_extended = positions.unsqueeze(2)  # [batch, seq, 1, dim]
            distances = torch.norm(
                pos_extended - positions.unsqueeze(1),  # [batch, 1, seq, dim]
                dim=-1, p=2
            )  # [batch, seq, seq]
        else:
            # Use 1D positions
            pos_diff = positions.unsqueeze(2) - positions.unsqueeze(1)  # [batch, seq, seq]
            distances = torch.abs(pos_diff)
        
        # Normalize distances for numerical stability
        if torch.max(distances) > 0:
            distances = distances / (torch.max(distances) + 1e-8)
        
        # Calculate velocity for each head using the parametric form
        velocity = torch.clamp(self.velocity_scale * self.velocity_bias, 0.0, 0.999)  # [heads, 1]
        
        # Calculate gamma (Lorentz factor) for each head
        gamma = 1.0 / torch.sqrt(1.0 - velocity**2 + 1e-8)  # [heads, 1]
        
        # Prepare distances for broadcasting
        distances = distances.unsqueeze(1)  # [batch, 1, seq, seq]
        
        # Calculate relativistic modifiers for all heads simultaneously
        # This creates a tensor of shape [batch, heads, seq, seq]
        rel_modifiers = torch.exp(
            -distances * gamma.view(1, num_heads, 1, 1) * velocity.view(1, num_heads, 1, 1)
        )
        
        # Apply modifiers to attention weights
        modified_attn = attn_weights * rel_modifiers
        
        # Rescale to ensure proper probability distribution
        row_sums = modified_attn.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        modified_attn = modified_attn / row_sums
        
        return modified_attn


class RelativisticPositionalEncoding(nn.Module):
    """
    Positional encoding with relativistic considerations.
    
    This module extends standard positional encodings by incorporating
    relativistic concepts, where the effective distance between positions
    is modulated by a learnable "velocity" parameter. This creates a
    non-uniform encoding of positions, with effective compression
    or dilation based on relativistic "proper distance" concepts.
    
    Args:
        hidden_dim (int): Embedding dimension
        max_len (int, optional): Maximum sequence length to pre-compute. Defaults to 5000.
        dropout (float, optional): Dropout probability. Defaults to 0.1.
    """
    
    def __init__(self, hidden_dim: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        self.dropout = nn.Dropout(p=dropout)
        
        # Relativistic parameter (learnable)
        self.velocity = nn.Parameter(torch.Tensor([0.5]))
        
        # Create standard positional encoding buffer
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2) * -(math.log(10000.0) / hidden_dim))
        
        # Initialize buffer for positional encodings
        # We'll transform these with relativistic effects during forward pass
        pe = torch.zeros(1, max_len, hidden_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe_base', pe)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Add relativistic positional encodings to the input.
        
        Args:
            x (Tensor): Input tensor [batch_size, seq_len, hidden_dim]
            
        Returns:
            Tensor: Input with added positional encodings
        """
        seq_len = x.size(1)
        
        if seq_len > self.max_len:
            raise ValueError(f"Input sequence length {seq_len} exceeds maximum length {self.max_len}")
        
        # Compute relativistic position encodings
        v = torch.clamp(self.velocity, 0.0, 0.999)
        gamma = 1.0 / torch.sqrt(1.0 - v**2)
        
        # Create position indices
        positions = torch.arange(seq_len, device=x.device).float()
        
        # Apply relativistic dilation to positions
        # This creates non-uniform position encodings based on relativistic concepts
        rel_positions = positions / gamma  # Contracted positions
        
        # Interpolate from base positional encodings
        rel_positions = rel_positions.clamp(0, self.max_len - 1)
        rel_idx_low = rel_positions.floor().long()
        rel_idx_high = (rel_idx_low + 1).clamp(max=self.max_len - 1)
        rel_weight_high = rel_positions - rel_positions.floor()
        rel_weight_low = 1.0 - rel_weight_high
        
        # Interpolate positional encodings
        pe = self.pe_base[0, rel_idx_low] * rel_weight_low.unsqueeze(-1) + \
             self.pe_base[0, rel_idx_high] * rel_weight_high.unsqueeze(-1)
        
        # Add to input and apply dropout
        return self.dropout(x + pe.unsqueeze(0))


class RelativisticTemporalAttention(nn.Module):
    """
    Attention mechanism for sequences with relativistic time dilation effects.
    
    This module is designed for temporal sequences where the Terrell-Penrose effect
    inspires a non-uniform processing of time. Different parts of a sequence are
    processed with different "time dilation" factors, allowing the network to
    automatically focus on relevant temporal scales.
    
    Args:
        hidden_dim (int): Feature dimension
        num_heads (int, optional): Number of attention heads. Defaults to 8.
        dropout (float, optional): Dropout probability. Defaults to 0.1.
        max_velocity (float, optional): Maximum "velocity" parameter. Defaults to 0.9.
    """
    
    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1,
                 max_velocity: float = 0.9):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Standard attention components
        self.self_attn = RelativisticSelfAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            max_velocity=max_velocity
        )
        
        # Temporal processing components
        self.time_embed = nn.Linear(1, hidden_dim)
        self.time_attn_norm = nn.LayerNorm(hidden_dim)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Time dilation parameters
        self.time_dilation = nn.Parameter(torch.zeros(1))
    
    def forward(self, x: Tensor, timestamps: Optional[Tensor] = None, 
                mask: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass with relativistic temporal attention.
        
        Args:
            x (Tensor): Input features [batch_size, seq_len, hidden_dim]
            timestamps (Tensor, optional): Timestamps for each element [batch_size, seq_len].
                                          Defaults to None (uses position indices).
            mask (Tensor, optional): Attention mask [batch_size, seq_len]. 
                                    Defaults to None.
            
        Returns:
            Tensor: Processed features [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len, _ = x.size()
        
        # Generate timestamps if not provided
        if timestamps is None:
            timestamps = torch.arange(seq_len, device=x.device).float()
            timestamps = timestamps.expand(batch_size, seq_len)
        
        # Apply relativistic time dilation to timestamps
        # v represents "velocity through time" affecting how time intervals are perceived
        v = torch.tanh(self.time_dilation) * 0.99  # Constrain < 1
        gamma = 1.0 / torch.sqrt(1.0 - v**2 + 1e-8)
        
        # Transform timestamps to account for relativistic effects
        rel_timestamps = timestamps / gamma
        rel_timestamps = rel_timestamps.unsqueeze(-1)  # [batch, seq_len, 1]
        
        # Embed dilated timestamps and add to input
        time_features = self.time_embed(rel_timestamps)
        x = x + time_features
        
        # Self-attention with relativistic effects
        residual = x
        x = self.time_attn_norm(x)
        x = self.self_attn(x, attention_mask=mask, positions=rel_timestamps)
        x = residual + x
        
        # Feed-forward network
        residual = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = residual + x
        
        return x
