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
    Self-attention mechanism incorporating relativistic time dilation and distortion.
    
    This attention mechanism is inspired by the Terrell-Penrose effect, where
    rapidly moving objects appear rotated rather than contracted. Similarly,
    this module implements attention where each attention head operates in a 
    different "reference frame" with its own relativistic velocity parameter.
    
    This creates a multi-perspective attention mechanism where information
    between tokens is processed as if affected by relativistic distortions
    across different reference frames.
    
    Args:
        hidden_dim (int): Dimension of input features
        num_heads (int, optional): Number of attention heads. Defaults to 8.
        dropout (float, optional): Dropout probability. Defaults to 0.1.
        max_velocity (float, optional): Maximum velocity parameter (0-1). Defaults to 0.9.
        bias (bool, optional): Whether to include bias terms. Defaults to True.
    """
    
    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1,
                 max_velocity: float = 0.9, bias: bool = True):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert self.head_dim * num_heads == hidden_dim, "hidden_dim must be divisible by num_heads"
        
        # Standard attention components
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        
        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        
        # Relativistic parameters - each head has its own "velocity"
        # initialized to linearly increasing values
        velocities = torch.linspace(0.1, max_velocity, num_heads)
        self.velocity = nn.Parameter(velocities)
        
        # Initialize parameters
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize attention parameters."""
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1/math.sqrt(2))
        nn.init.xavier_uniform_(self.k_proj.weight, gain=1/math.sqrt(2))
        nn.init.xavier_uniform_(self.v_proj.weight, gain=1/math.sqrt(2))
        nn.init.xavier_uniform_(self.out_proj.weight)
        
        if self.q_proj.bias is not None:
            nn.init.zeros_(self.q_proj.bias)
            nn.init.zeros_(self.k_proj.bias)
            nn.init.zeros_(self.v_proj.bias)
            nn.init.zeros_(self.out_proj.bias)
    
    def forward(self, x: Tensor, attention_mask: Optional[Tensor] = None, 
                positions: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass of relativistic self-attention.
        
        Args:
            x (Tensor): Input tensor of shape [batch_size, seq_len, hidden_dim]
            attention_mask (Tensor, optional): Attention mask of shape [batch_size, seq_len].
                                              1 indicates value token, 0 indicates padding.
                                              Defaults to None.
            positions (Tensor, optional): Position tensor for tokens [batch_size, seq_len, dim].
                                         Used to compute "spacetime" distances between tokens.
                                         Defaults to None.
            
        Returns:
            Tensor: Output tensor of shape [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len, _ = x.size()
        
        # Compute query, key, value tensors
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose for matrix multiplication: [batch, heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention scores
        scale = self.head_dim ** -0.5
        attn_weights = torch.matmul(q, k.transpose(-1, -2)) * scale  # [batch, heads, seq_len, seq_len]
        
        # Apply relativistic effects to attention weights for each head
        if positions is not None:
            attn_weights = self._apply_relativistic_effects(attn_weights, positions)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Convert mask of shape [batch_size, seq_len] to [batch_size, 1, 1, seq_len]
            mask = attention_mask.unsqueeze(1).unsqueeze(2).to(dtype=torch.bool)
            
            # Replace masked positions with large negative value
            attn_weights = attn_weights.masked_fill(~mask, float("-inf"))
        
        # Normalized attention weights
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)  # [batch, heads, seq_len, head_dim]
        
        # Transpose back: [batch, seq_len, heads, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        
        # Combine heads
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_dim)
        
        # Project to output
        output = self.out_proj(attn_output)
        output = self.output_dropout(output)
        
        return output
    
    def _apply_relativistic_effects(self, attn_weights: Tensor, positions: Tensor) -> Tensor:
        """
        Apply relativistic effects to attention weights.
        
        Args:
            attn_weights (Tensor): Raw attention weights [batch, heads, seq_len, seq_len]
            positions (Tensor): Token positions [batch, seq_len, dim]
            
        Returns:
            Tensor: Modified attention weights with relativistic effects
        """
        batch_size, num_heads, seq_len, _ = attn_weights.size()
        
        # Compute "spacetime distances" between token positions
        # In a simplified relativistic model, the distance affects information propagation
        if positions.size(-1) >= 2:
            # Use at least 2D positions to calculate distances
            pos_diffs = positions.unsqueeze(2) - positions.unsqueeze(1)  # [batch, seq, seq, dim]
            distances = torch.norm(pos_diffs, dim=-1)  # [batch, seq, seq]
        else:
            # If positions are 1D, use simple differences
            pos_diffs = positions.unsqueeze(2) - positions.unsqueeze(1)  # [batch, seq, seq, 1]
            distances = torch.abs(pos_diffs.squeeze(-1))  # [batch, seq, seq]
        
        # Normalize distances to [0, 1] range for stability
        if distances.max() > 0:
            distances = distances / (distances.max() + 1e-8)
        
        # For each head, apply its own relativistic transformation
        for h in range(num_heads):
            # Get head's velocity parameter (clamped for stability)
            v_h = torch.clamp(torch.abs(self.velocity[h]), 0.0, 0.999)
            
            # Compute relativistic gamma factor
            gamma = 1.0 / torch.sqrt(1.0 - v_h**2)
            
            # Apply Terrell-Penrose-inspired effect:
            # Attention between distant tokens is transformed by relativistic factor
            attn_modifier = torch.exp(-distances * gamma * v_h).unsqueeze(1)  # [batch, 1, seq, seq]
            
            # Apply the modifier to this head's attention weights
            head_selector = torch.zeros_like(attn_weights)
            head_selector[:, h:h+1, :, :] = 1.0
            
            # Blend the original weights with the transformed weights
            # This creates a "relativistic aberration" effect on attention
            attn_weights = attn_weights * (1.0 - head_selector) + \
                           attn_weights * attn_modifier * head_selector
        
        return attn_weights


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
