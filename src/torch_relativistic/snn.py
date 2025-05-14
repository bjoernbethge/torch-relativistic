"""
Relativistic Spiking Neural Network modules inspired by the Terrell-Penrose effect.

This module provides SNN components that incorporate relativistic concepts into
spiking neural networks. The key insight is that light travel time effects in the
Terrell-Penrose effect have analogies to signal propagation delays in SNNs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple, List, Union, Dict, Any


class RelativisticLIFNeuron(nn.Module):
    """
    Leaky Integrate-and-Fire neuron with relativistic time effects.
    
    This spiking neuron model incorporates concepts from relativity theory,
    particularly inspired by the Terrell-Penrose effect, where different signal
    arrival times lead to perceptual transformations. In this neuron model,
    inputs from different sources reach the neuron with different effective delays
    based on their "causal distance" and a relativistic velocity parameter.
    
    Args:
        input_size (int): Number of input connections to the neuron
        threshold (float, optional): Firing threshold. Defaults to 1.0.
        beta (float, optional): Membrane potential decay factor. Defaults to 0.9.
        dt (float, optional): Time step size. Defaults to 1.0.
        requires_grad (bool, optional): Whether causal parameters are learnable. Defaults to True.
        
    Attributes:
        causal_distances (Parameter): Learnable distances representing causal relationships
        velocity (Parameter): Relativistic velocity parameter (as fraction of c)
    """
    
    def __init__(self, input_size: int, threshold: float = 1.0, beta: float = 0.9, 
                 dt: float = 1.0, requires_grad: bool = True):
        super().__init__()
        self.input_size = input_size
        self.threshold = threshold
        self.beta = beta
        self.dt = dt
        
        # Learnable causal structure between inputs
        # (abstract representation of spacetime distances)
        self.causal_distances = nn.Parameter(
            torch.randn(input_size) * 0.01,
            requires_grad=requires_grad
        )
        
        # Relativistic velocity as learnable parameter
        # (initialized to 0.5c)
        self.velocity = nn.Parameter(
            torch.Tensor([0.5]),
            requires_grad=requires_grad
        )
    
    def forward(self, input_spikes: Tensor, prev_state: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """
        Forward pass of the relativistic LIF neuron.
        
        Args:
            input_spikes (Tensor): Incoming spikes [batch_size, input_size]
            prev_state (Tuple[Tensor, Tensor]): (membrane potential, previous spikes)
            
        Returns:
            Tuple[Tensor, Tuple[Tensor, Tensor]]: (output spikes, (new membrane potential, output spikes))
        """
        prev_potential, prev_spikes = prev_state
        batch_size = input_spikes.size(0)
        
        # Calculate relativistic time dilation
        v = torch.clamp(self.velocity, 0.0, 0.999)  # Constrain to < c
        gamma = 1.0 / torch.sqrt(1.0 - v**2)
        
        # Relativistic arrival times for signals from different inputs
        # (inspired by different light travel times in Terrell-Penrose effect)
        arrival_delays = gamma * torch.abs(self.causal_distances) * v
        delay_factors = torch.exp(-arrival_delays)  # Exponential attenuation with delay
        
        # Apply causality-based weighting to input spikes
        # This simulates that information from different "distances" is processed differently
        effective_inputs = input_spikes * delay_factors.unsqueeze(0)
        
        # Standard LIF dynamics
        new_potential = prev_potential * self.beta + torch.sum(effective_inputs, dim=1)
        
        # Spike generation
        new_spikes = (new_potential > self.threshold).float()
        
        # Reset potential after spike
        new_potential = new_potential * (1.0 - new_spikes)
        
        return new_spikes, (new_potential, new_spikes)
    
    def init_state(self, batch_size: int, device: torch.device = None) -> Tuple[Tensor, Tensor]:
        """
        Initialize the neuron state.
        
        Args:
            batch_size (int): Batch size
            device (torch.device, optional): Device to create tensors on. Defaults to None.
            
        Returns:
            Tuple[Tensor, Tensor]: (initial membrane potential, initial spikes)
        """
        device = device or self.causal_distances.device
        return (
            torch.zeros(batch_size, device=device),
            torch.zeros(batch_size, device=device)
        )


class TerrellPenroseSNN(nn.Module):
    """
    Spiking Neural Network architecture inspired by the Terrell-Penrose effect.
    
    This SNN architecture incorporates relativistic concepts from the Terrell-Penrose
    effect, where information arrival times lead to perceptual transformations. In this
    network, different neural pathways operate with different effective time dilations,
    allowing the network to process temporal information across multiple effective
    timescales simultaneously.
    
    Args:
        input_size (int): Input dimension
        hidden_size (int): Hidden layer size
        output_size (int): Output dimension 
        simulation_steps (int, optional): Number of time steps to simulate. Defaults to 100.
        beta (float, optional): Membrane potential decay factor. Defaults to 0.9.
        
    Note:
        The network integrates outputs over time with relativistically-inspired
        weighting, giving different importance to spikes at different time points.
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int, 
                 simulation_steps: int = 100, beta: float = 0.9):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.simulation_steps = simulation_steps
        
        # Input layer with relativistic information processing
        self.input_layer = RelativisticLIFNeuron(
            input_size, 
            threshold=1.0,
            beta=beta
        )
        
        # Hidden layer with relativistic effects
        self.hidden_layer = RelativisticLIFNeuron(
            hidden_size, 
            threshold=0.8,
            beta=beta
        )
        
        # Connections between layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
        # Readout parameters
        self.output_scale = nn.Parameter(torch.ones(output_size))
        self.output_bias = nn.Parameter(torch.zeros(output_size))
    
    def forward(self, x: Tensor, initial_state: Optional[Dict[str, Tuple[Tensor, Tensor]]] = None) -> Tensor:
        """
        Forward pass of the SNN.
        
        Args:
            x (Tensor): Input tensor [batch_size, input_size] or [batch_size, time_steps, input_size]
            initial_state (Dict, optional): Initial states for neurons. Defaults to None.
            
        Returns:
            Tensor: Network output [batch_size, output_size]
        """
        # Handle both static and temporal inputs
        if x.dim() == 2:
            # Static input - repeat for all simulation steps
            batch_size, _ = x.size()
            x = x.unsqueeze(1).expand(-1, self.simulation_steps, -1)
        elif x.dim() == 3:
            batch_size, time_steps, _ = x.size()
            if time_steps < self.simulation_steps:
                # Pad with zeros if input has fewer time steps
                padding = torch.zeros(batch_size, self.simulation_steps - time_steps, 
                                      self.input_size, device=x.device)
                x = torch.cat([x, padding], dim=1)
            elif time_steps > self.simulation_steps:
                # Truncate if input has more time steps
                x = x[:, :self.simulation_steps, :]
        else:
            raise ValueError(f"Expected input dims 2 or 3, got {x.dim()}")
        
        batch_size = x.size(0)
        device = x.device
        
        # Initialize neuron states if not provided
        if initial_state is None:
            h1_state = self.input_layer.init_state(batch_size, device)
            h2_state = self.hidden_layer.init_state(batch_size, device)
        else:
            h1_state = initial_state.get('input_layer', self.input_layer.init_state(batch_size, device))
            h2_state = initial_state.get('hidden_layer', self.hidden_layer.init_state(batch_size, device))
        
        # Output storage
        outputs = []
        spikes_hidden = []
        
        # Simulate SNN for multiple time steps
        for t in range(self.simulation_steps):
            # Input layer
            out_1, h1_state = self.input_layer(x[:, t], h1_state)
            
            # Hidden layer
            hidden_inputs = self.fc1(out_1)
            out_2, h2_state = self.hidden_layer(hidden_inputs, h2_state)
            spikes_hidden.append(out_2)
            
            # Output layer
            current_output = self.fc2(out_2)
            outputs.append(current_output)
        
        # Stack outputs over time dimension
        outputs = torch.stack(outputs, dim=1)  # [batch_size, time_steps, output_size]
        
        # Apply relativistic time weighting
        # (later time steps receive different weights based on "relativistic velocity")
        v = torch.clamp(self.input_layer.velocity, 0.0, 0.999)
        gamma = 1.0 / torch.sqrt(1.0 - v**2)
        
        # Create time-step-dependent weights with relativistic inspiration
        time_steps = torch.arange(self.simulation_steps, device=device).float()
        time_weights = torch.exp(-(gamma - 1.0) * time_steps)
        time_weights = time_weights / time_weights.sum()  # Normalize weights
        
        # Apply weighted summation over time
        weighted_output = torch.sum(outputs * time_weights.view(1, -1, 1), dim=1)
        
        # Apply output scaling and bias
        return weighted_output * self.output_scale + self.output_bias
    
    def get_spike_history(self, x: Tensor) -> Dict[str, torch.Tensor]:
        """
        Get spike history for visualization and analysis.
        
        Args:
            x (Tensor): Input tensor
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing spike histories
        """
        batch_size = x.size(0)
        device = x.device
        
        # Initialize states
        h1_state = self.input_layer.init_state(batch_size, device)
        h2_state = self.hidden_layer.init_state(batch_size, device)
        
        # Make sure input has time dimension
        if x.dim() == 2:
            x = x.unsqueeze(1).expand(-1, self.simulation_steps, -1)
        
        # Record spike history
        input_spikes = []
        hidden_spikes = []
        
        # Run simulation
        for t in range(self.simulation_steps):
            # Input layer
            out_1, h1_state = self.input_layer(x[:, t], h1_state)
            input_spikes.append(out_1)
            
            # Hidden layer
            hidden_inputs = self.fc1(out_1)
            out_2, h2_state = self.hidden_layer(hidden_inputs, h2_state)
            hidden_spikes.append(out_2)
        
        # Stack over time dimension
        input_spikes = torch.stack(input_spikes, dim=1)  # [batch_size, time_steps, input_size]
        hidden_spikes = torch.stack(hidden_spikes, dim=1)  # [batch_size, time_steps, hidden_size]
        
        return {
            'input_spikes': input_spikes,
            'hidden_spikes': hidden_spikes
        }


class RelativeSynapticPlasticity(nn.Module):
    """
    Synaptic plasticity rule inspired by relativistic time effects.
    
    This module implements a learning rule for spiking neural networks that
    incorporates relativistic concepts. The key insight is that synaptic
    weight updates are affected by the "relativistic frame" of reference,
    which depends on the activity level in different parts of the network.
    
    Args:
        input_size (int): Size of presynaptic population
        output_size (int): Size of postsynaptic population
        learning_rate (float, optional): Base learning rate. Defaults to 0.01.
        max_velocity (float, optional): Maximum "velocity" parameter (0-1). Defaults to 0.9.
    """
    
    def __init__(self, input_size: int, output_size: int, learning_rate: float = 0.01,
                 max_velocity: float = 0.9):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.max_velocity = max_velocity
        
        # Synaptic weights
        self.weights = nn.Parameter(torch.randn(output_size, input_size) * 0.1)
        
        # Relativistic parameters
        self.velocity = nn.Parameter(torch.zeros(1))
        
        # Synaptic activity trackers
        self.register_buffer('pre_trace', torch.zeros(input_size))
        self.register_buffer('post_trace', torch.zeros(output_size))
        
        # Decay rates for traces
        self.pre_decay = 0.9
        self.post_decay = 0.9
    
    def forward(self, pre_spikes: Tensor) -> Tensor:
        """
        Forward pass computing postsynaptic activity.
        
        Args:
            pre_spikes (Tensor): Presynaptic spike vector [batch_size, input_size]
            
        Returns:
            Tensor: Postsynaptic potentials [batch_size, output_size]
        """
        # Calculate relativistic gamma factor
        v = torch.clamp(self.velocity, -self.max_velocity, self.max_velocity)
        gamma = 1.0 / torch.sqrt(1.0 - v**2)
        
        # Apply relativistic weight transformation
        # This represents how the effectiveness of synapses changes with network activity
        effective_weights = self.weights * gamma
        
        # Compute postsynaptic potentials
        post_activity = torch.matmul(pre_spikes, effective_weights.t())
        
        return post_activity
    
    def update_traces(self, pre_spikes: Tensor, post_spikes: Tensor):
        """
        Update activity traces for plasticity.
        
        Args:
            pre_spikes (Tensor): Presynaptic spike vector
            post_spikes (Tensor): Postsynaptic spike vector
        """
        with torch.no_grad():
            # Update presynaptic trace
            self.pre_trace = self.pre_trace * self.pre_decay + pre_spikes.mean(0)
            
            # Update postsynaptic trace
            self.post_trace = self.post_trace * self.post_decay + post_spikes.mean(0)
    
    def update_weights(self, pre_spikes: Tensor, post_spikes: Tensor):
        """
        Update synaptic weights based on relativistic STDP rule.
        
        Args:
            pre_spikes (Tensor): Presynaptic spike vector
            post_spikes (Tensor): Postsynaptic spike vector
        """
        # Current "velocity" is based on overall network activity
        v = torch.clamp(self.velocity, -self.max_velocity, self.max_velocity)
        gamma = 1.0 / torch.sqrt(1.0 - v**2)
        
        # Update traces
        self.update_traces(pre_spikes, post_spikes)
        
        # Relativistic STDP rule
        # The effective learning rate is modulated by gamma factor
        # representing how time dilates in different activity regimes
        with torch.no_grad():
            # Pre-post correlation
            dw = self.learning_rate * gamma * torch.outer(
                post_spikes.mean(0), 
                self.pre_trace
            )
            
            # Post-pre correlation (with relativistic time shift)
            dw -= self.learning_rate * gamma * torch.outer(
                self.post_trace,
                pre_spikes.mean(0)
            )
            
            # Update weights
            self.weights.add_(dw)
            
            # Update "velocity" based on overall activity
            activity_level = (pre_spikes.mean() + post_spikes.mean()) / 2
            target_v = torch.tanh(activity_level * 5) * self.max_velocity
            self.velocity.data = self.velocity.data * 0.9 + target_v * 0.1
