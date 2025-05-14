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
    Optimized Spiking Neural Network architecture inspired by the Terrell-Penrose effect.
    
    This SNN architecture integrates relativistic concepts through parameter sharing,
    attention mechanisms and adaptive time-dependent weighting. The implementation
    uses vectorized operations for efficient time step computation and surrogate
    gradients for stable training.
    
    Args:
        input_size (int): Input dimension
        hidden_size (int): Size of hidden layers
        output_size (int): Output dimension
        simulation_steps (int, optional): Number of time steps to simulate. Default: 100.
        beta (float, optional): Membrane decay factor. Default: 0.9.
        dropout (float, optional): Dropout probability. Default: 0.1.
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int, 
                 simulation_steps: int = 100, beta: float = 0.9, dropout: float = 0.1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.simulation_steps = simulation_steps
        
        # Gemeinsamer Basis-Neuron mit relativistischen Effekten
        self.base_neuron = RelativisticLIFNeuron(max(input_size, hidden_size), beta=beta)
        
        # Adaptive neuronale Parameter
        self.input_threshold = nn.Parameter(torch.ones(1) * 1.0)
        self.hidden_threshold = nn.Parameter(torch.ones(1) * 0.8)
        
        # Trainierbare Zeitkonstanten
        self.input_beta = nn.Parameter(torch.ones(1) * beta)
        self.hidden_beta = nn.Parameter(torch.ones(1) * beta)
        
        # Verbindungen zwischen Schichten
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
        # Batch-Normalisierung für stabileres Training
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(output_size)
        
        # Dropout für Regularisierung
        self.dropout = nn.Dropout(dropout)
        
        # Aufmerksamkeitsmechanismus für zeitliche Integration
        self.time_attention = nn.Parameter(torch.ones(simulation_steps) / simulation_steps)
        
        # Relativistische Gewichtungsfaktoren
        self.lorentz_factor = nn.Parameter(torch.tensor([0.5]))
        
        # Surrogate Gradient Funktionsparameter
        self.surrogate_scale = nn.Parameter(torch.tensor([10.0]))
    
    def surrogate_spike_function(self, x: Tensor, threshold: Tensor) -> Tensor:
        """
        Differentiable approximation of the spike function (FastSigmoid).
        
        Args:
            x (Tensor): Membrane potentials
            threshold (Tensor): Threshold for spikes
            
        Returns:
            Tensor: Spike output with surrogate gradients
        """
        # Im Forward-Pass: Binäre Spikes
        spikes = (x > threshold).float()
        
        # Im Backward-Pass: FastSigmoid als Surrogate-Gradient
        if self.training:
            scale = self.surrogate_scale
            x_normalized = (x - threshold) * scale
            grad_scale = torch.sigmoid(x_normalized) * (1 - torch.sigmoid(x_normalized)) * scale
            
            # Gradient-Ersetzung durch benutzerdefinierte Autograd-Funktion
            class SurrogateSpike(torch.autograd.Function):
                @staticmethod
                def forward(ctx, input, grad_scale):
                    ctx.save_for_backward(grad_scale)
                    return (input > 0).float()
                
                @staticmethod
                def backward(ctx, grad_output):
                    grad_scale, = ctx.saved_tensors
                    return grad_output * grad_scale, None
            
            spikes = SurrogateSpike.apply(x - threshold, grad_scale)
            
        return spikes
    
    def forward(self, x: Tensor, initial_state: Optional[Dict[str, Tuple[Tensor, Tensor]]] = None) -> Tensor:
        """
        Forward pass of the SNN with vectorized time step computation and attention.
        
        Args:
            x (Tensor): Input tensor [batch_size, input_size] or [batch_size, time_steps, input_size]
            initial_state (Dict, optional): Initial states for neurons. Default: None.
            
        Returns:
            Tensor: Network output [batch_size, output_size]
        """
        # Handle both static and temporal inputs
        if x.dim() == 2:
            batch_size, _ = x.size()
            x = x.unsqueeze(1).expand(-1, self.simulation_steps, -1)
        elif x.dim() == 3:
            batch_size, time_steps, _ = x.size()
            if time_steps < self.simulation_steps:
                padding = torch.zeros(batch_size, self.simulation_steps - time_steps, 
                                     self.input_size, device=x.device)
                x = torch.cat([x, padding], dim=1)
            elif time_steps > self.simulation_steps:
                x = x[:, :self.simulation_steps, :]
        else:
            raise ValueError(f"Expected input dimensions 2 or 3, got {x.dim()}")
        
        batch_size = x.size(0)
        device = x.device
        
        # initialize neuron states
        if initial_state is None:
            input_membrane = torch.zeros(batch_size, device=device)
            input_spikes = torch.zeros(batch_size, device=device)
            hidden_membrane = torch.zeros(batch_size, device=device)
            hidden_spikes = torch.zeros(batch_size, device=device)
        else:
            (input_membrane, input_spikes) = initial_state.get('input_layer', 
                                                             (torch.zeros(batch_size, device=device), 
                                                              torch.zeros(batch_size, device=device)))
            (hidden_membrane, hidden_spikes) = initial_state.get('hidden_layer', 
                                                               (torch.zeros(batch_size, device=device), 
                                                                torch.zeros(batch_size, device=device)))
        
        # output storage for all time steps
        all_outputs = []
        all_hidden_spikes = []
        
        # calculate relativistic Lorentz factor
        v = torch.clamp(self.lorentz_factor, 0.0, 0.999)  # limit to < c
        gamma = 1.0 / torch.sqrt(1.0 - v**2)
        
        # calculate relativistic arrival times with vectorization
        delays = gamma * torch.abs(self.base_neuron.causal_distances[:self.input_size]) * v
        input_delay_factors = torch.exp(-delays).unsqueeze(0)  # [1, input_size]
        
        hidden_delays = gamma * torch.abs(self.base_neuron.causal_distances[:self.hidden_size]) * v
        hidden_delay_factors = torch.exp(-hidden_delays).unsqueeze(0)  # [1, hidden_size]
        
        # simulate SNN for multiple time steps
        for t in range(self.simulation_steps):
            # input layer with relativistic processing
            effective_inputs = x[:, t] * input_delay_factors
            input_membrane = input_membrane * self.input_beta + torch.sum(effective_inputs, dim=1)
            input_spikes = self.surrogate_spike_function(input_membrane, self.input_threshold)
            input_membrane = input_membrane * (1.0 - input_spikes)
            
            # hidden layer
            hidden_inputs = self.fc1(input_spikes)
            # BatchNorm only during training
            if self.training and batch_size > 1:  # BatchNorm requires more than one sample
                hidden_inputs = self.bn1(hidden_inputs)
            
            effective_hidden = hidden_inputs * hidden_delay_factors[:, :self.hidden_size]
            hidden_membrane = hidden_membrane * self.hidden_beta + torch.sum(effective_hidden, dim=1)
            hidden_spikes = self.surrogate_spike_function(hidden_membrane, self.hidden_threshold)
            hidden_membrane = hidden_membrane * (1.0 - hidden_spikes)
            
            # collect hidden spikes for analysis
            all_hidden_spikes.append(hidden_spikes)
            
            # output layer with dropout
            output = self.fc2(self.dropout(hidden_spikes) if self.training else hidden_spikes)
            if self.training and batch_size > 1:
                output = self.bn2(output)
            
            all_outputs.append(output)
        
        # stack output over time dimension
        all_outputs = torch.stack(all_outputs, dim=1)  # [batch_size, time_steps, output_size]
        
        # apply attention weighting over time
        attention_weights = F.softmax(self.time_attention, dim=0)
        
        # time-dependent relativistic weighting
        time_steps = torch.arange(self.simulation_steps, device=device).float()
        relativistic_weights = torch.exp(-(gamma - 1.0) * time_steps)
        combined_weights = attention_weights * relativistic_weights
        combined_weights = combined_weights / combined_weights.sum()  # normalize weights
        
        # apply weighted summation over time
        weighted_output = torch.sum(all_outputs * combined_weights.view(1, -1, 1), dim=1)
        
        return weighted_output
    
    def get_spike_history(self, x: Tensor) -> Dict[str, torch.Tensor]:
        """
        Get spike history for visualization and analysis.
        
        Args:
            x (Tensor): Input tensor
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing spike histories
        """
        # This function implements its own simulation
        # to capture the complete spike history
        
        batch_size = x.size(0)
        device = x.device
        
        # ensure input has time dimension
        if x.dim() == 2:
            x = x.unsqueeze(1).expand(-1, self.simulation_steps, -1)
        elif x.dim() == 3:
            time_steps = x.size(1)
            if time_steps < self.simulation_steps:
                padding = torch.zeros(batch_size, self.simulation_steps - time_steps, 
                                     self.input_size, device=device)
                x = torch.cat([x, padding], dim=1)
            elif time_steps > self.simulation_steps:
                x = x[:, :self.simulation_steps, :]
        
        # initialize neuron states
        input_membrane = torch.zeros(batch_size, device=device)
        input_spikes = torch.zeros(batch_size, device=device)
        hidden_membrane = torch.zeros(batch_size, device=device)
        hidden_spikes = torch.zeros(batch_size, device=device)
        
        # calculate relativistic factors
        v = torch.clamp(self.lorentz_factor, 0.0, 0.999)
        gamma = 1.0 / torch.sqrt(1.0 - v**2)
        
        delays = gamma * torch.abs(self.base_neuron.causal_distances[:self.input_size]) * v
        input_delay_factors = torch.exp(-delays).unsqueeze(0)
        
        hidden_delays = gamma * torch.abs(self.base_neuron.causal_distances[:self.hidden_size]) * v
        hidden_delay_factors = torch.exp(-hidden_delays).unsqueeze(0)
        
        # capture spike history
        input_spikes_history = []
        hidden_spikes_history = []
        
        # perform simulation
        for t in range(self.simulation_steps):
            # input layer
            effective_inputs = x[:, t] * input_delay_factors
            input_membrane = input_membrane * self.input_beta + torch.sum(effective_inputs, dim=1)
            input_spikes = (input_membrane > self.input_threshold).float()  # use hard threshold for visualization
            input_membrane = input_membrane * (1.0 - input_spikes)
            input_spikes_history.append(input_spikes)
            
            # hidden layer
            hidden_inputs = self.fc1(input_spikes)
            effective_hidden = hidden_inputs * hidden_delay_factors[:, :self.hidden_size]
            hidden_membrane = hidden_membrane * self.hidden_beta + torch.sum(effective_hidden, dim=1)
            hidden_spikes = (hidden_membrane > self.hidden_threshold).float()  # Verwende harte Schwelle für Visualisierung
            hidden_membrane = hidden_membrane * (1.0 - hidden_spikes)
            hidden_spikes_history.append(hidden_spikes)
        
        # stack over time dimension
        input_spikes_history = torch.stack(input_spikes_history, dim=1)   # [batch_size, time_steps, input_size]
        hidden_spikes_history = torch.stack(hidden_spikes_history, dim=1) # [batch_size, time_steps, hidden_size]
        
        return {
            'input_spikes': input_spikes_history,
            'hidden_spikes': hidden_spikes_history,
            'lorentz_factor': gamma.item(),
            'attention_weights': F.softmax(self.time_attention, dim=0).detach().cpu().numpy()
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
