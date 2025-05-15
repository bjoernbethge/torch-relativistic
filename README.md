# torch-relativistic

A PyTorch extension that implements neural network components inspired by relativistic physics, particularly the Terrell-Penrose effect.

## Overview

RelativisticTorch provides neural network modules that incorporate concepts from special relativity into machine learning. The key insight is that the Terrell-Penrose effect, where rapidly moving objects appear rotated rather than contracted, can inspire novel information processing paradigms in neural networks.

This library includes:
- Relativistic Graph Neural Networks (GNNs)
- Relativistic Spiking Neural Networks (SNNs)
- Relativistic attention mechanisms
- Transformations inspired by special relativity

## Installation

```bash
pip install -e .
```

## Components

### Relativistic Graph Neural Networks

GNN modules that process information as if affected by relativistic phenomena:

```python
import torch
from relativistic_torch.gnn import RelativisticGraphConv, MultiObserverGNN

# Create a simple graph
num_nodes = 10
feature_dim = 16
edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 0],
                           [1, 0, 2, 1, 3, 2, 4, 3, 0, 4]], dtype=torch.long)
node_features = torch.randn(num_nodes, feature_dim)

# Create a relativistic GNN layer
conv = RelativisticGraphConv(
    in_channels=feature_dim,
    out_channels=32,
    max_relative_velocity=0.8
)

# Process the graph
output_features = conv(node_features, edge_index)
print(f\"Output shape: {output_features.shape}\")  # [10, 32]

# Multi-observer GNN processes the graph from multiple relativistic perspectives
multi_observer_gnn = MultiObserverGNN(
    feature_dim=feature_dim,
    hidden_dim=32,
    output_dim=8,
    num_observers=4
)

output = multi_observer_gnn(node_features, edge_index)
print(f\"Multi-observer output shape: {output.shape}\")  # [10, 8]
```

### Relativistic Spiking Neural Networks

SNN components that incorporate relativistic time dilation:

```python
import torch
from relativistic_torch.snn import RelativisticLIFNeuron, TerrellPenroseSNN

# Create input spikes (batch_size=32, input_size=10)
input_spikes = torch.bernoulli(torch.ones(32, 10) * 0.3)

# Create a relativistic LIF neuron
neuron = RelativisticLIFNeuron(
    input_size=10,
    threshold=1.0,
    beta=0.9
)

# Initialize neuron state
initial_state = neuron.init_state(batch_size=32)

# Process input spikes
output_spikes, new_state = neuron(input_spikes, initial_state)
print(f\"Output spikes shape: {output_spikes.shape}\")  # [32]

# Create a complete SNN
snn = TerrellPenroseSNN(
    input_size=10,
    hidden_size=20,
    output_size=5,
    simulation_steps=100
)

# Process input
output = snn(input_spikes)
print(f\"SNN output shape: {output.shape}\")  # [32, 5]

# Get spike history for visualization
spike_history = snn.get_spike_history(input_spikes)
print(f\"Hidden spike history shape: {spike_history['hidden_spikes'].shape}\")  # [32, 100, 20]
```

### Relativistic Attention Mechanism

Attention where different heads operate in different reference frames:

```python
import torch
from relativistic_torch.attention import RelativisticSelfAttention

# Create input sequence (batch_size=16, seq_len=24, feature_dim=64)
seq = torch.randn(16, 24, 64)

# Create relativistic self-attention module
attention = RelativisticSelfAttention(
    hidden_dim=64,
    num_heads=8,
    dropout=0.1,
    max_velocity=0.9
)

# Optional: Create positions for spacetime distances
positions = torch.randn(16, 24, 3)  # 3D positions for each token

# Process sequence
output = attention(seq, positions=positions)
print(f\"Output shape: {output.shape}\")  # [16, 24, 64]
```

### Relativistic Transformations

Apply transformations inspired by special relativity to feature vectors:

```python
import torch
from relativistic_torch.transforms import TerrellPenroseTransform, LorentzBoost

# Create feature vectors (batch_size=8, feature_dim=64)
features = torch.randn(8, 64)

# Apply Terrell-Penrose inspired transformation
transform = TerrellPenroseTransform(
    feature_dim=64,
    max_velocity=0.9,
    mode=\"rotation\"
)

transformed = transform(features)
print(f\"Transformed shape: {transformed.shape}\")  # [8, 64]

# For spacetime features (batch_size=8, feature_dim=8 including 4D spacetime)
spacetime_features = torch.randn(8, 8)

# Apply Lorentz boost
boost = LorentzBoost(
    feature_dim=8,
    time_dim=0,  # First dimension is time
    max_velocity=0.8
)

boosted = boost(spacetime_features)
print(f\"Boosted shape: {boosted.shape}\")  # [8, 8]
```

