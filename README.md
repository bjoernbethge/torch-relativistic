# 🌌 torch-relativistic

<div align="center">

<!-- Badges -->
[![PyPI version](https://badge.fury.io/py/torch-relativistic.svg)](https://badge.fury.io/py/torch-relativistic)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/bjoernbethge/torch-relativistic/actions/workflows/tests.yml/badge.svg)](https://github.com/bjoernbethge/torch-relativistic/actions/workflows/tests.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

<!-- Logo/Header -->
<h3>🚀 A PyTorch extension implementing neural networks inspired by relativistic physics</h3>

*Harness the power of the Terrell-Penrose effect for novel information processing paradigms* ⚡

</div>

---

## 🌟 Overview

**torch-relativistic** provides neural network modules that incorporate concepts from **special relativity** into machine learning. The key insight is that the **Terrell-Penrose effect**, where rapidly moving objects appear rotated rather than contracted, can inspire revolutionary information processing paradigms in neural networks.

### 🎯 Key Features

- 🧠 **Relativistic Graph Neural Networks (GNNs)** - Process graphs with relativistic information propagation
- ⚡ **Relativistic Spiking Neural Networks (SNNs)** - Time dilation effects in spiking neurons  
- 🎭 **Relativistic Attention Mechanisms** - Multi-reference frame attention heads
- 🌀 **Relativistic Transformations** - Lorentz boosts and Terrell-Penrose transforms
- 🔬 **Physics-Inspired Architecture** - Grounded in real relativistic physics

---

## 📦 Installation

### Quick Install
```bash
pip install torch-relativistic
```

### Development Install
```bash
git clone https://github.com/yourusername/torch-relativistic.git
cd torch-relativistic
pip install -e .
```

### Requirements
- 🐍 Python ≥ 3.11
- 🔥 PyTorch ≥ 2.0.0  
- 📊 PyTorch Geometric ≥ 2.6.1
- 🔢 NumPy ≥ 1.20.0

---

## 🚀 Quick Start

```python
import torch
from torch_relativistic import RelativisticGraphConv

# Create a relativistic GNN layer
conv = RelativisticGraphConv(16, 32, max_relative_velocity=0.8)
x = torch.randn(10, 16)
edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)

# Process with relativistic effects
output = conv(x, edge_index)  # Shape: [10, 32]
```

---

## 📚 Components

### 🌐 Relativistic Graph Neural Networks

GNN modules that process information as if affected by relativistic phenomena:

```python
import torch
from torch_relativistic.gnn import RelativisticGraphConv, MultiObserverGNN

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
print(f"Output shape: {output_features.shape}")  # [10, 32]

# Multi-observer GNN processes the graph from multiple relativistic perspectives
multi_observer_gnn = MultiObserverGNN(
    feature_dim=feature_dim,
    hidden_dim=32,
    output_dim=8,
    num_observers=4
)

output = multi_observer_gnn(node_features, edge_index)
print(f"Multi-observer output shape: {output.shape}")  # [10, 8]
```

### ⚡ Relativistic Spiking Neural Networks

SNN components that incorporate relativistic time dilation:

```python
import torch
from torch_relativistic.snn import RelativisticLIFNeuron, TerrellPenroseSNN

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
print(f"Output spikes shape: {output_spikes.shape}")  # [32]

# Create a complete SNN
snn = TerrellPenroseSNN(
    input_size=10,
    hidden_size=20,
    output_size=5,
    simulation_steps=100
)

# Process input
output = snn(input_spikes)
print(f"SNN output shape: {output.shape}")  # [32, 5]

# Get spike history for visualization
spike_history = snn.get_spike_history(input_spikes)
print(f"Hidden spike history shape: {spike_history['hidden_spikes'].shape}")  # [32, 100, 20]
```

### 🎭 Relativistic Attention Mechanism

Attention where different heads operate in different reference frames:

```python
import torch
from torch_relativistic.attention import RelativisticSelfAttention

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
print(f"Output shape: {output.shape}")  # [16, 24, 64]
```

### 🌀 Relativistic Transformations

Apply transformations inspired by special relativity to feature vectors:

```python
import torch
from torch_relativistic.transforms import TerrellPenroseTransform, LorentzBoost

# Create feature vectors (batch_size=8, feature_dim=64)
features = torch.randn(8, 64)

# Apply Terrell-Penrose inspired transformation
transform = TerrellPenroseTransform(
    feature_dim=64,
    max_velocity=0.9,
    mode="rotation"
)

transformed = transform(features)
print(f"Transformed shape: {transformed.shape}")  # [8, 64]

# For spacetime features (batch_size=8, feature_dim=8 including 4D spacetime)
spacetime_features = torch.randn(8, 8)

# Apply Lorentz boost
boost = LorentzBoost(
    feature_dim=8,
    time_dim=0,  # First dimension is time
    max_velocity=0.8
)

boosted = boost(spacetime_features)
print(f"Boosted shape: {boosted.shape}")  # [8, 8]
```

---

## 🧪 Development

### Running Tests
```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=torch_relativistic
```

### Code Quality
```bash
# Format code
black src/ tests/

# Check linting
ruff check src/ tests/

# Type checking
mypy src/
```

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### 🛠️ Development Setup

1. **Fork** the repository
2. **Clone** your fork: `git clone https://github.com/yourusername/torch-relativistic.git`
3. **Install** in development mode: `pip install -e ".[dev]"`
4. **Create** a feature branch: `git checkout -b feature/amazing-feature`
5. **Make** your changes and add tests
6. **Run** tests: `pytest tests/`
7. **Submit** a pull request

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- 🌌 Inspired by Einstein's **Special Theory of Relativity**
- 🔬 Built on the **Terrell-Penrose effect** from relativistic physics
- 🔥 Powered by **PyTorch** and **PyTorch Geometric**
- ⚡ Thanks to the open-source ML community

---

## 📞 Contact & Links

<div align="center">

[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/yourusername/torch-relativistic)
[![PyPI](https://img.shields.io/badge/PyPI-3775A9?style=for-the-badge&logo=pypi&logoColor=white)](https://pypi.org/project/torch-relativistic/)
[![Documentation](https://img.shields.io/badge/Docs-4285F4?style=for-the-badge&logo=read-the-docs&logoColor=white)](https://torch-relativistic.readthedocs.io/)
[![Email](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:bjoern.bethge@gmail.com)

**Made with ❤️ and ⚛️ physics**

</div>

---

<div align="center">
<sub>Built with 🔥 PyTorch • Inspired by 🌌 Einstein • Powered by ⚛️ Physics</sub>
</div>
