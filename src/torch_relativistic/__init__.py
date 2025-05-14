"""
RelativisticTorch - PyTorch extension inspired by the Terrell-Penrose effect

This package provides neural network modules that incorporate concepts from
relativistic physics, particularly the Terrell-Penrose effect, into machine learning.
It includes implementations for both Graph Neural Networks (GNNs) and Spiking Neural
Networks (SNNs), enabling novel information processing paradigms.

Main components:
- `gnn`: Relativistic Graph Neural Network modules
- `snn`: Relativistic Spiking Neural Network modules
- `attention`: Relativistic attention mechanisms
- `transforms`: Relativistic space-time transforms for neural network features
- `utils`: Utility functions for relativistic computations

Authors: Claude AI
"""

from relativistic_torch.gnn import RelativisticGraphConv, MultiObserverGNN
from relativistic_torch.snn import RelativisticLIFNeuron, TerrellPenroseSNN
from relativistic_torch.attention import RelativisticSelfAttention
from relativistic_torch.transforms import TerrellPenroseTransform

__all__ = [
    'RelativisticGraphConv',
    'MultiObserverGNN',
    'RelativisticLIFNeuron',
    'TerrellPenroseSNN',
    'RelativisticSelfAttention',
    'TerrellPenroseTransform',
]
