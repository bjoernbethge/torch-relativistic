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

Authors: Björn Bethge
"""

__version__ = "0.1.2"

from torch_relativistic.gnn import RelativisticGraphConv, MultiObserverGNN
from torch_relativistic.snn import RelativisticLIFNeuron, TerrellPenroseSNN
from torch_relativistic.attention import RelativisticSelfAttention
from torch_relativistic.transforms import TerrellPenroseTransform

__all__ = [
    'RelativisticGraphConv',
    'MultiObserverGNN',
    'RelativisticLIFNeuron',
    'TerrellPenroseSNN',
    'RelativisticSelfAttention',
    'TerrellPenroseTransform',
]

# Typing support for mypy
if False:
    from .gnn import RelativisticGraphConv, MultiObserverGNN
    from .snn import RelativisticLIFNeuron, TerrellPenroseSNN
    from .attention import RelativisticSelfAttention
    from .transforms import TerrellPenroseTransform

# Hinweis für mypy: torch_geometric hat keine Typ-Stubs, daher ggf. type: ignore[import-untyped] in gnn.py verwenden.
