# GitHub Copilot Instructions for torch-relativistic

## Project Overview

**torch-relativistic** is a PyTorch extension that implements neural network components inspired by relativistic physics, particularly the **Terrell-Penrose effect**. This library provides novel information processing paradigms by incorporating concepts from special relativity into machine learning architectures.

### Key Concepts
- The Terrell-Penrose effect describes how rapidly moving objects appear rotated rather than contracted
- Neural network modules process information as if affected by relativistic phenomena
- Components include relativistic GNNs, SNNs, attention mechanisms, and transformations

## Technology Stack

- **Language**: Python ≥ 3.11
- **Deep Learning Framework**: PyTorch ≥ 2.0.0
- **Graph Neural Networks**: PyTorch Geometric ≥ 2.6.1
- **Package Manager**: uv (Astral)
- **Additional Dependencies**: NumPy, Polars, Plotly Express, Astroquery

## Project Structure

```
torch-relativistic/
├── src/torch_relativistic/          # Main source code
│   ├── __init__.py                  # Package initialization, exports main classes
│   ├── gnn.py                       # Relativistic Graph Neural Networks
│   ├── snn.py                       # Relativistic Spiking Neural Networks
│   ├── attention.py                 # Relativistic attention mechanisms
│   ├── transforms.py                # Relativistic transformations (Lorentz boosts, etc.)
│   ├── utils.py                     # Utility functions (Lorentz factor, etc.)
│   └── data/                        # Data processing modules
├── tests/                           # Test suite
│   ├── test_basic.py                # Basic import and instantiation tests
│   └── __init__.py
├── examples/                        # Usage examples
├── pyproject.toml                   # Project configuration and dependencies
└── README.md                        # Documentation
```

## Development Setup

### Installation

```bash
# Clone the repository
git clone https://github.com/bjoernbethge/torch-relativistic.git
cd torch-relativistic

# Install uv package manager (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies and dev tools
uv sync --dev

# Or install in editable mode with pip
pip install -e ".[dev]"
```

### Dependencies

The project uses `uv` as the package manager with the following key dependencies:
- `torch==2.7.0` (from PyTorch CUDA repository)
- `torch-geometric>=2.6.1`
- `numpy>=1.20.0`
- `polars>=1.30.0`
- `plotly-express>=0.4.1`
- `astroquery>=0.4.10`

Development dependencies:
- `black>=25.1.0` (code formatting)
- `mypy>=1.15.0` (type checking)
- `pytest>=8.3.5` (testing)
- `ruff>=0.11.9` (linting)

## Building and Testing

### Running Tests

```bash
# Run all tests with verbose output
uv run pytest tests/ -v

# Run tests with pip installation
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=torch_relativistic
```

### Code Quality Checks

```bash
# Format code with black (apply formatting)
uv run black src/ tests/

# Check formatting without changes
uv run black --check src/ tests/

# Run linting with ruff
uv run ruff check src/ tests/

# Run type checking with mypy
uv run mypy src/
```

### Full CI Pipeline (as in GitHub Actions)

```bash
# Install dependencies
uv sync --dev

# Run tests
uv run pytest tests/ -v

# Type checking
uv run mypy src/

# Linting
uv run ruff check src/ tests/
uv run black --check src/ tests/
```

## Code Style and Conventions

### General Guidelines

1. **Type Hints**: Always use type hints for function parameters and return values
2. **Formatting**: Use `black` for code formatting (enforced in CI)
3. **Linting**: Follow `ruff` linting rules
4. **Docstrings**: Use clear docstrings for public APIs, classes, and complex functions
5. **Language**: Code comments and docstrings are in English, though some test comments may be in German

### Naming Conventions

- **Classes**: PascalCase (e.g., `RelativisticGraphConv`, `TerrellPenroseSNN`)
- **Functions/Methods**: snake_case (e.g., `lorentz_factor`, `init_state`)
- **Constants**: UPPER_SNAKE_CASE
- **Private members**: Prefix with underscore (e.g., `_compute_boost`)

### PyTorch Conventions

- Use `torch.nn.Module` as base class for all neural network components
- Implement `forward()` method for all modules
- Use `torch.Tensor` for type hints
- Prefer functional API (`torch.nn.functional`) for stateless operations

### Physics-Inspired Naming

- Use physics terminology accurately (e.g., `lorentz_factor`, `velocity`, `boost`)
- Parameters like `max_relative_velocity` should be in range [0, 1) representing fraction of speed of light
- Time dilation, length contraction, and relativistic effects should follow special relativity equations

## Key Components

### 1. Relativistic Graph Neural Networks (gnn.py)

- `RelativisticGraphConv`: Basic GNN layer with relativistic message passing
- `MultiObserverGNN`: Processes graphs from multiple relativistic reference frames
- Uses PyTorch Geometric's `MessagePassing` as base class

### 2. Relativistic Spiking Neural Networks (snn.py)

- `RelativisticLIFNeuron`: Leaky Integrate-and-Fire neuron with time dilation
- `TerrellPenroseSNN`: Complete SNN with multiple layers
- State-based computation with explicit state management

### 3. Relativistic Attention (attention.py)

- `RelativisticSelfAttention`: Multi-head attention where each head operates in different reference frames
- Supports optional position-based spacetime distance computation

### 4. Relativistic Transformations (transforms.py)

- `TerrellPenroseTransform`: Feature transformation inspired by Terrell-Penrose effect
- `LorentzBoost`: Applies Lorentz transformation to feature vectors
- Both inherit from `torch.nn.Module`

### 5. Utilities (utils.py)

- `lorentz_factor`: Computes γ factor from velocity
- `LeviCivitaTensor`: Four-dimensional Levi-Civita tensor for relativistic calculations
- Helper functions for spacetime computations

## Testing Guidelines

### Test Structure

- Tests are organized in `tests/` directory
- Use `pytest` fixtures for reusable test components
- Parametrize tests for multiple scenarios when applicable

### Test Fixtures

```python
@pytest.fixture
def torch_relativistic_classes():
    """Provides access to all major classes for testing"""
    # Returns dictionary of classes
```

### What to Test

1. **Module imports**: Ensure all modules can be imported
2. **Class instantiation**: Verify classes can be created with default parameters
3. **Forward passes**: Test that modules process inputs correctly
4. **Shape preservation**: Verify output shapes match expected dimensions
5. **Edge cases**: Test boundary conditions (e.g., velocity = 0, velocity → 1)

## Common Patterns

### Creating a Module

```python
import torch
import torch.nn as nn

class NewRelativisticModule(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, max_velocity: float = 0.9):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_velocity = max_velocity
        # Initialize layers
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Implement forward pass
        return output
```

### Using Relativistic Effects

```python
from torch_relativistic.utils import lorentz_factor

# Compute time dilation
velocity = 0.8  # 80% speed of light
gamma = lorentz_factor(velocity)

# Apply relativistic transformation
time_dilated = time_interval * gamma
```

## CI/CD

### GitHub Actions Workflows

1. **tests.yml**: Runs on push to `main`/`develop` and PRs
   - Tests on Python 3.11 and 3.12
   - Runs pytest, mypy, ruff, and black checks

2. **python-publish.yml**: Publishes to PyPI on new releases

3. **jekyll-gh-pages.yml**: Builds GitHub Pages documentation

## Common Commands

```bash
# Development setup
uv sync --dev

# Run tests
uv run pytest tests/ -v

# Format code
uv run black src/ tests/

# Lint code
uv run ruff check src/ tests/

# Type check
uv run mypy src/

# Install package locally
pip install -e .

# Build distribution
python -m build
```

## Important Notes

1. **PyTorch Geometric**: Some type stubs are missing; use `# type: ignore[import-untyped]` where necessary
2. **CUDA**: PyTorch is configured to use CUDA 12.8 (see `pyproject.toml`)
3. **Python Version**: Minimum Python 3.11 required (uses modern type hints)
4. **Velocity Range**: All velocity parameters should be in [0, 1) representing fraction of speed of light (c=1)

## Physics Background

When contributing, keep in mind:
- Special relativity principles should be mathematically accurate
- Lorentz transformations preserve spacetime intervals
- Time dilation: γ = 1/√(1 - v²/c²)
- The Terrell-Penrose effect is about apparent rotation, not just contraction
- Velocities should never equal or exceed the speed of light (v < c)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes following the code style guidelines
4. Add tests for new functionality
5. Ensure all tests pass: `uv run pytest tests/ -v`
6. Run code quality checks (black, ruff, mypy)
7. Submit a pull request

## Contact

- **Author**: Björn Bethge
- **Email**: bjoern.bethge@gmail.com
- **Repository**: https://github.com/bjoernbethge/torch-relativistic
- **PyPI**: https://pypi.org/project/torch-relativistic/
