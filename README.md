# torch-template

A modern PyTorch project template using uv for dependency management.

## Features

- Fast dependency management with [uv](https://github.com/astral-sh/uv)
- Clean project structure for PyTorch applications
- Type annotations with proper tooling support
- Optimized for performance and developer experience

## Getting Started

### Prerequisites

- Python 3.8+
- [uv](https://github.com/astral-sh/uv) installed

### Installation

1. Clone the repository
```bash
git clone https://github.com/bjoernbethge/torch-template.git
cd torch-template
```

2. Create and activate a virtual environment with uv
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies
```bash
uv pip install -e .
```

## Development

### Adding Dependencies

Add new dependencies to your project:

```bash
# Add a runtime dependency
uv pip install torch

# Add a development dependency
uv pip install --dev pytest

# Update pyproject.toml after installing dependencies
uv pip freeze > requirements.txt
```

Then manually update your `pyproject.toml` with the new dependencies.

### Benefits of uv

- **Speed**: uv is significantly faster than pip for dependency resolution
- **Reliability**: Consistent dependency resolution across environments
- **Caching**: Efficient caching of downloaded packages
- **Compatibility**: Works with existing Python projects and tools

### Common uv Commands

```bash
# Install packages
uv pip install <package>

# Install from requirements file
uv pip install -r requirements.txt

# Update packages
uv pip install --upgrade <package>

# List installed packages
uv pip list

# Show package info
uv pip show <package>
```

## Project Structure

```
torch-template/
├── .python-version       # Python version specification
├── pyproject.toml        # Project metadata and dependencies
├── README.md             # This file
└── src/                  # Source code
    └── torch_template/   # Main package
        ├── __init__.py   # Package initialization
        └── py.typed      # Marker file for type checkers
```

## License

[MIT](LICENSE)
