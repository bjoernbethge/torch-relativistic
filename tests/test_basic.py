"""Basic tests for torch-relativistic"""

import torch
import pytest


def test_imports():
    """Test that main modules can be imported"""
    try:
        from torch_relativistic.gnn import RelativisticGraphConv
        from torch_relativistic.snn import RelativisticLIFNeuron
        from torch_relativistic.attention import RelativisticSelfAttention
        from torch_relativistic.transforms import TerrellPenroseTransform
        print("✅ All imports successful")
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")


def test_basic_functionality():
    """Test basic functionality of main components"""
    from torch_relativistic.gnn import RelativisticGraphConv
    
    # Test RelativisticGraphConv
    conv = RelativisticGraphConv(16, 32)
    x = torch.randn(10, 16)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    
    try:
        out = conv(x, edge_index)
        assert out.shape == (10, 32), f"Expected shape (10, 32), got {out.shape}"
        print("✅ RelativisticGraphConv works")
    except Exception as e:
        pytest.fail(f"RelativisticGraphConv failed: {e}")


def test_version():
    """Test that version is accessible"""
    import torch_relativistic
    assert hasattr(torch_relativistic, '__version__')
    assert torch_relativistic.__version__ == "0.1.2"
    print(f"✅ Version is {torch_relativistic.__version__}")


if __name__ == "__main__":
    test_imports()
    test_basic_functionality()
    test_version()
    print("🎉 All basic tests passed!")
