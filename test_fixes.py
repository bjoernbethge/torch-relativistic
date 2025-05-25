#!/usr/bin/env python3
"""
Test script to verify that all fixes work
"""

def test_imports():
    """Test that imports work correctly"""
    print("Testing imports...")
    try:
        import torch_relativistic
        print(f"✅ torch_relativistic imported, version: {torch_relativistic.__version__}")
        
        from torch_relativistic.gnn import RelativisticGraphConv
        print("✅ RelativisticGraphConv imported")
        
        from torch_relativistic.snn import RelativisticLIFNeuron
        print("✅ RelativisticLIFNeuron imported")
        
        from torch_relativistic.attention import RelativisticSelfAttention
        print("✅ RelativisticSelfAttention imported")
        
        from torch_relativistic.transforms import TerrellPenroseTransform
        print("✅ TerrellPenroseTransform imported")
        
        assert True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        assert False

def test_basic_functionality():
    """Test basic functionality"""
    print("\nTesting basic functionality...")
    try:
        import torch
        from torch_relativistic.gnn import RelativisticGraphConv
        
        # Test RelativisticGraphConv
        conv = RelativisticGraphConv(16, 32)
        x = torch.randn(10, 16)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        
        out = conv(x, edge_index)
        print(f"✅ RelativisticGraphConv output shape: {out.shape}")
        
        assert True
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        assert False

if __name__ == "__main__":
    print("🧪 Running fix verification tests...\n")
    
    success = True
    success &= test_imports()
    success &= test_basic_functionality()
    
    if success:
        print("\n🎉 All fixes verified successfully!")
        print("\nNext steps:")
        print("1. Run 'uv run pytest tests/' to run full test suite")
        print("2. Run 'uv run black src/' to format code")
        print("3. Run 'uv run ruff check src/' to check linting")
        print("4. Update your GitHub repository URLs in pyproject.toml")
    else:
        print("\n❌ Some fixes failed - check the errors above")
