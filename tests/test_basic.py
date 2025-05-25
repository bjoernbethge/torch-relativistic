"""Basic tests for torch-relativistic"""
import torch
import pytest
import importlib.util


@pytest.fixture
def torch_relativistic_modules():
    """Fixture für die wichtigsten Module"""
    import torch_relativistic
    return {
        'main': torch_relativistic,
        'gnn': importlib.util.find_spec('torch_relativistic.gnn'),
        'snn': importlib.util.find_spec('torch_relativistic.snn'),
        'attention': importlib.util.find_spec('torch_relativistic.attention'),
        'transforms': importlib.util.find_spec('torch_relativistic.transforms'),
        'utils': importlib.util.find_spec('torch_relativistic.utils')
    }


@pytest.fixture
def torch_relativistic_classes():
    """Fixture für die wichtigsten Klassen"""
    from torch_relativistic.gnn import RelativisticGraphConv, MultiObserverGNN
    from torch_relativistic.snn import RelativisticLIFNeuron, TerrellPenroseSNN
    from torch_relativistic.attention import RelativisticSelfAttention
    from torch_relativistic.transforms import TerrellPenroseTransform, LorentzBoost
    from torch_relativistic.utils import lorentz_factor, LeviCivitaTensor

    return {
        'RelativisticGraphConv': RelativisticGraphConv,
        'MultiObserverGNN': MultiObserverGNN,
        'RelativisticLIFNeuron': RelativisticLIFNeuron,
        'TerrellPenroseSNN': TerrellPenroseSNN,
        'RelativisticSelfAttention': RelativisticSelfAttention,
        'TerrellPenroseTransform': TerrellPenroseTransform,
        'LorentzBoost': LorentzBoost,
        'lorentz_factor': lorentz_factor,
        'LeviCivitaTensor': LeviCivitaTensor
    }


class TestImports:
    """Tests für die Import-Funktionalität"""

    def test_module_imports(self, torch_relativistic_modules):
        """Prüft, ob alle Module importiert werden können"""
        for name, module in torch_relativistic_modules.items():
            if name != 'main':
                assert module is not None, f"Modul {name} konnte nicht importiert werden"

    @pytest.mark.parametrize("class_name", [
        "RelativisticGraphConv", "MultiObserverGNN", "RelativisticLIFNeuron",
        "TerrellPenroseSNN", "RelativisticSelfAttention", "TerrellPenroseTransform",
        "LorentzBoost"
    ])
    def test_class_instantiation(self, torch_relativistic_classes, class_name):
        """Prüft, ob Klassen instanziiert werden können"""
        cls = torch_relativistic_classes[class_name]

        if class_name == "RelativisticGraphConv":
            instance = cls(4, 8)
        elif class_name == "MultiObserver
