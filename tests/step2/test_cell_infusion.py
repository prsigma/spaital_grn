import unittest
import importlib.util
import sys
import types
from pathlib import Path
import torch


def _load_spatial_fusion_model():
    root = Path(__file__).resolve().parents[2]
    pkg_root = root / "RNA_RIBO"
    step2_root = pkg_root / "step2"
    model_path = step2_root / "model.py"
    fusion_path = step2_root / "fusion.py"

    if "RNA_RIBO" not in sys.modules:
        pkg = types.ModuleType("RNA_RIBO")
        pkg.__path__ = [str(pkg_root)]
        sys.modules["RNA_RIBO"] = pkg
    if "RNA_RIBO.step2" not in sys.modules:
        subpkg = types.ModuleType("RNA_RIBO.step2")
        subpkg.__path__ = [str(step2_root)]
        sys.modules["RNA_RIBO.step2"] = subpkg

    fusion_spec = importlib.util.spec_from_file_location(
        "RNA_RIBO.step2.fusion", fusion_path
    )
    fusion_mod = importlib.util.module_from_spec(fusion_spec)
    assert fusion_spec.loader is not None
    fusion_spec.loader.exec_module(fusion_mod)
    sys.modules["RNA_RIBO.step2.fusion"] = fusion_mod

    model_spec = importlib.util.spec_from_file_location(
        "RNA_RIBO.step2.model", model_path
    )
    model_mod = importlib.util.module_from_spec(model_spec)
    assert model_spec.loader is not None
    model_spec.loader.exec_module(model_mod)
    sys.modules["RNA_RIBO.step2.model"] = model_mod
    return model_mod.SpatialFusionModel


SpatialFusionModel = _load_spatial_fusion_model()


def _identity_sparse(n: int) -> torch.Tensor:
    idx = torch.arange(n, dtype=torch.long)
    indices = torch.stack([idx, idx], dim=0)
    values = torch.ones(n, dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, values, (n, n)).coalesce()


class TestCellInfusion(unittest.TestCase):
    def test_cell_infusion_forward_shapes(self):
        n_cells = 6
        n_genes = 10
        dim = 8

        model = SpatialFusionModel(
            n_genes=n_genes,
            num_cells=n_cells,
            dim=dim,
            encoder_hidden=16,
            encoder_layers=2,
            gcn_hidden=8,
            gcn_layers=2,
            gene_dim=None,
        )
        rna = torch.randn(n_cells, n_genes)
        ribo = torch.randn(n_cells, n_genes)
        adj = _identity_sparse(n_cells)
        cell_idx = torch.arange(n_cells, dtype=torch.long)

        out = model(rna, ribo, adj, cell_idx=cell_idx)

        self.assertEqual(out["z_rna"].shape, (n_cells, dim))
        self.assertEqual(out["z_ribo"].shape, (n_cells, dim))
        self.assertEqual(out["fused"].shape, (n_cells, dim))
        self.assertEqual(out["h_final"].shape, (n_cells, dim))
        self.assertEqual(out["weights"].shape, (n_cells, 2))

    def test_cell_idx_length_mismatch_raises(self):
        n_cells = 5
        n_genes = 9
        model = SpatialFusionModel(
            n_genes=n_genes,
            num_cells=n_cells,
            dim=8,
            encoder_hidden=16,
            encoder_layers=2,
            gcn_hidden=8,
            gcn_layers=2,
            gene_dim=None,
        )
        rna = torch.randn(n_cells, n_genes)
        ribo = torch.randn(n_cells, n_genes)
        adj = _identity_sparse(n_cells)
        bad_cell_idx = torch.arange(n_cells - 1, dtype=torch.long)

        with self.assertRaises(ValueError):
            model(rna, ribo, adj, cell_idx=bad_cell_idx)


if __name__ == "__main__":
    unittest.main()
