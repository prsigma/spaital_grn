import csv
import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
import torch


def _load_step2_modules():
    root = Path(__file__).resolve().parents[2]
    pkg_root = root / "RNA_RIBO"
    step2_root = pkg_root / "step2"

    if "RNA_RIBO" not in sys.modules:
        pkg = types.ModuleType("RNA_RIBO")
        pkg.__path__ = [str(pkg_root)]
        sys.modules["RNA_RIBO"] = pkg
    if "RNA_RIBO.step2" not in sys.modules:
        subpkg = types.ModuleType("RNA_RIBO.step2")
        subpkg.__path__ = [str(step2_root)]
        sys.modules["RNA_RIBO.step2"] = subpkg

    fusion_spec = importlib.util.spec_from_file_location(
        "RNA_RIBO.step2.fusion", step2_root / "fusion.py"
    )
    fusion_mod = importlib.util.module_from_spec(fusion_spec)
    assert fusion_spec.loader is not None
    fusion_spec.loader.exec_module(fusion_mod)
    sys.modules["RNA_RIBO.step2.fusion"] = fusion_mod

    model_spec = importlib.util.spec_from_file_location(
        "RNA_RIBO.step2.model", step2_root / "model.py"
    )
    model_mod = importlib.util.module_from_spec(model_spec)
    assert model_spec.loader is not None
    model_spec.loader.exec_module(model_mod)
    sys.modules["RNA_RIBO.step2.model"] = model_mod

    ckpt_spec = importlib.util.spec_from_file_location(
        "RNA_RIBO.step2.checkpoint_gene_rank", step2_root / "checkpoint_gene_rank.py"
    )
    ckpt_mod = importlib.util.module_from_spec(ckpt_spec)
    assert ckpt_spec.loader is not None
    ckpt_spec.loader.exec_module(ckpt_mod)
    sys.modules["RNA_RIBO.step2.checkpoint_gene_rank"] = ckpt_mod
    return model_mod, ckpt_mod


MODEL_MOD, CKPT_MOD = _load_step2_modules()
SpatialFusionModel = MODEL_MOD.SpatialFusionModel
GeneRankReference = CKPT_MOD.GeneRankReference
evaluate_checkpoint_gene_rank = CKPT_MOD.evaluate_checkpoint_gene_rank
load_gene_rank_reference = CKPT_MOD.load_gene_rank_reference
topk_overlap_ratio = CKPT_MOD.topk_overlap_ratio


class TestCheckpointGeneRank(unittest.TestCase):
    def test_load_gene_rank_reference_aligns_and_orders(self):
        var_names = ["g0", "g1", "g2", "g3"]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=["gene", "rank"])
            writer.writeheader()
            writer.writerow({"gene": "g2", "rank": 2})
            writer.writerow({"gene": "gX", "rank": 1})
            writer.writerow({"gene": "g1", "rank": 3})
            csv_path = Path(fp.name)

        try:
            ref = load_gene_rank_reference(
                csv_path=csv_path,
                var_names=var_names,
                gene_col="gene",
                rank_col="rank",
                topk=100,
            )
            self.assertListEqual(ref.gene_names.tolist(), ["g2", "g1"])
            self.assertListEqual(ref.var_indices.tolist(), [2, 1])
            self.assertListEqual(ref.reference_topk.tolist(), ["g2", "g1"])
        finally:
            csv_path.unlink(missing_ok=True)

    def test_topk_overlap_ratio(self):
        model_rank = np.array(["a", "b", "c", "d"], dtype=object)
        ref_rank = np.array(["b", "c", "e", "f"], dtype=object)
        overlap, ratio = topk_overlap_ratio(model_rank, ref_rank, topk=3)
        self.assertEqual(overlap, 2)
        self.assertAlmostEqual(ratio, 2 / 3, places=6)

    def test_evaluate_checkpoint_gene_rank_small(self):
        torch.manual_seed(0)
        n_cells = 4
        n_genes = 5
        dim = 4

        w_tx = torch.full((n_cells, n_genes), 1.0 / n_genes, dtype=torch.float32)
        w_ribo = torch.full((n_cells, n_genes), 1.0 / n_genes, dtype=torch.float32)
        model = SpatialFusionModel(
            n_genes=n_genes,
            num_cells=n_cells,
            dim=dim,
            encoder_hidden=8,
            encoder_layers=2,
            gcn_hidden=4,
            gcn_layers=2,
            gene_dim=dim,
            w_tx=w_tx,
            w_ribo=w_ribo,
        )

        ref = GeneRankReference(
            gene_names=np.array(["g0", "g2", "g3"], dtype=object),
            var_indices=np.array([0, 2, 3], dtype=np.int64),
            reference_topk=np.array(["g0", "g2"], dtype=object),
        )
        cell_idx = torch.arange(n_cells, dtype=torch.long)
        ribo_norm = torch.tensor(
            [
                [0.2, 0.5, 0.1],
                [0.1, 0.3, 0.7],
                [0.4, 0.2, 0.2],
                [0.8, 0.6, 0.3],
            ],
            dtype=torch.float32,
        )

        out = evaluate_checkpoint_gene_rank(
            model=model,
            cell_idx=cell_idx,
            ribo_norm=ribo_norm,
            reference=ref,
            topk=2,
            cell_chunk_size=2,
            gene_chunk_size=2,
            corr_block_size=2,
            save_cosine=False,
        )

        self.assertIn("metric_topk_overlap", out)
        self.assertIn("overlap_count", out)
        self.assertIn("gene_scores", out)
        self.assertIn("model_ranked_genes", out)
        self.assertGreaterEqual(out["metric_topk_overlap"], 0.0)
        self.assertLessEqual(out["metric_topk_overlap"], 1.0)
        self.assertEqual(out["topk"], 2)
        self.assertEqual(len(out["gene_scores"]), 3)
        self.assertEqual(len(out["model_ranked_genes"]), 3)


if __name__ == "__main__":
    unittest.main()
