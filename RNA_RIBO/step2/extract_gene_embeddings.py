#!/usr/bin/env python
"""
Extract gene embeddings (E_tx/E_ribo) from a trained model and map to gene names.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import torch


def _load_args(run_dir: Path) -> dict:
    args_path = run_dir / "args.json"
    if not args_path.exists():
        raise FileNotFoundError(f"args.json not found in {run_dir}")
    return json.loads(args_path.read_text())


def _select_state_dict(obj: object) -> dict:
    if isinstance(obj, dict) and "state_dict" in obj and "E_tx" not in obj and "E_ribo" not in obj:
        return obj["state_dict"]
    if isinstance(obj, dict):
        return obj
    raise TypeError(f"Unexpected model file type: {type(obj)}")


def _require_key(state: dict, key: str) -> torch.Tensor:
    if key in state:
        return state[key]
    for k in state.keys():
        if k.endswith(f".{key}"):
            return state[k]
    raise KeyError(f"{key} not found in model state_dict")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract E_tx/E_ribo with gene names.")
    parser.add_argument("--run_dir", required=True, type=str, help="Run directory containing args.json and model.")
    parser.add_argument("--model_path", type=str, default=None, help="Path to model .pt (default: run_dir/model_best_ari.pt)")
    parser.add_argument("--h5ad", type=str, default=None, help="Path to h5ad (default: read from args.json)")
    parser.add_argument("--out", type=str, default=None, help="Output .npz path (default: run_dir/gene_embeddings.npz)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    model_path = Path(args.model_path) if args.model_path else run_dir / "model_best_ari.pt"

    if args.h5ad:
        h5ad_path = Path(args.h5ad)
    else:
        h5ad_path = Path(_load_args(run_dir)["h5ad"])

    out_path = Path(args.out) if args.out else run_dir / "gene_embeddings.npz"

    adata = ad.read_h5ad(h5ad_path, backed="r")
    gene_names = np.asarray(adata.var_names, dtype=str)

    state = _select_state_dict(torch.load(model_path, map_location="cpu"))
    e_tx = _require_key(state, "E_tx").detach().cpu().numpy()
    e_ribo = _require_key(state, "E_ribo").detach().cpu().numpy()

    if e_tx.shape[0] != gene_names.shape[0] or e_ribo.shape[0] != gene_names.shape[0]:
        raise ValueError(
            "Gene count mismatch: "
            f"var_names={gene_names.shape[0]} E_tx={e_tx.shape[0]} E_ribo={e_ribo.shape[0]}"
        )

    np.savez(out_path, gene_names=gene_names, E_tx=e_tx, E_ribo=e_ribo)
    print(f"Saved {out_path}")
    print(f"gene_names: {gene_names.shape} E_tx: {e_tx.shape} E_ribo: {e_ribo.shape}")


if __name__ == "__main__":
    main()
