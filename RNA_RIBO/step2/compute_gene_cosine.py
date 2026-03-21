#!/usr/bin/env python
"""
Compute gene-gene cosine similarities from gene_embeddings.npz.

Outputs two TSV files:
1) E_tx vs E_tx  (geneA, geneB, cosine)
2) E_ribo vs E_tx (geneA uses ribo, geneB uses RNA)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def _write_tsv(
    path: Path,
    gene_a: np.ndarray,
    gene_b: np.ndarray,
    sim: np.ndarray,
    header: bool = True,
    precision: int = 6,
    upper_tri: bool = False,
) -> None:
    fmt = f"{{:.{precision}f}}"
    with path.open("w", encoding="utf-8") as f:
        if header:
            f.write("geneA\tgeneB\tcosine\n")
        for i, ga in enumerate(gene_a):
            row = sim[i]
            if upper_tri:
                j_start = i
            else:
                j_start = 0
            for j in range(j_start, len(gene_b)):
                f.write(f"{ga}\t{gene_b[j]}\t{fmt.format(row[j])}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute gene-gene cosine similarity TSVs.")
    parser.add_argument(
        "--npz",
        type=str,
        required=True,
        help="Path to gene_embeddings.npz (contains gene_names, E_tx, E_ribo).",
    )
    parser.add_argument(
        "--out_tx",
        type=str,
        default=None,
        help="Output TSV for E_tx vs E_tx (default: alongside npz).",
    )
    parser.add_argument(
        "--out_ribo_tx",
        type=str,
        default=None,
        help="Output TSV for E_ribo vs E_tx (default: alongside npz).",
    )
    parser.add_argument("--no_header", action="store_true", help="Do not write TSV header.")
    parser.add_argument("--precision", type=int, default=6, help="Decimal places for cosine values.")
    args = parser.parse_args()

    npz_path = Path(args.npz)
    data = np.load(npz_path, allow_pickle=True)

    gene_names = data["gene_names"].astype(str)
    e_tx = data["E_tx"]
    e_ribo = data["E_ribo"]

    out_tx = Path(args.out_tx) if args.out_tx else npz_path.with_name("gene_cosine_tx.tsv")
    out_ribo_tx = (
        Path(args.out_ribo_tx) if args.out_ribo_tx else npz_path.with_name("gene_cosine_ribo_tx.tsv")
    )

    sim_tx = cosine_similarity(e_tx)
    sim_ribo_tx = cosine_similarity(e_ribo, e_tx)

    _write_tsv(
        out_tx,
        gene_names,
        gene_names,
        sim_tx,
        header=not args.no_header,
        precision=args.precision,
        upper_tri=True,
    )
    _write_tsv(
        out_ribo_tx, gene_names, gene_names, sim_ribo_tx, header=not args.no_header, precision=args.precision
    )

    print(f"Saved {out_tx}")
    print(f"Saved {out_ribo_tx}")


if __name__ == "__main__":
    main()
