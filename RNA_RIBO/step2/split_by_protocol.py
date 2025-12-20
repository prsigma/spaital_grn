#!/usr/bin/env python
"""
按 obs['protocol-replicate'] 拆分 h5ad，输出到指定目录（默认与源文件同目录）。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad


def split_h5ad_by_protocol(src: Path, out_dir: Path | None = None, col: str = "protocol-replicate"):
    src = Path(src)
    out_dir = out_dir or src.parent
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(src)
    if col not in adata.obs:
        raise KeyError(f"Column '{col}' not found in obs")

    values = adata.obs[col].unique().tolist()
    print(f"Unique values in {col}: {values}")

    for v in values:
        subset = adata[adata.obs[col] == v].copy()
        out_path = out_dir / f"{src.stem}_{v}_with_domain.h5ad"
        subset.write_h5ad(out_path)
        print(f"Written {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Split h5ad by protocol-replicate column.")
    parser.add_argument("--src", required=True, type=str, help="源 h5ad 路径")
    parser.add_argument("--out_dir", type=str, default=None, help="输出目录（默认同源目录）")
    parser.add_argument("--col", type=str, default="protocol-replicate", help="拆分使用的 obs 列名")
    args = parser.parse_args()
    split_h5ad_by_protocol(Path(args.src), Path(args.out_dir) if args.out_dir else None, args.col)


if __name__ == "__main__":
    main()
