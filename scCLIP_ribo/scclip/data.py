from torch.utils.data import Dataset, random_split, Subset
try:
    import muon as mu
    from mudata import MuData
except ImportError:  # allow lightweight usage when muon is absent
    mu = None
    MuData = None
try:
    import scanpy as sc
    _scanpy_import_error = None
except Exception as e:
    sc = None
    _scanpy_import_error = e
import numpy as np
import scipy
from anndata import AnnData
import anndata as ad
import os
import pandas as pd
from typing import Union, List
from sklearn.preprocessing import maxabs_scale, LabelEncoder
from pathlib import Path


def _require_scanpy():
    if sc is None:
        raise ImportError(
            "scanpy is required for this operation; please install scanpy in the scCLIP environment"
        ) from _scanpy_import_error

class BaseDataset(Dataset):
    def __init__(
        self,
        data_dir: str = None,
        modality: str = "multiome",
        backed: bool = False,
        n_top_genes: int = None,
        n_top_peaks: int = None,
        split: Union[float, str, list] = 0.9,
        linked: Union[bool, int] = 100_000,
        mask: float = None,
        binary: bool = True,
        verbose: bool = False,
        use_seq: bool = False,
        cell_type: str = "cell_type",
    ):
        super().__init__()
        self.data_dir = data_dir if type(data_dir) == str else str(data_dir)
        self.modality = modality
        self.backed = backed
        self.n_top_genes = n_top_genes
        self.n_top_peaks = n_top_peaks
        self.verbose = verbose
        self.linked = linked
        self.mask = mask
        self.binary = binary
        self.use_seq = use_seq
        self.cell_type = cell_type

        self.read()
        if not self.backed:
            if self.modality == "multiome":
                self._preprocess_multiome()
            elif self.modality == "rna":
                self._preprocess_rna()
            elif self.modality == "atac":
                self._preprocess_atac()
            else:
                raise ValueError(f"Modality {self.modality} not supported")
        if self.cell_type in self.mdata.obs.columns:
            self.le = LabelEncoder()
            self.cell_types = self.le.fit_transform(self.mdata.obs[self.cell_type])
        else:
            self.cell_types = None
        self.train_dataset, self.val_dataset = self._split(split)

        print(
            "RNA shape",
            self.mdata.mod["rna"].shape,
            "atac shape",
            self.mdata.mod["atac"].shape,
            flush=True,
        )

    def __len__(self):
        return self.mdata.mod["atac"].shape[0]

    def collate(self, batch):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def read(self):
        if self.modality == "multiome":
            if mu is None or MuData is None:
                raise ImportError("muon is required for multiome data loading")
            assert os.path.isfile(
                os.path.join(self.data_dir)
            ), f"MultiOme file {self.data_dir} not found"
            self.mdata = mu.read(self.data_dir, backed=self.backed)
        else:
            if MuData is None:
                raise ImportError("MuData is required for single-modality h5ad loading")
            assert os.path.isfile(self.data_dir), f'File {self.data_dir} not found'
            self.mdata = MuData({self.modality:mu.read(self.data_dir, backed=self.backed)})
        
        if 'atac' in self.mdata.mod.keys():
            self.mdata.mod['atac'].var_names_make_unique()
        if 'rna' in self.mdata.mod.keys():
            self.mdata.mod['rna'].var_names_make_unique()

        # print('Backed mode: {}'.format(self.mdata.isbacked))

    def _split(self, split):
        if isinstance(split, float):
            return self.random_split(split)
        elif isinstance(split, str):  # split only one cell type
            if split.endswith("_r"):
                split = split.strip("_r")
                reverse = True
            else:
                reverse = False
            assert (
                split in self.mdata.obs["cell_type"].unique()
            ), f"Cell type {split} not found"
            if reverse:
                train_idx = np.where(self.mdata.obs["cell_type"] != split)[0]
                val_idx = np.where(self.mdata.obs["cell_type"] == split)[0]
            else:
                train_idx = np.where(self.mdata.obs["cell_type"] == split)[0]
                val_idx = np.where(self.mdata.obs["cell_type"] != split)[0]

            return Subset(self, train_idx), Subset(self, val_idx)

        elif isinstance(split, list):  # split multiple cell types # TO DO
            train_idx = self.mdata.obs[self.mdata.obs["cell_type"].isin(split)].index
            val_idx = self.mdata.obs[~self.mdata.obs["cell_type"].isin(split)].index
            return Subset(self, train_idx), Subset(self, val_idx)

    def random_split(self, train_ratio=0.9):
        train_size = int(train_ratio * len(self))
        val_size = len(self) - train_size
        return random_split(self, [train_size, val_size])

    ## RNA specific functions

    def get_rna(self, index):
        x = self.mdata["rna"].X[index].toarray().squeeze()
        if self.mask is not None:
            index = np.where(x > 0)[0]
            index = np.random.choice(
                index, size=int(len(index) * self.mask), replace=False
            )
            x[index] = 0
        return x

    def _preprocess_rna(self):
        _require_scanpy()
        rna = self.mdata.mod["rna"]
        # rna.var_names_make_unique()

        # rna.var['mt'] = rna.var_names.str.startswith('MT-')
        # sc.pp.calculate_qc_metrics(rna, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

        # # control quality
        # mu.pp.filter_var(rna, 'n_cells_by_counts', lambda x: x >= 3)
        # mu.pp.filter_obs(rna, 'n_genes_by_counts', lambda x: (x >= 200) & (x < 5000))
        # mu.pp.filter_obs(rna, 'total_counts', lambda x: x < 15000)
        # mu.pp.filter_obs(rna, 'pct_counts_mt', lambda x: x < 20)

        rna = rna[
            :,
            [
                gene
                for gene in rna.var_names
                if not str(gene).startswith(tuple(["ERCC", "MT-", "mt-", "mt"]))
            ],
        ].copy()
        # sc.pp.filter_cells(rna, min_genes=200) #, max_counts=15000, max_genes=5000)
        # sc.pp.filter_genes(rna, min_cells=3)
        sc.pp.normalize_total(rna, target_sum=1e4)
        sc.pp.log1p(rna)

        if isinstance(self.n_top_genes, int):
            if self.n_top_genes > 0:
                sc.pp.highly_variable_genes(
                    rna, n_top_genes=self.n_top_genes, inplace=False, subset=True
                )  # batch_key='batch',
        elif self.n_top_genes is not None:
            if len(self.n_top_genes) != len(rna.var_names):
                rna = self.reindex_genes(rna, self.n_top_genes)

        if self.binary:
            rna.X = maxabs_scale(rna.X)
        self.mdata.mod["rna"] = rna

    def _transform_rna(self, x):
        x = x / x.sum() * 1e4
        x = np.log1p(x)
        return x

        self.mdata.mod["rna"] = annotate_gene(self.mdata.mod["rna"])

    def reindex_genes(self, adata, genes):
        idx = [i for i, g in enumerate(genes) if g in adata.var_names]
        print("There are {} gene in selected genes".format(len(idx)))
        if len(idx) == len(genes):
            adata = adata[:, genes].copy()
        else:
            new_X = scipy.sparse.lil_matrix((adata.shape[0], len(genes)))
            new_X[:, idx] = adata[:, genes[idx]].X
            adata = AnnData(new_X.tocsr(), obs=adata.obs, var={"var_names": genes})
        return adata

    ## ATAC specific functions

    def get_atac(self, index):
        x = self.mdata["atac"].X[index].toarray().squeeze()
        if self.mask is not None:
            index = np.where(x > 0)[0]
            index = np.random.choice(
                index, size=int(len(index) * self.mask), replace=False
            )
            x[index] = 0
        return x

    def _preprocess_atac(self):
        _require_scanpy()
        atac = self.mdata.mod["atac"]
        # sc.pp.calculate_qc_metrics(atac, percent_top=None, log1p=False, inplace=True)
        # mu.pp.filter_var(atac, 'n_cells_by_counts', lambda x: x >= 10)
        # mu.pp.filter_obs(atac, 'n_genes_by_counts', lambda x: (x >= 500) & (x <= 15000))
        # mu.pp.filter_obs(atac, 'total_counts', lambda x: (x >= 1000) & (x <= 40000))

        atac.X[atac.X > 0] = 1
        atac.var_names_make_unique()

        if isinstance(self.n_top_peaks, int):
            # epi.pp.select_var_feature(atac, nb_features=self.n_top_peaks, show=False, copy=False)
            sc.pp.highly_variable_genes(
                atac,
                n_top_genes=self.n_top_peaks,
                batch_key="batch",
                inplace=False,
                subset=True,
            )
        elif self.n_top_peaks is not None:
            if len(self.n_top_peaks) != len(atac.var_names):
                raise ValueError('n_top_peaks must be None or a list of length {}'.format(len(atac.var_names)))
                # atac = self.reindex_peak(atac, self.n_top_peaks)
        elif self.linked:
            print(
                "Linking {} peaks to {} genes".format(
                    atac.shape[1], self.mdata.mod["rna"].shape[1]
                ),
                flush=True,
            )
            gene_peak_links = self._get_gene_peak_links(dist=self.linked)
            peak_index = np.unique(gene_peak_links[1])
            gene_index = np.unique(gene_peak_links[0])
            atac = atac[:, peak_index].copy()
            self.mdata.mod["rna"] = self.mdata.mod["rna"][:, gene_index].copy()

        self.mdata.mod["atac"] = atac

    def _transform_atac(self, x):
        x[x > 1] = 1
        return x
    

    ## Multiome specific functions

    def get_multiome(self, index):
        return {
            "atac": self.get_atac(
                index
            ),  # self.mdata.mod['atac'].X[index].toarray().squeeze(),
            "rna": self.get_rna(
                index
            ),  # self.mdata.mod['rna'].X[index].toarray().squeeze()
        }

    def _preprocess_multiome(self):
        _require_scanpy()
        self._preprocess_rna()
        self._preprocess_atac()
        mu.pp.intersect_obs(self.mdata)

    def _transform_multiome(self, batch):
        return {
            "atac": self._transform_atac(batch["atac"]),
            "rna": self._transform_rna(batch["rna"]),
        }

        

    def concat_mudata(self):
        from anndata import concat

        if self.modality == "multiome":
            keys = []
            atac = concat(
                [mdata.mod["atac"] for mdata in self.mdata],
                label="dataset",
                keys=self.data_dir,
                index_unique="#",
            )
            rna = concat(
                [mdata.mod["rna"] for mdata in self.mdata],
                label="dataset",
                keys=self.data_dir,
                index_unique="#",
            )
            mdata = MuData({"atac": atac, "rna": rna})
            mu.pp.intersect_obs(mdata)
            return mdata
        elif self.modality == "rna":
            rna = concat(
                [mdata.mod["rna"] for mdata in self.mdata],
                label="dataset",
                keys=self.data_dir,
                index_unique="#",
            )
            return MuData({"rna": rna})
        elif self.modality == "atac":
            atac = concat(
                [mdata.mod["atac"] for mdata in self.mdata],
                label="dataset",
                keys=self.data_dir,
                index_unique="#",
            )
            return MuData({"atac": atac})


class ATACDataset(BaseDataset):
    def __init__(
        self,
        data_dir: str = None,
        backed: bool = False,
        **kwargs,
    ):
        super().__init__(data_dir, "atac", backed, **kwargs)

    def __getitem__(self, index):
        x = self.get_atac(index)
        if self.backed:
            x = self._transform_atac(x)
        batch = {"atac": x}
        if self.cell_types is not None:
            batch.update({"cell_type": self.cell_types[index]})
        return batch


class RNADataset(BaseDataset):
    def __init__(
        self,
        data_dir: str = None,
        backed: bool = False,
        **kwargs,
    ):
        super().__init__(data_dir, "rna", backed, **kwargs)

    def __getitem__(self, index):
        x = self.get_rna(index)
        if self.backed:
            x = self._transform_rna(x)
        batch = {"rna": x}
        if self.cell_types is not None:
            batch.update({"cell_type": self.cell_types[index]})
        return batch


class MultiOmeDataset(BaseDataset):
    def __init__(
        self,
        data_dir: str = None,
        backed: bool = False,
        **kwargs,
    ):
        super().__init__(data_dir, "multiome", backed, **kwargs)

    def __getitem__(self, index):
        if self.verbose:
            t0 = time.time()
        x = self.get_multiome(index)
        if self.backed:
            x = self._transform_multiome(x)
        if self.verbose:
            print("Time to get item: {} s".format(time.time() - t0), flush=True)
        if self.cell_types is not None:
            x.update({"cell_type": self.cell_types[index]})
        return x


class MTDataset(Dataset):
    def __init__(self, data_dir: List = None, **kwargs):
        super().__init__()

        # data_dir_keys = {key.split('_', 1)[0]:key.split('_', 1)[1] for key in data_dir} #;print(data_dir_keys)
        _keys = {}
        for key in data_dir:
            mod, name = key.split("_", 1)
            print(mod, name)
            if mod not in _keys:
                _keys[mod] = []
            _keys[mod].append(str(_datasets[mod][name]))

        print(_keys)
        self.datasets = {}
        self.datasets["atac"] = (
            ATACDataset(_keys["atac"][0], **kwargs)
            if len(_keys["atac"]) == 1
            else ConcatDataset([ATACDataset(name, **kwargs) for name in _keys["atac"]])
        )
        self.datasets["rna"] = (
            RNADataset(_keys["rna"][0], **kwargs)
            if len(_keys["rna"]) == 1
            else ConcatDataset([ATACDataset(name, **kwargs) for name in _keys["rna"]])
        )
        self.datasets["multiome"] = (
            MultiOmeDataset(_keys["multiome"][0], **kwargs)
            if len(_keys["multiome"]) == 1
            else ConcatDataset(
                [MultiOmeDataset(name, **kwargs) for name in _keys["multiome"]]
            )
        )
        print(
            len(self.datasets["atac"]),
            len(self.datasets["rna"]),
            len(self.datasets["multiome"]),
        )

    def __getitem__(self, index):
        atac = self.datasets["atac"][index]
        rna = self.datasets["rna"][index]
        multiome = self.datasets["multiome"][index]
        return {"atac": atac["atac"], "rna": rna["rna"], "multiome": multiome}

    def __len__(self):
        return min(
            [
                len(self.datasets["atac"]),
                len(self.datasets["rna"]),
                len(self.datasets["multiome"]),
            ]
        )


class RnaRiboDataset(Dataset):
    """Dataset for paired RNA (X) and RIBO (layers['ribo']) stored in a single h5ad."""

    def __init__(
        self,
        data_dir: str,
        split: float = 0.9,
        seed: int = 42,
        layer_name: str = "ribo",
        log_path: str | None = None,
        clip_min: float = -6.0,
        clip_max: float = 6.0,
    ):
        super().__init__()
        try:
            split = float(split)
        except Exception:
            raise ValueError(f"split must be float in [0,1], got {split}")
        if not (0.0 < split < 1.0):
            raise ValueError(f"split must be between 0 and 1, got {split}")
        self.path = Path(data_dir)
        if not self.path.exists():
            raise FileNotFoundError(f"h5ad not found: {self.path}")
        adata = ad.read_h5ad(self.path)
        # 持有 obs 以便下游可视化/监督
        self.obs = adata.obs.copy()
        if layer_name not in adata.layers:
            raise ValueError(f"Layer '{layer_name}' missing in {self.path}")

        self.rna = np.asarray(adata.X, dtype=np.float32)
        self.ribo = np.asarray(adata.layers[layer_name], dtype=np.float32)
        # Clip extreme values to stabilize training
        self.rna = np.clip(self.rna, clip_min, clip_max)
        self.ribo = np.clip(self.ribo, clip_min, clip_max)
        # Replace NaN/Inf with zeros to avoid NaN loss
        self.rna, rna_nan, rna_inf = self._clean_array(self.rna)
        self.ribo, ribo_nan, ribo_inf = self._clean_array(self.ribo)
        if self.rna.shape != self.ribo.shape:
            raise ValueError(f"RNA shape {self.rna.shape} != RIBO shape {self.ribo.shape}")

        self.n_cells, self.n_feats = self.rna.shape
        if "label" in self.obs.columns:
            self.label_encoder = LabelEncoder()
            self.labels = self.label_encoder.fit_transform(self.obs["label"])
            self.num_labels = int(self.labels.max() + 1)
        else:
            self.label_encoder = None
            self.labels = None
            self.num_labels = None

        rng = np.random.default_rng(seed)
        indices = np.arange(self.n_cells)
        rng.shuffle(indices)
        split_idx = int(self.n_cells * split)
        self.train_indices = indices[:split_idx]
        self.val_indices = indices[split_idx:]
        self.train_dataset = Subset(self, self.train_indices)
        self.val_dataset = Subset(self, self.val_indices)

        if log_path:
            stats = self._summaries()
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write(f"[RnaRiboDataset] path={self.path}, cells={self.n_cells}, feats={self.n_feats}\n")
                fh.write(f"  RNA  min={stats['rna']['min']:.3f} max={stats['rna']['max']:.3f} mean={stats['rna']['mean']:.3f} std={stats['rna']['std']:.3f}\n")
                fh.write(f"  RIBO min={stats['ribo']['min']:.3f} max={stats['ribo']['max']:.3f} mean={stats['ribo']['mean']:.3f} std={stats['ribo']['std']:.3f}\n")
                fh.write(f"  train_size={len(self.train_dataset)}, val_size={len(self.val_dataset)}\n")
                fh.write(f"  cleaned_nan_inf: rna_nan={rna_nan} rna_inf={rna_inf} ribo_nan={ribo_nan} ribo_inf={ribo_inf}\n")
        else:
            if any(v > 0 for v in (rna_nan, rna_inf, ribo_nan, ribo_inf)):
                print(
                    f"[RnaRiboDataset] cleaned NaN/Inf -> rna_nan={rna_nan}, rna_inf={rna_inf}, ribo_nan={ribo_nan}, ribo_inf={ribo_inf}",
                    flush=True,
                )

    def _summaries(self):
        out = {}
        for name, arr in [("rna", self.rna), ("ribo", self.ribo)]:
            flat = arr.reshape(-1)
            out[name] = {
                "min": float(np.nanmin(flat)),
                "max": float(np.nanmax(flat)),
                "mean": float(np.nanmean(flat)),
                "std": float(np.nanstd(flat)),
            }
        return out

    @staticmethod
    def _clean_array(arr):
        nan_count = int(np.isnan(arr).sum())
        inf_count = int(np.isinf(arr).sum())
        if nan_count or inf_count:
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return arr.astype(np.float32), nan_count, inf_count

    def __len__(self):
        return self.n_cells

    def __getitem__(self, idx):
        batch = {"atac": self.ribo[idx], "rna": self.rna[idx]}
        if self.labels is not None:
            batch["label"] = self.labels[idx]
        return batch


from pathlib import Path

mdata_dir = Path(__file__).parent.parent / "data/multiome"
atac_dir = Path.home() / "data/scATAC"
rna_dir = Path.home() / "data/scRNA"


_multiome_datasets = {
    "human_brain_3k": mdata_dir / "human_brain_3k.h5mu",
    "AD": mdata_dir / "AD.h5mu",
    "Pbmc10k": mdata_dir / "10x-Multiome-Pbmc10k-multi.h5mu",
    "Cortex": mdata_dir / "Cortex-2021.h5mu",
    "fetal": mdata_dir / "fetal.h5mu",
    "Chen": mdata_dir / "Chen-2019-multi.h5mu",
    "Ma": mdata_dir / "Ma-2020-multi.h5mu",
    "scorch_bin": mdata_dir / "scorch_bin_annotated.h5mu",
    "scorch": mdata_dir / "scorch_multi.h5mu",
    "ADRD": mdata_dir / "ADRD.h5mu",
    "train_multi": mdata_dir / "train_multi.h5mu",
}


# ATAC-only
_atac_datasets = {
    "adult_atlas": atac_dir / "Cell_zhang_GSE184462/atac_adult_atlas.h5ad",
    "atlas": atac_dir
    / "Cell_zhang_GSE184462/atac_atlas.h5ad",  # Cell paper and Science paper
    "fetal_old": atac_dir / "Science_Domcke_Fetal/atac_fetal_atlas.h5ad",
    "fetal": atac_dir / "Science_Domcke_Fetal/fetal_atlas_original.h5ad",
    "fetal_bin": atac_dir / "Science_Domcke_Fetal/fetal_atlas_original.h5ad",
    "scorch": mdata_dir / "SCORCH/scorch_atac_only.h5ad",
    "AD": atac_dir / "atac_AD/atac_AD.h5ad",
    "AD_bin": atac_dir / "atac_AD/atac_AD_bin.h5ad",
    "Domcke": mdata_dir / "Domcke-2020/Domcke-2020.h5ad",
}


# RNA-only
_rna_datasets = {
    "scorch": mdata_dir / "SCORCH/scorch_rna_only.h5ad",
    "AD": rna_dir / "rna_AD/rna_AD2.h5ad",  # rna_AD.h5ad backed not very well
    "adult_HCL": rna_dir / "human_atlas_10x/adult_HCL_20201006.h5ad",
    "fetal_HCL": rna_dir / "human_atlas_10x/fetal_HCL_20201006.h5ad",
    "fetal": rna_dir / "fetal_GSE156793/fetal_atlas_GSE156793.h5ad",
    "tabula_sapiens": rna_dir / "tabula_sapiens/tabula_sapiens.h5ad",
    "Cao": mdata_dir / "Cao-2020/Cao-2020.h5ad",
}

_datasets = {
    "multiome": _multiome_datasets,
    "atac": _atac_datasets,
    "rna": _rna_datasets,
    "rna_ribo": {},  # user-provided h5ad with X=rna, layers['ribo']=translation
}


# Paired
# 'scorch_bin': mdata_dir / 'SCORCH/scorch_bin_annotated.h5mu',
# 'scorch': mdata_dir / 'SCORCH/scorch_annotated.h5mu',
# 'scorch_fc': mdata_dir / 'SCORCH/scorch_fc.h5mu',
# 'scorch_na': mdata_dir / 'SCORCH/scorch_na.h5mu',
# 'ADRD': mdata_dir / 'ADRD/ADRD.h5mu',
# 'ADRD_bin': mdata_dir / 'ADRD/ADRD_bin.h5mu',
# 'ADRD': mdata_dir / 'ADRD/ADRD_ccRE.h5mu',
# 'human_brain_3k': mdata_dir / '10x/human/human_brain_3k/processed/cCRE.h5mu',
# 'human_brain_3k_bin': mdata_dir / '10x/human/human_brain_3k/processed/bin.h5mu',
# 'human_brain_3k': mdata_dir / '10x/human/human_brain_3k/processed/ccRE.h5mu',
# 'pbmc_10k_chromium-x': mdata_dir / '10x/human/pbmc_10k_chromium-x/processed/cCRE.h5mu',
# 'pbmc': mdata_dir / '10x/human/pbmc_granulocyte_sorted_10k/processed/cCRE.h5mu',
# 'pbmc_bin': mdata_dir / '10x/human/pbmc_granulocyte_sorted_10k/processed/bin.h5mu',
# 'pbmc_unsorted_3k': mdata_dir / '10x/human/pbmc_unsorted_3k/processed/cCRE.h5mu',
# 'lymph_node_lymphoma_14k': mdata_dir / '10x/human/lymph_node_lymphoma_14k/processed/cCRE.h5mu',
# 'pbmc_granulocyte_sorted_10k': mdata_dir / '10x/human/pbmc_granulocyte_sorted_10k/processed/cCRE.h5mu',
# 'pbmc_unsorted_10k': mdata_dir / '10x/human/pbmc_unsorted_10k/processed/cCRE.h5mu',

# 'open2022': mdata_dir / 'openproblem2022/train_multi.h5mu',
# 'Cortex': mdata_dir / 'Trevino-2021/Cortex-2021.h5mu',
# 'fetal': mdata_dir/'fetal/fetal.h5mu',

# scGLUE
# 'Chen': mdata_dir / 'Chen-2019/Chen-2019-multi.h5mu',
# 'Pbmc10k': mdata_dir / '10x-Multiome-Pbmc10k/10x-Multiome-Pbmc10k-multi.h5mu',
# 'Ma': mdata_dir / 'Ma-2020/Ma-2020-multi.h5mu'

# Unpaired
# 'Muto': mdata_dir / 'Muto-2021/Muto-2021-multi.h5mu',
# 'Yao':
# 'Cortext'

import torch

# from pytorch_lightning import LightningDataModule
import lightning.pytorch as pl
from torch.utils.data import DataLoader


class BaseDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = None,
        batch_size: int = 512,
        eval_batch_size: int = None,
        num_workers: int = 4,
        pin_memory: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.eval_batch_size = (
            eval_batch_size if eval_batch_size is not None else batch_size
        )
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.dataset = self.dataset_cls(self.data_dir, **kwargs)
        self.set_train_dataset()
        self.set_val_dataset()

    # @staticmethod
    # def add_model_specific_args(parent_parser):
    #     parser = parent_parser.add_argument_group("DataModule")
    #     parser.add_argument("--batch_size", type=int, default=512)
    #     parser.add_argument("--num_workers", type=int, default=4)
    #     parser.add_argument("--data_dir", type=str)
    #     parser.add_argument("--backed", action='store_true', default=False)
    #     parser.add_argument("--split", default=0.9)
    #     parser.add_argument("--n_top_genes", type=int, default=2000)
    #     # parser.add_argument("--binary", action='store_true')
    #     # parser.add_argument("--linked", type=int, default=100000)

    #     return parent_parser

    @property
    def dataset_cls(self):
        raise NotImplementedError("return tuple of dataset class")

    @property
    def dataset_name(self):
        raise NotImplementedError("return name of dataset")

    def set_train_dataset(self):
        self.train_dataset = self.dataset.train_dataset

    def set_val_dataset(self):
        self.val_dataset = self.dataset.val_dataset

    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            # collate_fn=self.train_dataset.collate,
        )
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.val_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            # collate_fn=self.val_dataset.collate,
        )
        return loader

    def test_dataloader(self):
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            # collate_fn=self.test_dataset.collate,
        )
        return loader

    def dataloader(self, dataset=None, shuffle=False):
        loader = DataLoader(
            dataset or self.dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            # collate_fn=self.train_dataset.collate,
        )
        return loader


class MultiOmeDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return MultiOmeDataset

    @property
    def dataset_name(self):
        return "multiome"


class RNADataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return RNADataset

    @property
    def dataset_name(self):
        return "rna"


class ATACDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return ATACDataset

    @property
    def dataset_name(self):
        return "atac"


class RnaRiboDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return RnaRiboDataset

    @property
    def dataset_name(self):
        return "rna_ribo"


_datamodules = {
    "atac": ATACDataModule,
    "rna": RNADataModule,
    "multiome": MultiOmeDataModule,
    "rna_ribo": RnaRiboDataModule,
}

_dataset = {
    "atac": ATACDataset,
    "rna": RNADataset,
    "multiome": MultiOmeDataset,
    "rna_ribo": RnaRiboDataset,
}


class MixDataModule(BaseDataModule):
    def __init__(self, data_dir=None, modality="multiome", *args, **kwargs):
        self.modality = modality
        if data_dir in _datasets[modality]:
            data_dir = _datasets[modality][data_dir]
        super().__init__(str(data_dir), *args, **kwargs)

    @property
    def dataset_cls(self):
        return _dataset[self.modality]

    @property
    def dataset_name(self):
        return self.modality


class MTDataModule(BaseDataModule):
    def __init__(self, _config, dist=False):
        datamodule_keys = _config["datasets"]
        assert len(datamodule_keys) > 0

        super().__init__()

        self.dm_keys = datamodule_keys
        print(datamodule_keys[0].split("_", 1))
        self.dm_dicts = {
            key: _datamodules[key.split("_")[0]](
                _datasets[key.split("_")[0]][key.split("_", 1)[1]]
            )
            for key in datamodule_keys
        }
        print(self.dm_dicts.keys())
        self.dms = [v for k, v in self.dm_dicts.items()]

        self.batch_size = self.dms[0].batch_size
        # self.vocab_size = self.dms[0].vocab_size
        self.num_workers = self.dms[0].num_workers

        self.dist = dist

        self.train_dataset = ConcatDataset([dm.train_dataset for dm in self.dms])
        self.val_dataset = ConcatDataset([dm.val_dataset for dm in self.dms])

        if self.dist:
            self.train_sampler = DistributedSampler(self.train_dataset, shuffle=True)
            self.val_sampler = DistributedSampler(self.val_dataset, shuffle=True)
            # self.test_sampler = DistributedSampler(self.test_dataset, shuffle=False)
        else:
            self.train_sampler = None
            self.val_sampler = None
