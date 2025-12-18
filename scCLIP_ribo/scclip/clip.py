import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
import os

import numpy as np
import copy
from dataclasses import dataclass
from typing import Optional, Union, Tuple, Any
from transformers.models.clip.modeling_clip import (
    BaseModelOutputWithPooling,
    ModelOutput,
)
from transformers import PreTrainedModel


from .lightning import LitModule
from .vit import ViTConfig, ViTModel

import time
from tqdm import tqdm
from anndata import AnnData, concat
from torch.utils.data import Subset

from scclip.plot import plot_umap, plot_paired_umap
from scclip.metrics import matching_metrics
from scclip.logger import create_logger
import scanpy as sc


# contrastive loss function, adapted from
# https://sachinruk.github.io/blog/pytorch/pytorch%20lightning/loss%20function/gpu/2021/03/07/CLIP.html
def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(
        logits, torch.arange(len(logits), device=logits.device)
    )


def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    atac_loss = contrastive_loss(similarity.T)
    return (caption_loss + atac_loss) / 2.0


class CLIPLoss(nn.Module):
    def __init__(self, logit_scale=2.6592, requires_grad=False):
        super().__init__()
        self.logit_scale = nn.Parameter(
            torch.ones([]) * logit_scale, requires_grad=requires_grad
        )

    def forward(self, atac_embeds, rna_embeds):
        # normalized features
        # atac_embeds = atac_embeds / atac_embeds.norm(dim=-1, keepdim=True)
        # rna_embeds = rna_embeds / rna_embeds.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_atac = torch.matmul(atac_embeds, rna_embeds.t()) * logit_scale
        logits_per_rna = logits_per_atac.T

        loss = clip_loss(logits_per_atac)

        return loss, logits_per_atac  # , logits_per_rna


def kl_div(mu, var):
    return (
        kl_divergence(
            Normal(mu, var.sqrt()), Normal(torch.zeros_like(mu), torch.ones_like(var))
        )
        .sum(dim=1)
        .mean()
    )


class CLIPModel(LitModule):
    def __init__(
        self,
        config,
        atac_config,
        rna_config,
    ):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        self._nan_warned = False

        self.atac_model = ViTModel(atac_config)
        self.rna_model = ViTModel(rna_config)
        self.num_labels = getattr(config, "num_labels", None)
        self.label_loss_weight = getattr(config, "label_loss_weight", 0.0)

        # config.hidden_size = self.atac_model.config.hidden_size
        self.atac_projection = nn.Linear(
            self.atac_model.config.hidden_size, config.projection_dim, bias=False
        )
        self.rna_projection = nn.Linear(
            self.rna_model.config.hidden_size, config.projection_dim, bias=False
        )

        self.criterion = CLIPLoss(
            self.config.logit_scale, requires_grad=self.config.requires_grad
        )
        self.label_classifier = (
            nn.Linear(config.projection_dim, self.num_labels)
            if self.num_labels
            else None
        )
        self.label_ce = nn.CrossEntropyLoss() if self.label_classifier else None

        print(f"atac_num_patches: {self.atac_model.embeddings.num_patches}", flush=True)
        print(f"rna_num_patches: {self.rna_model.embeddings.num_patches}", flush=True)

    def forward(
        self,
        atac_values=None,
        rna_values=None,
    ):
        atac_embeds = self._get_atac_features(atac_values)
        rna_embeds = self._get_rna_features(rna_values)

        return atac_embeds, rna_embeds

    def _step(self, batch, batch_idx, mode):  # TO DO: add reconstruction
        # 清理输入，避免 NaN/Inf 传播
        atac_in = torch.nan_to_num(batch["atac"], nan=0.0, posinf=0.0, neginf=0.0)
        rna_in = torch.nan_to_num(batch["rna"], nan=0.0, posinf=0.0, neginf=0.0)
        if (
            not self._nan_warned
            and (not torch.isfinite(batch["atac"]).all()
                 or not torch.isfinite(batch["rna"]).all())
        ):
            print(
                f"[CLIPModel] 非有限输入 (step={batch_idx}) "
                f"nan_atac={torch.isnan(batch['atac']).sum().item()} "
                f"nan_rna={torch.isnan(batch['rna']).sum().item()}",
                flush=True,
            )
            self._nan_warned = True

        atac_embeds, rna_embeds = self(atac_in, rna_in)

        if (
            not self._nan_warned
            and (not torch.isfinite(atac_embeds).all()
                 or not torch.isfinite(rna_embeds).all())
        ):
            print(
                f"[CLIPModel] 非有限 embedding (step={batch_idx}) "
                f"nan_atac_embeds={torch.isnan(atac_embeds).sum().item()} "
                f"nan_rna_embeds={torch.isnan(rna_embeds).sum().item()}",
                flush=True,
            )
            self._nan_warned = True
        loss, similarity = self.criterion(atac_embeds, rna_embeds)
        label_loss = None
        if (
            self.label_classifier is not None
            and "label" in batch
            and self.label_loss_weight > 0
        ):
            labels = batch["label"].long()
            atac_logits = self.label_classifier(atac_embeds)
            rna_logits = self.label_classifier(rna_embeds)
            label_loss = (
                self.label_ce(atac_logits, labels) + self.label_ce(rna_logits, labels)
            ) / 2.0
            loss = loss + self.label_loss_weight * label_loss

        acc, matchscore, foscttm = matching_metrics(similarity)
        log_dict = {
            f"acc/{mode}": acc,
            f"matchscore/{mode}": matchscore,
            f"foscttm/{mode}": foscttm,
            f"loss/{mode}": loss,
        }
        if label_loss is not None:
            log_dict[f"label_loss/{mode}"] = label_loss

        # logit_scale learnable
        if self.config.requires_grad:
            log_dict.update({"logit_scale": self.criterion.logit_scale})

        if mode == "predict":
            return atac_embeds, rna_embeds, log_dict

        # log_dict.update({f'loss/{mode}': loss})
        self.log_dict(log_dict, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        # At the moment fast initialization is not supported
        # for composite models
        kwargs["_fast_init"] = False
        return super().from_pretrained(*args, **kwargs)

    def _get_rna_features(self, rna_values=None):
        rna_outputs = self.rna_model(rna_values)

        rna_features = rna_outputs[1]
        rna_features = self.rna_projection(rna_features)

        if self.config.normalize:
            denom = rna_features.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            rna_features = rna_features / denom

        return rna_features

    def _get_atac_features(self, atac_values=None):
        atac_outputs = self.atac_model(atac_values)

        atac_features = atac_outputs[1]  # pooled_output
        atac_features = self.atac_projection(atac_features)

        if self.config.normalize:
            denom = atac_features.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            atac_features = atac_features / denom

        return atac_features

    def _get_batch_features(
        self, dataloader, modality="rna", cell_type="cell_type", out_dir="."
    ):
        if isinstance(dataloader.dataset, Subset):
            obs = dataloader.dataset.dataset.mdata.obs.iloc[dataloader.dataset.indices]
        else:
            obs = dataloader.dataset.mdata.obs

        self.to("cuda")
        fn = self._get_rna_features if modality == "rna" else self._get_atac_features
        adata = torch.concat(
            [
                fn(batch[modality].to("cuda")).detach().cpu()
                for batch in tqdm(dataloader, desc=modality)
            ]
        ).numpy()
        adata = AnnData(adata, obs=obs)
        sc.settings.figdir = out_dir
        plot_umap(adata, color=cell_type, metric="cosine", save=f"_{modality}.png")
        adata.write(f"{out_dir}/{modality}.h5ad")
        return adata

    def get_batch_features(
        self, dataloader, atac=None, rna=None, celltype="cell_type", out_dir="."
    ):
        log = create_logger("", fh=out_dir + "/log.txt")
        if not self.config.normalize:
            out_dir = f"{out_dir}_no_norm"

        if dataloader is not None:
            rna_embeds = self._get_batch_features(
                dataloader, modality="rna", out_dir=out_dir
            )
            atac_embeds = self._get_batch_features(
                dataloader, modality="atac", out_dir=out_dir
            )

        if rna is not None:
            rna_embeds = self._get_batch_features(
                rna.dataloader(), out_dir=out_dir, modality="rna"
            )
        if atac is not None:
            atac_embeds = self._get_batch_features(
                atac.dataloader(), out_dir=out_dir, modality="atac"
            )

        if atac is not None and rna is not None or dataloader is not None:
            acc, match_score, foscttm = matching_metrics(
                x=atac_embeds.obsm["X_umap"], y=rna_embeds.obsm["X_umap"]
            )
            if log is not None:
                log.info(
                    f"atac-rna\nacc: {acc:.4f}\nmatching score: {match_score:.4f}\nfoscttm: {foscttm:.4f}"
                )
            else:
                print(
                    f"atac-rna\nacc: {acc:.4f}\nmatching score: {match_score:.4f}\nfoscttm: {foscttm:.4f}",
                    flush=True,
                )
            concat_embeds = concat(
                [atac_embeds, rna_embeds],
                label="modality",
                keys=["atac", "rna"],
                index_unique="#",
            )
            sc.settings.figdir = out_dir
            if dataloader is not None:
                plot_umap(
                    concat_embeds,
                    color=[celltype, "modality"],
                    metric="cosine",
                    save="_concat.png",
                )
                # plot_paired_umap(concat_embeds, color=celltype, save=os.path.join(out_dir, 'umap_concat.png'))
            else:
                plot_umap(
                    concat_embeds, color=celltype, metric="cosine", save="_concat.png"
                )
            concat_embeds.write(f"{out_dir}/concat.h5ad")
