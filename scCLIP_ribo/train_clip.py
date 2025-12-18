#!/usr/bin/env python

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    EarlyStopping,
    Callback,
    ModelCheckpoint,
)

from scclip.data import MixDataModule
from scclip.data import RnaRiboDataModule
from scclip.clip import CLIPModel
from scclip.vit import ViTConfig
from scclip.callback import Monitor, LossLoggingCallback
from scclip.config import get_model_config
from scclip.logger import create_logger


import os
import argparse
import logging


from pathlib import Path

HOME = Path.home()
print("Start", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="clip")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--checkpoint", type=str, default=None)
    # DataModule
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--data_dir", type=str, default="human_brain_3k")
    parser.add_argument("--backed", action="store_true", default=False)
    parser.add_argument("--split", type=float, default=0.9)
    parser.add_argument("--n_top_genes", type=int, default=None)
    parser.add_argument("--binary", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dist", type=str, default=None)
    parser.add_argument("--mask", type=float, default=None)
    parser.add_argument("--peak_dist", type=int, default=10_000)
    parser.add_argument(
        "--experiment", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--mod", type=str, default="multiome")
    parser.add_argument("--atac", type=str, default=None)
    parser.add_argument("--rna", type=str, default=None)

    # Module
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--vae_path", type=str, default=None)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--ffn_dim", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.1)
    parser.add_argument(
        "--use_imputed", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--projection_dim", type=int, default=128)
    parser.add_argument(
        "--requires_grad", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--normalize", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--label_loss_weight", type=float, default=0.8)
    parser.add_argument("--version", type=str, default="")

    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--fast_dev_run", action="store_true", default=False)

    # parser.add_argument('--version', type=str, default='v2')
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--no_scalex", action="store_true", default=False)
    parser.add_argument(
        "--use_val", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--test_data", type=str, default="ADRD")
    parser.add_argument("--cell_type", type=str, default="cell_type")
    parser.add_argument(
        "--use_seq", action=argparse.BooleanOptionalAction, default=False
    )
    # 温度初始化略高，并默认可学习，提高判别力
    parser.add_argument("--logit_scale", type=float, default=3.0)  # exp(3)≈20.1, small data更易分辨
    parser.add_argument("--num_patches", type=int, default=128)
    parser.add_argument(
        "--early_stop", action=argparse.BooleanOptionalAction, default=False
    )

    # parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    seed_everything(args.seed)

    logger_obj = logging.getLogger("scclip_run")
    logger_obj.setLevel(logging.INFO)
    logger_obj.handlers.clear()

    if args.checkpoint is None:
        if args.mod == "rna_ribo":
            dm = RnaRiboDataModule(
                data_dir=args.data_dir,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                split=args.split,
            )
            feature_size = dm.dataset.n_feats
            args.num_labels = dm.dataset.num_labels
            if dm.dataset.labels is not None:
                label_counts = dm.dataset.obs["label"].value_counts()
                logger_obj.info(f"labels detected: {dm.dataset.label_encoder.classes_.tolist()}")
                logger_obj.info(f"label counts: {label_counts.to_dict()}")
        else:
            dm = MixDataModule(
                data_dir=args.data_dir,
                modality=args.mod,
                batch_size=args.batch_size,
                linked=args.dist,
                split=args.split,
                n_top_genes=args.n_top_genes,
                binary=args.binary,
                use_seq=args.use_seq,
                mask=args.mask,
            )
            feature_size = dm.dataset.mdata.mod["rna"].shape[1]
            args.peaks = dm.dataset.mdata.mod["atac"].var
            args.genes = dm.dataset.mdata.mod["rna"].var
            args.num_labels = None
            args.label_loss_weight = 0.0  # 非 rna_ribo 模态不使用标签监督

        model_config = get_model_config("small")
        atac_config = ViTConfig(
            **{
                "modality": "atac",
                "num_patches": args.num_patches,
                "feature_size": feature_size if args.mod == "rna_ribo" else dm.dataset.mdata.mod["atac"].shape[1],
                "attention_probs_dropout_prob": args.dropout,
                "hidden_dropout_prob": args.dropout,
                **model_config,
            }
        )

        rna_config = ViTConfig(
            **{
                "modality": "rna",
                "num_patches": args.num_patches,
                "attention_probs_dropout_prob": args.dropout,
                "hidden_dropout_prob": args.dropout,
                "feature_size": feature_size if args.mod == "rna_ribo" else dm.dataset.mdata.mod["rna"].shape[1],
                **model_config,
            }
        )

        model = CLIPModel(
            args,
            atac_config=atac_config,
            rna_config=rna_config,
        )

        # print(model, flush=True)

        # out_dir (use basename to avoid nesting absolute paths)
        data_label = Path(args.data_dir).stem if Path(args.data_dir).suffix else Path(args.data_dir).name
        if args.experiment:
            args.default_root_dir = f"results/{data_label}/{args.mod}_{args.logit_scale}_{args.requires_grad}_{args.max_steps}_{args.lr}_{args.version}"
        else:
            args.default_root_dir = (
                f"results/{data_label}/{args.mod}_{args.logit_scale}_{args.max_steps}"
            )
        # os.makedirs(args.default_root_dir, exist_ok=True)
        print("default_root_dir:", args.default_root_dir, flush=True)
        os.makedirs(args.default_root_dir, exist_ok=True)
        log_file = Path(args.default_root_dir) / "run.log"
        fh = logging.FileHandler(log_file)
        formatter = logging.Formatter("%(asctime)s - %(message)s")
        fh.setFormatter(formatter)
        logger_obj.addHandler(fh)
        logger_obj.addHandler(logging.StreamHandler())
        logger_obj.info(
            f"hyperparams mod={args.mod} batch_size={args.batch_size} lr={args.lr} "
            f"logit_scale={args.logit_scale} max_steps={args.max_steps} "
            f"label_loss_weight={args.label_loss_weight} num_labels={getattr(args, 'num_labels', None)}"
        )
        logger_obj.info(
            f"start training mod={args.mod} data={args.data_dir} feature_size={feature_size} "
            f"train={len(dm.train_dataset)} val={len(dm.val_dataset)}"
        )

        # trainer
        try:
            logger = TensorBoardLogger(
                save_dir=args.default_root_dir, default_hp_metric=False, version=""
            )
        except ModuleNotFoundError:
            logger_obj.warning("TensorBoard not available; falling back to CSVLogger.")
            csv_dir = Path(args.default_root_dir) / "csv_logs"
            csv_dir.mkdir(parents=True, exist_ok=True)
            logger = CSVLogger(save_dir=args.default_root_dir, name="csv_logs")
        checkpoint_cb = ModelCheckpoint(
            monitor="loss/val",
            mode="min",
            save_top_k=1,
            filename="best-{epoch:02d}",
            save_last=True,
        )
        callbacks = [LossLoggingCallback(), checkpoint_cb]
        if args.mod != "rna_ribo":
            callbacks.append(Monitor(dm, metric="cosine"))
        # LearningRateMonitor could be added here if needed
        if args.early_stop:
            callbacks.append(EarlyStopping(monitor="loss/val", mode="min", patience=10))
        trainer = Trainer(
            callbacks=callbacks,
            accelerator="gpu",
            devices=1,
            gradient_clip_val=5,
            num_sanity_val_steps=0,
            logger=logger,
            max_steps=args.max_steps,
            fast_dev_run=args.fast_dev_run,
        )

        # fit
        trainer.fit(model, dm)
        logger_obj.info("training finished")

    else:
        model = CLIPModel.load_from_checkpoint(args.checkpoint)
        print("normalize", args.normalize, flush=True)
        model.config.normalize = args.normalize
        args.default_root_dir = args.checkpoint.split("lightning_logs/")[0]

        os.makedirs(args.default_root_dir, exist_ok=True)
        log_file = Path(args.default_root_dir) / "run.log"
        fh = logging.FileHandler(log_file)
        formatter = logging.Formatter("%(asctime)s - %(message)s")
        fh.setFormatter(formatter)
        logger_obj.addHandler(fh)
        logger_obj.addHandler(logging.StreamHandler())
        logger_obj.info(f"loaded checkpoint {args.checkpoint} for mod={args.mod}")

        if args.mod == "rna_ribo":
            dm = RnaRiboDataModule(
                data_dir=args.data_dir,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                split=args.split,
            )
        else:
            dm = MixDataModule(
                data_dir=args.data_dir,
                modality=args.mod,
                batch_size=args.batch_size,
                n_top_peaks=model.config.peaks,
                n_top_genes=model.config.genes.index,
                binary=model.config.binary,
                use_seq=model.config.use_seq,
            )

    if not args.fast_dev_run:
        if args.mod == "rna_ribo":
            logger_obj.info("Skipping get_batch_features for rna_ribo modality")
        else:
            out_dir = os.path.join(args.default_root_dir, args.data_dir)
            os.makedirs(out_dir, exist_ok=True)

            if args.mod == "multiome":
                if args.data_dir == model.config.data_dir:
                    dataloader = dm.dataloader() #val_dataloader()
                else:
                    dataloader = dm.dataloader()
            if args.rna:
                rna_dm = MixDataModule(
                    data_dir=args.data_dir,
                    modality="rna",
                    batch_size=args.batch_size,
                    n_top_peaks=model.config.peaks,
                    n_top_genes=model.config.genes.index,
                    binary=model.config.binary,
                    use_seq=model.config.use_seq,
                )
            else:
                rna_dm = None
            if args.atac:
                atac_dm = MixDataModule(
                    data_dir=args.data_dir,
                    modality="atac",
                    batch_size=args.batch_size,
                    n_top_peaks=model.config.peaks,
                    n_top_genes=model.config.genes.index,
                    binary=model.config.binary,
                    use_seq=model.config.use_seq,
                )
            else:
                atac_dm = None

            model.get_batch_features(dataloader, atac_dm, rna_dm, out_dir=out_dir)
