#!/usr/bin/env python3
"""
Fine-tune Surya on binary EUV-wave-event classification.

Adapted from ``3_finetune_template_1D.py``. Same shape — ``build_datasets()`` /
``build_model()`` hold the task-specific content, ``build_trainer()`` and ``main()`` are
generic — with the additions the wave scaling study needs:

* **Its own DataLoaders.** ``build_helio_dataloaders()`` hard-codes ``drop_last=True`` for
  validation and does not expose ``prefetch_factor``. On a 24-sample validation set the
  first silently discards samples from the metric that selects checkpoints, and the second
  is the only bound on worker memory (``ts`` is 1.74 GB per sample). See
  ``experiments/wave_common.py:build_wave_loader()``.
* **Event-level stratified subsets** (``--train-n``). The catalog stores each event twice,
  as ``wave`` at ``start_time + 30 min`` and ``no wave`` at ``start_time - 30 min``, so
  subsetting by event is exactly class-balanced and the subsets are nested across N —
  which is what lets a learning curve attribute a change to sample count alone.
  ``waveDSDataset``'s ``max_number_of_samples`` cannot do this: it head-slices a
  time-sorted frame, so ``10`` means the 5 *earliest* pairs.
* **Pre-flight assertions.** Every failure this task has actually hit was silent and still
  produced a healthy-looking loss curve: a patch embedding left random because its shape
  disagreed with the checkpoint, LoRA target names that matched nothing, a head frozen at
  initialization. These are checked before the first step.
* **Gradient clipping, gradient accumulation and a wall-clock cap**, so one run in an
  unattended schedule cannot starve the rest.
* **A resource guard** that aborts on runaway memory instead of letting the machine die.
* **Held-out evaluation** of the best checkpoint on the official test split, plus figures
  written to disk and to wandb.

Usage — an LR probe run, then a scaling run. ``experiments/run_surya_schedule.sh`` drives
the whole sequence; these are what it invokes.

    python -m downstream_apps.test.4_finetune_wave_1D --run-name probe_lr1e-4 \
        --lr 1e-4 --max-steps 60 --accum 1 --train-n 48 \
        --max-time 00:25:00 --no-eval-test --no-checkpoint

    python -m downstream_apps.test.4_finetune_wave_1D --run-name surya_D_n384 \
        --train-n 384 --max-epochs 12 --max-time 04:40:00 --accum 4 --lr 1e-4
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Must be set BEFORE torch is imported: cuBLAS reads this once, when it initializes, so
# setting it later has no effect and every run under training.deterministic then warns.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from pathlib import Path
from typing import Tuple

import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from torch.utils.data import DataLoader

from downstream_apps.test.configs import TrainingConfig, load_wave_config
from downstream_apps.test.experiments import wave_common as wc
from downstream_apps.test.lightning_modules.pl_simple_baseline import WaveLightningModule
from downstream_apps.test.metrics.template_metrics import WaveMetrics
from workshop_infrastructure.assets import ensure_assets
from workshop_infrastructure.utils import (
    UploadBestCheckpointToS3,
    apply_peft_lora,
    build_scalers,
    load_pretrained_weights,
)

DEFAULT_CONFIG = Path(__file__).parent / "configs" / "config_wave_full.yaml"
RESULTS_DIR = Path(__file__).parent / "experiments" / "results"

_DETERMINISTIC_CLI = {"false": False, "warn": "warn", "true": True}

# Trainable-parameter counts for the regimes this study runs, pinned so a silently
# misconfigured run fails at startup instead of in the loss curve. Keys are
# (use_lora, lora_r, penultimate_linear_layer, freeze_backbone).
_EXPECTED_TRAINABLE = {
    (True, 8, False, False): 1_518_081,
    (True, 8, True, False): 3_157_761,
    (False, 8, False, True): 2_561,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(DEFAULT_CONFIG),
                   help="Run config YAML (default: this app's config_wave_full.yaml).")
    # Dev toggles
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--train_baseline", action="store_true",
                   help="Train the AIA193 running-difference logistic baseline instead of "
                        "Surya, through these same DataLoaders.")
    p.add_argument("--no-eval-test", action="store_true",
                   help="Skip the held-out test-split evaluation of the best checkpoint.")
    p.add_argument("--no-checkpoint", action="store_true",
                   help="Do not write checkpoints. Each one is 1.7 GB (the state dict carries "
                        "the frozen backbone, not just the adapters), so the LR probe — whose "
                        "weights are never used — should not write three of them.")
    # Overrides that already exist on TrainingConfig
    p.add_argument("--max-epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--lr", type=float, default=None,
                   help="Override training.learning_rate. The config's 0.01 was tuned on "
                        "the 16 K-param logistic baseline and DIVERGES Surya.")
    p.add_argument("--s3-cache-dir", default=None)
    p.add_argument("--deterministic", choices=tuple(_DETERMINISTIC_CLI), default=None)
    # Run-scale settings with no field on TrainingConfig
    p.add_argument("--prefetch-factor", type=int, default=1,
                   help="DataLoader batches prefetched per worker. Worst-case worker RSS is "
                        "num_workers * this * batch_size * 1.74 GB, so 1 is not a tuning "
                        "detail: the PyTorch default of 2 doubles it.")
    p.add_argument("--accum", type=int, default=4,
                   help="accumulate_grad_batches. One 65,536-token sample already saturates "
                        "the GPU, so a larger effective batch comes from here, not from "
                        "--batch-size (which is flat-to-worse in s/sample above 2).")
    p.add_argument("--max-time", default=None,
                   help='Hard wall-clock cap. Accepts "HH:MM:SS" or "MM:SS" as well as the '
                        '"DD:HH:MM:SS" Lightning itself requires. Training stops cleanly at '
                        "the cap with the best checkpoint intact, so one overrunning run "
                        "cannot starve the rest of a schedule.")
    p.add_argument("--max-steps", type=int, default=-1,
                   help="Cap on optimizer steps (not batches). Used by the LR probe.")
    p.add_argument("--train-n", type=int, default=None,
                   help="Training samples, event-level stratified (even). Default: all "
                        "complete pairs.")
    p.add_argument("--subset-seed", type=int, default=42,
                   help="Selects which events. Keep fixed across a scaling study so the "
                        "subsets stay nested.")
    p.add_argument("--lora-r", type=int, default=None, help="Override model.lora_config.r.")
    p.add_argument("--penultimate", dest="penultimate", action="store_true", default=None,
                   help="Turn model.penultimate_linear_layer back on (adds 1.64 M params "
                        "and, with no nonlinearity after it, no representational capacity).")
    p.add_argument("--no-penultimate", dest="penultimate", action="store_false")
    p.add_argument("--run-name", default=None,
                   help="Names the wandb run, the CSV log directory and the checkpoint "
                        "directory. Defaults to job_id.")
    p.add_argument("--cache-budget-gb", type=float, default=34.0,
                   help="Fast local storage to spend on pinned frames, in GB. Clamped to the "
                        "filesystem's free space less a checkpoint reserve. 34 GB covers the "
                        "48-frame validation split (~28 GB) plus a few training frames, which "
                        "is where nearly all the value is: at 380 MB/s the smallest training "
                        "run is GPU-bound whether or not its frames are cached, whereas "
                        "validation is re-read on every epoch of every run.")
    p.add_argument("--cache-priority-n", type=int, default=48,
                   help="After the validation split, pin this nested training prefix. "
                        "Because the event subsets are nested, frames pinned for N=48 "
                        "are also read by every larger run. 0 disables train pinning.")
    p.add_argument("--disk-cache", action="store_true",
                   help="Use the stock s3_mode read path (every frame written to "
                        "s3_cache_dir) instead of reading through RAM. The configured "
                        "cache is on an EFS mount measured at 9 MB/s against 400 MB/s "
                        "for the memory path, so this is for debugging only.")
    p.add_argument("--rss-ceiling-gb", type=float, default=70.0,
                   help="Abort if the process tree's proportional set size crosses this.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Resource guard
# ---------------------------------------------------------------------------

def normalize_max_time(value: str | None) -> str | None:
    """Pad a duration to the ``DD:HH:MM:SS`` that ``Trainer(max_time=...)`` demands.

    Lightning rejects ``"01:50:00"`` outright — it wants four fields, so an hours-minutes-
    seconds duration has to carry an explicit day count. That is a surprising failure to hit
    after a model has loaded, and a schedule of runs written in the obvious ``HH:MM:SS`` would
    fail one run at a time, so the shorter forms are accepted and padded here instead.

    >>> normalize_max_time("01:50:00")
    '00:01:50:00'
    >>> normalize_max_time("00:04:40:00")
    '00:04:40:00'
    """
    if value is None:
        return None
    parts = str(value).strip().split(":")
    if not 2 <= len(parts) <= 4 or not all(p.isdigit() for p in parts):
        raise ValueError(
            f"--max-time {value!r} is not a duration. Use DD:HH:MM:SS, HH:MM:SS or MM:SS."
        )
    parts = ["0"] * (4 - len(parts)) + parts
    return ":".join(f"{int(p):02d}" for p in parts)


class ResourceGuard(L.Callback):
    """Log GPU and host memory periodically, and abort before the machine dies.

    The hazard here is host RAM, not VRAM. One sample of ``ts`` is
    ``(13, 2, 4096, 4096)`` fp32 = 1.74 GB, and every DataLoader worker holds
    ``prefetch_factor * batch_size`` of them, so a mis-set worker count is the difference
    between 27.8 GB and 56 GB on a 124 GB machine that is also holding a 1.8 GB checkpoint
    and pinned copies of every batch. An OOM kill takes the whole schedule down with it;
    raising here loses one run.

    Proportional set size (PSS) is used rather than RSS: workers share the parent's
    copy-on-write pages, so summing RSS across the tree double-counts them and would
    trip the ceiling on a healthy run.
    """

    def __init__(self, every_n_steps: int = 25, ceiling_gb: float = 70.0):
        self.every_n_steps = every_n_steps
        self.ceiling_gb = ceiling_gb
        self.peak_host_gb = 0.0
        self._proc = None

    def _tree_memory_gb(self) -> tuple[float, str]:
        import psutil
        if self._proc is None:
            self._proc = psutil.Process()
        procs = [self._proc] + self._proc.children(recursive=True)
        total, kind = 0.0, "pss"
        for p in procs:
            try:
                info = p.memory_full_info()
                total += getattr(info, "pss", None) or info.rss
                if not hasattr(info, "pss"):
                    kind = "rss"
            except Exception:
                continue  # a worker exiting between listing and reading is not an error
        return total / 2**30, kind

    def _report(self, trainer, tag: str) -> None:
        host_gb, kind = self._tree_memory_gb()
        self.peak_host_gb = max(self.peak_host_gb, host_gb)
        msg = f"[RES] {tag} host {kind}={host_gb:.1f} GB (peak {self.peak_host_gb:.1f})"
        if torch.cuda.is_available():
            alloc = torch.cuda.max_memory_allocated() / 2**30
            reserved = torch.cuda.max_memory_reserved() / 2**30
            msg += f" | cuda peak alloc={alloc:.1f} GB reserved={reserved:.1f} GB"
        print(msg, flush=True)
        if host_gb > self.ceiling_gb:
            raise RuntimeError(
                f"Host memory {host_gb:.1f} GB crossed the {self.ceiling_gb:.1f} GB ceiling. "
                f"Lower --num-workers or --prefetch-factor (worst case is "
                f"num_workers * prefetch_factor * batch_size * 1.74 GB), or raise "
                f"--rss-ceiling-gb if this machine really has the headroom."
            )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % self.every_n_steps == 0:
            self._report(trainer, f"epoch {trainer.current_epoch} batch {batch_idx}")

    def on_validation_epoch_end(self, trainer, pl_module):
        if not trainer.sanity_checking:
            self._report(trainer, f"epoch {trainer.current_epoch} val end")


# ---------------------------------------------------------------------------
# Build functions
# ---------------------------------------------------------------------------

def build_datasets(
    cfg: TrainingConfig,
    scalers,
    args: argparse.Namespace,
) -> Tuple[DataLoader, DataLoader, DataLoader, dict]:
    """Create the train / val / test DataLoaders, subset at the event level.

    Returns the three loaders plus a dict of the subset statistics, which are asserted
    against the requested counts and then logged.
    """
    train_ds, val_ds, test_ds = wc.build_wave_datasets(
        cfg, scalers, include_test=True, in_memory=not args.disk_cache
    )
    print(f"[DATA] read path: "
          f"{'stock s3_mode=' + cfg.data.s3_mode if args.disk_cache else 'in-memory (BytesIO -> h5netcdf)'}")

    stats = {}
    for name, ds, n in [
        ("train", train_ds, args.train_n),
        ("val", val_ds, None),
        ("test", test_ds, None),
    ]:
        info = wc.event_stratified_subset(ds, n_samples=n, seed=args.subset_seed)
        cached, total = wc.cache_coverage(ds)
        print(f"[DATA] {name}: {info.describe()}")
        print(f"[DATA] {name}: {total} distinct frames, {cached} cached "
              f"({100 * cached / max(total, 1):.1f}%)")
        stats[name] = {
            "n_samples": info.n_samples, "n_events": info.n_events,
            "n_positive": info.n_positive, "n_negative": info.n_negative,
            "frames": total, "frames_cached": cached,
        }

    # Decide what goes on fast local storage. The read path sustains ~175-420 MB/s from S3
    # against ~3.1 s of GPU per sample, so a fully-streaming run is data-bound; local disk is
    # 350 MB/s but there is only ~66 GB of it against a 481 GB working set. The budget goes
    # to the validation split (re-read every epoch) and then to the smallest nested training
    # subset, whose frames every larger run also reads. See plan_pinned_frames().
    pinned, pin_report = wc.plan_pinned_frames(
        val_ds, train_ds,
        budget_gb=args.cache_budget_gb,
        train_priority_n=args.cache_priority_n or None,
        subset_seed=args.subset_seed,
    )
    n_pinned = wc.pin_frames(val_ds, extra_datasets=(train_ds, test_ds), uris=pinned)
    if n_pinned:
        print(f"[DATA] pinned {n_pinned} frames (~{pin_report['estimated_gb']} GB) to "
              f"{cfg.data.s3_cache_dir}: {pin_report['val_frames_pinned']}/"
              f"{pin_report['val_frames_total']} val + "
              f"{pin_report['train_frames_pinned']} train "
              f"(priority N={pin_report['train_priority_n']})")
    stats["cache_plan"] = pin_report

    if args.train_n is not None and stats["train"]["n_samples"] != args.train_n:
        raise AssertionError(
            f"Requested --train-n {args.train_n} but the subset has "
            f"{stats['train']['n_samples']} samples."
        )

    workers = cfg.num_workers
    common = dict(
        batch_size=cfg.batch_size,
        num_workers=workers,
        prefetch_factor=args.prefetch_factor,
        seed=cfg.seed,
    )
    train_loader = wc.build_wave_loader(train_ds, shuffle=True, drop_last=True, **common)
    # drop_last=False on the held-out splits: they are small enough that one dropped
    # sample is percent-level noise in the metric that selects checkpoints.
    val_loader = wc.build_wave_loader(val_ds, shuffle=False, drop_last=False, **common)
    test_loader = wc.build_wave_loader(test_ds, shuffle=False, drop_last=False, **common)
    return train_loader, val_loader, test_loader, stats


def build_metrics() -> dict:
    """The four metric modes. ``val_loss`` is what ModelCheckpoint monitors."""
    return {
        "train_loss": WaveMetrics("train_loss"),
        "val_loss": WaveMetrics("val_loss"),
        "train_metrics": WaveMetrics("train_metrics"),
        "val_metrics": WaveMetrics("val_metrics"),
    }


def build_model(cfg: TrainingConfig, scalers, train_baseline: bool = False) -> L.LightningModule:
    """Instantiate the model, run the pre-flight assertions, and wrap it for Lightning."""
    metrics = build_metrics()

    if train_baseline:
        from functools import partial

        from downstream_apps.test.models.simple_baseline import (
            RunningDifferenceLogisticModel,
            destandardize_channels,
        )
        # channel_index must come from the config, not a literal: AIA193 is index 3 in the
        # 13-channel stack but index 0 whenever data.channels is narrowed to ["aia193"].
        # img_size is deliberately not passed — the model pools with a fixed kernel and then
        # averages over whatever remains, so it is resolution-independent.
        model = RunningDifferenceLogisticModel(
            channel_index=cfg.data.channels.index("aia193"),
        )
        preprocess_fn = partial(
            destandardize_channels, channel_order=cfg.data.channels, scalers=scalers
        )
        print(f"[MODEL] running-difference baseline, "
              f"{sum(p.numel() for p in model.parameters()):,} parameters")
        return WaveLightningModule(model, metrics, lr=cfg.learning_rate,
                                   batch_size=cfg.batch_size, preprocess_fn=preprocess_fn)

    from workshop_infrastructure.models.finetune_models import HelioSpectformer1D
    model = HelioSpectformer1D.from_config(
        cfg.model,
        num_outputs=1,
        dtype=cfg.dtype,
        use_latitude_in_learned_flow=cfg.use_latitude_in_learned_flow,
    )
    load_pretrained_weights(model, cfg.model.pretrained_path)
    # Bug 1's guard: with time_dim=1 the tokenizer is Conv2d(13, ...) while the
    # checkpoint's is Conv2d(26, ...), the key is skipped, and every image is tokenized by
    # random weights for the whole run.
    wc.assert_tokenizer_loaded(model, cfg.model.pretrained_path)

    if cfg.model.freeze_backbone:
        for name, param in model.named_parameters():
            if name.startswith("backbone."):
                param.requires_grad = False
    if cfg.model.use_lora:
        model = apply_peft_lora(model, cfg.model.lora_config)
        wc.assert_lora_adapted_attention(model)

    key = (cfg.model.use_lora, cfg.model.lora_config.r,
           cfg.model.penultimate_linear_layer, cfg.model.freeze_backbone)
    wc.assert_trainable_count(model, _EXPECTED_TRAINABLE.get(key))
    if key not in _EXPECTED_TRAINABLE:
        print(f"[PREFLIGHT] no pinned trainable count for {key}; reported, not asserted")

    return WaveLightningModule(model, metrics, lr=cfg.learning_rate, batch_size=cfg.batch_size)


def build_trainer(
    cfg: TrainingConfig,
    args: argparse.Namespace,
    run_name: str,
) -> Tuple[L.Trainer, ModelCheckpoint, ResourceGuard, CSVLogger]:
    """Configure loggers, callbacks and the Trainer.

    The CSVLogger is returned explicitly rather than being fished back out of the Trainer:
    ``Trainer.log_dir`` is the *first* logger's directory, which is wandb's whenever wandb is
    enabled, so reading ``metrics.csv`` from it would silently find nothing and the
    training-curve figure would come out empty.
    """
    loggers = []
    if not args.no_wandb:
        loggers.append(WandbLogger(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=run_name,
            log_model=False,
            save_dir=os.environ.get("TMPDIR", "./wandb/wandb_tmp"),
        ))
    csv_logger = CSVLogger("runs", name=run_name)
    loggers.append(csv_logger)

    ckpt_dir = Path(cfg.output.ckpt_dir) / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="best-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        # save_top_k=0 still tracks best_model_score, so the probe's comparison table works
        # without any 1.7 GB files being written.
        save_top_k=0 if args.no_checkpoint else 1,
        save_last=False,
    )
    upload_cb = UploadBestCheckpointToS3(
        checkpoint_cb=checkpoint_cb,
        bucket=cfg.output.s3_bucket,
        prefix=cfg.output.s3_prefix,
        fixed_key_name=(cfg.output.s3_best_key or None),
    )
    guard = ResourceGuard(ceiling_gb=args.rss_ceiling_gb)

    trainer = L.Trainer(
        max_epochs=args.max_epochs if args.max_epochs is not None else cfg.max_epochs,
        max_steps=args.max_steps,
        max_time=normalize_max_time(args.max_time),
        accumulate_grad_batches=args.accum,
        # Surya diverged once already at lr=0.01. Clipping is cheap insurance that a bad
        # batch cannot take the run with it, and it costs nothing when gradients are small.
        gradient_clip_val=1.0,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices="auto",
        strategy="auto",
        precision="bf16-mixed" if torch.cuda.is_available() else "32-true",
        deterministic=cfg.deterministic,
        benchmark=False,
        logger=loggers,
        callbacks=[checkpoint_cb, upload_cb, guard],
        log_every_n_steps=5,
        # The overnight schedule's logs are the primary record of what happened, and a
        # progress bar redirected to a file writes one carriage-returned line per update —
        # thousands of them, with the ResourceGuard's memory reports buried inside. Off unless
        # someone is actually watching a terminal.
        enable_progress_bar=sys.stdout.isatty(),
    )
    return trainer, checkpoint_cb, guard, csv_logger


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_logits(lit_model: L.LightningModule, loader: DataLoader, device) -> tuple:
    """Run one forward pass over ``loader``, returning ``(logits, labels)`` as NumPy.

    Used instead of ``trainer.validate`` for the held-out evaluation because it also yields
    the per-sample scores the ROC curve and score histogram need, so the split is read
    once rather than twice — at ~3 s of GPU per sample that is worth arranging.
    """
    lit_model.eval().to(device)
    logits, labels = [], []
    for batch in loader:
        labels.append(batch["forecast"].float().numpy().reshape(-1))
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        if lit_model.preprocess_fn is not None:
            batch = lit_model.preprocess_fn(batch)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = lit_model(batch)
        logits.append(out.float().cpu().numpy().reshape(-1))
    return np.concatenate(logits), np.concatenate(labels)


def classification_report(logits: np.ndarray, labels: np.ndarray) -> dict:
    """BCE, accuracy, AUROC and the confusion counts at threshold 0.

    The splits are exactly class-balanced by construction, so 0.5 accuracy and 0.5 AUROC
    are exactly chance, and a BCE of ln 2 = 0.6931 is what predicting the base rate scores.
    """
    from sklearn.metrics import roc_auc_score

    p = 1.0 / (1.0 + np.exp(-logits))
    eps = 1e-7
    bce = float(-(labels * np.log(p + eps) + (1 - labels) * np.log(1 - p + eps)).mean())
    pred = (logits > 0).astype(np.float32)
    n_classes = len(np.unique(labels))
    return {
        "n": int(labels.size),
        "bce": bce,
        "accuracy": float((pred == labels).mean()),
        "auroc": float(roc_auc_score(labels, logits)) if n_classes == 2 else float("nan"),
        "positive_rate": float(labels.mean()),
        "tp": int(((pred == 1) & (labels == 1)).sum()),
        "fp": int(((pred == 1) & (labels == 0)).sum()),
        "tn": int(((pred == 0) & (labels == 0)).sum()),
        "fn": int(((pred == 0) & (labels == 1)).sum()),
    }


# plot_run() used to be defined here. It moved to experiments/wave_common.py, unchanged, so
# that 1_baseline_template_diego.ipynb can call the same implementation: this module's name
# starts with a digit, which makes it unimportable, so the notebook had no way to reach it
# and would otherwise have needed a second copy of the same figure code.
plot_run = wc.plot_run


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def apply_cli_overrides(cfg: TrainingConfig, args: argparse.Namespace) -> None:
    """Fold the per-run CLI flags into the loaded config, in one place."""
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
    if args.lr is not None:
        cfg.learning_rate = args.lr
    if args.s3_cache_dir is not None:
        cfg.data.s3_cache_dir = args.s3_cache_dir
    if args.deterministic is not None:
        cfg.deterministic = _DETERMINISTIC_CLI[args.deterministic]
    if args.lora_r is not None:
        cfg.model.lora_config.r = args.lora_r
    if args.penultimate is not None:
        cfg.model.penultimate_linear_layer = args.penultimate


def main() -> None:
    args = parse_args()
    torch.set_float32_matmul_precision("medium")

    cfg = load_wave_config(args.config)
    L.seed_everything(cfg.seed, workers=True)
    apply_cli_overrides(cfg, args)
    run_name = args.run_name or cfg.job_id

    print(f"[RUN] {run_name}: lr={cfg.learning_rate} batch={cfg.batch_size} "
          f"accum={args.accum} (effective {cfg.batch_size * args.accum}) "
          f"workers={cfg.num_workers} prefetch={args.prefetch_factor} "
          f"lora_r={cfg.model.lora_config.r} "
          f"penultimate={cfg.model.penultimate_linear_layer} max_time={args.max_time}")

    ensure_assets(cfg, which=["scalers"] if args.train_baseline else ["scalers", "weights"])
    scalers = build_scalers(info=cfg.data.scalers_path)

    train_loader, val_loader, test_loader, data_stats = build_datasets(cfg, scalers, args)
    lit_model = build_model(cfg, scalers, train_baseline=args.train_baseline)

    # The shape assertion needs a real batch. Read it through a throwaway single-process
    # loader rather than `next(iter(val_loader))`: taking one batch from the real loader
    # would spawn its 8 persistent workers and fill their prefetch queues, which at 1.74 GB
    # a sample is ~28 GB of RAM committed before training has started. One sample from the
    # local cache costs a second.
    wc.assert_batch_shape(
        next(iter(wc.build_wave_loader(
            val_loader.dataset, batch_size=1, num_workers=0, prefetch_factor=None,
            seed=cfg.seed, shuffle=False, drop_last=False,
        ))),
        n_channels=len(cfg.data.channels),
        n_timesteps=cfg.model.time_embedding.time_dim,
        img_size=cfg.model.img_size,
    )

    trainer, checkpoint_cb, guard, csv_logger = build_trainer(cfg, args, run_name)
    summary = {
        "run_name": run_name,
        "config": str(args.config),
        "learning_rate": cfg.learning_rate,
        "batch_size": cfg.batch_size,
        "accum": args.accum,
        "effective_batch": cfg.batch_size * args.accum,
        "max_epochs": trainer.max_epochs,
        "max_steps": args.max_steps,
        "max_time": args.max_time,
        "lora_r": cfg.model.lora_config.r,
        "penultimate_linear_layer": cfg.model.penultimate_linear_layer,
        "use_lora": cfg.model.use_lora,
        "train_baseline": args.train_baseline,
        "subset_seed": args.subset_seed,
        "trainable_params": sum(p.numel() for p in lit_model.parameters() if p.requires_grad),
        "data": data_stats,
    }

    evals: dict = {}
    try:
        trainer.fit(lit_model, train_loader, val_loader)

        best = checkpoint_cb.best_model_path
        summary["best_checkpoint"] = best or None
        summary["best_val_loss"] = (
            float(checkpoint_cb.best_model_score)
            if checkpoint_cb.best_model_score is not None else None
        )
        summary["epochs_completed"] = trainer.current_epoch
        summary["global_step"] = trainer.global_step
        summary["samples_seen"] = trainer.global_step * cfg.batch_size * args.accum
        summary["peak_host_gb"] = guard.peak_host_gb
        if torch.cuda.is_available():
            summary["peak_cuda_reserved_gb"] = torch.cuda.max_memory_reserved() / 2**30
        print(f"[CKPT] best: {best or 'none saved'}  val_loss={summary['best_val_loss']}")

        if best and not args.no_eval_test:
            # Restore the best weights before the held-out pass: at the end of fit the
            # module holds the LAST epoch's weights, which on a set this small is often
            # well past the point val_loss started rising.
            state = torch.load(best, weights_only=False, map_location="cpu")["state_dict"]
            lit_model.load_state_dict(state)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            for split, loader in [("val", val_loader), ("test", test_loader)]:
                logits, labels = collect_logits(lit_model, loader, device)
                report = classification_report(logits, labels)
                evals[split] = {"logits": logits, "labels": labels, "report": report}
                summary[f"{split}_best"] = report
                print(f"[EVAL] {split}: " + " ".join(f"{k}={v:.4f}" if isinstance(v, float)
                                                     else f"{k}={v}" for k, v in report.items()))

        figures = plot_run(run_name, Path(csv_logger.log_dir), evals, RESULTS_DIR)
        summary["figures"] = [str(f) for f in figures]

        if not args.no_wandb:
            import wandb
            if wandb.run is not None:
                wandb.run.summary.update(
                    {k: v for k, v in summary.items() if not isinstance(v, (dict, list))}
                )
                for split in evals:
                    wandb.run.summary.update(
                        {f"{split}_best_{k}": v for k, v in evals[split]["report"].items()}
                    )
                wandb.log({Path(f).stem: wandb.Image(str(f)) for f in figures})
    except BaseException as exc:
        # Recorded, then re-raised. The schedule continues past a failed run, and without
        # this the run would leave no summary at all — indistinguishable, later, from a run
        # that was never started.
        summary["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        # The summary is written on every path, including a crash or a wall-clock cap, so
        # summarize_results.py sees what each run actually achieved rather than only the
        # ones that finished cleanly.
        try:
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            out_json = RESULTS_DIR / f"{run_name}_summary.json"
            out_json.write_text(json.dumps(summary, indent=2, default=float))
            print(f"[OUT] {out_json}")
        except Exception as exc:
            print(f"[OUT] failed to write the run summary: {exc}")
        # Without finishing the run, the next one in the schedule reuses this process's
        # wandb run and the two sets of curves land on top of each other.
        if not args.no_wandb:
            try:
                import wandb
                if wandb.run is not None:
                    wandb.finish()
            except Exception as exc:  # a failed teardown must not mask a training error
                print(f"[WANDB] finish failed: {exc}")


if __name__ == "__main__":
    main()
