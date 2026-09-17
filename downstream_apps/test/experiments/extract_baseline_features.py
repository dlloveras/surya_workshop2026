#!/usr/bin/env python3
"""
Phase 1 — one data pass that does three jobs.

Run this first. It is the only script here that touches S3 in bulk, and everything
downstream (the 1.1 figures, the whole baseline arm, and the Surya runs' warm cache)
depends on what it leaves behind.

The three jobs:

1. **Warm the S3 cache.** The full 13-channel NetCDF files are downloaded, because the
   Surya runs need all 13 channels later and a cached file costs nothing to re-open. Only
   ~2% of the frames this task needs were already cached.
2. **Cache the running-difference baseline's features.** Only AIA193 is *decoded* — the
   channel the baseline uses — which is what makes this pass cheap: 1.2 s of CPU per
   sample against 15.2 s for all 13 channels (0.40 s/frame to read one channel vs 4.69 s,
   plus 0.39 s of signum-log against 4.67 s). The result is a small ``.npz`` that makes
   every baseline fit in Phase 3 instant.
3. **Save full-resolution arrays for the task 1.1 figures**, for a few matched
   wave/no-wave pairs.

The cache is exactly equivalent to reading the live dataset: no augmentation is active on
this path, so there is no random draw to freeze. ``num_mask_aia_channels`` defaults to 0
and is never passed, ``drop_hmi_probability`` is 0.0, and ``random_vert_flip`` defaults to
False — so ``phase="train"`` and ``phase="val"`` differ only in which index they read.

Features are cached at ``--pool-kernel 8`` (512x512, 1 MB/sample) rather than at the
baseline's 32 (128x128, 64 KB/sample). Mean pooling composes exactly — pooling a
kernel-8 map by a further 4 is identical to pooling the original by 32 — so the larger
cache costs ~0.5 GB in total and buys the pooling-kernel sweep for free.

Usage:
    python -m downstream_apps.test.experiments.extract_baseline_features \
        --config downstream_apps/test/configs/config_wave_full.yaml \
        --train-n 384 --num-workers 12
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from downstream_apps.test.configs import load_wave_config
from downstream_apps.test.experiments import wave_common as wc
from downstream_apps.test.models.simple_baseline import destandardize_channels
from workshop_infrastructure.assets import ensure_assets
from workshop_infrastructure.utils import build_scalers

# The channel the running-difference baseline uses, and the only one decoded here.
RD_CHANNEL = "aia193"

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "config_wave_full.yaml"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--train-n", type=int, default=384,
                   help="Training samples to extract (even; event-level stratified).")
    p.add_argument("--subset-seed", type=int, default=42,
                   help="Selects which events. Must match the value the Surya runs use.")
    p.add_argument("--num-workers", type=int, default=12,
                   help="At one channel a sample is 0.134 GB, so 12 workers cost ~2 GB.")
    p.add_argument("--prefetch-factor", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--pool-kernel", type=int, default=8,
                   help="Base mean-pool kernel for the cached features. Must divide "
                        "img_size, and 32 // this must be an integer.")
    p.add_argument("--figure-events", type=int, default=3,
                   help="Matched pairs to also save at full resolution, for task 1.1.")
    p.add_argument("--out", default=str(RESULTS_DIR / "rd_features.npz"))
    p.add_argument("--splits", default="train,val,test",
                   help="Comma-separated. Add 'repro' for the pseudo-split that "
                        "reproduces what data.max_samples selects.")
    p.add_argument("--repro-samples", type=int, default=10,
                   help="Size of the 'repro' pseudo-split, matching the max_samples "
                        "value being reproduced.")
    p.add_argument("--force", action="store_true",
                   help="Re-extract a split even if its per-split .npz already exists.")
    p.add_argument("--disk-cache", action="store_true",
                   help="Use the stock s3_mode read path (writes every frame to "
                        "s3_cache_dir) instead of reading through RAM.")
    return p.parse_args()


def extract_split(name, dataset, args, scalers, figure_events, figures_dir):
    """Run one pass over ``dataset``, returning the cached feature arrays."""
    loader = wc.build_wave_loader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        seed=args.subset_seed,
        shuffle=False,       # order must match df_valid_indices so labels line up
        drop_last=False,     # every sample is wanted
    )

    meta = dataset.df_valid_indices
    events = meta[wc.EVENT_COL].to_numpy()
    expected_index = [pd.Timestamp(t).isoformat() for t in meta["ds_index"]]

    cached, total = wc.cache_coverage(dataset)
    print(f"[{name}] {len(dataset)} samples, {total} distinct frames, "
          f"{cached} already cached ({100 * cached / max(total, 1):.1f}%)")

    feats, labels, indices = [], [], []
    pos = 0
    t0 = time.time()
    for batch in loader:
        signum_log = destandardize_channels(batch, channel_order=[RD_CHANNEL], scalers=scalers)
        pooled = wc.pooled_running_difference(
            signum_log["ts"], channel_index=0, pool_kernel=args.pool_kernel
        )
        feats.append(pooled.numpy().astype(np.float32))
        labels.append(batch["forecast"].numpy().astype(np.float32))
        indices.extend(batch["ds_index"])

        # Full-resolution arrays for the task 1.1 figures. Written from this same pass so
        # the plotted difference is the one the baseline trains on, not a re-derivation.
        for i in range(pooled.shape[0]):
            if name == "train" and events[pos + i] in figure_events:
                save_figure_arrays(
                    figures_dir, signum_log["ts"][i], meta.iloc[pos + i], args.pool_kernel
                )
        pos += pooled.shape[0]

        done = pos
        rate = done / (time.time() - t0)
        print(f"[{name}] {done}/{len(dataset)} samples  {rate:.2f} samples/s  "
              f"eta {(len(dataset) - done) / max(rate, 1e-9) / 60:.1f} min", flush=True)

    X = np.concatenate(feats)
    y = np.concatenate(labels)
    if list(indices) != expected_index:
        raise AssertionError(
            f"[{name}] sample order drifted from df_valid_indices, so labels and features "
            f"may be mismatched. First divergence at "
            f"{next(i for i, (a, b) in enumerate(zip(indices, expected_index)) if a != b)}."
        )
    if not np.array_equal(y, meta["label"].to_numpy().astype(np.float32)):
        raise AssertionError(f"[{name}] collected labels differ from the dataset's labels.")
    print(f"[{name}] done in {(time.time() - t0) / 60:.1f} min — "
          f"X {X.shape} {X.nbytes / 2**20:.0f} MiB, y balance {y.mean():.3f}")
    return X, y, np.array(indices), events


def save_figure_arrays(figures_dir, ts_signum_log, row, pool_kernel):
    """Write one sample's full-resolution AIA193 frames, running difference and pooled map.

    float16 on purpose: these arrays exist to be plotted, and at 4096x4096 each float32
    frame is 67 MiB. The physics is unaffected at plotting precision, and the pooled map
    the baseline actually consumes is written from the float32 computation above.
    """
    frames = ts_signum_log.numpy()                     # (1, T, H, W), signum-log space
    prev, now = frames[0, 0], frames[0, -1]
    diff = (now - prev).astype(np.float32)
    pooled = wc.repool(diff[None], pool_kernel)[0]
    pooled32 = wc.repool(diff[None], 32)[0]
    label = "wave" if float(row["label"]) == 1.0 else "no_wave"
    stem = f"{pd.Timestamp(row[wc.EVENT_COL]).strftime('%Y%m%dT%H%M')}_{label}"
    out = figures_dir / f"{stem}.npz"
    np.savez(
        out,
        now=now.astype(np.float16),
        prev=prev.astype(np.float16),
        running_diff=diff.astype(np.float16),
        pooled_base=pooled.astype(np.float32),
        pooled_32=pooled32.astype(np.float32),
        ds_index=str(row["ds_index"]),
        event=str(row[wc.EVENT_COL]),
        label=float(row["label"]),
        pool_kernel=pool_kernel,
    )
    print(f"[fig] wrote {out.name}")


def main() -> None:
    args = parse_args()
    if 32 % args.pool_kernel:
        raise SystemExit(f"--pool-kernel {args.pool_kernel} must divide 32 so the cached "
                         f"map can be re-pooled to the baseline's 32.")

    cfg = load_wave_config(args.config)
    ensure_assets(cfg, which=["scalers"])
    scalers = build_scalers(info=cfg.data.scalers_path)

    # Decode one channel instead of thirteen. The .nc files are downloaded whole either
    # way (s3_mode="download" fetches the object, then xarray extracts variables), so the
    # Surya runs still get a warm cache while this pass pays 1.2 s/sample instead of 15.2.
    cfg.data.channels = [RD_CHANNEL]
    print(f"[cfg] decoding channels={cfg.data.channels}, "
          f"frames={cfg.data.time_delta_input_minutes}, "
          f"tolerance={cfg.data.ds_time_tolerance}")

    train, val, test = wc.build_wave_datasets(
        cfg, scalers, include_test=True, in_memory=not args.disk_cache
    )

    # "repro" is a fourth pseudo-split: the chronologically first N samples of the
    # UNSUBSETTED train match, which is exactly what data.max_samples selects (waveDSDataset
    # head-slices a frame sorted by ds_index). It is kept separate from the train split
    # because the event-level subset's earliest samples are a different, later set of events,
    # so reproducing a max_samples run from the subset would not be reproducing it at all.
    repro = None
    if "repro" in args.splits:
        import copy

        repro = copy.copy(train)
        repro.valid_indices = list(train.valid_indices)
        repro.df_valid_indices = train.df_valid_indices.iloc[: args.repro_samples]
        repro.valid_indices = repro.valid_indices[: args.repro_samples]
        repro.adjusted_length = len(repro.valid_indices)

    print(f"[cfg] read path: {'stock s3_mode=' + cfg.data.s3_mode if args.disk_cache else 'in-memory (BytesIO -> h5netcdf)'}")
    subsets = {}
    for name, ds, n in [("train", train, args.train_n), ("val", val, None), ("test", test, None)]:
        info = wc.event_stratified_subset(ds, n_samples=n, seed=args.subset_seed)
        subsets[name] = (ds, info)
        print(f"[{name}] {info.describe()}")
    if repro is not None:
        labels = repro.df_valid_indices["label"].to_numpy()
        dates = sorted({str(d)[:10] for d in repro.df_valid_indices[wc.EVENT_COL]})
        subsets["repro"] = (repro, None)
        print(f"[repro] {len(repro)} earliest samples of the unsubsetted match "
              f"({int(labels.sum())} wave, {int((1 - labels).sum())} no wave), "
              f"events {dates[0]}..{dates[-1]}")

    # This pass reads every frame exactly once, so nothing benefits from being pinned to
    # local disk during it. It is pinned anyway, for the *next* consumer: the Surya runs
    # re-read the validation split on every epoch, and warming it here costs nothing extra.
    n_pinned = wc.pin_frames(val)
    print(f"[cache] pinned {n_pinned} validation frames to {cfg.data.s3_cache_dir}")

    figures_dir = RESULTS_DIR / "figure_arrays"
    figures_dir.mkdir(parents=True, exist_ok=True)
    # Take the figure pairs from the *training* subset: task 1.1 is an illustration of
    # what the model is fitted on, and it keeps val/test unlooked-at.
    train_events = subsets["train"][0].df_valid_indices[wc.EVENT_COL].unique()
    figure_events = set(train_events[: args.figure_events])
    print(f"[fig] full-resolution arrays for {sorted(figure_events)}")

    # Each split is written as soon as it finishes, and reused if already on disk. This pass
    # is long enough, and shares bandwidth with enough else, that losing a completed split to
    # a restart is a real cost — and the splits are independent, so there is no reason to.
    out = {}
    out_path = Path(args.out)
    wanted = [s.strip() for s in args.splits.split(",") if s.strip()]
    for name in wanted:
        part_path = out_path.with_name(f"{out_path.stem}_{name}.npz")
        if part_path.exists() and not args.force:
            part = np.load(part_path, allow_pickle=False)
            out.update({k: part[k] for k in part.files})
            print(f"[{name}] reusing {part_path.name} "
                  f"({part[f'X_{name}'].shape[0]} samples) — pass --force to re-extract")
            continue
        ds, info = subsets[name]
        X, y, idx, events = extract_split(
            name, ds, args, scalers, figure_events if name == "train" else set(), figures_dir
        )
        part = {
            f"X_{name}": X,
            f"y_{name}": y,
            f"index_{name}": idx,
            f"event_{name}": events.astype(str),
        }
        part_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(part_path, **part)
        print(f"[{name}] wrote {part_path.name}")
        out.update(part)

    # The merged file is rebuilt from EVERY per-split file on disk, not only the splits this
    # invocation was asked for. Otherwise `--splits repro` would replace a merged cache
    # containing train/val/test with one containing only the 10-sample repro split — the
    # per-split files survive, but every consumer reads the merged one.
    for part_path in sorted(out_path.parent.glob(f"{out_path.stem}_*.npz")):
        part = np.load(part_path, allow_pickle=False)
        for key in part.files:
            out.setdefault(key, part[key])

    out["pool_kernel"] = np.array(args.pool_kernel)
    out["subset_seed"] = np.array(args.subset_seed)
    out["channel"] = np.array(RD_CHANNEL)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **out)
    splits_in = sorted(k[2:] for k in out if k.startswith("X_"))
    print(f"[out] wrote {out_path} ({out_path.stat().st_size / 2**20:.1f} MiB) "
          f"containing splits: {', '.join(splits_in)}")

    summary = {
        name: {
            "n_samples": int(info.n_samples),
            "n_events": int(info.n_events),
            "n_positive": int(info.n_positive),
            "n_negative": int(info.n_negative),
            "n_dropped_unpaired": int(info.n_dropped_unpaired),
            "n_events_available": int(info.n_events_available),
        }
        for name, (_, info) in subsets.items() if info is not None
    }
    summary["pool_kernel"] = args.pool_kernel
    summary["subset_seed"] = args.subset_seed
    summary["ds_time_tolerance"] = cfg.data.ds_time_tolerance
    (RESULTS_DIR / "dataset_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
