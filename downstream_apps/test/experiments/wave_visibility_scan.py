#!/usr/bin/env python3
"""
Phase 2 (extended) — is +30 min after onset a good place to sample a wave?

The catalog puts every positive sample at ``start_time + 30 min`` and every negative at
``start_time - 30 min`` (std 0.0 across all 914 events, verified). With input frames at
``[-12, 0]`` the positive sample's running difference therefore spans **+18 to +30 min**
after onset. EUV waves are brightest early and fade as they expand and cool, so that
window may sit in the faint tail of the visibility period — in which case the reason a
classifier cannot generalize is the *label timing*, not the sample count, and no amount of
extra data or GPU time fixes it.

This script measures it. For each of a few events it extracts the AIA193 running
difference at onset offsets +12, +24, +36 and +48 min and reports the contrast of the
signal against that event's own quiet-Sun floor, taken from the negative sample's running
difference at ``-30 min``. Because the control comes from the same event an hour earlier,
the comparison is not confounded by active-region brightness, limb position or solar cycle.

Three contrast statistics are reported, all computed on the solar disk only:

* ``p999`` — the 99.9th percentile of \\|RD\\|. A wave is a thin bright arc covering a
  small fraction of the disk, so a high quantile is what tracks it; the mean does not.
* ``std``  — standard deviation of RD. Broader, less sensitive to a single bright pixel.
* ``pooled_max`` — max \\|RD\\| of the 32x32-mean-pooled 128x128 map. This is the one that
  matters for the baseline, because it is literally a feature the logistic model sees.

Each is divided by the same statistic on the control window to give an SNR, so a value of
1.0 means "indistinguishable from the quiet Sun an hour earlier".

Frames are taken from the **train** split's index only, so the verdict is not read off
held-out data.

Usage:
    python -m downstream_apps.test.experiments.wave_visibility_scan --n-events 6
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from downstream_apps.test.configs import load_wave_config
from downstream_apps.test.experiments import wave_common as wc
from workshop_infrastructure.assets import ensure_assets
from workshop_infrastructure.utils import build_scalers

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "config_wave_full.yaml"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RD_CHANNEL = "aia193"

# Onset offsets to probe, in minutes. The catalog's positive sample sits at +30, so +24 and
# +36 bracket it and +12 tests whether the signal peaks earlier.
DEFAULT_OFFSETS = (12, 24, 36, 48)
# The control window: the negative sample's own running difference, 30 min BEFORE onset.
CONTROL_OFFSET = -30
# How far a requested onset offset may be moved to reach a frame that actually exists.
# Catalog start_times are not on the Surya grid — only 8.3% of them have a minute divisible by
# 12, while every index timestamp does — so a requested "+24 min" has to snap to the nearest
# available frame. Half a cadence bounds the error at 6 min, and the achieved offset is
# recorded and plotted rather than the requested one.
SNAP_TOLERANCE = pd.Timedelta(minutes=6)
# Analysis radius as a fraction of the frame width. Measured on a real AIA193 frame rather
# than taken from the plate scale: the radial median brightness rises to a peak of 2.49x the
# disk median at r = 1600 px (frac 0.391) — that is limb brightening — and falls away beyond
# it, so the limb sits at frac ~0.39.
#
# 0.36 stops short of it on purpose. A static limb cancels in a running difference, but the
# brightness gradient there is so steep that sub-pixel pointing jitter leaves a bright ring in
# the difference (visible in the task 1.1 figures), and that ring would dominate the 99.9th
# percentile this script uses to track the wave front. Excluding it keeps 85% of the disk area,
# which is far more than a wave detection needs.
DISK_FRACTION = 0.36


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--n-events", type=int, default=6)
    p.add_argument("--offsets", default=",".join(str(o) for o in DEFAULT_OFFSETS))
    p.add_argument("--subset-seed", type=int, default=42,
                   help="Draws the events from the same seeded pool the training runs use.")
    p.add_argument("--replot", action="store_true",
                   help="Recompute the verdict and redraw the figure from the "
                        "existing wave_visibility.csv, reading no frames.")
    return p.parse_args()


def nearest_frame(stamps: np.ndarray, target: pd.Timestamp) -> "pd.Timestamp | None":
    """Nearest timestamp in ``stamps`` to ``target``, or None if none is within tolerance.

    ``stamps`` must be sorted. Needed because the catalog's event times are not aligned to the
    12-minute cadence the Surya index is built on, so "onset + 24 min" is almost never itself
    an available frame.
    """
    target64 = np.datetime64(target)
    i = int(np.searchsorted(stamps, target64))
    candidates = [j for j in (i - 1, i) if 0 <= j < len(stamps)]
    if not candidates:
        return None
    best = min(candidates, key=lambda j: abs(stamps[j] - target64))
    if abs(stamps[best] - target64) > np.timedelta64(SNAP_TOLERANCE):
        return None
    return pd.Timestamp(stamps[best])


def resolve_pair(stamps: np.ndarray, onset: pd.Timestamp, offset_min: int,
                 step: pd.Timedelta) -> "tuple[pd.Timestamp, pd.Timestamp, float] | None":
    """Resolve one running-difference pair for ``onset + offset_min``.

    Returns ``(previous_frame, now_frame, achieved_offset_minutes)``, or None if either frame is
    unavailable. The "previous" frame is taken one cadence step before the *snapped* now frame,
    so the difference always spans exactly one step — which is what the model sees.
    """
    now = nearest_frame(stamps, onset + pd.Timedelta(minutes=offset_min))
    if now is None:
        return None
    prev = nearest_frame(stamps, now - step)
    if prev is None or prev == now:
        return None
    achieved = (now - onset).total_seconds() / 60.0
    return prev, now, achieved


def disk_mask(side: int) -> np.ndarray:
    """Boolean mask of the solar disk for a ``side x side`` frame."""
    y, x = np.ogrid[:side, :side]
    c = (side - 1) / 2.0
    return ((y - c) ** 2 + (x - c) ** 2) <= (DISK_FRACTION * side) ** 2


def contrast_stats(rd: np.ndarray, mask: np.ndarray) -> dict:
    """Contrast statistics for one running-difference map, on-disk only."""
    on_disk = rd[mask]
    pooled = wc.repool(np.where(mask, rd, 0.0)[None].astype(np.float32), 32)[0]
    return {
        "p999": float(np.percentile(np.abs(on_disk), 99.9)),
        "std": float(on_disk.std()),
        "pooled_max": float(np.abs(pooled).max()),
    }


def load_signum_log_frame(dataset, timestamp) -> np.ndarray:
    """Return AIA193 at ``timestamp`` in **signum-log** space, shape ``(H, W)``.

    Signum-log rather than physical units on purpose: it is the space the baseline's
    features live in and the space the running difference is taken in during training, so a
    contrast measured here is a contrast the model could actually use. ``transform_data``
    applies signum-log *and* the z-score, so the scaler's ``inverse_transform`` undoes only
    the second stage — see the THREE SPACES block in
    ``workshop_infrastructure/datasets/helio.py``.
    """
    normalized = dataset.transform_data(
        dataset.load_nc_data(dataset.index.loc[timestamp, "path"], timestamp, [RD_CHANNEL])
    )
    return dataset.scalers[RD_CHANNEL].inverse_transform(normalized[0])


def main() -> None:
    args = parse_args()
    offsets = [int(o) for o in args.offsets.split(",")]

    if args.replot:
        csv = RESULTS_DIR / "wave_visibility.csv"
        if not csv.is_file():
            raise SystemExit(f"--replot needs {csv}, which does not exist yet.")
        print(f"[scan] redrawing from {csv} — no frames read")
        report(pd.read_csv(csv), offsets)
        return

    cfg = load_wave_config(args.config)
    cfg.data.channels = [RD_CHANNEL]
    ensure_assets(cfg, which=["scalers"])
    scalers = build_scalers(info=cfg.data.scalers_path)

    train_ds, _ = wc.build_wave_datasets(cfg, scalers)
    info = wc.event_stratified_subset(train_ds, n_samples=None, seed=args.subset_seed)
    print(f"[scan] train pool: {info.describe()}")

    meta = train_ds.df_valid_indices.reset_index()
    stamps = np.sort(np.asarray(train_ds.index.index.values, dtype="datetime64[ns]"))

    # An event is usable only if every pair the scan needs resolves to real frames — one per
    # probed offset plus the control — after snapping to the cadence grid.
    step = pd.Timedelta(minutes=abs(cfg.data.time_delta_input_minutes[0]))
    wanted = list(offsets) + [CONTROL_OFFSET]
    usable = {}
    for event in meta[wc.EVENT_COL].drop_duplicates().sort_values():
        onset = pd.Timestamp(event)
        resolved = {o: resolve_pair(stamps, onset, o, step) for o in wanted}
        if all(v is not None for v in resolved.values()):
            usable[event] = resolved
        if len(usable) >= args.n_events:
            break
    if not usable:
        raise SystemExit(
            f"No event has all {len(wanted)} running-difference pairs available within "
            f"{SNAP_TOLERANCE} of the requested offsets. Widen SNAP_TOLERANCE or drop an offset."
        )
    if len(usable) < args.n_events:
        print(f"[scan] only {len(usable)} of {args.n_events} requested events have all "
              f"{len(wanted)} pairs available; continuing with those")
    print(f"[scan] events: {list(usable)}")

    mask = disk_mask(cfg.model.img_size)
    rows = []
    for event, resolved in usable.items():
        print(f"[scan] {event}", flush=True)
        c_prev, c_now, c_achieved = resolved[CONTROL_OFFSET]
        control = contrast_stats(
            load_signum_log_frame(train_ds, c_now) - load_signum_log_frame(train_ds, c_prev),
            mask,
        )
        for offset in offsets:
            prev, now, achieved = resolved[offset]
            signal = contrast_stats(
                load_signum_log_frame(train_ds, now) - load_signum_log_frame(train_ds, prev),
                mask,
            )
            row = {"event": str(event), "offset_min": offset,
                   "achieved_offset_min": achieved, "control_achieved_min": c_achieved}
            for key in signal:
                row[key] = signal[key]
                row[f"{key}_control"] = control[key]
                row[f"{key}_snr"] = signal[key] / control[key] if control[key] else float("nan")
            rows.append(row)
            print(f"        +{offset:3d} min (actual {achieved:+6.1f})  "
                  f"p999 snr {row['p999_snr']:5.2f}  std snr {row['std_snr']:5.2f}  "
                  f"pooled snr {row['pooled_max_snr']:5.2f}", flush=True)

    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / "wave_visibility.csv", index=False)
    report(df, offsets)


def report(df: pd.DataFrame, offsets: list[int]) -> None:
    """Summarize the scan, write the verdict, and draw the figure."""
    summary = df.groupby("offset_min")[["p999_snr", "std_snr", "pooled_max_snr"]].agg(
        ["median", "mean", "std"])
    print("\n[scan] SNR vs onset offset (1.0 = indistinguishable from the same event "
          "one hour earlier)")
    print(summary.round(3).to_string())

    # Everything below is on the ACHIEVED offset axis. Snapping to the 12-minute cadence moves
    # a requested offset by up to 6 min, so the requested value is only a grouping label —
    # interpolating the +30 verdict on it would read the curve at the wrong place.
    med = df.groupby("offset_min")["pooled_max_snr"].median()
    x = df.groupby("offset_min")["achieved_offset_min"].median()
    best_offset = float(x.loc[med.idxmax()])
    at_30 = float(np.interp(30, x.to_numpy(dtype=float), med.to_numpy()))
    verdict = {
        "offsets": offsets,
        "achieved_offsets_median": {int(k): float(v) for k, v in x.items()},
        "control_offset_min": CONTROL_OFFSET,
        "n_events": int(df["event"].nunique()),
        "events": sorted(df["event"].unique().tolist()),
        "median_pooled_snr_by_offset": {int(k): float(v) for k, v in med.items()},
        "best_offset_min": best_offset,
        "best_median_pooled_snr": float(med.max()),
        "median_pooled_snr_at_catalog_offset_30": at_30,
        "decay_from_best_to_30_pct": float(100 * (1 - at_30 / med.max())) if med.max() else None,
        "verdict": (
            f"Signal peaks at +{best_offset:.0f} min (median pooled SNR "
            f"{med.max():.2f}); at the catalog's +30 min it is {at_30:.2f}, i.e. "
            f"{100 * (1 - at_30 / med.max()):.0f}% lower. "
            + ("+30 min is a POOR sampling point — the label timing, not the sample count, "
               "is the primary limit. Emit a re-timed catalog and re-run."
               if med.max() > 0 and at_30 < 0.7 * med.max() else
               "+30 min is an ACCEPTABLE sampling point — the signal has not substantially "
               "decayed by then, so sample count remains the binding constraint.")
        ),
    }
    (RESULTS_DIR / "wave_visibility.json").write_text(json.dumps(verdict, indent=2))
    print(f"\n[scan] {verdict['verdict']}")
    print(f"[out] {RESULTS_DIR / 'wave_visibility.json'}")

    plot(df, med, x, best_offset, at_30)


def plot(df: pd.DataFrame, med: pd.Series, x: pd.Series,
         best_offset: float, at_30: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    for ax, key, title in zip(
        axes,
        ["p999_snr", "std_snr", "pooled_max_snr"],
        ["99.9th pct of |RD|", "std of RD", "max |RD| of the 128x128 pooled map"],
    ):
        for event, g in df.groupby("event"):
            ax.plot(g["achieved_offset_min"], g[key], "-o", ms=3, lw=1, alpha=0.45,
                    color="0.4", label="_")
        m = df.groupby("offset_min")[key].median()
        # Median plotted at the median ACHIEVED offset, so it lies on the same axis as the
        # per-event traces above rather than up to 6 min away from them.
        ax.plot(x.reindex(m.index).to_numpy(), m.to_numpy(), "-o", lw=2.5, color="C0",
                label="median")
        ax.axhline(1.0, ls=":", c="C3", label="quiet Sun (same event, -30 min)")
        ax.axvline(30, ls="--", c="C1", label="catalog positive (+30 min)")
        ax.set(xlabel="minutes after onset", ylabel="SNR vs control", title=title)
        ax.legend(fontsize=7)
    fig.suptitle("Running-difference contrast vs where the positive sample is taken "
                 f"(peak +{best_offset:.0f} min, +30 min at {at_30:.2f})")
    out = RESULTS_DIR / "wave_visibility_scan.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[out] {out}")


if __name__ == "__main__":
    main()
