#!/usr/bin/env python3
"""
Task 1.1 — can a wave be seen at all, in the representation the baseline is given?

For each of a few matched wave / no-wave pairs, plots three panels per sample:

1. **AIA193 at t**, the "now" frame the model receives.
2. **The full-resolution running difference**, ``AIA193(t) - AIA193(t - 12 min)``.
3. **The 32x32-mean-pooled 128x128 map**, which is what
   ``RunningDifferenceLogisticModel`` actually consumes. Everything the baseline can
   possibly use is in this panel; if the wave is not visible here, the baseline cannot see
   it either.

The comparison is controlled: the negative sample is the *same event* one hour earlier
(``start_time - 30 min`` against ``start_time + 30 min``), so the two rows differ in wave
activity and almost nothing else — same active region, same limb position, same epoch of
the solar cycle.

Frame ordering is verified rather than assumed. ``HelioNetCDFDataset`` sorts
``time_delta_input_minutes`` ascending and always places the offset-0 "now" frame last, so
``x[:, -1] - x[:, 0]`` is genuinely now-minus-12-minutes and the plotted difference carries
the same sign as the feature the baseline trains on.

Reads the arrays ``extract_baseline_features.py`` wrote; run that first.

Usage:
    python -m downstream_apps.test.experiments.plot_task_1_1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent / "results"
ARRAYS_DIR = RESULTS_DIR / "figure_arrays"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arrays-dir", default=str(ARRAYS_DIR))
    p.add_argument("--out", default=str(RESULTS_DIR / "task_1_1_wave_visibility.png"))
    p.add_argument("--rd-percentile", type=float, default=99.5,
                   help="Symmetric colour limit for the difference panels, as a percentile "
                        "of |RD| over the pair. Shared within a pair so the two rows are "
                        "directly comparable.")
    return p.parse_args()


def load_pairs(arrays_dir: Path) -> list[tuple[str, dict, dict]]:
    """Group the saved per-sample arrays into (event, wave, no_wave) triples."""
    pairs = {}
    for path in sorted(arrays_dir.glob("*.npz")):
        stem = path.stem
        # Order matters: "_wave" is a suffix of "_no_wave", so test the longer one first.
        for suffix, label in (("_no_wave", "no_wave"), ("_wave", "wave")):
            if stem.endswith(suffix):
                event_key = stem[: -len(suffix)]
                pairs.setdefault(event_key, {})[label] = dict(
                    np.load(path, allow_pickle=False))
                break
    complete = [(k, v["wave"], v["no_wave"]) for k, v in sorted(pairs.items())
                if "wave" in v and "no_wave" in v]
    if not complete:
        raise SystemExit(
            f"No complete wave/no-wave pairs found in {arrays_dir}. Run "
            f"extract_baseline_features.py with --figure-events >= 1 first."
        )
    return complete


def main() -> None:
    args = parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # SunPy registers the instrument colour maps ("sdoaia193", ...) on import. Worth having:
    # the AIA193 map is what a solar physicist reads without having to think about it. The
    # frames are in signum-log space rather than DN, so the colour bar is labelled as such.
    try:
        import sunpy.visualization.colormaps  # noqa: F401
    except Exception:
        pass
    intensity_cmap = "sdoaia193" if "sdoaia193" in plt.colormaps() else "inferno"

    pairs = load_pairs(Path(args.arrays_dir))
    print(f"[1.1] {len(pairs)} matched pairs: {[p[0] for p in pairs]}")

    n_rows = 2 * len(pairs)
    fig, axes = plt.subplots(n_rows, 3, figsize=(11.5, 3.9 * n_rows),
                             constrained_layout=True)
    axes = np.atleast_2d(axes)

    for i, (event, wave, no_wave) in enumerate(pairs):
        # One colour scale per pair, from both samples, so "brighter" means brighter and
        # not "differently normalized".
        both = np.concatenate([np.abs(wave["running_diff"]).ravel(),
                               np.abs(no_wave["running_diff"]).ravel()])
        rd_lim = float(np.percentile(both, args.rd_percentile))
        pool_lim = float(np.percentile(
            np.abs(np.concatenate([wave["pooled_32"].ravel(), no_wave["pooled_32"].ravel()])),
            99.9))

        for j, (label, sample) in enumerate([("wave", wave), ("no wave", no_wave)]):
            row = 2 * i + j
            now = sample["now"].astype(np.float32)
            vmin, vmax = np.percentile(now, [1, 99.5])

            im = axes[row, 0].imshow(now, origin="lower", cmap=intensity_cmap,
                                     vmin=vmin, vmax=vmax)
            axes[row, 0].set_title(f"{label} — AIA193 at t\n{str(sample['ds_index'])}",
                                   fontsize=9)
            fig.colorbar(im, ax=axes[row, 0], fraction=0.046, label="signum-log DN")

            im = axes[row, 1].imshow(sample["running_diff"].astype(np.float32),
                                     origin="lower", cmap="RdBu_r",
                                     vmin=-rd_lim, vmax=rd_lim)
            axes[row, 1].set_title(f"{label} — running difference\n"
                                   f"AIA193(t) − AIA193(t−12 min), 4096²", fontsize=9)
            fig.colorbar(im, ax=axes[row, 1], fraction=0.046)

            im = axes[row, 2].imshow(sample["pooled_32"], origin="lower", cmap="RdBu_r",
                                     vmin=-pool_lim, vmax=pool_lim, interpolation="nearest")
            axes[row, 2].set_title(f"{label} — 32× mean-pooled, 128²\n"
                                   f"(what the baseline sees)", fontsize=9)
            fig.colorbar(im, ax=axes[row, 2], fraction=0.046)

            for ax in axes[row]:
                ax.set_xticks([])
                ax.set_yticks([])
        axes[2 * i, 0].set_ylabel(f"event {event}", fontsize=9)

    fig.suptitle("Task 1.1 — matched wave / no-wave pairs of the same event, one hour apart.\n"
                 "Each pair shares its colour limits; the right column is the only "
                 "information the running-difference baseline receives.", fontsize=11)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"[out] {out}")

    # A number to go with the picture: how much bigger is the pooled signal in the
    # positive sample than in its own matched negative?
    print("\n[1.1] pooled-map contrast, wave vs its matched no-wave control")
    for event, wave, no_wave in pairs:
        w = float(np.abs(wave["pooled_32"]).max())
        n = float(np.abs(no_wave["pooled_32"]).max())
        print(f"  {event}: max|pooled| wave {w:.4f}  no wave {n:.4f}  ratio {w / n:.2f}")


if __name__ == "__main__":
    main()
