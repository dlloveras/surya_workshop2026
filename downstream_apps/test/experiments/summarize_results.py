#!/usr/bin/env python3
"""
Collect every run's results into the study's headline figure and the answer to task 2.4.

Reads the ``*_summary.json`` files ``4_finetune_wave_1D.py`` writes and the
``baseline_study.json`` from Phase 3, and produces:

* **val loss vs N**, with the running-difference baseline's learning curve on the same axes
  and the base-rate line (ln 2) marked. This is the primary result: it separates "Surya
  helps" from "more data helps" from "neither helps at this scale".
* **the iso-compute slice** — validation loss against *samples seen* rather than against
  training-set size, read off the per-epoch CSV logs. A larger N looks better at fixed
  epochs partly because it took more gradient steps; this panel removes that confound
  without any extra runs.
* **a written answer to 2.4** in ``task_2_4_answer.md``, stating what the curves show,
  including the ablations and the held-out test numbers.

Usage:
    python -m downstream_apps.test.experiments.summarize_results
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import re
from pathlib import Path

import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent / "results"
CHANCE_BCE = float(math.log(2))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", default=str(RESULTS_DIR))
    p.add_argument("--runs-dir", default="runs",
                   help="Where the CSVLogger wrote per-epoch metrics.")
    return p.parse_args()


# Runs that are not points on the scaling curve. Probes deliberately stop after 60 steps, and
# the smoke test is one epoch on 8 samples; plotting either alongside the real runs would put a
# point on the curve that was never trying to fit anything.
_EXCLUDED_PREFIXES = ("probe_", "smoke_")


def load_summaries(results_dir: Path) -> list[dict]:
    """Every finished run's summary JSON, excluding probes and smoke tests."""
    out = []
    for path in sorted(results_dir.glob("*_summary.json")):
        if path.name.startswith(_EXCLUDED_PREFIXES):
            continue
        try:
            out.append(json.loads(path.read_text()))
        except json.JSONDecodeError:
            print(f"[warn] {path.name} is not valid JSON; skipping")
    return out


def epoch_curve(runs_dir: Path, run_name: str) -> "np.ndarray | None":
    """Per-epoch ``(samples_seen, val_loss)`` for one run, from its CSV log.

    Takes the *newest* version directory: a re-run of the same name appends a new one, and
    the last one is the one whose summary JSON is on disk.
    """
    import pandas as pd

    versions = sorted(glob.glob(str(runs_dir / run_name / "version_*")),
                      key=lambda p: int(re.search(r"version_(\d+)", p).group(1)))
    for version in reversed(versions):
        csv = Path(version) / "metrics.csv"
        if not csv.is_file():
            continue
        df = pd.read_csv(csv)
        if "val_loss" not in df or "epoch" not in df:
            continue
        per_epoch = df.dropna(subset=["val_loss"]).groupby("epoch")["val_loss"].mean()
        if per_epoch.empty:
            continue
        return per_epoch
    return None


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    runs_dir = Path(args.runs_dir)
    summaries = load_summaries(results_dir)
    if not summaries:
        print(f"[summarize] no run summaries in {results_dir}; nothing to do")
        return

    scaling, ablations = [], []
    for s in summaries:
        name = s.get("run_name", "")
        entry = {
            "run_name": name,
            "n_train": s.get("data", {}).get("train", {}).get("n_samples"),
            "best_val_loss": s.get("best_val_loss"),
            "epochs_completed": s.get("epochs_completed"),
            "samples_seen": s.get("samples_seen"),
            "lora_r": s.get("lora_r"),
            "penultimate": s.get("penultimate_linear_layer"),
            "lr": s.get("learning_rate"),
            "val_best": s.get("val_best"),
            "test_best": s.get("test_best"),
            "trainable_params": s.get("trainable_params"),
        }
        (ablations if name.startswith("surya_E") else scaling).append(entry)
    scaling = [e for e in scaling if e["n_train"] and e["best_val_loss"] is not None]
    scaling.sort(key=lambda e: e["n_train"])

    baseline_path = results_dir / "baseline_study.json"
    baseline = json.loads(baseline_path.read_text()) if baseline_path.is_file() else None

    plot(scaling, baseline, runs_dir, results_dir)
    write_answer(scaling, ablations, baseline, results_dir)


def plot(scaling, baseline, runs_dir: Path, results_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    ax = axes[0]
    if scaling:
        ax.plot([e["n_train"] for e in scaling], [e["best_val_loss"] for e in scaling],
                "-o", lw=2, label="Surya + LoRA (best val loss)")
        for e in scaling:
            if e.get("test_best"):
                ax.plot(e["n_train"], e["test_best"]["bce"], "x", ms=9, c="C0",
                        label="_" if e is not scaling[0] else "Surya, held-out test")
    if baseline and baseline.get("learning_curve"):
        curve = baseline["learning_curve"]
        ns = [c["n_train"] for c in curve]
        ax.plot(ns, [c["best_val_loss_median"] for c in curve], "-s", lw=2, c="C2",
                label="running-difference baseline (median of seeds)")
        ax.fill_between(ns, [c["best_val_loss_min"] for c in curve],
                        [c["best_val_loss_max"] for c in curve], alpha=0.18, color="C2")
    ax.axhline(CHANCE_BCE, ls=":", c="0.4", label="base rate (ln 2 = 0.693)")
    ax.set(xscale="log", xlabel="training samples N", ylabel="best validation BCE",
           title="Does validation loss improve with more images?")
    if scaling or baseline:
        ticks = sorted({e["n_train"] for e in scaling} |
                       ({c["n_train"] for c in baseline["learning_curve"]} if baseline else set()))
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticks, fontsize=8)
    ax.legend(fontsize=8)

    # Iso-compute: val loss against samples seen, so a run with a bigger N is not credited
    # for the extra gradient steps its fixed epoch count bought it.
    ax = axes[1]
    for e in scaling:
        per_epoch = epoch_curve(runs_dir, e["run_name"])
        if per_epoch is None:
            continue
        # samples_seen accumulates batch_size * accum per optimizer step; per epoch it is
        # simply the training-set size, so epoch index * N is exact and does not depend on
        # how the run was chunked.
        seen = (per_epoch.index.to_numpy() + 1) * e["n_train"]
        ax.plot(seen, per_epoch.to_numpy(), "-o", ms=3, label=f"N={e['n_train']}")
    ax.axhline(CHANCE_BCE, ls=":", c="0.4", label="base rate (ln 2)")
    ax.set(xscale="log", xlabel="training samples seen (epochs × N)",
           ylabel="validation BCE", title="Iso-compute view: same axes, equal samples seen")
    ax.legend(fontsize=8)

    out = results_dir / "val_loss_vs_n.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[out] {out}")


def write_answer(scaling, ablations, baseline, results_dir: Path) -> None:
    """Write the task 2.4 answer as markdown, stating only what the numbers support."""
    lines = ["# Task 2.4 — does validation loss improve as more images are used?", ""]

    if not scaling:
        lines += ["No Surya run produced a validation loss, so this question is unanswered. "
                  "Check `experiments/results/logs/` for why each run failed.", ""]
    else:
        lines += [f"Base rate for a balanced binary task is `ln 2 = {CHANCE_BCE:.4f}`. "
                  "Any loss at or above that is a model that has learned nothing about "
                  "the input.", "",
                  "## Surya + LoRA", "",
                  "| N (train) | epochs | best val BCE | val acc | val AUROC | "
                  "test BCE | test acc | test AUROC |",
                  "|---|---|---|---|---|---|---|---|"]
        for e in scaling:
            v, t = e.get("val_best") or {}, e.get("test_best") or {}
            lines.append(
                f"| {e['n_train']} | {e['epochs_completed']} | {e['best_val_loss']:.4f} | "
                f"{v.get('accuracy', float('nan')):.3f} | {v.get('auroc', float('nan')):.3f} | "
                f"{t.get('bce', float('nan')):.4f} | {t.get('accuracy', float('nan')):.3f} | "
                f"{t.get('auroc', float('nan')):.3f} |")
        lines.append("")

        first, last = scaling[0], scaling[-1]
        delta = last["best_val_loss"] - first["best_val_loss"]
        improved = delta < 0
        beat_chance = [e for e in scaling if e["best_val_loss"] < CHANCE_BCE]
        lines += [
            "## Answer", "",
            f"Going from N={first['n_train']} to N={last['n_train']}, best validation BCE "
            f"{'fell' if improved else 'rose'} by {abs(delta):.4f} "
            f"({first['best_val_loss']:.4f} → {last['best_val_loss']:.4f}), so on this "
            f"evidence more images **{'do' if improved else 'do not'}** improve validation "
            f"loss over the range tested.", "",
            f"{len(beat_chance)} of {len(scaling)} runs beat the base rate. "
            + ("At least one configuration therefore extracts real signal from the input."
               if beat_chance else
               "No configuration beat the base rate, so nothing here has yet demonstrated "
               "that the input is predictive at this sample size — the ranking of the runs "
               "is not meaningful until one of them does."), "",
            "**Caveat that limits every number above:** the validation split is 24 samples, "
            "so one sample is 4.2% of the metric and differences smaller than roughly 0.05 "
            "in BCE are not resolvable. The 48-sample test column is the more trustworthy "
            "comparison, and it was looked at once, after checkpoint selection.", "",
        ]

    if baseline and baseline.get("learning_curve"):
        curve = baseline["learning_curve"]
        b_first, b_last = curve[0], curve[-1]
        lines += [
            "## Running-difference baseline, same axes", "",
            f"| N | best val BCE (median of {len(b_first['seeds'])} seeds) | val AUROC |",
            "|---|---|---|",
        ]
        for c in curve:
            lines.append(f"| {c['n_train']} | {c['best_val_loss_median']:.4f} | "
                         f"{c['best_val_auroc_median']:.3f} |")
        lines += ["",
                  f"The baseline goes {b_first['best_val_loss_median']:.4f} → "
                  f"{b_last['best_val_loss_median']:.4f} over the same range. "
                  f"`ConstantProbabilityModel`, which learns only the base rate, scores "
                  f"{baseline['constant_probability']['val_loss']:.4f}.", ""]

    if ablations:
        lines += ["## Ablations", "",
                  "| run | LoRA r | penultimate head | trainable | best val BCE |",
                  "|---|---|---|---|---|"]
        for e in ablations:
            loss = (f"{e['best_val_loss']:.4f}" if e["best_val_loss"] is not None
                    else "did not finish")
            params = f"{e['trainable_params']:,}" if e["trainable_params"] else "?"
            lines.append(f"| {e['run_name']} | {e['lora_r']} | {e['penultimate']} | "
                         f"{params} | {loss} |")
        lines.append("")

    lines += ["## Figures", "",
              "- `val_loss_vs_n.png` — the headline curve plus the iso-compute view",
              "- `baseline_study.png` — task 1.2, the baseline learning curve and sweeps",
              "- `task_1_1_wave_visibility.png` — task 1.1, matched pairs",
              "- `wave_visibility_scan.png` — contrast vs where the positive is sampled",
              "- `<run>_curves.png`, `<run>_roc.png` — per-run training curves and ROC", ""]

    out = results_dir / "task_2_4_answer.md"
    out.write_text("\n".join(lines))
    print(f"[out] {out}")
    print("\n".join(lines[:24]))


if __name__ == "__main__":
    main()
