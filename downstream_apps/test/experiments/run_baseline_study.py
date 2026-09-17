#!/usr/bin/env python3
"""
Phase 3 — the baseline arm: task 1.2, a learning curve, and two hyperparameter sweeps.

Everything here runs on the cached AIA193 running-difference features that
``extract_baseline_features.py`` wrote, so a fit that costs ~3 s of GPU per sample when
trained through the live dataset costs milliseconds here. That is what turns "does
validation loss improve with more images?" from an assertion into a measured curve.

The model is exactly ``RunningDifferenceLogisticModel``: a 32x32 mean pool of
``AIA193(now) - AIA193(now - 12 min)`` in signum-log space, flattened into one linear layer,
trained with BCE-with-logits. The cache is stored at pool kernel 8 and re-pooled here,
which is numerically identical (mean pooling composes over equal-sized blocks) and is what
makes the pooling sweep free.

Four experiments:

1. **1.2 reproduction.** ``runs/wave_classification_baseline_running_diff/version_3``
   trained 16,385 parameters on 10 samples at lr=0.01: train loss 0.726 -> 0.0065, train
   accuracy 1.0 from epoch 1, validation loss rising monotonically 0.760 -> 1.006. The tell
   that this is memorization rather than miscalibration is that validation AUROC froze at
   exactly 0.4 from epoch 0 to 13 — the *ranking* of the validation samples never changed,
   so the model only rescaled a direction fixed in the first few steps. Reproduced here
   with the same 10 chronologically-earliest samples, which is what
   ``max_samples: 10`` actually selects.
2. **Learning curve** over N, event-level stratified and nested, several seeds per N.
3. **Weight-decay sweep** at the largest N.
4. **Pooling-kernel sweep** at the largest N.

Usage:
    python -m downstream_apps.test.experiments.run_baseline_study
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from downstream_apps.test.experiments import wave_common as wc

RESULTS_DIR = Path(__file__).resolve().parent / "results"
BASE_POOL = 32          # the kernel RunningDifferenceLogisticModel uses
CHANCE_BCE = float(np.log(2))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", default=str(RESULTS_DIR / "rd_features.npz"))
    p.add_argument("--epochs", type=int, default=500,
                   help="Epochs per fit. Full-batch gradient descent on <=384 samples, so "
                        "these are cheap; the curve is read off the best epoch, not the last.")
    p.add_argument("--lr", type=float, default=0.01,
                   help="Matches the config's diagnostic value, which is what version_3 used.")
    p.add_argument("--seeds", default="0,1,2,3,4",
                   help="Subset draws per N. Reported as median with spread.")
    p.add_argument("--curve-n", default="8,16,32,64,128,256,384")
    p.add_argument("--weight-decays", default="0,1e-4,1e-3,1e-2,1e-1,1")
    p.add_argument("--pool-kernels", default="8,16,32,64,128")
    p.add_argument("--repro-n", type=int, default=10,
                   help="Fallback size when the cache has no 'repro' split.")
    p.add_argument("--out", default=str(RESULTS_DIR / "baseline_study.json"))
    return p.parse_args()


# ---------------------------------------------------------------------------
# The model, and one fit
# ---------------------------------------------------------------------------

def fit_logistic(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int,
    lr: float,
    weight_decay: float = 0.0,
    seed: int = 0,
    batch_size: int | None = None,
) -> dict:
    """Fit the linear readout and return its per-epoch history plus the best-epoch metrics.

    ``batch_size=None`` means full-batch gradient descent, which is the right default for the
    learning curve: on <=384 samples it optimizes the same objective with less gradient noise,
    so a change in the curve is a property of the data rather than of the batch schedule.
    Pass an explicit ``batch_size`` to reproduce a specific Lightning run, whose minibatch
    schedule is part of what produced its behaviour.
    """
    from sklearn.metrics import roc_auc_score

    torch.manual_seed(seed)
    n_features = X_train.shape[1] * X_train.shape[2]
    model = torch.nn.Linear(n_features, 1)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    xt = torch.from_numpy(X_train.reshape(len(X_train), -1))
    yt = torch.from_numpy(y_train).unsqueeze(1)
    xv = torch.from_numpy(X_val.reshape(len(X_val), -1))
    yv = torch.from_numpy(y_val).unsqueeze(1)

    generator = torch.Generator().manual_seed(seed)
    history = []
    for epoch in range(epochs):
        model.train()
        if batch_size is None:
            opt.zero_grad()
            out = model(xt)
            loss = loss_fn(out, yt)
            loss.backward()
            opt.step()
        else:
            order = torch.randperm(len(xt), generator=generator)
            for start in range(0, len(order), batch_size):
                sel = order[start: start + batch_size]
                opt.zero_grad()
                loss = loss_fn(model(xt[sel]), yt[sel])
                loss.backward()
                opt.step()
            # Report the epoch's training loss over the whole set, so it is comparable with
            # the validation loss below rather than being the last minibatch's value.
            with torch.no_grad():
                out = model(xt)
                loss = loss_fn(out, yt)
        with torch.no_grad():
            model.eval()
            vout = model(xv)
            vloss = loss_fn(vout, yv)
            history.append({
                "epoch": epoch,
                "train_loss": float(loss),
                "train_accuracy": float(((out > 0).float() == yt).float().mean()),
                "val_loss": float(vloss),
                "val_accuracy": float(((vout > 0).float() == yv).float().mean()),
                "val_auroc": float(roc_auc_score(y_val, vout.numpy().ravel())),
            })

    best = min(history, key=lambda h: h["val_loss"])
    return {
        "history": history,
        "best_epoch": best["epoch"],
        "best_val_loss": best["val_loss"],
        "best_val_accuracy": best["val_accuracy"],
        "best_val_auroc": best["val_auroc"],
        "final_train_loss": history[-1]["train_loss"],
        "final_val_loss": history[-1]["val_loss"],
        "n_train": int(len(y_train)),
        "n_params": n_features + 1,
    }


def constant_probability_baseline(y_train: np.ndarray, y_val: np.ndarray) -> dict:
    """``ConstantProbabilityModel``, solved in closed form.

    A single learned logit independent of the input, so the optimum is the training base
    rate and the validation loss follows immediately. This is the number any
    input-dependent model has to beat; on a balanced split it is ln 2 = 0.6931.
    """
    rate = float(np.clip(y_train.mean(), 1e-6, 1 - 1e-6))
    p = np.full_like(y_val, rate)
    bce = float(-(y_val * np.log(p) + (1 - y_val) * np.log(1 - p)).mean())
    return {"train_base_rate": rate, "val_loss": bce,
            "val_accuracy": float(max(y_val.mean(), 1 - y_val.mean()))}


# ---------------------------------------------------------------------------
# Subset selection over the cached features
# ---------------------------------------------------------------------------

def event_subset_indices(events: np.ndarray, labels: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Row indices for a seeded, event-level stratified subset of the cached features.

    The same rule ``wave_common.event_stratified_subset()`` applies to a live dataset, so a
    baseline fit at N and a Surya run at N see the same events.
    """
    import pandas as pd

    per_event = pd.Series(labels).groupby(events).nunique()
    complete = np.sort(per_event.index[per_event == 2].to_numpy())
    n_events = n // 2
    if n_events > len(complete):
        raise ValueError(f"n={n} needs {n_events} events, only {len(complete)} available")
    order = np.random.default_rng(seed).permutation(len(complete))
    chosen = complete[np.sort(order[:n_events])]
    return np.flatnonzero(np.isin(events, chosen))


def main() -> None:
    args = parse_args()
    data = np.load(args.features, allow_pickle=False)
    base_kernel = int(data["pool_kernel"])
    factor = BASE_POOL // base_kernel
    if BASE_POOL % base_kernel:
        raise SystemExit(f"cached pool kernel {base_kernel} does not divide {BASE_POOL}")

    def features(split: str, kernel: int = BASE_POOL) -> np.ndarray:
        return wc.repool(data[f"X_{split}"], kernel // base_kernel)

    Xtr, ytr = features("train"), data["y_train"]
    Xva, yva = features("val"), data["y_val"]
    events_tr = data["event_train"]
    print(f"[data] train {Xtr.shape} val {Xva.shape} "
          f"(cached at kernel {base_kernel}, re-pooled by {factor} to {BASE_POOL})")
    print(f"[data] class balance train {ytr.mean():.3f} val {yva.mean():.3f}")

    results = {
        "lr": args.lr,
        "chance_bce": CHANCE_BCE,
        "constant_probability": constant_probability_baseline(ytr, yva),
        "n_train_available": int(len(ytr)),
        "pool_kernel_cached": base_kernel,
    }
    print(f"[base] ConstantProbabilityModel val_loss "
          f"{results['constant_probability']['val_loss']:.4f} (chance {CHANCE_BCE:.4f})")

    # --- 1. reproduce version_3 ------------------------------------------------
    results["reproduction"] = reproduce_overfit(data, Xtr, ytr, Xva, yva, args, base_kernel)

    # --- 2. learning curve ----------------------------------------------------
    seeds = [int(s) for s in args.seeds.split(",")]
    curve = []
    for n in [int(v) for v in args.curve_n.split(",")]:
        if n > len(ytr):
            print(f"[curve] skipping N={n}: only {len(ytr)} cached training samples")
            continue
        per_seed = []
        for seed in seeds:
            idx = event_subset_indices(events_tr, ytr, n, seed)
            fit = fit_logistic(Xtr[idx], ytr[idx], Xva, yva,
                               epochs=args.epochs, lr=args.lr, seed=seed)
            per_seed.append(fit)
        vl = [f["best_val_loss"] for f in per_seed]
        au = [f["best_val_auroc"] for f in per_seed]
        entry = {
            "n_train": n,
            "best_val_loss_median": float(np.median(vl)),
            "best_val_loss_min": float(np.min(vl)),
            "best_val_loss_max": float(np.max(vl)),
            "best_val_auroc_median": float(np.median(au)),
            "final_train_loss_median": float(np.median([f["final_train_loss"] for f in per_seed])),
            "seeds": seeds,
            "per_seed_val_loss": [float(v) for v in vl],
            "per_seed_val_auroc": [float(a) for a in au],
        }
        curve.append(entry)
        print(f"[curve] N={n:4d}  val_loss median {entry['best_val_loss_median']:.4f} "
              f"[{entry['best_val_loss_min']:.4f}, {entry['best_val_loss_max']:.4f}]  "
              f"AUROC {entry['best_val_auroc_median']:.3f}  "
              f"train_loss {entry['final_train_loss_median']:.4f}")
    results["learning_curve"] = curve

    n_max = max(e["n_train"] for e in curve)

    # --- 3. weight decay ------------------------------------------------------
    wd_rows = []
    for wd in [float(v) for v in args.weight_decays.split(",")]:
        vl, au = [], []
        for seed in seeds:
            idx = event_subset_indices(events_tr, ytr, n_max, seed)
            fit = fit_logistic(Xtr[idx], ytr[idx], Xva, yva, epochs=args.epochs,
                               lr=args.lr, weight_decay=wd, seed=seed)
            vl.append(fit["best_val_loss"])
            au.append(fit["best_val_auroc"])
        wd_rows.append({"weight_decay": wd, "n_train": n_max,
                        "best_val_loss_median": float(np.median(vl)),
                        "best_val_auroc_median": float(np.median(au))})
        print(f"[wd]    {wd:<8g} val_loss {wd_rows[-1]['best_val_loss_median']:.4f}  "
              f"AUROC {wd_rows[-1]['best_val_auroc_median']:.3f}")
    results["weight_decay_sweep"] = wd_rows

    # --- 4. pooling kernel ----------------------------------------------------
    pool_rows = []
    for kernel in [int(v) for v in args.pool_kernels.split(",")]:
        if kernel % base_kernel or (data["X_train"].shape[1] * base_kernel) % kernel:
            print(f"[pool]  skipping kernel {kernel}: not reachable from the cached {base_kernel}")
            continue
        Xk, Xvk = features("train", kernel), features("val", kernel)
        vl, au = [], []
        for seed in seeds:
            idx = event_subset_indices(events_tr, ytr, n_max, seed)
            fit = fit_logistic(Xk[idx], ytr[idx], Xvk, yva, epochs=args.epochs,
                               lr=args.lr, seed=seed)
            vl.append(fit["best_val_loss"])
            au.append(fit["best_val_auroc"])
        pool_rows.append({"pool_kernel": kernel, "side": int(Xk.shape[1]),
                          "n_features": int(Xk.shape[1] ** 2), "n_train": n_max,
                          "best_val_loss_median": float(np.median(vl)),
                          "best_val_auroc_median": float(np.median(au))})
        print(f"[pool]  kernel {kernel:4d} -> {Xk.shape[1]:3d}x{Xk.shape[1]:<3d} "
              f"val_loss {pool_rows[-1]['best_val_loss_median']:.4f}  "
              f"AUROC {pool_rows[-1]['best_val_auroc_median']:.3f}")
    results["pooling_sweep"] = pool_rows

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"[out] {out_path}")
    # Figure next to the JSON, not in a fixed directory: a --out into /tmp for a dry run
    # should not overwrite the real figure.
    plot(results, out_path.with_name(out_path.stem + ".png"))


def reproduce_overfit(data, Xtr, ytr, Xva, yva, args, base_kernel: int) -> dict:
    """Refit on the 10 samples ``max_samples: 10`` selects, and test for the memorization signature.

    Uses the ``repro`` pseudo-split when the feature cache has one: those are the
    chronologically first 10 samples of the **unsubsetted** train match, which is exactly what
    ``waveDSDataset`` returns under ``max_samples: 10`` (it head-slices a frame sorted by
    ``ds_index``, so the cap yields the 5 earliest event pairs rather than a sample). Falling
    back to the earliest 10 of the event-level subset would be reproducing a different run:
    those are a later, sparser set of events.

    What still differs from ``version_3``, unavoidably: validation here is this study's
    24-sample balanced split, not the 27-sample one the looser 4-day match tolerance produced.
    So the exact numbers cannot match. What is being tested is whether the *signature*
    reappears — training loss collapsing while validation loss rises and validation AUROC stays
    frozen, which together say the model memorized rather than merely miscalibrated.
    """
    if "X_repro" in data.files:
        Xr = wc.repool(data["X_repro"], BASE_POOL // base_kernel)
        yr = data["y_repro"]
        events = data["event_repro"]
        source = "repro split (earliest samples of the unsubsetted match)"
    else:
        order = np.argsort(data["index_train"])[: args.repro_n]
        Xr, yr, events = Xtr[order], ytr[order], data["event_train"][order]
        source = ("earliest samples of the event-level subset — NOT what max_samples selects; "
                  "re-run the extractor with --splits repro for a faithful reproduction")
    print(f"[repro] source: {source}")
    fit = fit_logistic(Xr, yr, Xva, yva, epochs=15, lr=args.lr, seed=42, batch_size=2)
    hist = fit["history"]
    aurocs = [h["val_auroc"] for h in hist]
    train_fell = hist[-1]["train_loss"] < 0.1 * hist[0]["train_loss"]
    val_rose = hist[-1]["val_loss"] > hist[0]["val_loss"]
    # Two versions of "frozen", because they answer different questions. version_3's AUROC was
    # identical from epoch 0 to 13; the substantive claim it supported is weaker — that the
    # model stopped changing its RANKING of the validation samples early and thereafter only
    # rescaled a direction already fixed. "Frozen from epoch 0" is the strict reading;
    # "frozen over the tail" is the claim, and a different validation split can satisfy the
    # second without the first.
    auroc_frozen = float(np.std(aurocs)) < 1e-6
    tail = aurocs[len(aurocs) // 3:]
    auroc_frozen_tail = float(np.std(tail)) < 1e-6
    out = {
        "n_train": int(len(yr)),
        "source": source,
        "class_balance": float(yr.mean()),
        "dates": sorted({str(e)[:10] for e in events}),
        "train_loss_first": hist[0]["train_loss"],
        "train_loss_last": hist[-1]["train_loss"],
        "train_accuracy_last": hist[-1]["train_accuracy"],
        "val_loss_first": hist[0]["val_loss"],
        "val_loss_last": hist[-1]["val_loss"],
        "val_auroc_values": aurocs,
        "val_auroc_unique": sorted({round(a, 9) for a in aurocs}),
        "val_auroc_below_chance_throughout": bool(max(aurocs) < 0.5),
        "train_loss_collapsed": bool(train_fell),
        "val_loss_rose": bool(val_rose),
        "val_auroc_frozen": bool(auroc_frozen),
        "val_auroc_frozen_over_tail": bool(auroc_frozen_tail),
        "val_auroc_tail_spread": float(max(tail) - min(tail)),
        # The overfit itself: training loss collapses while validation loss rises. This is the
        # part that has to reproduce for the version_3 diagnosis to stand.
        "overfit_reproduced": bool(train_fell and val_rose),
        "signature_matches_version_3": bool(train_fell and val_rose and auroc_frozen),
        "history": hist,
    }
    print(f"[repro] {out['n_train']} samples from {out['dates'][0]}..{out['dates'][-1]}: "
          f"train {out['train_loss_first']:.3f}->{out['train_loss_last']:.4f}, "
          f"val {out['val_loss_first']:.3f}->{out['val_loss_last']:.3f}, "
          f"AUROC unique values {out['val_auroc_unique']}")
    print(f"[repro] overfit reproduced (train collapses, val rises): "
          f"{out['overfit_reproduced']}")
    print(f"[repro] val AUROC below chance throughout: "
          f"{out['val_auroc_below_chance_throughout']}; frozen from epoch 0: "
          f"{out['val_auroc_frozen']}; frozen over the tail: "
          f"{out['val_auroc_frozen_over_tail']} (tail spread "
          f"{out['val_auroc_tail_spread']:.4f})")
    return out


def plot(results: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True)

    # (a) the reproduction
    hist = results["reproduction"]["history"]
    ep = [h["epoch"] for h in hist]
    ax = axes[0, 0]
    ax.plot(ep, [h["train_loss"] for h in hist], "-o", ms=3, label="train_loss")
    ax.plot(ep, [h["val_loss"] for h in hist], "-s", ms=3, label="val_loss")
    ax.axhline(CHANCE_BCE, ls=":", c="0.5", label="base rate (ln 2)")
    ax2 = ax.twinx()
    ax2.plot(ep, [h["val_auroc"] for h in hist], "-^", ms=3, c="C3", label="val_auroc")
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("val AUROC", color="C3")
    # Title states what was measured, not what was expected: the three conditions are the
    # memorization signature being tested for, and any of them can come out false.
    repro = results["reproduction"]
    marks = [("train collapses", repro["train_loss_collapsed"]),
             ("val rises", repro["val_loss_rose"]),
             ("AUROC below chance", repro["val_auroc_below_chance_throughout"]),
             ("AUROC frozen over tail", repro["val_auroc_frozen_over_tail"])]
    verdict = ", ".join(f"{name}: {'yes' if ok else 'NO'}" for name, ok in marks)
    ax.set(xlabel="epoch", ylabel="BCE",
           title=f"1.2 — {repro['n_train']} samples, lr={results.get('lr', 0.01)}\n{verdict}")
    ax.legend(fontsize=8, loc="center right")

    # (b) the learning curve
    curve = results["learning_curve"]
    ns = [e["n_train"] for e in curve]
    ax = axes[0, 1]
    ax.plot(ns, [e["best_val_loss_median"] for e in curve], "-o", label="best val_loss (median)")
    ax.fill_between(ns, [e["best_val_loss_min"] for e in curve],
                    [e["best_val_loss_max"] for e in curve], alpha=0.2)
    ax.plot(ns, [e["final_train_loss_median"] for e in curve], "-s", label="final train_loss")
    ax.axhline(CHANCE_BCE, ls=":", c="0.5", label="base rate (ln 2)")
    ax.set(xscale="log", xlabel="training samples N", ylabel="BCE",
           title="1.2 — does val loss improve with more images?")
    ax.set_xticks(ns)
    ax.set_xticklabels(ns)
    ax.legend(fontsize=8)

    # (c) weight decay
    wd = results["weight_decay_sweep"]
    ax = axes[1, 0]
    xs = [max(e["weight_decay"], 1e-5) for e in wd]
    ax.plot(xs, [e["best_val_loss_median"] for e in wd], "-o")
    ax.axhline(CHANCE_BCE, ls=":", c="0.5")
    ax.set(xscale="log", xlabel="weight decay (0 plotted at 1e-5)", ylabel="best val BCE",
           title=f"weight decay at N={wd[0]['n_train']}")

    # (d) pooling kernel
    pool = results["pooling_sweep"]
    ax = axes[1, 1]
    ax.plot([e["pool_kernel"] for e in pool],
            [e["best_val_loss_median"] for e in pool], "-o")
    ax.axhline(CHANCE_BCE, ls=":", c="0.5")
    ax.set(xscale="log", xlabel="mean-pool kernel", ylabel="best val BCE",
           title=f"pooling kernel at N={pool[0]['n_train']} (32 = the baseline)")
    ax.set_xticks([e["pool_kernel"] for e in pool])
    ax.set_xticklabels([f"{e['pool_kernel']}\n{e['side']}²" for e in pool])

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[out] {out}")


if __name__ == "__main__":
    main()
