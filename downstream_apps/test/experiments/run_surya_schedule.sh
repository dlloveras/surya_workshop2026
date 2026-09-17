#!/usr/bin/env bash
# Phases 4-5 — the unattended Surya schedule.
#
# Runs sequentially, each run under a hard wall-clock cap, and CONTINUES ON FAILURE: a run
# that OOMs, diverges or hits its cap must not take the rest of the night with it. Runs are
# ordered by increasing training-set size so an overrun still leaves the smaller runs
# complete, which is what makes the val-loss-vs-N curve readable even if the schedule is
# cut short.
#
# Phase 4 (the LR probe) exists because a previous run at lr=0.01 diverged outright — losses
# of 10^2-10^4, exact 0.0 values under bf16, val accuracy and AUROC pinned at 0.5. That LR
# had been tuned on the 16 K-parameter logistic baseline. The probe keeps the largest LR
# whose loss decreases without saturating, and Phase 5 uses it.
#
# Batch size is 2 with --accum 4 rather than the largest batch that fits: one 65,536-token
# sample already saturates the GPU, so s/sample is flat-to-worse in batch size (3.05 at B=1,
# 3.11 at B=2, 3.59 at B=3). --prefetch-factor 1 with 8 workers bounds worker RSS at
# 8 * 1 * 2 * 1.74 GB = 27.8 GB; the PyTorch default of 2 would double it.
#
# --num-workers 8 is a MEASURED optimum, not a CPU-count heuristic. Each worker fetches its
# frames serially, so num_workers is the number of whole-frame S3 reads in flight, and that
# has a sharp knee:
#     4 workers  352 MB/s     8 workers  380 MB/s     12 workers  166 MB/s
# At 12 the rate more than halves while a separate process on the same machine still reads at
# 424 MB/s, so it is a concurrency knee rather than a bandwidth ceiling. At 8 workers the data
# path costs 2.97 s/sample against 3.11 s/sample of GPU, so the GPU is (just) the limiter.
# Do not raise this to "use the other 8 CPUs".
#
# Usage:
#   bash downstream_apps/test/experiments/run_surya_schedule.sh            # everything
#   PHASES=probe bash .../run_surya_schedule.sh                            # LR probe only
#   PHASES=scaling LR=1e-4 bash .../run_surya_schedule.sh                  # skip the probe
#   SMOKE=1 bash .../run_surya_schedule.sh                                 # 8-sample smoke test

set -uo pipefail   # deliberately NOT -e: a failing run is expected and handled below

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT" || exit 1

PY="${PY:-/home/jovyan/envs/surya_WS/bin/python}"
ENTRY="downstream_apps.test.4_finetune_wave_1D"
RESULTS="downstream_apps/test/experiments/results"
LOGS="$RESULTS/logs"
mkdir -p "$LOGS"

# 8 workers on 16 CPUs: leave each worker two threads rather than letting every one of them
# spin up 16 and thrash. This matters here because signum-log normalization is the dominant
# CPU cost per sample.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PHASES="${PHASES:-probe,scaling,ablations}"
SMOKE="${SMOKE:-0}"
LR="${LR:-}"
SUBSET_SEED="${SUBSET_SEED:-42}"
COMMON="--num-workers 8 --prefetch-factor 1 --subset-seed $SUBSET_SEED"

log()  { printf '\n=== [%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

# Run one configuration. Never returns non-zero: the schedule must survive any single run.
run() {
  local name="$1"; shift
  local logfile="$LOGS/${name}.log"
  log "START $name  ->  $logfile"
  printf '    %s -u -m %s --run-name %s %s %s\n' "$PY" "$ENTRY" "$name" "$COMMON" "$*"
  local t0=$SECONDS
  "$PY" -u -m "$ENTRY" --run-name "$name" $COMMON "$@" > "$logfile" 2>&1
  local rc=$?
  local mins=$(( (SECONDS - t0) / 60 ))
  if [ $rc -eq 0 ]; then
    log "DONE  $name in ${mins}m"
    grep -E '^\[CKPT\]|^\[EVAL\]|^\[OUT\]' "$logfile" | sed 's/^/    /'
  else
    log "FAIL  $name (exit $rc) after ${mins}m — continuing with the schedule"
    tail -n 25 "$logfile" | sed 's/^/    | /'
  fi
  # Free the GPU before the next run starts, and give any orphaned spawn workers a moment
  # to exit so they are not holding 1.74 GB buffers when the next run allocates.
  sleep 20
  return 0
}

# --------------------------------------------------------------------------
# Smoke test — the last gate before the unattended stretch
# --------------------------------------------------------------------------
if [ "$SMOKE" = "1" ]; then
  run "smoke_n8" --train-n 8 --max-epochs 1 --accum 1 --lr "${LR:-1e-4}" --max-time 00:30:00
  log "smoke complete"
  exit 0
fi

# --------------------------------------------------------------------------
# Phase 4 — LR probe
# --------------------------------------------------------------------------
if [[ ",$PHASES," == *,probe,* ]]; then
  for lr in 3e-5 1e-4 3e-4; do
    # accum 1 so 60 optimizer steps really is 60 batches: the point is to watch the first
    # ~120 samples, not to reach a good model.
    # --train-n 48 so the probe reads the frames pinned to local disk instead of pulling a
    # fresh set: three probes at 120 samples each would otherwise cost ~25 min of S3 for
    # data that is thrown away.
    run "probe_lr${lr}" --lr "$lr" --max-steps 60 --accum 1 --train-n 48 \
        --max-epochs 50 --max-time 00:25:00 --no-eval-test --no-checkpoint
  done
  log "LR probe complete — inspect $LOGS/probe_lr*.log and $RESULTS/probe_lr*_summary.json"
  # The chooser writes its pick to a file as well as printing the table, so a single
  # unattended invocation carries the probe's answer into the scaling phase instead of
  # silently falling back to the config default.
  PROBE_PICK="$RESULTS/probe_recommended_lr.txt"
  rm -f "$PROBE_PICK"
  PROBE_PICK="$PROBE_PICK" "$PY" - <<'PYPROBE'
import json, glob, math, os
rows = []
for path in sorted(glob.glob("downstream_apps/test/experiments/results/probe_lr*_summary.json")):
    d = json.loads(open(path).read())
    lr = d.get("learning_rate")
    vl = d.get("best_val_loss")
    rows.append((lr, vl, d.get("global_step"), path))
print("\n  lr        best_val_loss   steps")
for lr, vl, steps, _ in sorted(rows):
    flag = ""
    if vl is None or not math.isfinite(vl):
        flag = "  <-- no finite val loss"
    elif vl > math.log(2):
        flag = "  <-- no better than the base rate (ln 2 = 0.6931)"
    print(f"  {lr:<9} {vl!s:<15} {steps!s:<6}{flag}")
ok = [(lr, vl) for lr, vl, _, _ in rows if vl is not None and math.isfinite(vl) and vl < math.log(2)]
if ok:
    # Largest LR that still improves on the base rate: with only 10-30 epochs per run, the
    # binding constraint is how far the model gets, so the fastest stable rate wins.
    best = max(ok, key=lambda t: t[0])
    print(f"\n  RECOMMENDED LR: {best[0]}  (val_loss {best[1]:.4f})")
    pick = os.environ.get("PROBE_PICK")
    if pick:
        with open(pick, "w") as fh:
            fh.write(str(best[0]))
        print(f"  written to {pick} — the scaling phase picks it up automatically")
else:
    print("\n  No LR beat the base rate in 60 steps. The scaling phase will fall back to the "
          "config default; consider widening the grid downward (1e-5, 3e-6) first.")
PYPROBE
fi

if [ -z "$LR" ] && [ -s "$RESULTS/probe_recommended_lr.txt" ]; then
  LR="$(cat "$RESULTS/probe_recommended_lr.txt")"
  log "using the LR probe's pick: $LR"
fi
if [ -z "$LR" ]; then
  # Reached when the probe was skipped, or when no LR beat the base rate in 60 steps. 1e-4 is
  # the config default and the midpoint of the probe grid.
  LR=1e-4
  log "no probe recommendation available — defaulting the scaling phase to $LR"
fi

# --------------------------------------------------------------------------
# Phase 5 — scaling study, increasing N
# --------------------------------------------------------------------------
if [[ ",$PHASES," == *,scaling,* ]]; then
  #      name          train-n  epochs  max_time
  # Epochs fall as N rises so every run fits the night: at ~3.3 s/sample the estimates are
  # 1.7 h, 2.4 h, 2.8 h and 4.4 h, and each cap is set ~10% above its estimate. A run that
  # hits its cap stops cleanly with its best checkpoint intact, and summarize_results.py
  # plots validation loss against samples SEEN as well as against N, so unequal epoch counts
  # do not confound the comparison.
  #
  # If the night is shorter than expected, run the ablations before surya_D rather than after
  # it: they are the cheapest runs here (N=48, locally cached) and D is both the most
  # expensive and the least load-bearing point on the curve.
  #     PHASES=scaling ... ; then PHASES=ablations ...
  #      name          train-n  epochs  max_time
  set -- "surya_A_n48   48  30  01:50:00" \
         "surya_B_n96   96  24  02:35:00" \
         "surya_C_n192 192  16  03:10:00" \
         "surya_D_n384 384  12  04:40:00"
  for spec in "$@"; do
    read -r name n epochs cap <<< "$spec"
    run "$name" --train-n "$n" --max-epochs "$epochs" --max-time "$cap" \
        --accum 4 --lr "$LR"
  done
  log "scaling study complete"
fi

# --------------------------------------------------------------------------
# Phase 5E — ablations at the largest N that finished
# --------------------------------------------------------------------------
if [[ ",$PHASES," == *,ablations,* ]]; then
  # N=48 rather than the largest N, for two reasons. Both ablations have to fit in whatever
  # the scaling study leaves, and N=48's frames are the ones pinned to local disk (the event
  # subsets are nested, so the smallest subset's frames are cached for every run) — which
  # makes these the cheapest runs available and keeps them GPU-bound. A capacity question
  # (r=4 vs 8, redundant head layer on vs off) is also sharpest where data is scarcest.
  # Raise ABL_N if the schedule finishes early.
  ABL_N="${ABL_N:-48}"
  ABL_EPOCHS="${ABL_EPOCHS:-24}"
  run "surya_E_r4_n${ABL_N}" --train-n "$ABL_N" --max-epochs "$ABL_EPOCHS" \
      --max-time 01:25:00 --accum 4 --lr "$LR" --lora-r 4
  run "surya_E_pen_n${ABL_N}" --train-n "$ABL_N" --max-epochs "$ABL_EPOCHS" \
      --max-time 01:25:00 --accum 4 --lr "$LR" --penultimate
  log "ablations complete"
fi

log "schedule finished"
"$PY" -m downstream_apps.test.experiments.summarize_results || true
