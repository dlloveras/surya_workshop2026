# Wave-classification experiments

Binary classification of EUV wave events: does a LoRA fine-tune of Surya beat a
running-difference logistic baseline, and does either improve as more images are used?

Everything reusable lives in `workshop_infrastructure/`; everything here is either a
driver, an analysis, or a measurement that the drivers depend on.

## Run order

```bash
cd <repo root>
PY=/home/jovyan/envs/surya_WS/bin/python      # the env with lightning + peft + wandb

# Phase 1 — one data pass: warm the cache, cache the baseline's features, save 1.1 arrays
OMP_NUM_THREADS=2 $PY -m downstream_apps.test.experiments.extract_baseline_features \
    --train-n 384 --num-workers 8

# Phase 2 — task 1.1 figures, and whether +30 min is a good place to sample a wave
$PY -m downstream_apps.test.experiments.plot_task_1_1
$PY -m downstream_apps.test.experiments.wave_visibility_scan --n-events 6

# Phase 3 — the baseline arm: 1.2 reproduction, learning curve, sweeps (seconds, CPU only)
$PY -m downstream_apps.test.experiments.run_baseline_study

# Smoke test, then Phases 4-5 — LR probe and the Surya scaling study
SMOKE=1 bash downstream_apps/test/experiments/run_surya_schedule.sh
bash downstream_apps/test/experiments/run_surya_schedule.sh          # probe, then read its table
PHASES=scaling,ablations LR=<probe's pick> bash .../run_surya_schedule.sh

# Collect everything into the headline figure and the written answer to 2.4
$PY -m downstream_apps.test.experiments.summarize_results
```

Phase 1 must run first: Phases 2 and 3 read what it writes, and it leaves the validation
split on fast local storage, which is what keeps the Surya runs GPU-bound.

## Files

| path | responsibility |
|---|---|
| `wave_common.py` | Shared helpers: event-level stratified subsetting, the pre-flight assertions, the running-difference feature, dataset/DataLoader construction, and the local-cache plan. Imported by everything else so none of it is duplicated. |
| `extract_baseline_features.py` | Phase 1. Restartable per split. |
| `plot_task_1_1.py` | Task 1.1 — matched wave/no-wave pairs at three levels of detail. |
| `wave_visibility_scan.py` | Task 1.1 extended — running-difference contrast against onset offset. |
| `run_baseline_study.py` | Phase 3 — task 1.2, the learning curve, weight-decay and pooling sweeps. |
| `run_surya_schedule.sh` | Phases 4-5 — sequential runs, per-run wall-clock cap, continue-on-failure. |
| `summarize_results.py` | The val-loss-vs-N figure, the iso-compute view, and `task_2_4_answer.md`. |
| `results/` | Figures, metrics JSON, per-split feature caches, and `logs/`. |

Also outside this directory, and part of the same work:
`../4_finetune_wave_1D.py` (the fine-tuning entry point),
`../datasets/wave_dataset_memory.py` (the in-memory read path),
`../configs/config_wave_full.yaml` (run-scale config).

## Four measurements that shaped all of the above

Every one of these overturned an assumption, so they are recorded rather than left in a
commit message.

**1. The pretrained tokenizer was silently discarded.** The checkpoint's patch embedding is
`Conv2d(26, 1280, 16, 16)` — 13 channels × **2** timesteps. The config's `time_dim: 1` built
a 13-channel one, the shapes disagreed, `load_pretrained_weights()` skipped the key, and it
stayed at its random initialization. Measured: `time_dim=1` loads 156/159 tensors,
`time_dim=2` loads 157/159 (the two missing ones are the pretraining decoder, which
`finetune=True` does not build). Every Surya result before this fix used a random tokenizer.
`wave_common.assert_tokenizer_loaded()` now blocks a run that regresses it.

**2. LoRA never adapted attention.** `target_modules: [q_proj, v_proj, k_proj, out_proj,
fc1, fc2]` — Surya has none of those names; q/k/v are fused into `attn.qkv` and the output
projection is `attn.proj`. PEFT raises only when *nothing* matches, so the run silently
trained `fc1`/`fc2` alone: 2,666,241 trainable parameters where 3,157,761 was intended.
`assert_lora_adapted_attention()` blocks it.

**3. The head was half redundant.** `head_linear` (1280→1280) feeds `head_unembed` (1280→1)
with no nonlinearity between them, so at eval the pair is exactly one linear map — 1,639,680
parameters, 52% of all trainable weights, contributing zero representational capacity.
`penultimate_linear_layer: false` is now the default (2,561-parameter head, 1,518,081
trainable in total) and the A/B is one of the ablations.

**4. The configured cache was 40× slower than the network.** One 0.59 GB frame:

| destination | throughput |
|---|---|
| S3 → memory (BytesIO) | 292 MB/s single, ~400 MB/s aggregate |
| S3 → local NVMe | 352 MB/s |
| S3 → EFS scratch (the configured `s3_cache_dir`) | 8.9 MB/s, and *worse* at higher concurrency |

The EFS mount's throughput is a property of the filesystem — bursting mode allots ~50 KB/s
per GB stored, so ~210 MB/s for the whole 4.2 TB volume, shared with every client mounting
it. So the read path was changed to fetch whole objects into RAM and hand the buffer to
h5netcdf (`../datasets/wave_dataset_memory.py`), which needs no disk at all. The local NVMe
is used only to pin the validation split, the one split re-read every epoch: measured 0.8 min
to extract val from local disk against 3.5 min streaming it.

Concurrency then has a sharp knee, and it is `num_workers` — each worker fetches serially, so
that *is* the number of frames in flight:

| frames in flight | throughput | s/sample |
|---|---|---|
| 4 | 352 MB/s | 3.18 |
| 8 | **380 MB/s** | **2.97** |
| 12 | 166 MB/s | 6.74 |

At 12 the rate more than halves while a separate process on the same machine still reads at
424 MB/s — a concurrency knee, not a bandwidth ceiling. `--num-workers 8` puts the data path
at 2.97 s/sample against 3.11 s/sample of GPU, so the GPU is (just) the limiter. Do not raise
it to "use the other 8 CPUs".
