"""
Shared helpers for the wave-classification experiments.

Three things are needed by more than one script in this directory, so they live here
once rather than being copied (and drifting) across the feature extractor, the baseline
study and the fine-tuning entry point:

1. **Event-level stratified subsetting.** The catalog stores each event twice — once as
   ``wave`` at ``start_time + 30 min`` and once as ``no wave`` at ``start_time - 30 min``
   — so an event is a *matched pair* of samples. Subsetting by event therefore gives
   exact class balance for free, and never puts the two halves of one event on opposite
   sides of a split boundary. ``waveDSDataset``'s own ``max_number_of_samples`` cannot do
   this: it head-slices a frame sorted by ``ds_index``, so ``max_number_of_samples=10``
   returns the 5 *earliest* pairs (all June-July 2010), which is a chronological bias, not
   a sample.

2. **Pre-flight assertions.** Every failure mode this task has actually hit is silent:
   a tokenizer that stays random because its shape disagreed with the checkpoint, LoRA
   target names that match nothing, a head frozen at its initialization. Each one still
   produces a plausible-looking loss curve. These helpers turn them into startup errors.

3. **The running-difference feature.** ``RunningDifferenceLogisticModel`` computes it on
   the GPU during training; ``extract_baseline_features.py`` computes it once and caches
   it. ``pooled_running_difference()`` is the single definition both use, and
   ``assert_feature_matches_model()`` checks they agree numerically.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Column in the wave catalog that identifies an *event*. Both the "wave" row
# (mid_time = start_time + 30 min) and the "no wave" row (mid_time - 30 min) of one
# event share it, which is what makes pair-level subsetting possible.
EVENT_COL = "start_time"
POSITIVE_CLASS = "wave"


def wave_label_transform(class_series: pd.Series) -> pd.Series:
    """Map the catalog's ``class`` column to float32 1.0 / 0.0.

    Passed to ``waveDSDataset(label_transform=...)``. Kept here so the notebooks, the
    feature extractor and the training script cannot disagree on which class is positive.
    """
    return (class_series == POSITIVE_CLASS).astype(np.float32)


# ---------------------------------------------------------------------------
# Event-level subsetting
# ---------------------------------------------------------------------------

@dataclass
class SubsetInfo:
    """What a subsetting call actually did, for logging and assertions."""
    n_samples: int
    n_events: int
    n_positive: int
    n_negative: int
    n_dropped_unpaired: int
    n_events_available: int

    def describe(self) -> str:
        return (
            f"{self.n_samples} samples / {self.n_events} events "
            f"({self.n_positive} wave, {self.n_negative} no wave) "
            f"from {self.n_events_available} complete pairs available; "
            f"dropped {self.n_dropped_unpaired} unpaired sample(s)"
        )


def event_stratified_subset(
    dataset,
    n_samples: int | None = None,
    seed: int = 42,
) -> SubsetInfo:
    """Restrict ``dataset`` in place to a seeded, event-level stratified subset.

    Only events whose *both* samples survived the Surya-index match are kept, so the
    result is exactly 50/50 by construction and accuracy of 0.5 is exactly chance. On the
    train split at ``ds_time_tolerance: 15min`` that costs 18 of 1210 matched samples and
    buys perfect balance.

    Args:
        dataset: A ``waveDSDataset`` (already constructed, so the Surya-index match has
            happened). Modified in place.
        n_samples: Number of samples to keep — must be even, since events come in pairs.
            ``None`` keeps every complete pair.
        seed: Selects which events. Fixed across the runs of a scaling study so that the
            N=48 subset is a subset of the N=96 subset is a subset of N=192, …; the
            learning curve then isolates sample count from which events were drawn.

    Returns:
        A ``SubsetInfo`` describing the selection.

    Raises:
        ValueError: if ``n_samples`` is odd, or larger than the available paired samples.
    """
    df = dataset.df_valid_indices
    if EVENT_COL not in df.columns:
        raise ValueError(
            f"The wave catalog must carry a {EVENT_COL!r} column to subset by event; "
            f"got columns {sorted(df.columns)[:8]}…"
        )

    # Positional order of df_valid_indices matches dataset.valid_indices, and
    # waveDSDataset.__getitem__ indexes both by position. Any mask must be applied to
    # both, in the same order.
    labels = df["label"].to_numpy()
    events = df[EVENT_COL].to_numpy()

    per_event = pd.Series(labels).groupby(events).nunique()
    complete = np.sort(per_event.index[per_event == 2].to_numpy())
    n_dropped = int(len(labels) - np.isin(events, complete).sum())

    if n_samples is None:
        chosen_events = complete
    else:
        if n_samples % 2 != 0:
            raise ValueError(
                f"n_samples must be even: events contribute one 'wave' and one 'no wave' "
                f"sample each, so an odd count cannot be class-balanced (got {n_samples})."
            )
        n_events = n_samples // 2
        if n_events > len(complete):
            raise ValueError(
                f"Requested {n_samples} samples ({n_events} events) but only "
                f"{len(complete)} complete event pairs ({2 * len(complete)} samples) "
                f"survived the Surya-index match. Loosen data.ds_time_tolerance or "
                f"lower --train-n."
            )
        # Permute the *sorted* event list so the choice depends only on the seed, not on
        # dict/CSV ordering. Taking a prefix makes the subsets nested across N, which is
        # what lets the learning curve attribute a change to sample count alone.
        order = np.random.default_rng(seed).permutation(len(complete))
        chosen_events = complete[np.sort(order[:n_events])]

    keep = np.flatnonzero(np.isin(events, chosen_events))

    dataset.df_valid_indices = df.iloc[keep]
    dataset.valid_indices = [dataset.valid_indices[i] for i in keep]
    dataset.adjusted_length = len(keep)

    kept_labels = dataset.df_valid_indices["label"].to_numpy()
    info = SubsetInfo(
        n_samples=len(keep),
        n_events=len(chosen_events),
        n_positive=int((kept_labels == 1).sum()),
        n_negative=int((kept_labels == 0).sum()),
        n_dropped_unpaired=n_dropped,
        n_events_available=len(complete),
    )
    if info.n_positive != info.n_negative:
        raise AssertionError(
            f"Event-level subsetting should be exactly balanced but produced "
            f"{info.n_positive} wave / {info.n_negative} no wave. This means an event "
            f"contributed two samples of the same class."
        )
    return info


def dataset_frame_uris(dataset) -> list[str]:
    """Return the distinct NetCDF paths a dataset will read, in chronological order.

    Used to report how much of the S3 cache is already warm before a run starts, and to
    size the download.
    """
    offsets = dataset.time_delta_input_minutes
    stamps = sorted({ts + dt for ts in dataset.valid_indices for dt in offsets})
    return [dataset.index.loc[ts, "path"] for ts in stamps]


def cache_coverage(dataset) -> tuple[int, int]:
    """Return ``(n_cached, n_total)`` for the frames ``dataset`` will read.

    Uses the dataset's own cache-path scheme, so the answer reflects what the training
    run will actually find rather than a guess at the naming.
    """
    uris = dataset_frame_uris(dataset)
    # The in-memory dataset consults local_cache_dir, not s3_cache_dir, and only for the
    # frames on its whitelist. Ask it where it would look rather than assuming.
    resolve = getattr(dataset, "_local_cache_path", None) or getattr(
        dataset, "_s3_cache_path", None)
    root = getattr(dataset, "local_cache_dir", None) or dataset.s3_cache_dir
    if not root or resolve is None:
        return 0, len(uris)
    cached = sum(1 for u in uris if u.startswith("s3://") and os.path.exists(resolve(u)))
    return cached, len(uris)


# Frames per gigabyte of cache, for turning a byte budget into a frame count. One
# 13-channel 4096x4096 SDO frame is ~0.59 GB on disk.
FRAME_GB = 0.59

# Space to keep free on the cache filesystem. A Lightning checkpoint of this model is ~1.45 GB
# (the state dict carries the frozen backbone, not only the adapters), and a full schedule
# writes one per run.
RESERVE_GB = 15.0


def plan_pinned_frames(
    val_dataset,
    train_dataset,
    budget_gb: float,
    train_priority_n: int | None = None,
    subset_seed: int = 42,
) -> tuple[set[str], dict]:
    """Choose which frames to keep on fast local storage, given a byte budget.

    The S3 read path sustains ~175 MB/s inside the DataLoader, against ~3.1 s of GPU per
    sample, so a run that streams every frame is **data-bound at roughly twice the GPU's
    cost**. Local disk is 350 MB/s but only ~66 GB (≈110 frames) of it is free, against a
    481 GB working set. So the budget has to be spent where it buys the most epochs.

    Two facts make the choice easy:

    * The **validation split is read on every epoch** and is small (48 frames, 28 GB). It is
      pinned first, unconditionally: over a 30-epoch run those 48 frames would otherwise be
      fetched 1,440 times.
    * The event-level subsets are **nested** — the N=48 subset's events are a subset of
      N=96's, which are a subset of N=192's. So pinning the frames of the *smallest*
      training subset makes those frames local for every larger run too. The small runs
      become GPU-bound, and the large ones still get a partial hit.

    Args:
        val_dataset: The already-subset validation dataset. Pinned first.
        train_dataset: The already-subset training dataset, at its full N. Its
            ``train_priority_n``-sample nested prefix is pinned with whatever budget remains.
        budget_gb: Total bytes to spend, in GB. Should be below the free space on the cache
            filesystem, since the frames land there.
        train_priority_n: Which nested prefix of the training subset to prioritize. Defaults
            to the smallest scaling run (48). ``None`` disables train pinning.
        subset_seed: Must match the seed the runs use, or the "nested prefix" is a different
            set of events.

    Returns:
        ``(uris, report)`` — the whitelist, and a dict describing how the budget was spent.
    """
    import copy
    import shutil

    # Clamp to what the cache filesystem can actually give. Lightning checkpoints hold the
    # frozen backbone as well as the adapters (~1.45 GB each), and a schedule of six runs
    # therefore needs ~9 GB that must not be eaten by cached frames.
    cache_root = getattr(val_dataset, "local_cache_dir", None)
    if cache_root and os.path.isdir(cache_root):
        free_gb = shutil.disk_usage(cache_root).free / 2**30
        already_gb = sum(
            e.stat().st_size for e in os.scandir(cache_root) if e.is_file()) / 2**30
        allowed = already_gb + max(free_gb - RESERVE_GB, 0.0)
        if budget_gb > allowed:
            print(f"[cache] budget {budget_gb:.0f} GB exceeds what {cache_root} can give "
                  f"({free_gb:.0f} GB free, {already_gb:.0f} GB already cached, "
                  f"{RESERVE_GB:.0f} GB reserved for checkpoints); using {allowed:.0f} GB")
            budget_gb = allowed

    val_uris = [u for u in dataset_frame_uris(val_dataset) if u.startswith("s3://")]
    budget_frames = int(budget_gb / FRAME_GB)
    chosen = list(val_uris[:budget_frames])
    report = {
        "budget_gb": budget_gb,
        "budget_frames": budget_frames,
        "val_frames_pinned": len(chosen),
        "val_frames_total": len(val_uris),
        "train_frames_pinned": 0,
        "train_priority_n": train_priority_n,
    }

    remaining = budget_frames - len(chosen)
    if train_priority_n and remaining > 0:
        # A shallow copy shares the dataframe and index but gets its own valid_indices, so
        # subsetting the copy does not disturb the dataset that is about to be trained on.
        probe = copy.copy(train_dataset)
        probe.valid_indices = list(train_dataset.valid_indices)
        probe.df_valid_indices = train_dataset.df_valid_indices
        try:
            event_stratified_subset(probe, n_samples=train_priority_n, seed=subset_seed)
        except ValueError as exc:
            report["train_pin_skipped"] = str(exc)
        else:
            train_uris = [u for u in dataset_frame_uris(probe) if u.startswith("s3://")]
            take = train_uris[:remaining]
            chosen.extend(take)
            report["train_frames_pinned"] = len(take)
            report["train_frames_priority_total"] = len(train_uris)

    uris = set(chosen)
    report["total_frames_pinned"] = len(uris)
    report["estimated_gb"] = round(len(uris) * FRAME_GB, 1)
    return uris, report


def pin_frames(dataset, extra_datasets=(), uris: set[str] | None = None) -> int:
    """Apply a local-disk cache whitelist to one or more datasets, and return its size.

    Only meaningful for ``waveDSDatasetInMemory``, whose ``cache_paths`` attribute this
    sets. Call it after subsetting, and before the DataLoaders are built: the loaders use
    spawn, so the attribute is pickled to the workers as it stands at that moment.

    Args:
        dataset: Dataset whose frames are pinned when ``uris`` is not given.
        extra_datasets: Other datasets that should honour the same whitelist. They share the
            cache directory, so listing them lets them *read* the pinned frames — without
            this, a train dataset would re-fetch a frame the val dataset had already cached.
        uris: An explicit whitelist, e.g. from ``plan_pinned_frames()``. When ``None``, all
            of ``dataset``'s own frames are used.

    Returns:
        Number of frames whitelisted, or 0 if these datasets have no local cache.
    """
    if not hasattr(dataset, "cache_paths"):
        return 0
    if uris is None:
        uris = {u for u in dataset_frame_uris(dataset) if u.startswith("s3://")}
    for ds in (dataset, *extra_datasets):
        if hasattr(ds, "cache_paths"):
            ds.cache_paths = set(uris)
    return len(uris)


# ---------------------------------------------------------------------------
# Pre-flight assertions
# ---------------------------------------------------------------------------

def count_matched_pretrained(model: torch.nn.Module, checkpoint_path: str) -> tuple[int, int]:
    """Count how many checkpoint tensors are present in ``model`` *after* loading.

    Call this after ``load_pretrained_weights()``. For each checkpoint tensor it looks for
    a model tensor — under the checkpoint's own key or under ``backbone.<key>``, the two
    spellings the loader tries — that is bit-identical to it.

    This is deliberately a check on the *outcome* rather than a second copy of the
    loader's matching rule: a tensor whose shape disagreed was skipped and still holds its
    random initialization, so it fails the comparison, which is exactly the question
    ("did the pretrained tokenizer actually land?") that bug 1 was about.

    Returns:
        ``(n_matched, n_checkpoint_tensors)``. For Surya 366M the ceiling is 157/159: the
        two ``unembed.unembed.0.*`` tensors are the pretraining decoder, which the
        fine-tuning wrapper builds with ``finetune=True`` and therefore does not have.
    """
    ckpt = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
    state = model.state_dict()
    matched = 0
    for key, value in ckpt.items():
        for candidate in (key, f"backbone.{key}"):
            have = state.get(candidate)
            if have is not None and have.shape == value.shape:
                if torch.equal(have.detach().cpu(), value):
                    matched += 1
                break
    return matched, len(ckpt)


def assert_tokenizer_loaded(model: torch.nn.Module, checkpoint_path: str) -> int:
    """Assert the pretrained patch embedding actually landed, and return the match count.

    The failure this guards is silent and total: with ``time_dim: 1`` the model builds a
    13-channel ``Conv2d`` while the checkpoint's is ``Conv2d(26, 1280, 16, 16)``
    (13 channels x 2 timesteps), the shapes disagree, ``load_pretrained_weights()`` skips
    the key, and every image entering the transformer is tokenized by *random* weights.
    The loss still falls.
    """
    matched, total = count_matched_pretrained(model, checkpoint_path)
    print(f"[PREFLIGHT] pretrained tensors matched: {matched} / {total}")
    if matched < total - 2:
        raise AssertionError(
            f"Only {matched} of {total} pretrained tensors are present in the model. "
            f"Expected {total - 2} (everything but the two pretraining-decoder tensors, "
            f"unembed.unembed.0.weight/bias, which finetune=True does not build).\n"
            f"The usual cause is model.time_embedding.time_dim disagreeing with the "
            f"checkpoint's tokenizer: it is Conv2d(in_channels * time_dim, ...), and the "
            f"checkpoint was trained with time_dim=2 (26 input channels)."
        )
    return matched


def assert_lora_adapted_attention(model: torch.nn.Module) -> list[str]:
    """Assert LoRA adapters reached attention, not just the feed-forward layers.

    PEFT raises only when *no* ``target_modules`` entry matches anything, so a list of
    plausible-but-wrong names (``q_proj``, ``k_proj``, ``v_proj``, ``out_proj`` — none of
    which exist in Surya) is silently reduced to whatever else happened to match. That is
    how a run trained ``fc1``/``fc2`` alone and reported 2,666,241 trainable parameters
    where 3,157,761 was intended.
    """
    adapted = sorted({
        name.split(".lora_A")[0].replace("base_model.model.", "")
        for name, _ in model.named_parameters()
        if ".lora_A" in name
    })
    missing = [
        needle for needle in ("attn.qkv", "attn.proj")
        if not any(a.endswith(needle) for a in adapted)
    ]
    if missing:
        raise AssertionError(
            f"LoRA did not adapt {', '.join(missing)}. Surya fuses q/k/v into attn.qkv and "
            f"names the output projection attn.proj; PEFT silently ignores target_modules "
            f"entries that match nothing.\nAdapted modules were: {adapted}"
        )
    n_attn = sum(1 for a in adapted if ".attn." in a)
    print(f"[PREFLIGHT] LoRA adapted {len(adapted)} modules, {n_attn} of them in attention")
    return adapted


def assert_trainable_count(model: torch.nn.Module, expected: int | None = None) -> int:
    """Print (and optionally assert) the trainable parameter count.

    Reference values for this task's config, pinned by ``tests/test_lora_setup.py``:

    ==============================================  =========
    regime                                          trainable
    ==============================================  =========
    LoRA r=8, penultimate_linear_layer: false       1,518,081
    LoRA r=8, penultimate_linear_layer: true        3,157,761
    linear probe (head only), penultimate false         2,561
    ==============================================  =========
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[PREFLIGHT] trainable: {trainable:,} / {total:,} ({100 * trainable / total:.3f}%)")
    if expected is not None and trainable != expected:
        raise AssertionError(
            f"Expected {expected:,} trainable parameters, got {trainable:,}. Check "
            f"model.use_lora, model.lora_config.r, model.penultimate_linear_layer and the "
            f"head_* modules apply_peft_lora() reported as modules_to_save."
        )
    return trainable


def assert_batch_shape(batch: dict, n_channels: int, n_timesteps: int, img_size: int) -> None:
    """Assert the input stack has the shape the backbone's tokenizer was trained for."""
    ts = batch["ts"]
    expected = (n_channels, n_timesteps, img_size, img_size)
    if tuple(ts.shape[1:]) != expected:
        raise AssertionError(
            f"batch['ts'] has shape {tuple(ts.shape)}; expected (B, {', '.join(map(str, expected))}). "
            f"ts.shape[2] is the number of input timesteps and must be 2 for this "
            f"checkpoint — it comes from model.time_embedding.time_dim and "
            f"data.time_delta_input_minutes."
        )
    print(f"[PREFLIGHT] batch['ts'] shape {tuple(ts.shape)}, dtype {ts.dtype}")


# ---------------------------------------------------------------------------
# The running-difference feature
# ---------------------------------------------------------------------------

def pooled_running_difference(
    ts_signum_log: torch.Tensor,
    channel_index: int,
    pool_kernel: int,
) -> torch.Tensor:
    """AIA193 running difference (now - previous), mean-pooled.

    This is the single definition of the baseline's feature.
    ``RunningDifferenceLogisticModel.forward()`` computes it per training step;
    ``extract_baseline_features.py`` computes it once and caches the result.

    Mean pooling composes: pooling a ``pool_kernel=8`` map by a further factor of 4 is
    identical to pooling the original by 32, because every block is the same size. That
    is what lets the cache be written once at kernel 8 (512x512, 1 MB/sample) and still
    serve the kernel-16/32/64/128 sweep exactly.

    Args:
        ts_signum_log: ``(B, C, T, H, W)`` in **signum-log** space — z-score undone,
            log compression retained. Use ``destandardize_channels()`` to get there from
            what the dataset returns.
        channel_index: Index of AIA193 within the C dimension.
        pool_kernel: Mean-pool kernel size (and stride).

    Returns:
        ``(B, H // pool_kernel, W // pool_kernel)``.
    """
    x = ts_signum_log[:, channel_index, :, :, :]           # (B, T, H, W)
    diff = x[:, -1] - x[:, 0]                              # now - previous
    pooled = torch.nn.functional.avg_pool2d(diff.unsqueeze(1), kernel_size=pool_kernel)
    return pooled.squeeze(1)


def repool(features: np.ndarray, factor: int) -> np.ndarray:
    """Mean-pool cached feature maps by a further integer ``factor``.

    Args:
        features: ``(N, S, S)`` cached at some base kernel.
        factor: Additional pooling factor; must divide ``S``.

    Returns:
        ``(N, S // factor, S // factor)``.
    """
    if factor == 1:
        return features
    n, s, _ = features.shape
    if s % factor:
        raise ValueError(f"repool factor {factor} does not divide the {s}x{s} cached map.")
    out = features.reshape(n, s // factor, factor, s // factor, factor)
    return out.mean(axis=(2, 4))


# ---------------------------------------------------------------------------
# Datasets and DataLoaders
# ---------------------------------------------------------------------------

def build_wave_datasets(
    cfg,
    scalers,
    include_test: bool = False,
    in_memory: bool = True,
    local_cache_budget_gb: float = 40.0,
):
    """Build the train / val (/ test) ``waveDSDataset`` objects described by ``cfg``.

    Thin wrapper over ``build_helio_datasets()`` that adds three things this task needs:
    the wave-specific keyword arguments in one place, the **test** split, and the choice of
    read path.

    ``in_memory=True`` (the default) uses ``waveDSDatasetInMemory``, which fetches each S3
    object into RAM and hands the buffer to h5netcdf rather than writing 0.59 GB to disk
    first. On this machine that is 400 MB/s against 9 MB/s, because the configured cache
    directory is on a throughput-capped EFS mount — see that class's module docstring for
    the measurements. Set ``in_memory=False`` for the stock ``s3_mode``-driven path.

    ``build_helio_datasets()`` returns exactly two datasets — one from
    ``data.train_data_path`` with ``phase="train"`` and one from ``data.valid_data_path``
    with ``phase="val"``. To get the third split we call it a second time with
    ``valid_data_path`` pointed at the test index and keep the second dataset. That is the
    right ``phase`` for a held-out split (no channel masking, no flips), and it means the
    ~20 generic dataset arguments still come from the one place that owns them.

    Returns:
        ``(train, val)`` or ``(train, val, test)``.
    """
    from workshop_infrastructure.datasets.builders import build_helio_datasets

    if in_memory:
        from downstream_apps.test.datasets.wave_dataset_memory import (
            waveDSDatasetInMemory as dataset_cls,
        )
        read_kwargs = dict(
            # Populated after subsetting by pin_frames(): the whitelist has to be the
            # frames the *subset* reads, which is not known until the subset exists.
            cache_paths=None,
            local_cache_dir=cfg.data.s3_cache_dir,
            local_cache_budget_gb=local_cache_budget_gb,
        )
    else:
        from downstream_apps.test.datasets.wave_dataset import waveDSDataset as dataset_cls
        read_kwargs = {}

    task_kwargs = dict(
        **read_kwargs,
        return_surya_stack=True,
        # None on purpose: max_number_of_samples head-slices chronologically. Subsetting
        # is event_stratified_subset()'s job, after the match has happened.
        max_number_of_samples=None,
        label_transform=wave_label_transform,
        ds_wave_index_path=cfg.data.wave_index_path,
        ds_time_column=cfg.data.ds_time_column,
        ds_class_column=cfg.data.ds_class_column,
        ds_time_tolerance=cfg.data.ds_time_tolerance,
        ds_match_direction=cfg.data.ds_match_direction,
    )

    train, val = build_helio_datasets(cfg, dataset_cls, scalers=scalers, **task_kwargs)
    if not include_test:
        return train, val

    test_index = str(cfg.data.train_data_path).replace("_train.csv", "_test.csv")
    if not os.path.isfile(test_index):
        raise FileNotFoundError(
            f"Test index not found at {test_index}. It is derived from "
            f"data.train_data_path by swapping _train.csv for _test.csv."
        )
    original = cfg.data.valid_data_path
    try:
        cfg.data.valid_data_path = test_index
        _, test = build_helio_datasets(cfg, dataset_cls, scalers=scalers, **task_kwargs)
    finally:
        cfg.data.valid_data_path = original
    return train, val, test


def build_wave_loader(
    dataset,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int | None,
    seed: int,
    shuffle: bool,
    drop_last: bool,
    persistent_workers: bool = True,
):
    """Build a DataLoader with the two knobs ``build_helio_dataloaders()`` does not expose.

    ``build_helio_dataloaders()`` is the right entry point for a normal run, but it hard-codes
    ``drop_last=True`` for *both* loaders and does not expose ``prefetch_factor``. Both matter
    here:

    * ``drop_last=True`` on a 24-sample validation set at batch 2 is survivable, but on any
      odd-sized split it silently discards a sample — 1 of 27 is 3.7% of the metric that
      selects checkpoints.
    * ``prefetch_factor`` is the only bound on worker memory. ``ts`` is
      ``(B, 13, 2, 4096, 4096)`` fp32 = 1.74 GB per sample, so worst-case resident set is
      ``num_workers * prefetch_factor * batch_size * 1.74 GB``: 27.8 GB at
      ``8 x 1 x 2``, but 55.7 GB at PyTorch's default factor of 2.

    ``persistent_workers`` defaults to ``True``, matching ``build_helio_dataloaders()``: a
    batch-script run does one ``fit()`` and wants to pay ``spawn``'s re-import cost once.
    **Pass ``False`` in a notebook.** Persistent workers make the DataLoader stateful: the
    worker pool outlives the iterator, so ``DataLoader.__iter__`` takes torch's
    ``self._iterator._reset(self)`` branch instead of building a fresh pool. If anything killed
    those workers between two ``fit()`` calls — a ``KeyboardInterrupt``, a cell restart, an OOM
    reaper — the reset handshake talks to dead PIDs, times out on an empty queue, and raises
    ``DataLoader worker (pid(s) ...) exited unexpectedly``. Lightning then tears the fit loop
    down, and teardown calls ``_DataFetcher.reset()`` before ``iter(combined_loader)`` ever
    succeeded, so the error you actually see is the misleading secondary
    ``RuntimeError: Please call iter(combined_loader) first.`` With ``persistent_workers=False``
    every ``iter()`` builds a fresh pool, so the loader is reusable across fits and interrupts
    cost nothing but the next epoch's startup.

    Everything else — spawn (the dataset holds a boto3 handle that does not survive fork),
    persistent workers, the explicit generator and the per-worker seeding — matches
    ``build_helio_dataloaders()`` exactly, by reusing its ``_seed_worker``. That function is
    private but importing it is deliberate: its whole purpose is to displace Lightning's
    auto-injected worker init, which derives worker seeds from ambient global RNG state, and
    a second copy of it here could drift out of step with the datasets it seeds.
    """
    from functools import partial

    from torch.utils.data import DataLoader
    from workshop_infrastructure.datasets.builders import _seed_worker

    kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
    )
    if num_workers > 0:
        kwargs["multiprocessing_context"] = "spawn"
        kwargs["persistent_workers"] = persistent_workers
        kwargs["worker_init_fn"] = partial(_seed_worker, base_seed=seed)
        if prefetch_factor is not None:
            kwargs["prefetch_factor"] = prefetch_factor

    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(seed)
        kwargs["generator"] = generator

    return DataLoader(dataset, shuffle=shuffle, **kwargs)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_run(run_name: str, csv_dir: Path, evals: dict, out_dir: Path) -> list[Path]:
    """Write the training-curve and ROC figures for one run. Returns the paths written.

    Lives here rather than in ``4_finetune_wave_1D.py`` because both that script and
    ``1_baseline_template_diego.ipynb`` need it, and a module whose name starts with a digit
    cannot be imported — so the notebook had no way to call the script's copy.

    Every number plotted comes from the CSV logger's ``metrics.csv``, which is the same
    stream Lightning sends to wandb. That is deliberate: the local PNG and the wandb
    dashboard cannot disagree, because they are two renderings of one set of values.
    Lightning writes one row per logging event, so training rows (every
    ``log_every_n_steps``) and validation rows (once per epoch) interleave with NaNs in each
    other's columns; ``groupby("epoch").mean()`` collapses that into one row per epoch.

    Args:
        run_name: Used in the titles and the output filenames.
        csv_dir: The ``CSVLogger`` version directory holding ``metrics.csv``. Missing or
            absent file means the curve figure is skipped rather than raising — a run that
            died before its first logging event should not also lose its ROC figure.
        evals: ``{split: {"logits": ndarray, "labels": ndarray, "report": dict}}`` for the
            held-out evaluation. Pass ``{}`` to plot the training curves only.
        out_dir: Directory for the PNGs; created if needed.

    Returns:
        The paths written, in order.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    metrics_csv = Path(csv_dir) / "metrics.csv"
    if metrics_csv.is_file():
        df = pd.read_csv(metrics_csv)
        agg = df.groupby("epoch").mean(numeric_only=True)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        for col, style in [("train_loss", "-o"), ("val_loss", "-s")]:
            if col in agg:
                axes[0].plot(agg.index, agg[col], style, ms=4, label=col)
        # ln 2 is what a model predicting the class base rate scores, and the splits are
        # exactly balanced, so this line is the "learned nothing" reference.
        axes[0].axhline(np.log(2), ls=":", c="0.5", label="base rate (ln 2)")
        axes[0].set(xlabel="epoch", ylabel="BCE", title=f"{run_name} — loss")
        axes[0].legend(fontsize=8)
        for col, style in [("train_metric_accuracy", "-o"), ("val_metric_accuracy", "-s"),
                           ("val_metric_auroc", "-^")]:
            if col in agg:
                axes[1].plot(agg.index, agg[col], style, ms=4, label=col)
        axes[1].axhline(0.5, ls=":", c="0.5", label="chance")
        axes[1].set(xlabel="epoch", ylabel="score", ylim=(0, 1),
                    title=f"{run_name} — accuracy / AUROC")
        axes[1].legend(fontsize=8)
        path = out_dir / f"{run_name}_curves.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        written.append(path)

    if evals:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        for split, payload in evals.items():
            logits, labels = payload["logits"], payload["labels"]
            if len(np.unique(labels)) == 2:
                fpr, tpr, _ = roc_curve(labels, logits)
                axes[0].plot(fpr, tpr, label=f"{split} (AUROC {payload['report']['auroc']:.3f})")
            axes[1].hist(logits[labels == 0], bins=12, alpha=0.6, label=f"{split} no wave")
            axes[1].hist(logits[labels == 1], bins=12, alpha=0.6, label=f"{split} wave")
        axes[0].plot([0, 1], [0, 1], ":", c="0.5")
        axes[0].set(xlabel="false positive rate", ylabel="true positive rate",
                    title=f"{run_name} — ROC (best checkpoint)")
        axes[0].legend(fontsize=8)
        axes[1].axvline(0, ls=":", c="0.5")
        axes[1].set(xlabel="logit", ylabel="count", title="score distribution")
        axes[1].legend(fontsize=8)
        path = out_dir / f"{run_name}_roc.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        written.append(path)

    return written
