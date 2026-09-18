"""Tests for the DataLoader memory knobs in build_helio_dataloaders().

Each of these three arguments is a multiplier on a 1.625 GiB-per-sample tensor, and this
builder is reached by every forked downstream app, so their defaults are pinned here
rather than left to PyTorch's.

CPU-only: dataset construction is stubbed out, so nothing touches S3 or a NetCDF file.
"""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import TensorDataset

from workshop_infrastructure.datasets import builders


@pytest.fixture
def stub_datasets(monkeypatch):
    """Replace dataset construction; we are testing loader wiring, not the datasets."""
    dataset = TensorDataset(torch.zeros(8, 1))
    monkeypatch.setattr(
        builders, "build_helio_datasets",
        lambda cfg, cls, scalers=None, **kw: (dataset, dataset),
    )
    return dataset


class Cfg:
    """The four fields build_helio_dataloaders reads off a TrainingConfig."""
    batch_size = 2
    num_workers = 3
    seed = 42


def _build(stub, **kwargs):
    return builders.build_helio_dataloaders(Cfg(), None, scalers=None, **kwargs)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

def test_prefetch_factor_defaults_to_one_not_torch_default(stub_datasets):
    """PyTorch defaults to 2, which doubles every worker's resident set. At 1.625 GiB a
    sample that is the difference between a run that fits and an OOM kill."""
    train, val = _build(stub_datasets)
    assert train.prefetch_factor == 1
    assert val.prefetch_factor == 1


def test_persistent_and_pin_defaults_preserve_existing_behaviour(stub_datasets):
    train, _ = _build(stub_datasets)
    assert train.persistent_workers is True
    assert train.pin_memory is True


def test_only_the_training_loader_shuffles_and_drops_last(stub_datasets):
    train, val = _build(stub_datasets)
    assert train.drop_last is True
    assert val.drop_last is True, "unchanged default: opt in to False rather than out"
    assert isinstance(train.sampler, torch.utils.data.RandomSampler)
    assert isinstance(val.sampler, torch.utils.data.SequentialSampler)


# ---------------------------------------------------------------------------
# Overrides
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("prefetch", [1, 2, 4])
def test_prefetch_factor_is_honoured(stub_datasets, prefetch):
    train, val = _build(stub_datasets, prefetch_factor=prefetch)
    assert train.prefetch_factor == prefetch
    assert val.prefetch_factor == prefetch


def test_notebook_settings(stub_datasets):
    """What 2_finetune_template_1D_diego.ipynb passes."""
    train, val = _build(
        stub_datasets, num_workers=2, prefetch_factor=1,
        persistent_workers=False, pin_memory=False, drop_last_val=False,
    )
    assert (train.num_workers, train.prefetch_factor) == (2, 1)
    assert train.persistent_workers is False
    assert train.pin_memory is False
    assert val.drop_last is False, "every validation sample must reach val_loss"
    assert train.drop_last is True, "training still drops a ragged tail"


def test_num_workers_override_beats_config(stub_datasets):
    train, _ = _build(stub_datasets, num_workers=2)
    assert train.num_workers == 2  # Cfg.num_workers is 3


# ---------------------------------------------------------------------------
# num_workers == 0
# ---------------------------------------------------------------------------

def test_worker_only_kwargs_are_omitted_at_zero_workers(stub_datasets):
    """torch raises outright if prefetch_factor, persistent_workers or
    multiprocessing_context are passed with num_workers=0, so they must be conditional."""
    train, val = _build(stub_datasets, num_workers=0, prefetch_factor=1)
    for loader in (train, val):
        assert loader.num_workers == 0
        assert loader.prefetch_factor is None
        assert loader.persistent_workers is False


# ---------------------------------------------------------------------------
# Seeding is unaffected by the new arguments
# ---------------------------------------------------------------------------

def test_shuffle_order_depends_only_on_the_seed(stub_datasets):
    """An explicit generator is what keeps the epoch order from depending on ambient RNG
    state; the memory arguments must not disturb it."""
    torch.manual_seed(1234)
    first, _ = _build(stub_datasets, num_workers=0)
    order_a = list(iter(first.sampler))

    torch.manual_seed(9999)  # different ambient state, same config
    second, _ = _build(stub_datasets, num_workers=0, prefetch_factor=2, pin_memory=False)
    order_b = list(iter(second.sampler))

    assert order_a == order_b
