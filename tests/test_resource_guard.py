"""Tests for host-memory detection and the DataLoader memory budget.

These pin the two mistakes that let a container get OOM-killed silently:

* a memory limit read from the host rather than the cgroup, and
* a guard ceiling set *above* that limit, so it could never fire first.

CPU-only and filesystem-only; no GPU, no S3, no model.
"""

from __future__ import annotations

import lightning as L
import pytest

from workshop_infrastructure import resource_guard as rg


# ---------------------------------------------------------------------------
# detect_memory_ceiling_gib
# ---------------------------------------------------------------------------

def _fake_cgroup(monkeypatch, tmp_path, *, v2=None, v1=None, host_bytes=None):
    """Point the module's three lookups at files under tmp_path."""
    real_is_file = rg.Path.is_file

    v2_path = tmp_path / "memory.max"
    v1_path = tmp_path / "memory.limit_in_bytes"
    if v2 is not None:
        v2_path.write_text(v2)
    if v1 is not None:
        v1_path.write_text(v1)

    mapping = {
        "/sys/fs/cgroup/memory.max": v2_path,
        "/sys/fs/cgroup/memory/memory.limit_in_bytes": v1_path,
    }

    def fake_init_is_file(self):
        target = mapping.get(str(self))
        return real_is_file(target) if target is not None else real_is_file(self)

    real_read_text = rg.Path.read_text

    def fake_read_text(self, *a, **k):
        target = mapping.get(str(self))
        return real_read_text(target, *a, **k) if target is not None else real_read_text(self, *a, **k)

    monkeypatch.setattr(rg.Path, "is_file", fake_init_is_file)
    monkeypatch.setattr(rg.Path, "read_text", fake_read_text)

    if host_bytes is not None:
        import psutil

        class FakeVM:
            total = host_bytes

        monkeypatch.setattr(psutil, "virtual_memory", lambda: FakeVM())


def test_reads_cgroup_v2_limit(monkeypatch, tmp_path):
    _fake_cgroup(monkeypatch, tmp_path, v2=str(60 * rg.GIB))
    assert rg.detect_memory_ceiling_gib() == pytest.approx(60.0)


def test_cgroup_v2_max_falls_through_to_v1(monkeypatch, tmp_path):
    """A literal "max" means unlimited, not a parse error."""
    _fake_cgroup(monkeypatch, tmp_path, v2="max\n", v1=str(32 * rg.GIB))
    assert rg.detect_memory_ceiling_gib() == pytest.approx(32.0)


def test_falls_back_to_host_when_cgroup_unlimited(monkeypatch, tmp_path):
    # cgroup v1 reports a sentinel near 2**63 rather than a word when unlimited.
    _fake_cgroup(
        monkeypatch, tmp_path, v2="max\n", v1=str(2 ** 63 - 4096),
        host_bytes=248 * rg.GIB,
    )
    assert rg.detect_memory_ceiling_gib() == pytest.approx(248.0)


def test_detects_a_real_limit_on_this_machine():
    """Whatever this machine is, the number must be positive and finite."""
    assert rg.detect_memory_ceiling_gib() > 0


# ---------------------------------------------------------------------------
# ResourceGuard ceiling — the regression test that matters
# ---------------------------------------------------------------------------

def test_default_ceiling_is_below_the_detected_limit():
    """The bug this guards against: a hard-coded 70 GB ceiling on a 60 GiB container can
    never fire before the kernel's OOM killer, so the guard was decorative."""
    guard = rg.ResourceGuard()
    assert guard.ceiling_gb < guard.detected_gib
    assert guard.ceiling_gb == pytest.approx(
        rg.DEFAULT_CEILING_FRACTION * rg.detect_memory_ceiling_gib()
    )


def test_explicit_ceiling_is_honoured():
    assert rg.ResourceGuard(ceiling_gb=12.5).ceiling_gb == pytest.approx(12.5)


def test_guard_raises_when_over_ceiling_and_names_the_knobs():
    guard = rg.ResourceGuard(ceiling_gb=0.001)  # certainly exceeded
    with pytest.raises(RuntimeError) as excinfo:
        guard._report(None, "test")
    message = str(excinfo.value)
    for knob in ("num_workers", "prefetch_factor", "pin_memory"):
        assert knob in message, f"the error should tell the reader to change {knob}"


def test_guard_is_a_lightning_callback():
    """It has to be usable in Trainer(callbacks=[...])."""
    assert isinstance(rg.ResourceGuard(), L.Callback)


# ---------------------------------------------------------------------------
# Page cache must not count against the budget
# ---------------------------------------------------------------------------

def _fake_memory_stat(monkeypatch, current_gib, anon_gib, file_gib, shmem_gib):
    monkeypatch.setattr(rg, "memory_current_gib", lambda: current_gib)
    monkeypatch.setattr(
        rg.Path, "is_file", lambda self: str(self) == "/sys/fs/cgroup/memory.stat"
    )
    monkeypatch.setattr(
        rg.Path, "read_text",
        lambda self, *a, **k: (
            f"anon {int(anon_gib * rg.GIB)}\n"
            f"file {int(file_gib * rg.GIB)}\n"
            f"shmem {int(shmem_gib * rg.GIB)}\n"
        ),
    )


def test_unreclaimable_excludes_page_cache(monkeypatch):
    """Reading SDO frames fills page cache, which the kernel evicts rather than OOM-kill.

    Counting it would abort runs that were never in danger — measured on an idle container
    here, 13.3 GiB charged to the cgroup of which 12.2 GiB was cache from frames read.
    """
    _fake_memory_stat(monkeypatch, current_gib=13.28, anon_gib=0.99, file_gib=12.21, shmem_gib=0.0)
    unreclaimable, cache = rg.memory_unreclaimable_gib()
    assert unreclaimable == pytest.approx(1.07, abs=0.05)
    assert cache == pytest.approx(12.21, abs=0.05)


def test_unreclaimable_still_counts_shmem(monkeypatch):
    """``shmem`` is nested inside ``file`` in cgroup v2, but tmpfs cannot be reclaimed
    without swap — and DataLoader workers pass every collated batch through /dev/shm.

    Measured mid-training: current 40.6, file 30.2 of which shmem 18.0. Subtracting all of
    ``file`` reported 9.2 GiB of pressure when the real figure was 28.4 GiB, which would
    have made the guard useless in the other direction from the 70 GB ceiling.
    """
    _fake_memory_stat(
        monkeypatch, current_gib=40.6, anon_gib=10.14, file_gib=30.23, shmem_gib=18.02
    )
    unreclaimable, cache = rg.memory_unreclaimable_gib()
    assert unreclaimable == pytest.approx(28.4, abs=0.2), "shmem must not be discounted"
    assert cache == pytest.approx(12.21, abs=0.05)


# ---------------------------------------------------------------------------
# describe_memory_budget
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "num_workers, prefetch, batch, expected_gib",
    [
        (2, 1, 2, 13.0),    # the notebook's setting after the fix
        (4, 2, 2, 52.0),    # what OOM-killed the container
        (8, 1, 2, 52.0),    # the batch script's old default
        (2, None, 2, 26.0),  # None means PyTorch's default of 2, not "no prefetch"
    ],
)
def test_budget_arithmetic(num_workers, prefetch, batch, expected_gib):
    line = rg.describe_memory_budget(num_workers, prefetch, batch)
    assert f"{expected_gib:.1f} GiB" in line, line


def test_budget_counts_both_worker_pools():
    """Train and val pools coexist: Lightning's sanity check spawns the validation pool
    before the first training batch, and the training iterator stays alive through it."""
    both = rg.describe_memory_budget(2, 1, 2, n_pools=2)
    one = rg.describe_memory_budget(2, 1, 2, n_pools=1)
    assert "13.0 GiB" in both
    assert "6.5 GiB" in one


def test_sample_size_matches_surya_stack():
    """1.625 GiB is 13 channels x 2 timesteps x 4096^2 x 4 bytes."""
    expected = 13 * 2 * 4096 * 4096 * 4 / rg.GIB
    assert expected == pytest.approx(1.625, abs=0.001)
    assert "1.625 GiB/sample" in rg.describe_memory_budget(1, 1, 1)
