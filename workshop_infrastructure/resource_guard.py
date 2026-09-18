"""
Host-memory budgeting shared by every Surya downstream app.

The hazard when fine-tuning Surya is **host RAM, not VRAM**. One sample of ``ts`` is
``(13, 2, 4096, 4096)`` fp32 = 1.625 GiB, and every DataLoader worker holds
``prefetch_factor * batch_size`` of them, so the difference between two worker settings
is tens of gigabytes. When host RAM runs out there is no exception to catch: the kernel's
OOM killer takes the process — and in a container, often the whole process tree with it.
What you see afterwards is a notebook cell with streamed output and no
``execution_count``, or a batch job that simply stopped.

Two pieces here, and the first is the reason the second exists.

``detect_memory_ceiling_gib()`` asks the OS how much memory this process may actually
use. That number is not what ``free`` prints and not what ``psutil.virtual_memory()``
returns: inside a container both report the **host node**, while the real limit is the
cgroup's. On the machine this was written for they differ by 4x (248 GiB host, 60 GiB
cgroup), which is exactly the kind of gap that turns a carefully budgeted run into an
OOM kill. Detecting it at runtime also keeps the number out of source comments, where
it goes stale silently the first time the deployment changes.

``ResourceGuard`` polls the process tree during training and raises before the OOM
killer does. Losing one run to a clean ``RuntimeError`` that names the knob to turn is
strictly better than losing the container and the rest of the schedule with it.
"""

from __future__ import annotations

import os
from pathlib import Path

import lightning as L
import torch

GIB = 2 ** 30

# Fraction of the detected ceiling ResourceGuard defaults to. The remainder covers what
# the guard cannot see coming: the transient NumPy peak inside a worker mid-normalization
# (helio.py's transform() chains four out-of-place ops on an 832 MiB array, then
# _load_and_stack_frames np.stacks a second full copy), the serialization buffer for a
# 1.8 GB checkpoint, and anything else sharing the cgroup — a notebook server, other
# kernels, editor processes.
DEFAULT_CEILING_FRACTION = 0.75


def detect_memory_ceiling_gib() -> float:
    """Return the memory this process may actually use, in GiB.

    Checks cgroup v2, then cgroup v1, then falls back to the host's physical memory.
    A cgroup limit of the literal ``"max"`` means unlimited, so it falls through too.

    Prefer this over ``psutil.virtual_memory().total`` anywhere a budget is being
    computed: under a container the latter reports the host node and will happily tell
    you there are 248 GiB available inside a 60 GiB cgroup.
    """
    # cgroup v2: a single unified hierarchy, one file.
    v2 = Path("/sys/fs/cgroup/memory.max")
    if v2.is_file():
        try:
            raw = v2.read_text().strip()
            if raw != "max":
                return int(raw) / GIB
        except (OSError, ValueError):
            pass

    # cgroup v1: limit_in_bytes, which reports a sentinel near 2**63 when unlimited
    # rather than a word like "max".
    v1 = Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    if v1.is_file():
        try:
            limit = int(v1.read_text().strip())
            if limit < 2 ** 62:
                return limit / GIB
        except (OSError, ValueError):
            pass

    import psutil

    return psutil.virtual_memory().total / GIB


def memory_current_gib() -> float | None:
    """Total charge against this process's cgroup in GiB, or ``None`` outside a cgroup.

    Counts everything sharing the cgroup — under JupyterHub that is the notebook server,
    every other kernel, and any editor or agent process, none of which is a child of this
    process. Summing PSS over ``psutil.Process().children()`` misses all of them, which is
    how a run that looks like it is using 30 GiB gets killed at a 60 GiB ceiling.

    **This total includes page cache, so it is not the right thing to guard on** — see
    :func:`memory_unreclaimable_gib`.
    """
    for path in (
        "/sys/fs/cgroup/memory.current",                    # v2
        "/sys/fs/cgroup/memory/memory.usage_in_bytes",      # v1
    ):
        p = Path(path)
        if p.is_file():
            try:
                return int(p.read_text().strip()) / GIB
            except (OSError, ValueError):
                pass
    return None


def memory_unreclaimable_gib() -> tuple[float, float] | None:
    """``(unreclaimable, page_cache)`` in GiB for this cgroup, or ``None`` if unavailable.

    This is the distinction that decides whether a memory reading is alarming.
    ``memory.current`` counts **page cache** — and reading SDO NetCDF files puts a lot of
    it there. Measured on this repo: 13.3 GiB charged to the cgroup of which 12.2 GiB was
    cache from frames already read. Cache is reclaimable: under pressure the kernel evicts
    it instead of invoking the OOM killer, so counting it would make a guard abort runs
    that were never in danger.

    What cannot be reclaimed is anonymous memory and tmpfs. Both matter here: the decoded
    tensors are anonymous, and DataLoader workers pass collated batches through
    ``/dev/shm``, whose pages are ``shmem`` and can only go to swap — of which this
    container has none.

    **``shmem`` is counted inside ``file``**, so the reclaimable part is ``file - shmem``,
    not ``file``. Subtracting all of ``file`` would discount the worker queues, which are
    the single largest term in a Surya run: measured mid-training here, ``file`` was
    30.2 GiB of which 18.0 GiB was ``shmem``, so guarding on ``current - file`` reported
    9.2 GiB when real pressure was 28.4 GiB.
    """
    stat = Path("/sys/fs/cgroup/memory.stat")
    current = memory_current_gib()
    if current is None or not stat.is_file():
        return None
    try:
        fields = dict(
            (k, int(v)) for k, v in (line.split() for line in stat.read_text().splitlines())
        )
    except (OSError, ValueError):
        return None
    if "file" not in fields:
        return None
    cache = max(fields["file"] - fields.get("shmem", 0), 0) / GIB
    return max(current - cache, 0.0), cache


def describe_memory_budget(
    num_workers: int,
    prefetch_factor: int | None,
    batch_size: int,
    gib_per_sample: float = 1.625,
    n_pools: int = 2,
) -> str:
    """One line stating worst-case DataLoader memory against the detected ceiling.

    Worst case is ``num_workers * prefetch_factor * batch_size * gib_per_sample`` per
    loader, and ``n_pools=2`` by default because the train and validation worker pools
    coexist: Lightning keeps the training iterator alive while it validates.

    ``gib_per_sample`` defaults to Surya's 13-channel, 2-timestep, 4096x4096 fp32 stack.
    A task with different channels or timesteps should pass its own.
    """
    pools = n_pools * num_workers * (prefetch_factor or 2) * batch_size * gib_per_sample
    ceiling = detect_memory_ceiling_gib()
    line = (
        f"[RES] DataLoader worst case {pools:.1f} GiB "
        f"({n_pools} pools x {num_workers} workers x prefetch {prefetch_factor or 2} "
        f"x batch {batch_size} x {gib_per_sample:.3f} GiB/sample) "
        f"against a {ceiling:.1f} GiB ceiling"
    )
    # What else is already in the cgroup matters: the headroom is the ceiling minus
    # everything sharing it, not minus zero. Page cache is reported separately because it
    # is reclaimable and so does not compete for the budget.
    split = memory_unreclaimable_gib()
    if split is not None:
        line += f"; {split[0]:.1f} GiB already in use ({split[1]:.1f} GiB reclaimable cache)"
    return line


class ResourceGuard(L.Callback):
    """Log GPU and host memory periodically, and abort before the machine dies.

    Proportional set size (PSS) is used rather than RSS: workers share the parent's
    copy-on-write pages, so summing RSS across the tree double-counts them and would
    trip the ceiling on a perfectly healthy run.

    Args:
        every_n_steps: How often to sample during training.
        ceiling_gb: Abort above this many GiB. ``None`` (the default) means
            ``DEFAULT_CEILING_FRACTION`` of :func:`detect_memory_ceiling_gib` — pass a
            number only to deliberately override a detected limit.
    """

    def __init__(self, every_n_steps: int = 25, ceiling_gb: float | None = None):
        self.every_n_steps = every_n_steps
        self.detected_gib = detect_memory_ceiling_gib()
        if ceiling_gb is None:
            self.ceiling_gb = DEFAULT_CEILING_FRACTION * self.detected_gib
            source = f"{DEFAULT_CEILING_FRACTION:.0%} of a detected {self.detected_gib:.1f} GiB"
        else:
            self.ceiling_gb = float(ceiling_gb)
            source = f"explicit (detected ceiling is {self.detected_gib:.1f} GiB)"
        self.peak_host_gb = 0.0
        self._proc = None
        # Printed rather than left in a comment: this is the number the run is actually
        # budgeted against, and it is a property of the machine, not of the source.
        print(f"[RES] host memory ceiling {self.ceiling_gb:.1f} GiB — {source}", flush=True)

    def _tree_memory_gb(self) -> tuple[float, str]:
        import psutil
        if self._proc is None:
            self._proc = psutil.Process()
        procs = [self._proc] + self._proc.children(recursive=True)
        total, kind = 0.0, "pss"
        for p in procs:
            try:
                info = p.memory_full_info()
                total += getattr(info, "pss", None) or info.rss
                if not hasattr(info, "pss"):
                    kind = "rss"
            except Exception:
                continue  # a worker exiting between listing and reading is not an error
        return total / GIB, kind

    def _report(self, trainer, tag: str) -> None:
        # Two numbers, because they answer different questions. cgroup memory.current is
        # authoritative — it is what the OOM killer compares against the limit, and it
        # counts every process sharing the cgroup. Tree PSS attributes: it says how much
        # of that is this run's own workers rather than a notebook server or a second
        # kernel. Guard on the first; report both so a near-miss is diagnosable.
        tree_gb, kind = self._tree_memory_gb()
        split = memory_unreclaimable_gib()
        if split is not None:
            budget_gb, cache_gb = split
        else:
            budget_gb, cache_gb = tree_gb, None
        self.peak_host_gb = max(self.peak_host_gb, budget_gb)

        msg = f"[RES] {tag} host {budget_gb:.1f} GiB (peak {self.peak_host_gb:.1f})"
        if cache_gb is not None:
            msg += f" [cgroup unreclaimable; +{cache_gb:.1f} cache, this tree {kind}={tree_gb:.1f}]"
        if torch.cuda.is_available():
            alloc = torch.cuda.max_memory_allocated() / GIB
            reserved = torch.cuda.max_memory_reserved() / GIB
            msg += f" | cuda peak alloc={alloc:.1f} GiB reserved={reserved:.1f} GiB"
        print(msg, flush=True)

        if budget_gb > self.ceiling_gb:
            raise RuntimeError(
                f"Unreclaimable host memory {budget_gb:.1f} GiB crossed the "
                f"{self.ceiling_gb:.1f} GiB ceiling (machine limit "
                f"{self.detected_gib:.1f} GiB; this process tree "
                f"accounts for {tree_gb:.1f} GiB of it). Lower num_workers or "
                f"prefetch_factor — worst case is num_workers * prefetch_factor * "
                f"batch_size * 1.625 GiB per loader, and the train and val pools coexist "
                f"— or set pin_memory=False to drop one host copy of each in-flight "
                f"batch. Raise ceiling_gb only if this machine really has the headroom."
            )

    def on_train_start(self, trainer, pl_module):
        # Before the first batch: an unaffordable configuration should fail in seconds,
        # not eight minutes into an epoch.
        self._report(trainer, "train start")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % self.every_n_steps == 0:
            self._report(trainer, f"epoch {trainer.current_epoch} batch {batch_idx}")

    def on_validation_epoch_end(self, trainer, pl_module):
        if not trainer.sanity_checking:
            self._report(trainer, f"epoch {trainer.current_epoch} val end")
