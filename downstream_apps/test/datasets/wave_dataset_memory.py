"""
An in-memory-reading ``waveDSDataset``, and why this task needs one.

``HelioNetCDFDataset``'s three S3 modes all assume the local filesystem is fast:
``download`` and ``simplecache`` write the whole 0.59 GB object into ``s3_cache_dir``
before opening it, and ``stream`` serves HDF5's random access as many small ranged GETs
(measured ~9x slower than fetching the object whole). That assumption does not hold on
this machine. Measured, one 13-channel SDO frame:

===========================================  ==========
destination                                  throughput
===========================================  ==========
S3 -> memory (BytesIO), concurrency 16          292 MB/s
S3 -> local NVMe (/home/jovyan), conc. 16       352 MB/s
S3 -> EFS scratch (the configured cache), 8     8.9 MB/s
S3 -> EFS scratch, concurrency 16               5.9 MB/s
===========================================  ==========

The configured ``s3_cache_dir`` is on an EFS mount whose throughput is a property of the
*filesystem*, not of this process: EFS bursting mode allots roughly 50 KB/s per GB stored,
which for the 4.2 TB filesystem is ~210 MB/s shared by every client mounting it, and
another tenant was consuming almost all of it. Raising concurrency made it *worse*, which
is the signature of a contended filesystem rather than a slow client. The local NVMe is
fast but only has 77 GB free, against a ~500 GB working set.

So the fix is to stop writing the frame down at all: fetch the whole object into RAM and
hand the buffer to ``h5netcdf``, which gives HDF5 the random access it needs without
touching a disk. One in-flight frame costs 0.59 GB of RAM, so eight workers holding two
frames each is ~9.4 GB — affordable on 124 GB, and the network path is 30-40x faster than
the one it replaces.

Two consequences worth being explicit about:

* **Nothing is cached, so every epoch re-reads from S3.** At ~1.18 GB per sample (two
  frames) and ~300 MB/s that is ~3.9 s of network per sample against ~3.1 s of GPU, so the
  data pipeline becomes the (mild) limiter rather than the GPU. ``cache_paths`` exists for
  the one case where that matters: the validation split is read every single epoch, and it
  is small enough (48 frames, 28 GB) to pin on the local NVMe.
* **This is a read-path change only.** The bytes decoded, the channels extracted and the
  normalization applied are identical to the parent class's, so a run using this dataset is
  numerically the same as one using ``s3_mode: download`` — just faster here.
"""

from __future__ import annotations

import io
import os
import threading
from uuid import uuid4

import numpy as np
import xarray as xr

from downstream_apps.test.datasets.wave_dataset import waveDSDataset
from workshop_infrastructure.utils import make_s3_client, parse_s3_uri

try:
    from boto3.s3.transfer import TransferConfig
except Exception:  # pragma: no cover
    TransferConfig = None


class waveDSDatasetInMemory(waveDSDataset):
    """``waveDSDataset`` that fetches S3 objects into RAM instead of onto disk.

    Overrides the public ``load_nc_data()`` only. Local (non-``s3://``) paths and every
    other part of the pipeline — frame sampling, channel masking, signum-log
    normalization, the label merge — are the parent classes' unchanged.

    Additional Args:
        cache_paths: S3 URIs that *should* be written to ``local_cache_dir`` after being
            fetched, because they will be read again many times. Pass the validation
            split's frames here: they are re-read every epoch, and pinning 48 of them costs
            28 GB. Everything not listed is fetched to memory and discarded, so the train
            split cannot evict them. ``None`` disables disk caching entirely.
        local_cache_dir: Where cached frames go. Must be on *fast* local storage — the
            point of this class is that the configured ``s3_cache_dir`` is not.
        local_cache_budget_gb: Refuse to write more than this into ``local_cache_dir``,
            so a mis-sized ``cache_paths`` cannot fill the root filesystem.
        connect_timeout_s, read_timeout_s: Socket timeouts for the S3 client. Set explicitly
            because the shared ``make_s3_client()`` helper exposes none, and its defaults let
            a stalled connection retry for minutes.
        fetch_timeout_s: Wall-clock deadline for one whole-object fetch, including boto3's
            internal retries. A healthy fetch is 1.4-3.3 s.
        fetch_attempts: How many times to retry a failed or timed-out fetch before giving up
            and failing the sample (which fails the run, deliberately: a silently skipped
            frame would change the dataset without changing any reported count).
    """

    def __init__(
        self,
        cache_paths: set[str] | None = None,
        local_cache_dir: str | None = None,
        local_cache_budget_gb: float = 60.0,
        s3_boto3_pool_size: int = 64,
        connect_timeout_s: float = 10.0,
        read_timeout_s: float = 45.0,
        fetch_timeout_s: float = 180.0,
        fetch_attempts: int = 4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.cache_paths = set(cache_paths) if cache_paths else set()
        self.local_cache_dir = local_cache_dir
        self.local_cache_budget_bytes = int(local_cache_budget_gb * 2**30)
        self.s3_boto3_pool_size = s3_boto3_pool_size
        # A healthy 0.59 GB fetch takes 1.4-3.3 s, so 180 s is ~50x the expected time: long
        # enough that a merely slow transfer is never abandoned, short enough that a hang
        # costs one frame rather than the run.
        self.connect_timeout_s = connect_timeout_s
        self.read_timeout_s = read_timeout_s
        self.fetch_timeout_s = fetch_timeout_s
        self.fetch_attempts = fetch_attempts
        # boto3 clients are not fork-safe and the loaders use spawn, so the client is
        # built lazily, once per worker process, and kept for the process's lifetime
        # (the parent class builds a fresh client per download, which costs a TLS
        # handshake on every frame).
        self._client = None
        self._client_lock = threading.Lock()
        if self.local_cache_dir:
            os.makedirs(self.local_cache_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Pickling
    # ------------------------------------------------------------------

    def __getstate__(self):
        """Drop the client and its lock so the dataset can cross a spawn boundary.

        DataLoader workers are started with ``multiprocessing_context="spawn"``, which
        pickles the dataset. Neither a boto3 client nor a ``threading.Lock`` is picklable,
        and neither *should* cross the boundary: a client carries connection state that is
        only valid in the process that opened it. Both are rebuilt on first use in the
        worker.
        """
        state = self.__dict__.copy()
        state["_client"] = None
        state["_client_lock"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._client_lock = threading.Lock()

    # ------------------------------------------------------------------
    # S3 client
    # ------------------------------------------------------------------

    def _get_client(self):
        """Return this process's S3 client, building it on first use.

        Deliberately does **not** use ``workshop_infrastructure.utils.make_s3_client``, which
        is otherwise the right helper: it exposes no socket timeouts, and botocore's defaults
        combined with ``mode="adaptive"`` and ``max_attempts=10`` can leave a single stalled
        connection retrying for many minutes. That is survivable interactively and fatal for
        an unattended schedule — observed here, one hung fetch blocked all eight DataLoader
        workers for ten minutes, because with ``prefetch_factor=1`` the main process consumes
        batches in order and every other worker was left waiting on the stalled one. Short
        timeouts and fewer attempts turn a hang into a fast retry.
        """
        if self._client is None:
            if self._client_lock is None:  # unpickled without __setstate__ running
                self._client_lock = threading.Lock()
            with self._client_lock:
                if self._client is None:
                    import boto3
                    from botocore import UNSIGNED
                    from botocore.config import Config as BotoConfig

                    from workshop_infrastructure.utils import detect_ec2_region
                    anon = bool(self.s3_storage_options.get("anon")
                                or self.s3fs_kwargs.get("anon"))
                    region = (os.environ.get("AWS_REGION")
                              or os.environ.get("AWS_DEFAULT_REGION")
                              or detect_ec2_region())
                    kwargs = dict(
                        max_pool_connections=max(self.s3_boto3_pool_size,
                                                 self.s3_boto3_max_concurrency * 2),
                        connect_timeout=self.connect_timeout_s,
                        read_timeout=self.read_timeout_s,
                        retries={"max_attempts": 4, "mode": "adaptive"},
                    )
                    if anon:
                        kwargs["signature_version"] = UNSIGNED
                    self._client = boto3.client("s3", region_name=region,
                                                config=BotoConfig(**kwargs))
        return self._client

    def _reset_client(self) -> None:
        """Discard the client, tearing down its connection pool.

        Called after a stalled fetch. Closing the pool is what makes an abandoned download
        thread fail instead of holding its socket, and the next call builds a fresh client.
        """
        client, self._client = self._client, None
        if client is not None:
            try:
                client.close()
            except Exception:
                pass

    def _transfer_config(self):
        part = self.s3_boto3_part_size_mb * 1024 * 1024
        return TransferConfig(
            multipart_threshold=part,
            multipart_chunksize=part,
            max_concurrency=self.s3_boto3_max_concurrency,
            use_threads=True,
            io_chunksize=1024 * 1024,
        )

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def _local_cache_path(self, s3_uri: str) -> str:
        """Local path for a cached frame, using the parent's naming so it is recognizable."""
        return os.path.join(self.local_cache_dir, os.path.basename(self._s3_cache_path(s3_uri)))

    def _cache_dir_bytes(self) -> int:
        try:
            with os.scandir(self.local_cache_dir) as it:
                return sum(e.stat().st_size for e in it if e.is_file())
        except FileNotFoundError:
            return 0

    def _fetch_to_memory(self, s3_uri: str) -> io.BytesIO:
        """Fetch a whole S3 object into a ``BytesIO``, under a wall-clock deadline per attempt.

        The deadline is a watchdog *around* boto3, not a replacement for the socket timeouts
        set in ``_get_client()``. Both are needed: the socket timeouts bound an individual
        request, and this bounds the whole multipart transfer, including boto3's own internal
        retry loop. Without the outer bound one unlucky object can stall a worker — and with
        ``prefetch_factor=1`` a stalled worker stalls the run.

        A timed-out attempt is abandoned rather than cancelled (boto3 offers no cancellation):
        the buffer is closed and the client's connection pool torn down, which makes the
        orphaned daemon thread fail on its next write instead of lingering with a socket and a
        0.59 GB buffer.
        """
        if TransferConfig is None:
            raise ImportError("boto3 is required for the in-memory read path.")
        bucket, key = parse_s3_uri(s3_uri)

        last_error: BaseException | None = None
        for attempt in range(1, self.fetch_attempts + 1):
            buf = io.BytesIO()
            outcome: dict = {}

            def _download() -> None:
                try:
                    self._get_client().download_fileobj(
                        bucket, key, buf, Config=self._transfer_config())
                    outcome["ok"] = True
                except BaseException as exc:  # reported to the caller below
                    outcome["error"] = exc

            thread = threading.Thread(target=_download, daemon=True)
            thread.start()
            thread.join(self.fetch_timeout_s)

            if outcome.get("ok"):
                buf.seek(0)
                return buf

            if thread.is_alive():
                last_error = TimeoutError(
                    f"{s3_uri} did not finish within {self.fetch_timeout_s}s "
                    f"(attempt {attempt}/{self.fetch_attempts})")
            else:
                last_error = outcome.get("error", RuntimeError(f"unknown failure on {s3_uri}"))
            self.logger.warning(f"Fetch failed, retrying: {last_error}")
            print(f"[S3] retry {attempt}/{self.fetch_attempts}: {last_error}", flush=True)
            self._reset_client()
            buf.close()

        raise RuntimeError(f"Failed to fetch {s3_uri} after "
                           f"{self.fetch_attempts} attempts") from last_error

    def _maybe_write_cache(self, s3_uri: str, buf: io.BytesIO) -> None:
        """Persist a whitelisted frame to fast local storage, budget permitting."""
        if not self.local_cache_dir or s3_uri not in self.cache_paths:
            return
        path = self._local_cache_path(s3_uri)
        if os.path.exists(path):
            return
        size = buf.getbuffer().nbytes
        if self._cache_dir_bytes() + size > self.local_cache_budget_bytes:
            return  # silently stream instead; the budget is a ceiling, not a target
        tmp = f"{path}.{os.getpid()}.{uuid4().hex}.partial"
        try:
            with open(tmp, "wb") as fh:
                fh.write(buf.getbuffer())
            os.replace(tmp, path)  # atomic: a losing racer never publishes a truncated file
        except BaseException:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise

    def load_nc_data(self, filepath: str, timestep, channels: list[str]) -> np.ndarray:
        """Return ``(C, H, W)`` for ``channels``, reading S3 objects through RAM.

        Falls back to the parent implementation for local paths, so an index of local
        NetCDF files behaves exactly as before.
        """
        self._ensure_logger()

        if not self._is_s3_path(filepath):
            return super().load_nc_data(filepath, timestep, channels)

        cached = self._local_cache_path(filepath) if self.local_cache_dir else None
        if cached and os.path.exists(cached):
            self.logger.info(f"Reading local cache {cached}.")
            with xr.open_dataset(cached, engine="h5netcdf", chunks=None, cache=False) as ds:
                return ds[channels].to_array().load().to_numpy()

        self.logger.info(f"Fetching {filepath} into memory.")
        buf = self._fetch_to_memory(filepath)
        try:
            self._maybe_write_cache(filepath, buf)
            buf.seek(0)
            with xr.open_dataset(buf, engine="h5netcdf", chunks=None, cache=False) as ds:
                return ds[channels].to_array().load().to_numpy()
        finally:
            buf.close()  # release the 0.59 GB before the next frame is fetched
