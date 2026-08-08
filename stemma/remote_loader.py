"""Selective safetensors reader over HTTP Range requests (and local mmap).

This module is the load-bearing piece of Stemma's *low transfer* claim: it never
downloads a checkpoint. It reads the 8-byte safetensors size prefix, then the
JSON header, and afterwards only the exact byte ranges of the rows of the
tensors a caller actually asks for. The same public API works against a local
directory so that the benchmark can compare bytes-read like with like.

Claim: low-transfer -- every byte that crosses the wire is accounted for in
:class:`stemma.types.TransferStats`, and the row-range machinery below is what
turns "read a 15 GB model" into "read a few hundred kilobytes".

safetensors layout recap (frozen by the format, not by us)::

    [0:8]        little-endian uint64 header length N
    [8:8+N]      UTF-8 JSON header: name -> {dtype, shape, data_offsets:[b,e]}
                 plus an optional "__metadata__" entry
    [8+N:]       tensor payload; a tensor's absolute range is (8+N+b, 8+N+e)
"""

from __future__ import annotations

import json
import math
import mmap
import os
import struct
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import requests

from .types import ModelRef, TensorMeta, TransferStats
from .utils import (
    atomic_write_json,
    default_cache_dir,
    get_logger,
    human_bytes,
    is_local_path,
    local_path_of,
    stable_hash,
)

LOG = get_logger(__name__)

#: Number of contiguous row blocks :func:`select_rows` spreads its budget over.
#: 16 keeps the sample stratified across the whole tensor while turning a
#: per-row request storm into 16 Range requests. Measured against the Hub, a
#: redirect-following Range request costs ~1.0 s and bandwidth is ~24 MB/s, so
#: a round trip is worth roughly 24 MB of payload -- request count, not byte
#: count, is what dominates wall-clock here. See :func:`select_rows`.
SAMPLE_BLOCKS: int = 16

#: Default number of concurrent in-flight Range requests. Remote reads are
#: latency-bound, so this is the main wall-clock lever; it does not change how
#: many bytes cross the wire.
DEFAULT_MAX_WORKERS: int = 8

#: Two row runs whose byte gap is smaller than this are fetched as one request.
#: Downloading a little slack is cheaper than paying another round trip.
COALESCE_GAP: int = 64 * 1024

#: Hub endpoint template. ``?download=true`` is deliberately *not* appended --
#: it only changes the Content-Disposition and would not help a Range read.
HF_ENDPOINT: str = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")

USER_AGENT: str = "stemma/0.1"

#: Canonical single-file shard name and the sharded-index sidecar.
SINGLE_FILE: str = "model.safetensors"
INDEX_FILE: str = "model.safetensors.index.json"

_HEADER_LEN_BYTES: int = 8
_MAX_HEADER_BYTES: int = 512 * 1024 * 1024  # sanity guard against a corrupt prefix

_RETRY_DELAYS: Tuple[float, ...] = (0.5, 1.0, 2.0)
_RETRY_STATUS: frozenset[int] = frozenset({408, 425, 429, 500, 502, 503, 504})

_TIMEOUT: Tuple[float, float] = (10.0, 120.0)


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #


class RemoteLoaderError(RuntimeError):
    """Base class for every failure raised by this module.

    Claim: infra -- a single catchable base keeps the CLI and benchmark from
    having to know the transport details.
    """


class UnsupportedDtype(RemoteLoaderError):
    """Raised when a safetensors dtype string has no numpy decoding path.

    Claim: low-transfer -- we decode bytes ourselves rather than pulling in a
    full framework loader, so an unknown dtype must fail loudly instead of
    silently mis-reading the payload.
    """


class RangeUnsupported(RemoteLoaderError):
    """Raised when a server answers 200 to a Range request.

    Claim: low-transfer -- a 200 means the server is about to hand us the whole
    file, which is exactly the thing this project promises never to do; we abort
    rather than quietly downloading gigabytes.
    """


# --------------------------------------------------------------------------- #
# dtype decoding
# --------------------------------------------------------------------------- #

#: safetensors dtype string -> little-endian numpy dtype, for the cases numpy
#: can represent natively.
_NUMPY_DTYPE: Dict[str, np.dtype] = {
    "F64": np.dtype("<f8"),
    "F32": np.dtype("<f4"),
    "F16": np.dtype("<f2"),
    "I64": np.dtype("<i8"),
    "I32": np.dtype("<i4"),
    "I16": np.dtype("<i2"),
    "I8": np.dtype("i1"),
    "U64": np.dtype("<u8"),
    "U32": np.dtype("<u4"),
    "U16": np.dtype("<u2"),
    "U8": np.dtype("u1"),
    "BOOL": np.dtype("?"),
}

#: Bytes per element for every dtype we know, including the ones numpy cannot
#: represent natively (BF16, FP8). Needed to compute row byte ranges.
_ITEMSIZE: Dict[str, int] = {
    "F64": 8,
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "I64": 8,
    "I32": 4,
    "I16": 2,
    "I8": 1,
    "U64": 8,
    "U32": 4,
    "U16": 2,
    "U8": 1,
    "BOOL": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "F8_E4M3FN": 1,
    "F8_E4M3FNUZ": 1,
    "F8_E5M2FNUZ": 1,
}

_F8_TABLES: Dict[str, np.ndarray] = {}


def _build_f8_table(kind: str) -> np.ndarray:
    """Build the 256-entry float32 lookup table for an FP8 encoding.

    Claim: low-transfer -- decoding FP8 checkpoints in pure numpy means we can
    Range-read a quantised model without installing a framework that knows the
    format.

    ``kind`` is ``"E4M3"`` (4 exponent / 3 mantissa bits, bias 7, no infinities,
    NaN only when exponent *and* mantissa are all ones) or ``"E5M2"``
    (5 exponent / 2 mantissa bits, bias 15, IEEE-like inf/NaN).
    """
    if kind == "E4M3":
        exp_bits, man_bits, bias, ieee_specials = 4, 3, 7, False
    elif kind == "E5M2":
        exp_bits, man_bits, bias, ieee_specials = 5, 2, 15, True
    else:  # pragma: no cover - guarded by callers
        raise UnsupportedDtype(f"unknown fp8 encoding: {kind}")

    man_max = (1 << man_bits) - 1
    exp_max = (1 << exp_bits) - 1
    table = np.zeros(256, dtype=np.float32)
    for code in range(256):
        sign = -1.0 if (code >> 7) & 1 else 1.0
        exp = (code >> man_bits) & exp_max
        man = code & man_max
        if exp == exp_max:
            if ieee_specials:
                value = math.inf if man == 0 else math.nan
            else:
                # E4M3 has no infinities: the all-ones pattern is NaN, every
                # other exp==max code is a perfectly ordinary large normal.
                value = math.nan if man == man_max else (1.0 + man / (man_max + 1)) * 2.0 ** (exp - bias)
        elif exp == 0:
            value = (man / (man_max + 1)) * 2.0 ** (1 - bias)
        else:
            value = (1.0 + man / (man_max + 1)) * 2.0 ** (exp - bias)
        table[code] = np.float32(sign * value)
    return table


def f8_table(kind: str) -> np.ndarray:
    """Return (and memoise) the FP8 decode table for ``"E4M3"`` / ``"E5M2"``.

    Claim: low-transfer -- memoising keeps per-tensor decode cost negligible so
    the byte budget, not the CPU, stays the binding constraint.
    """
    tbl = _F8_TABLES.get(kind)
    if tbl is None:
        tbl = _build_f8_table(kind)
        _F8_TABLES[kind] = tbl
    return tbl


def itemsize_of(st_dtype: str) -> int:
    """Bytes per element for a safetensors dtype string.

    Claim: low-transfer -- row byte ranges are computed from this, so it is the
    arithmetic that decides how few bytes a request actually asks for.
    """
    try:
        return _ITEMSIZE[st_dtype.upper()]
    except KeyError as exc:
        raise UnsupportedDtype(f"unsupported safetensors dtype {st_dtype!r}") from exc


def decode_buffer(raw: bytes | bytearray | memoryview, st_dtype: str) -> np.ndarray:
    """Decode a raw safetensors byte buffer into a flat numpy array.

    Claim: low-transfer -- BF16 and FP8 are widened here with bit tricks and
    lookup tables instead of torch, so a Range read stays a pure-numpy path with
    no framework download.

    BF16 is the top 16 bits of an IEEE float32, so widening to ``uint32``,
    shifting left by 16 and reinterpreting as ``float32`` is exact.
    """
    dt = st_dtype.upper()
    native = _NUMPY_DTYPE.get(dt)
    if native is not None:
        return np.frombuffer(raw, dtype=native)
    if dt == "BF16":
        u16 = np.frombuffer(raw, dtype="<u2")
        u32 = u16.astype(np.uint32) << np.uint32(16)
        return u32.view(np.float32)
    if dt in ("F8_E4M3", "F8_E4M3FN", "F8_E4M3FNUZ"):
        codes = np.frombuffer(raw, dtype=np.uint8)
        return f8_table("E4M3")[codes]
    if dt in ("F8_E5M2", "F8_E5M2FNUZ"):
        codes = np.frombuffer(raw, dtype=np.uint8)
        return f8_table("E5M2")[codes]
    raise UnsupportedDtype(f"unsupported safetensors dtype {st_dtype!r}")


# --------------------------------------------------------------------------- #
# Row selection + range coalescing
# --------------------------------------------------------------------------- #


def select_rows(
    n_rows: int,
    max_rows: Optional[int] = None,
    row_stride: Optional[int] = None,
    *,
    blocks: int = SAMPLE_BLOCKS,
) -> np.ndarray:
    """Deterministically choose which rows of a 2D tensor to fetch.

    Claim: low-transfer -- subsampling rows is the single biggest byte lever we
    have; making the choice depend only on ``(n_rows, max_rows, row_stride,
    blocks)`` means two different models are always compared row-for-row, so
    the saving costs us nothing in comparability.

    **Block-stratified, not evenly spaced.**  The obvious implementation picks
    ``max_rows`` rows spread uniformly with ``linspace``.  That is statistically
    fine and catastrophically slow: spreading 256 rows over a 49152-row
    embedding leaves ~220 KiB between consecutive rows, far more than
    :data:`COALESCE_GAP`, so the reader issues *one HTTP request per row*.
    Measured against the Hub, a redirect-following Range request costs ~1.0 s
    (the ``resolve`` endpoint 302s to a signed CDN URL and the redirect is
    re-resolved every time), so a single 256-row tensor would take ~4 minutes.

    Instead the budget is spent as ``blocks`` contiguous runs placed evenly
    across the row range.  The byte count is *identical* -- still ``max_rows``
    rows -- but each run is one Range request, so the same tensor costs ~16
    requests instead of 256.  Statistically this is stratified sampling: the
    strata still span the whole tensor, which matters because row order is
    rarely meaningless (embedding rows are token-id ordered, hence roughly
    frequency ordered, so a single contiguous block from the top would be a
    badly biased sample).
    """
    n_rows = int(n_rows)
    if n_rows <= 0:
        return np.zeros(0, dtype=np.int64)
    if row_stride is not None and int(row_stride) > 1:
        base = np.arange(0, n_rows, int(row_stride), dtype=np.int64)
    else:
        base = np.arange(n_rows, dtype=np.int64)

    if max_rows is None or not (0 < int(max_rows) < base.size):
        return base.astype(np.int64, copy=False)

    want = int(max_rows)
    n_blocks = max(1, min(int(blocks), want))
    per_block = max(1, want // n_blocks)
    # Evenly spaced block start positions over the available index range.
    if n_blocks == 1:
        starts = np.array([0], dtype=np.int64)
    else:
        starts = np.rint(
            np.linspace(0.0, float(base.size - per_block), n_blocks)
        ).astype(np.int64)
    picked: List[np.ndarray] = []
    for s in starts:
        s = int(max(0, min(s, base.size - 1)))
        picked.append(np.arange(s, min(s + per_block, base.size), dtype=np.int64))
    pick = np.unique(np.concatenate(picked)) if picked else np.zeros(0, dtype=np.int64)
    if pick.size > want:
        pick = pick[:want]
    return base[pick].astype(np.int64, copy=False)


def coalesce_ranges(
    ranges: Sequence[Tuple[int, int]],
    gap: int = COALESCE_GAP,
    *,
    max_slack: Optional[int] = None,
) -> List[Tuple[int, int]]:
    """Merge sorted half-open byte ranges separated by less than ``gap`` bytes.

    Claim: low-transfer -- trading a few kilobytes of slack for one fewer HTTP
    round trip is what keeps a strided row read from turning into thousands of
    requests.

    ``max_slack`` caps the total number of *unwanted* bytes merging may pull in.
    Without it a fixed 64 KiB gap would swallow a whole small tensor (many
    narrow rows sit closer than 64 KiB apart), which would silently break the
    very claim this module exists to support; callers pass the size of the
    payload they actually want, bounding over-fetch at 2x.
    """
    out: List[Tuple[int, int]] = []
    slack = 0
    for start, end in sorted(ranges):
        if end <= start:
            continue
        if out:
            prev_start, prev_end = out[-1]
            hole = start - prev_end
            if hole <= gap and (max_slack is None or slack + max(hole, 0) <= max_slack):
                slack += max(hole, 0)
                out[-1] = (prev_start, max(prev_end, end))
                continue
        out.append((start, end))
    return out


# --------------------------------------------------------------------------- #
# The source
# --------------------------------------------------------------------------- #


class SafeTensorsSource:
    """Range-reading accessor for a safetensors checkpoint, remote or local.

    Claim: low-transfer -- this class is the implementation of the headline
    "never download the checkpoint" requirement: it fetches the header, then
    only the byte ranges of the rows requested, and reports every byte it moved
    through :attr:`stats`.

    Usage::

        with SafeTensorsSource("Qwen/Qwen2.5-0.5B-Instruct") as src:
            meta = src.index()["model.embed_tokens.weight"]
            W = src.get_tensor(meta.name, max_rows=512)
            print(src.stats.bytes_read, src.total_size())
    """

    def __init__(
        self,
        ref: ModelRef,
        *,
        revision: str = "main",
        token: Optional[str] = None,
        cache_dir: Optional[str] = None,
        session: Optional["requests.Session"] = None,
        max_workers: int = DEFAULT_MAX_WORKERS,
    ) -> None:
        """Bind to a Hub repo id or a local directory without touching the network.

        Claim: low-transfer -- construction is free; nothing is read until the
        caller asks for the index, the config or a tensor.
        """
        self.ref: ModelRef = ref
        self.revision: str = revision
        self.token: Optional[str] = token or _token_from_env()
        self.cache_dir: Path = Path(cache_dir) if cache_dir else default_cache_dir()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        #: How many Range requests may be in flight at once. Remote reads are
        #: latency-bound (~0.87 s per request, almost all of it redirect
        #: resolution), so concurrency is the only lever that does not cost
        #: extra bytes. Set to 1 to force the sequential path.
        self.max_workers: int = max(1, int(max_workers))

        self.local: bool = is_local_path(ref)
        self.root: Optional[Path] = local_path_of(ref) if self.local else None

        #: Row indices used by the most recent :meth:`get_tensor` call.
        self.last_rows: Optional[np.ndarray] = None

        self._stats = TransferStats()
        self._stats_lock = threading.Lock()
        self._session: Optional["requests.Session"] = session
        self._own_session: bool = session is None
        self._closed = False

        self._files: Optional[List[str]] = None
        self._index: Optional[Dict[str, TensorMeta]] = None
        self._config: Optional[Dict[str, Any]] = None
        self._file_sizes: Dict[str, int] = {}
        self._total_size: Optional[int] = None
        self._mmaps: Dict[str, Tuple[Any, mmap.mmap]] = {}

    # ---------------------------------------------------------------- dunder

    def __enter__(self) -> "SafeTensorsSource":
        """Enter the context manager.

        Claim: infra -- deterministic cleanup of sockets and mmaps.
        """
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """Leave the context manager, closing sockets and mmaps.

        Claim: infra.
        """
        self.close()

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        """Short debugging representation including bytes moved so far.

        Claim: infra.
        """
        kind = "local" if self.local else "hub"
        return (
            f"<SafeTensorsSource {kind} {self.ref!r} rev={self.revision!r} "
            f"read={human_bytes(self._stats.bytes_read)} reqs={self._stats.requests}>"
        )

    # ---------------------------------------------------------------- public

    @property
    def stats(self) -> TransferStats:
        """Live, additive transfer accounting for this source.

        Claim: low-transfer -- this object *is* the evidence for the claim; the
        benchmark divides :attr:`TransferStats.full_size_bytes` by
        ``bytes_read`` to report the reduction factor.
        """
        return self._stats

    def files(self) -> List[str]:
        """Names of the safetensors shards that make up this checkpoint.

        Claim: low-transfer -- resolved from the tiny index sidecar (or a repo
        file listing), never by probing shard payloads.
        """
        if self._files is None:
            self._files = self._resolve_files()
        return list(self._files)

    def index(self) -> Dict[str, TensorMeta]:
        """Return ``name -> TensorMeta`` across every shard, from headers alone.

        Claim: low-transfer -- knowing dtype, shape and absolute byte range for
        every tensor after reading only the headers is what makes selective
        reads possible at all.
        """
        if self._index is None:
            merged: Dict[str, TensorMeta] = {}
            for fname in self.files():
                header, size = self._load_header(fname)
                if size:
                    self._file_sizes[fname] = size
                base = header["__stemma_payload_start__"]
                for name, entry in header.items():
                    if name in ("__metadata__", "__stemma_payload_start__"):
                        continue
                    if not isinstance(entry, dict) or "data_offsets" not in entry:
                        continue
                    beg, end = entry["data_offsets"]
                    merged[name] = TensorMeta(
                        name=name,
                        dtype=str(entry.get("dtype", "F32")).upper(),
                        shape=tuple(int(x) for x in entry.get("shape", ())),
                        begin=int(base) + int(beg),
                        end=int(base) + int(end),
                        file=fname,
                    )
            self._index = merged
            if len(self._file_sizes) == len(self._files or []) and self._file_sizes:
                self._total_size = int(sum(self._file_sizes.values()))
                self._stats.full_size_bytes = max(
                    self._stats.full_size_bytes, self._total_size
                )
        return self._index

    def total_size(self) -> int:
        """Total size in bytes of all shards, without downloading any of them.

        Claim: low-transfer -- the denominator of the reduction factor comes
        from ``Content-Length`` / the ``Content-Range`` total suffix, so proving
        the saving never costs a download.
        """
        if self._total_size is not None:
            return self._total_size
        files = self.files()
        for fname in files:
            if fname in self._file_sizes:
                continue
            size = self._file_size(fname)
            if size:
                self._file_sizes[fname] = size
        total = int(sum(self._file_sizes.get(f, 0) for f in files))
        self._total_size = total
        self._stats.full_size_bytes = max(self._stats.full_size_bytes, total)
        return total

    def get_tensor(
        self,
        name: str,
        *,
        dtype: Any = np.float32,
        max_rows: Optional[int] = None,
        row_stride: Optional[int] = None,
    ) -> np.ndarray:
        """Read one tensor, fetching only the byte ranges of the rows requested.

        Claim: low-transfer -- for a 2D row-major tensor only the selected rows'
        byte ranges cross the wire, coalesced into as few requests as possible;
        the chosen indices are exposed on :attr:`last_rows` so a second model can
        be read row-for-row identically.
        """
        meta = self._meta(name)
        rows = select_rows(meta.shape[0] if meta.shape else 0, max_rows, row_stride)
        return self.get_tensor_rows(name, rows, dtype=dtype)

    def get_tensor_rows(
        self,
        name: str,
        rows: Optional[Sequence[int]] = None,
        *,
        dtype: Any = np.float32,
    ) -> np.ndarray:
        """Read an explicit set of rows of a tensor.

        Claim: low-transfer -- lets a caller force the *same* row set on two
        models (so a comparison is apples-to-apples) while still paying only for
        those rows' bytes.
        """
        meta = self._meta(name)
        shape = meta.shape
        esize = itemsize_of(meta.dtype)

        if len(shape) < 2:
            # 0-D / 1-D tensors are tiny; read them whole.
            raw = self._read_ranges(meta.file, [(meta.begin, meta.end)])
            arr = decode_buffer(raw, meta.dtype)
            self.last_rows = np.arange(shape[0] if shape else 1, dtype=np.int64)
            arr = arr.reshape(shape) if shape else arr.reshape(())
            return _finalize(arr, dtype)

        n_rows = int(shape[0])
        row_elems = 1
        for d in shape[1:]:
            row_elems *= int(d)
        row_nbytes = row_elems * esize

        if rows is None:
            sel = np.arange(n_rows, dtype=np.int64)
        else:
            sel = np.asarray(list(rows), dtype=np.int64).reshape(-1)
            if sel.size:
                if sel.min() < 0 or sel.max() >= n_rows:
                    raise IndexError(
                        f"row index out of range for {name!r} with {n_rows} rows"
                    )
                sel = np.unique(sel)
        self.last_rows = sel

        if sel.size == 0:
            empty = np.zeros((0,) + tuple(int(d) for d in shape[1:]), dtype=np.float32)
            return _finalize(empty, dtype)

        wanted = [
            (meta.begin + int(r) * row_nbytes, meta.begin + (int(r) + 1) * row_nbytes)
            for r in sel
        ]
        runs = coalesce_ranges(wanted, COALESCE_GAP, max_slack=sel.size * row_nbytes)
        buffers = [self._read_ranges(meta.file, [run]) for run in runs]

        out = np.empty(sel.size * row_nbytes, dtype=np.uint8)
        run_i = 0
        for i, (start, end) in enumerate(wanted):
            while not (runs[run_i][0] <= start and end <= runs[run_i][1]):
                run_i += 1
            off = start - runs[run_i][0]
            chunk = np.frombuffer(buffers[run_i], dtype=np.uint8, count=row_nbytes, offset=off)
            out[i * row_nbytes : (i + 1) * row_nbytes] = chunk

        arr = decode_buffer(out.tobytes(), meta.dtype)
        arr = arr.reshape((sel.size,) + tuple(int(d) for d in shape[1:]))
        return _finalize(arr, dtype)

    def get_tensors(self, names: Sequence[str], **kw: Any) -> Dict[str, np.ndarray]:
        """Read several tensors with the same row policy.

        Claim: low-transfer -- a sketch needs a handful of tensors, and batching
        them through one source reuses the cached header and the TCP session.
        """
        out: Dict[str, np.ndarray] = {}
        for n in names:
            out[n] = self.get_tensor(n, **kw)
        return out

    def config(self) -> Dict[str, Any]:
        """Return ``config.json`` (a small whole-file read), cached on disk.

        Claim: low-transfer -- architecture metadata comes from a few kilobytes
        of JSON rather than from inspecting weights.
        """
        if self._config is not None:
            return self._config
        if self.local:
            assert self.root is not None
            path = self.root / "config.json"
            if path.is_file():
                data = path.read_bytes()
                self._bump(bytes_read=len(data))
                self._bump(requests=1)
                try:
                    self._config = json.loads(data.decode("utf-8"))
                except Exception:
                    LOG.warning("could not parse local config.json for %s", self.ref)
                    self._config = {}
            else:
                self._config = {}
            return self._config

        key = stable_hash((self.ref, self.revision, "config.json"))
        cached = self._cache_read(f"config-{key}.json")
        if cached is not None:
            self._config = cached if isinstance(cached, dict) else {}
            return self._config

        resp = self._request("GET", self._url("config.json"), allow_404=True)
        if resp is None:
            self._config = {}
            return self._config
        try:
            self._config = json.loads(resp.content.decode("utf-8"))
        except Exception:
            LOG.warning("could not parse config.json for %s", self.ref)
            self._config = {}
        self._cache_write(f"config-{key}.json", self._config)
        return self._config

    def close(self) -> None:
        """Release mmaps and, if we created it, the HTTP session.

        Claim: infra -- long benchmark runs open hundreds of sources.
        """
        if self._closed:
            return
        self._closed = True
        for fh, mm in self._mmaps.values():
            try:
                mm.close()
            except Exception:  # pragma: no cover
                pass
            try:
                fh.close()
            except Exception:  # pragma: no cover
                pass
        self._mmaps.clear()
        if self._own_session and self._session is not None:
            try:
                self._session.close()
            except Exception:  # pragma: no cover
                pass
        self._session = None

    # ------------------------------------------------------------- internals

    def _meta(self, name: str) -> TensorMeta:
        """Look a tensor up in the header index, with a helpful error.

        Claim: low-transfer.
        """
        idx = self.index()
        try:
            return idx[name]
        except KeyError as exc:
            raise KeyError(f"{name!r} not found in {self.ref!r} ({len(idx)} tensors)") from exc

    # -- session / requests ------------------------------------------------

    def _bump(
        self,
        *,
        bytes_read: int = 0,
        requests: int = 0,
        seconds: float = 0.0,
        cache_hits: int = 0,
    ) -> None:
        """Accumulate transfer statistics atomically.

        Claim: low-transfer -- ``bytes_read`` is a headline number, and Range
        reads are issued from a thread pool. ``+=`` on an int is a non-atomic
        read-modify-write even under the GIL, so concurrent workers would
        silently under-count. Every mutation goes through this lock.
        """
        with self._stats_lock:
            # NOTE: these four lines must stay raw field increments. Routing
            # them back through _bump() would re-enter a non-reentrant Lock and
            # deadlock the very first request.
            self._stats.bytes_read += int(bytes_read)
            self._stats.requests += int(requests)
            self._stats.seconds += float(seconds)
            self._stats.cache_hits += int(cache_hits)

    def _sess(self) -> "requests.Session":
        """Return the shared :class:`requests.Session`, creating it on demand.

        Claim: low-transfer -- connection reuse removes a TLS handshake per
        Range request, which dominates latency once payloads are this small.
        """
        if self._session is None:
            self._session = requests.Session()
            self._own_session = True
            # The default urllib3 pool holds 10 connections; concurrent Range
            # reads above that would serialise on the pool (and log
            # "Connection pool is full"). Size it to the worker count.
            pool = max(10, self.max_workers * 2)
            adapter = requests.adapters.HTTPAdapter(
                pool_connections=pool, pool_maxsize=pool
            )
            self._session.mount("https://", adapter)
            self._session.mount("http://", adapter)
        headers = {"User-Agent": USER_AGENT}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        self._session.headers.update(headers)
        return self._session

    def _url(self, filename: str) -> str:
        """Build the Hub resolve URL for one file of this repo/revision.

        Claim: low-transfer.
        """
        return f"{HF_ENDPOINT}/{self.ref}/resolve/{self.revision}/{filename}"

    def _request(
        self,
        method: str,
        url: str,
        *,
        headers: Optional[Dict[str, str]] = None,
        allow_404: bool = False,
        require_206: bool = False,
    ) -> Optional["requests.Response"]:
        """Perform one HTTP request with 3 attempts and exponential backoff.

        Claim: low-transfer -- a 200 answer to a Range request means the server
        ignored us and is about to stream the whole shard, so we raise
        :class:`RangeUnsupported` instead of paying for the download.
        """
        sess = self._sess()
        last_exc: Optional[BaseException] = None
        for attempt in range(len(_RETRY_DELAYS)):
            t0 = time.perf_counter()
            try:
                resp = sess.request(
                    method,
                    url,
                    headers=headers,
                    timeout=_TIMEOUT,
                    allow_redirects=True,
                    stream=True,
                )
            except Exception as exc:  # network hiccup -> retry
                self._bump(seconds=time.perf_counter() - t0)
                self._bump(requests=1)
                last_exc = exc
                LOG.debug("request error (%s) attempt %d: %s", url, attempt + 1, exc)
                time.sleep(_RETRY_DELAYS[attempt])
                continue

            status = resp.status_code
            if status == 404 and allow_404:
                self._bump(requests=1)
                self._bump(seconds=time.perf_counter() - t0)
                resp.close()
                return None

            if status in _RETRY_STATUS and attempt < len(_RETRY_DELAYS) - 1:
                delay = _retry_after(resp, _RETRY_DELAYS[attempt])
                self._bump(requests=1)
                self._bump(seconds=time.perf_counter() - t0)
                resp.close()
                LOG.debug("HTTP %s from %s, retrying in %.1fs", status, url, delay)
                time.sleep(delay)
                continue

            if require_206 and status == 200:
                self._bump(requests=1)
                self._bump(seconds=time.perf_counter() - t0)
                resp.close()  # never read the body: that is the whole point
                raise RangeUnsupported(
                    f"server answered 200 (whole file) to a Range request for {url}; "
                    "refusing to download the full shard"
                )
            if require_206 and status != 206:
                self._bump(requests=1)
                self._bump(seconds=time.perf_counter() - t0)
                resp.close()
                raise RemoteLoaderError(f"HTTP {status} for Range request {url}")

            if status >= 400:
                self._bump(requests=1)
                self._bump(seconds=time.perf_counter() - t0)
                resp.close()
                raise RemoteLoaderError(f"HTTP {status} for {url}")

            if method.upper() == "HEAD":
                body = b""
                resp.close()
            else:
                body = resp.content
            self._bump(requests=1)
            self._bump(seconds=time.perf_counter() - t0)
            self._bump(bytes_read=len(body))
            self._note_content_range(url, resp)
            return resp

        raise RemoteLoaderError(f"request failed after retries: {url} ({last_exc})")

    def _note_content_range(self, url: str, resp: "requests.Response") -> None:
        """Learn a shard's total size for free from a ``Content-Range`` header.

        Claim: low-transfer -- the ``/NNN`` suffix of a 206 response gives us
        the denominator of the reduction factor without any extra request.
        """
        cr = resp.headers.get("Content-Range")
        if not cr or "/" not in cr:
            return
        tail = cr.rsplit("/", 1)[-1].strip()
        if not tail.isdigit():
            return
        fname = url.rsplit("/", 1)[-1]
        self._file_sizes.setdefault(fname, int(tail))

    # -- disk cache ---------------------------------------------------------

    def _cache_read(self, name: str) -> Optional[Any]:
        """Read a cached JSON blob, counting a hit in the transfer stats.

        Claim: low-transfer -- header caching is what makes comparing one model
        against many candidates cost a single header read.
        """
        path = self.cache_dir / name
        if not path.is_file():
            return None
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            LOG.debug("ignoring corrupt cache entry %s", path)
            return None
        self._bump(cache_hits=1)
        return obj

    def _cache_write(self, name: str, obj: Any) -> None:
        """Persist a JSON blob to the on-disk cache.

        Claim: low-transfer.
        """
        try:
            atomic_write_json(self.cache_dir / name, obj)
        except Exception as exc:  # pragma: no cover - cache is best-effort
            LOG.debug("could not write cache %s: %s", name, exc)

    # -- shard discovery ----------------------------------------------------

    def _resolve_files(self) -> List[str]:
        """Find the safetensors shard filenames for this checkpoint.

        Claim: low-transfer -- prefers the small ``index.json`` sidecar, then a
        single ``model.safetensors``, and only then a repo file listing; no
        payload is ever touched.
        """
        if self.local:
            return self._resolve_files_local()
        return self._resolve_files_remote()

    def _resolve_files_local(self) -> List[str]:
        """Enumerate local shards from ``index.json`` or a directory glob.

        Claim: low-transfer.
        """
        root = self.root
        assert root is not None
        if root.is_file() and root.suffix == ".safetensors":
            self.root = root.parent
            return [root.name]
        idx = root / INDEX_FILE
        if idx.is_file():
            try:
                data = json.loads(idx.read_text(encoding="utf-8"))
                self._bump(bytes_read=idx.stat().st_size)
                self._bump(requests=1)
                wm = data.get("weight_map", {})
                files = sorted({str(v) for v in wm.values()})
                if files:
                    return files
            except Exception:
                LOG.warning("could not parse %s", idx)
        single = root / SINGLE_FILE
        if single.is_file():
            return [SINGLE_FILE]
        files = sorted(p.name for p in root.glob("*.safetensors"))
        if not files:
            raise RemoteLoaderError(f"no *.safetensors found under {root}")
        return files

    def _resolve_files_remote(self) -> List[str]:
        """Enumerate Hub shards: index sidecar, single file, then repo listing.

        Claim: low-transfer -- three cheap probes replace cloning the repo.
        """
        key = stable_hash((self.ref, self.revision, "__shards__"))
        cached = self._cache_read(f"shards-{key}.json")
        if isinstance(cached, list) and cached:
            return [str(x) for x in cached]

        files: List[str] = []
        resp = self._request("GET", self._url(INDEX_FILE), allow_404=True)
        if resp is not None:
            try:
                data = json.loads(resp.content.decode("utf-8"))
                wm = data.get("weight_map", {}) or {}
                files = sorted({str(v) for v in wm.values()})
            except Exception:
                LOG.warning("could not parse %s for %s", INDEX_FILE, self.ref)

        if not files:
            head = self._request("HEAD", self._url(SINGLE_FILE), allow_404=True)
            if head is not None:
                size = _size_from_headers(head.headers)
                if size:
                    self._file_sizes[SINGLE_FILE] = size
                files = [SINGLE_FILE]

        if not files:
            files = self._list_repo_safetensors()

        if not files:
            raise RemoteLoaderError(
                f"no *.safetensors found in {self.ref!r} at revision {self.revision!r}"
            )
        self._cache_write(f"shards-{key}.json", files)
        return files

    def _list_repo_safetensors(self) -> List[str]:
        """Last-resort shard discovery via ``huggingface_hub.list_repo_files``.

        Claim: low-transfer -- a metadata-only API call, imported lazily so the
        package still works without ``huggingface_hub`` installed.
        """
        try:
            from huggingface_hub import list_repo_files  # lazy: optional at import time
        except Exception as exc:
            LOG.debug("huggingface_hub unavailable: %s", exc)
            return []
        try:
            t0 = time.perf_counter()
            names = list_repo_files(self.ref, revision=self.revision, token=self.token)
            self._bump(requests=1)
            self._bump(seconds=time.perf_counter() - t0)
        except Exception as exc:
            LOG.debug("list_repo_files failed for %s: %s", self.ref, exc)
            return []
        return sorted(n for n in names if n.endswith(".safetensors"))

    # -- sizes --------------------------------------------------------------

    def _file_size(self, filename: str) -> int:
        """Size of one shard, from the filesystem or a ``HEAD`` request.

        Claim: low-transfer -- never obtained by downloading.
        """
        if filename in self._file_sizes:
            return self._file_sizes[filename]
        if self.local:
            assert self.root is not None
            path = self.root / filename
            size = int(path.stat().st_size) if path.is_file() else 0
            self._file_sizes[filename] = size
            return size
        resp = self._request("HEAD", self._url(filename), allow_404=True)
        if resp is None:
            return 0
        size = _size_from_headers(resp.headers)
        if not size:
            # Fall back to a 1-byte Range read whose Content-Range carries the total.
            r = self._request(
                "GET",
                self._url(filename),
                headers={"Range": "bytes=0-0"},
                require_206=True,
            )
            if r is not None:
                size = self._file_sizes.get(filename, 0)
        if size:
            self._file_sizes[filename] = int(size)
        return int(size or 0)

    # -- header reading -----------------------------------------------------

    def _load_header(self, filename: str) -> Tuple[Dict[str, Any], int]:
        """Parse one shard's safetensors header, using the on-disk cache.

        Claim: low-transfer -- exactly two small Range reads (8 bytes, then N
        bytes of JSON) buy the byte map of an entire multi-gigabyte shard.

        Returns ``(header, file_size)`` where ``header`` additionally carries the
        synthetic key ``"__stemma_payload_start__"`` == ``8 + N``.
        """
        key = stable_hash((self.ref, self.revision, filename))
        cache_name = f"header-{key}.json"
        if not self.local:
            cached = self._cache_read(cache_name)
            if isinstance(cached, dict) and "header" in cached:
                header = cached["header"]
                header["__stemma_payload_start__"] = int(cached.get("payload_start", 0))
                return header, int(cached.get("file_size", 0) or 0)

        if self.local:
            mm = self._mmap(filename)
            self._bump(bytes_read=_HEADER_LEN_BYTES)
            self._bump(requests=1)
            n = struct.unpack("<Q", bytes(mm[0:_HEADER_LEN_BYTES]))[0]
            _check_header_len(n, filename)
            raw = bytes(mm[_HEADER_LEN_BYTES : _HEADER_LEN_BYTES + n])
            self._bump(bytes_read=len(raw))
            self._bump(requests=1)
            file_size = len(mm)
        else:
            r0 = self._request(
                "GET",
                self._url(filename),
                headers={"Range": f"bytes=0-{_HEADER_LEN_BYTES - 1}"},
                require_206=True,
            )
            assert r0 is not None
            prefix = r0.content
            if len(prefix) < _HEADER_LEN_BYTES:
                raise RemoteLoaderError(f"short header prefix for {filename}")
            n = struct.unpack("<Q", prefix[:_HEADER_LEN_BYTES])[0]
            _check_header_len(n, filename)
            r1 = self._request(
                "GET",
                self._url(filename),
                headers={"Range": f"bytes={_HEADER_LEN_BYTES}-{_HEADER_LEN_BYTES + n - 1}"},
                require_206=True,
            )
            assert r1 is not None
            raw = r1.content
            file_size = int(self._file_sizes.get(filename, 0))

        try:
            header = json.loads(raw.decode("utf-8"))
        except Exception as exc:
            raise RemoteLoaderError(f"invalid safetensors header in {filename}: {exc}") from exc
        if not isinstance(header, dict):
            raise RemoteLoaderError(f"invalid safetensors header in {filename}")

        payload_start = _HEADER_LEN_BYTES + int(n)
        if not self.local:
            self._cache_write(
                cache_name,
                {"header": header, "payload_start": payload_start, "file_size": file_size},
            )
        header["__stemma_payload_start__"] = payload_start
        return header, int(file_size)

    # -- payload reading ----------------------------------------------------

    def _mmap(self, filename: str) -> mmap.mmap:
        """Open (and memoise) a read-only mmap of a local shard.

        Claim: low-transfer -- mmap means the local path touches only the pages
        of the rows we slice, so local and remote byte accounting are comparable.
        """
        got = self._mmaps.get(filename)
        if got is not None:
            return got[1]
        assert self.root is not None
        path = self.root / filename
        fh = open(path, "rb")
        mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        self._mmaps[filename] = (fh, mm)
        return mm

    def _read_ranges(self, filename: str, ranges: Sequence[Tuple[int, int]]) -> bytes:
        """Read and concatenate half-open absolute byte ranges of one shard.

        Claim: low-transfer -- this is the only place payload bytes are ever
        touched, remote or local, so ``stats.bytes_read`` is exact by
        construction.
        """
        wanted = [(s, e) for s, e in ranges if e > s]
        if not wanted:
            return b""

        if self.local:
            chunks: List[bytes] = []
            mm = self._mmap(filename)
            for start, end in wanted:
                data = bytes(mm[start:end])
                self._bump(bytes_read=len(data))
                self._bump(requests=1)
                chunks.append(data)
            return b"".join(chunks)

        # Remote. Latency, not bandwidth, is the binding constraint: measured
        # against the Hub a Range request costs ~0.87 s, essentially all of it
        # resolving the 302 from /resolve/ to a signed CDN URL. That URL cannot
        # be cached and replayed -- reusing it returns 403 (verified: 0/6
        # succeeded) -- so every request pays the redirect again. The cost is
        # pure round-trip latency, which parallelises almost perfectly, so the
        # runs of one tensor are fetched concurrently. Bytes read are unchanged;
        # only wall-clock improves.
        url = self._url(filename)

        def _fetch(rng: Tuple[int, int]) -> bytes:
            start, end = rng
            resp = self._request(
                "GET",
                url,
                headers={"Range": f"bytes={start}-{end - 1}"},
                require_206=True,
            )
            assert resp is not None
            data = resp.content
            if len(data) != end - start:
                raise RemoteLoaderError(
                    f"short Range read for {filename}: asked {end - start} bytes, "
                    f"got {len(data)}"
                )
            return data

        if len(wanted) == 1 or self.max_workers <= 1:
            return b"".join(_fetch(r) for r in wanted)

        from concurrent.futures import ThreadPoolExecutor

        workers = min(int(self.max_workers), len(wanted))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            # executor.map preserves input order, so the concatenation stays
            # byte-identical to the sequential path.
            return b"".join(pool.map(_fetch, wanted))


# --------------------------------------------------------------------------- #
# Module-level helpers
# --------------------------------------------------------------------------- #


def _finalize(arr: np.ndarray, dtype: Any) -> np.ndarray:
    """Cast to the requested dtype and guarantee a writable, owned array.

    Claim: infra -- ``np.frombuffer`` views are read-only and alias a temporary
    buffer; downstream sketch/direction code normalises matrices in place, so we
    hand back something it can safely write to.
    """
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    if not arr.flags.writeable or arr.base is not None:
        arr = np.array(arr, copy=True)
    return arr


def _token_from_env() -> Optional[str]:
    """Pick up a Hub token from the usual environment variables.

    Claim: infra -- gated repos would otherwise 401 in the middle of a benchmark.
    """
    for var in ("STEMMA_HF_TOKEN", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        val = os.environ.get(var)
        if val:
            return val.strip()
    return None


def _retry_after(resp: "requests.Response", fallback: float) -> float:
    """Honour a ``Retry-After`` header, clamped to something sane.

    Claim: infra -- being polite to the Hub keeps long benchmark sweeps alive.
    """
    raw = resp.headers.get("Retry-After")
    if raw:
        try:
            return max(0.0, min(30.0, float(raw.strip())))
        except ValueError:
            pass
    return fallback


def _size_from_headers(headers: Any) -> int:
    """Extract a file size from HF/CDN response headers.

    Claim: low-transfer -- the Hub reports the real (LFS) size in
    ``x-linked-size``; ``Content-Length`` is the fallback.
    """
    for key in ("x-linked-size", "X-Linked-Size", "Content-Length", "content-length"):
        val = headers.get(key)
        if val:
            try:
                n = int(str(val).strip())
            except ValueError:
                continue
            if n > 0:
                return n
    return 0


def _check_header_len(n: int, filename: str) -> None:
    """Reject an implausible safetensors header length.

    Claim: infra -- a corrupt or non-safetensors file would otherwise make us
    request a gigantic range, which is precisely what this project forbids.
    """
    if n <= 0 or n > _MAX_HEADER_BYTES:
        raise RemoteLoaderError(f"implausible safetensors header length {n} in {filename}")


def open_model(ref: ModelRef, **kw: Any) -> SafeTensorsSource:
    """Open a checkpoint (Hub repo id or local directory) for selective reads.

    Claim: low-transfer -- the one entry point every other Stemma module uses,
    so no module can accidentally take a non-Range path to the weights.
    """
    return SafeTensorsSource(ref, **kw)


def measure_full_download_bytes(ref: ModelRef, **kw: Any) -> int:
    """How many bytes a naive full download of ``ref`` would have cost.

    Claim: low-transfer -- this is the denominator of the headline reduction
    factor, and it is obtained from headers/HEAD only: calling it downloads no
    payload whatsoever.
    """
    with SafeTensorsSource(ref, **kw) as src:
        return int(src.total_size())


__all__ = [
    "COALESCE_GAP",
    "RemoteLoaderError",
    "UnsupportedDtype",
    "RangeUnsupported",
    "SafeTensorsSource",
    "coalesce_ranges",
    "decode_buffer",
    "f8_table",
    "itemsize_of",
    "measure_full_download_bytes",
    "open_model",
    "select_rows",
]
