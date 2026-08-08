"""Small shared helpers: logging, seeding, tensor-name parsing, byte formatting.

Claim: infra -- this module underpins all four claims but proves none by itself.
The one load-bearing piece is :func:`role_of` / :func:`layer_index_of`, which
put every architecture into the same (role, relative-depth) coordinate system so
that sketches from different model families are comparable at all.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
import tempfile
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .types import DEPTH_BUCKETS, ROLES

_LOG_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    """Return a package logger, configuring a single stderr handler once.

    Claim: infra.
    """
    global _LOG_CONFIGURED
    if not _LOG_CONFIGURED:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        root = logging.getLogger("stemma")
        root.addHandler(handler)
        root.setLevel(os.environ.get("STEMMA_LOG_LEVEL", "WARNING").upper())
        _LOG_CONFIGURED = True
    return logging.getLogger(name if name.startswith("stemma") else f"stemma.{name}")


def set_seed(seed: int) -> None:
    """Seed python/numpy (and torch when present) for reproducible sketches.

    Claim: infra.
    """
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    try:  # torch is optional for the pure-analysis path
        import torch

        torch.manual_seed(seed)
    except Exception:  # pragma: no cover - torch always present in dev env
        pass


def is_local_path(ref: str) -> bool:
    """True when ``ref`` points at a directory on disk rather than a Hub repo.

    Claim: infra -- lets the benchmark run identical code paths on locally
    generated ground-truth models and on real Hub repos.
    """
    if ref.startswith("file://"):
        return True
    p = Path(ref).expanduser()
    if p.exists():
        return True
    # "org/name" with no local counterpart is a Hub id.
    return os.sep in ref and Path(ref).parent.exists() and Path(ref).suffix == ""


def local_path_of(ref: str) -> Path:
    """Normalise a local model reference to a :class:`Path`.

    Claim: infra.
    """
    if ref.startswith("file://"):
        return Path(ref[len("file://") :]).expanduser()
    return Path(ref).expanduser()


# --------------------------------------------------------------------------- #
# Tensor-name -> (role, layer index) parsing
# --------------------------------------------------------------------------- #

_ROLE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    # --- embeddings ---------------------------------------------------------
    ("embed", re.compile(r"(embed_tokens|\bwte\b|word_embeddings|tok_embeddings|embeddings\.word)")),
    # --- output head (checked before generic proj patterns) ------------------
    ("lm_head", re.compile(r"(lm_head|\boutput\.weight$|score\.weight$|embed_out)")),
    # --- attention ----------------------------------------------------------
    ("attn_q", re.compile(r"(q_proj|query|\bwq\b|attn\.c_attn|query_key_value|\bWq\b)")),
    ("attn_k", re.compile(r"(k_proj|\bkey\b|\bwk\b)")),
    ("attn_v", re.compile(r"(v_proj|\bvalue\b|\bwv\b)")),
    ("attn_o", re.compile(r"(o_proj|out_proj|\bwo\b|attn\.c_proj|dense\.weight)")),
    # --- mlp ----------------------------------------------------------------
    ("mlp_in", re.compile(r"(gate_proj|up_proj|\bw1\b|\bw3\b|mlp\.c_fc|fc_in|intermediate\.dense)")),
    ("mlp_out", re.compile(r"(down_proj|\bw2\b|mlp\.c_proj|fc_out|output\.dense)")),
    # --- norms (last: matches many names) -----------------------------------
    ("norm", re.compile(r"(layernorm|layer_norm|\bln_\d*\b|\bln_f\b|rmsnorm|\bnorm\.weight$)", re.I)),
)

_LAYER_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\.layers?\.(\d+)\."),
    re.compile(r"\.h\.(\d+)\."),
    re.compile(r"\.blocks?\.(\d+)\."),
    re.compile(r"\.layer\.(\d+)\."),
    re.compile(r"\btransformer\.(\d+)\."),
)


def role_of(tensor_name: str) -> Optional[str]:
    """Map a tensor name onto one of :data:`stemma.types.ROLES`.

    Claim: infra (enables low-transfer) -- role mapping is what lets the sketcher
    request six specific tensors instead of an entire checkpoint.

    Returns ``None`` for tensors we deliberately ignore (biases, rotary caches,
    position embeddings, buffers).
    """
    name = tensor_name
    if name.endswith(".bias"):
        return None
    if re.search(r"(rotary|inv_freq|masked_bias|attn\.bias|position_ids|wpe)", name):
        return None
    for role, pat in _ROLE_PATTERNS:
        if pat.search(name):
            return role
    return None


def layer_index_of(tensor_name: str) -> Optional[int]:
    """Extract the transformer block index from a tensor name, if any.

    Claim: infra.
    """
    for pat in _LAYER_PATTERNS:
        m = pat.search(tensor_name)
        if m:
            return int(m.group(1))
    return None


def relative_depth(idx: int, n_layers: int) -> float:
    """Map an absolute layer index to [0, 1].

    Claim: infra -- relative depth is what makes a 12-layer model comparable to
    a 24-layer one, which matters for the distillation and pruning cases.
    """
    if n_layers <= 1:
        return 0.0
    return float(np.clip(idx / (n_layers - 1), 0.0, 1.0))


def depth_bucket(rel_depth: float) -> int:
    """Index of the nearest entry in :data:`stemma.types.DEPTH_BUCKETS`.

    Claim: infra.
    """
    arr = np.asarray(DEPTH_BUCKETS, dtype=np.float64)
    return int(np.argmin(np.abs(arr - float(rel_depth))))


def slot_index(role: str, bucket: int) -> int:
    """Flat index of a (role, depth-bucket) slot in the sketch vector.

    Claim: infra -- freezes the sketch wire format.
    """
    return ROLES.index(role) * len(DEPTH_BUCKETS) + bucket


def human_bytes(n: float) -> str:
    """Format a byte count for CLI/benchmark output.

    Claim: low-transfer (presentation only).
    """
    n = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024.0:
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024.0
    return f"{n:.1f} PiB"


def stable_hash(obj: Any) -> str:
    """Deterministic short hash used for on-disk cache keys.

    Claim: infra.
    """
    payload = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def atomic_write_json(path: os.PathLike | str, obj: Any) -> None:
    """Write JSON via a temp file + rename so partial results never land.

    Claim: infra.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=2, ensure_ascii=False, default=_json_default)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def default_cache_dir() -> Path:
    """Where Stemma caches safetensors headers and model cards.

    Claim: low-transfer -- header caching removes repeat requests when the same
    model is compared against many candidates.
    """
    d = Path(os.environ.get("STEMMA_CACHE", Path.home() / ".cache" / "stemma"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def short_id(ref: str, width: int = 34) -> str:
    """Shorten a model reference for table/graph labels.

    Claim: infra.
    """
    ref = str(ref)
    if len(ref) <= width:
        return ref
    return ref[: width - 3] + "..."
