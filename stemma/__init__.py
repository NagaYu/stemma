"""Stemma -- model provenance from weights alone.

Stemma recovers derivation *direction*, *multi-parent merges* and *mixing
ratios* between neural-network checkpoints by reading a few megabytes of
safetensors payload over HTTP Range requests, and reports the result as an
AI Bill of Materials with per-edge confidence.

Claim: infra -- this package root binds the four headline claims together
(direction, merge-recovery, low-transfer, low-false-positive) behind one import
surface, and it does so *lazily* so that a missing optional dependency (faiss,
gradio, graphviz, datasets) can never break ``import stemma``.

Everything below is resolved on first attribute access via :pep:`562`
``__getattr__``; importing this module touches nothing but the standard library.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, List

__version__ = "0.1.0"

#: Public name -> module it lives in. Resolution is deferred until the name is
#: actually touched, which is what keeps ``import stemma`` dependency-free.
_EXPORTS: Dict[str, str] = {
    # --- shared wire-format types (stemma.types only needs numpy) ------------
    "Sketch": ".types",
    "DirectionVerdict": ".types",
    "MergeDecomposition": ".types",
    "Phylogeny": ".types",
    "BOM": ".types",
    # --- the five entry points ----------------------------------------------
    "sketch_model": ".sketch",
    "estimate_direction": ".direction",
    "decompose_merge": ".merge_decompose",
    "build_phylogeny": ".phylogeny",
    "trace": ".phylogeny",
    "open_model": ".remote_loader",
}

__all__: List[str] = ["__version__", *sorted(_EXPORTS)]


def __getattr__(name: str) -> Any:
    """Resolve a public Stemma symbol on first use.

    Claim: infra -- deferring every sibling import to attribute-access time is
    what lets the CLI render ``--help``, and the test suite collect, even while
    an optional backend (faiss/gradio/graphviz) is absent or a sibling module is
    still under construction.
    """
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        module = importlib.import_module(module_name, __name__)
    except Exception as exc:  # pragma: no cover - depends on install state
        raise AttributeError(
            f"stemma.{name} is unavailable: importing {module_name.lstrip('.')!r} "
            f"failed with {type(exc).__name__}: {exc}"
        ) from exc
    value = getattr(module, name)
    globals()[name] = value  # cache so the next access is a plain global lookup
    return value


def __dir__() -> List[str]:
    """List the lazily-exported names alongside the already-resolved globals.

    Claim: infra -- keeps tab-completion and ``help(stemma)`` honest even though
    the symbols are not bound until first use.
    """
    return sorted(set(__all__) | set(_EXPORTS))
