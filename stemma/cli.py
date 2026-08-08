"""Stemma command-line interface.

The headline command is ``stemma trace org/model``: it builds the lineage DAG
for one model against a candidate universe, propagates license facts along the
edges, flags conflicts, writes an AI-BOM and prints how few bytes it had to read
to get there.

Claim: infra -- the CLI proves nothing on its own; it is the surface through
which the direction, merge-recovery, low-transfer and low-false-positive claims
are exercised and reported. Every heavy sibling module is imported *inside* the
subcommand handler that needs it, so ``--help`` renders even when a sibling is
missing, broken or mid-rewrite.

Exit codes (documented contract, relied on by CI and by ``benchmarks/run.py``):

===== ==========================================================================
``0`` success -- the command completed and no rights conflict was detected
``1`` error   -- bad arguments, unreadable model, or an unhandled failure
``2`` conflicts -- ``trace`` completed *successfully* but found at least one
      :class:`~stemma.types.RightsConflict`. This is not an error: the run is
      valid and the BOM was written. It is a distinct code so that a pipeline
      can gate a release on "no license conflicts" without parsing stdout.
===== ==========================================================================
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import textwrap
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .types import TransferStats
from .utils import atomic_write_json, get_logger, human_bytes, short_id

LOG = get_logger(__name__)

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_CONFLICTS = 2

#: Printed verbatim at the end of every ``trace`` run. Wording is deliberately
#: evidential: Stemma never asserts infringement.
DISCLAIMER = (
    "DISCLAIMER: Stemma reports STATISTICAL EVIDENCE about weight-level "
    "similarity and derivation direction, with a confidence attached to every "
    "edge. It does NOT establish provenance as fact and does NOT constitute a "
    "legal determination of license compliance or infringement. A human MUST "
    "review these findings before any action is taken."
)

#: Fallback candidate universe: small, public, quick-to-range-read checkpoints
#: spanning several families so a trace has something to work with out of the
#: box. Overridden by ``--universe`` or by a local ``bench_models/``.
DEFAULT_UNIVERSE: Tuple[str, ...] = (
    "hf-internal-testing/tiny-random-gpt2",
    "sshleifer/tiny-gpt2",
    "openai-community/gpt2",
    "openai-community/gpt2-medium",
    "distilbert/distilgpt2",
    "EleutherAI/pythia-70m",
    "EleutherAI/pythia-160m",
    "HuggingFaceTB/SmolLM2-135M",
    "HuggingFaceTB/SmolLM2-135M-Instruct",
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
)

_RULE = "-" * 72


# --------------------------------------------------------------------------- #
# Universe loading
# --------------------------------------------------------------------------- #


def parse_universe_text(text: str) -> List[str]:
    """Parse a candidate universe from free text (one model ref per line).

    Claim: low-false-positive -- the size and composition of the candidate
    universe is the knob that decides how many unrelated models a trace has the
    opportunity to wrongly attach, so it is explicit and user-controlled rather
    than implicit.

    Blank lines and ``#`` comments are ignored; a JSON document of the form
    ``{"models": [...]}`` (or a bare JSON list) is also accepted.
    """
    text = (text or "").strip()
    if not text:
        return []
    if text[0] in "[{":
        try:
            return _refs_from_json(json.loads(text))
        except json.JSONDecodeError:
            LOG.warning("universe text looked like JSON but did not parse; treating as lines")
    out: List[str] = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip().strip(",").strip('"').strip("'")
        if line:
            out.append(line)
    return _dedupe(out)


def _refs_from_json(doc: Any) -> List[str]:
    """Pull model references out of a parsed JSON universe / ground-truth doc.

    Claim: infra -- accepts both the hand-written ``{"models": ["a", "b"]}``
    shape and the benchmark's richer ``ground_truth.json`` records so the same
    file can drive both the CLI and the harness.
    """
    if isinstance(doc, dict):
        entries = doc.get("models", doc.get("universe", []))
    else:
        entries = doc
    if not isinstance(entries, (list, tuple)):
        return []
    refs: List[str] = []
    for entry in entries:
        if isinstance(entry, str):
            refs.append(entry)
        elif isinstance(entry, dict):
            # Prefer a real on-disk path (the benchmark writes both).
            ref = entry.get("path") or entry.get("id") or entry.get("model_id")
            if isinstance(ref, str) and ref:
                refs.append(ref)
    return _dedupe(refs)


def load_universe(path: Optional[str] = None, *, include_defaults: bool = True) -> List[str]:
    """Resolve the candidate universe for a trace or an index build.

    Claim: low-false-positive -- a trace can only be as honest as the negative
    examples it is given, so the universe defaults to a *mixed-family* list of
    small public models plus any locally generated ground-truth models rather
    than to the target's own family.

    Resolution order: ``--universe`` file when given; otherwise a local
    ``bench_models/ground_truth.json`` (if present) merged with
    :data:`DEFAULT_UNIVERSE`.
    """
    if path:
        p = Path(path).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"universe file not found: {p}")
        text = p.read_text(encoding="utf-8")
        refs = parse_universe_text(text)
        if not refs:
            raise ValueError(f"universe file {p} contained no model references")
        return refs

    refs: List[str] = []
    gt = _find_ground_truth()
    if gt is not None:
        try:
            refs.extend(_refs_from_json(json.loads(gt.read_text(encoding="utf-8"))))
            LOG.info("universe seeded from %s (%d models)", gt, len(refs))
        except Exception as exc:  # pragma: no cover - corrupt local file
            LOG.warning("could not read %s: %s", gt, exc)
    if include_defaults or not refs:
        refs.extend(DEFAULT_UNIVERSE)
    return _dedupe(refs)


def _find_ground_truth() -> Optional[Path]:
    """Locate a locally generated ``bench_models/ground_truth.json``, if any.

    Claim: infra -- lets ``stemma trace`` work fully offline against the
    benchmark's synthetic lineages with no extra flags.
    """
    candidates = [
        Path.cwd() / "bench_models" / "ground_truth.json",
        _repo_root() / "bench_models" / "ground_truth.json",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def _repo_root() -> Path:
    """Directory that contains the ``stemma`` package (the source checkout).

    Claim: infra.
    """
    return Path(__file__).resolve().parent.parent


def _dedupe(items: Sequence[str]) -> List[str]:
    """Order-preserving de-duplication of model references.

    Claim: infra -- a duplicated candidate would double-count as evidence.
    """
    seen: set[str] = set()
    out: List[str] = []
    for it in items:
        if it not in seen:
            seen.add(it)
            out.append(it)
    return out


# --------------------------------------------------------------------------- #
# Small formatting helpers (stdlib only -- no rich dependency)
# --------------------------------------------------------------------------- #


def _emit(line: str = "") -> None:
    """Write one line to stdout.

    Claim: infra -- single funnel for CLI output so ``--json`` can keep stdout
    machine-clean by routing prose to stderr instead.
    """
    print(line)


def _emit_err(line: str = "") -> None:
    """Write one line to stderr.

    Claim: infra.
    """
    print(line, file=sys.stderr)


def _wrap(text: str, width: int = 72, indent: str = "") -> str:
    """Hard-wrap prose for terminal output.

    Claim: infra.
    """
    return textwrap.fill(text, width=width, initial_indent=indent, subsequent_indent=indent)


def _print_disclaimer(stream_err: bool = False) -> None:
    """Print the mandatory human-review disclaimer.

    Claim: infra (safety) -- the project spec requires this on every ``trace``
    run; it is what keeps a statistical result from being read as a legal one.
    """
    out = _emit_err if stream_err else _emit
    out("")
    out(_RULE)
    for line in _wrap(DISCLAIMER).splitlines():
        out(line)
    out(_RULE)


def _coerce_stats(obj: Any) -> Optional[TransferStats]:
    """Normalise whatever a sibling attached as transfer accounting.

    Claim: low-transfer -- the headline "read X of Y" line must be computed from
    real byte counters, so accept a :class:`TransferStats`, a plain dict, or a
    dataclass and reject anything else rather than fabricating numbers.
    """
    if obj is None:
        return None
    if isinstance(obj, TransferStats):
        return obj
    if is_dataclass(obj) and not isinstance(obj, type):
        obj = asdict(obj)
    if isinstance(obj, dict):
        fields = {"bytes_read", "requests", "seconds", "full_size_bytes", "cache_hits"}
        kw = {k: v for k, v in obj.items() if k in fields}
        if kw:
            try:
                return TransferStats(**kw)
            except Exception:  # pragma: no cover - defensive
                return None
    return None


def _stats_from_phylogeny(p: Any) -> Optional[TransferStats]:
    """Dig the aggregate :class:`TransferStats` out of a phylogeny's metadata.

    Claim: low-transfer -- the DAG builder is the only component that knows how
    many models it touched, so the byte total is read back from its ``meta``.
    """
    meta = getattr(p, "meta", None)
    if not isinstance(meta, dict):
        return None
    for key in ("transfer", "stats", "transfer_stats"):
        st = _coerce_stats(meta.get(key))
        if st is not None:
            return st
    return None


def _transfer_line(stats: Optional[TransferStats], seconds: float) -> str:
    """Render the low-transfer headline: read X of Y (Nx less) in T s.

    Claim: low-transfer -- this single line is the user-facing form of the
    claim that Stemma answers a provenance question without downloading the
    checkpoints.
    """
    if stats is None or stats.bytes_read <= 0:
        return f"read (byte accounting unavailable) in {seconds:.1f} s"
    read = human_bytes(stats.bytes_read)
    if stats.full_size_bytes > 0:
        factor = stats.full_size_bytes / max(stats.bytes_read, 1)
        return (
            f"read {read} of {human_bytes(stats.full_size_bytes)} "
            f"({factor:.0f}x less than a full download) in {seconds:.1f} s "
            f"[{stats.requests} requests]"
        )
    return f"read {read} in {seconds:.1f} s [{stats.requests} requests]"


def _ancestor_tree_lines(p: Any, root: str, *, max_depth: int = 16) -> List[str]:
    """Render the ancestors of ``root`` as an ASCII tree with per-edge confidence.

    Claim: direction -- the tree only exists because every edge carries an
    orientation; a symmetric fingerprint could produce the node set but never
    the parent-of arrows or the confidence attached to them.
    """
    lines = [short_id(root, 60)]
    try:
        parents_of: Callable[[str], List[Any]] = p.parents_of
    except AttributeError:  # pragma: no cover - defensive
        return lines

    def walk(node: str, prefix: str, depth: int, path: Tuple[str, ...]) -> None:
        if depth >= max_depth:
            lines.append(prefix + "`-- ... (depth limit)")
            return
        edges = sorted(parents_of(node), key=lambda e: -float(getattr(e, "confidence", 0.0)))
        for i, e in enumerate(edges):
            last = i == len(edges) - 1
            conn = "`-- " if last else "|-- "
            pad = "    " if last else "|   "
            parent = str(getattr(e, "parent", "?"))
            bits = [str(getattr(e, "relation", "derived"))]
            bits.append(f"conf {float(getattr(e, 'confidence', 0.0)):.2f}")
            w = getattr(e, "weight", None)
            if w is not None:
                bits.append(f"mix {float(w):.2f}")
            lines.append(f"{prefix}{conn}{short_id(parent, 48)}  [{', '.join(bits)}]")
            for ev in list(getattr(e, "evidence", []) or [])[:2]:
                lines.append(f"{prefix}{pad}  . {short_id(str(ev), 66)}")
            if parent in path:
                lines.append(f"{prefix}{pad}  (cycle -- not expanded)")
                continue
            walk(parent, prefix + pad, depth + 1, path + (parent,))

    walk(root, "", 0, (root,))
    if len(lines) == 1:
        lines.append("`-- (no ancestors above the relatedness threshold)")
    return lines


def _conflict_lines(conflicts: Sequence[Any]) -> List[str]:
    """Render detected rights conflicts, severity first.

    Claim: low-false-positive -- each conflict is printed with the confidence of
    the *edge chain* that produced it, so a low-confidence lineage cannot masquerade
    as a hard license finding.
    """
    if not conflicts:
        return ["  none detected"]
    order = {"high": 0, "warning": 1, "info": 2}
    ranked = sorted(
        conflicts, key=lambda c: (order.get(str(getattr(c, "severity", "info")), 3), -float(getattr(c, "confidence", 0.0)))
    )
    out: List[str] = []
    for c in ranked:
        sev = str(getattr(c, "severity", "info")).upper()
        kind = str(getattr(c, "kind", "unknown"))
        conf = float(getattr(c, "confidence", 0.0))
        out.append(f"  [{sev:<7}] {kind}  (confidence {conf:.2f})")
        msg = str(getattr(c, "message", "")).strip()
        if msg:
            out.extend(_wrap(msg, width=66, indent="            ").splitlines())
        path = list(getattr(c, "path", []) or [])
        if path:
            out.append("            path: " + " -> ".join(short_id(x, 28) for x in path))
    return out


def _json_dump(obj: Any) -> str:
    """Serialise a CLI result for ``--json``.

    Claim: infra -- machine-readable output is what lets the benchmark and any
    downstream policy engine consume Stemma without screen-scraping.
    """
    import numpy as np

    def default(o: Any) -> Any:
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if is_dataclass(o) and not isinstance(o, type):
            return asdict(o)
        if isinstance(o, Path):
            return str(o)
        return str(o)

    return json.dumps(obj, indent=2, ensure_ascii=False, default=default)


def _default_out_path(fmt: str) -> str:
    """Default ``--out`` filename for each ``--format``.

    Claim: infra.
    """
    return {
        "json": "bom.json",
        "spdx": "bom.spdx.json",
        "mermaid": "phylogeny.mmd",
        "dot": "phylogeny.dot",
    }.get(fmt, "bom.json")


def _call_flexible(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Call ``fn`` dropping keyword arguments it does not accept.

    Claim: infra -- sibling modules are written concurrently against the same
    contract; tolerating an absent optional keyword (e.g. ``offline``) keeps the
    CLI usable rather than turning a signature nit into a crash.
    """
    import inspect

    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):  # pragma: no cover - builtins
        return fn(*args, **kwargs)
    params = sig.parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return fn(*args, **kwargs)
    accepted = {k: v for k, v in kwargs.items() if k in params}
    dropped = sorted(set(kwargs) - set(accepted))
    if dropped:
        LOG.debug("dropping unsupported kwargs %s for %s", dropped, getattr(fn, "__name__", fn))
    return fn(*args, **accepted)


# --------------------------------------------------------------------------- #
# Subcommand: trace
# --------------------------------------------------------------------------- #


def cmd_trace(args: argparse.Namespace) -> int:
    """Build a model's lineage, propagate rights, write the AI-BOM, report bytes.

    Claim: direction -- ``trace`` is the headline command and its whole output
    (an oriented ancestor tree, merge mixing weights, and license conflicts that
    only make sense along a direction) is unreachable for a symmetric similarity
    score.

    Returns :data:`EXIT_CONFLICTS` when the lineage is valid but at least one
    rights conflict was found, :data:`EXIT_OK` otherwise.
    """
    from .phylogeny import to_graphviz_dot, to_mermaid, trace as run_trace
    from .rights import bom_to_spdx, build_bom, detect_conflicts, fetch_license_facts, propagate

    target = args.model
    universe = load_universe(args.universe)
    universe = _dedupe([*universe, target])

    if not args.json:
        _emit(f"stemma trace  target={target}")
        _emit(f"  universe: {len(universe)} candidate models"
              f"{' (offline)' if args.offline else ''}")
        _emit("")

    t0 = time.time()
    # CONTRACT-CONCERN: `phylogeny.trace(target, universe, **kw)` is the only
    # place --offline can be expressed, but **kw is also forwarded to the loader.
    # Pass it only when it is actually requested, so the default path never
    # depends on a sibling accepting the keyword.
    trace_kw: Dict[str, Any] = {"k": args.k}
    if args.offline:
        trace_kw["offline"] = True
    phylo = _call_flexible(run_trace, target, universe, **trace_kw)
    nodes = list(getattr(phylo, "nodes", []) or [])

    facts: Dict[str, Any] = {}
    for node in nodes:
        try:
            facts[node] = _call_flexible(fetch_license_facts, node, offline=args.offline)
        except Exception as exc:
            LOG.warning("license lookup failed for %s: %s", node, exc)
    try:
        facts = propagate(phylo, facts) or facts
    except Exception as exc:
        LOG.warning("rights propagation failed: %s", exc)
    try:
        conflicts = list(detect_conflicts(phylo, facts) or [])
    except Exception as exc:
        LOG.warning("conflict detection failed: %s", exc)
        conflicts = []

    stats = _stats_from_phylogeny(phylo)
    elapsed = time.time() - t0

    bom = _call_flexible(
        build_bom, phylo, facts, conflicts, root=target, transfer=stats, sketches=None
    )

    out_path = Path(args.out) if args.out else Path(_default_out_path(args.format))
    if args.format == "json":
        payload: Any = json.loads(bom.to_json())
        atomic_write_json(out_path, payload)
        rendered = None
    elif args.format == "spdx":
        payload = bom_to_spdx(bom)
        atomic_write_json(out_path, payload)
        rendered = None
    elif args.format == "mermaid":
        rendered = to_mermaid(phylo, conflicts)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(rendered + "\n", encoding="utf-8")
    else:  # dot
        rendered = to_graphviz_dot(phylo, conflicts)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(rendered + "\n", encoding="utf-8")

    code = EXIT_CONFLICTS if conflicts else EXIT_OK

    if args.json:
        _emit(
            _json_dump(
                {
                    "target": target,
                    "universe_size": len(universe),
                    "bom": json.loads(bom.to_json()),
                    "nodes": nodes,
                    "edges": [asdict(e) for e in getattr(phylo, "edges", [])],
                    "conflicts": [asdict(c) for c in conflicts],
                    "transfer": asdict(stats) if stats else None,
                    "seconds": round(elapsed, 3),
                    "output_path": str(out_path),
                    "format": args.format,
                    "exit_code": code,
                    "disclaimer": DISCLAIMER,
                }
            )
        )
        _print_disclaimer(stream_err=True)
        return code

    _emit("Lineage (ancestors of the target, most confident first)")
    for line in _ancestor_tree_lines(phylo, target):
        _emit("  " + line)
    _emit("")

    roots = list(getattr(phylo, "root_candidates", []) or [])
    if roots:
        _emit("Root candidates: " + ", ".join(short_id(r, 40) for r in roots))
        _emit("")

    merge_edges = [
        e for e in getattr(phylo, "edges", []) if getattr(e, "weight", None) is not None
    ]
    if merge_edges:
        _emit("Recovered mixing ratios")
        for e in sorted(merge_edges, key=lambda e: (str(e.child), -float(e.weight or 0.0))):
            _emit(
                f"  {short_id(str(e.child), 34)} <- {short_id(str(e.parent), 34)}"
                f"  w={float(e.weight or 0.0):.3f}  conf={float(e.confidence):.2f}"
            )
        _emit("")

    _emit(f"License conflicts ({len(conflicts)})")
    for line in _conflict_lines(conflicts):
        _emit(line)
    _emit("")

    if rendered is not None and not args.out:
        _emit(rendered)
        _emit("")

    _emit(f"AI-BOM ({args.format}) written to {out_path}")
    _emit(_transfer_line(stats, elapsed))
    _print_disclaimer()
    if conflicts:
        _emit("")
        _emit(f"exit code {EXIT_CONFLICTS}: {len(conflicts)} rights conflict(s) require review")
    return code


# --------------------------------------------------------------------------- #
# Subcommand: sketch
# --------------------------------------------------------------------------- #


def cmd_sketch(args: argparse.Namespace) -> int:
    """Compute and optionally persist one model's invariant sketch.

    Claim: low-transfer -- prints the bytes actually read against the full
    checkpoint size, which is the per-model form of the low-transfer claim.
    """
    from .sketch import sketch_model

    t0 = time.time()
    sk = _call_flexible(
        sketch_model, args.ref, max_rows=args.max_rows, k=args.k, seed=args.seed
    )
    elapsed = time.time() - t0
    stats = _coerce_stats(getattr(sk, "stats", None))

    import numpy as np

    present = np.asarray(sk.present, dtype=bool)
    payload = sk.to_json()

    if args.out:
        atomic_write_json(args.out, payload)

    if args.json:
        _emit(_json_dump({"sketch": payload, "seconds": round(elapsed, 3),
                          "out": args.out, "slots_present": int(present.sum())}))
        return EXIT_OK

    _emit(f"sketch  {sk.model_id}   version={sk.version}")
    _emit(f"  dim={np.asarray(sk.vector).size}  slots present={int(present.sum())}/{present.size}")
    for key in ("architectures", "n_layers", "hidden_size", "vocab_size",
                "tensor_count", "param_count", "fused_qkv", "dtypes"):
        if key in (sk.meta or {}):
            _emit(f"  {key:<13} {sk.meta[key]}")
    _emit("  " + _transfer_line(stats, elapsed))
    if args.out:
        _emit(f"  written to {args.out}")
    return EXIT_OK


# --------------------------------------------------------------------------- #
# Subcommand: direction
# --------------------------------------------------------------------------- #


def cmd_direction(args: argparse.Namespace) -> int:
    """Decide which of two models is the ancestor, with per-feature evidence.

    Claim: direction -- this is the claim in its purest form: an ordered verdict
    with a log-likelihood ratio, where every symmetric baseline is pinned at
    "unknown" by construction.
    """
    from .direction import estimate_direction

    t0 = time.time()
    verdict = _call_flexible(
        estimate_direction, args.a, args.b, abstain=args.abstain, seed=args.seed
    )
    elapsed = time.time() - t0
    stats = _coerce_stats(getattr(verdict, "stats", None))

    if args.json:
        _emit(_json_dump({"verdict": asdict(verdict), "seconds": round(elapsed, 3)}))
        return EXIT_OK

    arrow = {"a->b": f"{short_id(args.a, 30)} -> {short_id(args.b, 30)}",
             "b->a": f"{short_id(args.b, 30)} -> {short_id(args.a, 30)}"}.get(
        verdict.direction, "unknown (abstained)"
    )
    _emit(f"direction  A={args.a}   B={args.b}")
    _emit(f"  verdict   {verdict.direction}   [{arrow}]")
    _emit(f"  llr       {verdict.llr:+.3f}   P(A is parent)={verdict.p_a_parent:.3f}"
          f"   confidence={verdict.confidence:.3f}")
    contribs = dict(verdict.contributions or {})
    if contribs:
        _emit("  top contributing features")
        for name, val in sorted(contribs.items(), key=lambda kv: -abs(kv[1]))[:8]:
            feat = float((verdict.features or {}).get(name, 0.0))
            _emit(f"    {name:<24} raw={feat:+.4f}  contribution={val:+.4f}")
    if verdict.evidence:
        _emit("  evidence")
        for line in verdict.evidence[:12]:
            for wrapped in _wrap(str(line), width=66, indent="    . ").splitlines():
                _emit(wrapped)
    _emit("  " + _transfer_line(stats, elapsed))
    _emit("")
    _emit(_wrap("Note: a direction verdict is statistical evidence, not proof of "
                "derivation. Human review required."))
    return EXIT_OK


# --------------------------------------------------------------------------- #
# Subcommand: decompose
# --------------------------------------------------------------------------- #


def cmd_decompose(args: argparse.Namespace) -> int:
    """Decompose a merged child over candidate parents and report mixing ratios.

    Claim: merge-recovery -- recovers *which* candidates were merged and in
    *what proportion*, neither of which any pairwise similarity score can express.
    """
    from .merge_decompose import decompose_merge

    t0 = time.time()
    dec = _call_flexible(
        decompose_merge,
        args.child,
        list(args.candidates),
        base=args.base,
        l1=args.l1,
        sum_to_one=not args.no_sum_to_one,
        support_threshold=args.support_threshold,
        coords=args.coords,
        seed=args.seed,
    )
    elapsed = time.time() - t0
    stats = _coerce_stats(getattr(dec, "stats", None))

    if args.json:
        _emit(_json_dump({"decomposition": asdict(dec), "mixing": dec.as_dict(),
                          "seconds": round(elapsed, 3)}))
        return EXIT_OK

    _emit(f"decompose  child={args.child}")
    _emit(f"  base={dec.base if dec.base else '(none / raw weights)'}   method={dec.method}")
    _emit("  mixing coefficients")
    import numpy as np

    coefs = np.asarray(dec.coefficients, dtype=float).ravel()
    selected = set(dec.selected or [])
    for cand, w in sorted(zip(dec.candidates, coefs), key=lambda kv: -float(kv[1])):
        mark = "*" if cand in selected else " "
        bar = "#" * int(round(max(0.0, min(1.0, float(w))) * 40))
        _emit(f"   {mark} {short_id(str(cand), 34):<34} {float(w):6.3f} |{bar}")
    _emit(f"  selected parents: {', '.join(short_id(s, 30) for s in dec.selected) or '(none)'}")
    _emit(f"  residual={dec.residual:.4f}   r2={dec.r2:.4f}")
    _emit("  " + _transfer_line(stats, elapsed))
    _emit("  (* = coefficient survived the support threshold)")
    return EXIT_OK


# --------------------------------------------------------------------------- #
# Subcommand: index
# --------------------------------------------------------------------------- #


def cmd_index(args: argparse.Namespace) -> int:
    """Sketch a whole universe and persist a nearest-neighbour index.

    Claim: low-transfer -- the index is built from sketches, so a universe of N
    models costs N cheap Range reads once and then answers candidate-parent
    queries with no further network traffic at all.
    """
    from .phylogeny import SketchIndex
    from .sketch import sketch_model

    refs = load_universe(args.universe, include_defaults=args.universe is None)
    sketches = []
    failures: List[Tuple[str, str]] = []
    t0 = time.time()
    total = TransferStats()
    for i, ref in enumerate(refs, 1):
        try:
            sk = _call_flexible(sketch_model, ref, max_rows=args.max_rows, seed=args.seed)
        except Exception as exc:
            LOG.warning("sketching %s failed: %s", ref, exc)
            failures.append((ref, f"{type(exc).__name__}: {exc}"))
            if not args.json:
                _emit(f"  [{i}/{len(refs)}] {short_id(ref, 40)}  FAILED ({type(exc).__name__})")
            continue
        sketches.append(sk)
        st = _coerce_stats(getattr(sk, "stats", None))
        if st is not None:
            total = total.add(st)
        if not args.json:
            _emit(f"  [{i}/{len(refs)}] {short_id(ref, 40)}  ok")

    if not sketches:
        _emit_err("error: no model could be sketched; index not written")
        return EXIT_ERROR

    idx = SketchIndex()
    idx.add(sketches)
    idx.save(args.out)
    elapsed = time.time() - t0

    if args.json:
        _emit(_json_dump({"out": args.out, "backend": idx.backend, "indexed": len(sketches),
                          "failed": failures, "seconds": round(elapsed, 3),
                          "transfer": asdict(total)}))
        return EXIT_OK

    _emit("")
    _emit(f"index written to {args.out}   backend={idx.backend}   "
          f"{len(sketches)} sketches ({len(failures)} failed)")
    _emit(_transfer_line(total, elapsed))
    return EXIT_OK


# --------------------------------------------------------------------------- #
# Subcommand: bench
# --------------------------------------------------------------------------- #


def cmd_bench(args: argparse.Namespace) -> int:
    """Run the benchmark harness in a subprocess and forward its exit code.

    Claim: infra -- the harness is what actually measures direction accuracy,
    merge F1/MAE, AUC/FPR@95TPR and bytes-per-decision; the CLI only launches it
    so that all four claims have one reproducible entry point.
    """
    runner = _repo_root() / "benchmarks" / "run.py"
    if not runner.is_file():
        _emit_err(f"error: benchmark runner not found at {runner}")
        return EXIT_ERROR
    cmd = [sys.executable, str(runner)]
    if args.quick:
        cmd.append("--quick")
    if args.out_dir:
        cmd += ["--out-dir", args.out_dir]
    cmd += list(args.extra or [])
    if not args.json:
        _emit("running: " + " ".join(cmd))
    env = dict(os.environ)
    env.setdefault("PYTHONPATH", str(_repo_root()))
    try:
        proc = subprocess.run(cmd, cwd=str(_repo_root()), env=env, check=False)
    except OSError as exc:
        _emit_err(f"error: could not launch benchmark: {exc}")
        return EXIT_ERROR
    return EXIT_OK if proc.returncode == 0 else EXIT_ERROR


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    """Construct the full argparse tree without importing any analysis module.

    Claim: infra -- ``--help`` must render (and be testable) while sibling
    modules are still being written, so nothing heavier than argparse is touched
    here; every real import happens inside a ``cmd_*`` handler.
    """
    parser = argparse.ArgumentParser(
        prog="stemma",
        description=(
            "Stemma -- recover model provenance (direction, merges, mixing ratios) "
            "from weights alone, reading as few bytes as possible over HTTP Range."
        ),
        epilog=(
            "Exit codes: 0 = ok, 1 = error, 2 = trace succeeded but rights "
            "conflicts were found (review required). "
            "Stemma reports statistical evidence, never a legal determination."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"stemma {_version()}")
    parser.add_argument(
        "-v", "--verbose", action="count", default=0,
        help="increase log verbosity (-v = INFO, -vv = DEBUG)",
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    # --- trace --------------------------------------------------------------
    p_trace = sub.add_parser(
        "trace",
        help="build a model's lineage, propagate rights and write an AI-BOM",
        description=(
            "Trace the lineage of one model against a candidate universe: retrieve "
            "candidate parents from sketches, orient each surviving pair, decompose "
            "multi-parent merges into mixing ratios, propagate license facts along "
            "the resulting DAG, detect conflicts and write an AI Bill of Materials. "
            "Prints a transfer-stats line showing how much less than a full "
            "download was read."
        ),
        epilog="Exit code 2 means the trace succeeded but rights conflicts were found.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_trace.add_argument("model", metavar="REF",
                         help="target model: Hugging Face repo id (org/model) or local directory")
    p_trace.add_argument("--universe", metavar="FILE", default=None,
                         help="candidate universe: text file with one model ref per line, or a "
                              "JSON file {\"models\": [...]}. Defaults to bench_models/"
                              "ground_truth.json (if present) plus a built-in small-model list.")
    p_trace.add_argument("--out", metavar="PATH", default=None,
                         help="where to write the rendered output (default depends on --format: "
                              "bom.json / bom.spdx.json / phylogeny.mmd / phylogeny.dot)")
    p_trace.add_argument("--format", choices=("json", "spdx", "mermaid", "dot"), default="json",
                         help="output format for --out (default: json)")
    p_trace.add_argument("--offline", action="store_true",
                         help="never touch the network: use only cached headers, local paths and "
                              "declared license metadata")
    p_trace.add_argument("--k", type=int, default=10, metavar="N",
                         help="number of candidate parents to retrieve per node (default: 10)")
    p_trace.add_argument("--json", action="store_true",
                         help="emit a machine-readable JSON result on stdout "
                              "(the disclaimer then goes to stderr)")
    p_trace.set_defaults(func=cmd_trace)

    # --- sketch -------------------------------------------------------------
    p_sketch = sub.add_parser(
        "sketch", help="compute one model's permutation/scale-invariant sketch",
        description="Range-read a handful of tensors and emit the fixed-length invariant sketch.",
    )
    p_sketch.add_argument("ref", metavar="REF", help="Hugging Face repo id or local directory")
    p_sketch.add_argument("--out", metavar="FILE", default=None, help="write the sketch as JSON")
    p_sketch.add_argument("--max-rows", type=int, default=2048, dest="max_rows",
                          help="rows sub-sampled per tensor (default: 2048)")
    p_sketch.add_argument("--k", type=int, default=64, help="singular values per tensor (default: 64)")
    p_sketch.add_argument("--seed", type=int, default=0, help="randomized-SVD seed (default: 0)")
    p_sketch.add_argument("--json", action="store_true", help="machine-readable output")
    p_sketch.set_defaults(func=cmd_sketch)

    # --- direction ----------------------------------------------------------
    p_dir = sub.add_parser(
        "direction", help="decide which of two models is the ancestor",
        description="Collect asymmetric evidence for A and B and report a signed llr verdict.",
    )
    p_dir.add_argument("a", metavar="A", help="first model ref")
    p_dir.add_argument("b", metavar="B", help="second model ref")
    p_dir.add_argument("--abstain", type=float, default=0.5, metavar="T",
                       help="abstain (direction=unknown) when |llr| < T (default: 0.5)")
    p_dir.add_argument("--seed", type=int, default=0, help="sampling seed (default: 0)")
    p_dir.add_argument("--json", action="store_true", help="machine-readable output")
    p_dir.set_defaults(func=cmd_direction)

    # --- decompose ----------------------------------------------------------
    p_dec = sub.add_parser(
        "decompose", help="recover merge parents and mixing ratios for a child model",
        description="Non-negative sparse decomposition of the child's task vector over candidates.",
    )
    p_dec.add_argument("child", metavar="CHILD", help="the (suspected) merged model")
    p_dec.add_argument("--candidates", nargs="+", required=True, metavar="REF",
                       help="candidate parent models")
    p_dec.add_argument("--base", metavar="REF", default=None,
                       help="shared base model for task vectors (inferred when omitted)")
    p_dec.add_argument("--l1", type=float, default=1e-3, help="L1 sparsity penalty (default: 1e-3)")
    p_dec.add_argument("--no-sum-to-one", action="store_true", dest="no_sum_to_one",
                       help="do not constrain coefficients to sum to 1")
    p_dec.add_argument("--support-threshold", type=float, default=0.05, dest="support_threshold",
                       help="minimum coefficient to count as a parent (default: 0.05)")
    p_dec.add_argument("--coords", type=int, default=200_000,
                       help="sub-sampled weight coordinates (default: 200000)")
    p_dec.add_argument("--seed", type=int, default=0, help="coordinate-sampling seed (default: 0)")
    p_dec.add_argument("--json", action="store_true", help="machine-readable output")
    p_dec.set_defaults(func=cmd_decompose)

    # --- index --------------------------------------------------------------
    p_idx = sub.add_parser(
        "index", help="sketch a universe and save a nearest-neighbour index",
        description="Build the candidate-retrieval index (faiss when installed, numpy otherwise).",
    )
    p_idx.add_argument("--universe", metavar="FILE", default=None,
                       help="universe file (see `stemma trace --universe`)")
    p_idx.add_argument("--out", metavar="PATH", default="stemma-index",
                       help="output index path (default: stemma-index)")
    p_idx.add_argument("--max-rows", type=int, default=2048, dest="max_rows",
                       help="rows sub-sampled per tensor (default: 2048)")
    p_idx.add_argument("--seed", type=int, default=0, help="sketching seed (default: 0)")
    p_idx.add_argument("--json", action="store_true", help="machine-readable output")
    p_idx.set_defaults(func=cmd_index)

    # --- bench --------------------------------------------------------------
    p_bench = sub.add_parser(
        "bench", help="run benchmarks/run.py and write results.json + figures",
        description="Shell out to the benchmark harness; --quick runs the reduced grid.",
    )
    p_bench.add_argument("--quick", action="store_true", help="reduced/fast benchmark grid")
    p_bench.add_argument("--out-dir", default=None, dest="out_dir",
                         help="forwarded to the harness as --out-dir")
    p_bench.add_argument("--json", action="store_true", help="suppress the launch banner")
    p_bench.add_argument("extra", nargs=argparse.REMAINDER,
                         help="extra arguments forwarded verbatim to benchmarks/run.py")
    p_bench.set_defaults(func=cmd_bench)

    return parser


def _version() -> str:
    """Package version string, resilient to a partially-installed package.

    Claim: infra.
    """
    try:
        from . import __version__

        return str(__version__)
    except Exception:  # pragma: no cover - defensive
        return "0.1.0"


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Parse arguments, dispatch to a subcommand and return a process exit code.

    Claim: infra -- one entry point for all four claims; it maps every failure
    onto the documented exit codes (0 ok / 1 error / 2 conflicts found) so that
    automation never has to parse prose.
    """
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if getattr(args, "command", None) is None:
        parser.print_help()
        return EXIT_ERROR

    if getattr(args, "verbose", 0):
        import logging

        logging.getLogger("stemma").setLevel(logging.DEBUG if args.verbose > 1 else logging.INFO)

    try:
        return int(args.func(args))
    except KeyboardInterrupt:  # pragma: no cover - interactive
        _emit_err("interrupted")
        return EXIT_ERROR
    except BrokenPipeError:  # pragma: no cover - `| head`
        return EXIT_OK
    except Exception as exc:
        LOG.debug("command failed", exc_info=True)
        _emit_err(f"error: {type(exc).__name__}: {exc}")
        return EXIT_ERROR


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
