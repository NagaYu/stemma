"""Stemma Gradio Space -- trace a model's lineage in the browser.

The whole analysis path lives in plain functions (:func:`run_trace`,
:func:`analyze`, :func:`render_dot_image`) that import nothing from gradio, so
the Space's logic is unit-testable in an environment where gradio is not
installed. Gradio itself is imported lazily inside :func:`build_ui` / :func:`main`.

Claim: infra -- the Space is the demo surface for the direction,
merge-recovery, low-transfer and low-false-positive claims; it computes nothing
new, it only renders what :mod:`stemma.phylogeny` and :mod:`stemma.rights`
produce, always alongside the mandatory human-review disclaimer.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from stemma.cli import (
    DEFAULT_UNIVERSE,
    DISCLAIMER,
    _coerce_stats,
    _stats_from_phylogeny,
    _transfer_line,
    load_universe,
    parse_universe_text,
)
from stemma.utils import get_logger, human_bytes, short_id

LOG = get_logger(__name__)

TITLE = "Stemma -- model provenance from weights alone"

INTRO = """
**Stemma** recovers *derivation direction*, *multi-parent merges* and *mixing
ratios* between checkpoints by reading a few megabytes of safetensors payload
over HTTP Range requests -- never a full download.

Give it a model id, optionally a candidate universe (one model reference per
line), and it returns a lineage DAG with a confidence on every edge, a
license-conflict panel, and a downloadable AI Bill of Materials.
"""

DISCLAIMER_MD = (
    "### Read this before acting on any result\n\n"
    "**Stemma outputs statistical evidence with a confidence score. "
    "It NEVER outputs a determination of infringement.**\n\n"
    "Weight-level similarity and derivation direction are *inferences* from a "
    "small sample of tensors. They can be wrong. Nothing here establishes "
    "provenance as fact, and nothing here is a legal determination of license "
    "compliance. **A human must review every finding before any action is taken.**"
)

EXAMPLE_ROWS: List[List[str]] = [
    ["HuggingFaceTB/SmolLM2-135M-Instruct",
     "HuggingFaceTB/SmolLM2-135M\nHuggingFaceTB/SmolLM2-360M\nQwen/Qwen2.5-0.5B\nEleutherAI/pythia-160m"],
    ["distilbert/distilgpt2",
     "openai-community/gpt2\nopenai-community/gpt2-medium\nEleutherAI/pythia-70m"],
    ["Qwen/Qwen2.5-0.5B-Instruct",
     "Qwen/Qwen2.5-0.5B\nHuggingFaceTB/SmolLM2-135M\nopenai-community/gpt2"],
]


# --------------------------------------------------------------------------- #
# Pure-python analysis (no gradio)
# --------------------------------------------------------------------------- #


def resolve_universe(universe_text: str = "", *, target: Optional[str] = None) -> List[str]:
    """Turn the textarea contents into the candidate universe for a trace.

    Claim: low-false-positive -- the negative set is what a false-positive rate
    is measured against, so an empty textarea falls back to a deliberately
    *mixed-family* default list rather than to the target's own family.
    """
    refs = parse_universe_text(universe_text)
    if not refs:
        try:
            refs = load_universe(None)
        except Exception as exc:  # pragma: no cover - defensive
            LOG.warning("universe fallback failed: %s", exc)
            refs = list(DEFAULT_UNIVERSE)
    if target:
        refs = [r for r in refs if r != target] + [target]
    seen: set[str] = set()
    return [r for r in refs if not (r in seen or seen.add(r))]


def run_trace(
    repo_id: str,
    universe_text: str = "",
    offline: bool = False,
    *,
    k: int = 10,
    out_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the full trace pipeline and return every renderable piece as a dict.

    Claim: direction -- the returned payload (oriented edges with confidence,
    merge mixing weights, and license conflicts that only exist along a
    direction) is exactly what a symmetric similarity score cannot produce.

    Each stage is guarded independently: a failure in rights propagation or in
    DOT rendering degrades that section only and is reported through
    ``result["errors"]`` instead of raising.
    """
    repo_id = (repo_id or "").strip()
    result: Dict[str, Any] = {
        "target": repo_id,
        "mermaid": "",
        "dot": "",
        "dot_png": None,
        "bom_path": None,
        "conflicts": [],
        "edges": [],
        "nodes": [],
        "stats": None,
        "seconds": 0.0,
        "errors": [],
    }
    if not repo_id:
        result["errors"].append("Enter a model id, e.g. `HuggingFaceTB/SmolLM2-135M-Instruct`.")
        return result

    out_dir = out_dir or tempfile.mkdtemp(prefix="stemma-space-")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    universe = resolve_universe(universe_text, target=repo_id)
    result["universe_size"] = len(universe)

    t0 = time.time()
    try:
        from stemma.phylogeny import trace as _trace

        kw: Dict[str, Any] = {"k": int(k)}
        if offline:
            kw["offline"] = True
        phylo = _trace(repo_id, universe, **kw)
    except Exception as exc:
        LOG.debug("trace failed", exc_info=True)
        result["errors"].append(f"Lineage reconstruction failed: {type(exc).__name__}: {exc}")
        result["seconds"] = time.time() - t0
        return result

    result["nodes"] = [str(n) for n in getattr(phylo, "nodes", []) or []]
    result["edges"] = [_edge_dict(e) for e in getattr(phylo, "edges", []) or []]
    result["stats"] = _stats_from_phylogeny(phylo)

    # --- rights -------------------------------------------------------------
    facts: Dict[str, Any] = {}
    conflicts: List[Any] = []
    try:
        from stemma.rights import detect_conflicts, fetch_license_facts, propagate

        for node in result["nodes"]:
            try:
                facts[node] = fetch_license_facts(node, offline=offline)
            except Exception as exc:
                LOG.info("license lookup failed for %s: %s", node, exc)
        facts = propagate(phylo, facts) or facts
        conflicts = list(detect_conflicts(phylo, facts) or [])
    except Exception as exc:
        LOG.debug("rights stage failed", exc_info=True)
        result["errors"].append(f"License analysis failed: {type(exc).__name__}: {exc}")
    result["conflicts"] = [_conflict_dict(c) for c in conflicts]

    # --- graph renderings ---------------------------------------------------
    try:
        from stemma.phylogeny import to_mermaid

        result["mermaid"] = to_mermaid(phylo, conflicts)
    except Exception as exc:
        LOG.debug("mermaid failed", exc_info=True)
        result["errors"].append(f"Mermaid rendering failed: {type(exc).__name__}: {exc}")
    try:
        from stemma.phylogeny import to_graphviz_dot

        result["dot"] = to_graphviz_dot(phylo, conflicts)
    except Exception as exc:
        LOG.debug("dot failed", exc_info=True)
        result["errors"].append(f"DOT rendering failed: {type(exc).__name__}: {exc}")
    if result["dot"]:
        result["dot_png"] = render_dot_image(result["dot"], out_dir)

    # --- BOM ----------------------------------------------------------------
    try:
        from stemma.rights import build_bom

        bom = build_bom(
            phylo, facts, conflicts, root=repo_id, transfer=result["stats"], sketches=None
        )
        path = Path(out_dir) / f"stemma-bom-{_slug(repo_id)}.json"
        path.write_text(bom.to_json(), encoding="utf-8")
        result["bom_path"] = str(path)
    except Exception as exc:
        LOG.debug("bom failed", exc_info=True)
        result["errors"].append(f"AI-BOM generation failed: {type(exc).__name__}: {exc}")

    result["seconds"] = time.time() - t0
    return result


def analyze(
    repo_id: str, universe_text: str = "", offline: bool = False
) -> Tuple[str, str, Optional[str], str]:
    """Trace ``repo_id`` and return (mermaid, conflicts markdown, BOM path, stats markdown).

    Claim: direction -- this is the Space's single analysis entry point and the
    unit-testable core of the UI; it exposes the oriented lineage, the rights
    conflicts implied by that orientation, and the byte cost of obtaining both.
    """
    res = run_trace(repo_id, universe_text, offline)
    return (
        mermaid_markdown(res),
        conflicts_markdown(res),
        res.get("bom_path"),
        stats_markdown(res),
    )


def _edge_dict(e: Any) -> Dict[str, Any]:
    """Flatten one :class:`~stemma.types.Edge` for the UI.

    Claim: infra.
    """
    return {
        "parent": str(getattr(e, "parent", "?")),
        "child": str(getattr(e, "child", "?")),
        "confidence": float(getattr(e, "confidence", 0.0)),
        "relation": str(getattr(e, "relation", "derived")),
        "weight": (None if getattr(e, "weight", None) is None else float(e.weight)),
        "evidence": [str(x) for x in (getattr(e, "evidence", []) or [])],
    }


def _conflict_dict(c: Any) -> Dict[str, Any]:
    """Flatten one :class:`~stemma.types.RightsConflict` for the UI.

    Claim: infra.
    """
    return {
        "descendant": str(getattr(c, "descendant", "?")),
        "ancestor": str(getattr(c, "ancestor", "?")),
        "kind": str(getattr(c, "kind", "unknown")),
        "severity": str(getattr(c, "severity", "info")),
        "message": str(getattr(c, "message", "")),
        "path": [str(x) for x in (getattr(c, "path", []) or [])],
        "confidence": float(getattr(c, "confidence", 0.0)),
    }


def _slug(ref: str) -> str:
    """Filesystem-safe version of a model reference.

    Claim: infra.
    """
    return "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in str(ref))[:80] or "model"


def render_dot_image(dot: str, out_dir: str) -> Optional[str]:
    """Rasterise a DOT graph to PNG when the graphviz *binary* is on PATH.

    Claim: infra -- the mermaid block is always available; the PNG is a pure
    enhancement, so a Space without the graphviz system package degrades to the
    text diagram instead of erroring.
    """
    if not dot:
        return None
    exe = shutil.which("dot")
    if not exe:
        LOG.info("graphviz `dot` binary not found; skipping PNG rendering")
        return None
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    src = out_dir_p / "phylogeny.dot"
    png = out_dir_p / "phylogeny.png"
    try:
        src.write_text(dot, encoding="utf-8")
        subprocess.run(
            [exe, "-Tpng", str(src), "-o", str(png)],
            check=True, capture_output=True, timeout=60,
        )
    except Exception as exc:
        LOG.info("graphviz rendering failed: %s", exc)
        return None
    return str(png) if png.is_file() else None


def mermaid_markdown(res: Dict[str, Any]) -> str:
    """Wrap the phylogeny in a fenced ```mermaid block for ``gr.Markdown``.

    Claim: merge-recovery -- the diagram is where multi-parent structure becomes
    visible: a node with two incoming edges, each labelled with its recovered
    mixing coefficient.
    """
    errors = [e for e in res.get("errors", []) if "rendering" in e.lower() or "Lineage" in e]
    mer = (res.get("mermaid") or "").strip()
    if not mer:
        body = "\n".join(f"- {e}" for e in errors) or "- No lineage could be reconstructed."
        return f"### Lineage\n\n_Nothing to draw._\n\n{body}"
    if mer.startswith("```"):
        block = mer
    else:
        block = f"```mermaid\n{mer}\n```"
    edges = res.get("edges") or []
    merges = [e for e in edges if e.get("weight") is not None]
    lines = [f"### Lineage of `{res.get('target', '')}`", "", block, ""]
    if merges:
        lines.append("**Recovered mixing ratios**")
        lines.append("")
        lines.append("| child | parent | mixing weight | confidence |")
        lines.append("| --- | --- | ---: | ---: |")
        for e in sorted(merges, key=lambda x: (x["child"], -(x["weight"] or 0.0))):
            lines.append(
                f"| `{short_id(e['child'], 40)}` | `{short_id(e['parent'], 40)}` "
                f"| {e['weight']:.3f} | {e['confidence']:.2f} |"
            )
        lines.append("")
    if errors:
        lines.append("_Partial result:_ " + "; ".join(errors))
    return "\n".join(lines)


def conflicts_markdown(res: Dict[str, Any]) -> str:
    """Render the license-conflict warnings panel with severity and confidence.

    Claim: low-false-positive -- every warning is printed together with the
    confidence of the lineage that implies it, so a weak edge can never read as
    a hard license finding.
    """
    errs = [e for e in res.get("errors", []) if "License" in e or "Lineage" in e]
    conflicts = res.get("conflicts") or []
    head = "### License warnings"
    if not conflicts:
        body = (
            "No license conflicts detected along the inferred lineage.\n\n"
            "_Absence of a warning is not a clearance: unknown or ungated "
            "ancestors may simply be missing from the candidate universe._"
        )
        return f"{head}\n\n{body}" + ("\n\n" + "\n".join(f"- {e}" for e in errs) if errs else "")
    order = {"high": 0, "warning": 1, "info": 2}
    icon = {"high": "HIGH", "warning": "WARNING", "info": "INFO"}
    rows = ["| severity | kind | confidence | detail |", "| --- | --- | ---: | --- |"]
    for c in sorted(conflicts, key=lambda c: (order.get(c["severity"], 3), -c["confidence"])):
        detail = c["message"].replace("|", "/").replace("\n", " ")
        if c.get("path"):
            detail += "  \n_path: " + " to ".join(short_id(p, 24) for p in c["path"]) + "_"
        rows.append(
            f"| **{icon.get(c['severity'], c['severity'].upper())}** | `{c['kind']}` "
            f"| {c['confidence']:.2f} | {detail} |"
        )
    tail = "\n\n" + "\n".join(f"- {e}" for e in errs) if errs else ""
    return f"{head}\n\n" + "\n".join(rows) + tail


def stats_markdown(res: Dict[str, Any]) -> str:
    """Render the transfer/latency panel: read X of Y, N times less, in T seconds.

    Claim: low-transfer -- this panel is the user-visible proof that the whole
    lineage was reconstructed from Range reads rather than from downloads.
    """
    stats = _coerce_stats(res.get("stats"))
    seconds = float(res.get("seconds") or 0.0)
    lines = ["### Transfer", "", "`" + _transfer_line(stats, seconds) + "`", ""]
    lines.append(
        f"- candidate universe: **{res.get('universe_size', 0)}** models, "
        f"lineage kept **{len(res.get('nodes') or [])}** nodes / "
        f"**{len(res.get('edges') or [])}** edges"
    )
    if stats is not None and stats.full_size_bytes > 0 and stats.bytes_read > 0:
        lines.append(
            f"- full checkpoints would have been **{human_bytes(stats.full_size_bytes)}**; "
            f"Stemma read **{human_bytes(stats.bytes_read)}** "
            f"({stats.full_size_bytes / max(stats.bytes_read, 1):.0f}x less)"
        )
    other = [e for e in res.get("errors", []) if "AI-BOM" in e]
    if other:
        lines.append("")
        lines.extend(f"- {e}" for e in other)
    return "\n".join(lines)


def analyze_ui(
    repo_id: str, universe_text: str, offline: bool, k: int = 10
) -> Tuple[str, str, Optional[str], Optional[str], str]:
    """Gradio callback: one trace, five outputs, never an exception.

    Claim: infra -- the Space must stay up; every failure is converted into
    rendered markdown so a broken model id degrades a panel instead of killing
    the worker.
    """
    try:
        res = run_trace(repo_id, universe_text, bool(offline), k=int(k))
    except Exception as exc:  # pragma: no cover - last-resort guard
        LOG.debug("analyze_ui failed", exc_info=True)
        tb = traceback.format_exc(limit=3)
        err = f"### Unexpected error\n\n```\n{type(exc).__name__}: {exc}\n{tb}\n```"
        return err, "### License warnings\n\n_Not computed._", None, None, "### Transfer\n\n_n/a_"
    return (
        mermaid_markdown(res),
        conflicts_markdown(res),
        res.get("bom_path"),
        res.get("dot_png"),
        stats_markdown(res),
    )


# --------------------------------------------------------------------------- #
# Gradio UI (imported lazily -- gradio is an optional dependency)
# --------------------------------------------------------------------------- #


def build_ui() -> Any:
    """Construct the Gradio Blocks app. Imports gradio lazily.

    Claim: infra -- keeping the import inside the function is what allows this
    module to be byte-compiled, imported and unit-tested in the (local)
    environment where gradio is not installed.
    """
    import gradio as gr

    with gr.Blocks(title=TITLE, analytics_enabled=False) as demo:
        gr.Markdown(f"# {TITLE}")
        gr.Markdown(INTRO)
        gr.Markdown(DISCLAIMER_MD)

        with gr.Row():
            with gr.Column(scale=2):
                repo = gr.Textbox(
                    label="Model id",
                    placeholder="org/model  (Hugging Face repo id, or a local directory)",
                    value="",
                    lines=1,
                )
                universe = gr.Textbox(
                    label="Candidate universe (optional)",
                    info="One model reference per line. Leave empty to use the built-in "
                         "mixed-family default list.",
                    lines=6,
                    placeholder="\n".join(DEFAULT_UNIVERSE[:5]),
                )
                with gr.Row():
                    offline = gr.Checkbox(label="Offline (cached headers only)", value=False)
                    k = gr.Slider(
                        label="Candidate parents per node (k)",
                        minimum=1, maximum=25, step=1, value=10,
                    )
                run_btn = gr.Button("Trace lineage", variant="primary")
            with gr.Column(scale=1):
                stats_md = gr.Markdown("### Transfer\n\n_Run a trace to see byte accounting._")
                bom_file = gr.File(label="AI-BOM (JSON)", interactive=False)

        graph_md = gr.Markdown("### Lineage\n\n_Run a trace to draw the DAG._")
        graph_img = gr.Image(
            label="Phylogeny (graphviz)",
            show_label=True,
            interactive=False,
            type="filepath",
        )
        conflicts_md = gr.Markdown("### License warnings\n\n_Run a trace._")

        gr.Examples(
            examples=EXAMPLE_ROWS,
            inputs=[repo, universe],
            label="Examples (small public models)",
        )

        gr.Markdown(DISCLAIMER_MD)
        gr.Markdown(f"<sub>{DISCLAIMER}</sub>")

        run_btn.click(
            fn=analyze_ui,
            inputs=[repo, universe, offline, k],
            outputs=[graph_md, conflicts_md, bom_file, graph_img, stats_md],
            api_name="trace",
        )
        repo.submit(
            fn=analyze_ui,
            inputs=[repo, universe, offline, k],
            outputs=[graph_md, conflicts_md, bom_file, graph_img, stats_md],
        )

    return demo


def main() -> None:
    """Launch the Space on 0.0.0.0.

    Claim: infra -- entry point for `python app.py` and for the Hugging Face
    Space runtime; gradio is imported here, never at module import time.
    """
    demo = build_ui()
    kwargs = {
        "server_name": os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        "server_port": int(os.environ.get("PORT", 7860)),
    }
    # `show_api` was removed in gradio 6 and raises TypeError there, but it is
    # still wanted on gradio 4/5 where the Space would otherwise advertise an
    # API tab. Pass it only when the running version accepts it rather than
    # pinning a version the Space runtime may not honour.
    import inspect

    if "show_api" in inspect.signature(demo.launch).parameters:
        kwargs["show_api"] = False
    demo.queue(default_concurrency_limit=2).launch(**kwargs)


if __name__ == "__main__":
    main()
