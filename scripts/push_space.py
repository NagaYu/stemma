#!/usr/bin/env python3
"""Assemble and publish the Stemma Gradio Space.

Stages exactly what the Space runtime needs -- ``app.py``, ``requirements.txt``,
a ``README.md`` carrying valid Space frontmatter, and the ``stemma`` package --
into a local directory, validates the frontmatter, prints the upload manifest,
and uploads only when explicitly told to.

Claim: infra -- the Space computes nothing new; it renders what
:mod:`stemma.phylogeny` and :mod:`stemma.rights` produce, always next to the
human-review disclaimer, so this packager's only job is to ship those modules
unchanged.

**Dry run by default.** Without ``--push`` nothing is created and nothing is
uploaded.

Usage::

    python scripts/push_space.py --repo-id org/stemma
    python scripts/push_space.py --repo-id org/stemma --push
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stemma.utils import human_bytes  # noqa: E402  (path bootstrap must come first)

#: Gradio major.minor.patch the Space frontmatter must declare. app.py targets
#: gradio 6.x; a Space pinned to 4.x renders a different Blocks API.
EXPECTED_SDK_VERSION = "6.22.0"

#: Top-level files copied verbatim into the Space, in presentation order.
TOP_LEVEL_FILES: Tuple[str, ...] = ("README.md", "app.py", "requirements.txt", "LICENSE")

#: Optional extra documents shipped so the Space's limitations are one click away.
DOC_FILES: Tuple[str, ...] = ("docs/FINDINGS.md", "CONTRACT.md")

#: Never staged, whatever the source tree looks like.
EXCLUDE_NAMES: frozenset[str] = frozenset(
    {"__pycache__", ".DS_Store", ".git", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
)


# --------------------------------------------------------------------------- #
# frontmatter
# --------------------------------------------------------------------------- #


def split_frontmatter(text: str) -> Tuple[Optional[str], str]:
    """Split a Markdown document into its leading YAML frontmatter and body.

    Claim: infra -- a Space that starts without a ``---`` block does not boot,
    so this check happens locally before anything is created on the Hub.
    """
    if not text.startswith("---"):
        return None, text
    lines = text.splitlines()
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            return "\n".join(lines[1:i]), "\n".join(lines[i + 1 :])
    return None, text


def parse_frontmatter(block: str) -> Dict[str, Any]:
    """Parse the flat top-level keys of a frontmatter block.

    Claim: infra -- PyYAML is not a declared dependency of this project, so the
    validator reads the handful of scalar/inline-list keys a Space frontmatter
    actually uses rather than pulling in a parser just to run a pre-flight check.

    Uses PyYAML when it happens to be installed; otherwise falls back to a small
    reader that understands ``key: value``, ``key: [a, b]`` and ``key:`` followed
    by ``  - item`` lines. Nested mappings are not supported (Spaces do not need
    them) and are reported as raw strings.
    """
    try:  # pragma: no cover - depends on environment
        import yaml  # type: ignore

        loaded = yaml.safe_load(block)
        if isinstance(loaded, dict):
            return loaded
    except Exception:
        pass

    out: Dict[str, Any] = {}
    key: Optional[str] = None
    for raw in block.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        if raw.startswith((" ", "\t")):
            item = raw.strip()
            if key is not None and item.startswith("-"):
                out.setdefault(key, [])
                if isinstance(out[key], list):
                    out[key].append(item[1:].strip().strip("'\""))
            continue
        if ":" not in raw:
            continue
        key, _, value = raw.partition(":")
        key = key.strip()
        value = value.strip()
        if not value:
            out[key] = []
        elif value.startswith("[") and value.endswith("]"):
            inner = value[1:-1].strip()
            out[key] = [v.strip().strip("'\"") for v in inner.split(",") if v.strip()] if inner else []
        else:
            out[key] = value.strip("'\"")
    return out


def validate_space_frontmatter(readme: Path, *, sdk_version: str = EXPECTED_SDK_VERSION) -> List[str]:
    """Check a README's frontmatter against what a Gradio Space requires.

    Claim: infra -- a wrong ``sdk_version`` or a missing ``app_file`` produces a
    Space that builds and then fails at runtime; catching it here keeps a broken
    demo from being the public face of the project.

    Returns a list of problems (empty means valid).
    """
    problems: List[str] = []
    if not readme.is_file():
        return [f"{readme} does not exist"]
    block, _ = split_frontmatter(readme.read_text(encoding="utf-8"))
    if block is None:
        return [f"{readme} does not start with a '---' YAML frontmatter block"]
    fm = parse_frontmatter(block)

    for required in ("title", "emoji", "colorFrom", "colorTo", "sdk", "app_file", "license"):
        if not fm.get(required):
            problems.append(f"frontmatter is missing required key '{required}'")

    if str(fm.get("sdk", "")).lower() != "gradio":
        problems.append(f"frontmatter sdk is {fm.get('sdk')!r}, expected 'gradio'")
    got = str(fm.get("sdk_version", "")).strip()
    if got != sdk_version:
        problems.append(
            f"frontmatter sdk_version is {got or '(missing)'!r}, expected {sdk_version!r} "
            "(the installed gradio, which app.py targets)"
        )
    app_file = str(fm.get("app_file", "")).strip()
    if app_file and not (REPO_ROOT / app_file).is_file():
        problems.append(f"frontmatter app_file '{app_file}' does not exist in the repo")
    return problems


def check_requirements(path: Path) -> List[str]:
    """Warn when the Space's requirements omit something ``app.py`` imports.

    Claim: infra -- the Space imports the stemma package plus gradio; a missing
    runtime dependency is a build that goes red on the Hub rather than locally.
    """
    warnings: List[str] = []
    if not path.is_file():
        return [f"{path} does not exist; the Space will install nothing"]
    text = path.read_text(encoding="utf-8").lower()
    for pkg in ("gradio", "numpy", "scipy", "requests", "safetensors", "huggingface_hub"):
        if pkg.lower() not in text:
            warnings.append(f"requirements.txt does not mention '{pkg}'")
    return warnings


# --------------------------------------------------------------------------- #
# staging
# --------------------------------------------------------------------------- #


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def stage_space(
    out_dir: Path,
    *,
    include_docs: bool = True,
    include_figures: bool = True,
    clean: bool = True,
) -> List[str]:
    """Copy the Space's runtime files into a clean staging directory.

    Claim: infra -- the published Space must run the *same* library code the
    benchmark measured, so the package is copied verbatim from the checkout
    instead of being pip-installed from an index at build time.

    Returns the list of staged paths, relative to ``out_dir``.
    """
    if clean and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    staged: List[str] = []

    for name in TOP_LEVEL_FILES:
        src = REPO_ROOT / name
        if src.is_file():
            _copy_file(src, out_dir / name)
            staged.append(name)

    pkg_src = REPO_ROOT / "stemma"
    if not pkg_src.is_dir():
        raise SystemExit(f"error: package directory not found at {pkg_src}")
    for p in sorted(pkg_src.rglob("*")):
        if any(part in EXCLUDE_NAMES for part in p.parts):
            continue
        if p.is_file() and p.suffix in (".py", ".json", ".txt", ".md"):
            rel = p.relative_to(REPO_ROOT)
            _copy_file(p, out_dir / rel)
            staged.append(str(rel))

    if include_docs:
        for name in DOC_FILES:
            src = REPO_ROOT / name
            if src.is_file():
                _copy_file(src, out_dir / name)
                staged.append(name)

    if include_figures:
        fig_src = REPO_ROOT / "figures"
        if fig_src.is_dir():
            for p in sorted(fig_src.rglob("*")):
                if p.is_file() and not any(part in EXCLUDE_NAMES for part in p.parts):
                    rel = p.relative_to(REPO_ROOT)
                    _copy_file(p, out_dir / rel)
                    staged.append(str(rel))

    return staged


def upload_plan(out_dir: Path) -> List[Tuple[str, int]]:
    """List every staged file with its size, in stable presentation order.

    Claim: infra -- the dry run must be exhaustive, so it walks the staging
    directory rather than echoing the intended file list.
    """
    files: List[Tuple[str, int]] = []
    for p in sorted(out_dir.rglob("*")):
        if p.is_file() and not any(part in EXCLUDE_NAMES for part in p.parts):
            files.append((str(p.relative_to(out_dir)), p.stat().st_size))
    rank = {name: i for i, name in enumerate(TOP_LEVEL_FILES)}
    files.sort(key=lambda kv: (rank.get(kv[0], len(TOP_LEVEL_FILES)), kv[0]))
    return files


def print_plan(
    plan: Sequence[Tuple[str, int]],
    *,
    repo_id: str,
    private: bool,
    sdk_version: str,
    out_dir: Path,
    pushing: bool,
) -> None:
    """Print the exact Space upload manifest.

    Claim: infra -- publishing a Space is a public, side-effecting act, so the
    default path stops after showing precisely what would be published.
    """
    verb = "UPLOADING" if pushing else "WOULD UPLOAD (dry run)"
    print("")
    print("=" * 72)
    print(f"{verb} -> https://huggingface.co/spaces/{repo_id}")
    print(f"  repo_type = space    space_sdk = gradio    sdk_version = {sdk_version}")
    print(f"  private   = {bool(private)}")
    print(f"  staged in = {out_dir}")
    print("-" * 72)
    total = 0
    for name, size in plan:
        total += size
        print(f"  {human_bytes(size):>10}  {name}")
    print("-" * 72)
    print(f"  {len(plan)} file(s), {human_bytes(total)} total")
    print("=" * 72)
    if not pushing:
        print("")
        print("Dry run: nothing was uploaded and no Space was created.")
        print("Re-run with --push to create the Space and upload the files above.")


def do_push(
    out_dir: Path, *, repo_id: str, private: bool, token: Optional[str], sdk: str = "gradio"
) -> str:
    """Create the Space if needed and upload the staged folder.

    Claim: infra -- ``create_repo(exist_ok=True)`` plus ``upload_folder`` makes
    redeploying idempotent, so the demo can be refreshed in lockstep with a
    benchmark rerun without hand-managing Hub state.
    """
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk=sdk,
        private=bool(private),
        exist_ok=True,
    )
    url = api.upload_folder(
        folder_path=str(out_dir),
        repo_id=repo_id,
        repo_type="space",
        commit_message="Deploy Stemma Gradio Space (app.py + stemma package)",
        ignore_patterns=["__pycache__/*", "*.pyc", ".DS_Store"],
    )
    return str(url)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    """Command-line surface for the Space packager.

    Claim: infra.
    """
    p = argparse.ArgumentParser(
        prog="push_space.py",
        description=(
            "Assemble the Stemma Gradio Space (app.py, requirements.txt, README.md "
            "with Space frontmatter, and the stemma package) and optionally push it. "
            "Dry run by default."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Nothing is created or uploaded unless --push is given.",
    )
    p.add_argument("--repo-id", required=True, metavar="ORG/NAME",
                   help="target Hugging Face Space id")
    p.add_argument("--out-dir", default="hf_space_export", metavar="DIR",
                   help="local staging directory (default: hf_space_export)")
    p.add_argument("--sdk-version", default=EXPECTED_SDK_VERSION, dest="sdk_version",
                   metavar="X.Y.Z",
                   help=f"gradio version the frontmatter must declare (default: {EXPECTED_SDK_VERSION})")
    p.add_argument("--no-docs", action="store_true",
                   help="do not ship docs/FINDINGS.md and CONTRACT.md alongside the app")
    p.add_argument("--no-figures", action="store_true", help="do not ship figures/")
    p.add_argument("--allow-frontmatter-problems", action="store_true",
                   help="stage and push even when README frontmatter validation fails")
    p.add_argument("--private", action="store_true", help="create the Space as private")
    p.add_argument("--token", default=None, metavar="TOKEN",
                   help="Hugging Face token; defaults to $HF_TOKEN / $HUGGINGFACE_HUB_TOKEN "
                        "or your cached `huggingface-cli login` credentials")
    p.add_argument("--push", action="store_true",
                   help="actually create the Space and upload (without this the script is a dry run)")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Validate, stage, print the manifest, and upload the Space only if asked.

    Claim: infra -- the Space is the project's public demo surface, so the
    happy path is "show me exactly what would go live" and publishing is opt-in.
    """
    args = build_parser().parse_args(list(argv) if argv is not None else None)

    readme = REPO_ROOT / "README.md"
    problems = validate_space_frontmatter(readme, sdk_version=args.sdk_version)
    if problems:
        print("README.md Space frontmatter problems:", file=sys.stderr)
        for prob in problems:
            print(f"  - {prob}", file=sys.stderr)
        if not args.allow_frontmatter_problems:
            print("refusing to stage; fix README.md or pass --allow-frontmatter-problems",
                  file=sys.stderr)
            return 1
        print("  (continuing because --allow-frontmatter-problems was given)", file=sys.stderr)
    else:
        print(f"README.md frontmatter OK (sdk=gradio, sdk_version={args.sdk_version})")

    for warn in check_requirements(REPO_ROOT / "requirements.txt"):
        print(f"warning: {warn}", file=sys.stderr)

    out_dir = Path(args.out_dir).expanduser()
    staged = stage_space(
        out_dir,
        include_docs=not args.no_docs,
        include_figures=not args.no_figures,
    )
    print(f"staged {len(staged)} file(s) into {out_dir}")

    if not (out_dir / "app.py").is_file():
        print("error: app.py was not staged; the Space would not boot", file=sys.stderr)
        return 1
    if not (out_dir / "stemma" / "__init__.py").is_file():
        print("error: the stemma package was not staged; the Space would not boot", file=sys.stderr)
        return 1

    plan = upload_plan(out_dir)
    print_plan(plan, repo_id=args.repo_id, private=args.private,
               sdk_version=args.sdk_version, out_dir=out_dir, pushing=bool(args.push))
    if not args.push:
        return 0

    token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    try:
        url = do_push(out_dir, repo_id=args.repo_id, private=args.private, token=token)
    except Exception as exc:
        print(f"error: upload failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print("")
    print(f"pushed: {url}")
    print(f"        https://huggingface.co/spaces/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
