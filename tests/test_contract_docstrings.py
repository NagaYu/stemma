"""Every public symbol must say which of the four claims it serves.

Claim: infra -- CONTRACT.md makes the ``Claim:`` line a hard requirement, for a
reason that is not bookkeeping: a research prototype whose code cannot say what
each function is evidence *for* is how overclaiming happens. This test is the
enforcement mechanism, and it fails loudly rather than warning.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import pkgutil
import re
from typing import Iterator, List, Tuple

import pytest

import stemma

#: The exact vocabulary CONTRACT.md permits after "Claim:".
ALLOWED_CLAIMS = frozenset(
    {"direction", "merge-recovery", "low-transfer", "low-false-positive", "infra"}
)

#: A line of the form "Claim: <value>", optionally followed by the required
#: one-sentence justification on the same line or the lines below.
CLAIM_RE = re.compile(r"^[ \t]*Claim:[ \t]*([A-Za-z][A-Za-z\-]*)", re.MULTILINE)

#: ``stemma.types`` is frozen by the project brief: it is a pure wire-format
#: module of dataclasses whose claim support is stated once in its module
#: docstring, and it may not be edited to add per-field ``Claim:`` lines. Its
#: *module* docstring is still checked below.
FROZEN_DATA_MODULES = frozenset({"stemma.types"})

#: The no-``print()`` rule exists so that *importable analysis code* stays silent
#: and never pollutes the Gradio app or the benchmark's stdout. ``stemma.cli`` is
#: the terminal-output layer itself -- CONTRACT.md requires it to print the
#: human-review disclaimer on every ``trace`` -- so it is the one exemption.
PRINTING_MODULES = frozenset({"stemma.cli"})


def _package_modules() -> List[str]:
    """Import names of every non-private module in the stemma package."""
    return [
        f"stemma.{m.name}"
        for m in pkgutil.iter_modules(stemma.__path__)
        if not m.name.startswith("_")
    ]


def _public_symbols(module) -> Iterator[Tuple[str, object]]:
    """Public functions and classes *defined in* ``module`` (not imported)."""
    for name, obj in sorted(vars(module).items()):
        if name.startswith("_"):
            continue
        if not (inspect.isfunction(obj) or inspect.isclass(obj)):
            continue
        if getattr(obj, "__module__", None) != module.__name__:
            continue
        yield name, obj


def _public_methods(cls) -> Iterator[Tuple[str, object]]:
    """Public methods, classmethods, staticmethods and properties of ``cls``."""
    for name, raw in sorted(vars(cls).items()):
        if name.startswith("_"):
            continue
        obj = raw.fget if isinstance(raw, property) else raw
        if isinstance(obj, (classmethod, staticmethod)):
            obj = obj.__func__
        if not inspect.isfunction(obj):
            continue
        yield name, obj


def _claim_of(obj) -> str | None:
    doc = inspect.getdoc(obj) or ""
    m = CLAIM_RE.search(doc)
    return m.group(1) if m else None


MODULES = _package_modules()


def test_the_package_exposes_the_modules_the_contract_names() -> None:
    """Every module CONTRACT.md specifies must exist and import cleanly."""
    required = {
        "stemma.utils",
        "stemma.types",
        "stemma.remote_loader",
        "stemma.sketch",
        "stemma.direction",
        "stemma.merge_decompose",
        "stemma.baselines",
        "stemma.phylogeny",
        "stemma.rights",
        "stemma.cli",
    }
    missing = sorted(required - set(MODULES))
    assert not missing, f"CONTRACT.md names modules that do not exist: {missing}"


@pytest.mark.parametrize("module_name", MODULES)
def test_module_docstring_states_its_claim_support(module_name: str) -> None:
    """Each module must say, in prose, which headline claim it supports."""
    mod = importlib.import_module(module_name)
    doc = inspect.getdoc(mod) or ""
    assert doc.strip(), f"{module_name} has no module docstring"
    assert "claim" in doc.lower(), (
        f"{module_name}'s module docstring never mentions a claim; CONTRACT.md "
        "requires every module to state what it is evidence for"
    )


@pytest.mark.parametrize("module_name", MODULES)
def test_public_functions_and_classes_declare_a_claim(module_name: str) -> None:
    """Walk the module and require a valid ``Claim:`` line on every public symbol."""
    if module_name in FROZEN_DATA_MODULES:
        pytest.skip(f"{module_name} is frozen; its claim support is stated module-wide")

    mod = importlib.import_module(module_name)
    problems: List[str] = []

    for name, obj in _public_symbols(mod):
        claim = _claim_of(obj)
        qual = f"{module_name}.{name}"
        if claim is None:
            problems.append(f"{qual}: no 'Claim:' line in the docstring")
        elif claim not in ALLOWED_CLAIMS:
            problems.append(f"{qual}: claim {claim!r} is not one of {sorted(ALLOWED_CLAIMS)}")

        if inspect.isclass(obj):
            for mname, meth in _public_methods(obj):
                mclaim = _claim_of(meth)
                mqual = f"{qual}.{mname}"
                if mclaim is None:
                    problems.append(f"{mqual}: no 'Claim:' line in the docstring")
                elif mclaim not in ALLOWED_CLAIMS:
                    problems.append(
                        f"{mqual}: claim {mclaim!r} is not one of {sorted(ALLOWED_CLAIMS)}"
                    )

    assert not problems, "\n".join(problems)


@pytest.mark.parametrize("module_name", MODULES)
def test_library_code_uses_the_logger_not_print(module_name: str) -> None:
    """No ``print()`` in library code -- CONTRACT.md reserves it for scripts.

    Claim: infra -- a library that prints cannot be embedded in the Gradio app
    or the benchmark harness without polluting their output.
    """
    if module_name in PRINTING_MODULES:
        pytest.skip(f"{module_name} is the terminal-output layer and must print")

    mod = importlib.import_module(module_name)
    tree = ast.parse(inspect.getsource(mod))
    offenders = [
        f"{module_name}:{node.lineno}"
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "print"
    ]
    assert not offenders, "print() found in library code:\n" + "\n".join(offenders)


def test_no_module_performs_network_io_at_import_time() -> None:
    """Importing any stemma module must not touch the network.

    Claim: infra -- ``import stemma`` happening to issue an HTTP request would
    make every transfer measurement in the project unreliable.
    """
    import requests

    calls: List[str] = []
    original = requests.Session.request

    def spy(self, method, url, *a, **kw):  # pragma: no cover - must never run
        calls.append(f"{method} {url}")
        raise AssertionError(f"network call at import time: {method} {url}")

    requests.Session.request = spy
    try:
        for name in MODULES:
            importlib.reload(importlib.import_module(name))
    finally:
        requests.Session.request = original
    assert not calls
