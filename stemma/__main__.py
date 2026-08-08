"""``python -m stemma`` shim.

Claim: infra -- gives the CLI a second, install-free entry point so the
benchmark and the CI harness can invoke Stemma from a source checkout without a
console script on PATH.
"""

from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
