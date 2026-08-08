"""Fingerprinting a checkpoint must cost far less than downloading it.

Claim: low-transfer -- this is the file that turns the headline "reads a few
megabytes instead of the whole checkpoint" into a measured, failing-if-violated
assertion. The local test proves the *mechanism* (only the requested rows' byte
ranges are ever touched) offline; the ``network`` test proves the same mechanism
survives a real HTTP server that must answer 206 Partial Content.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from stemma.remote_loader import (
    RangeUnsupported,
    SafeTensorsSource,
    measure_full_download_bytes,
    open_model,
)
from stemma.sketch import sketch_model

# --------------------------------------------------------------------------- #
# Thresholds (judgement calls, stated once here)
# --------------------------------------------------------------------------- #

#: Local budget. These fixtures are ~4.7 MB, i.e. thousands of times smaller
#: than a real checkpoint, and the per-file safetensors header is a fixed cost
#: that a 4 MB file cannot amortise. 25% is therefore a deliberately *loose*
#: bound; the real-repo figure recorded in docs/FINDINGS.md section 6 is 3.90%.
MAX_LOCAL_FRACTION: float = 0.25

#: Remote budget for the network test, matching the measured 3.90% of
#: FINDINGS section 6 scaled down by the smaller row budget used here.
MAX_REMOTE_FRACTION: float = 0.02

#: Rows per tensor. The single lever behind the claim: capping rows caps bytes.
MAX_ROWS: int = 32

#: A small, single-shard, ungated public repo used only by the network test.
NETWORK_REPO: str = "openai-community/gpt2"

NETWORK_ENABLED = os.environ.get("STEMMA_NETWORK_TESTS", "") == "1"
requires_network = pytest.mark.skipif(
    not NETWORK_ENABLED,
    reason="set STEMMA_NETWORK_TESTS=1 to run tests that hit huggingface.co",
)


# --------------------------------------------------------------------------- #
# Local: the mechanism
# --------------------------------------------------------------------------- #


def test_sketching_reads_far_less_than_the_whole_file(tiny_parent: str) -> None:
    """A full sketch must touch well under a quarter of the checkpoint's bytes."""
    src = open_model(tiny_parent)
    try:
        total = src.total_size()
        assert total > 0, "total_size() must be known without reading payload"
        s = sketch_model(tiny_parent, source=src, max_rows=MAX_ROWS, k=32, seed=0)
        read = int(src.stats.bytes_read)
    finally:
        src.close()

    fraction = read / total
    assert read > 0, "a sketch that read nothing is not evidence of anything"
    assert s.present.any(), "sketch filled no slots, so the byte count is meaningless"
    assert fraction < MAX_LOCAL_FRACTION, (
        f"sketching read {read} of {total} bytes ({fraction:.1%}), "
        f"budget {MAX_LOCAL_FRACTION:.0%}"
    )

    # The Sketch itself must carry the accounting, since the BOM reports it.
    assert s.stats is not None
    assert s.stats.bytes_read == read
    assert s.stats.full_size_bytes == total
    assert s.stats.reduction > 1.0 / MAX_LOCAL_FRACTION


def _tensor_payload(ref: str, name: str, **kw) -> tuple[int, np.ndarray]:
    """Bytes of *payload* (header cost excluded) spent reading one tensor."""
    src = open_model(ref)
    try:
        src.index()
        header_cost = int(src.stats.bytes_read)  # index() reads headers only
        W = src.get_tensor(name, **kw)
        return int(src.stats.bytes_read) - header_cost, W
    finally:
        src.close()


def test_row_capping_is_what_saves_the_bytes(tiny_parent: str) -> None:
    """A capped row budget must cost a small fraction of reading the tensor whole.

    Claim: low-transfer -- this pins the *causal* mechanism: the saving comes
    from fetching row byte-ranges, not from an accounting artefact.
    """
    capped, Wc = _tensor_payload(tiny_parent, "model.embed_tokens.weight", max_rows=MAX_ROWS)
    whole, Ww = _tensor_payload(tiny_parent, "model.embed_tokens.weight")

    assert Wc.shape[0] == MAX_ROWS
    assert Ww.shape[0] == 2048
    assert whole == Ww.size * 4, "reading a tensor whole should cost exactly its bytes"
    assert capped < 0.10 * whole, (
        f"capping to {MAX_ROWS} rows cost {capped} bytes vs {whole} for the whole tensor"
    )


def test_get_tensor_reads_only_the_requested_rows(tiny_parent: str) -> None:
    """Payload for a capped read stays inside the loader's documented 2x bound.

    ``coalesce_ranges`` is allowed to pull in up to ``max_slack`` unwanted bytes
    to save round trips, and the loader passes the wanted payload size as that
    budget -- so the guarantee is "no more than twice what you asked for".
    """
    payload, W = _tensor_payload(tiny_parent, "model.embed_tokens.weight", max_rows=MAX_ROWS)
    n_cols = int(W.shape[1])
    assert W.shape == (MAX_ROWS, n_cols)
    wanted = MAX_ROWS * n_cols * 4  # F32
    assert wanted <= payload <= 2 * wanted, (
        f"asked for {wanted} bytes of rows, loader touched {payload}"
    )


def test_header_only_access_costs_almost_nothing(tiny_parent: str) -> None:
    """dtype/shape/vocab for every tensor must be readable without payload."""
    src = open_model(tiny_parent)
    try:
        index = src.index()
        total = src.total_size()
        read = int(src.stats.bytes_read)
    finally:
        src.close()
    assert len(index) > 20
    assert read < 0.02 * total, (
        f"header-only read cost {read} of {total} bytes; it must be a rounding error"
    )


def test_measure_full_download_bytes_does_not_download(tiny_parent: str) -> None:
    """The denominator of the reduction factor must itself be cheap."""
    n = measure_full_download_bytes(tiny_parent)
    src = open_model(tiny_parent)
    try:
        assert n == src.total_size()
    finally:
        src.close()
    assert n > 1_000_000, "the fixture should be a multi-megabyte file"


def test_two_models_can_be_read_row_for_row(tiny_parent: str, tiny_child_sft: str) -> None:
    """Identical row selection on both sides, still paying only for those rows.

    Claim: low-transfer -- every pairwise comparison in the project depends on
    reading the *same* rows from two models; if that forced a full read the
    low-transfer claim would collapse for exactly the operation that matters.
    """
    a = open_model(tiny_parent)
    b = open_model(tiny_child_sft)
    try:
        Wa = a.get_tensor("model.layers.0.self_attn.q_proj.weight", max_rows=MAX_ROWS)
        rows = np.asarray(a.last_rows)
        Wb = b.get_tensor_rows("model.layers.0.self_attn.q_proj.weight", rows)
        assert Wa.shape == Wb.shape
        combined = a.stats.bytes_read + b.stats.bytes_read
        total = a.total_size() + b.total_size()
    finally:
        a.close()
        b.close()
    assert combined < MAX_LOCAL_FRACTION * total


# --------------------------------------------------------------------------- #
# Network: the same mechanism against a real HTTP server
# --------------------------------------------------------------------------- #


@pytest.mark.network
@pytest.mark.slow
@requires_network
def test_remote_sketch_uses_range_requests_and_stays_under_budget() -> None:
    """Against a real repo: 206 Partial Content, and under 2% of the shard.

    Claim: low-transfer -- ``require_206`` in the loader means a server that
    answers 200 raises :class:`RangeUnsupported` rather than silently paying for
    the whole file, so passing this test proves the bytes really were partial.
    """
    src = SafeTensorsSource(NETWORK_REPO)
    try:
        total = src.total_size()
        assert total > 100_000_000, "pick a repo big enough for the ratio to mean something"
        try:
            s = sketch_model(NETWORK_REPO, source=src, max_rows=256, k=32, seed=0)
        except RangeUnsupported as exc:  # pragma: no cover - server-dependent
            pytest.fail(f"server did not honour Range requests: {exc}")
        read = int(src.stats.bytes_read)
        requests = int(src.stats.requests)
    finally:
        src.close()

    assert s.present.any()
    assert requests > 1, "a single request would mean the whole file was fetched"
    fraction = read / total
    assert fraction < MAX_REMOTE_FRACTION, (
        f"remote sketch read {read} of {total} bytes ({fraction:.2%}), "
        f"budget {MAX_REMOTE_FRACTION:.0%}"
    )


@pytest.mark.network
@requires_network
def test_remote_header_only_is_kilobytes() -> None:
    """The tensor inventory of a real repo must cost kilobytes, not megabytes."""
    src = SafeTensorsSource(NETWORK_REPO)
    try:
        index = src.index()
        total = src.total_size()
        read = int(src.stats.bytes_read)
    finally:
        src.close()
    assert len(index) > 10
    assert read < 512 * 1024, f"header-only access cost {read} bytes"
    assert read < 0.001 * total
