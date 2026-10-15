"""Long operator chains keep their backward results and route errors while costing about n log n."""

from __future__ import annotations

import hashlib
import json
import tracemalloc
from pathlib import Path
from typing import Any

import pytest

from yian.compose import load_flow
from yian.compose.loader import ComposedFlow
from yian.expr.expressions import _balanced_concat
from yian.expr.lineage import SourceRef, ValueRoute
from yian.memory import contracts as memory_contracts
from yian.nodes import base as node_base
from yian.nodes.base import BackwardContext
from yian.run import resume_flow, run_flow
from yian.runtime import contracts, log as run_log
from yian.runtime.contracts import _validate_route
from yian.runtime.log import _route_from_payload, _route_payload
from yian.runtime.trace import RetainedTrace
from tests.engine._memory_fakes import _Memory

_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "route_chain_results.json"
"""Backward results recorded per chain, keyed by ``<kind>-<n>``.

Short chains keep their full results; ``_DIGEST_CASES`` keep only a sha256 digest.
"""


def _chain_flow(kind: str, n: int) -> tuple[ComposedFlow, dict[str, Any]]:
    """Return a flow calling an identity child on an ``n``-operand chain, and its inputs.

    ``list`` chains concatenate ``input.values`` (a balanced ``concat`` route);
    ``scalar`` chains add ``input.x`` (one ``scalar`` route with ``n`` operands).
    """
    if kind == "list":
        chain = " + ".join(["input.values"] * n)
        text = (
            "id: lc\nname: lc\nlong_term_memory: test\ninput:\n  values: !list(str)\n"
            "defs:\n  identity:\n    input:\n      values: !list(str)\n    nodes: {}\n"
            "    output: ${input.values}\n"
            "nodes:\n  outer:\n    kind: call\n    call: identity\n    input:\n"
            f"      values: ${{{chain}}}\n"
            "output: ${outer.output}\n"
        )
        return load_flow(text), {"values": ["v"]}
    chain = " + ".join(["input.x"] * n)
    text = (
        "id: bw\nname: bw\nlong_term_memory: test\ninput:\n  x: !int\n"
        "defs:\n  identity:\n    input:\n      v: !int\n    nodes: {}\n    output: ${input.v}\n"
        "nodes:\n  outer:\n    kind: call\n    call: identity\n    input:\n"
        f"      v: ${{{chain}}}\n"
        "output: ${outer.output}\n"
    )
    return load_flow(text), {"x": 1}


def _replace_run_id(value: Any, run_id: str) -> Any:
    """Return ``value`` with every string equal to ``run_id`` replaced by ``"RUN"``."""
    if isinstance(value, dict):
        return {key: _replace_run_id(item, run_id) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_replace_run_id(item, run_id) for item in value]
    return "RUN" if value == run_id else value


def _results(kind: str, n: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Run, resume and backward one chain flow; return its observable results as canonical JSON.

    The results are the feedback each learn context received, the downstream
    context of every backward context built, the resumed status and output, and
    the backward status, with the run ID replaced by ``"RUN"``.
    """
    flow, inputs = _chain_flow(kind, n)
    received: list[tuple[dict[str, Any], ...]] = []
    build = RetainedTrace.backward_context

    def spy(self: RetainedTrace, *args: Any, **kwargs: Any) -> BackwardContext:
        context = build(self, *args, **kwargs)
        received.append(context.downstream_context)
        return context

    monkeypatch.setattr(RetainedTrace, "backward_context", spy)
    memory = _Memory()
    log = tmp_path / f"{kind}-{n}.jsonl"
    run = run_flow(flow, inputs, log=log, learnable=True, memory={"test": memory})
    resumed = resume_flow(flow, log=log, memory={"test": memory})
    outcome = run.backward("good")
    results = {
        "feedback": [list(context.feedback) for context in memory.learn_contexts],
        "downstream_contexts": [list(contexts) for contexts in received],
        "resumed": {"status": resumed.status, "output": resumed.output},
        "backward_status": outcome.status,
    }
    text = json.dumps(_replace_run_id(results, run.run_id), sort_keys=True)
    assert run.run_id not in text
    return text


_CASES = [("list", 3), ("scalar", 3), ("list", 7)]
# The full list-7 results are too large to keep in the repository, so only their digest is recorded.
_DIGEST_CASES = {("list", 7)}


def _digest(text: str) -> str:
    """Return the sha256 hex digest of canonical results text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@pytest.mark.parametrize(("kind", "n"), _CASES)
def test_backward_results_match_the_recorded_results(kind, n, tmp_path, monkeypatch):
    recorded = json.loads(_FIXTURE.read_text(encoding="utf-8"))[f"{kind}-{n}"]
    results = _results(kind, n, tmp_path, monkeypatch)
    if (kind, n) in _DIGEST_CASES:
        assert _digest(results) == recorded
    else:
        assert results == json.dumps(recorded, sort_keys=True)


_SOURCE = ValueRoute("source", source=SourceRef("run", 1))
"""One valid source leaf shared by the route error trees."""


def _duplicate_names_and_bad_child() -> tuple[ValueRoute, dict[str, Any]]:
    """Return a record with duplicate names whose first child has a non-string expression."""
    bad = ValueRoute("source", source=SourceRef("run", 1), expression=5)  # type: ignore[arg-type]
    tree = ValueRoute("record", children=(bad, _SOURCE), names=("a", "a"))
    payload = _route_payload(ValueRoute("record", children=(_SOURCE, _SOURCE), names=("a", "b")))
    payload["names"] = ["a", "a"]
    payload["children"][0]["expression"] = 5
    return tree, payload


def _head_error_and_bad_child() -> tuple[ValueRoute, dict[str, Any]]:
    """Return a source route carrying a child whose expression is not a string."""
    bad = ValueRoute("source", source=SourceRef("run", 1), expression=5)  # type: ignore[arg-type]
    tree = ValueRoute("source", source=SourceRef("run", 1), children=(bad,))
    payload = _route_payload(_SOURCE)
    payload["children"] = [dict(_route_payload(_SOURCE), expression=5)]
    return tree, payload


def _bad_spans_and_bad_child_snapshot() -> tuple[ValueRoute, dict[str, Any]]:
    """Return an interpolation with misordered spans over a scalar whose snapshot holds a source."""
    child = ValueRoute("scalar", children=(_SOURCE,), result_type="Thing", snapshot=_SOURCE)
    tree = ValueRoute("interpolation", children=(child,), value_json='"ab"', spans=((2, 1),))
    valid_child = ValueRoute("scalar", children=(_SOURCE,), value_json="1")
    payload = _route_payload(
        ValueRoute("interpolation", children=(valid_child,), value_json='"ab"', spans=((0, 1),))
    )
    payload["spans"] = [[2, 1]]
    payload["children"][0].update(value_json=None, result_type="Thing", snapshot=_route_payload(_SOURCE))
    return tree, payload


def _snapshot_with_invalid_source() -> tuple[ValueRoute, dict[str, Any]]:
    """Return a scalar whose list snapshot contains a source route with an empty run ID."""
    snapshot = ValueRoute("list", children=(ValueRoute("source", source=SourceRef("", 1)),))
    tree = ValueRoute("scalar", children=(_SOURCE,), result_type="Thing", snapshot=snapshot)
    valid_snapshot = ValueRoute("list", children=(ValueRoute.literal(1),))
    payload = _route_payload(
        ValueRoute("scalar", children=(_SOURCE,), result_type="Thing", snapshot=valid_snapshot)
    )
    payload["snapshot"]["children"][0] = _route_payload(_SOURCE)
    payload["snapshot"]["children"][0]["source"]["run_id"] = ""
    return tree, payload


def _raised(call: Any, *args: Any) -> tuple[type[BaseException], str]:
    """Return the exception type and message ``call(*args)`` raises."""
    with pytest.raises(Exception) as caught:
        call(*args)
    return caught.type, str(caught.value)


@pytest.mark.parametrize(("build", "validated", "encoded", "decoded"), [
    (
        _duplicate_names_and_bad_child,
        (TypeError, "route expression must be a string"),
        (TypeError, "route expression must be a string"),
        (TypeError, "route expression must be a string"),
    ),
    (
        _head_error_and_bad_child,
        (ValueError, "route source cannot carry children"),
        (ValueError, "route source cannot carry children"),
        (TypeError, "route expression must be a string"),
    ),
    (
        _bad_spans_and_bad_child_snapshot,
        (ValueError, "partial snapshot cannot contain source routes"),
        (ValueError, "partial snapshot cannot contain source routes"),
        (ValueError, "partial snapshot cannot contain source routes"),
    ),
    (
        _snapshot_with_invalid_source,
        (ValueError, "source run_id must be non-empty"),
        (ValueError, "source run_id must be non-empty"),
        (ValueError, "source run_id must be non-empty"),
    ),
])
def test_route_errors_on_multi_error_trees(build, validated, encoded, decoded):
    """Validation, encoding and decoding each report the same first error they report today."""
    tree, payload = build()
    assert _raised(_validate_route, tree) == validated
    assert _raised(_route_payload, tree) == encoded
    assert _raised(_route_from_payload, payload) == decoded


def test_downstream_context_reads_are_unaliased():
    shared = {"note": ["kept"]}
    items = ({"first": shared, "again": shared}, {"second": shared})
    context = BackwardContext({}, downstream_context=items)

    read = context.downstream_context
    assert read == items
    read[0]["first"]["note"].append("changed")
    assert read[0]["again"] == read[1]["second"] == {"note": ["kept"]}
    assert shared == {"note": ["kept"]}
    assert context.downstream_context == items
    first, second = context.downstream_context, context.downstream_context
    first[1]["second"]["note"].clear()
    assert second == items


class _Checks:
    """Count route checks made through ``_validate_route``, recursive calls included."""

    def __init__(self) -> None:
        self.count = 0

    def measure(self, call: Any, *args: Any) -> int:
        """Return how many route checks ``call(*args)`` makes."""
        before = self.count
        call(*args)
        return self.count - before


@pytest.fixture
def route_checks(monkeypatch: pytest.MonkeyPatch) -> _Checks:
    """Count every ``_validate_route`` call made by contracts and the run log."""
    checks = _Checks()
    validate = contracts._validate_route

    def counted(route: ValueRoute, **kwargs: Any) -> None:
        checks.count += 1
        validate(route, **kwargs)

    # The log imports the name, and the recursion looks up the contracts global, so both are patched.
    monkeypatch.setattr(contracts, "_validate_route", counted)
    monkeypatch.setattr(run_log, "_validate_route", counted)
    return checks


def _node_count(route: ValueRoute) -> int:
    """Return the number of route nodes in ``route``, snapshots included."""
    snapshot = 0 if route.snapshot is None else _node_count(route.snapshot)
    return 1 + snapshot + sum(_node_count(child) for child in route.children)


def _concat_tree(leaves: int) -> ValueRoute:
    """Return a balanced ``concat`` route over ``leaves`` one-item source lists."""
    sources = [ValueRoute("source", source=SourceRef("run", i)) for i in range(leaves)]
    return _balanced_concat(sources, [1] * leaves)


@pytest.mark.parametrize("tree", [
    _concat_tree(64),
    _concat_tree(256),
    ValueRoute("scalar", children=(_SOURCE,) * 256, value_json="256"),
], ids=["concat-64", "concat-256", "scalar-256"])
def test_route_encode_and_decode_check_each_node_once(tree, route_checks):
    nodes = _node_count(tree)
    payload = _route_payload(tree)
    assert route_checks.measure(_route_payload, tree) == nodes
    assert route_checks.measure(_route_from_payload, payload) == nodes
    assert _route_from_payload(payload) == tree


def _run_and_resume(n: int, tmp_path: Path) -> None:
    """Run a learnable ``n``-operand list chain with a durable log, then resume it."""
    flow, inputs = _chain_flow("list", n)
    memory = _Memory()
    log = tmp_path / f"list-{n}.jsonl"
    run_flow(flow, inputs, log=log, learnable=True, memory={"test": memory})
    resume_flow(flow, log=log, memory={"test": memory})


def test_run_and_resume_route_checks_grow_linearly(tmp_path, route_checks):
    small = route_checks.measure(_run_and_resume, 50, tmp_path)
    large = route_checks.measure(_run_and_resume, 200, tmp_path)
    assert large / small <= 4.4, f"route checks: n=50 {small}, n=200 {large}"


_COUNTED = {
    "lookups": (RetainedTrace, "lookup_source"),
    "context_copies": (node_base, "_copy_backward_value"),
    "freezes": (memory_contracts, "_freeze_json"),
}
"""Calls counted during backward, by name: the owner and attribute each counter wraps.

``_copy_backward_value`` and ``_freeze_json`` are wrapped on their modules because
their recursion looks up that global, so recursive calls count too.
"""


def _backward_counts(kind: str, n: int, tmp_path: Path) -> dict[str, int]:
    """Run a learnable chain flow, then return how often each counted call ran during backward."""
    flow, inputs = _chain_flow(kind, n)
    memory = _Memory()
    run = run_flow(flow, inputs, log=tmp_path / f"{kind}-{n}.jsonl", learnable=True,
                   memory={"test": memory})
    counts = dict.fromkeys(_COUNTED, 0)

    def counter(name: str, call: Any) -> Any:
        def counted(*args: Any, **kwargs: Any) -> Any:
            counts[name] += 1
            return call(*args, **kwargs)
        return counted

    with pytest.MonkeyPatch.context() as patch:
        for name, (owner, attribute) in _COUNTED.items():
            patch.setattr(owner, attribute, counter(name, getattr(owner, attribute)))
        outcome = run.backward("good")
    assert outcome.status == "succeeded"
    return counts


@pytest.fixture(scope="module")
def backward_counts(tmp_path_factory: pytest.TempPathFactory) -> dict[tuple[str, int], dict[str, int]]:
    """Count backward calls once per chain kind at 32 and 128 operands."""
    tmp_path = tmp_path_factory.mktemp("backward-counts")
    return {(kind, n): _backward_counts(kind, n, tmp_path)
            for kind in ("list", "scalar") for n in (32, 128)}


@pytest.mark.parametrize("kind", ["list", "scalar"])
def test_backward_lookups_grow_near_linearly(kind, backward_counts):
    small, large = backward_counts[(kind, 32)]["lookups"], backward_counts[(kind, 128)]["lookups"]
    assert large / small <= 7, f"source lookups: n=32 {small}, n=128 {large}"


@pytest.mark.parametrize("kind", ["list", "scalar"])
def test_backward_context_copies_grow_near_linearly(kind, backward_counts):
    small = backward_counts[(kind, 32)]["context_copies"]
    large = backward_counts[(kind, 128)]["context_copies"]
    assert large / small <= 7, f"context copies: n=32 {small}, n=128 {large}"


@pytest.mark.parametrize("kind", ["list", "scalar"])
def test_backward_freeze_work_grows_near_linearly(kind, backward_counts):
    small = backward_counts[(kind, 32)]["freezes"]
    large = backward_counts[(kind, 128)]["freezes"]
    assert large / small <= 7, f"JSON freezes: n=32 {small}, n=128 {large}"


def _backward_peak(n: int, tmp_path: Path) -> int:
    """Run and resume a learnable ``n``-operand list chain, then return backward's peak traced bytes."""
    flow, inputs = _chain_flow("list", n)
    memory = _Memory()
    log = tmp_path / f"peak-list-{n}.jsonl"
    run = run_flow(flow, inputs, log=log, learnable=True, memory={"test": memory})
    resume_flow(flow, log=log, memory={"test": memory})
    was_tracing = tracemalloc.is_tracing()
    if was_tracing:
        tracemalloc.reset_peak()
    else:
        tracemalloc.start()
    # Memory already traced before backward (when tracing was on) is not backward's.
    baseline, _ = tracemalloc.get_traced_memory()
    try:
        outcome = run.backward("good")
        _, peak = tracemalloc.get_traced_memory()
        peak -= baseline
    finally:
        if not was_tracing:
            tracemalloc.stop()
    assert outcome.status == "succeeded"
    return peak


def test_backward_peak_memory_grows_near_linearly(tmp_path):
    small = _backward_peak(32, tmp_path)
    large = _backward_peak(128, tmp_path)
    assert large <= 8 * small, f"backward peak bytes: n=32 {small}, n=128 {large}"
