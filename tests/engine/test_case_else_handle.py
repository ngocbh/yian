"""CASE's else route has its own handle, so a node named `default` is ordinary.

A `then:` handle is its target node's name. The else (no-match) route takes
`ELSE_HANDLE`, which contains `#`, a character no node id may hold, so no `then:`
target can share the else route.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.engine.test_skipped_output_redirects import _call_redirect_flow, _router_in_flight
from yian.compose.helpers.errors import LoadError
from yian.compose.loader import load_flow
from yian.events import NodeStarted
from yian.nodes.base import ELSE_HANDLE
from yian.nodes.case import Case, CaseNode
from yian.runtime.common import NodeState
from yian.run import resume_flow, run_flow
from yian.runtime.state import commit_prepared, prepare_node_routed


_SEARCHED_FLOW = """
id: f
name: f
input: {n: int}
nodes:
  gate:
    kind: case
    cases:
      - when: "${input.n} > 5"
        then: default
    else: small
  default: {kind: code, code: "return 'big'", output: str}
  small: {kind: code, code: "return 'small'", output: str}
output: ${default.output | small.output}
"""

_NO_ELSE_FLOW = """
id: f
name: f
input: {n: int}
nodes:
  gate:
    kind: case
    cases:
      - when: "${input.n} > 5"
        then: default
  default: {kind: code, code: "return 'big'", output: str}
output: ${default.output :- "none"}
"""

_ON_FLOW = """
id: f
name: f
input: {s: str}
nodes:
  gate:
    kind: case
    on: ${input.s}
    cases:
      - when: a
        then: default
    else: other
  default: {kind: code, code: "return 'a'", output: str}
  other: {kind: code, code: "return 'other'", output: str}
output: ${default.output | other.output}
"""

_ELSE_LESS_FLOW = """
id: f
name: f
input: {n: int}
nodes:
  gate:
    kind: case
    cases:
      - when: "${input.n} > 5"
        then: big
  big: {kind: code, code: "return 'big'", output: str}
output: ${big.output :- "none"}
"""


def _started(result) -> set[str]:
    """Return the ids of every node that started in a run."""
    return {e.node_id for e in result.events if isinstance(e, NodeStarted)}


@pytest.mark.parametrize("n, ran, skipped, expected", [
    (1, "small", "default", "small"),
    (9, "default", "small", "big"),
])
def test_a_then_target_named_default_runs_only_on_its_own_branch(n, ran, skipped, expected):
    result = run_flow(load_flow(_SEARCHED_FLOW), {"n": n})

    assert result.status == "succeeded", result.failure
    assert result.output == expected
    assert ran in _started(result)
    assert skipped not in _started(result)


def test_a_then_target_named_default_does_not_run_on_no_match_without_else():
    result = run_flow(load_flow(_NO_ELSE_FLOW), {"n": 1})

    assert result.status == "succeeded", result.failure
    assert result.output == "none"
    assert "default" not in _started(result)


@pytest.mark.parametrize("s, ran, skipped, expected", [
    ("a", "default", "other", "a"),
    ("z", "other", "default", "other"),
])
def test_an_on_case_target_named_default_runs_only_on_its_own_branch(s, ran, skipped, expected):
    result = run_flow(load_flow(_ON_FLOW), {"s": s})

    assert result.status == "succeeded", result.failure
    assert result.output == expected
    assert ran in _started(result)
    assert skipped not in _started(result)


def test_the_else_handle_is_no_possible_node_id():
    assert ELSE_HANDLE == "#else"
    flow = """
id: f
name: f
nodes:
  "#else": {kind: code, code: "return 1", output: int}
output: ${#else.output}
"""
    with pytest.raises(LoadError, match="reserved separator"):
        load_flow(flow)


def test_a_case_may_not_take_the_else_handle_as_a_branch():
    with pytest.raises(ValueError, match="reserved for the else route"):
        Case(handle=ELSE_HANDLE, when="True")

    case = Case(handle="go", when="True")
    case.handle = ELSE_HANDLE
    with pytest.raises(ValueError, match="reserved for the else route"):
        CaseNode("g", [case]).validate_execution_setup()


def test_a_route_to_an_unknown_handle_named_default_is_rejected():
    state, _ = _router_in_flight(_call_redirect_flow())

    with pytest.raises(ValueError, match="matches no outgoing edge"):
        prepare_node_routed(state, "router", "default")


def test_the_else_handle_without_an_edge_skips_every_branch():
    state, log = _router_in_flight(_call_redirect_flow())

    commit_prepared(log, state, prepare_node_routed(state, "router", ELSE_HANDLE))

    assert state.edge_state["router-child-end"] is NodeState.SKIPPED


def test_a_log_routed_to_the_old_default_handle_does_not_resume(tmp_path: Path):
    path = tmp_path / "run.jsonl"
    first = run_flow(load_flow(_ELSE_LESS_FLOW), {"n": 1}, log=path)
    assert first.status == "succeeded", first.failure
    assert first.output == "none"

    lines = path.read_text().splitlines(keepends=True)
    records = [json.loads(line) for line in lines]
    routed = next(i for i, r in enumerate(records) if r["kind"] == "node_routed")
    assert records[routed]["payload"]["handle"] == ELSE_HANDLE
    records[routed]["payload"]["handle"] = "default"
    lines[routed] = json.dumps(records[routed]) + "\n"
    path.write_text("".join(lines))

    with pytest.raises(ValueError, match="matches no outgoing edge"):
        resume_flow(load_flow(_ELSE_LESS_FLOW), log=path)
