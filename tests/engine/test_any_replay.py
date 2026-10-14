"""Values stored for `Any` across a durable run log: replay and resume keep them unchanged.

A value typed `Any` keeps its natural carrier (an int stays an int, a bool stays a bool,
a list stays a `ListAnyValue`). Replay re-checks each recorded value against its declared
type and requires the same canonical wire back, so that check must be idempotent; a node
that runs after resume must see the same Python values a straight run sees.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from yian.compose.loader import load_flow
from yian.events import NodeStarted
from yian.nodes.base import _HostStop
from yian.run import resume_flow, run_flow
from yian.runtime.log import decode_record
from yian.typesys import ValueKind


_ANY_FLOW = """
id: any-replay
name: any_replay
input:
  x: !Any
nodes:
  mixed:
    kind: code
    output: Any
    code: |
      return [1, 2.0, True, None]
  obj:
    kind: code
    output: Any
    code: |
      return {"a": [1, None]}
  big:
    kind: code
    output: Any
    code: |
      return 2**80
  flt:
    kind: code
    output: Any
    code: |
      return 1.0
  items:
    kind: code
    output: list(Any)
    code: |
      return [1, "two"]
  nothing:
    kind: code
    output: Optional(Any)
    code: |
      return None
  rec:
    kind: code
    output:
      v: Any
    code: |
      return {"v": [1, True]}
  ask:
    kind: human_input
    input:
      mixed: ${mixed.output}
      obj: ${obj.output}
      big: ${big.output}
      flt: ${flt.output}
      items: ${items.output}
      nothing: ${nothing.output}
      rec: ${rec.output}
    prompt: "Go on?"
    output: str
  seen:
    kind: code
    input:
      x: ${input.x}
      mixed: ${mixed.output}
      obj: ${obj.output}
      big: ${big.output}
      flt: ${flt.output}
      items: ${items.output}
      nothing: ${nothing.output}
      rec: ${rec.output}
      answer: ${ask.output}
    code: |
      def names(v):
          if isinstance(v, list):
              return [names(i) for i in v]
          if isinstance(v, dict):
              return {k: names(i) for k, i in v.items()}
          return type(v).__name__
      return {k: names(v) for k, v in inputs.items() if k != "answer"}
output:
  x: ${input.x}
  mixed: ${mixed.output}
  obj: ${obj.output}
  big: ${big.output}
  flt: ${flt.output}
  items: ${items.output}
  nothing: ${nothing.output}
  rec: ${rec.output}
  seen: ${seen.output}
"""

_INPUT = {"x": [1, 2]}

_ANY_NODES = ("mixed", "obj", "big", "flt", "items", "nothing", "rec")

_EXPECTED_OUTPUT = {
    "x": [1, 2],
    "mixed": [1, 2.0, True, None],
    "obj": {"a": [1, None]},
    "big": 2**80,
    "flt": 1.0,
    "items": [1, "two"],
    "nothing": None,
    "rec": {"v": [1, True]},
    "seen": {
        "x": ["int", "int"],
        "mixed": ["int", "float", "bool", "NoneType"],
        "obj": {"a": ["int", "NoneType"]},
        "big": "int",
        "flt": "float",
        "items": ["int", "str"],
        "nothing": "NoneType",
        "rec": {"v": ["int", "bool"]},
    },
}

_EXPECTED_KINDS = {
    "mixed": ValueKind.LIST_ANY,
    "obj": ValueKind.OBJECT,
    "big": ValueKind.INTEGER,
    "flt": ValueKind.NUMBER,
    "items": ValueKind.LIST_ANY,
    "nothing": ValueKind.NONE,
    "rec": ValueKind.OBJECT,
}


def _answer(**request) -> str:
    """Answer the gate."""
    return "yes"


def _committed(path: Path) -> dict[str, dict]:
    """Return each Any node's committed wire value from the log at `path`."""
    values: dict[str, dict] = {}
    for line in path.read_text().splitlines():
        decode_record(line)  # every record still validates
        raw = json.loads(line)
        if raw["kind"] == "node_succeeded" and raw["payload"]["node_id"] in _ANY_NODES:
            values[raw["payload"]["node_id"]] = raw["payload"]["value"]
    return values


def _assert_exact(actual: object, expected: object) -> None:
    """Assert `actual == expected` with the same Python type at every level."""
    assert type(actual) is type(expected), (actual, expected)
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_exact(actual[key], expected[key])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for got, want in zip(actual, expected):
            _assert_exact(got, want)
    else:
        assert actual == expected


def _straight_run(path: Path):
    """Run the flow to completion with a log at `path`."""
    result = run_flow(load_flow(_ANY_FLOW), _INPUT, log=path, tools={"ask_user": _answer})
    assert result.status == "succeeded", result.failure
    return result


def test_a_straight_run_stores_any_values_in_their_natural_carriers(tmp_path: Path):
    """Each value typed `Any` is committed as its natural carrier, unwidened."""
    path = tmp_path / "straight.jsonl"
    result = _straight_run(path)

    _assert_exact(result.output, _EXPECTED_OUTPUT)
    committed = _committed(path)
    assert {node: value["kind"] for node, value in committed.items()} == _EXPECTED_KINDS
    _assert_exact(committed["mixed"]["value"], [1, 2.0, True, None])
    _assert_exact(committed["items"]["value"], [1, "two"])
    _assert_exact(committed["rec"]["value"], {"v": [1, True]})
    _assert_exact(committed["obj"]["value"], {"a": [1, None]})
    _assert_exact(committed["big"]["value"], 2**80)
    started = json.loads(path.read_text().splitlines()[0])["payload"]["initial_pool"]
    assert started["store"]["__start__"] == {"kind": ValueKind.OBJECT, "value": _INPUT}


def test_resuming_a_completed_any_run_returns_the_same_output(tmp_path: Path):
    """Recovering a finished log returns the same exact output without running a node."""
    path = tmp_path / "done.jsonl"
    first = _straight_run(path)
    before = path.read_bytes()

    resumed = resume_flow(load_flow(_ANY_FLOW), log=path, tools={"ask_user": _answer})

    assert resumed.status == "succeeded", resumed.failure
    _assert_exact(resumed.output, first.output)
    _assert_exact(resumed.input, _INPUT)
    assert not [event for event in resumed.events if isinstance(event, NodeStarted)]
    assert path.read_bytes() == before


def test_a_node_after_resume_reads_replayed_any_values_unchanged(tmp_path: Path):
    """Interrupted after the Any nodes commit, a resumed run matches a straight run."""
    straight = tmp_path / "straight.jsonl"
    expected = _straight_run(straight)

    path = tmp_path / "stopped.jsonl"
    interruption = _HostStop("interrupt before human answer")

    def interrupt(**request):
        raise interruption

    with pytest.raises(_HostStop) as caught:
        run_flow(load_flow(_ANY_FLOW), _INPUT, log=path, tools={"ask_user": interrupt})
    assert caught.value is interruption
    _assert_exact(_committed(path), _committed(straight))

    resumed = resume_flow(load_flow(_ANY_FLOW), log=path, tools={"ask_user": _answer})

    assert resumed.status == "succeeded", resumed.failure
    _assert_exact(resumed.output, expected.output)
    started = {event.node_id for event in resumed.events if isinstance(event, NodeStarted)}
    assert started.isdisjoint(_ANY_NODES)
    assert "seen" in started
    _assert_exact(_committed(path), _committed(straight))
