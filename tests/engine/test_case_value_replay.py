"""CASE value branches across a durable run log: replay and resume.

A typed CASE whose branch is a value records the lifted `__value_<n>` node and the
`__case_<id>` result node like any other node. Resuming a completed log replays their
recorded outputs, so the run returns the same value without starting either node or
calling the other branch's CODE again. A run stopped at a `human_input` ahead of the
CASE routes on the answer given at resume and returns the taken value.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import tests.engine._compose_codefns as codefns
from yian.compose.loader import load_flow
from yian.events import NodeStarted
from yian.nodes.base import _HostStop
from yian.run import resume_flow, run_flow


_ROUTE_FLOW = """
id: case-value-replay
name: case_value_replay
typedefs:
  Turn:
    reply: str
    exited: bool
input:
  cmd: str
  text: str
nodes:
  route:
    kind: case
    on: ${input.cmd}
    output: Turn
    cases:
      - when: exit
        then:
          reply: ${input.text}
          exited: true
    else: carry_on
  carry_on:
    kind: code
    code: tests.engine._compose_codefns:case_replay_carry_on
    input:
      text: ${input.text}
    output: Turn
output: ${route.output}
"""


_GATE_FLOW = """
id: case-value-gate
name: case_value_gate
typedefs:
  Turn:
    reply: str
    exited: bool
input:
  text: str
nodes:
  ask:
    kind: human_input
    prompt: "Command?"
  route:
    kind: case
    on: ${ask.output}
    output: Turn
    cases:
      - when: exit
        then:
          reply: ${input.text}
          exited: true
    else: carry_on
  carry_on:
    kind: code
    code: tests.engine._compose_codefns:case_replay_carry_on
    input:
      text: ${input.text}
    output: Turn
output: ${route.output}
"""


@pytest.fixture
def calls(monkeypatch) -> list[str]:
    """Install a counting CODE function on the test codefns module and return its log."""
    seen: list[str] = []

    def case_replay_carry_on(inputs: dict) -> dict:
        seen.append("carry_on")
        return {"reply": f"still here: {inputs['text']}", "exited": False}

    monkeypatch.setattr(
        codefns, case_replay_carry_on.__name__, case_replay_carry_on, raising=False
    )
    return seen


def _succeeded_values(path: Path) -> dict[str, object]:
    """Return each recorded node success in a run log, keyed by node id."""
    values: dict[str, object] = {}
    for line in path.read_text().splitlines():
        record = json.loads(line)
        if record["kind"] == "node_succeeded":
            payload = record["payload"]
            values[payload["node_id"]] = payload["value"]
    return values


def _started(result) -> set[str]:
    """Return the ids of every node that started in a run."""
    return {e.node_id for e in result.events if isinstance(e, NodeStarted)}


def test_resuming_a_run_that_took_a_value_branch_replays_the_value(
    tmp_path: Path, calls: list[str]
):
    """The value branch and result node are replayed from the log, not run again."""
    path = tmp_path / "exit.jsonl"
    first = run_flow(load_flow(_ROUTE_FLOW), {"cmd": "exit", "text": "bye"}, log=path)
    assert first.status == "succeeded", first.failure
    assert first.output == {"reply": "bye", "exited": True}
    assert calls == []

    recorded = _succeeded_values(path)
    assert recorded["__value_0"]["value"] == first.output
    assert recorded["__case_route"]["value"] == first.output
    assert "carry_on" not in recorded
    before = path.read_bytes()

    resumed = resume_flow(load_flow(_ROUTE_FLOW), log=path)

    assert resumed.status == first.status
    assert resumed.output == first.output
    assert _started(resumed) == set()
    assert calls == []
    assert path.read_bytes() == before


def test_resuming_a_run_that_took_a_node_branch_replays_without_running_code(
    tmp_path: Path, calls: list[str]
):
    """The CODE branch and the result node are replayed; the CODE is not called again."""
    path = tmp_path / "other.jsonl"
    first = run_flow(load_flow(_ROUTE_FLOW), {"cmd": "other", "text": "hi"}, log=path)
    assert first.status == "succeeded", first.failure
    assert first.output == {"reply": "still here: hi", "exited": False}
    assert calls == ["carry_on"]

    recorded = _succeeded_values(path)
    assert recorded["carry_on"]["value"] == first.output
    assert recorded["__case_route"]["value"] == first.output
    assert "__value_0" not in recorded
    before = path.read_bytes()

    calls.clear()
    resumed = resume_flow(load_flow(_ROUTE_FLOW), log=path)

    assert resumed.status == first.status
    assert resumed.output == first.output
    assert _started(resumed) == set()
    assert calls == []
    assert path.read_bytes() == before


def test_resume_at_a_gate_before_the_case_takes_the_value_branch(
    tmp_path: Path, calls: list[str]
):
    """A run stopped before the CASE routes on the answer given at resume."""
    path = tmp_path / "gate.jsonl"
    interruption = _HostStop("interrupt before human answer")

    def interrupt(**request):
        raise interruption

    with pytest.raises(_HostStop) as caught:
        run_flow(
            load_flow(_GATE_FLOW),
            {"text": "bye"},
            log=path,
            tools={"ask_user": interrupt},
        )
    assert caught.value is interruption
    assert "__value_0" not in _succeeded_values(path)
    assert "__case_route" not in _succeeded_values(path)

    resumed = resume_flow(
        load_flow(_GATE_FLOW), log=path, tools={"ask_user": lambda **request: "exit"}
    )

    assert resumed.status == "succeeded", resumed.failure
    assert resumed.output == {"reply": "bye", "exited": True}
    assert {"__value_0", "__case_route"} <= _started(resumed)
    assert "carry_on" not in _started(resumed)
    assert calls == []
