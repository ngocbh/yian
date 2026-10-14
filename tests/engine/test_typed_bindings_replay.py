"""Typed bindings across a durable run log: replay, resume, fingerprint, and recovery.

Replay folds recorded node results without binding completed nodes again; a node that
runs after resume binds (and type-checks) its inputs then. A declared binding type is
part of the flow fingerprint, and END's recorded value is re-validated against the
declared output types when the log is recovered.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import tests.engine._compose_codefns as codefns
from yian.compose.loader import load_flow
from yian.nodes.base import _HostStop
from yian.run import resume_flow, run_flow
from yian.typesys.values import TypeCheckError


_TYPED_FLOW = """
id: typed-replay
name: typed_replay
defs:
  bump:
    input:
      rounds: !int
    nodes:
      step:
        kind: code
        code: tests.engine._compose_codefns:replay_bump
        input:
          rounds: ${input.rounds}
        output: int
    output:
      rounds: ${step.output}
nodes:
  p:
    kind: code
    code: tests.engine._compose_codefns:replay_produce
    output: int
  use:
    kind: code
    code: tests.engine._compose_codefns:replay_use
    input:
      qty: !int ${p.output}
    output: int
  loop:
    kind: loop
    call: bump
    times: 2
    input:
      rounds: !int 0
output:
  total: !int ${use.output}
  rounds: ${loop.output.rounds}
"""


@pytest.fixture
def calls(monkeypatch) -> list[str]:
    """Install counting CODE functions on the test codefns module and return their log."""
    seen: list[str] = []

    def replay_produce(inputs: dict) -> int:
        seen.append("p")
        return 4

    def replay_use(inputs: dict) -> int:
        seen.append("use")
        return inputs["qty"] * 10

    def replay_bump(inputs: dict) -> int:
        seen.append("bump")
        return inputs["rounds"] + 1

    for fn in (replay_produce, replay_use, replay_bump):
        monkeypatch.setattr(codefns, fn.__name__, fn, raising=False)
    return seen


def test_resuming_a_completed_typed_run_replays_without_running_code(
    tmp_path: Path, calls: list[str]
):
    """A finished run with typed inputs, seeds, and outputs resumes to the same result."""
    path = tmp_path / "typed.jsonl"
    first = run_flow(load_flow(_TYPED_FLOW), {}, log=path)
    assert first.status == "succeeded", first.failure
    assert first.output == {"total": 40, "rounds": 2}
    assert sorted(calls) == ["bump", "bump", "p", "use"]

    calls.clear()
    resumed = resume_flow(load_flow(_TYPED_FLOW), log=path)

    assert resumed.status == first.status
    assert resumed.output == first.output
    assert calls == []


_GATE_FLOW = """
id: typed-gate-replay
name: typed_gate_replay
nodes:
  ask:
    kind: human_input
    prompt: "How many?"
  use:
    kind: code
    input:
      n: !int ${ask.output}
    code: |
      return inputs["n"] * 2
output: ${use.output}
"""


def _interrupt_then_resume(tmp_path: Path, answer: object):
    """Interrupt the gate before its answer, then resume the log with `answer`."""
    path = tmp_path / "gate.jsonl"
    interruption = _HostStop("interrupt before human answer")

    def interrupt(**request):
        raise interruption

    with pytest.raises(_HostStop) as caught:
        run_flow(load_flow(_GATE_FLOW), {}, log=path, tools={"ask_user": interrupt})
    assert caught.value is interruption

    return resume_flow(
        load_flow(_GATE_FLOW), log=path, tools={"ask_user": lambda **request: answer}
    )


def test_typed_consumer_after_resume_rejects_a_mistyped_answer(tmp_path: Path):
    """The consumer binds after resume, so its declared type checks the new answer."""
    result = _interrupt_then_resume(tmp_path, "three")

    assert result.status == "failed"
    assert result.failure.code == "binding.type"
    assert result.failure.node_id == "use"


def test_typed_consumer_after_resume_accepts_a_matching_answer(tmp_path: Path):
    """A well-typed answer supplied on resume binds and the run succeeds."""
    result = _interrupt_then_resume(tmp_path, 3)

    assert result.status == "succeeded", result.failure
    assert result.output == 6


def test_resume_rejects_a_changed_declared_binding_type(tmp_path: Path, calls: list[str]):
    """Changing only a binding's declared type changes the fingerprint, so resume refuses."""
    path = tmp_path / "typed.jsonl"
    assert run_flow(load_flow(_TYPED_FLOW), {}, log=path).status == "succeeded"
    changed = _TYPED_FLOW.replace("qty: !int ${p.output}", "qty: !float ${p.output}")
    assert changed != _TYPED_FLOW

    with pytest.raises(ValueError, match="flow fingerprint mismatch"):
        resume_flow(load_flow(changed), log=path)


def test_resume_rejects_a_recorded_end_value_outside_the_declared_output(tmp_path: Path):
    """A recorded END value that no longer fits the typed output fails recovery."""
    flow = """
id: typed-end
name: typed_end
nodes:
  n:
    kind: code
    code: |
      return 5
output:
  total: !int ${n.output}
"""
    path = tmp_path / "end.jsonl"
    first = run_flow(load_flow(flow), {}, log=path)
    assert first.status == "succeeded", first.failure
    assert first.output == {"total": 5}

    lines = path.read_text().splitlines()
    edited = 0
    for i, line in enumerate(lines):
        record = json.loads(line)
        payload = record["payload"]
        if record["kind"] == "node_succeeded" and payload["node_id"] == "__end__":
            assert payload["value"] == {"kind": "object", "value": {"total": 5}}
            payload["value"]["value"]["total"] = "five"
            lines[i] = json.dumps(record, separators=(",", ":"), sort_keys=True)
            edited += 1
    assert edited == 1
    path.write_text("\n".join(lines) + "\n")

    with pytest.raises(TypeCheckError, match="does not match declared type int"):
        resume_flow(load_flow(flow), log=path)
