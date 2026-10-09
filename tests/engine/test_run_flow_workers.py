"""`run_flow(num_workers=N)` threads the engine drive mode through from the compose layer.

A fan-out/fan-in code flow (Ollama-free) is run serial (`0`) and pooled (`>=1`); the
terminal output must be identical — workers are pure executors, so the result is
worker-count-independent (only intra-run concurrency changes).
"""

import threading

import pytest

from yian.compose import load_flow
from yian.run import _run_flow_controlled, run_flow
from yian.runtime.log import JsonlRunLog

# topic -> {positive, cautious} in parallel -> a code node fans them in.
_FANOUT = """
id: angles
name: angles
input:
  topic: str
nodes:
  pro:
    kind: code
    input:
      topic: ${input.topic}
    output: str
    code: tests.engine._compose_codefns:positive
  con:
    kind: code
    input:
      topic: ${input.topic}
    output: str
    code: tests.engine._compose_codefns:cautious
  merge:
    kind: code
    input:
      pro: ${pro.output}
      con: ${con.output}
    output: str
    code: tests.engine._compose_codefns:join_two
output: ${merge.output}
"""


def test_run_flow_serial_and_pooled_same_output():
    loaded = load_flow(_FANOUT)
    serial = run_flow(loaded, {"topic": "X"}, num_workers=0)
    pooled = run_flow(loaded, {"topic": "X"}, num_workers=4)
    assert serial.status == "succeeded", serial.failure
    assert pooled.status == "succeeded", pooled.failure
    assert serial.output == pooled.output


@pytest.mark.parametrize("runner", [run_flow, _run_flow_controlled])
def test_run_flow_defaults_to_eight_workers(runner, tmp_path):
    loaded = load_flow(_FANOUT)
    path = tmp_path / "run.jsonl"
    kwargs = {"stop_event": threading.Event()} if runner is _run_flow_controlled else {}
    res = runner(loaded, {"topic": "X"}, log=path, **kwargs)
    assert res.status == "succeeded", res.failure
    with JsonlRunLog.resume(path) as log:
        assert next(iter(log)).record.num_workers == 8
