"""`run_flow(num_workers=N)` threads the engine drive mode through from the compose layer.

A fan-out/fan-in code flow (Ollama-free) is run serial (`0`) and pooled (`>=1`); the
terminal output must be identical — workers are pure executors, so the result is
worker-count-independent (only intra-run concurrency changes).
"""

from concurrent.futures import Future
import gc
import threading

import pytest

from yian.compose import load_flow
from yian.events import NodeSucceeded
from yian.run import _run_flow_controlled, run_flow
from yian.runtime.eval_node import EvaluatedOutcome
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


_LONG_LOOP = """
id: counter
name: counter
input:
  k: int
nodes:
  l:
    kind: loop
    times: 12
    input:
      n: 0
    nodes:
      bump:
        kind: code
        input:
          n: ${input.n}
        output:
          n: int
        code: |
          return {"n": inputs["n"] + 1}
    output: ${bump.output}
output: ${l.output.n}
"""


def test_pooled_dispatcher_drops_read_results():
    # A completed Future holds its node's EvaluatedOutcome; the dispatcher must not keep
    # read ones alive, or memory grows with every evaluation of a long LOOP.
    loaded = load_flow(_LONG_LOOP)
    peak = 0

    def on_event(event):
        nonlocal peak
        if isinstance(event, NodeSucceeded):
            gc.collect()
            held = [
                obj for obj in gc.get_objects()
                if isinstance(obj, Future) and obj.done() and not obj.cancelled()
                and obj.exception() is None and isinstance(obj.result(), EvaluatedOutcome)
            ]
            peak = max(peak, len(held))

    res = run_flow(loaded, {"k": 0}, num_workers=2, on_event=on_event)
    assert res.status == "succeeded", res.failure
    assert res.output == 12
    # At most the one being committed plus finished-but-unread starts.
    assert peak <= 3
