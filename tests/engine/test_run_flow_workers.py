"""`run_flow(num_workers=N)` threads the engine drive mode through from the compose layer.

A fan-out/fan-in code flow (Ollama-free) is run serial (`0`) and pooled (`>=1`); the
terminal output must be identical — workers are pure executors, so the result is
worker-count-independent (only intra-run concurrency changes).
"""

from agent_composer.compose import load_flow
from agent_composer.run import run_flow

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
    assert serial.status == "succeeded", serial.error
    assert pooled.status == "succeeded", pooled.error
    assert serial.output == pooled.output


def test_run_flow_defaults_to_serial():
    loaded = load_flow(_FANOUT)
    res = run_flow(loaded, {"topic": "X"})  # no num_workers -> 0
    assert res.status == "succeeded", res.error
