"""Ints of any size are recorded in the run log, replayed and resumed exactly."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

import yian.json_codec as json_codec
from yian import load_flow, run_flow
from yian.events import NodeStarted, NodeSucceeded
from yian.nodes.base import _HostStop
from yian.run import resume_flow
from yian.runtime.log import decode_record
from yian.typesys import VariablePool


BIG = 10**4999 + 12345  # 5000 digits, past Python's default 4300-digit limit
DIGITS = json_codec.int_text(BIG)


@pytest.fixture(autouse=True)
def _default_int_digit_limit():
    """Run each test under Python's default int/str digit limit."""
    get = getattr(sys, "get_int_max_str_digits", None)
    if get is None:
        yield  # this Python has no limit
        return
    before = get()
    sys.set_int_max_str_digits(4300)
    try:
        yield
        after = get()
    finally:
        sys.set_int_max_str_digits(before)
    assert after == 4300, "yian code changed the global digit limit"


_INPUT_FLOW = """
id: big-input
name: big_input
input:
  xs: list[int]
nodes:
  n:
    kind: code
    input:
      xs: ${input.xs}
    code: |
      return inputs["xs"][0] % 10
output:
  seen: ${input.xs}
  mod: ${n.output}
"""

_OUTPUT_FLOW = """
id: big-output
name: big_output
input:
  n: int
nodes:
  t:
    kind: code
    code: |
      return 0
output: ${10 ** input.n + t.output}
"""

_GATE_FLOW = """
id: big-gate
name: big_gate
input:
  x: int
nodes:
  inc:
    kind: code
    input:
      x: ${input.x}
    code: |
      return inputs["x"] + 1
  ask:
    kind: human_input
    input:
      n: ${inc.output}
    prompt: "Answer?"
    output: str
  both:
    kind: code
    input:
      n: ${inc.output}
      answer: ${ask.output}
    code: |
      return {"n": inputs["n"], "answer": inputs["answer"]}
output: ${both.output}
"""


def _keyed_flow(value: str) -> str:
    """Return a flow whose `n` CODE node returns `{"b": 1, "a": <value>}`."""
    return f"""
id: big-keys
name: big_keys
nodes:
  n:
    kind: code
    code: |
      return {{"b": 1, "a": {value}}}
output: ${{n.output}}
"""


def _assert_replays(text: str, path: Path, first) -> None:
    """Assert resuming the completed log at `path` returns `first` without writing."""
    before = path.read_bytes()

    resumed = resume_flow(load_flow(text), log=path)

    assert resumed.status == first.status
    assert resumed.output == first.output
    assert not [event for event in resumed.events if isinstance(event, NodeStarted)]
    assert path.read_bytes() == before


def test_a_big_int_input_is_recorded_and_replayed(tmp_path: Path):
    path = tmp_path / "input.jsonl"

    result = run_flow(load_flow(_INPUT_FLOW), {"xs": [BIG, -BIG]}, log=path)

    assert result.status == "succeeded", result.failure
    assert result.output == {"seen": [BIG, -BIG], "mod": 5}
    assert result.input == {"xs": [BIG, -BIG]}
    text = path.read_text()
    assert DIGITS in text
    for line in text.splitlines():
        decode_record(line)
    _assert_replays(_INPUT_FLOW, path, result)


def test_a_big_int_input_runs_without_a_log():
    result = run_flow(load_flow(_INPUT_FLOW), {"xs": [BIG, -BIG]})

    assert result.status == "succeeded", result.failure
    assert result.output == {"seen": [BIG, -BIG], "mod": 5}


def test_a_big_int_output_is_recorded_and_replayed(tmp_path: Path):
    path = tmp_path / "output.jsonl"

    result = run_flow(load_flow(_OUTPUT_FLOW), {"n": 4999}, log=path)

    assert result.status == "succeeded", result.failure
    assert result.output == 10**4999
    assert json_codec.int_text(10**4999) in path.read_text()
    _assert_replays(_OUTPUT_FLOW, path, result)


@pytest.mark.parametrize("learnable", [False, True], ids=["plain", "learnable"])
def test_a_big_int_output_runs_without_a_log(learnable: bool):
    result = run_flow(load_flow(_OUTPUT_FLOW), {"n": 4999}, learnable=learnable)

    assert result.status == "succeeded", result.failure
    assert result.output == 10**4999


def test_node_events_hold_the_exact_big_ints(tmp_path: Path):
    result = run_flow(load_flow(_GATE_FLOW), {"x": BIG}, log=tmp_path / "events.jsonl",
                      tools={"ask_user": lambda **request: "yes"})

    assert result.status == "succeeded", result.failure
    assert result.input == {"x": BIG}
    outputs = {e.node_id: e.output for e in result.events if isinstance(e, NodeSucceeded)}
    assert outputs["inc"] == BIG + 1
    assert outputs["both"] == {"n": BIG + 1, "answer": "yes"}


def test_a_run_stopped_at_a_gate_resumes_with_the_big_int(tmp_path: Path):
    path = tmp_path / "gate.jsonl"
    interruption = _HostStop("interrupt before human answer")

    def interrupt(**request):
        raise interruption

    with pytest.raises(_HostStop) as caught:
        run_flow(load_flow(_GATE_FLOW), {"x": BIG}, log=path, tools={"ask_user": interrupt})
    assert caught.value is interruption
    assert json_codec.int_text(BIG + 1) in path.read_text()

    resumed = resume_flow(
        load_flow(_GATE_FLOW), log=path, tools={"ask_user": lambda **request: "yes"}
    )

    assert resumed.status == "succeeded", resumed.failure
    assert resumed.output == {"n": BIG + 1, "answer": "yes"}
    assert "inc" not in {e.node_id for e in resumed.events if isinstance(e, NodeStarted)}


def test_a_big_int_object_keeps_its_key_order_live_and_on_replay(tmp_path: Path):
    big_path = tmp_path / "big.jsonl"
    small_path = tmp_path / "small.jsonl"
    big_text = _keyed_flow("10**4999 + 12345")
    small_text = _keyed_flow("2")
    big = run_flow(load_flow(big_text), {}, log=big_path)
    small = run_flow(load_flow(small_text), {}, log=small_path)
    assert big.status == "succeeded", big.failure
    assert big.output == {"b": 1, "a": BIG}
    assert list(big.output) == list(small.output)

    big_replay = resume_flow(load_flow(big_text), log=big_path)
    small_replay = resume_flow(load_flow(small_text), log=small_path)

    assert big_replay.output == {"b": 1, "a": BIG}
    assert list(big_replay.output) == list(small_replay.output)


def test_a_pool_holding_a_big_int_round_trips_through_its_text():
    pool = VariablePool(start_id="start")
    pool.set("start", {"x": BIG, "xs": [-BIG]})
    pool.add_system("run_id", "run-big")

    loaded = VariablePool.loads(pool.dumps())

    assert loaded.get("start") == {"x": BIG, "xs": [-BIG]}
    assert loaded.dumps() == pool.dumps()
