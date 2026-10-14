"""A run records the values that differ from the built-in ones, and a resume checks them."""

from copy import deepcopy
import json

import pytest
from langchain_core.messages import AIMessage
from pydantic import ValidationError

import yian.llm_clients as llm_clients_mod
from yian import defaults, load_flow, run_flow
from yian.compile.expand import build_grow
from yian.compile.manifest import flow_fingerprint
from yian.defaults import Defaults, DefaultsError, DefaultsMismatchError
from yian.run import resume_flow
from yian.runtime.log import (
    MemoryRunLog,
    NodeExpandedRecord,
    RecordEnvelope,
    RunStartedRecord,
    decode_record,
    encode_record,
)
from yian.runtime.state import RunState, _replay_records
from tests.engine.test_agent import _FakeChat
from tests.engine.test_run_records import _RECORDED_ROUTE


_TWO_STEPS = """
id: two
name: two
nodes:
  first:
    kind: code
    code: |
      return {"n": 1}
  second:
    kind: code
    input:
      n: ${first.output.n}
    code: |
      return {"n": inputs["n"] + 1}
output: ${second.output}
"""

_MAP = """
id: fan
name: fan
input:
  xs: !list(int)
defs:
  one:
    input:
      v: !int
    nodes:
      echo:
        kind: code
        input:
          v: ${input.v}
        code: |
          return {"v": inputs["v"]}
    output: ${echo.output}
nodes:
  each:
    kind: map
    over: ${input.xs}
    call: one
    input:
      v: ${item}
output: ${each.output}
"""

_AGENT = """
id: talk
name: talk
nodes:
  ask:
    kind: agent
    prompt: say hi
output: ${ask.output}
"""


def _lines(path):
    return path.read_bytes().splitlines(keepends=True)


def _run_started_payload(path) -> dict:
    return json.loads(_lines(path)[0])["payload"]


def _rewrite_run_started(path, edit) -> None:
    """Apply `edit` to the `run_started` payload and write it back canonically."""
    lines = _lines(path)
    envelope = json.loads(lines[0])
    edit(envelope["payload"])
    lines[0] = encode_record(decode_record(json.dumps(envelope).encode())) + b"\n"
    path.write_bytes(b"".join(lines))


def _interrupted_run(path) -> None:
    """Run the two-step flow, then keep only the log up to START's success."""
    result = run_flow(load_flow(_TWO_STEPS), {}, log=path)
    assert result.status == "succeeded"
    path.write_bytes(b"".join(_lines(path)[:3]))


def test_resume_with_same_value_succeeds(set_defaults, tmp_path):
    path = tmp_path / "run.jsonl"
    set_defaults(MAX_TOTAL_NODES=5000)
    _interrupted_run(path)

    resumed = resume_flow(load_flow(_TWO_STEPS), log=path)

    assert resumed.status == "succeeded"
    assert resumed.output == {"n": 2}


def test_default_run_log_is_unchanged(tmp_path):
    envelope = decode_record(_RECORDED_ROUTE[0])
    assert envelope.record.defaults == {}
    assert encode_record(envelope) == _RECORDED_ROUTE[0]

    path = tmp_path / "run.jsonl"
    assert run_flow(load_flow(_TWO_STEPS), {}, log=path).status == "succeeded"
    assert "defaults" not in _run_started_payload(path)


def test_non_default_value_is_recorded(set_defaults, tmp_path):
    path = tmp_path / "run.jsonl"
    set_defaults(MAX_TOTAL_NODES=5000)

    assert run_flow(load_flow(_TWO_STEPS), {}, log=path).status == "succeeded"

    assert _run_started_payload(path)["defaults"] == {"MAX_TOTAL_NODES": 5000}
    assert decode_record(_lines(path)[0]).record.defaults == {"MAX_TOTAL_NODES": 5000}


def test_resume_with_a_different_value_fails(set_defaults, tmp_path):
    path = tmp_path / "run.jsonl"
    set_defaults(MAX_TOTAL_NODES=5000)
    _interrupted_run(path)
    before = path.read_bytes()
    set_defaults(MAX_TOTAL_NODES=6000)

    with pytest.raises(DefaultsMismatchError) as caught:
        resume_flow(load_flow(_TWO_STEPS), log=path)

    message = str(caught.value)
    assert "YIAN_MAX_TOTAL_NODES is 6000 here" in message
    assert "started with 5000" in message
    assert path.read_bytes() == before


def test_resume_without_the_variable_fails(set_defaults, monkeypatch, tmp_path):
    path = tmp_path / "run.jsonl"
    set_defaults(MAX_TOTAL_NODES=5000)
    _interrupted_run(path)
    monkeypatch.delenv("YIAN_MAX_TOTAL_NODES")
    defaults._reload()

    with pytest.raises(DefaultsMismatchError, match="set YIAN_MAX_TOTAL_NODES=5000"):
        resume_flow(load_flow(_TWO_STEPS), log=path)


def test_resume_of_default_run_with_a_variable_set_fails(set_defaults, tmp_path):
    path = tmp_path / "run.jsonl"
    _interrupted_run(path)
    set_defaults(MAX_TOTAL_NODES=5000)

    with pytest.raises(DefaultsMismatchError, match="unset YIAN_MAX_TOTAL_NODES"):
        resume_flow(load_flow(_TWO_STEPS), log=path)


def test_old_log_resumes_as_all_defaults(set_defaults, monkeypatch, tmp_path):
    plain = tmp_path / "plain.jsonl"
    _interrupted_run(plain)
    assert resume_flow(load_flow(_TWO_STEPS), log=plain).output == {"n": 2}

    stripped = tmp_path / "stripped.jsonl"
    set_defaults(MAX_TOTAL_NODES=5000)
    _interrupted_run(stripped)
    _rewrite_run_started(stripped, lambda payload: payload.pop("defaults"))
    monkeypatch.delenv("YIAN_MAX_TOTAL_NODES")
    defaults._reload()

    assert resume_flow(load_flow(_TWO_STEPS), log=stripped).output == {"n": 2}


def test_mismatch_is_reported_before_the_fingerprint(set_defaults, monkeypatch, tmp_path):
    monkeypatch.setattr(
        llm_clients_mod, "model_from_config", lambda cfg: _FakeChat([AIMessage(content="hi")]),
    )
    path = tmp_path / "run.jsonl"
    set_defaults(AGENT_RETRIES=0)
    zero_retries = load_flow(_AGENT)
    assert run_flow(zero_retries, {}, log=path).status == "succeeded"
    monkeypatch.delenv("YIAN_AGENT_RETRIES")
    defaults._reload()
    default_retries = load_flow(_AGENT)
    assert flow_fingerprint(default_retries.compiled) != flow_fingerprint(zero_retries.compiled)

    # Resuming a finished run is enough: the check runs before anything else in replay.
    with pytest.raises(DefaultsMismatchError) as caught:
        resume_flow(default_retries, log=path)

    assert "YIAN_AGENT_RETRIES" in str(caught.value)
    assert "fingerprint" not in str(caught.value)


def test_unknown_recorded_name_fails(tmp_path):
    path = tmp_path / "run.jsonl"
    _interrupted_run(path)
    _rewrite_run_started(path, lambda payload: payload.update(defaults={"MAX_NOT_A_THING": 1}))

    with pytest.raises(DefaultsMismatchError, match="YIAN_MAX_NOT_A_THING"):
        resume_flow(load_flow(_TWO_STEPS), log=path)


def test_variable_changed_after_first_read_is_ignored(monkeypatch, tmp_path):
    defaults.current()
    monkeypatch.setenv("YIAN_MAX_TOTAL_NODES", "5000")
    path = tmp_path / "run.jsonl"

    assert run_flow(load_flow(_TWO_STEPS), {}, log=path).status == "succeeded"

    assert "defaults" not in _run_started_payload(path)


def test_bad_variable_fails_before_load(monkeypatch, tmp_path):
    flow = load_flow(_TWO_STEPS)
    monkeypatch.setenv("YIAN_MAX_TOTAL_NODES", "abc")
    defaults._forget()

    with pytest.raises(DefaultsError, match="YIAN_MAX_TOTAL_NODES"):
        load_flow(_TWO_STEPS)

    defaults._forget()
    path = tmp_path / "run.jsonl"
    with pytest.raises(DefaultsError, match="YIAN_MAX_TOTAL_NODES"):
        run_flow(flow, {}, log=path)
    assert not path.exists()


def test_node_budget_over_the_limit_fails_live(set_defaults):
    set_defaults(MAX_TOTAL_NODES=3)

    result = run_flow(load_flow(_MAP), {"xs": [1, 2, 3, 4, 5]})

    assert result.status == "failed"
    assert "expansion exceeded node budget (3)" in result.failure.message


def test_replay_takes_the_node_budget_from_the_record(set_defaults, tmp_path):
    flow = load_flow(_MAP)
    path = tmp_path / "run.jsonl"
    assert run_flow(flow, {"xs": [1, 2, 3, 4, 5]}, log=path).status == "succeeded"
    envelopes = tuple(decode_record(line) for line in _lines(path))
    assert any(isinstance(envelope.record, NodeExpandedRecord) for envelope in envelopes)
    set_defaults.unchecked(max_total_nodes=2)

    state = _replay_records(
        RunState(deepcopy(flow.compiled)), envelopes, grow_builder=build_grow,
    )

    assert len(state.flow.nodes) > 2
    assert state.run_defaults == Defaults()


def test_replay_takes_a_recorded_node_budget_from_the_record(set_defaults, tmp_path):
    flow = load_flow(_MAP)
    path = tmp_path / "run.jsonl"
    set_defaults(MAX_TOTAL_NODES=5000)
    assert run_flow(flow, {"xs": [1, 2, 3, 4, 5]}, log=path).status == "succeeded"
    envelopes = tuple(decode_record(line) for line in _lines(path))
    set_defaults.unchecked(max_total_nodes=2)

    state = _replay_records(
        RunState(deepcopy(flow.compiled)), envelopes, grow_builder=build_grow,
    )

    assert state.run_defaults.max_total_nodes == 5000


def _run_started(**extra) -> RunStartedRecord:
    record = decode_record(_RECORDED_ROUTE[0]).record
    return RunStartedRecord(
        record.flow_fingerprint,
        record.initial_pool,
        record.start_wiring_override,
        record.effective_llm_by_static_path,
        record.num_workers,
        learnable=record.learnable,
        **extra,
    )


def test_recorded_values_encode_and_decode():
    envelope = RecordEnvelope(9, "direct-engine-test", 0, _run_started(defaults={"MAX_TOTAL_NODES": 5000}))

    encoded = encode_record(envelope)

    assert json.loads(encoded)["payload"]["defaults"] == {"MAX_TOTAL_NODES": 5000}
    assert decode_record(encoded).record.defaults == {"MAX_TOTAL_NODES": 5000}


def test_recorded_values_are_sorted_by_name():
    record = _run_started(defaults={"MAX_TOTAL_NODES": 5000, "AGENT_RETRIES": 0})

    encoded = encode_record(RecordEnvelope(9, "direct-engine-test", 0, record))

    assert list(json.loads(encoded)["payload"]["defaults"]) == ["AGENT_RETRIES", "MAX_TOTAL_NODES"]


@pytest.mark.parametrize(
    "recorded",
    [{"MAX_TOTAL_NODES": "5000"}, {"MAX_TOTAL_NODES": True}, {"MAX_TOTAL_NODES": 5.0}, {"": 1}],
    ids=["string", "bool", "float", "empty-name"],
)
def test_decode_rejects_a_bad_recorded_value(recorded):
    envelope = json.loads(_RECORDED_ROUTE[0])
    envelope["payload"]["defaults"] = recorded

    with pytest.raises(ValidationError, match="defaults"):
        decode_record(json.dumps(envelope).encode())


@pytest.mark.parametrize(
    "recorded",
    [
        {"MAX_TOTAL_NODES": "5000"},
        {"MAX_TOTAL_NODES": True},
        {"MAX_TOTAL_NODES": 5000.0},
        {"": 1},
        {1: 1},
        [("MAX_TOTAL_NODES", 5000)],
    ],
    ids=["string", "bool", "float", "empty-name", "int-name", "pairs"],
)
def test_record_rejects_a_bad_recorded_value(recorded):
    with pytest.raises((TypeError, ValueError)):
        _run_started(defaults=recorded)


def test_record_rejects_a_list_of_pairs_as_a_type_error():
    with pytest.raises(TypeError, match="mapping"):
        _run_started(defaults=[("MAX_TOTAL_NODES", 5000)])


def test_record_keeps_its_own_copy_of_the_values():
    recorded = {"MAX_TOTAL_NODES": 5000}
    record = _run_started(defaults=recorded)

    recorded["MAX_TOTAL_NODES"] = 6000

    assert record.defaults == {"MAX_TOTAL_NODES": 5000}


def test_append_revalidates_the_recorded_values():
    record = _run_started(defaults={"MAX_TOTAL_NODES": 5000})
    object.__setattr__(record, "defaults", {"MAX_TOTAL_NODES": "5000"})
    log = MemoryRunLog("direct-engine-test")

    with pytest.raises(TypeError, match="MAX_TOTAL_NODES"):
        log.append(record)
    assert log.next_sequence == 0
