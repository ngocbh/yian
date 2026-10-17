"""`RunStarted.call_time_llm`: the record of agents whose `llm_config` resolves when they run."""

import copy
import json

import pytest

from yian.compile.model import END_ID, START_ID, CompiledFlow, Edge
from yian.nodes.end import EndNode
from yian.nodes.start import StartNode
from yian.runtime.log import (
    SCHEMA_VERSION,
    MemoryRunLog,
    RecordEnvelope,
    RunStartedRecord,
    decode_record,
    encode_pool,
    encode_record,
)
from yian.runtime.state import RunStartData, RunState, commit_prepared, prepare_run_started
from yian.typesys.pool import VariablePool
from tests.engine._fakes import FuncNode

_FINGERPRINT = "a" * 64
_EFFECTIVE = {"agent": {"provider": "openai", "model": "gpt-test"}}
_CALL_TIME = {
    "selection": {
        "defaults": {"provider": "openai", "model": "gpt-saved"},
        "provider_models": {"openai": "gpt-saved"},
        "connections": {
            "openai": {
                "method": "api_key",
                "source": "missing",
                "credential_id": None,
                "endpoint": "https://api.openai.com/v1",
                "store_path": None,
            }
        },
        "api_endpoint": "https://api.openai.com/v1",
        "stray_model": False,
        "host": None,
    },
    "agents": {"chat": {"provider": "openai"}, "c/a": {}},
}

# A run without call-time agents encodes exactly as before the field existed.
_GOLDEN = (
    b'{"kind":"run_started","payload":{"effective_llm_by_static_path":{"agent":'
    b'{"model":"gpt-test","provider":"openai"}},"flow_fingerprint":"'
    + b"a" * 64
    + b'","initial_pool":{"start_id":"start","store":{"start":{"kind":"object","value":'
    b'{"name":"Ada"}}},"system":{"run_id":{"kind":"str","value":"r1"}}},"learnable":false,'
    b'"memories":null,"memory_snapshots":null,"memory_stacks":null,"num_workers":2,'
    b'"start_wiring_override":{"name":"${input.name}"}},"run_id":"r1","schema":9,"seq":0}'
)


def _record(**overrides):
    pool = VariablePool(start_id="start")
    pool.set("start", {"name": "Ada"})
    pool.add_system("run_id", "r1")
    fields = {
        "flow_fingerprint": _FINGERPRINT,
        "initial_pool": encode_pool(pool),
        "start_wiring_override": {"name": "${input.name}"},
        "effective_llm_by_static_path": _EFFECTIVE,
        "num_workers": 2,
    }
    fields.update(overrides)
    return RunStartedRecord(**fields)


def _encode(record):
    return encode_record(RecordEnvelope(SCHEMA_VERSION, "r1", 0, record))


def test_the_schema_stays_nine():
    assert SCHEMA_VERSION == 9
    assert json.loads(_encode(_record(call_time_llm=_CALL_TIME)))["schema"] == 9


def test_call_time_llm_round_trips():
    data = _encode(_record(call_time_llm=copy.deepcopy(_CALL_TIME)))
    assert json.loads(data)["payload"]["call_time_llm"] == _CALL_TIME
    decoded = decode_record(data).record
    assert decoded.call_time_llm == _CALL_TIME
    assert decoded == _record(call_time_llm=_CALL_TIME)
    assert _encode(decoded) == data


def test_a_run_without_call_time_agents_writes_the_same_bytes():
    record = _record()
    assert record.call_time_llm is None
    assert _encode(record) == _GOLDEN
    assert decode_record(_GOLDEN).record.call_time_llm is None


def test_the_record_keeps_its_own_copy():
    given = copy.deepcopy(_CALL_TIME)
    record = _record(call_time_llm=given)
    given["agents"]["chat"]["provider"] = "anthropic"
    assert record.call_time_llm == _CALL_TIME


_MALFORMED = [
    [],
    {},
    {"agents": _CALL_TIME["agents"]},
    {"selection": _CALL_TIME["selection"]},
    {**_CALL_TIME, "extra": 1},
    {**_CALL_TIME, "agents": {}},
    {**_CALL_TIME, "agents": {"chat": "openai"}},
    {**_CALL_TIME, "agents": {"": {}}},
    {**_CALL_TIME, "agents": []},
    {**_CALL_TIME, "selection": "x"},
    {**_CALL_TIME, "agents": {"chat": {"temperature": float("nan")}}},
]


@pytest.mark.parametrize("value", _MALFORMED)
def test_a_malformed_record_is_rejected(value):
    with pytest.raises((TypeError, ValueError), match="call_time_llm"):
        _record(call_time_llm=value)


@pytest.mark.parametrize("value", [*_MALFORMED[:-1], None])
def test_a_malformed_payload_is_rejected(value):
    wire = json.loads(_encode(_record()))
    wire["payload"]["call_time_llm"] = value
    with pytest.raises(ValueError):
        decode_record(json.dumps(wire))


def _flow() -> CompiledFlow:
    nodes = {
        START_ID: StartNode(START_ID),
        "work": FuncNode("work", lambda inputs: inputs),
        END_ID: EndNode.value(END_ID, output_name=None),
    }
    return CompiledFlow.from_parts(
        nodes=nodes,
        edges=[
            Edge(id="start-work", from_=START_ID, to="work"),
            Edge(id="work-end", from_="work", to=END_ID),
        ],
    )


def _start(state, call_time_llm):
    pool = VariablePool(start_id=state.flow.start_id)
    pool.set(state.flow.start_id, {})
    pool.add_system("run_id", "r1")
    return prepare_run_started(
        state,
        RunStartData(
            run_id="r1",
            flow_fingerprint=_FINGERPRINT,
            initial_pool=pool,
            start_wiring_override={},
            effective_llm_by_static_path=_EFFECTIVE,
            num_workers=1,
            call_time_llm=call_time_llm,
        ),
    )


def test_run_start_records_and_applies_call_time_llm():
    state = RunState(_flow())
    assert state.call_time_llm is None
    authority = MemoryRunLog("r1")
    commit_prepared(authority, state, _start(state, copy.deepcopy(_CALL_TIME)))
    assert state.call_time_llm == _CALL_TIME
    assert authority.records()[0].record.call_time_llm == _CALL_TIME


def test_run_start_without_call_time_agents_leaves_it_none():
    state = RunState(_flow())
    commit_prepared(MemoryRunLog("r1"), state, _start(state, None))
    assert state.call_time_llm is None


def test_run_start_requires_a_pristine_call_time_llm():
    state = RunState(_flow())
    state.call_time_llm = copy.deepcopy(_CALL_TIME)
    with pytest.raises(ValueError, match="pristine"):
        _start(state, None)
