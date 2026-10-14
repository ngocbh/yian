"""Public backward failures, durable attempt prefixes, and whole-pass retries."""

from asyncio import CancelledError
from concurrent.futures import CancelledError as FutureCancelledError
from copy import deepcopy
import json
from textwrap import indent

from langchain_core.messages import AIMessage
import pytest

import yian.llm_clients as llm_clients
from yian.compose import load_flow
from yian.memory import MemoryMutation, MemoryOperationError
from yian.nodes.base import BackwardResult
from yian.run import resume_flow, run_flow
from yian.runtime.log import (
    BackwardFinishedRecord, BackwardReadyRecord, BackwardStartedRecord,
    JsonlRunLog, MemoryCommitFailedRecord, MemoryCommitRequestedRecord,
    MemoryCommitSucceededRecord,
)
from tests.engine._memory_fakes import _Memory
from tests.engine.test_agent_backward import _BackwardChat, _reply
from tests.engine.test_run_log import _FileProxy
from tests.engine.test_run_memory import _records


# Fixed writer output; keep these bytes independent of the current codec.
_RECORDED_BACKWARD = (
    (
        b'{"kind":"run_started","payload":{"effective_llm_by_static_path":{},"flow_fingerprint":"9b3'
        b'2440fa3064ab7e910cb4c49539809ac9ae0a3390da9b8dbf83c0b6530c050","initial_pool":{"start_id":'
        b'"__start__","store":{"__start__":{"kind":"object","value":{}}},"system":{"now":{"kind":"da'
        b'tetime","value":"2026-10-02T04:21:25.257481+00:00"},"run_id":{"kind":"str","value":"histor'
        b'ical-backward"},"today":{"kind":"date","value":"2026-10-02"}}},"learnable":true,"memories"'
        b':[{"config":{},"conversation_eligible":false,"key":"test","owners":[{"node":null,"owner":['
        b'"root"],"scope":[]}],"type":"test"}],"memory_snapshots":[{"memory":"test","snapshot":{"nam'
        b'espace":"test"}}],"memory_stacks":[{"locator":["__end__"],"memories":["test"]},{"locator":'
        b'["__start__"],"memories":["test"]},{"locator":["work"],"memories":["test"]}],"num_workers"'
        b':0,"start_wiring_override":{}},"run_id":"historical-backward","schema":9,"seq":0}'
    ),
    (
        b'{"kind":"node_started","payload":{"node_id":"__start__"},"run_id":"historical-backward","s'
        b'chema":9,"seq":1}'
    ),
    (
        b'{"kind":"node_succeeded","payload":{"commit_target":"__start__","evidence":null,"forward_c'
        b'apture":{"bound_inputs":{},"evaluated_keys":[],"input_names":[],"items":[],"node_data":{"i'
        b'nput_output_paths":{}},"route":{"children":[],"content_index":0,"count":0,"expression":"{}'
        b'","kind":"record","names":[],"path":[],"result_type":null,"sizes":[],"snapshot":null,"sour'
        b'ce":null,"spans":[],"value_json":null},"scope":{"boundary":"start","flow_path":[],"occurre'
        b'nce_path":[],"static_path":["__start__"]},"variable_names":[]},"node_id":"__start__","valu'
        b'e":{"kind":"object","value":{}}},"run_id":"historical-backward","schema":9,"seq":2}'
    ),
    (
        b'{"kind":"node_started","payload":{"node_id":"work"},"run_id":"historical-backward","schema'
        b'":9,"seq":3}'
    ),
    (
        b'{"kind":"node_succeeded","payload":{"commit_target":"work","evidence":null,"forward_captur'
        b'e":{"bound_inputs":{},"evaluated_keys":[],"input_names":[],"items":[],"node_data":{},"rout'
        b'e":{"children":[],"content_index":0,"count":0,"expression":"{}","kind":"record","names":[]'
        b',"path":[],"result_type":null,"sizes":[],"snapshot":null,"source":null,"spans":[],"value_j'
        b'son":null},"scope":{"boundary":null,"flow_path":[],"occurrence_path":[],"static_path":["wo'
        b'rk"]},"variable_names":[]},"node_id":"work","value":{"kind":"int","value":3}},"run_id":"hi'
        b'storical-backward","schema":9,"seq":4}'
    ),
    (
        b'{"kind":"node_started","payload":{"node_id":"__end__"},"run_id":"historical-backward","sch'
        b'ema":9,"seq":5}'
    ),
    (
        b'{"kind":"node_succeeded","payload":{"commit_target":"__end__","evidence":null,"forward_cap'
        b'ture":{"bound_inputs":{"result":3},"evaluated_keys":["result"],"input_names":["result"],"i'
        b'tems":[],"node_data":{"input_output_paths":{"result":[]}},"route":{"children":[{"children"'
        b':[],"content_index":0,"count":0,"expression":"work.output","kind":"source","names":[],"pat'
        b'h":[],"result_type":null,"sizes":[],"snapshot":null,"source":{"execution_seq":3,"path":[],'
        b'"run_id":"historical-backward"},"spans":[],"value_json":null}],"content_index":0,"count":0'
        b',"expression":"{result: work.output}","kind":"record","names":["result"],"path":[],"result'
        b'_type":null,"sizes":[],"snapshot":null,"source":null,"spans":[],"value_json":null},"scope"'
        b':{"boundary":"end","flow_path":[],"occurrence_path":[],"static_path":["__end__"]},"variabl'
        b'e_names":[]},"node_id":"__end__","value":{"kind":"int","value":3}},"run_id":"historical-ba'
        b'ckward","schema":9,"seq":6}'
    ),
    (
        b'{"kind":"backward_started","payload":{"attempt_id":1,"feedback":null},"run_id":"historical'
        b'-backward","schema":9,"seq":7}'
    ),
    (
        b'{"kind":"backward_ready","payload":{"attempt_id":1,"attributed_results":[{"attribution":[]'
        b',"feedback":[],"invoked":false,"result":{"input_feedback":{},"self_feedback":[],"variable_'
        b'feedback":{}},"source":{"execution_seq":1,"path":[],"run_id":"historical-backward"}},{"att'
        b'ribution":[],"feedback":[],"invoked":false,"result":{"input_feedback":{},"self_feedback":['
        b'],"variable_feedback":{}},"source":{"execution_seq":3,"path":[],"run_id":"historical-backw'
        b'ard"}},{"attribution":[],"feedback":[],"invoked":false,"result":{"input_feedback":{},"self'
        b'_feedback":[],"variable_feedback":{}},"source":{"execution_seq":5,"path":[],"run_id":"hist'
        b'orical-backward"}}],"proposals":[{"memory":"test","mutations":[]}],"root_result":{"input_f'
        b'eedback":{},"self_feedback":[],"variable_feedback":{}}},"run_id":"historical-backward","sc'
        b'hema":9,"seq":8}'
    ),
    (
        b'{"kind":"memory_commit_requested","payload":{"attempt_id":1,"commit_id":"ac-memory-v3:97da'
        b'c2e25243bab896b6015ad9728ddabbe2a35f306e07b97b3ab7cf9be19263","memory":"test","mutations":'
        b'[]},"run_id":"historical-backward","schema":9,"seq":9}'
    ),
    (
        b'{"kind":"memory_commit_succeeded","payload":{"attempt_id":1,"commit_id":"ac-memory-v3:97da'
        b'c2e25243bab896b6015ad9728ddabbe2a35f306e07b97b3ab7cf9be19263","memory":"test"},"run_id":"h'
        b'istorical-backward","schema":9,"seq":10}'
    ),
    (
        b'{"kind":"backward_finished","payload":{"attempt_id":1,"attributed_results":[],"outcome":{"'
        b'failure":null,"feedback":{"input_feedback":{},"self_feedback":[],"variable_feedback":{}},"'
        b'learning":{"failures":[],"policies":[{"commit_id":"ac-memory-v3:97dac2e25243bab896b6015ad9'
        b'728ddabbe2a35f306e07b97b3ab7cf9be19263","failure":null,"memory":"test","status":"succeeded'
        b'"}],"status":"succeeded"},"status":"succeeded"}},"run_id":"historical-backward","schema":9'
        b',"seq":11}'
    ),
)


# Fixed writer output; keep these bytes independent of the current codec.
_RECORDED_BACKWARD_FAILED = (
    (
        b'{"kind":"run_started","payload":{"effective_llm_by_static_path":{},"flow_fingerprint":"9b3'
        b'2440fa3064ab7e910cb4c49539809ac9ae0a3390da9b8dbf83c0b6530c050","initial_pool":{"start_id":'
        b'"__start__","store":{"__start__":{"kind":"object","value":{}}},"system":{"now":{"kind":"da'
        b'tetime","value":"2026-10-02T04:21:25.298027+00:00"},"run_id":{"kind":"str","value":"histor'
        b'ical-backward_failed"},"today":{"kind":"date","value":"2026-10-02"}}},"learnable":true,"me'
        b'mories":[{"config":{},"conversation_eligible":false,"key":"test","owners":[{"node":null,"o'
        b'wner":["root"],"scope":[]}],"type":"test"}],"memory_snapshots":[{"memory":"test","snapshot'
        b'":{"namespace":"test"}}],"memory_stacks":[{"locator":["__end__"],"memories":["test"]},{"lo'
        b'cator":["__start__"],"memories":["test"]},{"locator":["work"],"memories":["test"]}],"num_w'
        b'orkers":0,"start_wiring_override":{}},"run_id":"historical-backward_failed","schema":9,"se'
        b'q":0}'
    ),
    (
        b'{"kind":"node_started","payload":{"node_id":"__start__"},"run_id":"historical-backward_fai'
        b'led","schema":9,"seq":1}'
    ),
    (
        b'{"kind":"node_succeeded","payload":{"commit_target":"__start__","evidence":null,"forward_c'
        b'apture":{"bound_inputs":{},"evaluated_keys":[],"input_names":[],"items":[],"node_data":{"i'
        b'nput_output_paths":{}},"route":{"children":[],"content_index":0,"count":0,"expression":"{}'
        b'","kind":"record","names":[],"path":[],"result_type":null,"sizes":[],"snapshot":null,"sour'
        b'ce":null,"spans":[],"value_json":null},"scope":{"boundary":"start","flow_path":[],"occurre'
        b'nce_path":[],"static_path":["__start__"]},"variable_names":[]},"node_id":"__start__","valu'
        b'e":{"kind":"object","value":{}}},"run_id":"historical-backward_failed","schema":9,"seq":2}'
    ),
    (
        b'{"kind":"node_started","payload":{"node_id":"work"},"run_id":"historical-backward_failed",'
        b'"schema":9,"seq":3}'
    ),
    (
        b'{"kind":"node_succeeded","payload":{"commit_target":"work","evidence":null,"forward_captur'
        b'e":{"bound_inputs":{},"evaluated_keys":[],"input_names":[],"items":[],"node_data":{},"rout'
        b'e":{"children":[],"content_index":0,"count":0,"expression":"{}","kind":"record","names":[]'
        b',"path":[],"result_type":null,"sizes":[],"snapshot":null,"source":null,"spans":[],"value_j'
        b'son":null},"scope":{"boundary":null,"flow_path":[],"occurrence_path":[],"static_path":["wo'
        b'rk"]},"variable_names":[]},"node_id":"work","value":{"kind":"int","value":3}},"run_id":"hi'
        b'storical-backward_failed","schema":9,"seq":4}'
    ),
    (
        b'{"kind":"node_started","payload":{"node_id":"__end__"},"run_id":"historical-backward_faile'
        b'd","schema":9,"seq":5}'
    ),
    (
        b'{"kind":"node_succeeded","payload":{"commit_target":"__end__","evidence":null,"forward_cap'
        b'ture":{"bound_inputs":{"result":3},"evaluated_keys":["result"],"input_names":["result"],"i'
        b'tems":[],"node_data":{"input_output_paths":{"result":[]}},"route":{"children":[{"children"'
        b':[],"content_index":0,"count":0,"expression":"work.output","kind":"source","names":[],"pat'
        b'h":[],"result_type":null,"sizes":[],"snapshot":null,"source":{"execution_seq":3,"path":[],'
        b'"run_id":"historical-backward_failed"},"spans":[],"value_json":null}],"content_index":0,"c'
        b'ount":0,"expression":"{result: work.output}","kind":"record","names":["result"],"path":[],'
        b'"result_type":null,"sizes":[],"snapshot":null,"source":null,"spans":[],"value_json":null},'
        b'"scope":{"boundary":"end","flow_path":[],"occurrence_path":[],"static_path":["__end__"]},"'
        b'variable_names":[]},"node_id":"__end__","value":{"kind":"int","value":3}},"run_id":"histor'
        b'ical-backward_failed","schema":9,"seq":6}'
    ),
    (
        b'{"kind":"backward_started","payload":{"attempt_id":1,"feedback":null},"run_id":"historical'
        b'-backward_failed","schema":9,"seq":7}'
    ),
    (
        b'{"kind":"backward_ready","payload":{"attempt_id":1,"attributed_results":[{"attribution":[]'
        b',"feedback":[],"invoked":false,"result":{"input_feedback":{},"self_feedback":[],"variable_'
        b'feedback":{}},"source":{"execution_seq":1,"path":[],"run_id":"historical-backward_failed"}'
        b'},{"attribution":[],"feedback":[],"invoked":false,"result":{"input_feedback":{},"self_feed'
        b'back":[],"variable_feedback":{}},"source":{"execution_seq":3,"path":[],"run_id":"historica'
        b'l-backward_failed"}},{"attribution":[],"feedback":[],"invoked":false,"result":{"input_feed'
        b'back":{},"self_feedback":[],"variable_feedback":{}},"source":{"execution_seq":5,"path":[],'
        b'"run_id":"historical-backward_failed"}}],"proposals":[{"memory":"test","mutations":[]}],"r'
        b'oot_result":{"input_feedback":{},"self_feedback":[],"variable_feedback":{}}},"run_id":"his'
        b'torical-backward_failed","schema":9,"seq":8}'
    ),
    (
        b'{"kind":"memory_commit_requested","payload":{"attempt_id":1,"commit_id":"ac-memory-v3:a47d'
        b'f09e6ee891e7fb93a8a3f4babfded8555cf7bd41b4d976bbd54d49c06dd9","memory":"test","mutations":'
        b'[]},"run_id":"historical-backward_failed","schema":9,"seq":9}'
    ),
    (
        b'{"kind":"memory_commit_failed","payload":{"attempt_id":1,"commit_id":"ac-memory-v3:a47df09'
        b'e6ee891e7fb93a8a3f4babfded8555cf7bd41b4d976bbd54d49c06dd9","failure":{"code":"declined","m'
        b'essage":"Declined.","retryable":false,"stage":"commit"},"memory":"test"},"run_id":"histori'
        b'cal-backward_failed","schema":9,"seq":10}'
    ),
    (
        b'{"kind":"backward_finished","payload":{"attempt_id":1,"attributed_results":[],"outcome":{"'
        b'failure":{"cause":{"code":"declined","message":"Declined.","retryable":false,"stage":"comm'
        b'it"},"cause_type":null,"code":"declined","memory":"test","message":"Declined.","node_id":n'
        b'ull,"node_started_seq":null,"run_id":null,"scope":null,"stage":"commit","traceback":null},'
        b'"feedback":{"input_feedback":{},"self_feedback":[],"variable_feedback":{}},"learning":{"fa'
        b'ilures":[{"failure":{"code":"declined","message":"Declined.","retryable":false,"stage":"co'
        b'mmit"},"memory":"test","operation":"commit"}],"policies":[{"commit_id":"ac-memory-v3:a47df'
        b'09e6ee891e7fb93a8a3f4babfded8555cf7bd41b4d976bbd54d49c06dd9","failure":{"code":"declined",'
        b'"message":"Declined.","retryable":false,"stage":"commit"},"memory":"test","status":"failed'
        b'"}],"status":"failed"},"status":"failed"}},"run_id":"historical-backward_failed","schema":'
        b'9,"seq":11}'
    ),
)


# Fixed writer output; keep these bytes independent of the current codec.
_RECORDED_FAILURE = (
    (
        b'{"kind":"run_started","payload":{"effective_llm_by_static_path":{},"flow_fingerprint":"ddc'
        b'273ce43ce37ff2d9719b9e217a3dd20fcdf29ebe21763fac72aeae2b6b1fe","initial_pool":{"start_id":'
        b'"__start__","store":{"__start__":{"kind":"object","value":{}}},"system":{"now":{"kind":"da'
        b'tetime","value":"2026-10-02T04:21:25.336197+00:00"},"run_id":{"kind":"str","value":"histor'
        b'ical-failure"},"today":{"kind":"date","value":"2026-10-02"}}},"learnable":true,"memories":'
        b'[{"config":{},"conversation_eligible":false,"key":"test","owners":[{"node":null,"owner":["'
        b'root"],"scope":[]}],"type":"test"}],"memory_snapshots":[{"memory":"test","snapshot":{"name'
        b'space":"test"}}],"memory_stacks":[{"locator":["__end__"],"memories":["test"]},{"locator":['
        b'"__start__"],"memories":["test"]},{"locator":["work"],"memories":["test"]}],"num_workers":'
        b'0,"start_wiring_override":{}},"run_id":"historical-failure","schema":9,"seq":0}'
    ),
    (
        b'{"kind":"node_started","payload":{"node_id":"__start__"},"run_id":"historical-failure","sc'
        b'hema":9,"seq":1}'
    ),
    (
        b'{"kind":"node_succeeded","payload":{"commit_target":"__start__","evidence":null,"forward_c'
        b'apture":{"bound_inputs":{},"evaluated_keys":[],"input_names":[],"items":[],"node_data":{"i'
        b'nput_output_paths":{}},"route":{"children":[],"content_index":0,"count":0,"expression":"{}'
        b'","kind":"record","names":[],"path":[],"result_type":null,"sizes":[],"snapshot":null,"sour'
        b'ce":null,"spans":[],"value_json":null},"scope":{"boundary":"start","flow_path":[],"occurre'
        b'nce_path":[],"static_path":["__start__"]},"variable_names":[]},"node_id":"__start__","valu'
        b'e":{"kind":"object","value":{}}},"run_id":"historical-failure","schema":9,"seq":2}'
    ),
    (
        b'{"kind":"node_started","payload":{"node_id":"work"},"run_id":"historical-failure","schema"'
        b':9,"seq":3}'
    ),
    (
        b'{"kind":"node_failed","payload":{"evidence":null,"failure":{"cause_type":"TypeCheckError",'
        b'"code":"contract.output","message":"\'bad\' does not match declared type int","node_id":"wor'
        b'k","trace":[{"key":"/s:root/s:node/s:work/s:output/s:contract/s:root"}],"traceback":"Trace'
        b'back (most recent call last):\\n  File \\"/Users/ngocbh/workspace/yian/src/agent_c'
        b'omposer/runtime/state.py\\", line 912, in prepare_node_succeeded\\n    value_wire, typed = _'
        b'canonical_output_value(\\n                        ^^^^^^^^^^^^^^^^^^^^^^^^\\n  File \\"/Users'
        b'/ngocbh/workspace/yian/src/yian/runtime/state.py\\", line 127, in _cano'
        b'nical_output_value\\n    return canonical_typed(output, declared_type)\\n           ^^^^^^^^'
        b'^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\\n  File \\"/Users/ngocbh/workspace/yian/src/agent_'
        b'composer/runtime/log.py\\", line 1433, in canonical_typed\\n    typed = build_value(value) i'
        b'f declared is None else build_value_as(declared, value)\\n                                 '
        b'                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\\n  File \\"/Users/ngocbh/workspace/a'
        b'gent-composer/src/yian/typesys/values.py\\", line 649, in build_value_as\\n    ret'
        b'urn _build_for_type(typ, value)\\n           ^^^^^^^^^^^^^^^^^^^^^^^^^^^\\n  File \\"/Users/n'
        b'gocbh/workspace/yian/src/yian/typesys/values.py\\", line 756, in _build'
        b'_for_type\\n    return _build_scalar_or_list(typ.kind, value, path=path)\\n           ^^^^^^'
        b'^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\\n  File \\"/Users/ngocbh/workspace/agent-compos'
        b'er/src/yian/typesys/values.py\\", line 798, in _build_scalar_or_list\\n    raise T'
        b"ypeCheckError(\\nyian.typesys.values.TypeCheckError: 'bad' does not match declare"
        b'd type int\\n"},"forward_capture":{"bound_inputs":{},"evaluated_keys":[],"input_names":[],"'
        b'items":[],"node_data":{},"route":{"children":[],"content_index":0,"count":0,"expression":"'
        b'{}","kind":"record","names":[],"path":[],"result_type":null,"sizes":[],"snapshot":null,"so'
        b'urce":null,"spans":[],"value_json":null},"scope":{"boundary":null,"flow_path":[],"occurren'
        b'ce_path":[],"static_path":["work"]},"variable_names":[]}},"run_id":"historical-failure","s'
        b'chema":9,"seq":4}'
    ),
)


_CHAIN = """id: backward-recovery
name: backward_recovery
long_term_memory: test
llm_config: {provider: ollama, model: local-test}
input:
  topic: str
nodes:
  first:
    kind: agent
    input:
      topic: ${input.topic}
    prompt: 'Draft ${topic}'
  second:
    kind: agent
    long_term_memory: test
    input:
      draft: ${first.output}
    prompt: 'Review ${draft}'
output: ${second.output}
"""


_RECORDED_BACKWARD_FLOW = """id: historical-backward
name: historical_backward
long_term_memory: test
nodes:
  work:
    kind: code
    code: return 3
    output: int
output: ${work.output}
"""


def _chain(tmp_path, monkeypatch, *, workers=0, backend=None):
    """Run the same real two-AGENT public flow before installing backward replies."""
    flow = load_flow(_CHAIN)
    backend = _Memory() if backend is None else backend
    chat = _BackwardChat([AIMessage(content="draft"), AIMessage(content="answer")])
    monkeypatch.setattr(llm_clients, "model_from_config", lambda config: chat)
    path = tmp_path / "chain.jsonl"
    result = run_flow(flow, {"topic": "reefs"}, log=path, learnable=True,
                      num_workers=workers, memory={"test": backend})
    assert result.status == "succeeded" and result.output == "answer"
    assert chat.calls == 2 and backend.learn_contexts == backend.commit_calls == []
    return flow, path, backend, result, chat


@pytest.mark.parametrize("failed", [False, True], ids=["success", "failure"])
def test_recorded_terminal_inspection_is_inert(tmp_path, monkeypatch, failed):
    from yian.nodes.code import CodeNode
    from yian.runtime.log import decode_record

    flow = load_flow(_RECORDED_BACKWARD_FLOW.replace("return 3", 'return "bad"') if failed else _RECORDED_BACKWARD_FLOW)
    wires = _RECORDED_FAILURE if failed else _RECORDED_BACKWARD[:7]
    path = tmp_path / "terminal.jsonl"
    original = b"\n".join(wires) + b"\n"
    path.write_bytes(original)
    backend = _Memory()
    monkeypatch.setattr(CodeNode, "forward", lambda *args, **kwargs: pytest.fail("terminal forward"))
    monkeypatch.setattr(CodeNode, "backward", lambda *args, **kwargs: pytest.fail("terminal backward"))
    monkeypatch.setattr(llm_clients, "model_from_config", lambda *args, **kwargs: pytest.fail("terminal model"))
    monkeypatch.setattr(JsonlRunLog, "append", lambda *args, **kwargs: pytest.fail("terminal append"))

    result = resume_flow(flow, log=path, memory={"test": backend})
    repeated = resume_flow(flow, log=path, memory={"test": backend})

    assert result.status == ("failed" if failed else "succeeded")
    assert result.output == (None if failed else 3)
    assert result.events == repeated.events == []
    assert result.backward_outcome is None
    if failed:
        assert result.failure == decode_record(wires[-1]).record.failure == repeated.failure
    assert backend.open_snapshots == backend.learn_contexts == backend.commit_calls == []
    assert backend.close_count == 0
    assert path.read_bytes() == original


@pytest.mark.parametrize("failed_commit,boundary,disposition", [
    (False, 7, "unrequested"),
    (False, 8, "unrequested"),
    (False, 9, "unknown"),
    (False, 10, "succeeded"),
    (False, 11, "succeeded"),
    (True, 10, "failed"),
    (True, 11, "failed"),
])
def test_recorded_backward_history_admits_a_later_attempt_at_every_boundary(
    tmp_path, monkeypatch, failed_commit, boundary, disposition,
):
    from yian.nodes.code import CodeNode
    from yian.runtime.log import SCHEMA_VERSION, decode_record

    flow = load_flow(_RECORDED_BACKWARD_FLOW)
    wires = (_RECORDED_BACKWARD_FAILED if failed_commit else _RECORDED_BACKWARD)[:boundary + 1]
    path = tmp_path / "attempt.jsonl"
    original = b"\n".join(wires) + b"\n"
    path.write_bytes(original)
    backend = _Memory()
    monkeypatch.setattr(CodeNode, "forward", lambda *args, **kwargs: pytest.fail("historical forward reran"))
    monkeypatch.setattr(CodeNode, "backward", lambda *args, **kwargs: pytest.fail("empty backward called node"))
    appended = []
    append = JsonlRunLog.append

    def observe_append(log, record):
        appended.append(record)
        return append(log, record)

    monkeypatch.setattr(JsonlRunLog, "append", observe_append)
    result = resume_flow(flow, log=path, memory={"test": backend})
    assert result.status == "succeeded" and result.output == 3 and result.events == []
    assert result.backward_outcome.learning.policies[0].status == disposition
    assert appended == [] and path.read_bytes() == original
    assert backend.open_snapshots == backend.learn_contexts == backend.commit_calls == []

    outcome = result.backward()
    assert outcome.status == "succeeded"
    assert backend.open_snapshots == [{"namespace": "test"}]
    assert len(backend.learn_contexts) == len(backend.commit_calls) == 1
    assert [record.attempt_id for record in appended if isinstance(record, BackwardStartedRecord)] == [2]
    current = path.read_bytes()
    assert current.startswith(original)
    added = current.splitlines()[len(wires):]
    assert added and all(decode_record(line).schema == SCHEMA_VERSION for line in added)
    before_reopen = len(appended)
    inspected = resume_flow(flow, log=path, memory={"test": object()})
    assert inspected.backward_outcome == outcome and inspected.events == []
    assert len(appended) == before_reopen and path.read_bytes() == current


@pytest.mark.parametrize("failed,value,output_type,feedback", [
    (False, {"items": [1, 2]}, "object", {"items": [None]}),
    (False, [[1], [2, 3]], "list[list[int]]", [None, [None]]),
    (False, [[1], [2, 3]], "list[list[int]]", [None, [None, None, None]]),
    (False, {"known": 1}, "object", {"missing": None}),
    (False, [1, 2], "list[int]", {"0": "Review."}),
    (True, 0, "int", {}),
    (True, 0, "int", []),
])
def test_invalid_root_shapes_preserve_the_public_forward_result_and_unopened_attempt(
    tmp_path, monkeypatch, failed, value, output_type, feedback,
):
    code = "if True:\n    raise RuntimeError('forward stopped')\nreturn 0" if failed else f"return {value!r}"
    flow = load_flow(f"""id: invalid-root-feedback
name: invalid_root_feedback
long_term_memory: test
nodes:
  work:
    kind: code
    code: |
{indent(code, '      ')}
    output: {output_type}
output: ${{work.output}}
""")
    backend = _Memory()
    path = tmp_path / "invalid.jsonl"
    result = run_flow(flow, {}, log=path, learnable=True, memory={"test": backend})
    assert result.status == ("failed" if failed else "succeeded")
    forward = deepcopy((result.input, result.status, result.output, result.failure, result.events))
    previous = (result.backward_outcome, result.learning)
    before = path.read_bytes()
    counters = (len(backend.open_snapshots), backend.close_count, len(backend.learn_contexts), len(backend.commit_calls))
    monkeypatch.setattr(llm_clients, "model_from_config", lambda config: pytest.fail("invalid feedback invoked model"))

    with pytest.raises((TypeError, ValueError)):
        result.backward(feedback)

    assert (result.input, result.status, result.output, result.failure, result.events) == forward
    assert result.backward_outcome is previous[0] and result.learning is previous[1]
    assert path.read_bytes() == before
    assert (len(backend.open_snapshots), backend.close_count, len(backend.learn_contexts), len(backend.commit_calls)) == counters
    assert not any(type(record) is BackwardStartedRecord for record in _records(path))
    inspected = resume_flow(flow, log=path, memory={"test": backend})
    assert inspected.events == [] and inspected.backward_outcome is None
    assert (inspected.input, inspected.status, inspected.output, inspected.failure) == forward[:4]
    assert path.read_bytes() == before
    assert (len(backend.open_snapshots), backend.close_count, len(backend.learn_contexts), len(backend.commit_calls)) == counters


@pytest.mark.parametrize("workers", [0, 2])
@pytest.mark.parametrize("invalid_result", [False, True])
def test_real_agent_failure_discards_partial_reverse_results_and_retries_every_callback(
    tmp_path, monkeypatch, workers, invalid_result,
):
    flow, path, backend, result, forward_chat = _chain(tmp_path, monkeypatch, workers=workers)
    forward = deepcopy((result.input, result.status, result.output, result.failure, result.events))
    forward_bytes = path.read_bytes()
    count = len(_records(path))
    bad = _reply(input_feedback={"topic": ["valid sibling"], "foreign": ["reject"]}) if invalid_result else AIMessage(content="not JSON")
    reverse = _BackwardChat([_reply(input_feedback={"draft": ["temporary"]}), bad])
    monkeypatch.setattr(llm_clients, "model_from_config", lambda config: reverse)

    outcome = result.backward("First review.")

    assert outcome.status == "failed" and outcome.failure.stage == "backward"
    assert outcome.failure.code == ("backward.result" if invalid_result else "backward.execute")
    assert outcome.failure.node_id == "first"
    assert outcome.feedback is None and outcome.learning.status == "incomplete"
    assert [policy.status for policy in outcome.learning.policies] == ["unrequested"]
    accepted = _records(path)[count:]
    assert [type(record) for record in accepted] == [BackwardStartedRecord, BackwardFinishedRecord]
    assert accepted[-1].attributed_results == ()
    assert path.read_bytes().startswith(forward_bytes)
    assert reverse.calls == 2 and forward_chat.calls == 2
    assert [json.loads(messages[1].content)["forward"]["node_id"] for messages in reverse.messages] == ["second", "first"]
    assert backend.learn_contexts == backend.commit_calls == [] and backend.close_count == 2
    assert (result.input, result.status, result.output, result.failure, result.events) == forward
    before = path.read_bytes()
    inspected = resume_flow(flow, log=path, memory={"test": backend})
    assert inspected.events == [] and inspected.backward_outcome == outcome
    assert (inspected.input, inspected.status, inspected.output, inspected.failure) == forward[:4]
    assert path.read_bytes() == before and reverse.calls == 2 and len(backend.open_snapshots) == 2

    retry = _BackwardChat([_reply(input_feedback={"draft": ["fresh draft"]}),
                           _reply(input_feedback={"topic": ["fresh topic"]})])
    monkeypatch.setattr(llm_clients, "model_from_config", lambda config: retry)
    completed = inspected.backward("Second review.")
    assert completed.status == "succeeded" and completed.feedback == BackwardResult({"topic": ("fresh topic",)})
    assert retry.calls == 2 and reverse.calls == 2 and forward_chat.calls == 2
    assert json.loads(retry.messages[0][1].content)["feedback"] == ["Second review."]
    assert [record.attempt_id for record in _records(path) if type(record) is BackwardStartedRecord] == [1, 2]
    assert len(backend.learn_contexts) == 1 and len(backend.commit_calls) == 1
    request, = [record for record in _records(path) if type(record) is MemoryCommitRequestedRecord]
    assert request.mutations == ()
    assert backend.commit_calls == [(request.commit_id, request.mutations)]
    assert backend.open_snapshots == [None, {"namespace": "test"}, {"namespace": "test"}]
    assert backend.close_count == 3 and all(session.closed for session in backend.sessions)
    assert (result.input, result.status, result.output, result.failure, result.events) == forward
    assert (inspected.input, inspected.status, inspected.output, inspected.failure, inspected.events) == (*forward[:4], [])


@pytest.mark.parametrize("last_kind,disposition", [
    ("backward_ready", "unrequested"),
    ("memory_commit_requested", "unknown"),
    ("memory_commit_succeeded", "succeeded"),
])
def test_terminal_resume_inspects_actual_public_attempt_prefixes_without_reopening_memory(
    tmp_path, monkeypatch, last_kind, disposition,
):
    flow, path, backend, result, forward_chat = _chain(tmp_path, monkeypatch)
    forward = deepcopy((result.input, result.status, result.output, result.failure, result.events))

    def propose(context):
        return [MemoryMutation("own/" + context.memory, {"retain": True})]

    backend.learn_outcomes = [propose]
    reverse = _BackwardChat([_reply(input_feedback={"draft": ["draft review"]}),
                             _reply(input_feedback={"topic": ["topic review"]})])
    monkeypatch.setattr(llm_clients, "model_from_config", lambda config: reverse)
    completed = result.backward("Review.")
    assert completed.status == "succeeded"
    request, = [record for record in _records(path) if type(record) is MemoryCommitRequestedRecord]
    assert len(request.mutations) == 1
    assert backend.commit_calls == [(request.commit_id, request.mutations)]
    complete_bytes = path.read_bytes()
    lines = complete_bytes.splitlines(keepends=True)
    boundary = next(index for index, line in enumerate(lines) if json.loads(line)["kind"] == last_kind)
    prefix_path = tmp_path / f"{last_kind}.jsonl"
    prefix_bytes = b"".join(lines[:boundary + 1])
    prefix_path.write_bytes(prefix_bytes)
    counters = (len(backend.open_snapshots), backend.close_count, len(backend.learn_contexts), len(backend.commit_calls))
    monkeypatch.setattr(llm_clients, "model_from_config", lambda config: pytest.fail("terminal inspection invoked model"))

    inspected = resume_flow(flow, log=prefix_path, memory={"test": object()})

    outcome = inspected.backward_outcome
    assert outcome.status == "failed" and outcome.failure.code == "backward.interrupted"
    assert outcome.feedback == completed.feedback == BackwardResult({"topic": ("topic review",)})
    assert outcome.learning.status == "incomplete"
    policy, = outcome.learning.policies
    assert policy.memory == "test" and policy.status == disposition and policy.failure is None
    assert (policy.commit_id is None) is (disposition == "unrequested")
    assert inspected.events == []
    assert (inspected.input, inspected.status, inspected.output, inspected.failure) == forward[:4]
    assert (result.input, result.status, result.output, result.failure, result.events) == forward
    assert prefix_path.read_bytes() == prefix_bytes and path.read_bytes() == complete_bytes
    assert (len(backend.open_snapshots), backend.close_count, len(backend.learn_contexts), len(backend.commit_calls)) == counters
    assert reverse.calls == forward_chat.calls == 2
    ready, = [record for record in _records(prefix_path) if type(record) is BackwardReadyRecord]
    assert ready.root_result == completed.feedback and len(ready.attributed_results) == 4
    assert len(ready.proposals) == 1 and all(len(proposal.mutations) == 1 for proposal in ready.proposals)


@pytest.mark.parametrize("failure_kind", ["learner", "duplicate_proposal"])
def test_late_learning_failure_keeps_completed_feedback_but_retry_rebuilds_every_proposal(tmp_path, monkeypatch, failure_kind):
    flow, path, backend, result, forward_chat = _chain(tmp_path, monkeypatch)
    forward = deepcopy((result.input, result.status, result.output, result.failure, result.events))
    before = path.read_bytes()
    count = len(_records(path))

    def duplicate(context):
        return [MemoryMutation("duplicate", {"attempt": 1}), MemoryMutation("duplicate", {"attempt": 1})]

    backend.learn_outcomes = [duplicate if failure_kind == "duplicate_proposal" else
                              MemoryOperationError("learner", "declined", "Learner declined.", False)]
    reverse = _BackwardChat([_reply(input_feedback={"draft": ["old draft"]}),
                             _reply(input_feedback={"topic": ["old topic"]})])
    monkeypatch.setattr(llm_clients, "model_from_config", lambda config: reverse)
    outcome = result.backward("First review.")
    assert outcome.status == "failed" and outcome.failure.stage == "learn"
    assert outcome.failure.code == ("declined" if failure_kind == "learner" else "memory.invalid_result")
    assert outcome.feedback == BackwardResult({"topic": ("old topic",)})
    assert outcome.learning.status == "failed" and outcome.learning.policies[0].status == "unrequested"
    accepted = _records(path)[count:]
    assert [type(record) for record in accepted] == [BackwardStartedRecord, BackwardFinishedRecord]
    assert len(accepted[-1].attributed_results) == 4
    assert path.read_bytes().startswith(before)
    assert reverse.calls == 2 and len(backend.learn_contexts) == 1 and backend.commit_calls == []
    assert outcome.failure.memory == backend.learn_contexts[0].memory == "test"
    assert backend.close_count == 2
    assert (result.input, result.status, result.output, result.failure, result.events) == forward
    before = path.read_bytes()
    inspected = resume_flow(flow, log=path, memory={"test": backend})
    assert inspected.events == [] and inspected.backward_outcome == outcome
    assert (inspected.input, inspected.status, inspected.output, inspected.failure) == forward[:4]
    assert path.read_bytes() == before and reverse.calls == 2 and len(backend.open_snapshots) == 2

    def fresh(context):
        return [MemoryMutation("fresh/" + context.memory, {"attempt": 2})]

    backend.learn_outcomes = [fresh]
    retry = _BackwardChat([_reply(input_feedback={"draft": ["fresh draft"]}),
                           _reply(input_feedback={"topic": ["fresh topic"]})])
    monkeypatch.setattr(llm_clients, "model_from_config", lambda config: retry)
    completed = inspected.backward("Second review.")
    assert completed.status == "succeeded" and completed.feedback == BackwardResult({"topic": ("fresh topic",)})
    assert retry.calls == reverse.calls == forward_chat.calls == 2
    assert len(backend.learn_contexts) == 2 and len(backend.commit_calls) == 1
    original, repeated = backend.learn_contexts
    assert original.memory == repeated.memory and original.episodes == repeated.episodes
    assert original.feedback != repeated.feedback
    committed = backend.commit_calls[0][1]
    assert len(committed) == 1 and all(mutation.mutation_id.startswith("fresh/") for mutation in committed)
    assert all(mutation.payload == {"attempt": 2} for mutation in committed)
    request, = [record for record in _records(path) if type(record) is MemoryCommitRequestedRecord]
    assert len(request.mutations) == 1 and request.attempt_id == 2
    assert backend.commit_calls == [(request.commit_id, request.mutations)]
    assert backend.open_snapshots == [None, {"namespace": "test"}, {"namespace": "test"}]
    assert backend.close_count == 3 and all(session.closed for session in backend.sessions)
    assert (result.input, result.status, result.output, result.failure, result.events) == forward
    assert (inspected.input, inspected.status, inspected.output, inspected.failure, inspected.events) == (*forward[:4], [])


def test_unmaterializable_owner_context_is_a_complete_known_failure_on_each_explicit_retry(tmp_path, monkeypatch):
    flow = load_flow("""id: deep-owner-context
name: deep_owner_context
long_term_memory: test
input:
  payload: object
nodes: {}
output: ${input.payload}
""")
    # A low memory bound lets a value well inside the yian value limit exceed the
    # learn-context bound.
    monkeypatch.setattr("yian.memory.contracts._MAX_JSON_NESTING", 10)
    value = {"leaf": "retained"}
    for _ in range(20):
        value = {"next": value}
    backend = _Memory()
    path = tmp_path / "deep.jsonl"
    result = run_flow(flow, {"payload": value}, log=path, learnable=True, memory={"test": backend})
    assert result.status == "succeeded" and result.output == value
    forward = deepcopy((result.input, result.status, result.output, result.failure, result.events))
    monkeypatch.setattr(llm_clients, "model_from_config", lambda config: pytest.fail("structural context failure invoked model"))
    before = path.read_bytes()
    first = result.backward("First review.")
    assert first.failure.stage == "learn" and first.failure.code == "memory.invalid_context"
    assert first.feedback == BackwardResult({"payload": ("First review.",)})
    assert first.learning.status == "failed" and first.learning.policies[0].status == "unrequested"
    assert path.read_bytes().startswith(before)
    assert (result.input, result.status, result.output, result.failure, result.events) == forward
    before = path.read_bytes()
    inspected = resume_flow(flow, log=path, memory={"test": backend})
    assert inspected.events == [] and inspected.backward_outcome == first
    assert (inspected.input, inspected.status, inspected.output, inspected.failure) == forward[:4]
    assert path.read_bytes() == before and len(backend.open_snapshots) == 2
    second = inspected.backward("Second review.")
    assert second.failure.code == first.failure.code and second.failure.memory == first.failure.memory
    assert second.feedback == BackwardResult({"payload": ("Second review.",)})
    assert backend.learn_contexts == backend.commit_calls == []
    records = _records(path)
    assert [record.attempt_id for record in records if type(record) is BackwardStartedRecord] == [1, 2]
    finished = [record for record in records if type(record) is BackwardFinishedRecord]
    assert len(finished) == 2 and all(len(record.attributed_results) == 2 for record in finished)
    assert not any(type(record) in (BackwardReadyRecord, MemoryCommitRequestedRecord) for record in records)
    assert backend.open_snapshots == [None, {"namespace": "test"}, {"namespace": "test"}]
    assert backend.close_count == 3 and all(session.closed for session in backend.sessions)
    assert (result.input, result.status, result.output, result.failure, result.events) == forward
    assert (inspected.input, inspected.status, inspected.output, inspected.failure, inspected.events) == (*forward[:4], [])


@pytest.mark.parametrize("error_type", [CancelledError, FutureCancelledError])
def test_reverse_cancellation_retains_started_only_and_resume_recomputes_the_entire_pass(tmp_path, monkeypatch, error_type):
    flow, path, backend, result, forward_chat = _chain(tmp_path, monkeypatch)
    forward = deepcopy((result.input, result.status, result.output, result.failure, result.events))
    before = path.read_bytes()
    count = len(_records(path))
    cancellation = error_type("stop reverse work")
    reverse = _BackwardChat([_reply(input_feedback={"draft": ["temporary"]}), cancellation])
    monkeypatch.setattr(llm_clients, "model_from_config", lambda config: reverse)
    with pytest.raises(error_type) as caught:
        result.backward("First review.")
    assert caught.value is cancellation
    assert [type(record) for record in _records(path)[count:]] == [BackwardStartedRecord]
    assert path.read_bytes().startswith(before)
    outcome = result.backward_outcome
    assert outcome.failure.code == "backward.interrupted" and outcome.feedback is None
    assert outcome.learning.status == "incomplete" and outcome.learning.policies[0].status == "unrequested"
    assert reverse.calls == 2 and forward_chat.calls == 2
    assert backend.learn_contexts == backend.commit_calls == [] and backend.close_count == 2
    assert (result.input, result.status, result.output, result.failure, result.events) == forward

    before = path.read_bytes()
    inspected = resume_flow(flow, log=path, memory={"test": backend})
    assert inspected.events == [] and inspected.backward_outcome == outcome
    assert (inspected.input, inspected.status, inspected.output, inspected.failure) == forward[:4]
    assert path.read_bytes() == before and reverse.calls == 2 and len(backend.open_snapshots) == 2

    retry = _BackwardChat([_reply(input_feedback={"draft": ["fresh draft"]}),
                           _reply(input_feedback={"topic": ["fresh topic"]})])
    monkeypatch.setattr(llm_clients, "model_from_config", lambda config: retry)
    assert inspected.backward("Second review.").feedback == BackwardResult({"topic": ("fresh topic",)})
    accepted = _records(path)[count:]
    assert [type(record) for record in accepted[:3]] == [BackwardStartedRecord, BackwardFinishedRecord, BackwardStartedRecord]
    assert accepted[1].outcome.feedback is None and accepted[1].attributed_results == ()
    assert accepted[2].attempt_id == 2
    assert retry.calls == 2 and reverse.calls == 2 and forward_chat.calls == 2
    assert len(backend.learn_contexts) == 1 and len(backend.commit_calls) == 1
    request, = [record for record in _records(path) if type(record) is MemoryCommitRequestedRecord]
    assert request.mutations == ()
    assert backend.commit_calls == [(request.commit_id, request.mutations)]
    assert backend.open_snapshots == [None, {"namespace": "test"}, {"namespace": "test"}]
    assert backend.close_count == 3 and all(session.closed for session in backend.sessions)
    assert (result.input, result.status, result.output, result.failure, result.events) == forward
    assert (inspected.input, inspected.status, inspected.output, inspected.failure, inspected.events) == (*forward[:4], [])


@pytest.mark.parametrize("phase,rejected_type", [
    ("started", BackwardStartedRecord),
    ("ready", BackwardReadyRecord),
    ("request", MemoryCommitRequestedRecord),
    ("partial_request", MemoryCommitRequestedRecord),
    ("success_observation", MemoryCommitSucceededRecord),
    ("failure_observation", MemoryCommitFailedRecord),
    ("finished", BackwardFinishedRecord),
    ("failed_finished", BackwardFinishedRecord),
])
def test_public_append_rejection_keeps_exact_authority_and_retries_after_every_phase(tmp_path, monkeypatch, phase, rejected_type):
    backend = _Memory(commit_outcomes=[MemoryOperationError("store", "declined", "Store declined.", False)]
                      if phase in ("failure_observation", "failed_finished") else [])
    flow, path, backend, result, forward_chat = _chain(tmp_path, monkeypatch, backend=backend)
    forward = deepcopy((result.input, result.status, result.output, result.failure, result.events))
    forward_bytes = path.read_bytes()
    count = len(_records(path))
    generation = 1

    def propose(context):
        return [MemoryMutation(f"pass-{generation}/{context.memory}", {"generation": generation})]

    backend.learn_outcomes = [propose] * 2
    reverse = _BackwardChat([_reply(input_feedback={"draft": ["old draft"]}),
                             _reply(input_feedback={"topic": ["old topic"]})])
    monkeypatch.setattr(llm_clients, "model_from_config", lambda config: reverse)
    append = JsonlRunLog.append
    error = OSError("rejected append")
    rejected_prefixes = []
    partial_writes = []

    def reject(log, record):
        if type(record) is not rejected_type:
            return append(log, record)
        rejected_prefixes.append(path.read_bytes())
        if phase == "partial_request":
            wrapped = log._file

            def partial_write(data):
                partial = data[:len(data) // 2]
                wrapped.write(partial)
                wrapped.flush()
                assert path.stat().st_size > len(rejected_prefixes[-1])
                partial_writes.append(partial)
                raise error

            log._file = _FileProxy(wrapped, on_write=partial_write)
            try:
                return append(log, record)
            finally:
                log._file = wrapped
        raise error

    monkeypatch.setattr(JsonlRunLog, "append", reject)
    with pytest.raises(OSError) as caught:
        result.backward("First review.")
    assert caught.value is error
    assert rejected_prefixes == [path.read_bytes()]
    assert bool(partial_writes) is (phase == "partial_request")
    assert path.read_bytes().startswith(forward_bytes)
    expected = [] if phase == "started" else [BackwardStartedRecord]
    if phase not in ("started", "ready"):
        expected.append(BackwardReadyRecord)
    if phase in ("success_observation", "failure_observation", "finished", "failed_finished"):
        expected.append(MemoryCommitRequestedRecord)
    if phase in ("finished", "failed_finished"):
        expected.append(MemoryCommitSucceededRecord if phase == "finished" else MemoryCommitFailedRecord)
    accepted = _records(path)[count:]
    assert [type(record) for record in accepted] == expected
    expected_callbacks = 0 if phase == "started" else 2
    expected_learns = 0 if phase == "started" else 1
    expected_commits = int(phase in ("success_observation", "failure_observation", "finished", "failed_finished"))
    assert reverse.calls == expected_callbacks and forward_chat.calls == 2
    assert len(backend.learn_contexts) == expected_learns and len(backend.commit_calls) == expected_commits
    if expected_commits:
        request, = [record for record in accepted if type(record) is MemoryCommitRequestedRecord]
        assert len(request.mutations) == 1
        assert backend.commit_calls == [(request.commit_id, request.mutations)]
    assert backend.close_count == 2 and all(session.closed for session in backend.sessions)
    assert (result.input, result.status, result.output, result.failure, result.events) == forward
    outcome = result.backward_outcome
    if phase == "started":
        assert outcome is None and result.learning is None
    else:
        assert outcome.failure.code == "backward.interrupted"
        assert outcome.feedback == (None if phase == "ready" else BackwardResult({"topic": ("old topic",)}))
        assert outcome.learning.status == ("failed" if phase == "failed_finished" else "incomplete")
        disposition = {"ready": "unrequested", "request": "unrequested", "partial_request": "unrequested",
                       "success_observation": "unknown", "failure_observation": "unknown",
                       "finished": "succeeded", "failed_finished": "failed"}[phase]
        assert outcome.learning.policies[0].status == disposition

    monkeypatch.setattr(JsonlRunLog, "append", append)
    before = path.read_bytes()
    inspected = resume_flow(flow, log=path, memory={"test": backend})
    assert inspected.events == [] and inspected.backward_outcome == outcome
    assert (inspected.input, inspected.status, inspected.output, inspected.failure) == forward[:4]
    assert path.read_bytes() == before and reverse.calls == expected_callbacks and len(backend.open_snapshots) == 2

    generation = 2
    retry = _BackwardChat([_reply(input_feedback={"draft": ["fresh draft"]}),
                           _reply(input_feedback={"topic": ["fresh topic"]})])
    monkeypatch.setattr(llm_clients, "model_from_config", lambda config: retry)
    completed = inspected.backward("Second review.")
    assert completed.status == "succeeded" and completed.feedback == BackwardResult({"topic": ("fresh topic",)})
    assert retry.calls == 2 and reverse.calls == expected_callbacks and forward_chat.calls == 2
    assert len(backend.learn_contexts) == expected_learns + 1
    assert len(backend.commit_calls) == expected_commits + 1
    records = _records(path)
    requests = [record for record in records if type(record) is MemoryCommitRequestedRecord]
    assert len(requests) == expected_commits + 1
    request = requests[-1]
    assert len(request.mutations) == 1
    assert backend.commit_calls[-1] == (request.commit_id, request.mutations)
    assert all(mutation.mutation_id.startswith("pass-2/") and mutation.payload == {"generation": 2}
               for mutation in backend.commit_calls[-1][1])
    assert [record.attempt_id for record in records if type(record) is BackwardStartedRecord] == ([1] if phase == "started" else [1, 2])
    if phase != "started":
        prior, = [record for record in records if type(record) is BackwardFinishedRecord and record.attempt_id == 1]
        assert prior.outcome.failure.code == "backward.interrupted"
        assert prior.outcome.feedback == outcome.feedback
    assert backend.open_snapshots == [None, {"namespace": "test"}, {"namespace": "test"}]
    assert backend.close_count == 3 and all(session.closed for session in backend.sessions)
    assert (result.input, result.status, result.output, result.failure, result.events) == forward
    assert (inspected.input, inspected.status, inspected.output, inspected.failure, inspected.events) == (*forward[:4], [])


def test_three_memory_failure_preserves_ordered_dispositions_and_admits_a_public_retry(tmp_path, monkeypatch):
    source = _CHAIN.replace("long_term_memory: test", "long_term_memory: A", 1)
    source = source.replace("  first:\n    kind: agent", "  first:\n    kind: agent\n    long_term_memory: B")
    flow = load_flow(source.replace("    long_term_memory: test", "    long_term_memory: C"))
    backends = {name: _Memory() for name in ("A", "B", "C")}
    backends["B"].commit_outcomes = [MemoryOperationError("store", "backward.interrupted", "Store declined.", True)]
    forward_chat = _BackwardChat([AIMessage(content="draft"), AIMessage(content="answer")])
    monkeypatch.setattr(llm_clients, "model_from_config", lambda config: forward_chat)
    path = tmp_path / "policies.jsonl"
    result = run_flow(flow, {"topic": "reefs"}, log=path, learnable=True, memory=backends)
    assert result.status == "succeeded" and result.output == "answer"
    forward = deepcopy((result.input, result.status, result.output, result.failure, result.events))
    forward_bytes = path.read_bytes()
    count = len(_records(path))
    learned = []

    def propose(memory_name, context):
        assert all(backend.commit_calls == [] for backend in backends.values())
        learned.append(memory_name)
        return [MemoryMutation("same-id", {"policy": memory_name})]

    for name, backend in backends.items():
        backend.learn_outcomes = [lambda context, name=name: propose(name, context)]
    reverse = _BackwardChat([_reply(input_feedback={"draft": ["draft review"]}),
                             _reply(input_feedback={"topic": ["topic review"]})])
    monkeypatch.setattr(llm_clients, "model_from_config", lambda config: reverse)
    outcome = result.backward("Review.")

    assert learned == ["A", "B", "C"]
    assert outcome.status == "failed" and outcome.failure.stage == "commit"
    assert outcome.failure.code == "backward.interrupted" and outcome.failure.memory == "B"
    assert outcome.feedback == BackwardResult({"topic": ("topic review",)})
    assert outcome.learning.status == "failed"
    assert [(policy.memory, policy.status) for policy in outcome.learning.policies] == [
        ("A", "succeeded"), ("B", "failed"), ("C", "unrequested"),
    ]
    assert outcome.learning.policies[1].failure == outcome.failure.cause
    assert outcome.failure.cause.retryable is True
    accepted = _records(path)[count:]
    assert [type(record) for record in accepted] == [
        BackwardStartedRecord, BackwardReadyRecord, MemoryCommitRequestedRecord,
        MemoryCommitSucceededRecord, MemoryCommitRequestedRecord, MemoryCommitFailedRecord,
        BackwardFinishedRecord,
    ]
    requests = [record for record in accepted if type(record) is MemoryCommitRequestedRecord]
    assert [record.memory for record in requests] == ["A", "B"]
    for request in requests:
        assert len(request.mutations) == 1
        assert backends[request.memory].commit_calls == [(request.commit_id, request.mutations)]
    assert [proposal.memory for proposal in accepted[1].proposals] == [
        backend.learn_contexts[0].memory for backend in backends.values()
    ]
    assert [proposal.mutations[0].payload for proposal in accepted[1].proposals] == [
        {"policy": name} for name in backends
    ]
    assert [len(backend.commit_calls) for backend in backends.values()] == [1, 1, 0]
    assert path.read_bytes().startswith(forward_bytes)
    assert reverse.calls == forward_chat.calls == 2
    assert all(len(backend.learn_contexts) == 1 and backend.close_count == 2 for backend in backends.values())
    assert all(session.closed for backend in backends.values() for session in backend.sessions)
    assert (result.input, result.status, result.output, result.failure, result.events) == forward
    before = path.read_bytes()
    retry = _BackwardChat([_reply(input_feedback={"draft": ["draft again"]}),
                           _reply(input_feedback={"topic": ["topic again"]})])
    monkeypatch.setattr(llm_clients, "model_from_config", lambda config: retry)

    inspected = resume_flow(flow, log=path, memory=backends)
    assert inspected.events == [] and inspected.backward_outcome == outcome
    assert (inspected.input, inspected.status, inspected.output, inspected.failure) == forward[:4]
    completed = inspected.backward("Second review.")
    assert completed.status == "succeeded" and completed.learning.status == "succeeded"
    assert [policy.status for policy in completed.learning.policies] == ["succeeded"] * 3
    assert path.read_bytes().startswith(before) and retry.calls == 2
    assert [len(backend.commit_calls) for backend in backends.values()] == [2, 2, 1]
    assert [record.attempt_id for record in _records(path) if type(record) is BackwardStartedRecord] == [1, 2]
    assert (result.input, result.status, result.output, result.failure, result.events) == forward
    assert (inspected.input, inspected.status, inspected.output, inspected.failure, inspected.events) == (*forward[:4], [])


@pytest.mark.parametrize("corruption", ["root_projection", "requested_batch"])
def test_public_replay_rejects_complete_record_corruption_before_any_new_effects(tmp_path, monkeypatch, corruption):
    flow, path, backend, result, forward_chat = _chain(tmp_path, monkeypatch)
    forward = deepcopy((result.input, result.status, result.output, result.failure, result.events))

    def propose(context):
        return [MemoryMutation("own/" + context.memory, {"retain": True})]

    backend.learn_outcomes = [propose]
    reverse = _BackwardChat([_reply(input_feedback={"draft": ["draft review"]}),
                             _reply(input_feedback={"topic": ["topic review"]})])
    monkeypatch.setattr(llm_clients, "model_from_config", lambda config: reverse)
    assert result.backward("Review.").status == "succeeded"
    request, = [record for record in _records(path) if type(record) is MemoryCommitRequestedRecord]
    assert len(request.mutations) == 1
    assert backend.commit_calls == [(request.commit_id, request.mutations)]
    previous = (result.backward_outcome, result.learning)
    counters = (len(backend.open_snapshots), backend.close_count, len(backend.learn_contexts), len(backend.commit_calls))
    lines = path.read_bytes().splitlines(keepends=True)
    kind = "backward_ready" if corruption == "root_projection" else "memory_commit_requested"
    index = next(index for index, line in enumerate(lines) if json.loads(line)["kind"] == kind)
    record = json.loads(lines[index])
    if corruption == "root_projection":
        record["payload"]["root_result"]["input_feedback"]["topic"] = ["forged projection"]
    else:
        record["payload"]["mutations"][0]["payload"] = {"forged": True}
    lines[index] = (json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
    corrupted = b"".join(lines)
    path.write_bytes(corrupted)
    monkeypatch.setattr(llm_clients, "model_from_config", lambda config: pytest.fail("corrupt replay invoked model"))

    message = "root result must match" if corruption == "root_projection" else "batch and identity must match"
    with pytest.raises(ValueError, match=message):
        result.backward("Review after tampering.")
    assert result.backward_outcome is previous[0] and result.learning is previous[1]
    with pytest.raises(ValueError, match=message):
        resume_flow(flow, log=path, memory={"test": backend})

    assert path.read_bytes() == corrupted
    assert (len(backend.open_snapshots), backend.close_count, len(backend.learn_contexts), len(backend.commit_calls)) == counters
    assert reverse.calls == forward_chat.calls == 2
    assert (result.input, result.status, result.output, result.failure, result.events) == forward
