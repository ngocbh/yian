"""Public backward composition through loaded flows and the real node hooks."""

from copy import deepcopy
import json
import threading

from langchain_core.messages import AIMessage
import pytest

import yian.llm_clients as llm_clients
from yian.compose import load_flow
from yian.events import NodeSucceeded
from yian.expr.builtins import TEMPLATE_FNS
from yian.memory import MemoryMutation
from yian.nodes.base import BackwardResult
from yian.nodes.code import CodeNode
from yian.run import resume_flow, run_flow
from yian.runtime.log import (
    BackwardFinishedRecord,
    BackwardReadyRecord,
    BackwardStartedRecord,
    JsonlRunLog,
    MemoryCommitRequestedRecord,
    NodeStartedRecord,
    NodeSucceededRecord,
    _evidence_payload,
)
from tests.engine.test_agent_backward import _BackwardChat, _reply
from tests.engine._memory_fakes import _Memory


def test_nested_call_case_selects_partial_feedback_without_promoting_child_defaults(tmp_path, monkeypatch):
    flow = load_flow("""id: selected-feedback
name: selected_feedback
input:
  selected: bool
  payload: object
defs:
  leaf:
    input:
      payload: object
      local: str = child default
    nodes: {}
    output:
      payload: ${input.payload}
      local: ${input.local}
  wrapper:
    input:
      selected: bool
      payload: object
    nodes:
      gate:
        kind: case
        cases:
          - when: input.selected
            then: chosen
        else: unselected
      chosen:
        kind: call
        call: leaf
        input:
          payload: ${input.payload}
      unselected:
        kind: call
        call: leaf
        input:
          payload: ${input.payload}
    output: ${chosen.output | unselected.output}
nodes:
  outer:
    kind: call
    call: wrapper
    input:
      selected: ${input.selected}
      payload: ${input.payload}
output: ${outer.output}
""")
    monkeypatch.setattr(llm_clients, "model_from_config", lambda config: pytest.fail("structural flow called a model"))
    path = tmp_path / "selected.jsonl"
    payload = {"title": "unchanged", "rows": [["a"], ["b", "c"]]}
    result = run_flow(flow, {"selected": True, "payload": payload}, log=path, learnable=True)
    assert result.status == "succeeded"
    assert result.output == {"payload": payload, "local": "child default"}
    before = path.read_bytes()
    with pytest.raises(ValueError, match="exactly 2"):
        result.backward({"payload": {"rows": [None, [None]]}})
    assert path.read_bytes() == before and result.backward_outcome is None

    selected = {"rows": [None, ["Revise b.", None]]}
    outcome = result.backward({"payload": selected, "local": "Keep the default."})
    assert outcome.status == "succeeded"
    assert outcome.feedback == BackwardResult({"payload": (selected,)})
    with JsonlRunLog.resume(path) as log:
        envelopes = tuple(log)
    starts = {envelope.seq: envelope.record.node_id for envelope in envelopes
              if type(envelope.record) is NodeStartedRecord}
    ready, = [envelope.record for envelope in envelopes if type(envelope.record) is BackwardReadyRecord]
    by_node = {starts[item.source.execution_seq]: item for item in ready.attributed_results}
    assert not any("unselected" in name for name in by_node)
    assert by_node["outer/gate"].invoked is False
    assert by_node["outer/gate"].feedback == ()
    assert by_node["outer/chosen/__start__"].result.input_feedback == {
        "payload": (selected,), "local": ("Keep the default.",),
    }
    assert result.output == {"payload": payload, "local": "child default"}


@pytest.mark.parametrize("feedback", ["Review every item.", [None, ""]])
def test_public_map_preserves_equal_item_positions_after_reversed_completion(tmp_path, monkeypatch, feedback):
    flow = load_flow("""id: ordered-map-feedback
name: ordered_map_feedback
llm_config: {provider: ollama, model: local-test}
input:
  values: list[str]
defs:
  item_flow:
    input:
      value: str
    nodes:
      gate:
        kind: code
        code: return True
        output: bool
      answer:
        kind: agent
        depends_on: [gate]
        input:
          value: ${input.value}
        prompt: ${value}
    output: ${answer.output}
nodes:
  each:
    kind: map
    call: item_flow
    over: ${input.values}
    input:
      value: ${item}
output: ${each.output}
""")
    second_done = threading.Event()
    original_forward = CodeNode.forward

    def gate(node, inputs):
        position, = node.execution_scope.occurrence_path
        if position.index == 0:
            assert second_done.wait(10), "second MAP item did not finish"
        return original_forward(node, inputs)

    def observe(event):
        if type(event) is NodeSucceeded and event.node_id == "each#1/__end__":
            second_done.set()

    monkeypatch.setattr(CodeNode, "forward", gate)
    forward_chats = []

    def forward_model(config):
        chat = _BackwardChat([AIMessage(content="same")])
        forward_chats.append(chat)
        return chat

    monkeypatch.setattr(llm_clients, "model_from_config", forward_model)
    path = tmp_path / "map.jsonl"
    result = run_flow(flow, {"values": ["same", "same"]}, log=path, learnable=True,
                      num_workers=4, on_event=observe)
    assert result.status == "succeeded" and result.output == ["same", "same"]
    assert len(forward_chats) == 2 and all(chat.calls == 1 for chat in forward_chats)
    with JsonlRunLog.resume(path) as log:
        forward_records = tuple(log)
    item_ends = [record.record.forward_capture.scope.occurrence_path[0].index
                 for record in forward_records if type(record.record) is NodeSucceededRecord
                 and record.record.forward_capture.scope.static_path == ("each", "__end__")]
    assert item_ends == [1, 0]
    before = path.read_bytes()
    with pytest.raises(ValueError, match="exactly 2"):
        result.backward([None])
    assert path.read_bytes() == before and result.backward_outcome is None

    backward_chats = []
    message = feedback if isinstance(feedback, str) else ""

    def backward_model(config):
        chat = _BackwardChat([_reply(input_feedback={"value": [message]})])
        backward_chats.append(chat)
        return chat

    monkeypatch.setattr(llm_clients, "model_from_config", backward_model)
    outcome = result.backward(feedback)
    expected_indices = [0, 1] if isinstance(feedback, str) else [1]
    assert outcome.status == "succeeded"
    assert outcome.feedback == BackwardResult({"values": tuple(
        [message if position == index else None for position in range(2)]
        for index in expected_indices
    )})
    assert len(backward_chats) == len(expected_indices)
    agent_records = {envelope.record.node_id: envelope.record for envelope in forward_records
                     if type(envelope.record) is NodeSucceededRecord
                     and envelope.record.forward_capture.scope.static_path == ("each", "answer")}
    observed_positions = []
    for chat in backward_chats:
        assert chat.calls == 1 and chat.bound_tools is None
        prompt = json.loads(chat.messages[0][1].content)
        assert prompt["feedback"] == [message]
        assert prompt["forward"]["inputs"] == {"value": "same"}
        details = [detail for entry in prompt["downstream_context"] for detail in entry["contexts"]]
        index = agent_records[prompt["forward"]["node_id"]].forward_capture.scope.occurrence_path[0].index
        assert any(detail["role"] == "output" and detail["result"] == ["same", "same"]
                   and detail["positions"] == [[index]] for detail in details)
        observed_positions.append(index)
    assert sorted(observed_positions) == expected_indices


def test_call_and_map_loop_calls_retain_each_agent_after_pruning_and_terminal_resume(tmp_path, monkeypatch):
    from yian.run import _replay_run_log

    flow = load_flow("""id: nested-agent-history
name: nested_agent_history
input:
  n: int
  values: list[int]
defs:
  agent_body:
    input:
      n: int
    nodes:
      author:
        kind: agent
        variables:
          tone: str = concise
        input:
          n: ${input.n}
        prompt: '${tone}: ${n}'
        output:
          n: int
    output: ${author.output}
  loop_body:
    input:
      n: int
    nodes:
      invoke:
        kind: call
        call: agent_body
        input:
          n: ${input.n}
    output: ${invoke.output}
  direct:
    llm_config: {provider: openai, model: no-structured-sentinel, openai: {temperature: 0.1}}
    input:
      n: int
    nodes:
      loop:
        kind: loop
        call: loop_body
        input:
          n: ${input.n}
        times: 2
    output: ${loop.output}
  mapped:
    llm_config: {provider: deepseek, model: no-structured-sentinel, deepseek: {temperature: 0.8}}
    input:
      n: int
    nodes:
      loop:
        kind: loop
        call: loop_body
        input:
          n: ${input.n}
        times: 2
    output: ${loop.output}
nodes:
  one:
    kind: call
    call: direct
    input:
      n: ${input.n}
  many:
    kind: map
    call: mapped
    over: ${input.values}
    input:
      n: ${item}
output:
  single: ${one.output}
  mapped: ${many.output}
""")
    forward_calls = []

    def forward_model(config):
        chat = _BackwardChat([AIMessage(content='{"n": 1}')])
        forward_calls.append((deepcopy(config), chat))
        return chat

    monkeypatch.setattr(llm_clients, "model_from_config", forward_model)
    path = tmp_path / "nested.jsonl"
    result = run_flow(flow, {"n": 3, "values": [10, 20]}, log=path, learnable=True)
    assert result.status == "succeeded"
    assert result.output == {"single": {"n": 1}, "mapped": [{"n": 1}, {"n": 1}]}
    assert len(forward_calls) == 6 and all(chat.calls == 1 for _, chat in forward_calls)
    with JsonlRunLog.resume(path) as log:
        envelopes = tuple(log)
        replay = _replay_run_log(flow, log)
    assert not any(type(envelope.record) is BackwardStartedRecord for envelope in envelopes)
    accepted_agents = {item.node_id: item for item in replay.trace.occurrences
                       if item.record.forward_capture.node_data.get("prompt_template") == "${tone}: ${n}"}
    assert len(accepted_agents) == 6
    assert sorted(json.dumps(config, sort_keys=True) for config, _ in forward_calls) == sorted(
        json.dumps(item.record.forward_capture.node_data["effective_llm_config"], sort_keys=True)
        for item in accepted_agents.values()
    )
    expected_map = envelopes[0].record.effective_llm_by_static_path
    for occurrence in accepted_agents.values():
        capture = occurrence.record.forward_capture
        scope = capture.scope
        positions = scope.occurrence_path
        body, = [position for position in positions if position.kind == "loop_body"]
        assert body.index in (0, 1)
        assert [position.kind for position in positions] == [
            "call" if scope.static_path[0] == "one" else "map_item", "loop_body", "call",
        ]
        assert capture.bound_inputs["n"] == (1 if body.index == 1 else (
            3 if scope.static_path[0] == "one" else [10, 20][positions[0].index]
        ))
        assert capture.node_data["variables"] == {"tone": "concise"}
        assert capture.node_data["variable_domains"] == {"tone": "str"}
        assert capture.node_data["rendered_prompt"] == f"concise: {capture.bound_inputs['n']}"
        selected = expected_map["/".join(scope.static_path)]
        assert capture.node_data["effective_llm_config"] == selected
        assert selected["provider"] == ("openai" if scope.static_path[0] == "one" else "deepseek")
        assert selected[selected["provider"]] == {
            "temperature": 0.1 if scope.static_path[0] == "one" else 0.8
        }
        assert occurrence.record.evidence is not None
        assert occurrence.record.evidence.prompt == capture.node_data["rendered_prompt"]

    prior_body = [item for item in replay.trace.occurrences
                  if item.record.forward_capture.scope.static_path[:-1] in (("one", "loop"), ("many", "loop"))
                  and any(position.kind == "loop_body" and position.index == 0
                          for position in item.record.forward_capture.scope.occurrence_path)]
    assert len(prior_body) == 9
    assert all(item.node_id not in replay.flow.nodes for item in prior_body)

    monkeypatch.setattr(llm_clients, "model_from_config", lambda config: pytest.fail("terminal resume invoked model"))
    before = path.read_bytes()
    inspected = resume_flow(flow, log=path, on_event=lambda event: pytest.fail("terminal resume emitted event"))
    assert inspected.events == [] and inspected.output == result.output
    assert inspected.backward_outcome is None and path.read_bytes() == before
    backward_calls = []

    def backward_model(config):
        chat = _BackwardChat([_reply(input_feedback={"n": ["Refine n."]},
                                    self_feedback=["Keep the record."],
                                    variable_feedback={"tone": ["Keep concise."]})])
        backward_calls.append((deepcopy(config), chat))
        return chat

    monkeypatch.setattr(llm_clients, "model_from_config", backward_model)
    outcome = inspected.backward("Review every iteration.")
    assert outcome.status == "succeeded"
    assert outcome.feedback == BackwardResult({
        "n": ("Refine n.",), "values": (["Refine n.", None], [None, "Refine n."]),
    })
    assert len(backward_calls) == len(accepted_agents)
    seen = []
    for config, chat in backward_calls:
        assert chat.calls == 1 and chat.bound_tools is None
        material = json.loads(chat.messages[0][1].content)["forward"]
        occurrence = accepted_agents[material["node_id"]]
        capture = occurrence.record.forward_capture
        assert material["node_started_seq"] == occurrence.source.execution_seq
        assert material["inputs"] == capture.bound_inputs
        assert material["output"] == {"n": 1}
        for key in ("prompt_template", "rendered_prompt", "variables", "variable_domains"):
            assert material[key] == capture.node_data[key]
        assert material["evaluation_evidence"] == _evidence_payload(occurrence.record.evidence)
        assert config == capture.node_data["effective_llm_config"]
        seen.append(occurrence.source)
    assert len(set(seen)) == len(accepted_agents)
    with JsonlRunLog.resume(path) as log:
        ready, = [envelope.record for envelope in log if type(envelope.record) is BackwardReadyRecord]
    assert len(ready.attributed_results) == len(replay.trace.occurrences)
    assert all(item.invoked for item in ready.attributed_results if item.source in seen)
    before = path.read_bytes()
    monkeypatch.setattr(llm_clients, "model_from_config", lambda config: pytest.fail("completed replay invoked model"))
    recovered = resume_flow(flow, log=path)
    assert recovered.events == [] and recovered.backward_outcome == outcome
    assert recovered.output == result.output and path.read_bytes() == before


def test_concat_repeat_and_opaque_routes_cross_public_call_without_reevaluation(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setitem(TEMPLATE_FNS, "composition_opaque", lambda value: calls.append(deepcopy(value)) or value)
    flow = load_flow("""id: composed-expressions
name: composed_expressions
long_term_memory: test
input:
  values: list[str]
defs:
  identity:
    input:
      values: list[str]
    nodes: {}
    output: ${input.values}
  combine:
    input:
      values: list[str]
    nodes:
      invoke:
        kind: call
        call: identity
        input:
          values: ${input.values + input.values * 2 + ['literal'] + composition_opaque(input.values)}
    output: ${invoke.output}
nodes:
  outer:
    kind: call
    call: combine
    input:
      values: ${input.values}
output: ${outer.output}
""")
    backend = _Memory()
    path = tmp_path / "expressions.jsonl"
    result = run_flow(flow, {"values": ["same"]}, log=path, learnable=True, memory={"test": backend})
    assert result.status == "succeeded"
    assert result.output == ["same", "same", "same", "literal", "same"]
    assert calls == [["same"]]
    outcome = result.backward(["first", "second", "third", "literal feedback", "opaque feedback"])
    assert outcome.status == "succeeded"
    assert outcome.feedback == BackwardResult({"values": (["first"], ["second"], ["third"])})
    context, = backend.learn_contexts
    stopped = [entry for entry in context.feedback if entry["channel"] == "stopped"]
    assert [entry["value"] for entry in stopped] == [["opaque feedback"]]
    assert all(entry["target_name"] == "values" for entry in stopped)
    assert any("composition_opaque(" in detail["expression"] and detail["result"] == ["same"]
               for entry in stopped for detail in entry["contribution_context"]["contexts"])
    assert calls == [["same"]]
    before = path.read_bytes()
    inspected = resume_flow(flow, log=path, memory={"test": backend})
    assert inspected.backward_outcome == outcome and inspected.events == []
    assert path.read_bytes() == before and calls == [["same"]]


@pytest.mark.parametrize("invalid_proposal", [False, True])
def test_owned_memories_learn_complete_history_and_reject_invalid_proposals(tmp_path, monkeypatch, invalid_proposal):
    flow = load_flow("""id: owned-composition
name: owned_composition
long_term_memory: outer
llm_config: {provider: ollama, model: local-test}
input:
  topic: str
defs:
  child:
    long_term_memory: inner
    input:
      topic: str
    nodes:
      work:
        kind: code
        code: return 7
        output: int
      author:
        kind: agent
        long_term_memory: local
        variables:
          tone: str = concise
        input:
          topic: ${input.topic}
        prompt: '${topic}: ${tone}'
    output: ${author.output}
nodes:
  gate:
    kind: case
    cases:
      - when: 'true'
        then: invoke
    else: unused
  invoke:
    kind: call
    call: child
    input:
      topic: ${input.topic}
  unused:
    kind: call
    call: child
    input:
      topic: ${input.topic}
output: ${invoke.output | unused.output}
""")
    backends = {key: _Memory(recall_text="Preserve detail.") for key in ("inner", "local", "outer")}
    forward_chat = _BackwardChat([AIMessage(content="answer")])
    monkeypatch.setattr(llm_clients, "model_from_config", lambda config: forward_chat)
    path = tmp_path / "owned.jsonl"
    result = run_flow(flow, {"topic": "reefs"}, learnable=True, log=path, memory=backends)
    assert result.status == "succeeded" and result.output == "answer"
    assert forward_chat.calls == 1
    assert all(backend.learn_contexts == backend.commit_calls == [] for backend in backends.values())
    with JsonlRunLog.resume(path) as log:
        forward = tuple(log)
    assert [spec.key for spec in forward[0].record.memories] == ["inner", "local", "outer"]
    author, = [item for item in forward if type(item.record) is NodeSucceededRecord
               and item.record.node_id == "invoke/author"]
    agent_sequence, = [item.seq for item in forward if type(item.record) is NodeStartedRecord
                       and item.record.node_id == "invoke/author"]
    assert author.seq > agent_sequence
    assert len(author.record.evidence.memory_sections) == 3

    # Each learner receives only its own memory's context despite inherited recall/evidence.
    def propose(context):
        assert backward_chat.calls == 1
        assert all(backend.commit_calls == [] for backend in backends.values())
        return [MemoryMutation(context.memory, {"seen": len(context.episodes)})]

    for key, backend in backends.items():
        backend.learn_outcomes = [(lambda context: ()) if invalid_proposal and key == "local" else propose]
    backward_chat = _BackwardChat([_reply(input_feedback={"topic": ["Refine topic."]},
                                         self_feedback=["Keep context.", "Avoid guessing."],
                                         variable_feedback={"tone": ["Stay concise.", "Be precise."]})])
    monkeypatch.setattr(llm_clients, "model_from_config", lambda config: backward_chat)
    outcome = result.backward("Review answer.")
    assert outcome.feedback == BackwardResult({"topic": ("Refine topic.",)})
    assert backward_chat.calls == 1 and backward_chat.bound_tools is None
    with JsonlRunLog.resume(path) as log:
        records = tuple(item.record for item in log)
    if invalid_proposal:
        assert outcome.status == "failed" and outcome.failure.stage == "learn"
        assert outcome.failure.code == "memory.invalid_result" and outcome.failure.memory == "local"
        assert not any(type(record) in (BackwardReadyRecord, MemoryCommitRequestedRecord) for record in records)
        assert all(backend.commit_calls == [] for backend in backends.values())
        assert type(records[-1]) is BackwardFinishedRecord
        return

    assert outcome.status == "succeeded" and outcome.learning.status == "succeeded"
    for key, backend in backends.items():
        context, = backend.learn_contexts
        assert context.memory == key
        (_, mutations), = backend.commit_calls
        assert [mutation.mutation_id for mutation in mutations] == [key]
        episode, = [episode for episode in context.episodes if episode.execution_seq == agent_sequence]
        assert episode.owner_static_path == ("invoke", "author")
        assert episode.output == "answer"
        entries = [entry for entry in context.feedback if entry["execution_seq"] == agent_sequence]
        assert all(entry["owner_static_path"] == ["invoke", "author"] for entry in entries)
        assert [entry["value"] for entry in entries if entry["channel"] == "output"] == ["Review answer."]
        assert [entry["value"] for entry in entries if entry["channel"] == "self"] == ["Keep context.", "Avoid guessing."]
        assert [entry["value"] for entry in entries if entry["channel"] == "variable"] == ["Stay concise.", "Be precise."]
        for channel, target in (("self", None), ("variable", "tone")):
            messages = [entry for entry in entries if entry["channel"] == channel]
            assert [entry["message_index"] for entry in messages] == [0, 1]
            assert all(entry["target_name"] == target and entry["contribution_context"] == {} for entry in messages)
    assert len(backends["local"].learn_contexts[0].episodes) == 1
    for key in ("inner", "outer"):
        assert any(episode.owner_static_path == ("invoke", "work")
                   for episode in backends[key].learn_contexts[0].episodes)


@pytest.mark.parametrize("action", ["noop", "add"])
def test_two_public_runs_reject_the_stale_reflective_snapshot_after_any_commit(tmp_path, monkeypatch, action):
    import yian.memory.reflective as reflective
    from yian.memory.reflective import ReflectiveMemory

    flow = load_flow("""id: reflective-composition
name: reflective_composition
long_term_memory: reflective
input:
  text: str
nodes: {}
output: ${input.text}
""")
    backend = ReflectiveMemory({"provider": "ollama", "model": "local-reflector"})
    payload = {"action": action, "lesson_id": None if action == "noop" else "one",
               "text": None if action == "noop" else "Keep the original detail."}
    learner = _BackwardChat([{"actions": [payload]}])
    schemas = []
    configurations = []

    def bind_schema(schema):
        schemas.append(schema)
        return learner

    learner.with_structured_output = bind_schema
    monkeypatch.setattr(reflective, "model_from_config",
                        lambda config: configurations.append(deepcopy(config)) or learner)
    monkeypatch.setattr(llm_clients, "model_from_config", lambda config: pytest.fail("identity flow called an AGENT model"))
    first_path, second_path = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    first = run_flow(flow, {"text": "A"}, run_id="snapshot-a", log=first_path,
                     learnable=True, memory={"reflective": backend})
    second = run_flow(flow, {"text": "B"}, run_id="snapshot-b", log=second_path,
                      learnable=True, memory={"reflective": backend})
    with JsonlRunLog.resume(first_path) as log:
        first_start = next(iter(log)).record
    with JsonlRunLog.resume(second_path) as log:
        second_start = next(iter(log)).record
    assert first_start.memory_snapshots == second_start.memory_snapshots
    initial = first_start.memory_snapshots[0].snapshot
    assert learner.calls == 0 and configurations == []

    winner = second.backward("B feedback")
    assert winner.status == "succeeded" and winner.learning.status == "succeeded"
    assert winner.feedback == BackwardResult({"text": ("B feedback",)})
    with JsonlRunLog.resume(second_path) as log:
        requests = [item.record for item in log if type(item.record) is MemoryCommitRequestedRecord]
    request, = requests
    assert len(request.mutations) == (0 if action == "noop" else 1)
    with backend.open_run() as session:
        committed = session.snapshot
    assert committed["revision"] == initial["revision"] + 1
    assert len(committed["lessons"]) == (0 if action == "noop" else 1)
    assert learner.calls == 1 and len(configurations) == len(schemas) == 1

    forward_state = deepcopy((first.status, first.output, first.failure))
    assert forward_state == ("succeeded", "A", None)
    stale = first.backward("A feedback")
    assert (first.status, first.output, first.failure) == forward_state
    assert stale.status == "failed" and stale.failure.stage == "learn"
    assert stale.failure.code == "reflective.stale_generation"
    assert stale.feedback == BackwardResult({"text": ("A feedback",)})
    assert stale.learning.status == "failed"
    assert [policy.status for policy in stale.learning.policies] == ["unrequested"]
    with JsonlRunLog.resume(first_path) as log:
        records = tuple(item.record for item in log)
    assert type(records[-1]) is BackwardFinishedRecord
    assert not any(type(record) in (BackwardReadyRecord, MemoryCommitRequestedRecord) for record in records)
    assert learner.calls == 1
    with backend.open_run() as session:
        assert session.snapshot == committed
    before = first_path.read_bytes()
    inspected = resume_flow(flow, log=first_path, memory={"reflective": object()})
    assert (inspected.status, inspected.output, inspected.failure) == forward_state
    assert inspected.backward_outcome == stale and inspected.events == []
    assert first_path.read_bytes() == before and learner.calls == 1
