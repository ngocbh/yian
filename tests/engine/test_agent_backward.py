"""One tool-free AGENT backward call over retained invocation material."""

from copy import deepcopy
import json

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import pytest

import yian.llm_clients as llm_clients
from yian.nodes.agent import AgentNode
from yian.nodes.base import BackwardContext, BackwardResult, validate_backward_result
from tests.engine.test_agent import _FakeChat


class _BackwardChat(_FakeChat):
    """Keep the existing model fake while recording messages and scripted errors."""

    def __init__(self, replies):
        super().__init__(replies)
        self.messages = []

    def invoke(self, messages):
        self.messages.append(deepcopy(messages))
        result = super().invoke(messages)
        if isinstance(result, BaseException):
            raise result
        return result


def _context(**overrides):
    values = {
        "inputs": {"topic": "reefs", "tags": ["a", "b"]},
        "input_names": ("topic", "tags"),
        "variable_names": ("tone",),
        "output": {"answer": "recorded answer", "tags": ["a", "b"]},
        "run_id": "recorded-run", "node_id": "recorded-node", "node_started_seq": 3,
        "node_data": {
            "prompt_template": "${topic}: ${tone}",
            "variables": {"tone": "concise"},
            "variable_domains": {"tone": 'Literal["concise","detailed"]'},
            "rendered_prompt": "reefs: concise",
            "effective_llm_config": {"provider": "test", "model": "recorded", "temperature": 0.4},
            "_evaluation_evidence": None,
        },
        "downstream_context": ({"message_index": 0, "contexts": [{"expression": "items[1]"}]},),
    }
    values.update(overrides)
    return BackwardContext(**values)


def _reply(**overrides):
    value = {"input_feedback": {}, "self_feedback": [], "variable_feedback": {}}
    value.update(overrides)
    return AIMessage(content=json.dumps(value))


def _retained_engine(monkeypatch):
    from yian.compile.model import START_ID, FlowOutput
    from yian.compose.helpers.types import read_variables
    from yian.nodes.contracts import Parameter
    from tests.engine._graph_builder import _engine, _graph

    node = AgentNode("writer", prompt="${topic}: ${tone}", tools=["not-used"],
                     llm_config={"provider": "test", "model": "recorded"},
                     variables=read_variables({"tone": "Literal[concise,detailed] = concise"}, {}))
    node.params = [Parameter("topic"), Parameter("tags")]
    flow = _graph([node], [(START_ID, "writer")],
                  wiring={"writer": {"topic": "${input.topic}", "tags": "${input.tags}"}},
                  outputs=[FlowOutput("result", "${writer.output}")])
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _config: _BackwardChat([AIMessage(content="forward answer")]))
    engine = _engine(flow, {"topic": "reefs", "tags": ["a", "b"]}, learnable=True)
    original = deepcopy(engine.state.flow)
    list(engine.run())
    return engine, node, original


def test_backward_context_exposes_only_accepted_evidence_and_replays_without_agent_calls(monkeypatch):
    import yian.nodes.agent.node as agent_module
    from yian.compile.expand import build_grow
    from yian.runtime.log import _evidence_payload
    from yian.runtime.state import RunState, _replay_records

    engine, node, original = _retained_engine(monkeypatch)
    occurrence = next(item for item in engine.state.trace.occurrences if item.node_id == "writer")
    accepted = occurrence.record.forward_capture.node_data
    expected = {**accepted, "_evaluation_evidence": _evidence_payload(occurrence.record.evidence)}
    assert "_evaluation_evidence" not in accepted
    assert engine.state.trace.backward_context(occurrence.source).node_data == expected
    node.prompt = "later prompt"
    node.llm_config["model"] = "later model"

    def forbidden(*_args, **_kwargs):
        pytest.fail("context reconstruction or replay invoked AGENT work")

    monkeypatch.setattr(AgentNode, "_resolve_variables", forbidden)
    monkeypatch.setattr(AgentNode, "backward", forbidden)
    monkeypatch.setattr(agent_module, "render_template_record", forbidden)
    monkeypatch.setattr(llm_clients, "model_from_config", forbidden)
    replay = _replay_records(RunState(original), tuple(engine.log), grow_builder=build_grow)
    assert replay.trace.backward_context(occurrence.source).node_data == expected
    assert replay.trace.occurrence(occurrence.source).record.forward_capture.node_data == accepted
    expected["_evaluation_evidence"]["interactions"].clear()
    assert replay.trace.backward_context(occurrence.source).node_data["_evaluation_evidence"]["interactions"]


def test_backward_context_rejects_captured_reserved_evidence_collision():
    from yian.compile.model import START_ID, FlowOutput
    from tests.engine._fakes import FuncNode
    from tests.engine._graph_builder import _engine, _graph

    node = FuncNode("custom", lambda _inputs: "done")
    node.capture_backward_context = lambda: {"_evaluation_evidence": None}
    engine = _engine(_graph([node], [(START_ID, "custom")], outputs=[FlowOutput("result", "${custom.output}")]), learnable=True)
    list(engine.run())
    occurrence = next(item for item in engine.state.trace.occurrences if item.node_id == "custom")
    with pytest.raises(ValueError, match="reserved.*_evaluation_evidence"):
        engine.state.trace.backward_context(occurrence.source)


def test_backward_context_allows_absent_node_data_without_changing_the_record():
    from yian.compile.model import START_ID
    from yian.runtime.state import commit_prepared, prepare_node_started, prepare_node_succeeded
    from tests.engine._graph_builder import _engine, _forward_capture, _graph

    engine = _engine(_graph([], []), learnable=True)
    state = engine.state
    commit_prepared(engine.log, state, prepare_node_started(state, START_ID))
    commit_prepared(engine.log, state, prepare_node_succeeded(
        state, START_ID, {}, forward_capture=_forward_capture(state.flow.nodes[START_ID], {}),
    ))
    occurrence, = state.trace.occurrences
    assert occurrence.record.forward_capture.node_data is None
    assert state.trace.backward_context(occurrence.source).node_data == {"_evaluation_evidence": None}
    assert occurrence.record.forward_capture.node_data is None


def test_agent_backward_uses_recorded_model_and_no_tools(monkeypatch):
    node = AgentNode("mutable-node", prompt="mutable prompt", tools=["never-granted"],
                     llm_config={"provider": "ambient", "model": "wrong"})
    context = _context()
    feedback = ({"tags": [None, "Check the second tag."]}, "Explain the answer.")
    expected = BackwardResult(
        {"tags": ([None, "Use a relevant tag."],)}, ("Preserve uncertainty.",), {"tone": ("Be precise.",)},
    )
    chat = _BackwardChat([_reply(input_feedback={"tags": [[None, "Use a relevant tag."]]},
                                 self_feedback=["Preserve uncertainty."], variable_feedback={"tone": ["Be precise."]})])
    configs = []

    def factory(config):
        configs.append(deepcopy(config))
        config["model"] = "factory mutation"
        return chat

    monkeypatch.setattr(llm_clients, "model_from_config", factory)
    result = node.backward(context, feedback)
    assert result == expected
    validate_backward_result(result, context)
    assert configs == [context.node_data["effective_llm_config"]]
    assert chat.calls == 1 and chat.bound_tools is None
    assert [type(message) for message in chat.messages[0]] == [SystemMessage, HumanMessage]
    prompt = json.loads(chat.messages[0][1].content)
    assert prompt["forward"]["inputs"] == context.inputs
    assert prompt["forward"]["output"] == context.output
    assert prompt["forward"]["prompt_template"] == "${topic}: ${tone}"
    assert prompt["forward"]["rendered_prompt"] == "reefs: concise"
    assert prompt["forward"]["variables"] == {"tone": "concise"}
    assert prompt["forward"]["variable_domains"] == context.node_data["variable_domains"]
    assert prompt["feedback"] == list(feedback)
    assert prompt["downstream_context"] == list(context.downstream_context)
    assert context.node_data["effective_llm_config"]["model"] == "recorded"
    assert node.prompt == "mutable prompt" and node.tools == ["never-granted"]


def test_agent_backward_keeps_visible_tool_evidence_inert_and_separate(monkeypatch):
    from yian.memory import EvaluationEvidence, ModelTextInteraction, ToolCallInteraction, ToolResultInteraction
    from yian.runtime.log import _evidence_payload

    evidence = EvaluationEvidence(None, {}, "Recorded base system.", "Recorded mode prompt.", interactions=(
        ToolCallInteraction("search", {"query": "reefs"}), ToolResultInteraction("search", "Found reefs."),
        ModelTextInteraction("Recorded visible answer."),
    ))
    data = _context().node_data
    data["_evaluation_evidence"] = _evidence_payload(evidence)
    context = _context(node_data=data)
    chat = _BackwardChat([_reply()])
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _config: chat)
    result = AgentNode("unused", tools=["search"]).backward(context, ("Review.",))
    assert result == BackwardResult()
    assert chat.calls == 1 and chat.bound_tools is None
    assert [type(message) for message in chat.messages[0]] == [SystemMessage, HumanMessage]
    prompt = json.loads(chat.messages[0][1].content)
    assert prompt["forward"]["evaluation_evidence"] == _evidence_payload(evidence)
    assert prompt["forward"]["rendered_prompt"] == "reefs: concise"


@pytest.mark.parametrize("feedback", [(), (None,), ({},), ({"tags": [None, None]},)])
def test_agent_backward_skips_empty_effective_feedback(monkeypatch, feedback):
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _config: pytest.fail("empty feedback constructed a model"))
    assert AgentNode("unused").backward(_context(node_data={}), feedback) == BackwardResult()


@pytest.mark.parametrize("feedback", [([None],), ({"tags": [None]},), ("review", {"unknown": None}), ["review"]])
def test_agent_backward_validates_all_feedback_before_skipping_or_calling(monkeypatch, feedback):
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _config: pytest.fail("invalid feedback constructed a model"))
    with pytest.raises((TypeError, ValueError)):
        AgentNode("unused").backward(_context(), feedback)


def test_agent_backward_empty_string_is_effective_and_nested_arrays_keep_their_shape(monkeypatch):
    chat = _BackwardChat([_reply(input_feedback={"tags": [[None, ""]]}, self_feedback=["", "second"],
                                 variable_feedback={"tone": ["first", "second"]})])
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _config: chat)
    context = _context()
    result = AgentNode("unused").backward(context, ("",))
    assert chat.calls == 1
    assert result.input_feedback == {"tags": ([None, ""],)}
    assert result.self_feedback == ("", "second")
    assert result.variable_feedback == {"tone": ("first", "second")}
    validate_backward_result(result, context)
    result.input_feedback["tags"][0][1] = "reader mutation"
    assert result.input_feedback["tags"] == ([None, ""],)


@pytest.mark.parametrize("key", [
    "prompt_template", "rendered_prompt", "variables", "variable_domains", "effective_llm_config", "_evaluation_evidence",
])
def test_agent_backward_rejects_missing_retained_material_without_fallback(monkeypatch, key):
    data = _context().node_data
    del data[key]
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _config: pytest.fail("missing capture constructed a model"))
    with pytest.raises(ValueError, match="recorded capture"):
        AgentNode("unused", llm_config={"provider": "ambient", "model": "fallback"}).backward(_context(node_data=data), ("review",))


@pytest.mark.parametrize("key,value", [
    ("prompt_template", None), ("rendered_prompt", []), ("variables", []),
    ("variables", {"foreign": "value"}), ("variables", None),
    ("variable_domains", {"tone": 1}), ("variable_domains", {}), ("variable_domains", None),
    ("effective_llm_config", None), ("effective_llm_config", {}),
    ("effective_llm_config", {"provider": "test", "model": " "}),
    ("effective_llm_config", {"provider": None, "model": "recorded"}),
    ("_evaluation_evidence", []),
])
def test_agent_backward_rejects_malformed_retained_capture(monkeypatch, key, value):
    data = _context().node_data
    data[key] = value
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _config: pytest.fail("malformed capture constructed a model"))
    with pytest.raises((TypeError, ValueError)):
        AgentNode("unused").backward(_context(node_data=data), ("review",))


def test_agent_backward_uses_partial_failure_capture_without_inventing_locals(monkeypatch):
    from yian.diagnostics import Failure

    data = _context().node_data
    data.update(variables=None, rendered_prompt=None)
    context = _context(inputs=None, output=None, failure=Failure("node.execute", "resolution failed", "recorded-node"), node_data=data)
    chat = _BackwardChat([_reply(self_feedback=["Check resolution."], variable_feedback={"tone": ["Use a valid default."]})])
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _config: chat)
    result = AgentNode("unused").backward(context, ("Explain the failure.",))
    validate_backward_result(result, context)
    prompt = json.loads(chat.messages[0][1].content)
    assert prompt["forward"]["inputs"] is None
    assert prompt["forward"]["variables"] is None and prompt["forward"]["rendered_prompt"] is None
    assert prompt["forward"]["failure"]["message"] == "resolution failed"
    assert result.input_feedback == {}


@pytest.mark.parametrize("response,code", [
    ("not JSON", "backward.execute"),
    ("[]", "backward.execute"),
    ('{"input_feedback":{},"self_feedback":[],"variable_feedback":{},"other":null}', "backward.execute"),
    ('{"input_feedback":{},"self_feedback":[],"self_feedback":[],"variable_feedback":{}}', "backward.execute"),
    ('{"input_feedback":{},"self_feedback":["\\ud800"],"variable_feedback":{}}', "backward.execute"),
    (_reply(input_feedback={"tags": "flat batch"}).content, "backward.execute"),
    (_reply(self_feedback=[None]).content, "backward.execute"),
    (_reply(variable_feedback={"tone": [1]}).content, "backward.execute"),
    (_reply(input_feedback={"tags": [False]}).content, "backward.execute"),
    (_reply(input_feedback={"foreign": ["bad"]}).content, "backward.result"),
    (_reply(variable_feedback={"foreign": ["bad"]}).content, "backward.result"),
    (_reply(input_feedback={"tags": [[None]]}).content, "backward.result"),
    (ValueError("model failed"), "backward.execute"),
])
def test_agent_backward_failure_is_attributed_without_forward_changes(monkeypatch, response, code):
    from yian.runtime.backward import _BackwardComputeError, compute_backward

    engine, node, _original = _retained_engine(monkeypatch)
    before = (tuple(engine.log), engine.state.result, engine.state.trace.occurrences, node.variables[0].default)
    chat = _BackwardChat([response])
    monkeypatch.setattr(llm_clients, "model_from_config", lambda _config: chat)
    with pytest.raises(_BackwardComputeError) as caught:
        compute_backward(engine.state.trace, "Review.")
    failure = caught.value.failure
    occurrence = next(item for item in engine.state.trace.occurrences if item.node_id == "writer")
    assert failure.code == code and failure.node_id == "writer"
    assert failure.node_started_seq == occurrence.source.execution_seq
    assert failure.run_id == engine.state.run_id and failure.scope == occurrence.record.forward_capture.scope
    assert chat.calls == 1 and chat.bound_tools is None
    assert before == (tuple(engine.log), engine.state.result, engine.state.trace.occurrences, node.variables[0].default)


def test_agent_backward_reads_no_live_render_model_or_variable_state(monkeypatch):
    import yian.nodes.agent.node as agent_module
    import yian.nodes.agent.memory as memory_adapter
    from yian.runtime.backward import compute_backward
    from yian.runtime.learning import validate_attributed_backward

    engine, node, _original = _retained_engine(monkeypatch)
    node.prompt = "wrong"
    node.llm_config = {"provider": "wrong", "model": "wrong"}
    node.variables.clear()
    node.params.clear()

    def forbidden(*_args, **_kwargs):
        pytest.fail("backward reconstructed live forward material")

    monkeypatch.setattr(AgentNode, "_resolve_variables", forbidden)
    monkeypatch.setattr(AgentNode, "declared_variables", forbidden)
    monkeypatch.setattr(agent_module, "render_template_record", forbidden)
    monkeypatch.setattr(agent_module, "canonical_type_label", forbidden)
    monkeypatch.setattr(memory_adapter, "system_prompt_with_memory", forbidden)
    configs = []
    chat = _BackwardChat([_reply(input_feedback={"topic": ["Use a specific reef."]}, variable_feedback={"tone": ["Be precise."]})])
    monkeypatch.setattr(llm_clients, "model_from_config", lambda config: configs.append(deepcopy(config)) or chat)
    root, attributed = compute_backward(engine.state.trace, "Review.")
    assert root.input_feedback == {"topic": ("Use a specific reef.",)}
    assert configs == [{"provider": "test", "model": "recorded"}]
    validate_attributed_backward(engine.state.trace, "Review.", root, attributed)
    assert chat.calls == 1


@pytest.mark.parametrize("stage", ["factory", "invoke"])
@pytest.mark.parametrize("kind", ["future_cancel", "async_cancel", "memory", "host_stop", "wrapped_memory", "wrapped_cancel", "nested_capability"])
def test_agent_backward_host_controls_preserve_original_identity(monkeypatch, stage, kind):
    from asyncio import CancelledError as AsyncCancelledError
    from concurrent.futures import CancelledError as FutureCancelledError
    from yian.nodes.base import _CapabilityFault, _HostStop
    from yian.runtime.backward import compute_backward

    originals = {
        "future_cancel": FutureCancelledError("cancelled"),
        "async_cancel": AsyncCancelledError("cancelled"),
        "memory": MemoryError("allocation failed"),
        "host_stop": _HostStop("stop"),
        "wrapped_memory": MemoryError("wrapped allocation failed"),
        "wrapped_cancel": FutureCancelledError("wrapped cancellation"),
        "nested_capability": RuntimeError("original capability failure"),
    }
    original = originals[kind]
    raised = original
    if kind.startswith("wrapped"):
        raised = ValueError("model wrapper")
        raised.__cause__ = original
    if kind == "nested_capability":
        raised = _CapabilityFault(_CapabilityFault(original))
    engine, _node, _program = _retained_engine(monkeypatch)
    before = tuple(engine.log), engine.state.result, engine.state.trace.occurrences
    chat = _BackwardChat([raised])

    def factory(_config):
        if stage == "factory":
            raise raised
        return chat

    monkeypatch.setattr(llm_clients, "model_from_config", factory)
    with pytest.raises(type(original)) as caught:
        compute_backward(engine.state.trace, "Review.")
    assert caught.value is original
    assert chat.calls == (stage == "invoke")
    assert before == (tuple(engine.log), engine.state.result, engine.state.trace.occurrences)
