"""Local variable resolution at the AGENT's ordinary prompt boundary."""

import pytest

import yian.nodes.agent.node as agent_module
from yian.compose.helpers.types import read_variables
from yian.expr import ExpressionError
from yian.nodes.agent import AgentNode
from yian.nodes.base import Node, Output
from yian.nodes.contracts import Parameter


def _retained_agent(node, inputs=None, **engine_options):
    from yian.compile.model import START_ID, FlowOutput
    from tests.engine._graph_builder import _engine, _graph

    supplied = inputs or {}
    node.params = [Parameter(name) for name in supplied]
    flow = _graph([node], [(START_ID, node.id)],
                  wiring={node.id: {name: f"${{input.{name}}}" for name in supplied}},
                  outputs=[FlowOutput("answer", f"${{{node.id}.output}}")])
    engine = _engine(flow, supplied, learnable=True, **engine_options)
    list(engine.run())
    occurrence = next(item for item in engine.state.trace.occurrences if item.node_id == node.id)
    return engine, occurrence.record


def test_retained_agent_captures_actual_render_locals_once(monkeypatch):
    node = AgentNode("writer", prompt="${instruction}: ${topic}; ${count}/${count}",
                     llm_config={"provider": "openai", "model": "chosen", "openai": {"temperature": 0.8}},
                     variables=read_variables({
                         "count": "float[0,4] = 4", "instruction": "str = ${input.topic}",
                     }, {}))
    resolve, render = agent_module.apply_defaults, agent_module.render_template_record
    resolutions, renders = [], []

    def resolve_once(declarations, values):
        resolutions.append(declarations)
        return resolve(declarations, values)

    def render_once(template, values):
        renders.append((template, values))
        return render(template, values)

    monkeypatch.setattr(agent_module, "apply_defaults", resolve_once)
    monkeypatch.setattr(agent_module, "render_template_record", render_once)
    monkeypatch.setattr(agent_module, "run_agent", lambda ctx: Output(ctx.prompt))
    engine, record = _retained_agent(node, {"topic": "reefs"})
    data = record.forward_capture.node_data
    assert data == {
        "prompt_template": "${instruction}: ${topic}; ${count}/${count}",
        "variables": {"count": 4, "instruction": "${input.topic}"},
        "variable_domains": {"count": "float[0,4]", "instruction": "str"},
        "rendered_prompt": "${input.topic}: reefs; 4/4",
        "effective_llm_config": {"provider": "openai", "model": "chosen", "openai": {"temperature": 0.8}},
    }
    assert len(resolutions) == len(renders) == 1
    assert type(data["variables"]["count"]) is int
    assert record.forward_capture.bound_inputs == {"topic": "reefs"}
    assert engine.state.effective_llm_by_static_path["writer"] == data["effective_llm_config"]


def test_retained_render_locals_are_detached_before_render_and_model_callbacks(monkeypatch):
    node = AgentNode("writer", prompt="${settings}",
                     llm_config={"provider": "test", "model": "original"},
                     variables=read_variables({"settings": "object = {items: [original]}"}, {}))
    render = agent_module.render_template_record

    def mutating_render(template, values):
        values["settings"]["items"].append("render mutation")
        return render(template, values)

    def mutating_run(ctx):
        node.prompt = "different template"
        node.variables[0].default["items"].append("default mutation")
        ctx.llm_config["model"] = "different model"
        return Output(ctx.prompt)

    monkeypatch.setattr(agent_module, "render_template_record", mutating_render)
    monkeypatch.setattr(agent_module, "run_agent", mutating_run)
    engine, record = _retained_agent(node)
    data = record.forward_capture.node_data
    assert data["variables"] == {"settings": {"items": ["original"]}}
    assert data["prompt_template"] == "${settings}"
    assert "render mutation" in data["rendered_prompt"]
    assert data["effective_llm_config"] == {"provider": "test", "model": "original"}
    assert engine.state.effective_llm_by_static_path["writer"] == data["effective_llm_config"]
    data["variables"]["settings"]["items"].append("reader mutation")
    assert record.forward_capture.node_data["variables"] == {"settings": {"items": ["original"]}}


@pytest.mark.parametrize("stage", ["resolve", "render", "empty"])
def test_retained_agent_distinguishes_partial_and_empty_render_locals(monkeypatch, stage):
    from yian.runtime.log import NodeFailedRecord

    node = AgentNode("writer", prompt="" if stage == "empty" else "${note}",
                     variables=[] if stage == "empty" else read_variables({"note": "Optional[str] = null"}, {}))
    calls = []
    monkeypatch.setattr(agent_module, "run_agent", lambda ctx: calls.append(ctx.prompt) or Output(ctx.prompt))
    if stage == "resolve":
        def fail_resolution():
            raise ValueError("variable resolution failed")
        monkeypatch.setattr(node, "_resolve_variables", fail_resolution)
    _, record = _retained_agent(node)
    data = record.forward_capture.node_data
    assert data["prompt_template"] == node.prompt
    assert data["variable_domains"] == ({} if stage == "empty" else {"note": "Optional[str]"})
    assert data["variables"] == (None if stage == "resolve" else {} if stage == "empty" else {"note": None})
    assert data["rendered_prompt"] == ("" if stage == "empty" else None)
    assert record.forward_capture.bound_inputs == {}
    assert calls == ([""] if stage == "empty" else [])
    if stage != "empty":
        assert type(record) is NodeFailedRecord
        assert record.failure.code == "node.execute"


def test_retained_agent_canonicalizes_supported_file_variable_without_changing_default(monkeypatch):
    from yian.nodes.contracts import ValueContract
    from yian.runtime.log import RecordEnvelope, SCHEMA_VERSION, decode_record, encode_record
    from yian.typesys import FileRef, Type, ValueKind

    file = FileRef(uri="file:///tmp/artifact.txt")
    node = AgentNode("writer", prompt="Inspect the attachment.", variables=[
        Parameter("artifact", ValueContract(Type(ValueKind.FILE)), has_default=True, default=file),
    ])
    monkeypatch.setattr(agent_module, "run_agent", lambda ctx: Output(ctx.prompt))
    _, record = _retained_agent(node)
    assert record.forward_capture.node_data["variables"] == {
        "artifact": {"uri": "file:///tmp/artifact.txt", "mime": None, "name": None},
    }
    assert record.forward_capture.node_data["variable_domains"] == {"artifact": "file"}
    assert node.variables[0].default is file
    envelope = RecordEnvelope(SCHEMA_VERSION, "run", 3, record)
    assert decode_record(encode_record(envelope)) == envelope


@pytest.mark.parametrize("number", [float("inf"), float("-inf"), float("nan")])
def test_retained_agent_fails_on_nonfinite_variable_model_extras(monkeypatch, number):
    from pydantic import BaseModel, ConfigDict
    from yian.compile.model import START_ID, END_ID
    from yian.nodes.binding import BindingError
    from yian.nodes.contracts import ValueContract
    from yian.runtime.log import NodeFailedRecord
    from yian.typesys import Type, ValueKind
    from tests.engine._graph_builder import _engine, _graph

    class Extras(BaseModel):
        model_config = ConfigDict(extra="allow")

    node = AgentNode("writer", prompt="ready", variables=[
        Parameter("settings", ValueContract(Type(ValueKind.OBJECT)), has_default=True,
                  default={"nested": Extras(child=Extras(lost=number))}),
    ])
    calls = []
    monkeypatch.setattr(agent_module, "run_agent", lambda ctx: calls.append(ctx.prompt) or Output(ctx.prompt))
    with pytest.raises(BindingError, match="non-finite"):
        node.forward({})
    engine = _engine(_graph([node], [(START_ID, "writer"), ("writer", END_ID)]), learnable=True)
    list(engine.run())
    assert calls == []
    failed = [envelope.record for envelope in engine.log
              if isinstance(envelope.record, NodeFailedRecord)]
    assert [record.failure.node_id for record in failed] == ["writer"]
    assert failed[0].failure.code == "node.execute"


@pytest.mark.parametrize("phase", ["domains", "copy"])
def test_render_capture_preparation_faults_escape_with_original_host_identity(monkeypatch, phase):
    from yian.runtime.eval_node import eval_node
    from yian.typesys import VariablePool

    expected = MemoryError("capture preparation failed")
    node = AgentNode("writer", prompt="${settings}", variables=read_variables({
        "settings": "object = {items: [1]}",
    }, {}))
    copy = agent_module.deepcopy
    calls = []

    def copy_or_fail(value):
        calls.append(value)
        if len(calls) == 2:
            raise expected
        return copy(value)

    def fail_domain(_type):
        raise expected

    if phase == "domains":
        monkeypatch.setattr(agent_module, "canonical_type_label", fail_domain)
    else:
        monkeypatch.setattr(agent_module, "deepcopy", copy_or_fail)
    monkeypatch.setattr(agent_module, "run_agent", lambda _ctx: pytest.fail("model ran after capture fault"))
    with pytest.raises(MemoryError) as caught:
        eval_node(node, None, VariablePool(), capture_bindings=True)
    assert caught.value is expected


def test_retained_agent_keeps_early_binding_unavailable_and_capture_off_dormant(monkeypatch):
    from yian.runtime.eval_node import EvaluationFailure, eval_node
    from yian.typesys import VariablePool

    node = AgentNode("writer", prompt="${topic}", variables=read_variables({"tone": "str = concise"}, {}))
    node.params = [Parameter("topic", required=True)]
    calls = []
    monkeypatch.setattr(node, "_resolve_variables", lambda: calls.append("resolve") or {"tone": "concise"})
    monkeypatch.setattr(agent_module, "run_agent", lambda ctx: Output(ctx.prompt))
    missing = eval_node(node, None, VariablePool(), capture_bindings=True)
    assert type(missing.outcome) is EvaluationFailure
    assert missing.bindings.bound_inputs is None
    assert missing.backward_capture.node_data == {}
    assert missing.backward_capture.variable_names == ("tone",)
    assert calls == []
    ordinary = eval_node(node, None, VariablePool(), wiring_override={"topic": "reefs"})
    retained = eval_node(node, None, VariablePool(), wiring_override={"topic": "reefs"}, capture_bindings=True)
    assert ordinary.outcome == retained.outcome == Output("reefs")
    assert ordinary.backward_capture is None
    assert retained.backward_capture.node_data["effective_llm_config"] is None
    assert calls == ["resolve", "resolve"]


def test_declared_variable_inventory_preserves_canonical_declarations():
    declarations = read_variables({"tone": "str = concise", "count": "int[1,12] = 4"}, {})
    node = AgentNode("writer", variables=declarations)

    inventory = node.declared_variables()

    assert type(inventory) is tuple
    assert len(inventory) == 2
    assert all(actual is declared for actual, declared in zip(inventory, declarations))
    assert Node("ordinary").declared_variables() == ()
    assert AgentNode("empty").declared_variables() == ()


def test_agent_renders_inputs_and_variable_defaults_without_execution_or_memory_scope(monkeypatch):
    calls = []
    monkeypatch.setattr(agent_module, "run_agent", lambda ctx: calls.append(ctx) or Output(ctx.prompt))
    node = AgentNode("writer", prompt="${instruction}: ${topic}, ${num_sentences} sentences, ${tone}.",
                     variables=read_variables({
                         "num_sentences": "int[1,12] = 4",
                         "tone": "Literal[concise,detailed,technical] = concise",
                         "instruction": 'str = "Summarize clearly"',
                     }, {}))
    node.params = [Parameter("topic")]
    assert node.execution_scope is None and node.memory_scope is None
    assert node.forward({"topic": "coral reefs"}).value == "Summarize clearly: coral reefs, 4 sentences, concise."
    assert len(calls) == 1


def test_agent_resolves_and_renders_once_with_repeated_literal_default_references(monkeypatch):
    node = AgentNode("writer", prompt="${count}/${count}: ${instruction}/${empty}",
                     variables=read_variables({
                         "count": "float[0,4] = 4",
                         "instruction": "str = ${input.topic}",
                         "empty": 'str = ""',
                     }, {}))
    defaults = agent_module.apply_defaults
    render = agent_module.render_template_record
    resolutions, records = [], []

    def resolve(variables, inputs):
        resolutions.append((variables, inputs))
        return defaults(variables, inputs)

    def render_once(prompt, record):
        records.append(record)
        return render(prompt, record)

    monkeypatch.setattr(agent_module, "apply_defaults", resolve)
    monkeypatch.setattr(agent_module, "render_template_record", render_once)
    monkeypatch.setattr(agent_module, "run_agent", lambda ctx: Output(ctx.prompt))
    assert node.forward({}).value == "4/4: ${input.topic}/"
    assert len(resolutions) == len(records) == 1
    assert resolutions[0] == (node.variables, {})
    assert type(records[0]["count"]) is int


def test_mutable_variable_defaults_are_detached_between_invocations(monkeypatch):
    node = AgentNode("writer", prompt="${settings}", variables=read_variables({
        "settings": "object = {nested: {items: [a, b]}}",
    }, {}))
    default = node.variables[0].default
    records = []
    render = agent_module.render_template_record

    def capture(prompt, record):
        records.append(record)
        return render(prompt, record)

    monkeypatch.setattr(agent_module, "render_template_record", capture)
    monkeypatch.setattr(agent_module, "run_agent", lambda ctx: Output(ctx.prompt))
    first = node.forward({}).value
    records[0]["settings"]["nested"]["items"].append("changed")
    second = node.forward({}).value
    assert first == second
    assert default == {"nested": {"items": ["a", "b"]}}
    assert records[0]["settings"]["nested"] is not records[1]["settings"]["nested"]
    assert records[1]["settings"]["nested"]["items"] is not default["nested"]["items"]


def test_nullable_variable_defaults_follow_existing_strict_prompt_rules(monkeypatch):
    node = AgentNode("writer", prompt="${null | 'fallback'}",
                     variables=read_variables({"note": "Optional[str] = null"}, {}))
    calls, records = [], []
    render = agent_module.render_template_record

    def capture(prompt, record):
        records.append(record)
        return render(prompt, record)

    monkeypatch.setattr(agent_module, "render_template_record", capture)
    monkeypatch.setattr(agent_module, "run_agent", lambda ctx: calls.append(ctx) or Output(ctx.prompt))
    assert node.forward({}).value == "fallback"
    assert records == [{"note": None}]
    node.prompt = "${note | 'fallback'}"
    assert node.forward({}).value == "fallback"
    node.prompt = "${note}"
    with pytest.raises(ExpressionError, match=r"unresolved reference \$\{note\}"):
        node.forward({})
    assert len(calls) == 2
    assert node.variables[0].default is None


@pytest.mark.parametrize("declared,inputs", [([], {"tone": "input"}), ([Parameter("tone")], {})])
def test_direct_variable_input_overlap_is_rejected_before_render_or_run(monkeypatch, declared, inputs):
    node = AgentNode("writer", prompt="${tone}",
                     variables=read_variables({"tone": "str = concise"}, {}))
    node.params = declared
    calls = []
    monkeypatch.setattr(agent_module, "render_template_record", lambda *_args: calls.append("render"))
    monkeypatch.setattr(agent_module, "run_agent", lambda _ctx: calls.append("run") or Output("bad"))
    with pytest.raises(ValueError, match="variable.*tone.*input"):
        node.forward(inputs)
    assert calls == []


def test_temperature_variable_renders_independently_of_effective_model_configuration(monkeypatch):
    from langchain_core.messages import AIMessage

    import yian.llm_clients as llm_clients
    from yian.compose import load_flow
    from yian.run import run_flow
    from tests.engine.test_agent import _FakeChat

    loaded = load_flow("""id: variable_temperature
name: variable_temperature
input: {topic: str}
llm_config: {provider: ollama, model: test, ollama: {options: {temperature: 0.8}}}
nodes:
  writer:
    kind: agent
    input: {topic: '${input.topic}'}
    variables: {temperature: 'float[0,1] = 0.4'}
    prompt: '${topic}: ${temperature}'
    output: str
output: ${writer.output}
""")
    chat = _FakeChat([AIMessage(content="answer")])
    configurations, messages, scopes = [], [], []
    invoke = chat.invoke
    resolve = AgentNode._resolve_variables

    def model(config):
        configurations.append(config)
        return chat

    def record_messages(prompt):
        messages.extend(prompt)
        return invoke(prompt)

    def resolve_without_scope(node):
        scopes.append((node.execution_scope, node.memory_scope))
        return resolve(node)

    monkeypatch.setattr(llm_clients, "model_from_config", model)
    monkeypatch.setattr(chat, "invoke", record_messages)
    monkeypatch.setattr(AgentNode, "_resolve_variables", resolve_without_scope)
    result = run_flow(loaded, {"topic": "reefs"}, learnable=False)

    assert result.status == "succeeded" and result.output == "answer"
    assert result.learning is None
    assert configurations == [
        {"provider": "ollama", "model": "test", "ollama": {"options": {"temperature": 0.8}}}
    ]
    assert messages[-1].content == "reefs: 0.4"
    assert chat.calls == 1 and scopes == [(None, None)]
    writer = loaded.compiled.nodes["writer"]
    assert writer.variables[0].default == 0.4


def test_ordinary_run_keeps_variables_out_of_wiring_and_pool_and_backward_dormant(monkeypatch):
    from yian.compile.model import END_ID, START_ID
    from yian.compose import load_flow
    from tests.engine._graph_builder import _engine

    flow = load_flow("""id: local_render
name: local_render
input: {topic: str}
nodes:
  writer:
    kind: agent
    input: {topic: '${input.topic}'}
    variables:
      instruction: str = ${absent.output}
      tone: str = concise
    prompt: '${topic}: ${tone}, ${instruction}'
    output: str
output: ${writer.output}
""").compiled

    def forbidden_backward(*_args, **_kwargs):
        raise AssertionError("ordinary forward run invoked backward")

    monkeypatch.setattr(AgentNode, "backward", forbidden_backward)
    monkeypatch.setattr(agent_module, "run_agent", lambda ctx: Output(ctx.prompt))
    assert set(flow.wiring["writer"]) == {"topic"}
    assert [parameter.name for parameter in flow.nodes["writer"].params] == ["topic"]
    assert all(edge.from_ != "absent" for edge in flow.edges)
    engine = _engine(flow, {"topic": "reefs"}, learnable=False)
    list(engine.run())

    expected = "reefs: concise, ${absent.output}"
    assert engine.state.pool.get(END_ID) == expected
    assert set(engine.state.pool.store) == {START_ID, "writer", END_ID}
    assert engine.state.pool.get(START_ID) == {"topic": "reefs"}
    assert engine.state.pool.get("writer") == expected
    assert engine.state.trace is None
