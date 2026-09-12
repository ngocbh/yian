"""Local variable resolution at the AGENT's ordinary prompt boundary."""

import pytest

import agent_composer.nodes.agent.node as agent_module
from agent_composer.compose.helpers.types import read_variables
from agent_composer.expr import ExpressionError
from agent_composer.nodes.agent import AgentNode
from agent_composer.nodes.base import Node, Output
from agent_composer.nodes.contracts import Parameter


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
    monkeypatch.setitem(agent_module.MODES, "plain", lambda ctx: calls.append(ctx) or Output(ctx.prompt))
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
    monkeypatch.setitem(agent_module.MODES, "plain", lambda ctx: Output(ctx.prompt))
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
    monkeypatch.setitem(agent_module.MODES, "plain", lambda ctx: Output(ctx.prompt))
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
    monkeypatch.setitem(agent_module.MODES, "plain", lambda ctx: calls.append(ctx) or Output(ctx.prompt))
    assert node.forward({}).value == "fallback"
    assert records == [{"note": None}]
    for prompt in ("${note}", "${note | 'fallback'}"):
        node.prompt = prompt
        with pytest.raises(ExpressionError, match="unresolved reference"):
            node.forward({})
    assert len(calls) == 1
    assert node.variables[0].default is None


@pytest.mark.parametrize("declared,inputs", [([], {"tone": "input"}), ([Parameter("tone")], {})])
def test_direct_variable_input_overlap_is_rejected_before_render_or_mode(monkeypatch, declared, inputs):
    node = AgentNode("writer", prompt="${tone}",
                     variables=read_variables({"tone": "str = concise"}, {}))
    node.params = declared
    calls = []
    monkeypatch.setattr(agent_module, "render_template_record", lambda *_args: calls.append("render"))
    monkeypatch.setitem(agent_module.MODES, "plain", lambda _ctx: calls.append("mode") or Output("bad"))
    with pytest.raises(ValueError, match="variable.*tone.*input"):
        node.forward(inputs)
    assert calls == []


def test_temperature_variable_renders_independently_of_effective_model_configuration(monkeypatch):
    from langchain_core.messages import AIMessage

    import agent_composer.llm_clients as llm_clients
    from agent_composer.compose import load_flow
    from agent_composer.run import run_flow
    from tests.engine.test_agent import _FakeChat

    loaded = load_flow("""id: variable_temperature
name: variable_temperature
input: {topic: str}
llm_config: {provider: test, model: test, temperature: 0.8}
nodes:
  writer:
    kind: agent
    mode: plain
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
    result = run_flow(loaded, {"topic": "reefs"}, learn=False)

    assert result.status == "succeeded" and result.output == "answer"
    assert result.learning is None
    assert configurations == [{"provider": "test", "model": "test", "temperature": 0.8}]
    assert messages[-1].content == "reefs: 0.4"
    assert chat.calls == 1 and scopes == [(None, None)]
    writer = loaded.compiled.nodes["writer"]
    assert writer.variables[0].default == 0.4


def test_ordinary_run_keeps_variables_out_of_wiring_and_pool_and_backward_dormant(monkeypatch):
    from agent_composer.compile.model import END_ID, START_ID
    from agent_composer.compose import load_flow
    from tests.engine._graph_builder import _engine

    flow = load_flow("""id: local_render
name: local_render
input: {topic: str}
nodes:
  writer:
    kind: agent
    mode: plain
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
    monkeypatch.setitem(agent_module.MODES, "plain", lambda ctx: Output(ctx.prompt))
    assert set(flow.wiring["writer"]) == {"topic"}
    assert [parameter.name for parameter in flow.nodes["writer"].params] == ["topic"]
    assert all(edge.from_ != "absent" for edge in flow.edges)
    engine = _engine(flow, {"topic": "reefs"}, learn=False)
    list(engine.run())

    expected = "reefs: concise, ${absent.output}"
    assert engine.state.pool.get(END_ID) == expected
    assert set(engine.state.pool.store) == {START_ID, "writer", END_ID}
    assert engine.state.pool.get(START_ID) == {"topic": "reefs"}
    assert engine.state.pool.get("writer") == expected
    assert engine.state.trace is None
