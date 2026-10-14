"""AGENT-local declaration loading without model execution."""

import pytest

from yian.compose import LoadError, load_flow
from yian.compose.nodes.agent import AgentDescriptor
from yian.compose.nodes.base import BuildContext
from yian.compose.source import SourceDocument
from yian.nodes.contracts import Parameter, ValueContract
from yian.typesys.types import read_typedefs


def test_source_free_agent_build_resolves_typed_variables_without_input_bindings():
    registry = read_typedefs({"Settings": {"limit": "int[1,12]", "tags": "list[str]"}})
    result = AgentDescriptor(id="writer", variables={
        "num_sentences": "int[1,12] = 4",
        "temperature": "float[0,1] = 0.4",
        "tone": "Literal[concise,detailed,technical] = concise",
        "instruction": 'str = "Summarize clearly"',
        "settings": 'Settings = {limit: 4, tags: [a, b]}',
        "items": 'list[int[1,12]] = [1, 4]',
        "optional": "Optional[str] = null",
        "empty": 'str = ""',
        "compatible": "float[0,4] = 4",
        "literal": "str = ${input.topic}",
    }).build(BuildContext(registry))
    variables = result.node.variables
    assert all(type(variable) is Parameter for variable in variables)
    assert all(type(variable.contract) is ValueContract for variable in variables)
    assert all(variable.has_default is True and variable.required is False for variable in variables)
    assert {variable.name: variable.default for variable in variables} == {
        "num_sentences": 4, "temperature": 0.4, "tone": "concise",
        "instruction": "Summarize clearly", "settings": {"limit": 4, "tags": ["a", "b"]},
        "items": [1, 4], "optional": None, "empty": "", "compatible": 4,
        "literal": "${input.topic}",
    }
    assert type(next(variable.default for variable in variables if variable.name == "compatible")) is int
    assert all(list(variable.iter_origins()) == [] for variable in variables)
    assert result.node.params == [] and result.wiring == {}
    assert result.node.llm_config == {}


@pytest.mark.parametrize("declaration", [
    "int", "Optional[str]", "int =", "int = 4.5", 'int = "4"',
    "int[1,12] = 13", "float[0,1] = 1.1", "Literal[a,b] = c",
    "str = null", "int = true", "list[int] = [1, wrong]", {"field": "str"}, 4,
])
def test_agent_variables_reject_missing_empty_or_invalid_defaults(declaration):
    with pytest.raises(LoadError):
        AgentDescriptor(id="writer", variables={"value": declaration}).build(BuildContext({}))


@pytest.mark.parametrize("variables", [None, [], "x", 4, {"": "int = 1"}, {1: "int = 1"}])
def test_agent_variables_reject_malformed_mappings_and_names(variables):
    with pytest.raises(LoadError):
        AgentDescriptor(id="writer", variables=variables).build(BuildContext({}))


def test_variable_reader_can_decline_a_source_free_declaration():
    calls = []

    def reader(node_id, declarations):
        calls.append((node_id, declarations))
        return None

    declarations = {"count": "int[1,12] = 4"}
    node = AgentDescriptor(id="writer", variables=declarations).build(
        BuildContext({}, variable_declaration_reader=reader),
    ).node
    assert calls == [("writer", declarations)]
    assert node.variables[0].default == 4
    assert list(node.variables[0].iter_origins()) == []


@pytest.mark.parametrize("returned", [(), {}, [object()]])
def test_variable_reader_rejects_a_noncanonical_return_value(returned):
    with pytest.raises(TypeError, match="variable declaration reader"):
        AgentDescriptor(id="writer").build(
            BuildContext({}, variable_declaration_reader=lambda _node, _raw: returned),
        )


def test_loaded_agent_prompt_can_name_inputs_and_variables_without_default_data_edges():
    text = """\
id: local-variables
name: local-variables
nodes:
  writer:
    kind: agent
    input:
      topic: supplied
    variables:
      tone: Literal[concise,detailed] = concise
      instruction: str = ${missing.output}
    prompt: "${topic}: ${tone}, ${instruction}"
    output: str
output: ${writer.output}
"""
    loaded = load_flow(text)
    writer = loaded.compiled.nodes["writer"]
    assert [parameter.name for parameter in writer.params] == ["topic"]
    assert [variable.name for variable in writer.variables] == ["tone", "instruction"]
    assert writer.variables[1].default == "${missing.output}"
    assert set(loaded.compiled.wiring["writer"]) == {"topic"}
    assert all(edge.from_ != "missing" for edge in loaded.compiled.edges)
    with pytest.raises(LoadError, match="not a declared input or variable"):
        load_flow(text.replace("${tone},", "${unknown},"))


def test_variable_input_collision_selects_the_variable_declaration_key():
    text = """\
id: variable-collision
name: variable-collision
nodes:
  writer:
    kind: agent
    input:
      topic: supplied
    variables:
      topic: str = fallback
    prompt: ${topic}
    output: str
output: ${writer.output}
"""
    document = SourceDocument.parse(text)
    with pytest.raises(LoadError, match="variable.*topic.*input") as caught:
        load_flow(text)
    cursor = document.root.child("nodes").child("writer").child("variables").child("topic")
    assert caught.value.diagnostic.frames[-1].range == cursor.key_range
    with pytest.raises(LoadError, match="variable.*topic.*input"):
        AgentDescriptor(id="writer", inputs={"topic": "supplied"},
                        variables={"topic": "str = fallback"}).build(BuildContext({}))


@pytest.mark.parametrize("declaration", ['int = "4"', "int[1,12] = 13", "str =", "Optional[str]"])
def test_invalid_variable_default_selects_the_declaration_value(declaration):
    text = f"""\
id: invalid-variable-default
name: invalid-variable-default
nodes:
  writer:
    kind: agent
    variables:
      value: '{declaration}'
    output: str
output: ${{writer.output}}
"""
    document = SourceDocument.parse(text)
    with pytest.raises(LoadError) as caught:
        load_flow(text)
    cursor = document.root.child("nodes").child("writer").child("variables").child("value")
    assert caught.value.diagnostic.frames[-1].range == cursor.range


@pytest.mark.parametrize(
    ("key", "message"),
    [
        ("''", "names must be non-empty strings"),
        ("1", "names in a flow must be text"),
        ("true", "names in a flow must be text"),
    ],
)
def test_invalid_variable_name_selects_its_declaration_key(key, message):
    text = f"""\
id: invalid-variable-name
name: invalid-variable-name
nodes:
  writer:
    kind: agent
    variables:
      {key}: int = 4
    output: str
output: ${{writer.output}}
"""
    document = SourceDocument.parse(text)
    variables = document.root.child("nodes").child("writer").child("variables")
    with pytest.raises(LoadError, match=message) as caught:
        load_flow(text)
    assert caught.value.diagnostic.frames[-1].range == variables.child(next(iter(variables.value))).key_range
