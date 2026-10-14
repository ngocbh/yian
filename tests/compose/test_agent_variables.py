"""AGENT-local declaration loading without model execution."""

import pytest

from yian.compose import LoadError, load_flow
from yian.compose.nodes.agent import AgentDescriptor
from yian.compose.nodes.base import BuildContext
from yian.compose.source import SourceDocument
from yian.nodes.contracts import Parameter, ValueContract
from yian.typesys.types import TypeRegistry, read_typedefs

from tests.compose._declarations import parameters_from_yaml


def _yaml_variables(declarations: str, registry: TypeRegistry | None = None) -> list[Parameter]:
    """Read `declarations` (indented `name: !T default` lines) as agent variables."""
    return parameters_from_yaml(
        "variables:\n" + declarations,
        registry or TypeRegistry(),
        field="variables",
        require_default=True,
    )


def _reading(parameters: list[Parameter], registry=None) -> BuildContext:
    """Return a build context whose variable reader returns `parameters`."""
    return BuildContext(registry or {}, variable_declaration_reader=lambda _node, _raw: parameters)


def test_agent_build_resolves_typed_variables_without_input_bindings():
    registry = read_typedefs({
        "Settings": {"limit": "int(1..12)", "tags": "list(str)"},
        "Tone": "Literal(concise, detailed, technical)",
    })
    variables = _yaml_variables(
        "  num_sentences: !int(1..12) 4\n"
        "  temperature: !float(0..1) 0.4\n"
        "  tone: !Tone concise\n"
        '  instruction: !str "Summarize clearly"\n'
        "  settings: !Settings {limit: 4, tags: [a, b]}\n"
        "  items: !list(int(1..12)) [1, 4]\n"
        "  optional: !Optional(str) null\n"
        '  empty: !str ""\n'
        "  compatible: !float(0..4) 4\n"
        "  literal: !str ${input.topic}\n",
        registry,
    )
    result = AgentDescriptor(id="writer", variables={variable.name: None for variable in variables}).build(
        _reading(variables, registry)
    )
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
    assert result.node.params == [] and result.wiring == {}
    assert result.node.llm_config == {}


@pytest.mark.parametrize("declaration", [
    "!int", "!Optional(str)", "!int ''", "!int 4.5", '!int "4"',
    "!int(1..12) 13", "!float(0..1) 1.1", "!AB c",
    "!str null", "!int true", "!list(int) [1, wrong]", "{field: str}", "4",
])
def test_agent_variables_reject_missing_empty_or_invalid_defaults(declaration):
    registry = read_typedefs({"AB": "Literal(a, b)"})
    with pytest.raises(LoadError):
        _yaml_variables(f"  value: {declaration}\n", registry)


@pytest.mark.parametrize("variables", [None, [], "x", 4, {"": "int"}, {1: "int"}])
def test_agent_variables_reject_malformed_mappings_and_names(variables):
    with pytest.raises(LoadError):
        AgentDescriptor(id="writer", variables=variables).build(BuildContext({}))


def test_variable_reader_can_decline_a_source_free_declaration():
    calls = []

    def reader(node_id, declarations):
        calls.append((node_id, declarations))
        return None

    declarations = {"count": "int(1..12)"}
    # Declining hands the declarations to the source-free reader, which takes a type only,
    # so the variable has no default.
    with pytest.raises(LoadError, match="variable 'count'"):
        AgentDescriptor(id="writer", variables=declarations).build(
            BuildContext({}, variable_declaration_reader=reader),
        )
    assert calls == [("writer", declarations)]


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
typedefs:
  Tone: Literal(concise, detailed)
nodes:
  writer:
    kind: agent
    input:
      topic: supplied
    variables:
      tone: !Tone concise
      instruction: !str ${missing.output}
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
      topic: !str fallback
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
        AgentDescriptor(id="writer", inputs={"topic": "supplied"}, variables={"topic": None}).build(
            _reading(_yaml_variables("  topic: !str fallback\n"))
        )


@pytest.mark.parametrize("declaration", ['!int "4"', "!int(1..12) 13", "!str", "!Optional(str)"])
def test_invalid_variable_default_selects_the_declaration_value(declaration):
    text = f"""\
id: invalid-variable-default
name: invalid-variable-default
nodes:
  writer:
    kind: agent
    variables:
      value: {declaration}
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
      {key}: !int 4
    output: str
output: ${{writer.output}}
"""
    document = SourceDocument.parse(text)
    variables = document.root.child("nodes").child("writer").child("variables")
    with pytest.raises(LoadError, match=message) as caught:
        load_flow(text)
    assert caught.value.diagnostic.frames[-1].range == variables.child(next(iter(variables.value))).key_range
