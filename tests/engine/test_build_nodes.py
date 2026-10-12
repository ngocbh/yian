"""Unit tests for the per-kind `descriptor.build(ctx)` — the leaf runtime Node builder.

A leaf descriptor (agent/code/tool) + a typedefs registry (via `BuildContext`) build the
matching runtime `Node`, stamped with:
  - `output_contract = ValueContract(read_type(descriptor.outputs, registry))`
    (None if no authored `output:`),
  - untyped `Parameter(name=k)` values plus flow-owned wiring from the descriptor's
    authored `input:` map; a TOOL's `input:` map binds the same way.

`desc.build(ctx)` returns a `BuildResult` (`.node`/`.wiring`/`.edges`). Mirrors the per-kind
ctor args.
"""

from pathlib import Path

import pytest

from yian.nodes.agent import AgentNode
from yian.nodes.code import CodeNode
from yian.nodes.contracts import Parameter, ValueContract
from yian.nodes.tool import ToolNode
from yian.llm_clients import LLMConfig
from yian.typesys.values import ValueKind
from yian.typesys.types import read_typedefs
from yian.compose import LoadError
from yian.compose.helpers.errors import LoadDiagnostics
from yian.compose.nodes.base import BuildContext
from yian.compose.parse.file import parse_file
from yian.compose.parse.nodes import parse_nodes
from yian.compose.source import SourceCursor, SourceDocument

_SEEDS = Path(__file__).resolve().parents[2] / "tests" / "seeds"


def _build(desc, registry=None, diagnostics=None):
    """Build a leaf descriptor and extract its `(node, wiring)` pair."""
    result = desc.build(
        BuildContext(
            registry if registry is not None else {},
            diagnostics=diagnostics or LoadDiagnostics(),
        )
    )
    return result.node, result.wiring


def _seed_nodes(name: str):
    path = _SEEDS / name
    document = SourceDocument.parse(path.read_text(), label=str(path))
    parse_file(document)
    return parse_nodes(document.root.child("nodes"))


# ---------- agent ----------


def test_agent_node_built_with_type_and_bindings():
    # seed 00 — a scalar (str) output, one ${input.topic} sink binding.
    desc = _seed_nodes("00-hello-agent.yaml")["note"]
    node, wiring = _build(desc)
    assert isinstance(node, AgentNode)
    assert node.id == "note"
    assert type(node.output_contract) is ValueContract
    assert node.output_contract.type.kind == ValueKind.STRING
    # the node-side signature is the param NAMES; the flow owns the source.
    assert [p.name for p in node.params] == ["topic"]
    assert wiring == {"topic": "${input.topic}"}
    # Body-node sink params are exact, source-neutral, and untyped.
    assert type(node.params[0]) is Parameter
    assert node.params[0].contract is None


def test_agent_node_object_output():
    # seed 01 score — an object {rating, rationale} output.
    desc = _seed_nodes("01-structured-agent.yaml")["score"]
    node, _ = _build(desc)
    assert isinstance(node, AgentNode)
    assert type(node.output_contract) is ValueContract
    assert node.output_contract.type.kind == ValueKind.OBJECT
    assert node.output_contract.type.fields["rating"].kind == ValueKind.NUMBER
    assert node.output_contract.type.fields["rationale"].kind == ValueKind.STRING


def test_agent_unknown_mode_is_loud_loaderror():
    # An unknown AGENT mode is rejected at build as the loader's one error type
    # (LoadError, loud) — not a bare ValueError leaking from the node ctor.
    text = (
        "id: bad-mode\nname: bad\ninput:\n  topic: str\n"
        "nodes:\n  note:\n    kind: agent\n    inputs:\n      topic: ${input.topic}\n"
        '    output: str\n    prompt: "hi ${topic}"\n    mode: react\n'
        "output: ${note.output}\n"
    )
    document = SourceDocument.parse(text)
    parse_file(document)
    desc = parse_nodes(document.root.child("nodes"))["note"]
    with pytest.raises(LoadError, match="unknown mode") as caught:
        _build(desc, diagnostics=LoadDiagnostics.from_document(document))
    expected = desc.field_ranges[("field", "mode")]
    assert caught.value.diagnostic.frames[0].range == expected
    assert caught.value.diagnostic.frames[0].text == text


def test_agent_invalid_llm_config_uses_the_config_field_range():
    text = (
        "id: bad-config\nname: bad\n"
        "nodes:\n  note:\n    kind: agent\n"
        "    llm_config:\n      provider: bogus_provider\n"
        "    prompt: hi\n"
        "output: ${note.output}\n"
    )
    document = SourceDocument.parse(text)
    parse_file(document)
    desc = parse_nodes(document.root.child("nodes"))["note"]

    with pytest.raises(LoadError, match="llm_config") as caught:
        _build(desc, diagnostics=LoadDiagnostics.from_document(document))

    expected = desc.field_ranges[("field", "llm_config")]
    assert caught.value.diagnostic.frames[0].range == expected
    assert caught.value.diagnostic.frames[0].text == text


@pytest.mark.parametrize("value", ["bogus", "1", "false", "[bogus]"])
def test_agent_non_mapping_llm_config_is_a_ranged_load_error(value):
    text = (
        "id: bad-config\nname: bad\n"
        "nodes:\n  note:\n    kind: agent\n"
        f"    llm_config: {value}\n"
        "    prompt: hi\n"
        "output: ${note.output}\n"
    )
    document = SourceDocument.parse(text)
    parse_file(document)
    desc = parse_nodes(document.root.child("nodes"))["note"]

    with pytest.raises(LoadError, match="must be a mapping") as caught:
        _build(desc, diagnostics=LoadDiagnostics.from_document(document))

    expected = desc.field_ranges[("field", "llm_config")]
    assert caught.value.diagnostic.frames[0].range == expected
    assert caught.value.diagnostic.frames[0].text == text


def test_agent_non_string_mode_is_a_ranged_load_error():
    text = (
        "id: bad-mode\nname: bad\n"
        "nodes:\n  note:\n    kind: agent\n"
        "    mode: [plain]\n"
        "    prompt: hi\n"
        "output: ${note.output}\n"
    )
    document = SourceDocument.parse(text)
    parse_file(document)
    desc = parse_nodes(document.root.child("nodes"))["note"]

    with pytest.raises(LoadError, match="must be a string") as caught:
        _build(desc, diagnostics=LoadDiagnostics.from_document(document))

    expected = desc.field_ranges[("field", "mode")]
    assert caught.value.diagnostic.frames[0].range == expected
    assert caught.value.diagnostic.frames[0].text == text


def test_agent_bad_mode_loaderror_is_located_via_load_flow():
    # The descriptor owns the precise `mode:` range, so the build-time LoadError points
    # directly at that field rather than relying on the conductor's node-header line.
    from yian.compose.loader import load_flow

    bad_mode = (
        "id: f\nname: f\n"
        "nodes:\n"  # line 3
        "  a:\n"  # line 4 — the node
        "    kind: agent\n"
        "    prompt: hi\n"
        "    mode: not_a_real_mode\n"
        "output: ${a.output}\n"
    )
    with pytest.raises(LoadError, match="unknown mode") as ei:
        load_flow(bad_mode, search_paths=["."])
    assert ei.value.diagnostic.frames[0].range.start_line == 7
    assert ei.value.diagnostic.frames[0].text == bad_mode


def test_agent_node_knobs():
    # seed 14 — mode / tools / llm_config carried onto the AgentNode.
    desc = _seed_nodes("14-agent-tools.yaml")["reviewer"]
    node, _ = _build(desc)
    assert isinstance(node, AgentNode)
    assert node.mode == "tool_calling"
    assert node.tools == ["get_web_data", "web_search", "ask_user"]
    # llm_config is a plain dict end-to-end.
    assert isinstance(node.llm_config, dict)
    assert node.llm_config["provider"] == "anthropic"
    assert node.llm_config["model"] == "claude-opus-4-8"
    assert node.llm_config["temperature"] == 0.3
    assert node.prompt.startswith("Research ${topic}")
    assert node.output_contract.type.fields["confidence"].kind == ValueKind.NUMBER


def test_node_name_becomes_title():
    # node_name -> the Node title; absent -> None. A constructed descriptor pins
    # the node_name -> title mapping (seed leaf nodes carry no node_name).
    note, _ = _build(_seed_nodes("00-hello-agent.yaml")["note"])
    assert note.title is None
    named, _ = _build(
        parse_nodes(
            SourceCursor.virtual(
                {
                    "n": {
                        "kind": "code",
                        "node_name": "Format verdict",
                        "inputs": {"x": "${input.x}"},
                        "outputs": "str",
                        "code": "m:f",
                    }
                }
            )
        )["n"],
    )
    assert named.title == "Format verdict"


# ---------- code ----------


def test_code_node_built():
    # seed 01 verdict — a CODE node (module:function ref).
    desc = _seed_nodes("01-structured-agent.yaml")["verdict"]
    node, wiring = _build(desc)
    assert isinstance(node, CodeNode)
    assert node.ref == "tests.seeds.fns:one_line_summary"
    assert node.output_contract.type.kind == ValueKind.STRING
    assert {p.name for p in node.params} == {"rating", "rationale"}
    assert wiring == {
        "rating": "${score.output.rating}",
        "rationale": "${score.output.rationale}",
    }


# ---------- tool ----------


def test_tool_node_built_with_args_bindings():
    # seed 08 news — tool_id + untyped `inputs` -> InputBindings.
    desc = _seed_nodes("08-tool-news.yaml")["news"]
    node, wiring = _build(desc)
    assert isinstance(node, ToolNode)
    assert node.tool_id == "get_facts"
    assert node.output_contract is None  # no `output:` declared
    assert {p.name for p in node.params} == {"symbol", "limit"}
    assert wiring == {"symbol": "${input.topic}", "limit": 10}
    # Inputs are exact untyped sink parameters.
    assert all(type(p) is Parameter and p.contract is None for p in node.params)


# ---------- registry-resolved output type ----------


def test_call_loop_descriptors_need_a_resolver():
    # call/map/loop share the kind-blind `desc.build(ctx)` path but resolve a CHILD via
    # `ctx.resolver`. A leaf-only `BuildContext` (resolver=None, the default) can't resolve
    # the callable, so building one is loud — a `LoadError` naming the unresolvable callable,
    # not a bare `TypeError` leaking from the `None(...)` resolver call.
    from yian.compose.parse.file import (
        CallDescriptor,
        MapDescriptor,
        LoopDescriptor,
    )

    for desc in (
        CallDescriptor(id="c", call="child"),
        MapDescriptor(id="m", call="child", over="${input.xs}"),
        LoopDescriptor(id="p", call="body", times=3),
    ):
        with pytest.raises(LoadError, match="could not resolve callable"):
            _build(desc)


def test_output_contract_resolves_registry_name():
    registry = read_typedefs({"Rating": {"category": "str", "score": "float"}})
    # an agent descriptor whose outputs name a registry record.
    nodes = parse_nodes(
        SourceCursor.virtual(
            {
                "synth": {
                    "kind": "agent",
                    "inputs": {"topic": "${input.topic}"},
                    "outputs": "Rating",
                    "prompt": "x",
                }
            }
        )
    )
    node, _ = _build(nodes["synth"], registry)
    assert isinstance(node, AgentNode)
    assert type(node.output_contract) is ValueContract
    assert node.output_contract.type.kind == ValueKind.OBJECT
    assert node.output_contract.type.fields["category"].kind == ValueKind.STRING
