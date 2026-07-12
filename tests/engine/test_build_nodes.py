"""Unit tests for build_leaf_node — the leaf runtime Node builder.

A leaf descriptor (agent/code/model/tool) + a typedefs registry build the
matching runtime `Node`, stamped with:
  - `output_type = read_type(descriptor.outputs, registry)` (None if no outputs),
  - `inputs = [InputBinding(name=k, source=v)]` from the descriptor's `inputs:`
    map (sink bindings — `type_str`/`type` lenient/None; the type is carried by the
    source); a TOOL's `args` map binds the same way (untyped).

Mirrors the per-kind ctor args. case/ref/map are later steps.
"""

from pathlib import Path

import pytest

from agent_composer.nodes.agent import AgentNode
from agent_composer.nodes.code import CodeNode
from agent_composer.nodes.model import ModelNode
from agent_composer.nodes.tool import ToolNode
from agent_composer.llm_clients import LLMConfig
from agent_composer.typesys.values import ValueKind
from agent_composer.typesys.types import read_typedefs
from agent_composer.compose import LoadError
from agent_composer.compose.build import build_leaf_node
from agent_composer.compose.parse.file import parse_file
from agent_composer.compose.helpers.reader import NodeReader
from agent_composer.compose.parse.nodes import parse_nodes

_SEEDS = Path(__file__).resolve().parents[2] / "tests" / "seeds"


def _seed_nodes(name: str):
    f = parse_file((_SEEDS / name).read_text())
    return parse_nodes(NodeReader.spanless(f.nodes))


# ---------- agent ----------


def test_agent_node_built_with_type_and_bindings():
    # seed 00 — a scalar (str) output, one ${input.topic} sink binding.
    desc = _seed_nodes("00-hello-agent.yaml")["note"]
    node, wiring = build_leaf_node(desc, {})
    assert isinstance(node, AgentNode)
    assert node.id == "note"
    assert node.output_type.kind == ValueKind.STRING
    # the node-side signature is the param NAMES; the flow owns the source.
    assert [p.name for p in node.params] == ["topic"]
    assert wiring == {"topic": "${input.topic}"}
    # sink params are lenient: the type travels with the source.
    assert node.params[0].type_str is None and node.params[0].type is None


def test_agent_node_object_output():
    # seed 01 score — an object {rating, rationale} output.
    desc = _seed_nodes("01-structured-agent.yaml")["score"]
    node, _ = build_leaf_node(desc, {})
    assert isinstance(node, AgentNode)
    assert node.output_type.kind == ValueKind.OBJECT
    assert node.output_type.fields["rating"].kind == ValueKind.NUMBER
    assert node.output_type.fields["rationale"].kind == ValueKind.STRING


def test_agent_unknown_mode_is_loud_loaderror():
    # An unknown AGENT mode is rejected at build as the loader's one error type
    # (LoadError, loud) — not a bare ValueError leaking from the node ctor.
    text = (
        "id: bad-mode\nname: bad\ninput:\n  topic: str\n"
        "nodes:\n  note:\n    kind: agent\n    inputs:\n      topic: ${input.topic}\n"
        "    output: str\n    prompt: \"hi ${topic}\"\n    mode: react\n"
        "output: ${note.output}\n"
    )
    desc = parse_nodes(NodeReader.spanless(parse_file(text).nodes))["note"]
    with pytest.raises(LoadError, match="unknown mode"):
        build_leaf_node(desc, {})


def test_agent_bad_mode_loaderror_is_located_via_load_flow():
    # Through the full load path the loader stamps the node's source line onto the
    # build-time LoadError (build_leaf_node raises it line-less; loader locates it),
    # so an invalid mode:/controls: points the author at the offending node.
    from agent_composer.compose.loader import load_flow

    bad_mode = (
        "id: f\nname: f\n"
        "nodes:\n"          # line 3
        "  a:\n"            # line 4 — the node
        "    kind: agent\n"
        "    prompt: hi\n"
        "    mode: not_a_real_mode\n"
        "output: ${a.output}\n"
    )
    with pytest.raises(LoadError, match="unknown mode") as ei:
        load_flow(bad_mode, search_paths=["."])
    assert ei.value.line == 4

    bad_controls = (
        "id: f\nname: f\n"
        "nodes:\n"
        "  a:\n"            # line 4 — the node
        "    kind: agent\n"
        "    prompt: hi\n"
        "    controls:\n"
        "      - bogus_control\n"
        "output: ${a.output}\n"
    )
    with pytest.raises(LoadError, match="unknown control") as ei:
        load_flow(bad_controls, search_paths=["."])
    assert ei.value.line == 4


def test_agent_node_knobs():
    # seed 14 — mode / tools / controls / llm_config carried onto the AgentNode.
    desc = _seed_nodes("14-agent-tools.yaml")["reviewer"]
    node, _ = build_leaf_node(desc, {})
    assert isinstance(node, AgentNode)
    assert node.mode == "tool_calling"
    assert node.tools == ["get_web_data", "web_search"]
    assert node.controls == ["ask_user"]
    # llm_config is a plain dict end-to-end.
    assert isinstance(node.llm_config, dict)
    assert node.llm_config["provider"] == "anthropic"
    assert node.llm_config["model"] == "claude-opus-4-8"
    assert node.llm_config["temperature"] == 0.3
    assert node.prompt.startswith("Research ${topic}")
    assert node.output_type.fields["confidence"].kind == ValueKind.NUMBER


def test_node_name_becomes_title():
    # node_name -> the Node title; absent -> None. A constructed descriptor pins
    # the node_name -> title mapping (seed leaf nodes carry no node_name).
    note, _ = build_leaf_node(_seed_nodes("00-hello-agent.yaml")["note"], {})
    assert note.title is None
    named, _ = build_leaf_node(
        parse_nodes(
            NodeReader.spanless(
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
        {},
    )
    assert named.title == "Format verdict"


# ---------- code ----------


def test_code_node_built():
    # seed 01 verdict — a CODE node (module:function ref).
    desc = _seed_nodes("01-structured-agent.yaml")["verdict"]
    node, wiring = build_leaf_node(desc, {})
    assert isinstance(node, CodeNode)
    assert node.ref == "tests.seeds.fns:one_line_summary"
    assert node.output_type.kind == ValueKind.STRING
    assert {p.name for p in node.params} == {"rating", "rationale"}
    assert wiring == {
        "rating": "${score.output.rating}",
        "rationale": "${score.output.rationale}",
    }


# ---------- model ----------


def test_model_node_built():
    # seed 07 predict — model_id / weights_uri / runtime, object output.
    desc = _seed_nodes("07-model-rating.yaml")["predict"]
    node, wiring = build_leaf_node(desc, {})
    assert isinstance(node, ModelNode)
    assert node.model_id == "topic-ranker-v1"
    assert node.weights_uri == "manifold://calpha/models/topic-ranker-v1.pt"
    assert node.runtime_name == "torchscript"
    assert node.output_type.fields["score"].kind == ValueKind.NUMBER
    assert node.output_type.fields["rank"].kind == ValueKind.INTEGER
    assert {p.name for p in node.params} == {"topic", "features"}
    assert wiring == {"topic": "${input.topic}", "features": "${input.features}"}


def test_model_node_run_raises_serving_not_wired():
    # The model_runtime seam was removed (dead plumbing); the MODEL kind still builds,
    # but running one is loud until ML serving is re-added.
    desc = _seed_nodes("07-model-rating.yaml")["predict"]
    node, _ = build_leaf_node(desc, {})
    with pytest.raises(NotImplementedError, match="not wired"):
        node.run({})


# ---------- tool ----------


def test_tool_node_built_with_args_bindings():
    # seed 08 news — tool_id + untyped `inputs` -> InputBindings.
    desc = _seed_nodes("08-tool-news.yaml")["news"]
    node, wiring = build_leaf_node(desc, {})
    assert isinstance(node, ToolNode)
    assert node.tool_id == "get_facts"
    assert node.output_type is None  # no outputs: declared
    assert {p.name for p in node.params} == {"symbol", "limit"}
    assert wiring == {"symbol": "${input.topic}", "limit": 10}
    # inputs are untyped sink params.
    assert all(p.type_str is None for p in node.params)


# ---------- registry-resolved output type ----------


def test_call_loop_descriptors_rejected_as_non_leaf():
    # call/map/loop share the KINDS table but need a resolver — `build_leaf_node` (the
    # leaf-only, resolver=None path) must reject them up front rather than build with a
    # missing resolver and fail with a confusing message deep in `build_call_node`.
    from agent_composer.compose.parse.file import CallDescriptor, MapDescriptor, LoopDescriptor

    for desc in (
        CallDescriptor(id="c", call="child"),
        MapDescriptor(id="m", call="child", over="${input.xs}"),
        LoopDescriptor(id="p", call="body", times=3),
    ):
        with pytest.raises(LoadError, match="is not a leaf node"):
            build_leaf_node(desc, {})


def test_output_type_resolves_registry_name():
    registry = read_typedefs({"Rating": {"category": "str", "score": "float"}})
    # an agent descriptor whose outputs name a registry record.
    nodes = parse_nodes(
        NodeReader.spanless(
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
    node, _ = build_leaf_node(nodes["synth"], registry)
    assert isinstance(node, AgentNode)
    assert node.output_type.kind == ValueKind.OBJECT
    assert node.output_type.fields["category"].kind == ValueKind.STRING
