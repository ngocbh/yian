"""Unit tests for parse_nodes — the keyed-map flat-body node descriptors.

Each `nodes:` entry (map key = node id, no `id:` field) carries a `kind:` plus
flat kind-fields; `parse_nodes` reads it into a typed per-kind descriptor and is
loud (a `LoadError` at the node's `.yaml` line) on a field illegal for the kind
or an unknown kind. These are DESCRIPTORS (validated parsed shape) — runtime
`Node`s are built later.
"""

from pathlib import Path

import pytest

from agent_composer.compose import LoadError
from agent_composer.compose.parser import (
    AgentDescriptor,
    CallDescriptor,
    CaseDescriptor,
    CodeDescriptor,
    ModelDescriptor,
    ToolDescriptor,
    parse_file,
)
from agent_composer.compose.helpers.reader import NodeReader
from agent_composer.compose.helpers.spanloader import load_with_spans
from agent_composer.compose.stages.parse import parse_nodes

_SEEDS = Path(__file__).resolve().parents[2] / "tests" / "seeds"


def _nodes(yaml_text: str):
    """Parse a bare `nodes:`-body mapping (the test inlines node maps directly).

    Span-aware: the reader is built over the source text so parse errors carry a
    `SourceRange` (and, derived from it, a legacy `.line`).
    """
    value, spans, root = load_with_spans(yaml_text, return_root=True)
    return parse_nodes(NodeReader(value, spans, root))


# ---------- reserved boundary ids ----------


@pytest.mark.parametrize("reserved", ["__start__", "__end__"])
def test_reserved_boundary_id_rejected(reserved):
    # an author node id equal to the synthesized START_ID/END_ID boundary id is a located
    # LoadError (authors write `input:`/`output:`, never a boundary node).
    with pytest.raises(LoadError, match="reserved for the synthesized START_ID/END_ID boundary"):
        _nodes(
            f"""
{reserved}:
  kind: agent
  prompt: hi
  output: str
"""
        )


# ---------- reserved resolver head literals ----------


@pytest.mark.parametrize("reserved", ["input", "output", "system", "item"])
def test_reserved_singular_head_rejected(reserved):
    """A node id equal to a resolver head literal is a located LoadError.
    Required so `${<node>.output.k}` and `${input.k}` are unambiguous."""
    with pytest.raises(LoadError, match=f"node id {reserved!r} is reserved"):
        _nodes(
            f"""
{reserved}:
  kind: agent
  prompt: hi
  output: str
"""
        )


@pytest.mark.parametrize("reserved", ["inputs", "outputs"])
def test_reserved_plural_head_rejected(reserved):
    """The plural forms are typo-catchers — a node id equal to `inputs`/`outputs`
    is a located LoadError naming the retirement of the plural form, so muscle-memory
    mistakes are loud."""
    with pytest.raises(LoadError, match=f"node id {reserved!r} is reserved.*plural head is retired"):
        _nodes(
            f"""
{reserved}:
  kind: agent
  prompt: hi
  output: str
"""
        )


# ---------- agent / code ----------


def test_agent_minimal_parses():
    node = _nodes(
        """
note:
  kind: agent
  input:
    topic: ${input.topic}
  output: str
  prompt: Write a note on ${topic}.
"""
    )["note"]
    assert isinstance(node, AgentDescriptor)
    assert node.id == "note"
    assert node.prompt == "Write a note on ${topic}."
    assert node.inputs == {"topic": "${input.topic}"}
    assert node.outputs == "str"
    assert node.tools == []
    assert node.controls == []
    assert node.mode == "tool_calling"
    assert node.llm_config == {}


def test_agent_knobs_parse():
    # seed 14 — mode / tools / controls / llm_config + object output
    f = parse_file((_SEEDS / "14-agent-tools.yaml").read_text())
    node = parse_nodes(NodeReader.spanless(f.nodes))["reviewer"]
    assert isinstance(node, AgentDescriptor)
    assert node.mode == "tool_calling"
    assert node.tools == ["get_web_data", "web_search"]
    assert node.controls == ["ask_user"]
    assert node.llm_config == {
        "provider": "anthropic",
        "model": "claude-opus-4-8",
        "temperature": 0.3,
    }
    assert node.outputs == {"claim": "str", "confidence": "float"}


def test_code_parses():
    # seed 01 verdict node
    f = parse_file((_SEEDS / "01-structured-agent.yaml").read_text())
    node = parse_nodes(NodeReader.spanless(f.nodes))["verdict"]
    assert isinstance(node, CodeDescriptor)
    assert node.code == "tests.seeds.fns:one_line_summary"
    assert node.inputs == {
        "rating": "${score.output.rating}",
        "rationale": "${score.output.rationale}",
    }
    assert node.outputs == "str"


# ---------- model / tool ----------


def test_model_parses():
    f = parse_file((_SEEDS / "07-model-rating.yaml").read_text())
    node = parse_nodes(NodeReader.spanless(f.nodes))["predict"]
    assert isinstance(node, ModelDescriptor)
    assert node.model_id == "topic-ranker-v1"
    assert node.weights_uri == "manifold://calpha/models/topic-ranker-v1.pt"
    assert node.runtime == "torchscript"
    assert node.inputs == {
        "topic": "${input.topic}",
        "features": "${input.features}",
    }
    assert node.outputs == {"score": "float", "rank": "int"}


def test_model_requires_model_id():
    with pytest.raises(LoadError) as exc:
        _nodes(
            """
predict:
  kind: model
  output:
    score: float
"""
        )
    assert "model_id" in str(exc.value)


def test_tool_parses():
    f = parse_file((_SEEDS / "08-tool-news.yaml").read_text())
    node = parse_nodes(NodeReader.spanless(f.nodes))["news"]
    assert isinstance(node, ToolDescriptor)
    assert node.tool_id == "get_facts"
    assert node.inputs == {"symbol": "${input.topic}", "limit": 10}


def test_tool_legacy_args_key_is_loud():
    # `args:` was retired in favor of `input:` (the standard sink surface) — a tool
    # authoring the old key is a located LoadError (unknown field), no silent alias.
    with pytest.raises(LoadError, match="args") as exc:
        _nodes(
            """
news:
  kind: tool
  tool_id: get_facts
  args:
    symbol: ${input.topic}
"""
        )
    assert "not allowed" in str(exc.value)
    assert exc.value.line is not None


# ---------- case / call ----------


def test_case_searched_parses():
    # seed 02 gate — searched form (no on:)
    f = parse_file((_SEEDS / "02-case.yaml").read_text())
    node = parse_nodes(NodeReader.spanless(f.nodes))["gate"]
    assert isinstance(node, CaseDescriptor)
    assert node.on is None
    assert node.cases == [{"when": "score.output >= 0.5", "then": "positive"}]
    assert node.else_ == "cautious"


def test_case_on_parses():
    # seed 06 route — on: form
    f = parse_file((_SEEDS / "06-case-on.yaml").read_text())
    node = parse_nodes(NodeReader.spanless(f.nodes))["route"]
    assert isinstance(node, CaseDescriptor)
    assert node.on == "${classify.output}"
    assert node.cases == [
        {"when": "pro", "then": "pro_note"},
        {"when": "con", "then": "con_note"},
        {"when": "mixed", "then": "choppy_note"},
    ]
    assert node.else_ == "choppy_note"


def test_case_carries_no_inputs():
    with pytest.raises(LoadError) as exc:
        _nodes(
            """
gate:
  kind: case
  input:
    x: ${score.output}
  cases:
    - when: "${score.output} >= 0.5"
      then: a
  else: b
"""
        )
    assert "inputs" in str(exc.value)


def test_call_ref_form_parses():
    # seed 04 — a plain `call` (REF) = single application.
    f = parse_file((_SEEDS / "04-call.yaml").read_text())
    node = parse_nodes(NodeReader.spanless(f.nodes))["research"]
    assert isinstance(node, CallDescriptor)
    assert node.kind == "call"                          # the REF discriminator
    assert node.call == "research-one"
    assert node.inputs == {"topic": "${input.topic}"}
    assert node.over is None
    assert node.parallel is False


# ---------- node-level asserts: surface ----------


def test_node_asserts_parse_on_code():
    node = _nodes(
        """
x:
  kind: code
  code: m:f
  input:
    n: ${input.v}
  output: int
  asserts:
    - "${output} >= 0"
"""
    )["x"]
    assert node.asserts == ["${output} >= 0"]


def test_node_asserts_rejected_on_case():
    with pytest.raises(LoadError, match="not allowed"):
        _nodes(
            """
gate:
  kind: case
  cases:
    - when: "${s.output} > 0"
      then: a
  asserts:
    - "${output} >= 0"
"""
        )


def test_node_asserts_must_be_list_of_str():
    with pytest.raises(LoadError):
        _nodes(
            """
x:
  kind: code
  code: m:f
  output: int
  asserts: not-a-list
"""
        )


def test_call_map_form_parses():
    # seed 05 — a `map` (kind: map + over:) = List.map.
    f = parse_file((_SEEDS / "05-call-map.yaml").read_text())
    node = parse_nodes(NodeReader.spanless(f.nodes))["research_each"]
    assert isinstance(node, CallDescriptor)
    assert node.kind == "map"                           # the MAP discriminator
    assert node.call == "research-one"
    assert node.over == "${input.topics}"
    assert node.inputs == {"topic": "${item}", "as_of": '${input.as_of:-"today"}'}
    assert node.parallel is True
    assert node.node_name == "Research each topic"  # optional human label


def test_map_without_over_is_loud():
    # `over:` is the iteration source — a `kind: map` without it is incomplete.
    with pytest.raises(LoadError) as exc:
        _nodes(
            """
n:
  kind: map
  call: child
"""
        )
    assert "over" in str(exc.value)


def test_over_on_kind_call_is_loud():
    # `over:`/`parallel:` are MAP-only — on `kind: call` (REF) they are not allowed (use `map`).
    with pytest.raises(LoadError) as exc:
        _nodes(
            """
n:
  kind: call
  call: child
  over: ${input.xs}
"""
        )
    assert "over" in str(exc.value)
    with pytest.raises(LoadError) as exc:
        _nodes(
            """
n:
  kind: call
  call: child
  parallel: true
"""
        )
    assert "parallel" in str(exc.value)


def test_legacy_ref_kind_is_unknown():
    # the collapse removed `kind: ref` — it is an unknown kind (MAP is back, kind: map valid).
    with pytest.raises(LoadError) as exc:
        _nodes("n:\n  kind: ref\n  call: child\n")
    assert "ref" in str(exc.value)
    assert "unknown kind" in str(exc.value)


# ---------- common fields + loud rules ----------


def test_node_name_depends_on_and_runs_after_captured():
    # depends_on + runs_after are run-ordering edges (feature D); node_name is the label.
    node = _nodes(
        """
fetch:
  kind: code
  node_name: Fetch values
  depends_on: [warm_cache]
  runs_after: [log_start]
  input:
    topic: ${input.topic}
  output:
    values: list[float]
  code: tests.seeds.fns:fetch_facts
"""
    )["fetch"]
    assert isinstance(node, CodeDescriptor)
    assert node.node_name == "Fetch values"
    assert node.depends_on == ["warm_cache"]
    assert node.runs_after == ["log_start"]


def test_depends_on_and_runs_after_default_empty():
    node = _nodes(
        """
note:
  kind: agent
  input:
    topic: ${input.topic}
  output: str
  prompt: hi
"""
    )["note"]
    assert node.depends_on == []
    assert node.runs_after == []
    assert node.node_name is None


def test_field_illegal_for_kind_is_loud():
    # `prompt` is an agent field, illegal on a code node.
    with pytest.raises(LoadError) as exc:
        _nodes(
            """
n:
  kind: code
  code: tests.seeds.fns:f
  prompt: not allowed here
"""
        )
    assert "prompt" in str(exc.value)
    assert exc.value.line is not None  # located at the node's .yaml line


def test_tool_id_illegal_on_agent_is_loud():
    with pytest.raises(LoadError) as exc:
        _nodes(
            """
n:
  kind: agent
  tool_id: get_facts
  output: str
  prompt: hi
"""
        )
    assert "tool_id" in str(exc.value)


def test_unknown_kind_is_loud():
    with pytest.raises(LoadError) as exc:
        _nodes(
            """
n:
  kind: wizard
  output: str
"""
        )
    assert "wizard" in str(exc.value)
    assert exc.value.line is not None


def test_missing_kind_is_loud():
    with pytest.raises(LoadError) as exc:
        _nodes(
            """
n:
  input:
    topic: ${input.topic}
  output: str
"""
        )
    assert "kind" in str(exc.value)


def test_seed_11_anchored_nodes_parse():
    # merged-anchor agent nodes all parse to AgentDescriptors
    f = parse_file((_SEEDS / "11-reuse-anchors.yaml").read_text())
    nodes = parse_nodes(NodeReader.spanless(f.nodes))
    for nid in ("pro", "con", "judge"):
        assert isinstance(nodes[nid], AgentDescriptor)
        assert nodes[nid].outputs == "str"
        assert nodes[nid].llm_config["provider"] == "anthropic"
