"""Compact (single-node) flows: a flow whose top-level body carries a node `kind:`
and no `nodes:` map. The parser desugars it into the canonical one-node flow before
compile (so the IR + engine are unchanged):

- the flow `id:` names the single node;
- the flow `input:` is the node signature, auto-wired by name (`p = ${input.p}`);
- the flow `output:` is the node's output TYPE, re-exported as the flow output;
- every other top-level key is the node body (kind + its logic fields).

Run tests use the `code` kind (Ollama-free); parser tests cover the desugar shape
and the rejections (non-leaf kind, missing id).
"""

import pytest

from yian.compose import LoadError, load_flow
from yian.run import run_flow
from yian.compose.parse.file import (
    flow_nodes_cursor,
    parse_file,
    parse_source_document,
)
from yian.compose.parse.nodes import parse_nodes
from yian.diagnostics import SourceRange


# --- a compact CODE flow runs end-to-end (auto-wire + output re-export) ------ #
_COMPACT_CODE = """
id: echo_one
name: echo_one
input:
  topic: str
output: str
kind: code
code: tests.engine._compose_codefns:echo
"""


def test_compact_code_runs():
    # `topic` (a flow input) auto-binds by name into the single node; echo returns it,
    # and the flow output is the node's output (no explicit `output: ${...}` wiring).
    loaded = load_flow(_COMPACT_CODE)
    res = run_flow(loaded, {"topic": "ACME"})
    assert res.status == "succeeded", res.failure
    assert res.output == "ACME"


def test_compact_desugars_to_one_node_keyed_by_flow_id():
    # The desugar keys the single node by the flow id and re-exports its output.
    f = parse_file(_COMPACT_CODE)
    assert set(f.nodes) == {"echo_one"}
    node = f.nodes["echo_one"]
    assert node["kind"] == "code"
    assert node["inputs"] == {"topic": "${input.topic}"}  # auto-wired by name
    assert f.outputs == "${echo_one.output}"             # single-value re-export


def test_compact_projection_keeps_logic_ranges_without_mutating_root():
    document = parse_source_document(_COMPACT_CODE, "compact.yaml")
    original = document.root.value
    parsed = parse_file(document)

    cursor = flow_nodes_cursor(document, parsed.nodes)
    descriptor = parse_nodes(cursor)["echo_one"]

    assert cursor.document is document
    assert descriptor.range == SourceRange(2, 1, 2, 3)
    assert cursor.child("echo_one").range_of("kind") == SourceRange(
        7, 7, 7, 11
    )
    assert descriptor.field_ranges[("field", "code")] == SourceRange(
        8, 7, 8, 41
    )
    assert ("field", "inputs") not in descriptor.field_ranges
    assert document.root.value == original


# --- a record output type carries through to structured generation ----------- #
_COMPACT_RECORD = """
id: plan
name: plan
input:
  topic: str
output:
  rating: str
  score: float
kind: code
code: tests.engine._compose_codefns:make_plan
"""


def test_compact_record_output_runs():
    loaded = load_flow(_COMPACT_RECORD)
    res = run_flow(loaded, {"topic": "ACME"})
    assert res.status == "succeeded", res.failure
    assert res.output == {"rating": "plan for ACME", "score": 0.9}


# --- rejections -------------------------------------------------------------- #
_COMPACT_NON_LEAF = """
id: router
name: router
input:
  x: str
kind: case
on: ${input.x}
cases:
  - when: a
    then: somewhere
"""


def test_compact_rejects_non_leaf_kind():
    # case/call/map reference other nodes a one-node flow has none of.
    with pytest.raises(LoadError, match="not allowed inline") as exc:
        parse_file(_COMPACT_NON_LEAF)

    assert exc.value.diagnostic.frames[-1].range == SourceRange(6, 7, 6, 11)
    assert exc.value.diagnostic.frames[-1].range.start_line == 6


_COMPACT_MODEL = """
id: rank
name: rank
input:
  topic: str
output: float
kind: model
model_id: x
"""


def test_compact_model_kind_is_removed():
    # A removed kind names its replacement instead of the non-leaf rejection.
    with pytest.raises(LoadError, match="kind 'model' was removed") as exc:
        parse_file(_COMPACT_MODEL)

    assert "not allowed inline" not in str(exc.value)
    assert exc.value.diagnostic.frames[-1].range.start_line == 7


_COMPACT_NO_ID = """
name: noid
input:
  topic: str
kind: code
code: tests.engine._compose_codefns:echo
"""


def test_compact_requires_id():
    document = parse_source_document(_COMPACT_NO_ID, "compact.yaml")

    with pytest.raises(LoadError, match="top-level `id:` is required") as exc:
        parse_file(document)

    assert exc.value.diagnostic.frames[-1].range == document.root.range
    assert (
        exc.value.diagnostic.frames[-1].range.start_line
        == document.root.range.start_line
    )


_COMPACT_BAD_INPUT = """
id: echo_one
name: echo_one
input: [topic]
kind: code
code: tests.engine._compose_codefns:echo
"""


def test_compact_invalid_input_has_value_range():
    with pytest.raises(LoadError, match="must be a param map") as exc:
        parse_file(_COMPACT_BAD_INPUT)

    assert exc.value.diagnostic.frames[-1].range == SourceRange(4, 8, 4, 15)
    assert exc.value.diagnostic.frames[-1].range.start_line == 4


_COMPACT_HYPHEN_ID = """
id: hello-agent
name: hello
input: {topic: str}
output: str
kind: code
code: tests.engine._compose_codefns:echo
"""


def test_compact_non_identifier_id_rejected_at_id_key():
    document = parse_source_document(_COMPACT_HYPHEN_ID, "compact.yaml")
    with pytest.raises(LoadError) as exc:
        parse_file(document)
    message = str(exc.value)
    assert "compact flow id 'hello-agent' also names its node" in message
    assert "use 'hello_agent'" in message
    assert "full `nodes:` form" in message
    assert exc.value.diagnostic.frames[-1].range == document.root.range_of("id")


def test_compact_digit_id_rejected_without_parser_dump():
    with pytest.raises(LoadError, match=r"compact flow id '2x'.*use '_2x'") as exc:
        load_flow(_COMPACT_HYPHEN_ID.replace("id: hello-agent", "id: '2x'"))
    assert "Unexpected token" not in str(exc.value)


def test_full_form_flow_id_may_keep_hyphens():
    flow = load_flow(
        """
id: hello-agent
name: hello
input: {topic: str}
nodes:
  greet:
    kind: code
    code: tests.engine._compose_codefns:echo
    input:
      topic: ${input.topic}
output: ${greet.output}
"""
    )
    assert "greet" in flow.compiled.nodes
