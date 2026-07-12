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

from agent_composer.compose import LoadError, load_flow
from agent_composer.run import run_flow
from agent_composer.compose.parser import parse_file


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
    assert res.status == "succeeded", res.error
    assert res.output == "ACME"


def test_compact_desugars_to_one_node_keyed_by_flow_id():
    # The desugar keys the single node by the flow id and re-exports its output.
    f = parse_file(_COMPACT_CODE)
    assert set(f.nodes) == {"echo_one"}
    node = f.nodes["echo_one"]
    assert node["kind"] == "code"
    assert node["inputs"] == {"topic": "${input.topic}"}  # auto-wired by name
    assert f.outputs == "${echo_one.output}"             # single-value re-export


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
    assert res.status == "succeeded", res.error
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
    with pytest.raises(LoadError, match="not allowed inline"):
        parse_file(_COMPACT_NON_LEAF)


_COMPACT_NO_ID = """
name: noid
input:
  topic: str
kind: code
code: tests.engine._compose_codefns:echo
"""


def test_compact_requires_id():
    with pytest.raises(LoadError, match="top-level `id:` is required"):
        parse_file(_COMPACT_NO_ID)
