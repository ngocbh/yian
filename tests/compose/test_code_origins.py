"""Compose provenance contracts for inline CODE source."""

from __future__ import annotations

import pytest

from agent_composer.compile import rewrite_flow_origins
from agent_composer.compile.manifest import flow_fingerprint, flow_manifest
from agent_composer.compose import load_flow
from agent_composer.diagnostics import OriginRef, SourceRange
from agent_composer.nodes.code import CodeNode


_CODE_ORIGIN = OriginRef("/s:root/s:node/s:worker/s:field/s:code")
_LINE_ORIGINS = (
    OriginRef("/s:root/s:node/s:worker/s:field/s:code/s:line/i:0"),
    OriginRef("/s:root/s:node/s:worker/s:field/s:code/s:line/i:1"),
)

_LITERAL_FLOW = """\
id: code-origins
name: code-origins
nodes:
  worker:
    kind: code
    outputs: int
    code: |-
      value = 1
      return value
output: ${worker.output}
"""


def _worker(text: str):
    """Load ``text`` and return its authored CODE node and resolver."""
    loaded = load_flow(text, source_label="code-origins.yaml")
    return loaded.compiled.nodes["worker"], loaded.origin_resolver


def test_literal_code_carries_whole_scalar_and_exact_constructed_lines():
    node, resolver = _worker(_LITERAL_FLOW)

    assert node.code_origin == _CODE_ORIGIN
    assert node.code_line_origins == _LINE_ORIGINS
    assert resolver is not None
    assert resolver.resolve(node.code_origin).range == SourceRange(7, 11, 10, 1)
    assert tuple(
        resolver.resolve(origin).range for origin in node.code_line_origins
    ) == (
        SourceRange(8, 7, 8, 16),
        SourceRange(9, 7, 9, 19),
    )


@pytest.mark.parametrize(
    "code_source",
    [
        '    code: "return 1"\n',
        "    code: >-\n      return 1\n",
    ],
    ids=["quoted", "folded"],
)
def test_nonliteral_code_carries_only_whole_scalar_fallback(code_source):
    text = (
        "id: code-origins\n"
        "name: code-origins\n"
        "nodes:\n"
        "  worker:\n"
        "    kind: code\n"
        "    outputs: int\n"
        f"{code_source}"
        "output: ${worker.output}\n"
    )

    node, resolver = _worker(text)

    assert node.code_origin == _CODE_ORIGIN
    assert node.code_line_origins == ()
    assert resolver is not None
    assert resolver.resolve(node.code_origin) is not None


def test_source_free_code_accepts_empty_origin_carriers():
    node = CodeNode(
        "worker",
        code="return 1",
        code_origin=None,
        code_line_origins=(),
    )

    assert node.code_origin is None
    assert node.code_line_origins == ()
    assert list(node.iter_local_origins()) == []


def test_code_origins_participate_in_local_traversal_and_rewrite():
    node_origin = OriginRef("node")
    code_origin = OriginRef("code")
    line_origins = (OriginRef("line/0"), OriginRef("line/1"))
    node = CodeNode(
        "worker",
        code="value = 1\nreturn value",
        code_origin=code_origin,
        code_line_origins=line_origins,
    )
    node.origin = node_origin

    assert list(node.iter_local_origins()) == [
        node_origin,
        code_origin,
        *line_origins,
    ]

    node.rewrite_local_origins(
        lambda origin: OriginRef(f"rewritten/{origin.key}")
    )

    assert list(node.iter_local_origins()) == [
        OriginRef("rewritten/node"),
        OriginRef("rewritten/code"),
        OriginRef("rewritten/line/0"),
        OriginRef("rewritten/line/1"),
    ]
    assert node.code_origin == OriginRef("rewritten/code")
    assert node.code_line_origins == (
        OriginRef("rewritten/line/0"),
        OriginRef("rewritten/line/1"),
    )


def test_code_origins_are_noncanonical_manifest_and_fingerprint_metadata():
    loaded = load_flow(_LITERAL_FLOW)
    original = loaded.compiled
    rewritten = rewrite_flow_origins(
        original,
        lambda origin: OriginRef(f"rewritten/{origin.key}"),
    )

    original_node = original.nodes["worker"]
    rewritten_node = rewritten.nodes["worker"]
    assert rewritten_node.code_origin != original_node.code_origin
    assert rewritten_node.code_line_origins != original_node.code_line_origins
    assert flow_manifest(rewritten) == flow_manifest(original)
    assert flow_fingerprint(rewritten) == flow_fingerprint(original)
