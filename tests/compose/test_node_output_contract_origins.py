"""Compose provenance contracts for authored node-output declarations."""

from __future__ import annotations

import pytest

from agent_composer.compose import LoadError, load_flow
from agent_composer.compose.nodes.base import BuildContext
from agent_composer.compose.nodes.code import CodeDescriptor
from agent_composer.compose.source import SourceCursor, SourceDocument
from agent_composer.diagnostics import OriginRef, OriginResolver
from agent_composer.nodes.contracts import ContractOrigins, ValueContract
from agent_composer.run import run_flow
from agent_composer.typesys.values import ValueKind


_NESTED_OUTPUT_FLOW = """\
id: nested-output-contract
name: nested-output-contract
nodes:
  emit:
    kind: code
    code: |-
      return {"profile": {"age": "bad", "name": "Ada"}, "ok": True}
    output:
      profile:
        age: int
        name: str
      ok: bool
output: ${emit.output}
"""


def _contract(loaded, node_id: str = "emit") -> ValueContract:
    """Return one loaded node's exact output contract."""
    contract = loaded.compiled.nodes[node_id].output_contract
    assert type(contract) is ValueContract
    return contract


def _assert_site(
    resolver: OriginResolver | None,
    origin: OriginRef | None,
    cursor: SourceCursor,
) -> OriginRef:
    """Assert that one contract origin resolves to an exact value occurrence."""
    assert resolver is not None
    assert isinstance(origin, OriginRef)
    frame = resolver.resolve(origin)
    assert frame is not None
    assert cursor.document is not None
    assert frame.text == cursor.document.text
    assert frame.range == cursor.range
    return origin


def test_full_code_output_compiles_root_and_nested_contract_origins():
    """One traversal retains the declaration root and every record field."""
    document = SourceDocument.parse(_NESTED_OUTPUT_FLOW, label="nested.yaml")
    loaded = load_flow(_NESTED_OUTPUT_FLOW, source_label="nested.yaml")
    output_cursor = document.root.child("nodes").child("emit").child("output")
    origins = _contract(loaded).origins
    entries = dict(origins.entries)

    _assert_site(loaded.origin_resolver, origins.default, output_cursor)
    _assert_site(
        loaded.origin_resolver,
        entries[("profile",)],
        output_cursor.child("profile"),
    )
    _assert_site(
        loaded.origin_resolver,
        entries[("profile", "age")],
        output_cursor.child("profile").child("age"),
    )
    _assert_site(
        loaded.origin_resolver,
        entries[("profile", "name")],
        output_cursor.child("profile").child("name"),
    )
    _assert_site(
        loaded.origin_resolver,
        entries[("ok",)],
        output_cursor.child("ok"),
    )


def test_nested_output_failure_selects_the_exact_contract_field():
    """A structural output failure resolves to its nested type occurrence."""
    document = SourceDocument.parse(_NESTED_OUTPUT_FLOW, label="nested.yaml")
    loaded = load_flow(_NESTED_OUTPUT_FLOW, source_label="nested.yaml")
    age_origin = dict(_contract(loaded).origins.entries)[("profile", "age")]

    result = run_flow(loaded, {})

    assert result.status == "failed"
    assert result.failure is not None
    assert result.failure.code == "contract.output"
    assert result.failure.trace[-1] == age_origin
    diagnostic = loaded.diagnose(result.failure)
    expected = (
        document.root.child("nodes")
        .child("emit")
        .child("output")
        .child("profile")
        .child("age")
    )
    assert diagnostic.frames[-1].range == expected.range


def test_root_output_failure_selects_the_contract_root():
    """A root mismatch resolves to the output declaration rather than its node."""
    text = """\
id: root-output-contract
name: root-output-contract
nodes:
  emit:
    kind: code
    code: |-
      return {"not": "an integer"}
    output: int
output: ${emit.output}
"""
    document = SourceDocument.parse(text, label="root.yaml")
    loaded = load_flow(text, source_label="root.yaml")
    root_origin = _contract(loaded).origins.default

    result = run_flow(loaded, {})

    assert result.status == "failed"
    assert result.failure is not None
    assert result.failure.code == "contract.output"
    assert result.failure.trace[-1] == root_origin
    diagnostic = loaded.diagnose(result.failure)
    expected = document.root.child("nodes").child("emit").child("output")
    assert diagnostic.frames[-1].range == expected.range


def test_list_form_output_keeps_winning_sites_and_record_semantics():
    """List-form records keep first-key order and the last value occurrence."""
    text = """\
id: list-output-contract
name: list-output-contract
nodes:
  emit:
    kind: code
    code: |-
      return {"profile": {"age": 7, "active": True}, "status": False}
    output:
      - profile:
          age: str
      - status: bool
      - profile:
          age: int
          active: bool
output: ${emit.output}
"""
    document = SourceDocument.parse(text, label="list.yaml")
    loaded = load_flow(text, source_label="list.yaml")
    contract = _contract(loaded)
    output_cursor = document.root.child("nodes").child("emit").child("output")
    entries = dict(contract.origins.entries)

    assert contract.type.fields is not None
    assert list(contract.type.fields) == ["profile", "status"]
    profile_type = contract.type.fields["profile"]
    assert profile_type.fields is not None
    assert profile_type.fields["age"].kind is ValueKind.INTEGER
    assert profile_type.fields["active"].kind is ValueKind.BOOLEAN
    winning_profile = output_cursor.child(2).child("profile")
    _assert_site(
        loaded.origin_resolver,
        entries[("profile",)],
        winning_profile,
    )
    _assert_site(
        loaded.origin_resolver,
        entries[("profile", "age")],
        winning_profile.child("age"),
    )
    _assert_site(
        loaded.origin_resolver,
        entries[("profile", "active")],
        winning_profile.child("active"),
    )
    _assert_site(
        loaded.origin_resolver,
        entries[("status",)],
        output_cursor.child(1).child("status"),
    )


def test_code_syntax_error_precedes_an_invalid_output_type():
    """Source-aware output compilation preserves CODE's validation order."""
    text = """\
id: invalid-code-before-output
name: invalid-code-before-output
nodes:
  broken:
    kind: code
    code: |-
      return (
    output: DefinitelyNotAType
output: ${broken.output}
"""
    document = SourceDocument.parse(text, label="precedence.yaml")
    code_cursor = document.root.child("nodes").child("broken").child("code")

    with pytest.raises(LoadError) as caught:
        load_flow(text, source_label="precedence.yaml")

    assert caught.value.diagnostic.message == "node 'broken': '(' was never closed"
    scalar_source = code_cursor.scalar_source
    assert scalar_source is not None
    assert scalar_source.constructed_line_ranges is not None
    assert caught.value.diagnostic.frames[-1].range == (
        scalar_source.constructed_line_ranges[0]
    )


def test_direct_descriptor_output_contract_remains_source_free():
    """A direct Python descriptor compiles its type without inventing origins."""
    descriptor = CodeDescriptor(
        id="emit",
        code='return {"profile": {"age": 7}}',
        outputs={"profile": {"age": "int"}},
    )

    node = descriptor.build(BuildContext(registry={})).node

    assert type(node.output_contract) is ValueContract
    assert node.output_contract.origins == ContractOrigins()
    assert node.output_contract.type.fields is not None
    assert node.output_contract.type.fields["profile"].fields is not None
    assert (
        node.output_contract.type.fields["profile"].fields["age"].kind
        is ValueKind.INTEGER
    )
