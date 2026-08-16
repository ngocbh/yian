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
    return _node_contract(loaded.compiled.nodes[node_id])


def _node_contract(node) -> ValueContract:
    """Return one exact node output contract."""
    contract = node.output_contract
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


def _assert_profile_contract_sites(
    resolver: OriginResolver | None,
    contract: ValueContract,
    output_cursor: SourceCursor,
) -> None:
    """Assert the declaration root and nested profile fields retain their sites."""
    entries = dict(contract.origins.entries)
    _assert_site(resolver, contract.origins.default, output_cursor)
    _assert_site(
        resolver,
        entries[("profile",)],
        output_cursor.child("profile"),
    )
    _assert_site(
        resolver,
        entries[("profile", "age")],
        output_cursor.child("profile").child("age"),
    )


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


def test_compact_root_output_contract_uses_the_top_level_output_site():
    """Compact syntax projects its authored output back to the physical scalar tree."""
    text = """\
id: compact_output_contract
name: compact_output_contract
kind: code
code: |-
  return {"profile": {"age": 7}}
output:
  profile:
    age: int
"""
    document = SourceDocument.parse(text, label="compact.yaml")
    loaded = load_flow(text, source_label="compact.yaml")

    _assert_profile_contract_sites(
        loaded.origin_resolver,
        _contract(loaded, "compact_output_contract"),
        document.root.child("output"),
    )


def test_multi_node_def_output_contract_uses_the_nested_node_site():
    """A full definition keeps its leaf node's physical output declaration."""
    text = """\
id: full-def-output-contract
name: full-def-output-contract
defs:
  child:
    nodes:
      emit:
        kind: code
        code: |-
          return {"profile": {"age": 7}}
        output:
          profile:
            age: int
    output: ${emit.output}
nodes:
  invoke:
    kind: call
    call: child
output: ${invoke.output}
"""
    document = SourceDocument.parse(text, label="full-def.yaml")
    loaded = load_flow(text, source_label="full-def.yaml")
    emit = loaded.compiled.nodes["invoke"].child.nodes["emit"]
    output_cursor = (
        document.root.child("defs")
        .child("child")
        .child("nodes")
        .child("emit")
        .child("output")
    )

    _assert_profile_contract_sites(
        loaded.origin_resolver,
        _node_contract(emit),
        output_cursor,
    )


def test_flat_def_output_contract_uses_the_projected_def_output_site():
    """A compact definition projects its leaf contract to the def's output field."""
    text = """\
id: flat-def-output-contract
name: flat-def-output-contract
defs:
  child:
    kind: code
    code: |-
      return {"profile": {"age": 7}}
    output:
      profile:
        age: int
nodes:
  invoke:
    kind: call
    call: child
output: ${invoke.output}
"""
    document = SourceDocument.parse(text, label="flat-def.yaml")
    loaded = load_flow(text, source_label="flat-def.yaml")
    child = loaded.compiled.nodes["invoke"].child.nodes["child"]
    output_cursor = document.root.child("defs").child("child").child("output")

    _assert_profile_contract_sites(
        loaded.origin_resolver,
        _node_contract(child),
        output_cursor,
    )


def test_nested_inline_child_output_contract_uses_its_physical_site():
    """Inline extraction retains the nested child's output declaration."""
    text = """\
id: inline-output-contract
name: inline-output-contract
nodes:
  invoke:
    kind: call
    nodes:
      emit:
        kind: code
        code: |-
          return {"profile": {"age": 7}}
        output:
          profile:
            age: int
    output: ${emit.output}
output: ${invoke.output}
"""
    document = SourceDocument.parse(text, label="inline.yaml")
    loaded = load_flow(text, source_label="inline.yaml")
    emit = loaded.compiled.nodes["invoke"].child.nodes["emit"]
    output_cursor = (
        document.root.child("nodes")
        .child("invoke")
        .child("nodes")
        .child("emit")
        .child("output")
    )

    _assert_profile_contract_sites(
        loaded.origin_resolver,
        _node_contract(emit),
        output_cursor,
    )


def test_external_alias_rebases_every_output_contract_origin():
    """An alias view owns distinct output keys resolving to canonical child sites."""
    child_text = """\
id: external-output-child
name: external-output-child
nodes:
  emit:
    kind: code
    code: |-
      return {"profile": {"age": 7}}
    output:
      profile:
        age: int
output: ${emit.output}
"""
    child_document = SourceDocument.parse(child_text, label="child.yaml")
    child = load_flow(child_text, source_label="child.yaml")
    parent_text = """\
id: external-output-parent
name: external-output-parent
uses:
  imported: child
nodes:
  invoke:
    kind: call
    call: imported
output: ${invoke.output}
"""
    parent = load_flow(
        parent_text,
        source_label="parent.yaml",
        child_resolver=lambda _path, _version=None: child,
    )
    canonical = _contract(child)
    invoke = parent.compiled.nodes["invoke"]
    rebased = _node_contract(invoke.child.nodes["emit"])
    canonical_origins = (
        canonical.origins.default,
        *(origin for _path, origin in canonical.origins.entries),
    )
    rebased_origins = (
        rebased.origins.default,
        *(origin for _path, origin in rebased.origins.entries),
    )

    assert _node_contract(invoke).origins == rebased.origins
    assert child.origin_resolver is not None
    assert parent.origin_resolver is not None
    for canonical_origin, rebased_origin in zip(
        canonical_origins,
        rebased_origins,
        strict=True,
    ):
        assert isinstance(canonical_origin, OriginRef)
        assert isinstance(rebased_origin, OriginRef)
        assert rebased_origin != canonical_origin
        assert parent.origin_resolver.resolve(rebased_origin) == (
            child.origin_resolver.resolve(canonical_origin)
        )
    _assert_profile_contract_sites(
        parent.origin_resolver,
        rebased,
        child_document.root.child("nodes").child("emit").child("output"),
    )


def test_compact_invalid_output_type_reports_the_top_level_output_site():
    """Compact output parsing reports its physical top-level type occurrence."""
    text = """\
id: invalid_compact_output
name: invalid_compact_output
kind: code
code: return 7
output: list[
"""
    document = SourceDocument.parse(text, label="invalid-compact.yaml")

    with pytest.raises(LoadError, match="malformed type expression") as caught:
        load_flow(text, source_label="invalid-compact.yaml")

    frame = caught.value.diagnostic.frames[-1]
    assert frame.text == document.text
    assert frame.range == document.root.child("output").range


def test_adaptive_generated_output_contracts_remain_source_free():
    """Desugared contracts without authorable type declarations invent no origins."""
    text = """\
id: adaptive-output-contracts
name: adaptive-output-contracts
input:
  topic: str
nodes:
  ask:
    kind: human_input
    input:
      topic: ${input.topic}
    adaptive_questions:
      prompt: Ask about ${topic}
output: ${ask.output}
"""
    nodes = load_flow(text).compiled.nodes

    assert _node_contract(nodes["ask__compose"]).origins == ContractOrigins()
    assert _node_contract(nodes["ask"]).origins == ContractOrigins()
