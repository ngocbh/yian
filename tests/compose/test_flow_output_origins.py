"""Integration tests for authored Compose flow-output origins."""

from __future__ import annotations

from agent_composer.compile.model import END_ID, CompiledFlow, FlowOutput
from agent_composer.compose import load_flow
from agent_composer.compose.build.pipeline import _assemble
from agent_composer.compose.loader import _CallableScope
from agent_composer.compose.source import SourceCursor, SourceDocument
from agent_composer.diagnostics import OriginRef, OriginResolver
from agent_composer.nodes.contracts import Binding


def _output(flow: CompiledFlow, name: str) -> FlowOutput:
    """Return one declared output by name."""
    matches = [output for output in flow.outputs if output.name == name]
    assert len(matches) == 1
    return matches[0]


def _assert_output_site(
    catalog: OriginResolver,
    flow: CompiledFlow,
    name: str,
    cursor: SourceCursor,
    *,
    source: object,
) -> OriginRef:
    """Assert one flow-output binding resolves its authored value occurrence."""
    output = _output(flow, name)
    assert output.binding.source == source
    assert isinstance(output.binding.origin, OriginRef)
    assert flow.wiring[END_ID][name] is output.binding

    frame = catalog.resolve(output.binding.origin)
    assert frame is not None
    assert frame.text == cursor.document.text
    assert frame.range == cursor.range
    return output.binding.origin


def test_bare_root_output_resolves_its_exact_value_and_reaches_end_by_identity():
    """A bare root output owns its value occurrence on the canonical binding."""
    text = """id: bare-output-origin
name: bare-output-origin
nodes:
  worker:
    kind: code
    code: return 1
    output: int
output: ${worker.output}
"""
    document = SourceDocument.parse(text)
    loaded = load_flow(text)

    _assert_output_site(
        loaded.origin_resolver,
        loaded.compiled,
        "result",
        document.root.child("output"),
        source="${worker.output}",
    )


def test_mapped_root_named_def_and_inline_outputs_keep_distinct_value_sites():
    """Output values stay distinct across mapping entries and lexical flows."""
    text = """id: lexical-output-origins
name: lexical-output-origins
defs:
  named:
    nodes:
      worker:
        kind: code
        code: return 2
        output: int
    output: ${worker.output}
nodes:
  worker:
    kind: code
    code: return 1
    output: int
  named_call:
    kind: call
    call: named
  inline_call:
    kind: call
    nodes:
      worker:
        kind: code
        code: return 3
        output: int
    output: ${worker.output}
output:
  root: ${worker.output}
  named: ${named_call.output}
  inline: ${inline_call.output}
"""
    document = SourceDocument.parse(text)
    loaded = load_flow(text)
    named = loaded.compiled.nodes["named_call"].child
    inline = loaded.compiled.nodes["inline_call"].child
    catalog = loaded.origin_resolver
    assert catalog is not None

    origins = {
        _assert_output_site(
            catalog,
            loaded.compiled,
            "root",
            document.root.child("output").child("root"),
            source="${worker.output}",
        ),
        _assert_output_site(
            catalog,
            loaded.compiled,
            "named",
            document.root.child("output").child("named"),
            source="${named_call.output}",
        ),
        _assert_output_site(
            catalog,
            loaded.compiled,
            "inline",
            document.root.child("output").child("inline"),
            source="${inline_call.output}",
        ),
        _assert_output_site(
            catalog,
            named,
            "result",
            document.root.child("defs").child("named").child("output"),
            source="${worker.output}",
        ),
        _assert_output_site(
            catalog,
            inline,
            "result",
            document.root.child("nodes").child("inline_call").child("output"),
            source="${worker.output}",
        ),
    }

    assert len(origins) == 5


def test_mapped_output_uses_the_effective_duplicate_value_occurrence():
    """A shadowed mapping pair cannot steal the final binding's origin."""
    text = """id: duplicate-output-origin
name: duplicate-output-origin
nodes:
  worker:
    kind: code
    code: return 1
    output: int
output:
  result: ignored
  result: ${worker.output}
"""
    document = SourceDocument.parse(text)
    loaded = load_flow(text)
    output_cursor = document.root.child("output")
    occurrences = output_cursor.mapping_occurrences("result")

    assert len(occurrences) == 2
    assert occurrences[0].range != occurrences[1].range
    _assert_output_site(
        loaded.origin_resolver,
        loaded.compiled,
        "result",
        occurrences[1],
        source="${worker.output}",
    )


def test_output_alias_and_merge_sites_keep_distinct_occurrences():
    """Shared definitions retain occurrence-local output identities."""
    text = """id: aliased-output-origins
name: aliased-output-origins
x-shared: &shared
  merged: &value ${worker.output}
nodes:
  worker:
    kind: code
    code: return 1
    output: int
output:
  <<: *shared
  alias: *value
"""
    document = SourceDocument.parse(text)
    loaded = load_flow(text)
    output_cursor = document.root.child("output")
    merged_cursor = output_cursor.child("merged")
    alias_cursor = output_cursor.child("alias")

    merged_origin = _assert_output_site(
        loaded.origin_resolver,
        loaded.compiled,
        "merged",
        merged_cursor,
        source="${worker.output}",
    )
    alias_origin = _assert_output_site(
        loaded.origin_resolver,
        loaded.compiled,
        "alias",
        alias_cursor,
        source="${worker.output}",
    )

    assert merged_cursor.range == alias_cursor.range
    assert merged_cursor.path != alias_cursor.path
    assert merged_origin != alias_origin


def test_output_rewrites_keep_authored_sites_with_final_binding_sources():
    """Desugaring changes output sources without moving their authored sites."""
    text = """id: rewritten-output-origins
name: rewritten-output-origins
defs:
  helper:
    nodes:
      value:
        kind: code
        code: return 7
        output: int
    output: ${value.output}
nodes:
  score:
    kind: code
    code: return 1
    output: int
  gate:
    kind: case
    cases:
      - when: ${score.output} > 0
        then: positive
    else: negative
  positive:
    kind: code
    code: return 1
    output: int
  negative:
    kind: code
    code: return 0
    output: int
output:
  via_call: call(helper)
  via_case: ${gate.output}
"""
    document = SourceDocument.parse(text)
    loaded = load_flow(text)
    output_cursor = document.root.child("output")

    call_origin = _assert_output_site(
        loaded.origin_resolver,
        loaded.compiled,
        "via_call",
        output_cursor.child("via_call"),
        source="${__call_0.output}",
    )
    case_origin = _assert_output_site(
        loaded.origin_resolver,
        loaded.compiled,
        "via_case",
        output_cursor.child("via_case"),
        source="${positive.output | negative.output}",
    )
    assert call_origin != case_origin


def test_compact_root_generated_reexport_is_originless():
    """A compact root's generated reexport is not the authored type site."""
    loaded = load_flow(
        """id: compact_root
name: compact_root
kind: code
code: return 1
output: int
"""
    )

    output = _output(loaded.compiled, "result")
    assert output.binding.source == "${compact_root.output}"
    assert output.binding.origin is None
    assert loaded.compiled.wiring[END_ID]["result"] is output.binding


def test_single_node_def_generated_reexport_is_originless():
    """A compact def's generated reexport is not the authored type site."""
    loaded = load_flow(
        """id: compact-def-output-origin
name: compact-def-output-origin
defs:
  compact:
    kind: code
    code: return 1
    output: int
nodes:
  invoke:
    kind: call
    call: compact
output: ${invoke.output}
"""
    )
    child = loaded.compiled.nodes["invoke"].child

    output = _output(child, "result")
    assert output.binding.source == "${compact.output}"
    assert output.binding.origin is None
    assert child.wiring[END_ID]["result"] is output.binding


def test_virtual_outputs_do_not_mint_or_erase_programmatic_origins():
    """Source-free assembly preserves absent and explicit binding origins."""
    explicit = OriginRef("programmatic-flow-output")
    outputs = {
        "virtual": "${worker.output}",
        "programmatic": Binding("${worker.output}", explicit),
    }
    loaded = _assemble(
        inputs_section={},
        nodes_cursor=SourceCursor.virtual(
            {
                "worker": {
                    "kind": "code",
                    "code": "return 1",
                    "outputs": "int",
                }
            }
        ),
        outputs_section=outputs,
        outputs_cursor=SourceCursor.virtual(
            {
                "virtual": "${worker.output}",
                "programmatic": "${worker.output}",
            }
        ),
        asserts_section=[],
        registry={},
        scope=_CallableScope({}, {}, None, {}),
        owner=("root",),
    )

    virtual = _output(loaded.compiled, "virtual")
    programmatic = _output(loaded.compiled, "programmatic")
    assert virtual.binding.origin is None
    assert programmatic.binding.origin == explicit
    assert loaded.compiled.wiring[END_ID]["virtual"] is virtual.binding
    assert loaded.compiled.wiring[END_ID]["programmatic"] is programmatic.binding


def test_physical_output_does_not_replace_an_explicit_programmatic_origin():
    """An existing carrier origin outranks an optional Compose source cursor."""
    explicit = OriginRef("programmatic-flow-output")
    document = SourceDocument.parse(
        "output:\n  result: ${worker.output}\n"
    )
    loaded = _assemble(
        inputs_section={},
        nodes_cursor=SourceCursor.virtual(
            {
                "worker": {
                    "kind": "code",
                    "code": "return 1",
                    "outputs": "int",
                }
            }
        ),
        outputs_section={
            "result": Binding("${worker.output}", explicit),
        },
        outputs_cursor=document.root.child("output"),
        asserts_section=[],
        registry={},
        scope=_CallableScope({}, {}, None, {}),
        owner=("root",),
    )

    output = _output(loaded.compiled, "result")
    assert output.binding.origin == explicit
    assert loaded.origin_resolver is None
    assert loaded.compiled.wiring[END_ID]["result"] is output.binding
