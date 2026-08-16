"""Integration tests for authored Compose assertion origins."""

from __future__ import annotations

from agent_composer.compile.expand import call_subgraph
from agent_composer.compile.model import END_ID, START_ID
from agent_composer.compose import load_flow
from agent_composer.compose.build.pipeline import _assemble
from agent_composer.compose.loader import _CallableScope
from agent_composer.compose.source import SourceCatalog, SourceCursor, SourceDocument
from agent_composer.diagnostics import OriginRef
from agent_composer.nodes.contracts import Assertion


def _assert_assertion_site(
    catalog: SourceCatalog,
    assertion: Assertion,
    cursor: SourceCursor,
    *,
    expression: str | None = None,
    display: str | None = None,
) -> OriginRef:
    """Assert one semantic assertion resolves its exact authored list item."""
    assert type(assertion) is Assertion
    if expression is not None:
        assert assertion.expression == expression
    if display is not None:
        assert assertion.display == display
    assert isinstance(assertion.origin, OriginRef)

    frame = catalog.resolve(assertion.origin)
    assert frame is not None
    assert frame.text == cursor.document.text
    assert frame.range == cursor.range
    return assertion.origin


def test_root_flow_assertions_keep_indexed_sites_through_rewrites():
    """Flow lowering keeps duplicate, pre/post, and inline-call occurrences exact."""
    text = """id: root-assertion-origins
name: root-assertion-origins
input:
  n: int
defs:
  truthy:
    kind: code
    code: return True
    output: bool
nodes:
  worker:
    kind: code
    code: return 1
    output: int
output: ${worker.output}
asserts:
  - ${input.n} >= 0
  - ${input.n} >= 0
  - call(truthy)
  - ${worker.output} >= 0
"""
    document = SourceDocument.parse(text)
    loaded = load_flow(text)
    catalog = loaded._local_catalog
    assertion_cursors = document.root.child("asserts")
    start = loaded.compiled.nodes[START_ID]
    end = loaded.compiled.nodes[END_ID]

    assert [assertion.display for assertion in start.post_asserts] == [
        "${input.n} >= 0",
        "${input.n} >= 0",
    ]
    assert [assertion.expression for assertion in start.post_asserts] == [
        "${output.n} >= 0",
        "${output.n} >= 0",
    ]
    assert [assertion.display for assertion in end.post_asserts] == [
        "call(truthy)",
        "${worker.output} >= 0",
    ]
    assert end.post_asserts[0].expression == "${__assert___call_0}"

    ordered = [
        (start.post_asserts[0], assertion_cursors.child(0)),
        (start.post_asserts[1], assertion_cursors.child(1)),
        (end.post_asserts[0], assertion_cursors.child(2)),
        (end.post_asserts[1], assertion_cursors.child(3)),
    ]
    origins = {
        _assert_assertion_site(catalog, assertion, cursor)
        for assertion, cursor in ordered
    }

    assert len(origins) == 4
    assert start.post_assert_ranges == [
        assertion_cursors.child(0).range,
        assertion_cursors.child(1).range,
    ]
    assert end.post_assert_ranges == [
        assertion_cursors.child(2).range,
        assertion_cursors.child(3).range,
    ]


def test_alias_occurrences_mint_distinct_node_and_flow_assertion_origins():
    """Alias uses remain distinct even when they share one definition range."""
    text = """id: aliased-assertion-origins
name: aliased-assertion-origins
input:
  n: int
nodes:
  first: &shared_node
    kind: code
    code: return 1
    output: int
    asserts:
      - ${output} >= 0
  second: *shared_node
output:
  first: ${first.output}
  second: ${second.output}
asserts:
  - &shared_rule ${input.n} >= 0
  - *shared_rule
"""
    document = SourceDocument.parse(text)
    loaded = load_flow(text)
    catalog = loaded._local_catalog

    first_cursor = (
        document.root.child("nodes")
        .child("first")
        .child("asserts")
        .child(0)
    )
    second_cursor = (
        document.root.child("nodes")
        .child("second")
        .child("asserts")
        .child(0)
    )
    node_origins = {
        _assert_assertion_site(
            catalog,
            loaded.compiled.nodes[node_id].post_asserts[0],
            cursor,
        )
        for node_id, cursor in (
            ("first", first_cursor),
            ("second", second_cursor),
        )
    }
    assert first_cursor.range == second_cursor.range
    assert first_cursor.path != second_cursor.path
    assert len(node_origins) == 2

    flow_cursors = document.root.child("asserts")
    start_assertions = loaded.compiled.nodes[START_ID].post_asserts
    flow_origins = {
        _assert_assertion_site(
            catalog,
            assertion,
            flow_cursors.child(index),
        )
        for index, assertion in enumerate(start_assertions)
    }
    assert flow_cursors.child(0).range == flow_cursors.child(1).range
    assert flow_cursors.child(0).path != flow_cursors.child(1).path
    assert len(flow_origins) == 2


def test_same_node_assertion_text_has_distinct_root_def_inline_and_flat_sites():
    """Lexically equal node assertions retain four physical identities."""
    assertion_text = "${output} >= 0"
    text = f"""id: lexical-node-assertions
name: lexical-node-assertions
defs:
  named:
    nodes:
      worker:
        kind: code
        code: return 2
        output: int
        asserts:
          - {assertion_text}
    output: ${{worker.output}}
  flat:
    kind: code
    code: return 3
    output: int
    asserts:
      - {assertion_text}
nodes:
  worker:
    kind: code
    code: return 1
    output: int
    asserts:
      - {assertion_text}
  named_call:
    kind: call
    call: named
  flat_call:
    kind: call
    call: flat
  inline_call:
    kind: call
    nodes:
      worker:
        kind: code
        code: return 4
        output: int
        asserts:
          - {assertion_text}
    output: ${{worker.output}}
output:
  root: ${{worker.output}}
  named: ${{named_call.output}}
  flat: ${{flat_call.output}}
  inline: ${{inline_call.output}}
"""
    document = SourceDocument.parse(text)
    loaded = load_flow(text)
    catalog = loaded._local_catalog

    sites = (
        (
            loaded.compiled.nodes["worker"].post_asserts[0],
            document.root.child("nodes")
            .child("worker")
            .child("asserts")
            .child(0),
        ),
        (
            loaded.compiled.nodes["named_call"]
            .child.nodes["worker"]
            .post_asserts[0],
            document.root.child("defs")
            .child("named")
            .child("nodes")
            .child("worker")
            .child("asserts")
            .child(0),
        ),
        (
            loaded.compiled.nodes["inline_call"]
            .child.nodes["worker"]
            .post_asserts[0],
            document.root.child("nodes")
            .child("inline_call")
            .child("nodes")
            .child("worker")
            .child("asserts")
            .child(0),
        ),
        (
            loaded.compiled.nodes["flat_call"]
            .child.nodes["flat"]
            .post_asserts[0],
            document.root.child("defs")
            .child("flat")
            .child("asserts")
            .child(0),
        ),
    )

    origins = {
        _assert_assertion_site(
            catalog,
            assertion,
            cursor,
            expression=assertion_text,
            display=assertion_text,
        )
        for assertion, cursor in sites
    }
    assert len(origins) == 4


def test_compact_root_node_assertion_uses_its_projected_physical_item():
    """Compact projection gives its synthesized node the top-level assert site."""
    text = """id: compact_assertion_origin
name: compact_assertion_origin
kind: code
code: return 1
output: int
asserts:
  - ${output} >= 0
"""
    document = SourceDocument.parse(text)
    loaded = load_flow(text)
    node = loaded.compiled.nodes["compact_assertion_origin"]
    cursor = document.root.child("asserts").child(0)

    _assert_assertion_site(
        loaded._local_catalog,
        node.post_asserts[0],
        cursor,
        expression="${output} >= 0",
        display="${output} >= 0",
    )
    assert node.post_assert_ranges == [cursor.range]


def test_duplicate_loop_assertions_keep_sites_when_the_driver_respawns():
    """One node's duplicate authored carriers survive LOOP reconstruction."""
    text = """id: loop-assertion-origins
name: loop-assertion-origins
nodes:
  drive:
    kind: loop
    times: 1
    input:
      n: 0
      exited: false
    asserts:
      - ${output.n} >= 0
      - ${output.n} >= 0
    nodes:
      bump:
        kind: code
        code: tests.engine._compose_codefns:loop_bump
        input:
          n: ${input.n}
          exited: ${input.exited}
        output:
          n: int
          exited: bool
    output: ${bump.output}
output: ${drive.output}
"""
    document = SourceDocument.parse(text)
    loaded = load_flow(text)
    catalog = loaded._local_catalog
    drive = loaded.compiled.nodes["drive"]
    cursors = (
        document.root.child("nodes").child("drive").child("asserts")
    )

    origins = {
        _assert_assertion_site(catalog, assertion, cursors.child(index))
        for index, assertion in enumerate(drive.post_asserts)
    }
    assert len(origins) == 2
    assert drive.post_assert_ranges == [
        cursors.child(0).range,
        cursors.child(1).range,
    ]

    respawned = drive.respawn(1)

    assert respawned.post_asserts == drive.post_asserts
    assert respawned.post_asserts is not drive.post_asserts
    assert [assertion.origin for assertion in respawned.post_asserts] == [
        assertion.origin for assertion in drive.post_asserts
    ]
    assert respawned.post_assert_ranges == drive.post_assert_ranges
    assert respawned.post_assert_ranges is not drive.post_assert_ranges


def test_def_boundary_and_call_relocation_reuse_authored_assertion_carriers():
    """Def START/END and a relocated CALL post-assert keep their source sites."""
    text = """id: nested-assertion-origins
name: nested-assertion-origins
defs:
  child:
    input:
      n: int
    nodes:
      worker:
        kind: code
        code: tests.engine._compose_codefns:double
        input:
          n: ${input.n}
        output: int
    output: ${worker.output}
    asserts:
      - ${input.n} >= 0
      - ${worker.output} >= 0
nodes:
  invoke:
    kind: call
    call: child
    input:
      n: 1
    asserts:
      - ${n} >= 0 and ${output} >= 0
output: ${invoke.output}
"""
    document = SourceDocument.parse(text)
    loaded = load_flow(text)
    catalog = loaded._local_catalog
    invoke = loaded.compiled.nodes["invoke"]
    child = invoke.child
    child_asserts = (
        document.root.child("defs").child("child").child("asserts")
    )
    call_cursor = (
        document.root.child("nodes")
        .child("invoke")
        .child("asserts")
        .child(0)
    )

    child_start = child.nodes[START_ID]
    child_end = child.nodes[END_ID]
    start_assertion = child_start.post_asserts[0]
    end_assertion = child_end.post_asserts[0]
    call_assertion = invoke.post_asserts[0]

    _assert_assertion_site(catalog, start_assertion, child_asserts.child(0))
    _assert_assertion_site(catalog, end_assertion, child_asserts.child(1))
    call_origin = _assert_assertion_site(catalog, call_assertion, call_cursor)
    assert child_start.post_assert_ranges == [child_asserts.child(0).range]
    assert child_end.post_assert_ranges == [child_asserts.child(1).range]
    assert invoke.post_assert_ranges == [call_cursor.range]

    cloned = call_subgraph(
        child,
        callsite="invoke",
        record={"n": 1},
        post_asserts=invoke.post_asserts,
        post_assert_ranges=invoke.post_assert_ranges,
        input_names={parameter.name for parameter in invoke.params},
    )
    relocated_end = cloned.nodes[cloned.end_id]
    relocated = relocated_end.post_asserts[-1]

    assert relocated is not call_assertion
    assert relocated.origin == call_origin
    assert relocated.display == call_assertion.display
    assert "${__assert_input_n}" in relocated.expression
    assert "${output}" in relocated.expression
    assert relocated_end.post_assert_ranges[-1] == call_cursor.range
    assert catalog.resolve(relocated.origin) == catalog.resolve(call_origin)


def test_virtual_assertions_do_not_mint_or_erase_programmatic_origins():
    """Source-free assembly leaves raw assertions empty and explicit origins intact."""
    raw = "${input.n} >= 0"
    explicit = OriginRef("programmatic-assertion")
    programmatic = Assertion(
        "${worker.output} >= 0",
        "programmatic worker assertion",
        explicit,
    )
    loaded = _assemble(
        inputs_section={"n": "int"},
        nodes_cursor=SourceCursor.virtual(
            {
                "worker": {
                    "kind": "code",
                    "code": "return 1",
                    "outputs": "int",
                }
            }
        ),
        outputs_section="${worker.output}",
        asserts_section=[raw, programmatic],
        asserts_cursor=SourceCursor.virtual([raw, programmatic.expression]),
        registry={},
        scope=_CallableScope({}, {}, None, {}),
        owner=("root",),
        n_lines={},
        s_lines={},
    )

    start = loaded.compiled.nodes[START_ID]
    end = loaded.compiled.nodes[END_ID]
    assert start.post_asserts[0].display == raw
    assert start.post_asserts[0].origin is None
    assert end.post_asserts[0].display == programmatic.display
    assert end.post_asserts[0].origin is explicit
    assert loaded._local_catalog.resolve(explicit) is None
