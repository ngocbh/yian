"""In-file `defs:` callables + defs-first `call` resolution — Ollama-free.

A top-level `defs:` section holds in-file callables (multi-node sub-flows). A `call:`
resolves **defs-first** (an in-file def), else the injected external `child_resolver`. A
flow whose calls are all in-file defs loads + runs with NO external resolver. These drive
real CODE-only children through `run_flow` (no LLM). Pinned here:

- a multi-node def runs end-to-end (a plain `call` and a mapped `call` over a def);
- a def calls another def (nested, resolver-free);
- a def SHADOWS an external flow of the same name (defs-first);
- recursive / mutual defs are loud (-> LoadError, not a load hang);
- a single-node def runs (auto-wire params by name) + its call is cross-flow type-checked;
- a `call` to a name that is neither a def nor (no external resolver) is loud.
"""

import pytest

from agent_composer.compose import loader as compose_loader
from agent_composer.compose import LoadError, load_flow
from agent_composer.compose.source import SourceDocument
from agent_composer.compose.source import document as source_document
from agent_composer.diagnostics import SourceRange
from agent_composer.nodes.contracts import Parameter, ValueContract
from agent_composer.run import run_flow
from agent_composer.compose.parse.file import parse_file
from agent_composer.typesys import Type, ValueKind

# --------------------------------------------------------------------------- #
# A multi-node def, called as a plain application (no over:).
# research_pair = data (make_report -> {report, n}) -> summary (echo the report).
# --------------------------------------------------------------------------- #

_DEFS_PARENT = """
id: defs-parent
name: defs_parent
input:
  topic: str
defs:
  research_pair:
    input:
      topic: str
    nodes:
      data:
        kind: code
        code: tests.engine._compose_codefns:make_report
        input:
          topic: ${input.topic}
        output:
          report: str
          n: int
      summary:
        kind: code
        code: tests.engine._compose_codefns:echo
        input:
          topic: ${data.output.report}
        output: str
    output: ${summary.output}
nodes:
  research:
    kind: call
    call: research_pair
    input:
      topic: ${input.topic}
output: ${research.output}
"""


def test_defs_section_parses():
    f = parse_file(_DEFS_PARENT)
    assert "research_pair" in f.defs
    assert "nodes" in f.defs["research_pair"]


def test_callable_scope_eagerly_builds_defs_and_memoizes_resolution(monkeypatch):
    """One scope eagerly builds local defs and resolves each external alias once."""
    events = []
    local_child = object()
    external_child = load_flow("id: external\nname: external\nnodes: {}\n")
    registry = object()
    body = {"nodes": {}}
    seen_scopes = []

    source_document = SourceDocument.parse("defs:\n  local:\n    nodes: {}\n")
    defs_cursor = source_document.root.child("defs")

    def fake_load_def(
        name,
        value,
        seen_registry,
        scope,
        seen_def_cursor=None,
    ):
        events.append(("def", name))
        seen_scopes.append(scope)
        assert value is body
        assert seen_registry is registry
        assert seen_def_cursor.document is source_document
        assert seen_def_cursor.path == defs_cursor.child("local").path
        return local_child

    def external(path, version=None):
        events.append(("external", path, version))
        return external_child

    monkeypatch.setattr(compose_loader, "_load_def", fake_load_def)
    scope = compose_loader._CallableScope(
        {"local": body},
        registry,
        external,
        {"remote": "library/child@declared-v1"},
        defs_cursor,
    )

    scope.eager_load_defs()
    assert events == [("def", "local")]
    assert seen_scopes == [scope]
    assert scope.resolve("local") is local_child
    assert scope.resolve("local") is local_child

    first = scope.resolve("remote", "caller-version-is-ignored")
    second = scope.resolve("remote", "another-caller-version")
    assert events == [
        ("def", "local"),
        ("external", "library/child", "declared-v1"),
    ]
    assert first is not external_child
    assert first.compiled is not external_child.compiled
    assert second is first


def test_multiple_defs_share_one_parent_document_parse(monkeypatch):
    source_loads = 0
    original_source_init = source_document._DocumentLoader.__init__

    def count_source(self, *args, **kwargs):
        nonlocal source_loads
        source_loads += 1
        original_source_init(self, *args, **kwargs)

    monkeypatch.setattr(source_document._DocumentLoader, "__init__", count_source)

    load_flow(
        """id: two-defs
name: two_defs
defs:
  first:
    nodes:
      one:
        kind: code
        code: tests.engine._compose_codefns:echo
        input: {value: first}
        output: str
    output: ${one.output}
  second:
    nodes:
      two:
        kind: code
        code: tests.engine._compose_codefns:echo
        input: {value: second}
        output: str
    output: ${two.output}
nodes:
  root:
    kind: code
    code: tests.engine._compose_codefns:echo
    input: {value: root}
    output: str
output: ${root.output}
"""
    )

    assert source_loads == 1


def test_defs_callable_runs_end_to_end_without_external_resolver():
    # the call resolves to the in-file def -> NO external resolver needed.
    loaded = load_flow(_DEFS_PARENT)
    result = run_flow(loaded, {"topic": "ACME"})
    assert result.status == "succeeded"
    # research_pair: make_report -> {report: "report for ACME", ...} -> echo the report.
    assert result.output == "report for ACME"


# --------------------------------------------------------------------------- #
# A def, called as a MAPPED call (over:) — defs compose with iteration.
# --------------------------------------------------------------------------- #

_DEFS_MAP = """
id: defs-map
name: defs_map
input:
  topics: list[str]
defs:
  one:
    input:
      topic: str
    nodes:
      x:
        kind: code
        code: tests.engine._compose_codefns:echo
        input:
          topic: ${input.topic}
        output: str
    output: ${x.output}
nodes:
  each:
    kind: map
    call: one
    over: ${input.topics}
    input:
      topic: ${item}
output: ${each.output}
"""


def test_defs_callable_as_mapped_call():
    loaded = load_flow(_DEFS_MAP)
    result = run_flow(loaded, {"topics": ["A", "B", "C"]})
    assert result.status == "succeeded"
    assert result.output == ["A", "B", "C"]


# --------------------------------------------------------------------------- #
# A def calls another def (nested, resolver-free).
# --------------------------------------------------------------------------- #

_NESTED_DEFS = """
id: nested
name: nested
input:
  topic: str
defs:
  inner:
    input:
      topic: str
    nodes:
      x:
        kind: code
        code: tests.engine._compose_codefns:make_report
        input:
          topic: ${input.topic}
        output:
          report: str
          n: int
    output: ${x.output.report}
  outer:
    input:
      topic: str
    nodes:
      inner_call:
        kind: call
        call: inner
        input:
          topic: ${input.topic}
    output: ${inner_call.output}
nodes:
  go:
    kind: call
    call: outer
    input:
      topic: ${input.topic}
output: ${go.output}
"""


def test_def_calls_another_def():
    loaded = load_flow(_NESTED_DEFS)
    result = run_flow(loaded, {"topic": "ACME"})
    assert result.status == "succeeded"
    assert result.output == "report for ACME"


# --------------------------------------------------------------------------- #
# defs-first: an in-file def shadows an external flow of the same name.
# --------------------------------------------------------------------------- #

_SHADOW_PARENT = """
id: shadow-parent
name: shadow_parent
input:
  topic: str
defs:
  shared:
    input:
      topic: str
    nodes:
      x:
        kind: code
        code: tests.engine._compose_codefns:make_report
        input:
          topic: ${input.topic}
        output:
          report: str
          n: int
    output: ${x.output.report}
nodes:
  call_shared:
    kind: call
    call: shared
    input:
      topic: ${input.topic}
output: ${call_shared.output}
"""

_EXTERNAL_SHARED = """
id: shared
name: shared
input:
  topic: str
nodes:
  x:
    kind: code
    code: tests.engine._compose_codefns:echo
    input:
      topic: ${input.topic}
    output: str
output: ${x.output}
"""


def test_defs_shadow_external_of_same_name():
    def external(flow_id, version=None):
        return load_flow(_EXTERNAL_SHARED)

    loaded = load_flow(_SHADOW_PARENT, child_resolver=external)
    result = run_flow(loaded, {"topic": "ACME"})
    assert result.status == "succeeded"
    # the in-file def (make_report -> "report for ACME") won, NOT the external echo ("ACME").
    assert result.output == "report for ACME"


# --------------------------------------------------------------------------- #
# recursive / mutual defs are loud (finite baked call graph).
# --------------------------------------------------------------------------- #

_SELF_RECURSIVE = """
id: rec
name: rec
input:
  topic: str
defs:
  a:
    input:
      topic: str
    nodes:
      step:
        kind: call
        call: a
        input:
          topic: ${input.topic}
    output: ${step.output}
nodes:
  go:
    kind: call
    call: a
    input:
      topic: ${input.topic}
output: ${go.output}
"""

_MUTUAL_RECURSIVE = """
id: mutual
name: mutual
input:
  topic: str
defs:
  a:
    input:
      topic: str
    nodes:
      to_b:
        kind: call
        call: b
        input:
          topic: ${input.topic}
    output: ${to_b.output}
  b:
    input:
      topic: str
    nodes:
      to_a:
        kind: call
        call: a
        input:
          topic: ${input.topic}
    output: ${to_a.output}
nodes:
  go:
    kind: call
    call: a
    input:
      topic: ${input.topic}
output: ${go.output}
"""


def test_self_recursive_def_is_loud():
    with pytest.raises(LoadError) as exc:
        load_flow(_SELF_RECURSIVE)
    assert "recursive" in str(exc.value).lower()
    assert "'a'" in str(exc.value)


def test_mutually_recursive_defs_are_loud():
    with pytest.raises(LoadError) as exc:
        load_flow(_MUTUAL_RECURSIVE)
    assert "recursive" in str(exc.value).lower()


# --------------------------------------------------------------------------- #
# single-node def: a flat `kind:`+logic body whose `inputs:` are params (name -> TYPE),
# auto-wired by name into the one node; `outputs:` is the codomain type.
# --------------------------------------------------------------------------- #

_SINGLE_NODE_DEF = """
id: single
name: single
input:
  t: str
defs:
  passthru:
    kind: code
    code: tests.engine._compose_codefns:echo
    input: { topic: str }
    output: str
nodes:
  go:
    kind: call
    call: passthru
    input:
      topic: ${input.t}
output: ${go.output}
"""


def test_single_node_def_runs():
    # `topic` (a def param) auto-binds by name into the echo node; echo returns it.
    loaded = load_flow(_SINGLE_NODE_DEF)
    res = run_flow(loaded, {"t": "ACME"})
    assert res.status == "succeeded", res.failure
    assert res.output == "ACME"


def test_single_node_def_projected_header_and_logic_origins():
    loaded = load_flow(_SINGLE_NODE_DEF)
    node = loaded.compiled.nodes["go"].child.nodes["passthru"]
    resolver = loaded.origin_resolver
    assert resolver is not None

    assert node.origin is not None
    assert node.code_origin is not None
    assert resolver.resolve(node.origin).range.start_line == 7
    assert resolver.resolve(node.code_origin).range.start_line == 9


def test_single_node_def_generated_inputs_keep_binding_origin():
    loaded = load_flow(_SINGLE_NODE_DEF)
    child = loaded.compiled.nodes["go"].child
    binding = child.wiring["passthru"]["topic"]
    resolver = loaded.origin_resolver

    assert resolver is not None
    assert binding.origin is not None
    assert resolver.resolve(binding.origin).range == SourceRange(10, 21, 10, 24)


def test_single_node_def_generated_output_keeps_contract_origin():
    loaded = load_flow(_SINGLE_NODE_DEF)
    node = loaded.compiled.nodes["go"].child.nodes["passthru"]
    origin = node.output_contract.origins.default
    resolver = loaded.origin_resolver

    assert resolver is not None
    assert origin is not None
    assert resolver.resolve(origin).range == SourceRange(11, 13, 11, 16)


_SINGLE_NODE_DEF_BADTYPE = """
id: single
name: single
input:
  t: str
defs:
  passthru:
    kind: code
    code: tests.engine._compose_codefns:echo
    input: { topic: int }
    output: str
nodes:
  go:
    kind: call
    call: passthru
    input:
      topic: ${input.t}
output: ${go.output}
"""


def test_single_node_def_e06_mismatch():
    # the parent binds a str to a def param typed int -> cross-flow type error.
    with pytest.raises(LoadError):
        load_flow(_SINGLE_NODE_DEF_BADTYPE)


# --------------------------------------------------------------------------- #
# a call to a name that is neither a def nor an external flow (no resolver) is loud.
# --------------------------------------------------------------------------- #

_UNKNOWN_CALL = """
id: unk
name: unk
input:
  topic: str
defs:
  known:
    input:
      topic: str
    nodes:
      x:
        kind: code
        code: tests.engine._compose_codefns:echo
        input:
          topic: ${input.topic}
        output: str
    output: ${x.output}
nodes:
  go:
    kind: call
    call: missing
    input:
      topic: ${input.topic}
output: ${go.output}
"""


def test_call_to_unknown_callable_is_loud():
    with pytest.raises(LoadError) as exc:
        load_flow(_UNKNOWN_CALL)
    msg = str(exc.value)
    assert "missing" in msg
    # a bare call to a non-def/non-use callable is loud (external only via uses:)
    assert "uses:" in msg


def test_malformed_def_without_nodes_is_loud():
    text = """
id: bad
name: bad
input:
  topic: str
defs:
  oops:
    input:
      topic: str
    output: ${nope.output}
nodes:
  go:
    kind: call
    call: oops
    input:
      topic: ${input.topic}
output: ${go.output}
"""
    with pytest.raises(LoadError) as exc:
        load_flow(text)
    assert "nodes" in str(exc.value)


# --------------------------------------------------------------------------- #
# a def's `asserts:` is admitted and enforced at the REF/MAP child seam.
# --------------------------------------------------------------------------- #

_DEF_WITH_ASSERTS = """
id: da
name: da
input:
  topic: str
defs:
  d:
    input:
      topic: str
    nodes:
      x:
        kind: code
        code: tests.engine._compose_codefns:echo
        input:
          topic: ${input.topic}
        output: str
    output: ${x.output}
    asserts:
      - ${x.output} != ""
nodes:
  go:
    kind: call
    call: d
    input:
      topic: ${input.topic}
output: ${go.output}
"""


def test_def_asserts_are_enforced_at_the_child_seam():
    # a def's `asserts:` is admitted and enforced at the REF/MAP child seam.
    out = run_flow(load_flow(_DEF_WITH_ASSERTS), {"topic": "ACME"})
    assert out.status == "succeeded"  # the def's post assert `${x.output} != ""` holds
    assert out.output == "ACME"


# --------------------------------------------------------------------------- #
# a def-internal error names which def it came from (loud + located-by-def).
# --------------------------------------------------------------------------- #

_DEF_DANGLING_REF = """
id: dd
name: dd
input:
  topic: str
defs:
  enrich:
    input:
      topic: str
    nodes:
      x:
        kind: code
        code: tests.engine._compose_codefns:echo
        input:
          topic: ${nonexistent.output}
        output: str
    output: ${x.output}
nodes:
  go:
    kind: call
    call: enrich
    input:
      topic: ${input.topic}
output: ${go.output}
"""


def test_def_internal_error_names_the_def():
    with pytest.raises(LoadError) as exc:
        load_flow(_DEF_DANGLING_REF)
    msg = str(exc.value)
    assert "defs entry 'enrich'" in msg   # the error names which def failed
    assert "nonexistent" in msg


# --------------------------------------------------------------------------- #
# def body shape guards: a non-mapping body / an unknown field are loud.
# --------------------------------------------------------------------------- #


def test_def_non_mapping_body_is_loud():
    text = """
id: nm
name: nm
input:
  topic: str
defs:
  d: just a string
nodes:
  go:
    kind: call
    call: d
    input:
      topic: ${input.topic}
output: ${go.output}
"""
    with pytest.raises(LoadError) as exc:
        load_flow(text)
    assert "must be a mapping" in str(exc.value)
    assert "'d'" in str(exc.value)


def test_def_unknown_field_is_loud():
    text = """
id: uf
name: uf
input:
  topic: str
defs:
  d:
    input:
      topic: str
    nodes:
      x:
        kind: code
        code: tests.engine._compose_codefns:echo
        input:
          topic: ${input.topic}
        output: str
    output: ${x.output}
    bogus: 1
nodes:
  go:
    kind: call
    call: d
    input:
      topic: ${input.topic}
output: ${go.output}
"""
    with pytest.raises(LoadError) as exc:
        load_flow(text)
    assert "unknown field" in str(exc.value)
    assert "bogus" in str(exc.value)


# --------------------------------------------------------------------------- #
# eager build: a def that is NEVER called but has a broken body is still loud at load.
# --------------------------------------------------------------------------- #

_UNREFERENCED_BROKEN_DEF = """
id: ub
name: ub
input:
  topic: str
defs:
  used:
    input:
      topic: str
    nodes:
      x:
        kind: code
        code: tests.engine._compose_codefns:echo
        input:
          topic: ${input.topic}
        output: str
    output: ${x.output}
  unused_broken:
    input:
      topic: str
    nodes:
      y:
        kind: code
        code: tests.engine._compose_codefns:echo
        input:
          topic: ${nonexistent.output}
        output: str
    output: ${y.output}
nodes:
  go:
    kind: call
    call: used
    input:
      topic: ${input.topic}
output: ${go.output}
"""


def test_unreferenced_broken_def_is_loud_eager_build():
    # `unused_broken` is never reached by a call, but eager build validates every def.
    with pytest.raises(LoadError) as exc:
        load_flow(_UNREFERENCED_BROKEN_DEF)
    assert "defs entry 'unused_broken'" in str(exc.value)


# --------------------------------------------------------------------------- #
# nullable-strictness: a nullable source bound to a non-nullable child input is
# rejected unless the BINDING guarantees non-null (:-literal / :?).
# The child's own `default:` does NOT cover it (the parent's null shadows it).
# --------------------------------------------------------------------------- #

_NULLABLE_TO_DEFAULTED = """
id: nstrict
name: nstrict
input:
  x: Optional[str]
defs:
  child:
    kind: code
    code: tests.engine._compose_codefns:echo
    input:
      topic: str = "fallback"
    output: str
nodes:
  go:
    kind: call
    call: child
    input:
      topic: ${input.x}
output: ${go.output}
"""


def test_nullable_source_to_defaulted_child_input_is_strict():
    # child `topic` is non-nullable (has a default); x is nullable. The parent's null
    # would SHADOW the default -> rejected (the child default doesn't cover it).
    with pytest.raises(LoadError):
        load_flow(_NULLABLE_TO_DEFAULTED)


_NULLABLE_LIST_TO_STRICT_CHILD = """
id: nullable-list-strict-child
name: nullable_list_strict_child
input:
  items: list[Optional[int]]
defs:
  child:
    kind: code
    code: |
      return inputs["items"]
    input:
      items: list[int]
    output: list[int]
nodes:
  go:
    kind: call
    call: child
    input:
      items: ${input.items}
output: ${go.output}
"""


def test_nullable_list_element_source_to_strict_child_is_rejected():
    """Element nullability is rejected at load, before an actual null can run."""
    with pytest.raises(LoadError) as caught:
        load_flow(_NULLABLE_LIST_TO_STRICT_CHILD)

    assert "call node 'go': binding 'items'" in str(caught.value)
    assert (
        "child expects 'list[int]', source is 'list[Optional[int]]'"
        in str(caught.value)
    )


_NULLABLE_WITH_LITERAL_ESCAPE = """
id: nok
name: nok
input:
  x: Optional[str]
defs:
  child:
    kind: code
    code: tests.engine._compose_codefns:echo
    input: { topic: str }
    output: str
nodes:
  go:
    kind: call
    call: child
    input:
      topic: ${input.x:-"safe"}
output: ${go.output}
"""


def test_nullable_source_with_literal_escape_ok():
    # `:-safe` guarantees a non-null value (fires on null, present-null included) -> ok.
    loaded = load_flow(_NULLABLE_WITH_LITERAL_ESCAPE)
    res = run_flow(loaded, {"x": None})
    assert res.status == "succeeded", res.failure
    assert res.output == "safe"


# --------------------------------------------------------------------------- #
# parent-binds-null SHADOWS the child default (null != absent; like
# Python f(x=None)). apply_defaults fills only OMITTED inputs.
# --------------------------------------------------------------------------- #

_SHADOW = """
id: shadow
name: shadow
input:
  x: Optional[str]
defs:
  child:
    kind: code
    code: tests.engine._compose_codefns:echo
    input:
      topic: Optional[str] = "fallback"
    output: Optional[str]
nodes:
  go:
    kind: call
    call: child
    input:
      topic: ${input.x}
output: ${go.output}
"""


def test_parent_null_shadows_child_default():
    loaded = load_flow(_SHADOW)
    res = run_flow(loaded, {"x": None})
    assert res.status == "succeeded", res.failure
    assert res.output is None  # the parent's explicit null shadows the child default


def test_omitted_child_input_applies_explicit_null_default():
    loaded = load_flow(
        """id: omitted-null-default
name: omitted_null_default
defs:
  child:
    kind: code
    code: |
      return inputs["maybe"]
    input:
      maybe: Optional[int] = null
    output: Optional[int]
nodes:
  go:
    kind: call
    call: child
output: ${go.output}
"""
    )
    parameter = loaded.compiled.nodes["go"].child_inputs[0]

    assert type(parameter) is Parameter
    assert type(parameter.contract) is ValueContract
    assert parameter.contract.type == Type(kind=ValueKind.INTEGER, nullable=True)
    assert parameter.has_default is True
    assert parameter.default is None
    assert parameter.coerce_strings is True

    result = run_flow(loaded, {})
    assert result.status == "succeeded", result.failure
    assert result.output is None


def test_invalid_def_default_keeps_absolute_range_and_def_context():
    source = """id: invalid-def-default
name: invalid_def_default
defs:
  child:
    kind: code
    code: |
      return inputs["value"]
    input:
      value: int = null
    output: int
nodes:
  go:
    kind: call
    call: child
output: ${go.output}
"""

    with pytest.raises(LoadError, match="defs entry 'child'") as caught:
        load_flow(source)

    assert caught.value.diagnostic.frames[-1].range.start_line == 9


def test_def_yaml_merge_uses_local_nodes_override():
    """A located def reader must pair the effective value with its last YAML pair."""
    loaded = load_flow(
        """id: merged-def
name: merged_def
x-def-base: &def_base
  nodes:
    old:
      kind: code
      code: |
        return "old"
      output: str
  output: ${old.output}
defs:
  target:
    <<: *def_base
    nodes:
      new:
        kind: code
        code: |
          return "new"
        output: str
    output: ${new.output}
nodes:
  invoke:
    kind: call
    call: target
output: ${invoke.output}
"""
    )

    result = run_flow(loaded, {})

    assert result.status == "succeeded"
    assert result.output == "new"
