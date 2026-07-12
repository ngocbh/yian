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

from agent_composer.compose import LoadError, load_flow
from agent_composer.run import run_flow
from agent_composer.compose.parse.file import parse_file

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
    assert res.status == "succeeded", res.error
    assert res.output == "ACME"


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
    assert res.status == "succeeded", res.error
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
    assert res.status == "succeeded", res.error
    assert res.output is None  # the parent's explicit null shadows the child default
