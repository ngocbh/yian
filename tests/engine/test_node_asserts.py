"""Node-level `asserts:` — the per-node contract.

Compile: node-local validation + pre/post classification + stamping onto the Node.
Runtime: per-node enforcement in `Node.run` (pre before `_run`, post after) — e2e below.
"""

from types import SimpleNamespace

import pytest

from yian.compose import LoadError, load_flow
from yian.compose.helpers.errors import LoadDiagnostics
from yian.compose.validate.asserts import validate_node_asserts
from yian.diagnostics import OriginRef, SourceRange
from yian.nodes.base import Node
from yian.nodes.contracts import (
    Assertion,
    ContractOrigins,
    Parameter,
    PartialRecordContract,
    ValueContract,
)
from yian.run import run_flow
from yian.typesys.values import Type, ValueKind


def _flow(asserts: list[str], *, input_decl="n: ${input.v}", outputs="int") -> str:
    a = "\n".join(f'      - "{q}"' for q in asserts)
    return f"""
id: na
name: na
input:
  v: int
nodes:
  x:
    kind: code
    code: tests.engine._compose_codefns:double
    input:
      {input_decl}
    output: {outputs}
    asserts:
{a}
output: ${{x.output}}
"""


def _assertion_text(assertions: list[Assertion]) -> list[tuple[str, str]]:
    """Project carriers onto the executable and authored text under test here."""
    return [(assertion.expression, assertion.display) for assertion in assertions]


def _range_diagnostics() -> LoadDiagnostics:
    """Return source text large enough for direct validator ranges."""
    return LoadDiagnostics(label="flow.yaml", text=("x" * 80 + "\n") * 30)


# ---------- classify + stamp ----------


def test_node_asserts_classify_pre_and_post():
    loaded = load_flow(_flow(["output >= 0", "n >= 0"]))
    x = loaded.compiled.nodes["x"]
    assert _assertion_text(x.post_asserts) == [
        ("output >= 0", "output >= 0")
    ]
    assert _assertion_text(x.pre_asserts) == [("n >= 0", "n >= 0")]


def test_assertion_site_lookup_is_keyword_only() -> None:
    """A retired third positional argument cannot silently bind as a site lookup."""
    with pytest.raises(TypeError):
        validate_node_asserts({}, {}, lambda _node_id, _index: None)  # type: ignore[call-arg]


def test_node_assert_field_walk_uses_the_canonical_output_contract():
    """Node-local validation projects the type without replacing provenance."""
    contract_type = Type(
        kind=ValueKind.OBJECT,
        fields={"score": Type.scalar(ValueKind.NUMBER)},
        required=frozenset({"score"}),
    )
    origins = ContractOrigins(default=OriginRef("node-output"))
    node = Node(
        "x",
        output_contract=ValueContract(contract_type, origins),
    )
    descriptor = SimpleNamespace(
        asserts=["output.score >= 0"],
        assert_ranges=[],
    )

    validate_node_asserts({"x": node}, {"x": descriptor})

    assert node.post_asserts == [
        Assertion("output.score >= 0", "output.score >= 0")
    ]
    assert node.output_contract.type is contract_type
    assert node.output_contract.origins is origins


def test_node_assert_rejects_unknown_field_on_partial_output_contract():
    """Node-local asserts use the partial contract's closed root field set."""
    node = Node(
        "x",
        output_contract=PartialRecordContract(
            fields=(
                ("score", ValueContract(Type.scalar(ValueKind.NUMBER))),
                ("note", None),
            )
        ),
    )
    descriptor = SimpleNamespace(
        asserts=["output.missing >= 0"],
        assert_ranges=[],
    )

    with pytest.raises(LoadError, match="unknown field 'missing'"):
        validate_node_asserts({"x": node}, {"x": descriptor})


@pytest.mark.parametrize("pre", ["n >= 0", "${n} >= 0", "${n >= 0}"])
@pytest.mark.parametrize("post", ["output >= 0", "${output} >= 0", "${output >= 0}"])
def test_node_asserts_classify_same_across_spellings(pre, post):
    # A node assert's refs are extracted by PARSING the whole value (via condition_refs),
    # so the bare / mixed / whole-span spellings must classify identically: a `${n}`-only
    # (input) assert is pre, an `${output}`-reading assert is post.
    x = load_flow(_flow([post, pre])).compiled.nodes["x"]
    assert _assertion_text(x.post_asserts) == [(post, post)]
    assert _assertion_text(x.pre_asserts) == [(pre, pre)]


def test_duplicate_node_asserts_are_distinct_ordered_occurrences():
    """Equal authored text must not collapse occurrence identity or provenance."""
    text = "n >= 0"
    loaded = load_flow(_flow([text, text]))
    x = loaded.compiled.nodes["x"]

    assert _assertion_text(x.pre_asserts) == [(text, text), (text, text)]
    assert x.pre_asserts[0] is not x.pre_asserts[1]
    assert x.pre_asserts[0].origin is not None
    assert x.pre_asserts[1].origin is not None
    assert x.pre_asserts[0].origin != x.pre_asserts[1].origin
    assert loaded.origin_resolver is not None
    frames = [
        loaded.origin_resolver.resolve(assertion.origin)
        for assertion in x.pre_asserts
    ]
    assert all(frame is not None for frame in frames)
    assert frames[0].range != frames[1].range


@pytest.mark.parametrize("bad", ["nope > 0", "${nope} > 0", "${nope > 0}"])
def test_reject_unknown_input_ref_in_all_spellings(bad):
    with pytest.raises(LoadError, match="not a declared input"):
        load_flow(_flow([bad]))


# ---------- rejections (located LoadError) ----------


def test_reject_unknown_input_ref():
    source = _flow(["nope > 0", "input.v > 0"])
    expected_line = next(
        line_no
        for line_no, line in enumerate(source.splitlines(), start=1)
        if "nope > 0" in line
    )
    with pytest.raises(LoadError, match="not a declared input") as caught:
        load_flow(source)
    assert caught.value.diagnostic.frames[0].range.start_line == expected_line


def test_node_assert_field_error_prefers_the_exact_input_range():
    """A pre-loop field error uses its field before the asserts block or node."""
    input_range = SourceRange(5, 3, 5, 9)
    descriptor = SimpleNamespace(
        asserts=["output >= 0"],
        assert_ranges=[SourceRange(8, 5, 8, 20)],
        field_ranges={
            ("input", "output"): input_range,
            ("field", "asserts"): SourceRange(7, 3, 8, 20),
        },
        range=SourceRange(2, 1, 2, 2),
    )
    node = Node("x")
    node.params = [Parameter(name="output")]

    with pytest.raises(LoadError, match="collides") as caught:
        validate_node_asserts(
            {"x": node},
            {"x": descriptor},
            diagnostics=_range_diagnostics(),
        )

    assert caught.value.diagnostic.frames[0].range == input_range


def test_node_assert_error_falls_back_to_the_descriptor_range():
    """A source-backed node still locates an error when its item range is absent."""
    node_range = SourceRange(4, 3, 4, 4)
    descriptor = SimpleNamespace(
        asserts=["missing > 0"],
        assert_ranges=[],
        field_ranges={},
        range=node_range,
    )
    node = Node("x")
    node.params = []

    with pytest.raises(LoadError, match="not a declared input") as caught:
        validate_node_asserts(
            {"x": node},
            {"x": descriptor},
            diagnostics=_range_diagnostics(),
        )

    assert caught.value.diagnostic.frames[0].range == node_range


def test_node_assert_aggregate_uses_first_available_range():
    """A later authored range locates an aggregate whose first error is unlocated."""
    later_range = SourceRange(20, 3, 20, 19)
    descriptors = {
        "first": SimpleNamespace(
            asserts=["missing_first > 0"],
            assert_ranges=[],
            field_ranges={},
            range=None,
        ),
        "later": SimpleNamespace(
            asserts=["missing_later > 0"],
            assert_ranges=[later_range],
            field_ranges={},
            range=later_range,
        ),
    }
    nodes = {"first": Node("first"), "later": Node("later")}

    with pytest.raises(LoadError, match="missing_first") as caught:
        validate_node_asserts(
            nodes,
            descriptors,
            diagnostics=_range_diagnostics(),
        )

    assert caught.value.diagnostic.frames[0].range == later_range


def test_reject_pool_head_ref():
    with pytest.raises(LoadError, match="node-local"):
        load_flow(_flow(["input.v > 0"]))


def test_reject_inline_call_in_node_assert():
    # the retired `${flow(args)}` span form of an inline call is rejected in a node assert.
    flow = f"""
id: na
name: na
input:
  v: int
defs:
  dbl:
    input:
      n: int
    nodes:
      y:
        kind: code
        code: tests.engine._compose_codefns:double
        input:
          n: ${{input.n}}
        output: int
    output: ${{y.output}}
nodes:
  x:
    kind: code
    code: tests.engine._compose_codefns:double
    input:
      n: ${{input.v}}
    output: int
    asserts:
      - "${{ dbl(n=input.v) >= 0 }}"
output: ${{x.output}}
"""
    with pytest.raises(LoadError, match="inline"):
        load_flow(flow)


def test_reject_call_directive_in_node_assert():
    # the whole-value `call(...)` directive form of an inline call is also rejected in a
    # node assert — inline calls belong in a flow-level assert, never node-local.
    flow = f"""
id: na
name: na
input:
  v: int
defs:
  dbl:
    input:
      n: int
    nodes:
      y:
        kind: code
        code: tests.engine._compose_codefns:double
        input:
          n: ${{input.n}}
        output: int
    output: ${{y.output}}
nodes:
  x:
    kind: code
    code: tests.engine._compose_codefns:double
    input:
      n: ${{input.v}}
    output: int
    asserts:
      - call(dbl, n=1)
output: ${{x.output}}
"""
    with pytest.raises(LoadError, match="inline"):
        load_flow(flow)


def test_reject_input_named_output():
    with pytest.raises(LoadError, match="collides"):
        load_flow(_flow(["output >= 0"], input_decl="output: ${input.v}"))


def test_reject_dotted_field_on_scalar_output():
    with pytest.raises(LoadError):
        load_flow(_flow(["output.field >= 0"]))  # output is int (scalar)


def test_reject_node_asserts_on_mapped_call():
    flow = f"""
id: nam
name: nam
input:
  vs: list[int]
defs:
  dbl:
    input:
      n: int
    nodes:
      y:
        kind: code
        code: tests.engine._compose_codefns:double
        input:
          n: ${{input.n}}
        output: int
    output: ${{y.output}}
nodes:
  m:
    kind: map
    call: dbl
    over: ${{input.vs}}
    input:
      n: ${{item}}
    asserts:
      - "output >= 0"
output: ${{m.output}}
"""
    with pytest.raises(LoadError, match="mapped"):
        load_flow(flow)


# ---------- per-node enforcement (e2e via run_flow) ----------


def test_pre_assert_passes_and_runs():
    out = run_flow(load_flow(_flow(["n >= 0"])), {"v": 5})
    assert out.status == "succeeded" and out.output == 10


def test_pre_assert_violated_fails_before_body():
    out = run_flow(load_flow(_flow(["n >= 0"])), {"v": -1})
    assert out.status != "succeeded"
    assert out.failure is not None
    assert "pre-assert" in out.failure.message


def test_post_assert_violated_fails():
    out = run_flow(load_flow(_flow(["output >= 0"])), {"v": -1})  # double(-1) = -2
    assert out.status != "succeeded"
    assert out.failure is not None
    assert "post-assert" in out.failure.message


def test_post_assert_on_nonscalar_output_fails_cleanly():
    # ${output} is a record (dict); an ordered op RAISES inside evaluate_when_record — the
    # wrapper must turn that into a clean NodeFailed rather than crash the run.
    flow = """
id: rec
name: rec
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
    asserts:
      - "output >= 0"
output: ${x.output}
"""
    out = run_flow(load_flow(flow), {"topic": "ACME"})  # output is {report, n}
    assert out.status != "succeeded"  # clean RunFailed, no uncaught exception


def test_post_assert_record_field_passes():
    flow = """
id: recf
name: recf
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
    asserts:
      - "output.n >= 0"
output: ${x.output}
"""
    out = run_flow(load_flow(flow), {"topic": "ACME"})
    assert out.status == "succeeded"


def test_skipped_branch_target_asserts_do_not_fire():
    # `gate` routes to `lo`; `hi` (with an assert that WOULD fail) is skip-flooded -> its
    # asserts never fire (skip-correct), and the run succeeds with `lo`'s value.
    flow = """
id: skp
name: skp
input:
  v: int
nodes:
  gate:
    kind: case
    cases:
      - when: "input.v >= 100"
        then: hi
    else: lo
  hi:
    kind: code
    code: tests.engine._compose_codefns:double
    input:
      n: ${input.v}
    output: int
    asserts:
      - "output > 1000000"
  lo:
    kind: code
    code: tests.engine._compose_codefns:double
    input:
      n: ${input.v}
    output: int
output: ${gate.output}
"""
    out = run_flow(load_flow(flow), {"v": 5})  # -> lo (hi skip-flooded, its assert never fires)
    assert out.status == "succeeded" and out.output == 10


def test_absent_input_pre_assert_fails_cleanly():
    # a node runs with a required input bound from an omitted Optional flow input (-> None);
    # the assert-time bind raises BindingError, which must become a clean RunFailed.
    flow = """
id: ab
name: ab
input:
  maybe: Optional[int]
nodes:
  x:
    kind: code
    code: tests.engine._compose_codefns:double
    input:
      n: ${input.maybe}
    output: int
    asserts:
      - "n >= 0"
output: ${x.output}
"""
    out = run_flow(load_flow(flow), {})  # maybe omitted -> None -> required bind fails
    assert out.status != "succeeded"  # clean RunFailed, not an uncaught exception


_CHECKED_DEF = """
defs:
  checked:
    input:
      n: int
    nodes:
      y:
        kind: code
        code: tests.engine._compose_codefns:double
        input:
          n: ${input.n}
        output: int
        asserts:
          - "output >= 0"
    output: ${y.output}
"""


def test_def_internal_node_assert_via_call():
    flow = f"""
id: dr
name: dr
input:
  v: int
{_CHECKED_DEF}
nodes:
  call_it:
    kind: call
    call: checked
    input:
      n: ${{input.v}}
output: ${{call_it.output}}
"""
    loaded = load_flow(flow)
    assert run_flow(loaded, {"v": 5}).status == "succeeded"          # y=10, output>=0 holds
    bad = run_flow(loaded, {"v": -1})                                # y=-2 -> def-internal post-assert fails
    assert bad.status != "succeeded"


def test_def_internal_node_assert_via_map():
    flow = f"""
id: dm
name: dm
input:
  vs: list[int]
{_CHECKED_DEF}
nodes:
  m:
    kind: map
    call: checked
    over: ${{input.vs}}
    input:
      n: ${{item}}
output: ${{m.output}}
"""
    loaded = load_flow(flow)
    assert run_flow(loaded, {"vs": [1, 2, 3]}).status == "succeeded"
    assert run_flow(loaded, {"vs": [1, -1, 3]}).status != "succeeded"  # element -1 -> y=-2 fails
