"""`${<case>.output}` = the taken-branch value.

A `case` read as a VALUE: `${<case>.output}` desugars at load to a COALESCE over the
case's branch targets (`${t1.output | t2.output | … | else.output}`). Pure loader
sugar — skip-flood makes every non-taken branch null, so the coalesce yields the taken
value (exactly seed 02's hand-written join). Runtime IR unchanged.

Descriptor-level coverage exercises `expand_case_outputs` over constructed descriptors:
whole, dotted, embedded, `on:` form, nested cases flattened to leaves, non-case, and
fast-path references. It also covers loud rejection of a case value in a non-clean binding
position, a condition, or an assert. End-to-end coverage loads and runs CODE branches with
`output: ${gate.output}` and with a node consuming `${gate.output}`.

(The `then:/else: ${call}` inline-call branch form is covered by
`tests/engine/test_case_call.py`.)
"""

import pytest

from yian.compose import LoadError, expand_case_outputs, load_flow
from yian.compose.helpers.errors import LoadDiagnostics
from yian.run import run_flow
from yian.compose.parse.file import (
    CaseDescriptor,
    CodeDescriptor,
    HumanInputDescriptor,
    LoopDescriptor,
)
from yian.diagnostics import SourceRange


# --------------------------------------------------------------------------- #
# Descriptor-level `expand_case_outputs` coverage
# --------------------------------------------------------------------------- #


def _gate(then="pro", else_="caut"):
    return CaseDescriptor(
        id="gate", cases=[{"when": "${s.output} >= 0.5", "then": then}], else_=else_
    )


def _code(nid, **inputs):
    return CodeDescriptor(id=nid, code="m:f", inputs=inputs)


def test_whole_ref_expands_to_coalesce():
    descs = {"gate": _gate(), "c": _code("c", v="${gate.output}")}
    new, _ = expand_case_outputs(descs, None, [])
    assert new["c"].inputs["v"] == "${pro.output | caut.output}"


def test_dotted_case_ref_expands_per_branch():
    descs = {"gate": _gate(), "c": _code("c", v="${gate.output.report}")}
    new, _ = expand_case_outputs(descs, None, [])
    assert new["c"].inputs["v"] == "${pro.output.report | caut.output.report}"


def test_embedded_case_ref_expands():
    descs = {"gate": _gate(), "c": _code("c", v="pre ${gate.output} post")}
    new, _ = expand_case_outputs(descs, None, [])
    assert new["c"].inputs["v"] == "pre ${pro.output | caut.output} post"


def test_on_form_case_value_uses_then_targets():
    gate = CaseDescriptor(
        id="gate",
        on="${x.output.tag}",
        cases=[{"when": "a", "then": "na"}, {"when": "b", "then": "nb"}],
    )
    descs = {"gate": gate, "c": _code("c", v="${gate.output}")}
    new, _ = expand_case_outputs(descs, None, [])
    assert new["c"].inputs["v"] == "${na.output | nb.output}"  # no else -> just thens


def test_outputs_section_ref_expands():
    descs = {"gate": _gate()}
    _, new_out = expand_case_outputs(descs, "${gate.output}", [])
    assert new_out == "${pro.output | caut.output}"


def test_case_of_case_value_flattens_to_leaves():
    # a case whose target is itself a case -> ${<outer>.output} flattens through the
    # nested case to its LEAF (non-case) targets (sound via the veto, which skip-floods
    # the nested gate when the outer branch loses).
    outer = CaseDescriptor(
        id="outer", cases=[{"when": "w", "then": "inner"}], else_="x"
    )
    inner = CaseDescriptor(id="inner", cases=[{"when": "w", "then": "a"}], else_="b")
    descs = {"outer": outer, "inner": inner, "c": _code("c", v="${outer.output}")}
    new, _ = expand_case_outputs(descs, None, [])
    assert new["c"].inputs["v"] == "${a.output | b.output | x.output}"


def test_case_of_case_cycle_drops_arm_is_loud():
    # g1 -> g2 -> g1: the cycle-guarded flatten drops the cyclic arm -> empty leaves -> loud.
    g1 = CaseDescriptor(id="g1", cases=[{"when": "w", "then": "g2"}])  # no else
    g2 = CaseDescriptor(id="g2", cases=[{"when": "w", "then": "g1"}])  # no else
    descs = {"g1": g1, "g2": g2, "c": _code("c", v="${g1.output}")}
    with pytest.raises(LoadError, match="cycle"):
        expand_case_outputs(descs, None, [])


def test_non_case_ref_untouched():
    descs = {"gate": _gate(), "c": _code("c", v="${s.output}")}
    new, _ = expand_case_outputs(descs, None, [])
    assert new["c"].inputs["v"] == "${s.output}"


def test_multiple_case_values_in_one_binding_both_expand():
    g1 = CaseDescriptor(id="g1", cases=[{"when": "w", "then": "a"}], else_="b")
    g2 = CaseDescriptor(id="g2", cases=[{"when": "w", "then": "c"}], else_="d")
    descs = {
        "g1": g1,
        "g2": g2,
        "n": _code("n", v="pre ${g1.output} mid ${g2.output} post"),
    }
    new, _ = expand_case_outputs(descs, None, [])
    assert (
        new["n"].inputs["v"]
        == "pre ${a.output | b.output} mid ${c.output | d.output} post"
    )


def test_case_value_in_tool_inputs_expands():
    from yian.compose.parse.file import ToolDescriptor

    descs = {
        "gate": _gate(),
        "t": ToolDescriptor(id="t", tool_id="x", inputs={"q": "${gate.output}"}),
    }
    new, _ = expand_case_outputs(descs, None, [])
    assert new["t"].inputs["q"] == "${pro.output | caut.output}"


def test_case_value_in_human_input_and_loop_inputs_expands():
    hosts = (
        HumanInputDescriptor(
            id="human", prompt="Continue?", inputs={"q": "${gate.output}"}
        ),
        LoopDescriptor(id="loop", call="body", inputs={"q": "${gate.output}"}),
    )

    for host in hosts:
        new, _ = expand_case_outputs(
            {"gate": _gate(), host.id: host},
            None,
            [],
        )
        assert new[host.id].inputs["q"] == "${pro.output | caut.output}"


def test_case_with_no_branch_target_is_loud():
    # a malformed case (no then:, no else:) referenced as a value -> clear LoadError, not `${}`.
    bad = CaseDescriptor(id="gate", cases=[{"when": "w"}])  # no then, no else
    descs = {"gate": bad, "c": _code("c", v="${gate.output}")}
    with pytest.raises(LoadError, match="no resolvable branch value"):
        expand_case_outputs(descs, None, [])


def test_no_case_fast_path_is_identity():
    descs = {"c": _code("c", v="${x.output}")}
    out = {"r": "${c.output}"}
    new, new_out = expand_case_outputs(descs, out, [])
    assert new is descs and new_out is out  # untouched (no case node)


def test_case_ref_in_coalesce_is_loud():
    descs = {"gate": _gate(), "c": _code("c", v="${gate.output | other.output}")}
    with pytest.raises(LoadError, match="a case value can't be read in"):
        expand_case_outputs(descs, None, [])


def test_case_ref_in_default_is_loud():
    descs = {"gate": _gate(), "c": _code("c", v="${gate.output:-fallback}")}
    with pytest.raises(LoadError, match="a case value can't be read in"):
        expand_case_outputs(descs, None, [])


def test_case_ref_in_nested_default_value_is_loud():
    # ${x:-${gate.output}} — a case ref as the nested-default VALUE (inside an enclosing
    # span) must be LOUD, not silently expanded (a regression caught in review).
    descs = {"gate": _gate(), "c": _code("c", v="${x:-${gate.output}}")}
    with pytest.raises(LoadError, match="a case value can't be read in"):
        expand_case_outputs(descs, None, [])


def test_case_ref_in_required_message_is_loud():
    # ${input.t:?${gate.output}} — the :? message is a full expression,
    # so a case ref in it is a genuine ref and must be LOUD (parallel to the :- default).
    descs = {"gate": _gate(), "c": _code("c", v="${input.t:?${gate.output}}")}
    with pytest.raises(LoadError, match="a case value can't be read in"):
        expand_case_outputs(descs, None, [])


def test_case_ref_in_expression_is_loud():
    descs = {"gate": _gate(), "c": _code("c", v="${gate.output + 1}")}
    with pytest.raises(LoadError, match=r"can't be read in .*an expression") as info:
        expand_case_outputs(descs, None, [])
    assert "whole single reference" not in str(info.value)
    assert "call(...)" not in str(info.value)


def test_case_ref_in_assert_is_located():
    # The assert-position rejection locates at the `asserts:` line, not the `output:` line.
    descs = {"gate": _gate()}
    diagnostics = LoadDiagnostics(label="flow.yaml", text=(" " * 80 + "\n") * 50)
    assertion_range = SourceRange(42, 1, 42, 1)
    with pytest.raises(LoadError) as exc:
        expand_case_outputs(
            descs,
            None,
            ['${gate.output} != ""'],
            assertion_range_at=lambda _index: assertion_range,
            diagnostics=diagnostics,
        )
    assert exc.value.diagnostic.frames[0].range.start_line == 42


def test_case_ref_in_condition_is_loud():
    # another case whose when: reads ${gate.output} (a case value) -> rejected.
    other = CaseDescriptor(
        id="other", cases=[{"when": "${gate.output} == 1", "then": "z"}]
    )
    descs = {"gate": _gate(), "other": other}
    with pytest.raises(LoadError, match="condition"):
        expand_case_outputs(descs, None, [])


def test_case_ref_in_assert_is_loud():
    descs = {"gate": _gate()}
    with pytest.raises(LoadError, match="assert"):
        expand_case_outputs(descs, None, ['${gate.output} != ""'])


def test_case_ref_in_bare_condition_is_loud():
    # The bare condition spelling reads the case value too, so it is rejected the same way.
    other = CaseDescriptor(id="other", cases=[{"when": "gate.output == 1", "then": "z"}])
    descs = {"gate": _gate(), "other": other}
    with pytest.raises(LoadError, match="case node 'other' condition"):
        expand_case_outputs(descs, None, [])


def test_case_ref_in_whole_span_condition_is_loud():
    other = CaseDescriptor(id="other", cases=[{"when": "${gate.output == 1}", "then": "z"}])
    descs = {"gate": _gate(), "other": other}
    with pytest.raises(LoadError, match="case node 'other' condition"):
        expand_case_outputs(descs, None, [])


def test_case_ref_in_bare_assert_is_loud():
    descs = {"gate": _gate()}
    with pytest.raises(LoadError, match="assert"):
        expand_case_outputs(descs, None, ['gate.output != ""'])


def test_on_form_match_value_naming_a_case_is_not_a_read():
    # An `on:` case's `when:` is a match value (text), not a condition.
    other = CaseDescriptor(
        id="other", on="${s.output}", cases=[{"when": "gate.output", "then": "z"}]
    )
    descs = {"gate": _gate(), "other": other}
    expand_case_outputs(descs, None, [])


# --------------------------------------------------------------------------- #
# End-to-end load/run coverage with CODE branches (Ollama-free)
# (branches read ${input.X} only -> no data edges, so skip-flood is correct).
# --------------------------------------------------------------------------- #

_CASE_VALUE_FLOW = """
id: cv
name: case_value
input:
  seed: float
  topic: str
nodes:
  s:
    kind: code
    code: tests.engine._compose_codefns:score
    input:
      seed: ${input.seed}
    output: float
  gate:
    kind: case
    cases:
      - when: "${s.output} >= 0.5"
        then: pro
    else: caut
  pro:
    kind: code
    code: tests.engine._compose_codefns:positive
    input:
      topic: ${input.topic}
    output: str
  caut:
    kind: code
    code: tests.engine._compose_codefns:cautious
    input:
      topic: ${input.topic}
    output: str
output: ${gate.output}
"""


def test_case_value_flow_output_is_expanded():
    loaded = load_flow(_CASE_VALUE_FLOW)
    # `output: ${gate.output}` desugared to the branch coalesce (= seed 02's hand form).
    assert loaded.compiled.outputs[0].from_ == "${pro.output | caut.output}"


def test_case_value_returns_taken_branch():
    loaded = load_flow(_CASE_VALUE_FLOW)
    hit = run_flow(loaded, {"seed": 0.9, "topic": "ACME"})
    assert hit.status == "succeeded"
    assert hit.output == "pro case for ACME"  # the THEN branch ran
    miss = run_flow(loaded, {"seed": 0.1, "topic": "ACME"})
    assert miss.status == "succeeded"
    assert miss.output == "cautious note for ACME"  # the ELSE branch ran


_CASE_VALUE_CONSUMER = """
id: cvc
name: case_value_consumer
input:
  seed: float
  topic: str
nodes:
  s:
    kind: code
    code: tests.engine._compose_codefns:score
    input:
      seed: ${input.seed}
    output: float
  gate:
    kind: case
    cases:
      - when: "${s.output} >= 0.5"
        then: pro
    else: caut
  pro:
    kind: code
    code: tests.engine._compose_codefns:positive
    input:
      topic: ${input.topic}
    output: str
  caut:
    kind: code
    code: tests.engine._compose_codefns:cautious
    input:
      topic: ${input.topic}
    output: str
  summary:
    kind: code
    code: tests.engine._compose_codefns:echo
    input:
      topic: ${gate.output}
    output: str
output: ${summary.output}
"""


def test_node_consuming_case_value_runs():
    loaded = load_flow(_CASE_VALUE_CONSUMER)
    # summary consumes ${gate.output} -> data edges from BOTH branches; the taken one wins.
    edges = {(e.from_, e.to) for e in loaded.compiled.edges}
    assert ("pro", "summary") in edges and ("caut", "summary") in edges
    result = run_flow(loaded, {"seed": 0.9, "topic": "ACME"})
    assert result.status == "succeeded"
    assert result.output == "pro case for ACME"


def test_seed_22_case_value_output_expanded():
    from pathlib import Path

    seeds = Path(__file__).resolve().parents[2] / "tests" / "seeds"
    loaded = load_flow((seeds / "22-case-value.yaml").read_text())
    # Seed 22's `output: ${gate.output}` desugars to seed 02's hand-written join.
    assert loaded.compiled.outputs[0].from_ == "${positive.output | cautious.output}"


_CASE_OF_CASE_FLOW = """
id: coc
name: coc
input:
  a: float
  b: float
nodes:
  outer:
    kind: case
    cases:
      - when: "${input.a} >= 0.5"
        then: inner
    else: x
  inner:
    kind: case
    cases:
      - when: "${input.b} >= 0.5"
        then: hi
    else: lo
  hi:
    kind: code
    code: tests.engine._compose_codefns:echo
    input:
      topic: "HI"
    output: str
  lo:
    kind: code
    code: tests.engine._compose_codefns:echo
    input:
      topic: "LO"
    output: str
  x:
    kind: code
    code: tests.engine._compose_codefns:echo
    input:
      topic: "X"
    output: str
output: ${outer.output}
"""


def test_case_of_case_value_runs_taken_nested_leaf():
    loaded = load_flow(_CASE_OF_CASE_FLOW)
    # ${outer.output} flattened through `inner` to its leaves (+ outer's else `x`).
    assert loaded.compiled.outputs[0].from_ == "${hi.output | lo.output | x.output}"
    assert run_flow(loaded, {"a": 0.9, "b": 0.9}).output == "HI"  # outer->inner->hi
    assert run_flow(loaded, {"a": 0.9, "b": 0.2}).output == "LO"  # outer->inner->lo
    assert (
        run_flow(loaded, {"a": 0.2, "b": 0.9}).output == "X"
    )  # outer->x (inner skip-flooded)


_INLINE_INTO_CASE = """
id: cvi
name: case_value_into_inline
input:
  seed: float
  topic: str
defs:
  passthru:
    input:
      v: str
    nodes:
      x:
        kind: code
        code: tests.engine._compose_codefns:echo
        input:
          topic: ${input.v}
        output: str
    output: ${x.output}
nodes:
  s:
    kind: code
    code: tests.engine._compose_codefns:score
    input:
      seed: ${input.seed}
    output: float
  gate:
    kind: case
    cases:
      - when: "${s.output} >= 0.5"
        then: pro
    else: caut
  pro:
    kind: code
    code: tests.engine._compose_codefns:positive
    input:
      topic: ${input.topic}
    output: str
  caut:
    kind: code
    code: tests.engine._compose_codefns:cautious
    input:
      topic: ${input.topic}
    output: str
  wrap:
    kind: code
    code: tests.engine._compose_codefns:echo
    input:
      topic: call(passthru, v=${gate.output})
    output: str
output: ${wrap.output}
"""


def test_case_value_in_inline_call_arg_runs():
    # the desugar ordering contract: an inline call arg reading a case value
    # (call(passthru, v=${gate.output})) desugars to a synth call whose arg is then
    # expanded to the branch coalesce. If expansion missed the synth node, ${gate.output}
    # would resolve to null (a case writes no value) and the run would be wrong.
    loaded = load_flow(_INLINE_INTO_CASE)
    assert any(nid.startswith("__call_") for nid in loaded.compiled.nodes)
    result = run_flow(loaded, {"seed": 0.9, "topic": "ACME"})
    assert result.status == "succeeded"
    assert result.output == "pro case for ACME"  # the taken branch value flowed through


def test_case_value_in_coalesce_position_rejected_e2e():
    text = """
id: cv_bad
name: cv_bad
input:
  seed: float
nodes:
  s:
    kind: code
    code: tests.engine._compose_codefns:score
    input:
      seed: ${input.seed}
    output: float
  gate:
    kind: case
    cases:
      - when: "${s.output} >= 0.5"
        then: a
    else: b
  a:
    kind: code
    code: tests.engine._compose_codefns:echo
    input:
      seed: ${input.seed}
    output: float
  b:
    kind: code
    code: tests.engine._compose_codefns:echo
    input:
      seed: ${input.seed}
    output: float
output: ${gate.output | s.output}
"""
    with pytest.raises(LoadError, match="a case value can't be read in"):
        load_flow(text)
