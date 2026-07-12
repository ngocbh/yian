"""The whole-value `call(<flow-id>, kw=...)` compile-time directive.

The loader recognizes a whole-value `call(...)` directive ALONGSIDE the legacy inline
`${flow(args)}` form — both coexist. A directive is recognized IFF the WHOLE trimmed
field value is exactly `call( … )`: the first arg is the POSITIONAL flow id (bare, may
be hyphenated), the rest are `name=value` keywords. It desugars — through the SAME
`desugar_inline_calls` traversal + SHARED minter — into a synth `CallDescriptor(over=None)`,
the host binding rewritten to `${<synth>.output}`. Nesting is legal and desugars inner-first.

These tests drive the pass directly (rather than via a full load), so recognition/rewrite
is asserted on the returned descriptor map with a deterministic minter.
"""

import pytest

from agent_composer.compose import LoadError, desugar_inline_calls
from agent_composer.compose.parse.file import CodeDescriptor


def _ids():
    """A deterministic synth-id minter (`__call_0`, `__call_1`, …), matching the
    loader's minter shape."""
    counter = iter(range(1000))
    return lambda: f"__call_{next(counter)}"


def _code(nid, inputs):
    """A minimal CodeDescriptor whose `inputs:` carry the binding under test."""
    return CodeDescriptor(id=nid, code="mod:fn", inputs=inputs, outputs="str")


def test_whole_value_call_directive_recognized():
    descriptors = {"use": _code("use", {"topic": "call(summarize, messages=${messages})"})}
    new_descriptors, _, _ = desugar_inline_calls(descriptors, None, next_id=_ids())
    assert new_descriptors["use"].inputs["topic"] == "${__call_0.output}"
    synth = new_descriptors["__call_0"]
    assert synth.call == "summarize"
    assert synth.inputs == {"messages": "${messages}"}
    assert synth.over is None


def test_mid_value_call_is_literal_not_directive():
    descriptors = {"use": _code("use", {"topic": "pre call(x)"})}
    new_descriptors, _, _ = desugar_inline_calls(descriptors, None, next_id=_ids())
    assert new_descriptors["use"].inputs["topic"] == "pre call(x)"
    assert not any(nid.startswith("__call_") for nid in new_descriptors)


def test_trailing_content_after_close_paren_is_literal():
    # not whole-value (content after the close paren) -> left literal, no synth node.
    descriptors = {"use": _code("use", {"topic": "call(f) tail"})}
    new_descriptors, _, _ = desugar_inline_calls(descriptors, None, next_id=_ids())
    assert new_descriptors["use"].inputs["topic"] == "call(f) tail"
    assert not any(nid.startswith("__call_") for nid in new_descriptors)


def test_bare_id_with_hyphen_recognized():
    descriptors = {"use": _code("use", {"topic": "call(my-flow, topic=${input.t})"})}
    new_descriptors, _, _ = desugar_inline_calls(descriptors, None, next_id=_ids())
    assert new_descriptors["use"].inputs["topic"] == "${__call_0.output}"
    assert new_descriptors["__call_0"].call == "my-flow"


def test_positional_arg_after_flow_id_is_loud():
    descriptors = {"use": _code("use", {"topic": "call(f, 30)"})}
    with pytest.raises(LoadError, match="keyword"):
        desugar_inline_calls(descriptors, None, next_id=_ids())


def test_directive_capturing_item_is_loud():
    descriptors = {"use": _code("use", {"topic": "call(wrap, t=${item})"})}
    with pytest.raises(LoadError) as exc:
        desugar_inline_calls(descriptors, None, next_id=_ids())
    assert "item" in str(exc.value)


def test_nested_directive_desugars_inner_first():
    descriptors = {"use": _code("use", {"topic": "call(a, x=call(b))"})}
    new_descriptors, _, _ = desugar_inline_calls(descriptors, None, next_id=_ids())
    # inner `b` is minted first (inner-first), then outer `a`.
    inner = new_descriptors["__call_0"]
    outer = new_descriptors["__call_1"]
    assert inner.call == "b"
    assert outer.call == "a"
    assert outer.inputs["x"] == "${__call_0.output}"
    assert new_descriptors["use"].inputs["topic"] == "${__call_1.output}"


def test_shared_minter_yields_distinct_ids_across_bindings():
    descriptors = {
        "a": _code("a", {"topic": "call(f, x=${input.x})"}),
        "b": _code("b", {"topic": "call(g, y=${input.y})"}),
    }
    new_descriptors, _, _ = desugar_inline_calls(descriptors, None, next_id=_ids())
    synth_ids = {nid for nid in new_descriptors if nid.startswith("__call_")}
    assert synth_ids == {"__call_0", "__call_1"}  # distinct, no collision


def test_old_form_rejected_new_form_desugars():
    # the OLD-form `${flow(args)}` is now a hard-break LoadError; the NEW-form `call(...)`
    # desugars via the directive path.
    old_only = {"old": _code("old", {"topic": "${ enrich(topic=${input.topic}) }"})}
    with pytest.raises(LoadError, match="call\\("):
        desugar_inline_calls(old_only, None, next_id=_ids())
    new_only = {"new": _code("new", {"topic": "call(summarize, messages=${messages})"})}
    new_descriptors, _, _ = desugar_inline_calls(new_only, None, next_id=_ids())
    assert new_descriptors["new"].inputs["topic"].startswith("${__call_")
    synth = {d.call for nid, d in new_descriptors.items() if nid.startswith("__call_")}
    assert synth == {"summarize"}


def test_no_arg_directive_recognized():
    descriptors = {"use": _code("use", {"topic": "call(now)"})}
    new_descriptors, _, _ = desugar_inline_calls(descriptors, None, next_id=_ids())
    assert new_descriptors["use"].inputs["topic"] == "${__call_0.output}"
    synth = new_descriptors["__call_0"]
    assert synth.call == "now" and synth.inputs == {}


def test_keyword_literal_values_coerce_like_named_form():
    # a bare literal kwarg binds its typed value (int/bool/null), a quoted scalar unwraps.
    descriptors = {"use": _code("use", {"topic": 'call(f, n=30, flag=true, z=null, s="hi")'})}
    new_descriptors, _, _ = desugar_inline_calls(descriptors, None, next_id=_ids())
    synth = new_descriptors["__call_0"]
    assert synth.inputs == {"n": 30, "flag": True, "z": None, "s": "hi"}


def test_bare_bool_word_arg_stays_string():
    # deliberate: inline bare literals are a YAML-1.1 SUBSET — `yes`/`on`/`off`/`no` stay
    # strings (no boolean coercion), avoiding the YAML bool footgun. (A named call's YAML
    # `inputs:` map WOULD coerce `yes` -> True; the directive form intentionally does not.)
    descriptors = {"use": _code("use", {"topic": "call(f, flag=yes)"})}
    new_descriptors, _, _ = desugar_inline_calls(descriptors, None, next_id=_ids())
    assert new_descriptors["__call_0"].inputs == {"flag": "yes"}


def test_unbalanced_directive_parens_left_literal():
    # a `call(` with no matching close paren is NOT a whole-value directive — left literal.
    descriptors = {"use": _code("use", {"topic": "call(f, x=1"})}
    new_descriptors, _, _ = desugar_inline_calls(descriptors, None, next_id=_ids())
    assert new_descriptors["use"].inputs["topic"] == "call(f, x=1"
    assert not any(nid.startswith("__call_") for nid in new_descriptors)


def test_directive_in_outputs_and_asserts_is_desugared():
    # a directive placed in the flow `outputs:` map and in a flow-level assert both
    # desugar on the shared traversal.
    new_descriptors, new_outputs, new_asserts = desugar_inline_calls(
        {},
        {"result": "call(summarize, messages=${messages})"},
        asserts_section=["call(score, topic=${input.t})"],
        next_id=_ids(),
    )
    assert new_outputs["result"].startswith("${__call_")
    assert new_asserts[0].startswith("${__call_")
    synth = {d.call for nid, d in new_descriptors.items() if nid.startswith("__call_")}
    assert synth == {"summarize", "score"}


# --------------------------------------------------------------------------- #
# The hard-break: a FLOW call embedded inside a `${...}` span (the retired
# `${flow(args)}` form) is a LoadError pointing at the call(...) directive.
# A pure builtin inside `${...}` stays legal.
# --------------------------------------------------------------------------- #


def test_whole_value_flow_call_in_span_is_loud():
    descriptors = {"use": _code("use", {"topic": "${summarize(messages=${messages})}"})}
    with pytest.raises(LoadError, match="call\\("):
        desugar_inline_calls(descriptors, None, next_id=_ids())


def test_embedded_flow_call_in_comparison_is_loud():
    descriptors = {"use": _code("use", {"topic": "${score_one(t=${input.t})} >= -1"})}
    with pytest.raises(LoadError, match="call\\("):
        desugar_inline_calls(descriptors, None, next_id=_ids())


def test_embedded_flow_call_in_text_is_loud():
    descriptors = {"use": _code("use", {"topic": "pe=${relevance(t=${input.t})}"})}
    with pytest.raises(LoadError, match="call\\("):
        desugar_inline_calls(descriptors, None, next_id=_ids())


def test_coalesce_of_flow_calls_in_span_is_loud():
    descriptors = {"use": _code("use", {"topic": "${a(x=1) | b(y=2)}"})}
    with pytest.raises(LoadError, match="call\\("):
        desugar_inline_calls(descriptors, None, next_id=_ids())


def test_flow_call_in_outputs_span_is_loud():
    with pytest.raises(LoadError, match="call\\("):
        desugar_inline_calls({}, {"result": "${summarize(m=${m})}"}, next_id=_ids())


def test_pure_builtin_in_span_still_allowed():
    # `upper` is a TEMPLATE_FNS builtin — a call to it inside `${...}` is NOT a flow call,
    # so it passes through untouched (no synth node, no raise).
    descriptors = {"use": _code("use", {"topic": "${upper(${input.t})}"})}
    new_descriptors, _, _ = desugar_inline_calls(descriptors, None, next_id=_ids())
    assert new_descriptors["use"].inputs["topic"] == "${upper(${input.t})}"
    assert not any(nid.startswith("__call_") for nid in new_descriptors)
