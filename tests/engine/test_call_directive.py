"""The whole-value `call(<flow-id>, kw=...)` compile-time directive.

An unquoted value whose trimmed text starts with `call(` must be exactly one directive
`call( … )`, else it is a load error naming the problem; a quoted value is text. The first
arg is the POSITIONAL flow id (bare, may be hyphenated), the rest are `name=value`
keywords. It desugars — through the SAME
`desugar_inline_calls` traversal + SHARED minter — into a synth `CallDescriptor(over=None)`,
the host binding rewritten to `${<synth>.output}`. Nesting is legal and desugars inner-first.

These tests drive the pass directly (rather than via a full load), so recognition/rewrite
is asserted on the returned descriptor map with a deterministic minter.
"""

import pytest

from yian.compose import LoadError, desugar_inline_calls, load_flow
from yian.compose.desugar.inline_bodies import INLINE_FLOW_PREFIX
from yian.compose.helpers.errors import LoadDiagnostics
from yian.compose.parse.file import CodeDescriptor
from yian.diagnostics import OriginRef, SourceRange
from yian.nodes.contracts import Assertion
from yian.run import run_flow


def _ids():
    """A deterministic synth-id minter (`__call_0`, `__call_1`, …), matching the
    loader's minter shape."""
    counter = iter(range(1000))
    return lambda: f"__call_{next(counter)}"


def _code(nid, inputs):
    """A minimal CodeDescriptor whose internal `inputs` carry the binding under test."""
    return CodeDescriptor(id=nid, code="mod:fn", inputs=inputs, outputs="str")


def test_whole_value_call_directive_recognized():
    descriptors = {
        "use": _code("use", {"topic": "call(summarize, messages=${messages})"})
    }
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


@pytest.mark.parametrize(
    "value, problem",
    [
        ("call(f) tail", "text follows the closing `)`: 'tail'"),
        ("call(f, x=1", "`call(` is never closed"),
        ('call(f, x="a)', "`call(` is never closed"),
        ("call(f x=1)", "must start with a flow id (a bare name) and a comma, got 'f x=1'"),
        ("call(f.g)", "must start with a flow id (a bare name) and a comma, got 'f.g'"),
        ("call()", "must start with a flow id (a bare name) and a comma, got ''"),
        ("call(f, 30)", "passes a positional argument '30'"),
        ("call(f, a=1 b=2)", "keyword 'a' value '1 b=2' looks like two keywords"),
        ("call(f, x=call(g) tail)", "text follows the closing `)`: 'tail'"),
        ("call(f, x=call(g, 1))", "call(g, ...) passes a positional argument '1'"),
    ],
)
def test_malformed_directive_is_a_load_error_with_the_quote_escape(value, problem):
    # an unquoted value starting with `call(` must be exactly one well-formed directive.
    descriptors = {"use": _code("use", {"topic": value})}
    with pytest.raises(LoadError) as caught:
        desugar_inline_calls(descriptors, None, next_id=_ids())
    message = str(caught.value)
    assert message.startswith("use: ")
    assert problem in message
    assert "If you meant the text, quote the whole value:" in message


def test_quoted_directive_is_text():
    # a quoted binding is text, so quoting is the escape for a literal `call(...)`.
    descriptors = {
        "use": _code("use", {"topic": "call(f, x=1", "note": "call(f, x=1)"})
    }
    new_descriptors, _, _ = desugar_inline_calls(
        descriptors,
        None,
        binding_style_at=lambda node_id, field, key: "single",
        next_id=_ids(),
    )
    assert new_descriptors["use"].inputs == {"topic": "call(f, x=1", "note": "call(f, x=1)"}
    assert not any(nid.startswith("__call_") for nid in new_descriptors)


def test_quoted_directive_holding_a_span_is_loud():
    # quoting makes a `call(...)` text, which can't be what a span-bearing call meant.
    descriptors = {"use": _code("use", {"topic": "call(f, x=${input.x})"})}
    with pytest.raises(LoadError, match="a quoted `call\\(...\\)` is text"):
        desugar_inline_calls(
            descriptors,
            None,
            binding_style_at=lambda node_id, field, key: "double",
            next_id=_ids(),
        )


def test_quoted_nested_arg_is_text():
    descriptors = {"use": _code("use", {"topic": 'call(f, x="call(g", y=\'call(h)\')'})}
    new_descriptors, _, _ = desugar_inline_calls(descriptors, None, next_id=_ids())
    assert new_descriptors["__call_0"].inputs == {"x": "call(g", "y": "call(h)"}


def test_quoted_flow_output_is_text_and_unquoted_is_strict():
    _, new_outputs, _ = desugar_inline_calls(
        {},
        {"a": "call(f", "b": "call(g)"},
        output_style_at=lambda key: "double" if key == "a" else None,
        next_id=_ids(),
    )
    assert new_outputs == {"a": "call(f", "b": "${__call_0.output}"}
    with pytest.raises(LoadError, match="flow outputs: `call\\(` is never closed"):
        desugar_inline_calls({}, "call(f", next_id=_ids())


def test_malformed_directive_in_assert_is_loud_without_a_quote_hint():
    # an assert is an expression; quoting it does not make it text, so no quote hint.
    with pytest.raises(LoadError) as caught:
        desugar_inline_calls({}, None, asserts_section=["call(f) tail"], next_id=_ids())
    message = str(caught.value)
    assert "assert: text follows the closing `)`: 'tail'" in message
    assert "quote" not in message


def test_bare_id_with_hyphen_recognized():
    descriptors = {"use": _code("use", {"topic": "call(my-flow, topic=${input.t})"})}
    new_descriptors, _, _ = desugar_inline_calls(descriptors, None, next_id=_ids())
    assert new_descriptors["use"].inputs["topic"] == "${__call_0.output}"
    assert new_descriptors["__call_0"].call == "my-flow"


def test_positional_arg_after_flow_id_is_loud():
    descriptors = {"use": _code("use", {"topic": "call(f, 30)"})}
    with pytest.raises(LoadError, match="keyword"):
        desugar_inline_calls(descriptors, None, next_id=_ids())


def test_repeated_keyword_is_loud():
    descriptors = {"use": _code("use", {"topic": "call(f, a=1, a=2)"})}
    with pytest.raises(LoadError, match="'a' more than once"):
        desugar_inline_calls(descriptors, None, next_id=_ids())


def test_repeated_keyword_in_nested_directive_is_loud():
    descriptors = {"use": _code("use", {"topic": "call(f, x=call(g, b=1, b=2))"})}
    with pytest.raises(LoadError, match="'b' more than once"):
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


def test_outer_private_callee_is_rejected_at_binding_before_resolution(monkeypatch):
    private_name = INLINE_FLOW_PREFIX + "d" * 64
    text = f"""id: private_outer
name: private_outer
nodes:
  use:
    kind: code
    code: tests.engine._compose_codefns:echo
    input:
      topic: call({private_name})
    output: str
output: ${{use.output}}
"""
    resolved = []

    def resolve(*args, **kwargs):
        resolved.append((args, kwargs))
        raise AssertionError("the private callee reached child resolution")

    monkeypatch.setattr("yian.compose.nodes.call._resolve_child", resolve)

    with pytest.raises(LoadError) as caught:
        load_flow(text)

    expected_line = next(
        number
        for number, source_line in enumerate(text.splitlines(), start=1)
        if private_name in source_line
    )
    assert caught.value.diagnostic.frames[0].range.start_line == expected_line
    assert caught.value.diagnostic.frames[0].text == text
    assert INLINE_FLOW_PREFIX in str(caught.value)
    assert private_name not in str(caught.value)
    assert resolved == []


def test_nested_private_callee_is_rejected_at_binding_before_resolution(monkeypatch):
    private_name = INLINE_FLOW_PREFIX + "a" * 64
    text = f"""id: private_nested
name: private_nested
nodes:
  use:
    kind: code
    code: tests.engine._compose_codefns:echo
    input:
      topic: call(outer, x=call({private_name}))
    output: str
output: ${{use.output}}
"""
    resolved = []

    def resolve(*args, **kwargs):
        resolved.append((args, kwargs))
        raise AssertionError("the private callee reached child resolution")

    monkeypatch.setattr("yian.compose.nodes.call._resolve_child", resolve)

    with pytest.raises(LoadError) as caught:
        load_flow(text)

    expected_line = next(
        number
        for number, source_line in enumerate(text.splitlines(), start=1)
        if private_name in source_line
    )
    assert caught.value.diagnostic.frames[0].range.start_line == expected_line
    assert caught.value.diagnostic.frames[0].text == text
    assert INLINE_FLOW_PREFIX in str(caught.value)
    assert private_name not in str(caught.value)
    assert resolved == []


@pytest.mark.parametrize(
    ("section", "expected_line"),
    [("output", 17), ("assert", 23)],
)
def test_private_callee_in_flow_section_preserves_section_line(section, expected_line):
    private_name = INLINE_FLOW_PREFIX + "b" * 64
    expected_range = SourceRange(expected_line, 1, expected_line, 1)
    kwargs = (
        {"outputs_section": f"call({private_name})", "outputs_range": expected_range}
        if section == "output"
        else {
            "outputs_section": None,
            "asserts_section": [f"call({private_name})"],
            "assertion_range_at": lambda _index: expected_range,
        }
    )

    diagnostics = LoadDiagnostics(
        label="flow.yaml",
        text=(" " * 80 + "\n") * 30,
    )
    with pytest.raises(LoadError) as caught:
        desugar_inline_calls(
            {},
            next_id=_ids(),
            diagnostics=diagnostics,
            **kwargs,
        )

    assert caught.value.diagnostic.frames[0].range.start_line == expected_line
    assert caught.value.diagnostic.frames[0].label == "flow.yaml"
    assert INLINE_FLOW_PREFIX in str(caught.value)
    assert private_name not in str(caught.value)


@pytest.mark.parametrize("callee", ["__inline_flow", "x__inline_flow_"])
def test_private_callee_guard_matches_only_the_exact_prefix(callee):
    descriptors = {"use": _code("use", {"topic": f"call({callee})"})}

    new_descriptors, _, _ = desugar_inline_calls(descriptors, None, next_id=_ids())

    assert new_descriptors["__call_0"].call == callee


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
    old_only = {"old": _code("old", {"topic": "${ enrich(topic=input.topic) }"})}
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
    descriptors = {
        "use": _code("use", {"topic": 'call(f, n=30, flag=true, z=null, s="hi")'})
    }
    new_descriptors, _, _ = desugar_inline_calls(descriptors, None, next_id=_ids())
    synth = new_descriptors["__call_0"]
    assert synth.inputs == {"n": 30, "flag": True, "z": None, "s": "hi"}


def test_bare_bool_word_arg_stays_string():
    # deliberate: inline bare literals are a YAML-1.1 SUBSET — `yes`/`on`/`off`/`no` stay
    # strings (no boolean coercion), avoiding the YAML bool footgun. (A named call's YAML
    # authored `input:` map would coerce `yes` -> True; the directive form intentionally does not.)
    descriptors = {"use": _code("use", {"topic": "call(f, flag=yes)"})}
    new_descriptors, _, _ = desugar_inline_calls(descriptors, None, next_id=_ids())
    assert new_descriptors["__call_0"].inputs == {"flag": "yes"}


def test_directive_in_outputs_and_asserts_is_desugared():
    # a directive placed in the flow `output:` map and in a flow-level assert both
    # desugar on the shared traversal.
    new_descriptors, new_outputs, new_asserts = desugar_inline_calls(
        {},
        {"result": "call(summarize, messages=${messages})"},
        asserts_section=["call(score, topic=${input.t})"],
        next_id=_ids(),
    )
    assert new_outputs["result"].startswith("${__call_")
    assert new_asserts[0].expression.startswith("${__call_")
    assert new_asserts[0].display == "call(score, topic=${input.t})"
    synth = {d.call for nid, d in new_descriptors.items() if nid.startswith("__call_")}
    assert synth == {"summarize", "score"}


def test_directive_rewrite_preserves_existing_assertion_presentation():
    """Carrierized flow assertions retain presentation while executable text changes."""
    authored = Assertion(
        "call(score, topic=${input.t})",
        "score(topic) must pass",
        OriginRef("asserts/0"),
    )

    _, _, assertions = desugar_inline_calls(
        {}, None, asserts_section=[authored], next_id=_ids()
    )

    assert assertions[0].expression.startswith("${__call_")
    assert assertions[0].display == authored.display
    assert assertions[0].origin is authored.origin


def test_raw_duplicate_flow_assertions_become_distinct_ordered_carriers():
    """The front-end boundary retains occurrence identity even for equal YAML text."""
    text = "input.score > 0"

    _, _, assertions = desugar_inline_calls(
        {}, None, asserts_section=[text, text], next_id=_ids()
    )

    assert assertions == [Assertion(text, text), Assertion(text, text)]
    assert assertions[0] is not assertions[1]


# --------------------------------------------------------------------------- #
# The hard-break: a FLOW call embedded inside a `${...}` span (the retired
# `${flow(args)}` form) is a LoadError pointing at the call(...) directive.
# A pure builtin inside `${...}` stays legal.
# --------------------------------------------------------------------------- #


def test_whole_value_flow_call_in_span_is_loud():
    descriptors = {"use": _code("use", {"topic": "${summarize(messages=messages)}"})}
    with pytest.raises(LoadError, match="call\\("):
        desugar_inline_calls(descriptors, None, next_id=_ids())


def test_embedded_flow_call_in_comparison_is_loud():
    descriptors = {"use": _code("use", {"topic": "${score_one(t=input.t)} >= -1"})}
    with pytest.raises(LoadError, match="call\\("):
        desugar_inline_calls(descriptors, None, next_id=_ids())


def test_embedded_flow_call_in_text_is_loud():
    descriptors = {"use": _code("use", {"topic": "pe=${relevance(t=input.t)}"})}
    with pytest.raises(LoadError, match="call\\("):
        desugar_inline_calls(descriptors, None, next_id=_ids())


def test_coalesce_of_flow_calls_in_span_is_loud():
    descriptors = {"use": _code("use", {"topic": "${a(x=1) | b(y=2)}"})}
    with pytest.raises(LoadError, match="call\\("):
        desugar_inline_calls(descriptors, None, next_id=_ids())


def test_flow_call_in_outputs_span_is_loud():
    with pytest.raises(LoadError, match="call\\("):
        desugar_inline_calls({}, {"result": "${summarize(m=m)}"}, next_id=_ids())


def test_pure_builtin_in_span_still_allowed():
    # `upper` is a TEMPLATE_FNS builtin — a call to it inside `${...}` is NOT a flow call,
    # so it passes through untouched (no synth node, no raise).
    descriptors = {"use": _code("use", {"topic": "${upper(input.t)}"})}
    new_descriptors, _, _ = desugar_inline_calls(descriptors, None, next_id=_ids())
    assert new_descriptors["use"].inputs["topic"] == "${upper(input.t)}"
    assert not any(nid.startswith("__call_") for nid in new_descriptors)


def _echo_flow(binding: str, output: str = "${use.output}") -> str:
    """A flow whose CODE node `use` echoes its `topic` binding."""
    return f"""id: echo
name: echo
nodes:
  use:
    kind: code
    code: tests.engine._compose_codefns:echo
    input:
      topic: {binding}
    output: str
output: {output}
"""


@pytest.mark.parametrize(
    "binding, text",
    [
        ("'call(f, x=1'", "call(f, x=1"),
        ('"call(f) tail"', "call(f) tail"),
        ('"call(f, x=1)"', "call(f, x=1)"),
        ("|\n        call(f", "call(f\n"),
    ],
)
def test_quoted_binding_runs_as_text(binding, text):
    out = run_flow(load_flow(_echo_flow(binding)), {})
    assert out.status == "succeeded"
    assert out.output == text


@pytest.mark.parametrize(
    "binding, problem",
    [
        ("call(f, x=1", "`call(` is never closed"),
        ("call(f) tail", "text follows the closing `)`: 'tail'"),
    ],
)
def test_unquoted_malformed_binding_is_a_located_load_error(binding, problem):
    text = _echo_flow(binding)
    with pytest.raises(LoadError) as caught:
        load_flow(text)
    message = str(caught.value)
    assert f"use: {problem}" in message
    assert "If you meant the text, quote the whole value: `\"call(f" in message
    assert caught.value.diagnostic.frames[-1].range.start_line == 8


def test_quoted_flow_output_runs_as_text():
    out = run_flow(load_flow(_echo_flow("hi", output="'call(f'")), {})
    assert out.status == "succeeded"
    assert out.output == "call(f"


def test_unquoted_malformed_flow_output_is_loud():
    with pytest.raises(LoadError, match="flow outputs: text follows the closing"):
        load_flow(_echo_flow("hi", output="call(f) tail"))


def test_unquoted_malformed_map_over_is_loud():
    text = """id: m
name: m
nodes:
  many:
    kind: map
    call: f
    over: call(g
    input:
      topic: ${item}
output: ${many.output}
"""
    with pytest.raises(LoadError, match="many: `call\\(` is never closed") as caught:
        load_flow(text)
    assert caught.value.diagnostic.frames[-1].range.start_line == 7


def test_unquoted_directive_holding_a_separator_is_a_call_not_a_typed_binding():
    text = """id: spaced
name: spaced
defs:
  f:
    input:
      topic: str
    nodes:
      e:
        kind: code
        code: tests.engine._compose_codefns:echo
        input:
          topic: ${input.topic}
        output: str
    output: ${e.output}
nodes:
  use:
    kind: code
    code: tests.engine._compose_codefns:echo
    input:
      topic: call(f, topic = "a = b")
    output: str
output: ${use.output}
"""
    out = run_flow(load_flow(text), {})
    assert out.status == "succeeded"
    assert out.output == "a = b"


@pytest.mark.parametrize(
    "value, expected",
    [
        ("call(f, t=${input.x} == X)", "${input.x} == X"),
        ("call(f, t=a>=b, u=a<=b, v=a!=b)", "a>=b"),
        ("call(f, t=a=b)", "a=b"),
        ("call(f, note=a = b)", "a = b"),
    ],
)
def test_comparison_and_glued_equals_in_a_keyword_value_stay_text(value, expected):
    descriptors = {"use": _code("use", {"topic": value})}
    new_descriptors, _, _ = desugar_inline_calls(descriptors, None, next_id=_ids())
    assert new_descriptors["__call_0"].inputs["t" if "t=" in value else "note"] == expected


@pytest.mark.parametrize(
    "value, problem",
    [
        ("call(f, a=${x} b=${y})", "keyword 'a' value '${x} b=${y}' looks like two keywords"),
        ("call(f, =1)", "argument '=1' needs one name before `=`"),
        ("call(f, a b=1)", "argument 'a b=1' needs one name before `=`"),
    ],
)
def test_missing_comma_or_name_in_keywords_is_loud(value, problem):
    descriptors = {"use": _code("use", {"topic": value})}
    with pytest.raises(LoadError) as caught:
        desugar_inline_calls(descriptors, None, next_id=_ids())
    assert problem in str(caught.value)


def test_no_quote_hint_when_quoting_would_not_give_text():
    # quoting a span-bearing call is its own error, and quoting a typed binding drops the type.
    span = {"use": _code("use", {"topic": "call(f, topic=${input.x}"})}
    with pytest.raises(LoadError) as caught:
        desugar_inline_calls(span, None, next_id=_ids())
    assert "If you meant the text" not in str(caught.value)
    with pytest.raises(LoadError) as caught:
        load_flow(_echo_flow("str = call(f"))
    assert "If you meant the text" not in str(caught.value)
    with pytest.raises(LoadError) as caught:
        load_flow(_echo_flow("hi", output="str = call(f"))
    assert "If you meant the text" not in str(caught.value)
