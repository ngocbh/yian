"""Co-skip classifier: does a binding co-skip when all its producers are skipped?"""

from yian.expr import binding_co_skips


def test_plain_ref_co_skips():
    assert binding_co_skips("${a.output}") is True


def test_ref_coalesce_co_skips():
    assert binding_co_skips("${a.output | b.output}") is True


def test_ref_default_co_skips():
    assert binding_co_skips("${a.output:-${b.output}}") is True  # default is a ref


def test_literal_default_does_not_co_skip():
    assert binding_co_skips("${a.output:-null}") is False


def test_literal_coalesce_operand_does_not_co_skip():
    assert binding_co_skips('${a.output | "fallback"}') is False


def test_required_does_not_co_skip():
    assert binding_co_skips("${a.output:?missing}") is False


def test_embedded_text_does_not_co_skip():
    assert binding_co_skips("got ${a.output} here") is False


def test_non_string_and_non_ref_do_not_co_skip():
    assert binding_co_skips(30) is False
    assert binding_co_skips("plain literal") is False
    assert binding_co_skips("${input.x}") is True  # whole-string ref still co-skips structurally


def test_arithmetic_span_does_not_co_skip():
    # a computed span (not a pure ref group) is not a hard data dependency.
    assert binding_co_skips("${a.output + 1}") is False


def test_builtin_call_span_does_not_co_skip():
    # a builtin-call span is a computed value, not a pure ref, so it does not co-skip.
    assert binding_co_skips("${upper(a.output)}") is False


def test_bare_ref_default_co_skips():
    # `:-b.output` (a bare-ref default) keeps co-skipping, like `:-${b.output}`.
    assert binding_co_skips("${a.output:-b.output}") is True
