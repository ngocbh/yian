"""Load-time checks for reading a field a value's known type can't have."""

import pytest

from yian.compose import load_flow
from yian.compose.helpers.errors import LoadError
from yian.run import run_flow


def _flow(step_output: str, rest: str, *, typedefs: str = "") -> str:
    """A flow with one CODE node `step` typed `step_output`, plus `rest` (YAML tail)."""
    return (
        "id: p\nname: p\n"
        + typedefs
        + "input:\n  s: str\n"
        "nodes:\n"
        "  step:\n"
        "    kind: code\n"
        "    input:\n"
        "      x: ${input.s}\n"
        f"    output: {step_output}\n"
        "    code: return None\n"
        + rest
    )


def _load_error(text: str) -> str:
    with pytest.raises(LoadError) as exc:
        load_flow(text)
    return str(exc.value)


def test_optional_record_field_reads_null_when_the_whole_value_is_null():
    text = _flow(
        "Optional[Rec]",
        "output:\n  a: ${step.output.a}\n  note: 'a=${step.output.a}'\n",
        typedefs="typedefs:\n  Rec: {a: str}\n",
    )
    result = run_flow(load_flow(text), {"s": "x"})
    assert result.status == "succeeded"
    assert result.output == {"a": None, "note": "a="}


def test_case_when_on_a_missing_optional_record_input_is_falsy():
    """A `when:` reading a field of an omitted `Optional[Rec]` input takes the else branch."""
    text = (
        "id: p\nname: p\n"
        "typedefs:\n  Rec: {a: str}\n"
        "input:\n  s: str\n  maybe: Optional[Rec]\n"
        "nodes:\n"
        "  route:\n"
        "    kind: case\n"
        "    cases:\n"
        "      - when: ${input.maybe.a == 'x'}\n"
        "        then: hit\n"
        "    else: miss\n"
        "  hit:\n"
        "    kind: code\n"
        "    input:\n      s: ${input.s}\n"
        "    output: str\n"
        "    code: return 'hit'\n"
        "  miss:\n"
        "    kind: code\n"
        "    input:\n      s: ${input.s}\n"
        "    output: str\n"
        "    code: return 'miss'\n"
        "output: ${hit.output | miss.output}\n"
    )
    flow = load_flow(text)
    assert run_flow(flow, {"s": "x"}).output == "miss"
    assert run_flow(flow, {"s": "x", "maybe": {"a": "x"}}).output == "hit"
