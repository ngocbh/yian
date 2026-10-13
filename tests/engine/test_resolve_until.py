"""Resolve a WAIT ``until`` source at the node-owned binding boundary.

``resolve_until`` accepts raw or carrier-wrapped text, formats an object exposing
``isoformat()`` defensively, and rejects values that are not date/datetime-like.
"""

import pytest

from yian.compile.model import START_ID
from yian.diagnostics import OriginRef
from yian.nodes.binding import BindingError
from yian.nodes.contracts import Binding
from yian.nodes.wait.node import resolve_until
from yian.typesys.pool import VariablePool


def test_literal_iso_passthrough():
    assert resolve_until("2026-01-01T00:00:00", VariablePool()) == "2026-01-01T00:00:00"


def test_ref_iso_passthrough():
    pool = VariablePool()
    pool.set(START_ID, {"t": "2026-06-20T12:00:00"})
    assert resolve_until("${input.t}", pool) == "2026-06-20T12:00:00"


def test_non_date_error_is_origin_neutral_for_raw_and_wrapped_sources():
    """WAIT reports semantic source text without leaking the origin carrier repr."""
    pool = VariablePool()
    pool.set(START_ID, {"n": 5})
    raw_source = "${input.n}"
    expected = (
        "wait `until` ('${input.n}') did not resolve to a date/datetime: 5"
    )

    for source in (raw_source, Binding(raw_source, OriginRef("wait-until"))):
        with pytest.raises(ValueError) as caught:
            resolve_until(source, pool)
        assert str(caught.value) == expected


def test_literal_with_z_suffix_passes_through_unchanged():
    assert resolve_until("2026-01-02T00:00:00Z", VariablePool()) == "2026-01-02T00:00:00Z"


@pytest.mark.parametrize(
    "literal",
    ["20260102", "2026-W01-5", "2026-01-02T00:00:00.1234567"],
    ids=["basic-format", "week-date", "seven-digit-fraction"],
)
def test_literal_outside_the_grammar_names_the_accepted_forms(literal):
    with pytest.raises(BindingError) as caught:
        resolve_until(literal, VariablePool())

    message = str(caught.value)
    assert caught.value.failure_code == "contract.input"
    assert "2026-05-21" in message
    assert "2026-05-21T14:30[:00][Z|+07:00]" in message
    # The value is named once; the parser's message shows the format, not the value.
    assert message.count(literal) == 1


def test_deadline_that_overflows_in_utc_names_the_value():
    literal = "0001-01-01T00:00+01:00"
    with pytest.raises(BindingError) as caught:
        resolve_until(literal, VariablePool())

    assert caught.value.failure_code == "contract.input"
    assert literal in str(caught.value)


@pytest.mark.parametrize(
    ("literal", "reason"),
    [("2026-02-30", "not a valid date"), ("2026-05-21T24:00", "not a valid datetime")],
    ids=["impossible-date", "impossible-datetime"],
)
def test_impossible_literal_names_the_reason(literal, reason):
    with pytest.raises(BindingError) as caught:
        resolve_until(literal, VariablePool())

    assert caught.value.failure_code == "contract.input"
    assert reason in str(caught.value)
