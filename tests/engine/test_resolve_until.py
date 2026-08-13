"""Resolve a WAIT ``until`` source at the node-owned binding boundary.

``resolve_until`` accepts a string directly, formats an object exposing
``isoformat()`` defensively, and rejects values that are not date/datetime-like.
"""

import pytest

from agent_composer.compile.model import START_ID
from agent_composer.nodes.wait.node import resolve_until
from agent_composer.typesys.pool import VariablePool


def test_literal_iso_passthrough():
    assert resolve_until("2026-01-01T00:00:00", VariablePool()) == "2026-01-01T00:00:00"


def test_ref_iso_passthrough():
    pool = VariablePool()
    pool.set(START_ID, {"t": "2026-06-20T12:00:00"})
    assert resolve_until("${input.t}", pool) == "2026-06-20T12:00:00"


def test_non_date_raises_substring():
    pool = VariablePool()
    pool.set(START_ID, {"n": 5})
    with pytest.raises(ValueError, match="did not resolve to a date/datetime"):
        resolve_until("${input.n}", pool)
