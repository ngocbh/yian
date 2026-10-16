"""MCP timeout defaults."""

from __future__ import annotations

import pytest

from yian.defaults import DefaultsError, current


def test_mcp_timeout_defaults() -> None:
    values = current()

    assert values.mcp_connect_timeout_seconds == 30
    assert values.mcp_call_timeout_seconds == 300


def test_mcp_timeouts_can_be_set(set_defaults) -> None:
    values = set_defaults(MCP_CONNECT_TIMEOUT_SECONDS=5, MCP_CALL_TIMEOUT_SECONDS=60)

    assert values.mcp_connect_timeout_seconds == 5
    assert values.mcp_call_timeout_seconds == 60


@pytest.mark.parametrize("name", ["MCP_CONNECT_TIMEOUT_SECONDS", "MCP_CALL_TIMEOUT_SECONDS"])
def test_mcp_timeouts_reject_values_below_one(set_defaults, name) -> None:
    with pytest.raises(DefaultsError, match=f"YIAN_{name}='0' is out of range"):
        set_defaults(**{name: 0})
