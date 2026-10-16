"""TOOL retry defaults and the errors a host tool raises to steer retries."""

from __future__ import annotations

import pytest

import yian
import yian.tools as tools_package
from yian.defaults import DefaultsError, current
from yian.tools import FunctionTool, ToolSet
from yian.tools.errors import PermanentToolError, ToolCheckError, TransientToolError


def test_tool_retry_defaults() -> None:
    values = current()

    assert values.tool_retries == 0
    assert values.tool_retry_delay_seconds == 1
    assert values.tool_retry_max_delay_seconds == 30


def test_tool_retries_can_be_set(set_defaults) -> None:
    assert set_defaults(TOOL_RETRIES=2).tool_retries == 2


@pytest.mark.parametrize(
    "name", ["TOOL_RETRIES", "TOOL_RETRY_DELAY_SECONDS", "TOOL_RETRY_MAX_DELAY_SECONDS"]
)
def test_tool_retry_defaults_reject_negative_values(set_defaults, name) -> None:
    with pytest.raises(DefaultsError, match=f"YIAN_{name}='-1' is out of range"):
        set_defaults(**{name: -1})


def test_transient_tool_error_carries_retry_after() -> None:
    assert TransientToolError("x", retry_after=2.5).retry_after == 2.5
    assert TransientToolError("x", retry_after=0).retry_after == 0
    assert TransientToolError("x").retry_after is None
    assert str(TransientToolError("rate limited", retry_after=1)) == "rate limited"


@pytest.mark.parametrize("value", [-1, -0.5, float("nan"), float("inf")])
def test_transient_tool_error_rejects_a_negative_or_non_finite_retry_after(value) -> None:
    with pytest.raises(ValueError, match="retry_after"):
        TransientToolError("x", retry_after=value)


@pytest.mark.parametrize("value", [True, "1"])
def test_transient_tool_error_rejects_a_non_number_retry_after(value) -> None:
    with pytest.raises(TypeError, match="retry_after"):
        TransientToolError("x", retry_after=value)


def test_tool_errors_are_independent_exceptions() -> None:
    for error in (PermanentToolError, TransientToolError):
        assert issubclass(error, Exception)
    assert not issubclass(PermanentToolError, TransientToolError)
    assert not issubclass(TransientToolError, PermanentToolError)
    assert issubclass(ToolCheckError, TypeError)
    assert not issubclass(ToolCheckError, (PermanentToolError, TransientToolError))


def test_tool_errors_are_public() -> None:
    assert yian.PermanentToolError is PermanentToolError
    assert yian.TransientToolError is TransientToolError
    assert tools_package.PermanentToolError is PermanentToolError
    assert tools_package.TransientToolError is TransientToolError
    assert tools_package.ToolCheckError is ToolCheckError
    assert {"PermanentToolError", "TransientToolError", "ToolCheckError"} <= set(
        tools_package.__all__
    )
    assert "ToolCheckError" not in yian.__all__


def test_function_tool_awaitable_result_raises_a_tool_check_error() -> None:
    async def later() -> str:
        return "x"

    ft = FunctionTool(
        name="echo",
        description="",
        params_json_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        on_invoke=lambda ctx, args: later(),
    )

    with pytest.raises(ToolCheckError, match="returned an awaitable"):
        ToolSet({"echo": ft}).invoke("echo", {"text": "hi"})
