"""``FunctionTool``, ``function_tool``, and name and description overrides."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.utils.function_calling import convert_to_openai_tool

import yian
import yian.tools as tools_package
from yian.tools import (
    TOOL_REGISTRY,
    FunctionTool,
    ToolContext,
    ToolSet,
    function_tool,
    register_tool,
)
from yian.tools.types import tool_types
from yian.typesys.values import Type, TypeCheckError, ValueKind

_ECHO_SCHEMA = {
    "type": "object",
    "properties": {"text": {"type": "string"}},
    "required": ["text"],
}


@pytest.fixture(autouse=True)
def _restore_tool_registry():
    """Keep process-global registrations isolated between tests."""
    original = dict(TOOL_REGISTRY)
    yield
    TOOL_REGISTRY.clear()
    TOOL_REGISTRY.update(original)


def _recorder(result: Any = "done"):
    """Return an `on_invoke` that records its calls, and the list it records into."""
    seen: list[tuple[ToolContext[Any], dict[str, Any]]] = []

    def on_invoke(ctx: ToolContext[Any], args: dict[str, Any]) -> Any:
        seen.append((ctx, args))
        return result

    return on_invoke, seen


def _echo(schema: dict[str, Any] | None = None, name: str = "echo"):
    on_invoke, seen = _recorder()
    tool = FunctionTool(
        name=name,
        description="Echo.",
        params_json_schema=_ECHO_SCHEMA if schema is None else schema,
        on_invoke=on_invoke,
    )
    return tool, seen


# FunctionTool in a ToolSet


def test_function_tool_is_invoked_with_a_context_and_its_arguments():
    ft, seen = _echo()
    inputs = {"text": "hi"}

    result = ToolSet({"echo": ft}).for_run("r").invoke("echo", inputs, node="n")

    assert result == "done"
    ((ctx, args),) = seen
    assert isinstance(ctx, ToolContext)
    assert ctx.tool_name == "echo"
    assert ctx.node == "n"
    assert ctx.call_id == "r/n"
    assert args == {"text": "hi"}
    assert type(args) is dict
    assert args is not inputs


def test_function_tool_rejects_an_unknown_argument_before_on_invoke():
    ft, seen = _echo()

    with pytest.raises(ValueError, match="unsupported argument"):
        ToolSet({"echo": ft}).invoke("echo", {"text": "hi", "extra": 1})
    assert seen == []


@pytest.mark.parametrize("extra", [True, {"type": "integer"}])
def test_function_tool_accepts_unknown_arguments_when_the_schema_allows_them(extra):
    ft, seen = _echo({**_ECHO_SCHEMA, "additionalProperties": extra})

    ToolSet({"echo": ft}).invoke("echo", {"text": "hi", "extra": 1})

    assert seen[0][1] == {"text": "hi", "extra": 1}


def test_function_tool_checks_extra_arguments_against_an_additional_properties_schema():
    ft, seen = _echo({**_ECHO_SCHEMA, "additionalProperties": {"type": "integer"}})

    with pytest.raises(TypeCheckError, match="extra"):
        ToolSet({"echo": ft}).invoke("echo", {"text": "hi", "extra": "x"})
    assert seen == []


def test_function_tool_rejects_a_missing_required_argument():
    ft, seen = _echo()

    with pytest.raises(ValueError, match="text"):
        ToolSet({"echo": ft}).invoke("echo", {})
    assert seen == []


def test_function_tool_rejects_a_value_of_the_wrong_type():
    ft, seen = _echo()

    with pytest.raises(TypeCheckError, match="text"):
        ToolSet({"echo": ft}).invoke("echo", {"text": 3})
    assert seen == []


def test_function_tool_passes_an_untranslatable_property_through_unchecked():
    schema = {
        "type": "object",
        "properties": {"v": {"anyOf": [{"type": "string"}, {"type": "integer"}]}},
    }
    ft, seen = _echo(schema)

    ToolSet({"echo": ft}).invoke("echo", {"v": [1, 2]})

    assert seen[0][1] == {"v": [1, 2]}


def test_function_tool_accepts_an_integral_float_for_an_integer_as_given():
    schema = {"type": "object", "properties": {"n": {"type": "integer", "minimum": 1}}}
    ft, seen = _echo(schema)
    tools = ToolSet({"echo": ft})

    tools.invoke("echo", {"n": 3.0})

    assert seen[0][1] == {"n": 3.0}
    assert type(seen[0][1]["n"]) is float
    with pytest.raises(TypeCheckError):
        tools.invoke("echo", {"n": 3.5})
    with pytest.raises(TypeCheckError):
        tools.invoke("echo", {"n": 0.0})


def test_function_tool_trusted_invocation_validates_the_same_way():
    ft, seen = _echo()
    tools = ToolSet({"echo": ft})

    assert tools.invoke_trusted("echo", {"text": "hi"}) == "done"
    with pytest.raises(ValueError, match="unsupported argument"):
        tools.invoke_trusted("echo", {"text": "hi", "extra": 1})


def test_function_tool_key_must_match_its_name():
    ft, _ = _echo()

    with pytest.raises(ValueError, match="'other'.*'echo'"):
        ToolSet({"other": ft})


def test_function_tool_group_item_is_named_by_its_name():
    ft, seen = _echo()
    tools = ToolSet({"box": [ft]})

    assert tools.expand(["box"]) == ["echo"]
    tools.invoke("echo", {"text": "hi"})
    assert seen[0][0].tool_name == "echo"


def test_function_tool_cannot_be_the_ask_user_handler():
    ft, _ = _echo(name="ask_user")

    with pytest.raises(TypeError, match="plain callable"):
        ToolSet({"ask_user": ft})


def test_register_tool_registers_a_function_tool_and_returns_it_unchanged():
    ft, seen = _echo()

    assert register_tool("echo")(ft) is ft

    assert ToolSet().invoke("echo", {"text": "hi"}) == "done"
    assert seen[0][0].tool_name == "echo"


def test_register_tool_rejects_a_function_tool_with_another_name():
    ft, _ = _echo()

    with pytest.raises(ValueError, match="'other'.*'echo'"):
        register_tool("other")(ft)
    assert "other" not in TOOL_REGISTRY


def test_register_tool_overrides_a_function_tool_description():
    ft, _ = _echo()

    register_tool("echo", description_override="New.")(ft)

    assert TOOL_REGISTRY["echo"].description == "New."
    assert ft.description == "Echo."


# FunctionTool construction


@pytest.mark.parametrize(
    "changes, error",
    [
        ({"name": ""}, ValueError),
        ({"name": 3}, TypeError),
        ({"description": None}, TypeError),
        ({"params_json_schema": [1]}, TypeError),
        ({"params_json_schema": {"type": "string"}}, ValueError),
        ({"params_json_schema": {"properties": {}}}, ValueError),
        ({"on_invoke": "not callable"}, TypeError),
    ],
)
def test_function_tool_validates_its_fields(changes, error):
    fields = {
        "name": "echo",
        "description": "Echo.",
        "params_json_schema": _ECHO_SCHEMA,
        "on_invoke": lambda ctx, args: None,
    }
    with pytest.raises(error):
        FunctionTool(**{**fields, **changes})


def test_function_tool_rejects_a_coroutine_on_invoke():
    async def on_invoke(ctx, args):
        return None

    class AsyncCall:
        async def __call__(self, ctx, args):
            return None

    for handler in (on_invoke, AsyncCall()):
        with pytest.raises(TypeError, match="async"):
            FunctionTool(
                name="echo", description="", params_json_schema=_ECHO_SCHEMA, on_invoke=handler
            )


def test_function_tool_rejects_an_awaitable_result_at_call_time():
    async def later() -> str:
        return "x"

    class ReturnsAwaitable:
        def __call__(self, ctx, args):
            return later()

    ft = FunctionTool(
        name="echo", description="", params_json_schema=_ECHO_SCHEMA, on_invoke=ReturnsAwaitable()
    )

    with pytest.raises(TypeError, match="awaitable"):
        ToolSet({"echo": ft}).invoke("echo", {"text": "hi"})


def test_function_tool_without_a_signature_cannot_be_called_directly():
    ft, _ = _echo()
    (tool,) = ToolSet({"echo": ft}).resolve(["echo"])

    with pytest.raises(TypeError, match="ToolSet"):
        tool.func(text="hi")


# AGENT binding and types


def test_function_tool_binds_with_its_name_description_and_schema():
    ft, _ = _echo()

    (tool,) = ToolSet({"echo": ft}).resolve(["echo"])
    function = convert_to_openai_tool(tool)["function"]

    assert function["name"] == "echo"
    assert function["description"] == "Echo."
    assert function["parameters"]["properties"] == {"text": {"type": "string"}}
    assert function["parameters"]["required"] == ["text"]


def test_function_tool_binds_a_schema_with_references_inlined():
    schema = {
        "type": "object",
        "title": "Order",
        "$defs": {"Side": {"type": "string", "enum": ["buy", "sell"], "title": "Side"}},
        "properties": {"side": {"$ref": "#/$defs/Side"}, "qty": {"type": "integer"}},
        "required": ["side"],
    }
    ft, seen = _echo(schema, name="order")
    tools = ToolSet({"order": ft})

    (tool,) = tools.resolve(["order"])
    parameters = convert_to_openai_tool(tool)["function"]["parameters"]

    assert parameters["properties"]["side"] == {"type": "string", "enum": ["buy", "sell"]}
    assert parameters["properties"]["qty"] == {"type": "integer"}
    assert parameters["required"] == ["side"]
    tools.invoke("order", {"side": "buy"})
    with pytest.raises(TypeCheckError):
        tools.invoke("order", {"side": "hold"})


def test_tool_types_read_a_function_tool_json_schema():
    ft, _ = _echo()
    (tool,) = ToolSet({"echo": ft}).resolve(["echo"])

    types = tool_types(tool)

    assert types.inputs == {"text": Type.scalar(ValueKind.STRING)}
    assert types.required == frozenset({"text"})
    assert types.output is None


# function_tool


class Broker:
    def __init__(self) -> None:
        self.orders: list[tuple[str, int, str]] = []


def place(ctx: ToolContext[Broker], symbol: str, qty: int = 1) -> str:
    """Place an order.

    Args:
        symbol: The ticker.
        qty: How many shares.
    """
    ctx.context.orders.append((symbol, qty, ctx.call_id))
    return f"{symbol}:{qty}"


def add(a: int, b: int) -> int:
    """Add two integers.

    Args:
        a: The first integer.
        b: The second integer.
    """
    return a + b


def test_function_tool_builds_a_function_tool_from_a_function():
    ft = function_tool(place, name_override="buy", description_override="Buy.")

    assert isinstance(ft, FunctionTool)
    assert ft.name == "buy"
    assert ft.description == "Buy."
    assert set(ft.params_json_schema["properties"]) == {"symbol", "qty"}
    assert ft.params_json_schema["required"] == ["symbol"]


def test_function_tool_defaults_to_the_function_name_and_docstring():
    ft = function_tool(add)

    assert ft.name == "add"
    assert ft.description == "Add two integers."
    assert ft.params_json_schema["properties"]["a"]["description"] == "The first integer."


def test_function_tool_result_validates_like_a_function_and_passes_the_context():
    broker = Broker()
    ft = function_tool(place, name_override="buy", description_override="Buy.")
    tools = ToolSet({"buy": ft}, context=broker).for_run("r")

    assert tools.invoke("buy", {"symbol": "X", "qty": "2"}, node="n") == "X:2"

    assert broker.orders == [("X", 2, "r/n")]
    with pytest.raises(ValueError, match="unsupported argument"):
        tools.invoke("buy", {"symbol": "X", "ctx": "forged"})
    with pytest.raises(ValueError):
        tools.invoke("buy", {"symbol": "X", "qty": "many"})


def test_function_tool_without_context_gets_its_validated_arguments():
    ft = function_tool(add)

    assert ToolSet({"add": ft}).invoke("add", {"a": "3", "b": 4}) == 7
    ctx = ToolContext(call_id="c", attempt=1, tool_name="add", run_id="", node="", context=None)
    assert ft.on_invoke(ctx, {"a": 1, "b": "2"}) == 3


def test_function_tool_result_binds_and_reads_types_like_a_function():
    ft = function_tool(place, name_override="buy", description_override="Buy.")
    (tool,) = ToolSet({"buy": ft}).resolve(["buy"])

    function = convert_to_openai_tool(tool)["function"]
    types = tool_types(tool)

    assert function["name"] == "buy"
    assert function["description"] == "Buy."
    assert set(function["parameters"]["properties"]) == {"symbol", "qty"}
    assert function["parameters"]["properties"]["symbol"]["description"] == "The ticker."
    assert set(types.inputs) == {"symbol", "qty"}
    assert types.inputs["qty"] == Type.scalar(ValueKind.INTEGER)
    assert types.required == frozenset({"symbol"})
    assert types.output == Type.scalar(ValueKind.STRING)


def test_function_tool_works_as_a_decorator():
    @function_tool
    def shout(text: str) -> str:
        """Shout the text."""
        return text.upper()

    @function_tool(name_override="whisper")
    def quiet(text: str) -> str:
        """Whisper the text."""
        return text.lower()

    assert isinstance(shout, FunctionTool) and shout.name == "shout"
    assert quiet.name == "whisper" and quiet.description == "Whisper the text."
    tools = ToolSet({"box": [shout, quiet]})
    assert tools.invoke("shout", {"text": "hi"}) == "HI"
    assert tools.invoke("whisper", {"text": "HI"}) == "hi"


def test_function_tool_rejects_an_async_function():
    async def fetch(url: str) -> str:
        """Fetch a URL."""
        return url

    with pytest.raises(TypeError, match="async"):
        function_tool(fetch)


def test_function_tool_names_the_tool_when_an_annotation_cannot_be_resolved():
    def f(x: Foo) -> str:  # noqa: F821
        return ""

    with pytest.raises(TypeError, match="'f'"):
        function_tool(f)


class Opaque:
    """A class pydantic validates with isinstance but can't describe in JSON Schema."""


def test_function_tool_names_the_tool_when_an_argument_has_no_json_schema():
    def g(x: Opaque) -> str:
        return ""

    with pytest.raises(TypeError, match="'opaque_tool'"):
        function_tool(g, name_override="opaque_tool")


def test_function_tool_registers_with_register_tool():
    ft = function_tool(add)

    register_tool("add")(ft)

    assert ToolSet().invoke("add", {"a": 1, "b": 2}) == 3
    assert tool_types(TOOL_REGISTRY["add"]).output == Type.scalar(ValueKind.INTEGER)


def test_register_tool_description_override_on_a_function():
    register_tool("add", description_override="Add two ints.")(add)

    assert TOOL_REGISTRY["add"].description == "Add two ints."


def test_register_tool_description_override_on_a_function_tool_from_a_function():
    register_tool("add", description_override="Sum.")(function_tool(add))

    assert TOOL_REGISTRY["add"].description == "Sum."
    assert ToolSet().invoke("add", {"a": 1, "b": 2}) == 3


# Exports


def test_tool_names_are_exported_from_yian_and_yian_tools():
    from yian.tools.context import ToolContext as context_class
    from yian.tools.function_tools import FunctionTool as function_tool_class
    from yian.tools.function_tools import function_tool as function_tool_fn

    for module in (yian, tools_package):
        assert module.ToolContext is context_class
        assert module.FunctionTool is function_tool_class
        assert module.function_tool is function_tool_fn
        assert {"ToolContext", "FunctionTool", "function_tool"} <= set(module.__all__)

