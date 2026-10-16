"""Host tools that declare a first ``ToolContext`` parameter receive one per call."""

from __future__ import annotations

import re
from typing import Any

import pytest
from langchain_core.utils.function_calling import convert_to_openai_tool

from yian.nodes.human_input import HumanInputNode
from yian.nodes.base import Output
from yian.tools import (
    TOOL_REGISTRY,
    ToolSet,
    invoke_tool,
    invoke_tool_trusted,
    register_tool,
)
from yian.tools.context import ToolContext
from yian.tools.registry import _check_ask_user_handler
from yian.tools.types import tool_types
from yian.typesys.values import Type, ValueKind

_HEX32 = re.compile(r"[0-9a-f]{32}")


@pytest.fixture(autouse=True)
def _restore_tool_registry():
    """Keep process-global registrations isolated between tests."""
    original = dict(TOOL_REGISTRY)
    yield
    TOOL_REGISTRY.clear()
    TOOL_REGISTRY.update(original)


class Db:
    pass


_SEEN: list[tuple[ToolContext[Any], dict[str, Any]]] = []


@pytest.fixture(autouse=True)
def _clear_seen():
    _SEEN.clear()
    yield
    _SEEN.clear()


def place(ctx: ToolContext[Db], symbol: str, qty: int = 1) -> str:
    """Place an order.

    Args:
        symbol: The ticker.
        qty: How many shares.
    """
    _SEEN.append((ctx, {"symbol": symbol, "qty": qty}))
    return f"{symbol}:{qty}"


def test_unbound_toolset_passes_a_fresh_context():
    tools = ToolSet({"place": place})

    assert tools.invoke("place", {"symbol": "X"}, node="n") == "X:1"
    tools.invoke("place", {"symbol": "Y"}, node="n")

    (first, args), (second, _) = _SEEN
    assert args == {"symbol": "X", "qty": 1}
    assert first.tool_name == "place"
    assert first.node == "n"
    assert first.run_id == ""
    assert first.attempt == 1
    assert first.context is None
    assert _HEX32.fullmatch(first.call_id)
    assert _HEX32.fullmatch(second.call_id)
    assert first.call_id != second.call_id


def test_run_bound_toolset_passes_run_id_context_and_a_stable_call_id():
    db = Db()
    tools = ToolSet({"place": place}, context=db).for_run("run-1")

    tools.invoke("place", {"symbol": "X"}, node="loop#3/place")
    tools.invoke("place", {"symbol": "X"}, node="loop#3/place", call_id="x", attempt=2)

    (ctx, _), (explicit, _) = _SEEN
    assert ctx.context is db
    assert ctx.run_id == "run-1"
    assert ctx.node == "loop#3/place"
    assert ctx.call_id == "run-1/loop#3/place"
    assert explicit.call_id == "x"
    assert explicit.attempt == 2


def test_trusted_invocation_passes_the_same_context():
    tools = ToolSet({"place": place}).for_run("run-1")

    tools.invoke_trusted("place", {"symbol": "X"}, node="n", attempt=3)

    ((ctx, _),) = _SEEN
    assert ctx.call_id == "run-1/n"
    assert ctx.attempt == 3


def test_for_run_returns_a_bound_copy_and_leaves_the_original_unbound():
    db = Db()
    original = ToolSet({"place": place}, context=db)

    bound = original.for_run("run-1")

    assert bound is not original
    assert original.run_id == ""
    assert bound.run_id == "run-1"
    assert original.context is db
    assert bound.context is db
    original.invoke("place", {"symbol": "X"})
    assert _SEEN[0][0].run_id == ""
    assert _HEX32.fullmatch(_SEEN[0][0].call_id)


def test_bound_copy_of_the_global_registry_still_reads_it():
    register_tool("ctx_probe")(place)

    bound = ToolSet().for_run("run-9")

    assert bound.invoke("ctx_probe", {"symbol": "Z"}, node="t") == "Z:1"
    assert _SEEN[0][0].call_id == "run-9/t"
    assert _SEEN[0][0].tool_name == "ctx_probe"


def _late_context(symbol: str, ctx: ToolContext[Db]) -> str:
    """Put the context second."""
    return symbol


def test_a_non_first_context_parameter_is_rejected_at_construction():
    with pytest.raises(TypeError, match="first parameter"):
        ToolSet({"late": _late_context})


def test_a_non_first_context_parameter_is_rejected_at_registration():
    with pytest.raises(TypeError, match="first parameter"):
        register_tool("late")(_late_context)
    assert "late" not in TOOL_REGISTRY


def test_the_model_never_sees_the_context_parameter():
    tools = ToolSet({"place": place})
    (tool,) = tools.resolve(["place"])

    assert "ctx" not in tool.tool_call_schema.model_fields
    properties = convert_to_openai_tool(tool)["function"]["parameters"]["properties"]
    assert set(properties) == {"symbol", "qty"}
    with pytest.raises(ValueError, match="unsupported argument"):
        tools.invoke("place", {"symbol": "X", "ctx": "forged"})
    assert _SEEN == []


def test_description_and_parameter_descriptions_come_from_the_docstring():
    (tool,) = ToolSet({"place": place}).resolve(["place"])

    assert tool.description == "Place an order."
    properties = convert_to_openai_tool(tool)["function"]["parameters"]["properties"]
    assert properties["symbol"]["description"] == "The ticker."
    assert properties["qty"]["description"] == "How many shares."


def test_a_function_without_context_gets_exactly_its_validated_arguments():
    seen: list[tuple[Any, ...]] = []

    def add(a: int, b: int = 2) -> int:
        """Add two integers."""
        seen.append((a, b))
        return a + b

    assert ToolSet({"add": add}).invoke("add", {"a": "3"}) == 5
    assert seen == [(3, 2)]


def test_group_items_receive_the_context():
    tools = ToolSet({"trading": [place]}).for_run("r")

    tools.invoke("place", {"symbol": "X"}, node="n")

    assert _SEEN[0][0].tool_name == "place"
    assert _SEEN[0][0].call_id == "r/n"


def test_registered_tools_receive_the_context():
    register_tool("ctx_registered")(place)

    ToolSet().invoke("ctx_registered", {"symbol": "X"}, node="n")

    assert _SEEN[0][0].tool_name == "ctx_registered"


class _Broker:
    def __init__(self) -> None:
        self.calls: list[tuple[ToolContext[Any], str]] = []

    def submit(self, ctx: ToolContext[Db], symbol: str) -> str:
        """Submit one order."""
        self.calls.append((ctx, symbol))
        return symbol


def test_a_bound_method_receives_the_context():
    broker = _Broker()
    tools = ToolSet({"submit": broker.submit}).for_run("r")

    assert tools.invoke("submit", {"symbol": "X"}, node="n") == "X"

    ((ctx, symbol),) = broker.calls
    assert symbol == "X"
    assert ctx.call_id == "r/n"
    (tool,) = tools.resolve(["submit"])
    assert set(tool.tool_call_schema.model_fields) == {"symbol"}


def test_module_level_invocation_passes_an_unbound_context():
    register_tool("ctx_module")(place)

    assert invoke_tool("ctx_module", {"symbol": "X"}) == "X:1"
    assert invoke_tool_trusted("ctx_module", {"symbol": "Y"}) == "Y:1"

    (first, _), (second, _) = _SEEN
    assert first.run_id == "" and first.node == ""
    assert _HEX32.fullmatch(first.call_id)
    assert first.call_id != second.call_id


def test_calling_a_context_tool_outside_a_toolset_raises():
    (tool,) = ToolSet({"place": place}).resolve(["place"])

    with pytest.raises(TypeError, match="takes a ToolContext; invoke it through a run's ToolSet"):
        tool.func(symbol="X")


def test_an_async_function_is_rejected_at_construction():
    async def fetch(url: str) -> str:
        """Fetch a URL."""
        return url

    with pytest.raises(TypeError, match="async"):
        ToolSet({"t": fetch})


_ASK_FIELDS = ("question", "questions", "answer_schema", "title", "node_id", "options")


def test_ask_user_handler_with_context_receives_it_first():
    seen: list[tuple[ToolContext[Any], dict[str, Any]]] = []

    def ask(ctx: ToolContext, **request: Any) -> str:
        seen.append((ctx, request))
        return "yes"

    tools = ToolSet({"ask_user": ask}).for_run("r")

    assert tools.invoke("ask_user", {"question": "Q?", "options": ["A", "a"]}, node="agent") == "yes"

    ((ctx, request),) = seen
    assert ctx.tool_name == "ask_user"
    assert ctx.node == "agent"
    assert ctx.call_id == "r/agent"
    assert tuple(sorted(request)) == tuple(sorted(_ASK_FIELDS))
    assert request["options"] == ["A"]


def test_ask_user_handler_without_context_is_called_as_before():
    seen: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def handler(*args: Any, **request: Any) -> str:
        seen.append((args, request))
        return "ok"

    assert ToolSet({"ask_user": handler}).invoke("ask_user", {"question": "Q?"}) == "ok"
    assert seen == [((), {name: None for name in _ASK_FIELDS} | {"question": "Q?"})]


def test_human_input_passes_its_node_id_to_the_ask_user_handler():
    seen: list[ToolContext[Any]] = []

    def ask(ctx: ToolContext, **request: Any) -> str:
        seen.append(ctx)
        return "yes"

    tools = ToolSet({"ask_user": ask}).for_run("run-1")
    node = HumanInputNode("approval", prompt="Ship?")

    assert node.run({}, tools=tools) == Output("yes")
    assert seen[0].node == "approval"
    assert seen[0].call_id == "run-1/approval"


def test_check_ask_user_handler_binds_the_keywords_after_a_context():
    def ask(ctx: ToolContext, *, question, questions, answer_schema, title, node_id, options):
        return None

    def missing(ctx: ToolContext, *, question):
        return None

    _check_ask_user_handler(ask)
    with pytest.raises(TypeError, match="ask_user"):
        _check_ask_user_handler(missing)


def test_tool_types_skip_the_context_and_read_the_return_annotation():
    (tool,) = ToolSet({"place": place}).resolve(["place"])

    types = tool_types(tool)

    assert set(types.inputs) == {"symbol", "qty"}
    assert types.required == frozenset({"symbol"})
    assert types.output == Type.scalar(ValueKind.STRING)
