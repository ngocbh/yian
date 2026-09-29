"""Tool registry + the ``@register_tool`` decorator.

A tool self-registers by decorating its implementation; importing the module
that defines it populates ``TOOL_REGISTRY`` as a side effect. The engine's TOOL
node and the AGENT tool-calling loop look tools up here by id.

This is the package's tool *seam*: the core ships no domain tools — a host
application registers its own (e.g. ``@register_tool("add")``). Tool ids must be
unique across everything registered into the process. Model calls are validated
against their public schema; host-owned interactions may use the trusted schema
for injected metadata. The reserved ``ask_user`` entry has one process-wide host
owner per interaction lifecycle.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
import inspect
from threading import RLock
from types import MappingProxyType
from typing import Annotated, Any, Callable, Iterator

from langchain_core.tools import BaseTool, InjectedToolArg, StructuredTool

# Maps tool id -> a LangChain tool with a Pydantic argument schema. The supported
# ``register_tool`` path guarantees that schema, which the strict invocation boundary needs.
TOOL_REGISTRY: dict[str, BaseTool] = {}
_REGISTRY_LOCK = RLock()
_ASK_USER_ID = "ask_user"
_ASK_USER_OWNER: object | None = None


def register_tool(tool_id: str) -> Callable:
    """Register the decorated function as a tool under `tool_id`.

    The function's docstring becomes the tool description the model sees (dedented with
    `inspect.cleandoc`, so a multi-line docstring reads cleanly, not with source indentation).
    The original function is returned unchanged; the built LangChain tool is stored in
    `TOOL_REGISTRY`.

    Args:
        tool_id (`str`):
            The unique id to register under; a duplicate id raises.

    Returns:
        `Callable`: A decorator that registers its target and returns it unchanged.

    Raises:
        ValueError: If `tool_id` is already registered.

    Example:
        ```python
        @register_tool("add")
        def add(a: int, b: int) -> int:
            "Add two integers."
            return a + b
        ```
    """

    def decorator(fn: Callable) -> Callable:
        if tool_id == _ASK_USER_ID:
            raise ValueError(
                f"tool id {_ASK_USER_ID!r} is reserved; use install_ask_user()"
            )
        with _REGISTRY_LOCK:
            if tool_id in TOOL_REGISTRY:
                raise ValueError(f"Duplicate tool id '{tool_id}'")
            TOOL_REGISTRY[tool_id] = _tool_from_function(tool_id, fn)
        return fn

    return decorator


def _tool_from_function(tool_id: str, fn: Callable[..., Any]) -> BaseTool:
    """Build a tool named `tool_id` whose description is the dedented docstring."""
    description = inspect.cleandoc(fn.__doc__ or "").strip() or tool_id
    return StructuredTool.from_function(fn, name=tool_id, description=description)


def registered_tools() -> dict[str, BaseTool]:
    """Return a copy of the tools registered with `register_tool`.

    A run given `tools=` sees only those tools. A host that wants the registered
    tools as well builds the mapping itself, for example
    `tools={**registered_tools(), "ask_user": handler}`. Changing the copy does not
    change the registry. The copy is taken when you call this, so register every
    tool the run needs before the run starts.

    Returns:
        `dict[str, BaseTool]`:
            A new dict of tool id to tool, taken under the registry lock.
    """
    with _REGISTRY_LOCK:
        return dict(TOOL_REGISTRY)


class ToolSet:
    """The tools one run can use.

    With no mapping, the run uses the global registry. With a mapping, even an
    empty one, the run sees only that mapping's entries; there is no fallback to
    the registry. Two runs with different `ToolSet`s never see each other's entries.

    Args:
        tools (`Mapping[str, BaseTool | Callable[..., Any]] | None`, *optional*, defaults to `None`):
            The run's tools keyed by tool id. A function or method becomes a tool
            named by its key and described by its docstring. To bind values such
            as a session, use a closure or a bound method, not `functools.partial`.
            A `BaseTool` must already use its key as its name. `None` means the
            global registry. To keep the registered tools and add more, pass
            `{**registered_tools(), ...}`.

    Raises:
        `TypeError`:
            `tools` is not a mapping or `None`, a key is not a non-empty string,
            or an entry is not a `BaseTool`, function, or method.
        `ValueError`:
            A `BaseTool` entry's name differs from its key.

    Example:
        ```python
        def lookup(city: str) -> str:
            "Return the weather for one city."
            return "sunny"

        tools = ToolSet({"weather": lookup})
        tools.invoke("weather", {"city": "Paris"})
        ```
    """

    __slots__ = ("_tools",)

    def __init__(
        self, tools: Mapping[str, BaseTool | Callable[..., Any]] | None = None
    ) -> None:
        self._tools: MappingProxyType[str, BaseTool] | None = None
        if tools is None:
            return
        if not isinstance(tools, Mapping):
            raise TypeError("tools must be a mapping of tool id to tool, or None")
        built: dict[str, BaseTool] = {}
        for tool_id, entry in tools.items():
            if type(tool_id) is not str or not tool_id:
                raise TypeError("tool ids in tools must be non-empty strings")
            if isinstance(entry, BaseTool):
                if entry.name != tool_id:
                    raise ValueError(
                        f"tool {tool_id!r} is a BaseTool named {entry.name!r}; "
                        "its name must match its key"
                    )
                built[tool_id] = entry
            elif inspect.isroutine(entry):
                built[tool_id] = _tool_from_function(tool_id, entry)
            else:
                raise TypeError(
                    f"tools[{tool_id!r}] must be a BaseTool, a function, or a method, "
                    f"got {type(entry).__name__}; bind values in a def or lambda, "
                    "not functools.partial"
                )
        self._tools = MappingProxyType(built)

    def _get(self, tool_id: str) -> BaseTool:
        """Return this run's entry, or the global entry when the run has no mapping."""
        if self._tools is not None:
            tool = self._tools.get(tool_id)
            if tool is None:
                raise ValueError(
                    f"unknown tool id {tool_id!r}; known tools: {sorted(self._tools)}"
                )
            return tool
        with _REGISTRY_LOCK:
            tool = TOOL_REGISTRY.get(tool_id)
            if tool is None:
                raise ValueError(
                    f"unknown tool id {tool_id!r}; known tools: {sorted(TOOL_REGISTRY)}"
                )
            return tool

    def resolve(self, tool_ids: list[str]) -> list[BaseTool]:
        """Look up tools by id.

        Args:
            tool_ids (`list[str]`):
                The ids to resolve, in order.

        Returns:
            `list[BaseTool]`:
                The matching tools, in the same order as `tool_ids`.

        Raises:
            `ValueError`:
                An id is not visible to this run.
        """
        return [self._get(tool_id) for tool_id in tool_ids]

    def invoke(self, tool_id: str, inputs: dict[str, Any]) -> Any:
        """Validate and invoke one tool through its model-visible public schema.

        Unknown keys are rejected before Pydantic validation because tool schemas
        may otherwise ignore them. Exceptions from validation and the tool
        propagate unchanged.

        Args:
            tool_id (`str`):
                The id of the tool to invoke.
            inputs (`dict[str, Any]`):
                Public arguments keyed by parameter name.

        Returns:
            `Any`:
                The tool's result.

        Raises:
            `ValueError`:
                The id is not visible to this run, or an argument is not public.
        """
        tool = self._get(tool_id)
        payload = dict(inputs)
        _validate_invocation(tool_id, tool, payload, trusted=False)
        return tool.invoke(payload)

    def invoke_trusted(self, tool_id: str, inputs: dict[str, Any]) -> Any:
        """Invoke one tool through its full schema, including host-injected fields.

        Args:
            tool_id (`str`):
                The id of the tool to invoke.
            inputs (`dict[str, Any]`):
                Public and injected arguments keyed by parameter name.

        Returns:
            `Any`:
                The tool's result.

        Raises:
            `ValueError`:
                The id is not visible to this run, or an argument is not in the
                full schema.
        """
        tool = self._get(tool_id)
        payload = dict(inputs)
        _validate_invocation(tool_id, tool, payload, trusted=True)
        return tool.invoke(payload)


def resolve_tools(tool_ids: list[str]) -> list[BaseTool]:
    """Look up tools by id.

    Args:
        tool_ids (`list[str]`):
            The ids to resolve, in order.

    Returns:
        `list[BaseTool]`: The matching tools, in the same order as `tool_ids`.

    Raises:
        ValueError: If any id is not registered.
    """
    with _REGISTRY_LOCK:
        resolved = []
        for tid in tool_ids:
            if tid not in TOOL_REGISTRY:
                raise ValueError(
                    f"Unknown tool id '{tid}'. Known tools: {sorted(TOOL_REGISTRY)}"
                )
            resolved.append(TOOL_REGISTRY[tid])
        return resolved


def _registered_tool(tool_id: str) -> BaseTool:
    """Return one registry entry while holding mutation synchronization."""
    with _REGISTRY_LOCK:
        tool = TOOL_REGISTRY.get(tool_id)
        if tool is None:
            raise ValueError(
                f"unknown tool id {tool_id!r}; known tools: {sorted(TOOL_REGISTRY)}"
            )
        return tool


def _validate_invocation(
    tool_id: str,
    tool: BaseTool,
    inputs: dict[str, Any],
    *,
    trusted: bool,
) -> None:
    """Reject extra keys, then validate a supported Pydantic argument schema."""
    schema = tool.args_schema if trusted else tool.tool_call_schema
    if isinstance(schema, dict):
        raise TypeError(
            f"registered tool {tool_id!r} uses an unvalidated JSON args_schema; "
            "use a Pydantic argument schema or register the function with register_tool()"
        )
    if schema is not None and hasattr(schema, "model_fields"):
        allowed = frozenset(schema.model_fields)
        validate = schema.model_validate
    elif schema is not None and hasattr(schema, "__fields__"):
        allowed = frozenset(schema.__fields__)
        validate = schema.parse_obj
    else:
        raise TypeError(
            f"registered tool {tool_id!r} does not expose a supported argument schema"
        )

    unsupported = sorted(set(inputs) - allowed)
    if unsupported:
        scope = "trusted" if trusted else "public"
        raise ValueError(
            f"tool {tool_id!r} received unsupported argument(s) {unsupported} "
            f"on its {scope} invocation path; allowed: {sorted(allowed)}"
        )
    validate(inputs)


def invoke_tool(tool_id: str, inputs: dict[str, Any]) -> Any:
    """Validate and invoke one tool through its model-visible public schema.

    Unknown keys are rejected before Pydantic validation because tool schemas may
    otherwise ignore them. Exceptions from validation and the tool propagate to the
    node boundary unchanged.
    """
    tool = _registered_tool(tool_id)
    payload = dict(inputs)
    _validate_invocation(tool_id, tool, payload, trusted=False)
    return tool.invoke(payload)


def invoke_tool_trusted(tool_id: str, inputs: dict[str, Any]) -> Any:
    """Invoke one tool through its full schema, including host-injected fields."""
    tool = _registered_tool(tool_id)
    payload = dict(inputs)
    _validate_invocation(tool_id, tool, payload, trusted=True)
    return tool.invoke(payload)


@contextmanager
def install_ask_user(handler: Callable[..., Any]) -> Iterator[BaseTool]:
    """Install the process-wide ``ask_user`` owner for one host lifecycle.

    The model-visible schema contains only ``question``. Structured question and
    display metadata are injected solely through ``invoke_tool_trusted``. A second
    active owner is rejected, and cleanup removes only this installation's exact
    registry entry.
    """
    global _ASK_USER_OWNER

    def ask_user(
        question: str,
        questions: Annotated[list[dict[str, Any]] | None, InjectedToolArg] = None,
        answer_schema: Annotated[list[dict[str, Any]] | None, InjectedToolArg] = None,
        title: Annotated[str | None, InjectedToolArg] = None,
        node_id: Annotated[str | None, InjectedToolArg] = None,
    ) -> Any:
        """Ask the human a question and wait for the host-provided answer."""
        return handler(
            question=question,
            questions=questions,
            answer_schema=answer_schema,
            title=title,
            node_id=node_id,
        )

    tool = StructuredTool.from_function(
        ask_user,
        name=_ASK_USER_ID,
        description=(
            "Ask the human a question and wait for their answer. Use only when you "
            "need information that only the user can provide."
        ),
    )
    owner = object()
    with _REGISTRY_LOCK:
        if _ASK_USER_OWNER is not None or _ASK_USER_ID in TOOL_REGISTRY:
            raise ValueError("tool 'ask_user' already has an active owner")
        _ASK_USER_OWNER = owner
        TOOL_REGISTRY[_ASK_USER_ID] = tool

    try:
        yield tool
    finally:
        with _REGISTRY_LOCK:
            if _ASK_USER_OWNER is owner:
                if TOOL_REGISTRY.get(_ASK_USER_ID) is tool:
                    del TOOL_REGISTRY[_ASK_USER_ID]
                _ASK_USER_OWNER = None
