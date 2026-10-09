"""Tool registry + the ``@register_tool`` decorator.

A tool self-registers by decorating its implementation; importing the module
that defines it populates ``TOOL_REGISTRY`` as a side effect. A run's ``ToolSet``
reads this registry when the host passes no ``tools=`` mapping; the TOOL,
HUMAN_INPUT, and AGENT nodes look tools up by id through that ``ToolSet``.

This is the package's tool *seam*: the core ships no domain tools — a host
application registers its own (e.g. ``@register_tool("add")``). Tool ids must be
unique across everything registered into the process. Model calls are validated
against their public schema; host-owned interactions may use the trusted schema
for injected metadata. ``ToolSet`` holds one run's tools: this registry, or a
host mapping that replaces it and may hold named tool groups. The reserved
``ask_user`` id never lives here: each run gets its handler through
``ToolSet({"ask_user": handler, ...})``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import inspect
import json
from threading import RLock
from types import MappingProxyType
from typing import Annotated, Any, Callable

from langchain_core.tools import BaseTool, InjectedToolArg, StructuredTool
from pydantic import BeforeValidator, Field

from yian.skills.tool import LOAD_SKILL

# Maps tool id -> a LangChain tool with a Pydantic argument schema. The supported
# ``register_tool`` path guarantees that schema, which the strict invocation boundary needs.
TOOL_REGISTRY: dict[str, BaseTool] = {}
_REGISTRY_LOCK = RLock()
_ASK_USER_ID = "ask_user"
# Names a group tool may not take: the run's ask_user handler and the skills tool.
_RESERVED_TOOL_NAMES = frozenset({_ASK_USER_ID, LOAD_SKILL})
_MAX_OPTIONS = 6
_MAX_OPTION_CHARS = 80


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
        `ValueError`:
            `tool_id` is already registered, or is the reserved `ask_user` id.

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
                f"tool id {_ASK_USER_ID!r} is reserved; pass "
                "tools={**registered_tools(), 'ask_user': handler} to run_flow or resume_flow"
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


def _normalize_options(value: Any) -> list[str] | None:
    """Clean model-sent `ask_user` options; any shape is accepted, none raises.

    A JSON list string is parsed; any other string is one label. Non-string items
    are dropped. Labels are stripped to one line, empty and case-insensitive
    duplicate labels are dropped, long labels are truncated, and at most six are
    kept. Returns `None` when no label is left.
    """
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except ValueError:
            parsed = None
        value = parsed if isinstance(parsed, list) else [value]
    if not isinstance(value, list):
        return None
    labels: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            continue
        label = " ".join(item.split())
        if len(label) > _MAX_OPTION_CHARS:
            label = label[: _MAX_OPTION_CHARS - 1] + "…"
        if not label or label.casefold() in seen:
            continue
        seen.add(label.casefold())
        labels.append(label)
        if len(labels) == _MAX_OPTIONS:
            break
    return labels or None


def _ask_user_tool(handler: Callable[..., Any]) -> BaseTool:
    """Wrap a host handler in the fixed `ask_user` tool.

    The model-visible schema contains `question` and optional `options`, which are
    cleaned by `_normalize_options`. Structured question and display metadata are
    injected fields, so only a trusted invocation can set them. The handler always
    receives all six fields as keyword arguments.
    """

    def ask_user(
        question: str,
        options: Annotated[
            list[str] | None,
            BeforeValidator(_normalize_options),
            Field(
                description=(
                    "Optional 2-4 short answer choices. The user can always type a "
                    "different answer."
                )
            ),
        ] = None,
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
            options=options,
        )

    return StructuredTool.from_function(
        ask_user,
        name=_ASK_USER_ID,
        description=(
            "Ask the user one short question and wait for the answer. Use only when "
            "you need a decision or information that only the user can provide. Put "
            "the context they need in one or two sentences. When there are clear "
            "choices, pass 2-4 short `options`; the user can always type their own "
            "answer."
        ),
    )


def _group_tool(group: str, index: int, item: Any) -> BaseTool:
    """Return one group item as a tool named by its `name` or `__name__`."""
    where = f"tools[{group!r}][{index}]"
    if isinstance(item, BaseTool):
        return item
    if inspect.isroutine(item):
        if item.__name__ == "<lambda>":
            raise TypeError(
                f"{where} is a lambda, so it has no tool name; define it with def "
                "or wrap it in a BaseTool"
            )
        return _tool_from_function(item.__name__, item)
    raise TypeError(
        f"{where} must be a BaseTool, a function, or a method, got {type(item).__name__}"
    )


def _same_item(first: Any, other: Any) -> bool:
    """Return whether two group items are one tool: the same object or bound method."""
    # Each `obj.method` access builds a new bound method, but equal ones share
    # their function and instance.
    return first is other or (inspect.ismethod(first) and first == other)


def _check_ask_user_handler(handler: Callable[..., Any]) -> None:
    """Reject an `ask_user` handler that cannot take the six keyword fields."""
    try:
        signature = inspect.signature(handler)
    except (TypeError, ValueError):
        return  # No readable signature; the handler is called as is.
    try:
        signature.bind(
            question="",
            questions=None,
            answer_schema=None,
            title=None,
            node_id=None,
            options=None,
        )
    except TypeError as exc:
        raise TypeError(
            "tools['ask_user'] must accept question, questions, answer_schema, title, "
            "node_id, and options as keywords; define it as def handler(**request)"
        ) from exc


class ToolSet:
    """The tools one run can use.

    With no mapping, the run uses the global registry. With a mapping, even an
    empty one, the run sees only that mapping's entries; there is no fallback to
    the registry. Two runs with different `ToolSet`s never see each other's entries.

    An entry whose value is a list or tuple is a tool group. An AGENT node grants a
    group by its key, and `expand` replaces that key with the group's tool names.
    Every tool name in the run names one tool: the same tool object may sit in
    several groups, but a group tool may not share its name with a different tool
    or with a top-level key, or take a reserved name (`ask_user`, `load_skill`).

    Args:
        tools (`Mapping[str, BaseTool | Callable[..., Any] | Sequence[BaseTool | Callable[..., Any]]] | None`, *optional*, defaults to `None`):
            The run's tools keyed by tool id. A function or method becomes a tool
            named by its key and described by its docstring. To bind values such
            as a session, use a closure or a bound method, not `functools.partial`.
            A `BaseTool` must already use its key as its name. A list or tuple is a
            group keyed by its group name; each item is a `BaseTool` named by its
            `name`, or a function or method named by its `__name__` (not a
            lambda). The `ask_user` entry must be a plain callable; it is wrapped
            so the model sees `question` and `options`, and it receives
            `question`, `questions`, `answer_schema`, `title`, `node_id`, and
            `options` as keywords (define it as `def handler(**request)`). `None`
            means the global registry. To keep the registered tools and add more,
            pass `{**registered_tools(), ...}`.

    Raises:
        `TypeError`:
            `tools` is not a mapping or `None`, a key is not a non-empty string,
            an entry is not a `BaseTool`, function, method, or group of them, a
            group item is a lambda, or the `ask_user` handler cannot take its six
            keyword fields.
        `ValueError`:
            A `BaseTool` entry's name differs from its key, or a group tool's name
            is `ask_user` or `load_skill`, is a top-level key, or names a
            different tool elsewhere in the groups.

    Example:
        ```python
        def lookup(city: str) -> str:
            "Return the weather for one city."
            return "sunny"

        tools = ToolSet({"weather": lookup})
        tools.invoke("weather", {"city": "Paris"})

        def forecast(city: str) -> str:
            "Return tomorrow's weather for one city."
            return "rain"

        grouped = ToolSet({"toolbox": [lookup, forecast]})
        grouped.expand(["toolbox"])  # ["lookup", "forecast"]
        ```
    """

    __slots__ = ("_tools", "_groups")

    def __init__(
        self,
        tools: Mapping[str, BaseTool | Callable[..., Any] | Sequence[BaseTool | Callable[..., Any]]] | None = None,
    ) -> None:
        self._tools: MappingProxyType[str, BaseTool] | None = None
        self._groups: MappingProxyType[str, tuple[str, ...]] = MappingProxyType({})
        if tools is None:
            return
        if not isinstance(tools, Mapping):
            raise TypeError("tools must be a mapping of tool id to tool, or None")
        built: dict[str, BaseTool] = {}
        group_items: dict[str, list[tuple[Any, BaseTool]]] = {}
        for tool_id, entry in tools.items():
            if type(tool_id) is not str or not tool_id:
                raise TypeError("tool ids in tools must be non-empty strings")
            if tool_id == _ASK_USER_ID:
                if isinstance(entry, BaseTool) or not callable(entry):
                    raise TypeError(
                        "tools['ask_user'] must be a plain callable handler; "
                        "the engine supplies its schema"
                    )
                _check_ask_user_handler(entry)
                built[tool_id] = _ask_user_tool(entry)
            elif isinstance(entry, BaseTool):
                if entry.name != tool_id:
                    raise ValueError(
                        f"tool {tool_id!r} is a BaseTool named {entry.name!r}; "
                        "its name must match its key"
                    )
                built[tool_id] = entry
            elif inspect.isroutine(entry):
                built[tool_id] = _tool_from_function(tool_id, entry)
            elif isinstance(entry, (list, tuple)):
                group_items[tool_id] = [
                    (item, _group_tool(tool_id, index, item)) for index, item in enumerate(entry)
                ]
            else:
                raise TypeError(
                    f"tools[{tool_id!r}] must be a BaseTool, a function, a method, or a "
                    f"list of them, got {type(entry).__name__}; bind values in a def or "
                    "lambda, not functools.partial"
                )
        groups: dict[str, tuple[str, ...]] = {}
        owners: dict[str, tuple[str, Any]] = {}  # tool name -> (first group, its item)
        for group, items in group_items.items():
            names: list[str] = []
            for item, tool in items:
                name = tool.name
                if name in _RESERVED_TOOL_NAMES:
                    hint = (
                        "pass the handler as the top-level tools['ask_user'] entry"
                        if name == _ASK_USER_ID
                        else "granting skills binds it"
                    )
                    raise ValueError(
                        f"group {group!r} has a tool named {name!r}, a reserved name; {hint}"
                    )
                if name in tools:
                    raise ValueError(
                        f"tool {name!r} in group {group!r} clashes with tools[{name!r}]"
                    )
                names.append(name)
                if name not in owners:
                    owners[name] = (group, item)
                    built[name] = tool
                    continue
                # Host permission groups overlap, so one tool may sit in several of
                # them; two different tools under one name would be ambiguous.
                first_group, first_item = owners[name]
                if not _same_item(first_item, item):
                    where = (
                        f"appears twice in group {group!r}"
                        if first_group == group
                        else f"is in both groups {first_group!r} and {group!r}"
                    )
                    raise ValueError(f"tool {name!r} {where} as different tools")
            groups[group] = tuple(dict.fromkeys(names))
        self._tools = MappingProxyType(built)
        self._groups = MappingProxyType(groups)

    def _get(self, tool_id: str) -> BaseTool:
        """Return this run's entry, or the global entry when the run has no mapping.

        `ask_user` never comes from the registry: a run without its own handler
        cannot ask.
        """
        if tool_id == _ASK_USER_ID and (self._tools is None or tool_id not in self._tools):
            raise ValueError(
                "no ask_user handler for this run; pass "
                "tools={**registered_tools(), 'ask_user': handler} to run_flow or resume_flow"
            )
        if tool_id in self._groups:
            raise ValueError(
                f"{tool_id!r} is a tool group, not a tool; grant it in an AGENT "
                "node's tools: list"
            )
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

    @property
    def groups(self) -> tuple[str, ...]:
        """The names of this run's tool groups, in mapping order."""
        return tuple(self._groups)

    def expand(self, tool_ids: list[str]) -> list[str]:
        """Replace each group id with the names of the tools in that group.

        Args:
            tool_ids (`list[str]`):
                Granted ids, in order. An id that is not a group is kept as is.

        Returns:
            `list[str]`:
                The ids with every group replaced by its tool names, in group order.
                A name that repeats, from overlapping groups or a repeated grant,
                is kept only where it first appears.
        """
        expanded: list[str] = []
        for tool_id in tool_ids:
            expanded.extend(self._groups.get(tool_id, (tool_id,)))
        # Overlapping or repeated grants bind each tool once.
        return list(dict.fromkeys(expanded))

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
