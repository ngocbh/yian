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

A plain function or method becomes a tool through ``tools.schema.function_schema``:
its type hints give the argument model, and its docstring's first text block and
``Args`` entries give the description and the parameter descriptions. A first
parameter annotated ``ToolContext`` is hidden from the model; ``ToolSet.invoke``
passes the call's context (``tools.context``) there. A ``FunctionTool``
(``tools.function_tools``) supplies its own JSON Schema and ``on_invoke`` instead;
``ToolSet`` checks its arguments against that schema before the call.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
import functools
import inspect
import json
from threading import RLock
from types import MappingProxyType
from typing import Annotated, Any, Callable
from uuid import uuid4

from langchain_core.tools import BaseTool, InjectedToolArg, StructuredTool
from pydantic import BeforeValidator, Field

from yian.defaults import current
from yian.skills.tool import LOAD_SKILL
from yian.tools.context import ToolContext
from yian.tools.function_tools import FunctionTool
from yian.tools.schema import _is_tool_context, function_schema
from yian.tools.types import _input_type_from_json_schema
from yian.typesys.values import Type, TypeCheckError, ValueKind, build_value_as

# Maps tool id -> a LangChain tool with a Pydantic argument schema, or a
# ``FunctionTool``'s JSON Schema that yian checks itself. The supported ``register_tool``
# path guarantees one of these, which the strict invocation boundary needs.
TOOL_REGISTRY: dict[str, BaseTool] = {}
_REGISTRY_LOCK = RLock()
_ASK_USER_ID = "ask_user"
# Names a group tool may not take: the run's ask_user handler and the skills tool.
_RESERVED_TOOL_NAMES = frozenset({_ASK_USER_ID, LOAD_SKILL})
# One entry of a host `tools=` mapping or of a tool group.
_Entry = BaseTool | FunctionTool | Callable[..., Any]
# What evaluating a string annotation, or reading a signature, can raise.
_ANNOTATION_ERRORS = (NameError, AttributeError, SyntaxError, TypeError, ValueError)


def register_tool(tool_id: str, *, description_override: str | None = None) -> Callable:
    """Register the decorated function or `FunctionTool` as a tool under `tool_id`.

    The first text block of the function's docstring becomes the tool description the
    model sees, and its `Args` entries (Google, NumPy, or Sphinx style) describe the
    parameters. A first parameter annotated `ToolContext` or `ToolContext[T]` is hidden
    from the model and receives the call's context. A `FunctionTool` keeps its own
    schema and description. The target is returned unchanged; the built tool is
    stored in `TOOL_REGISTRY`.

    Args:
        tool_id (`str`):
            The unique id to register under; a duplicate id raises. A
            `FunctionTool`'s name must equal it.
        description_override (`str`, *optional*, defaults to `None`):
            The description the model sees. `None` keeps the docstring's (for a
            function) or the tool's own (for a `FunctionTool`).

    Returns:
        `Callable`: A decorator that registers its target and returns it unchanged.

    Raises:
        `ValueError`:
            `tool_id` is already registered, is the reserved `ask_user` id, or
            differs from a `FunctionTool`'s name.
        `TypeError`:
            The function is async, an annotation can't be resolved, or `ToolContext`
            appears anywhere but as the first parameter's type.

    Example:
        ```python
        @register_tool("add")
        def add(a: int, b: int) -> int:
            '''Add two integers.

            Args:
                a: The first integer.
                b: The second integer.
            '''
            return a + b

        @register_tool("place")
        def place(ctx: ToolContext[Broker], symbol: str) -> str:
            "Place one order."
            return ctx.context.submit(symbol, idempotency_key=ctx.call_id)
        ```
    """

    def decorator(fn: Callable | FunctionTool) -> Callable | FunctionTool:
        if tool_id == _ASK_USER_ID:
            raise ValueError(
                f"tool id {_ASK_USER_ID!r} is reserved; pass "
                "tools={**registered_tools(), 'ask_user': handler} to run_flow or resume_flow"
            )
        with _REGISTRY_LOCK:
            if tool_id in TOOL_REGISTRY:
                raise ValueError(f"Duplicate tool id '{tool_id}'")
            if isinstance(fn, FunctionTool):
                if fn.name != tool_id:
                    raise ValueError(
                        f"tool id {tool_id!r} names a FunctionTool named {fn.name!r}; "
                        "its name must match its id"
                    )
                tool = _tool_from_function_tool(fn, description=description_override)
            else:
                tool = _tool_from_function(tool_id, fn, description=description_override)
            TOOL_REGISTRY[tool_id] = tool
        return fn

    return decorator


class _HostTool(StructuredTool):
    """A tool yian built from a host function, invoked by `ToolSet` with a `ToolContext`.

    The LangChain fields (`args_schema`, `tool_call_schema`, `func`) serve model
    binding and type reading; `ToolSet` calls `yian_call` instead of `invoke`.

    Args:
        yian_call (`Callable[[ToolContext[Any], dict[str, Any]], Any]`):
            Validates the arguments and calls the host function, passing the
            context to a function that declares one.
    """

    yian_call: Callable[[ToolContext[Any], dict[str, Any]], Any] = Field(repr=False)


def _tool_from_function(
    tool_id: str, fn: Callable[..., Any], *, description: str | None = None
) -> _HostTool:
    """Build the tool `tool_id` from `fn`'s signature and docstring.

    `description` replaces the docstring's description; with neither, the
    description is `tool_id`. Raises `TypeError` as `function_schema` does.
    """
    schema = function_schema(fn, name_override=tool_id, description_override=description)

    # `functools.wraps` keeps the return annotation `tools.types` reads from `func`.
    @functools.wraps(fn)
    def func(*args: Any, **kwargs: Any) -> Any:
        if schema.takes_context:
            raise TypeError(
                f"tool {tool_id!r} takes a ToolContext; invoke it through a run's ToolSet"
            )
        return fn(*args, **kwargs)

    def yian_call(ctx: ToolContext[Any], payload: dict[str, Any]) -> Any:
        data = schema.params_model.model_validate(payload)
        args, kwargs = schema.to_call_args(data)
        if schema.takes_context:
            return fn(ctx, *args, **kwargs)
        return fn(*args, **kwargs)

    return _HostTool(
        name=tool_id,
        description=schema.description or tool_id,
        args_schema=schema.params_model,
        func=func,
        yian_call=yian_call,
    )


def _tool_from_function_tool(ft: FunctionTool, *, description: str | None = None) -> _HostTool:
    """Build the tool for `ft`, named by its name.

    A tool from `function_tool` carries the function's pydantic model and return
    annotation, as a plain function tool does; any other carries the JSON Schema,
    which `ToolSet` checks before calling `on_invoke`. `description` replaces
    `ft.description`.
    """
    schema = ft._schema

    # Unannotated, so `tools.types` reads no return type from a JSON-Schema tool.
    def func(*args, **kwargs):
        raise TypeError(f"tool {ft.name!r} is a FunctionTool; invoke it through a run's ToolSet")

    if schema is not None and schema.return_annotation is not inspect.Signature.empty:
        func.__annotations__ = {"return": schema.return_annotation}

    def yian_call(ctx: ToolContext[Any], payload: dict[str, Any]) -> Any:
        # A deep copy, so `on_invoke` can't change values the caller still holds.
        result = ft.on_invoke(ctx, copy.deepcopy(payload))
        if inspect.isawaitable(result):
            if inspect.iscoroutine(result):
                result.close()  # Never awaited; closing avoids a "never awaited" warning.
            raise TypeError(
                f"tool {ft.name!r} returned an awaitable; yian calls tools synchronously, "
                "so on_invoke must return the result"
            )
        return result

    return _HostTool(
        name=ft.name,
        description=description or ft.description,
        args_schema=ft.params_json_schema if schema is None else schema.params_model,
        func=func,
        yian_call=yian_call,
    )


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
    duplicate labels are dropped, labels longer than
    `current().max_question_option_chars` are truncated, and at most
    `current().max_question_options` are kept. Returns `None` when no label is left.
    """
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except ValueError:
            parsed = None
        value = parsed if isinstance(parsed, list) else [value]
    if not isinstance(value, list):
        return None
    caps = current()
    labels: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            continue
        label = " ".join(item.split())
        if len(label) > caps.max_question_option_chars:
            label = label[: caps.max_question_option_chars - 1] + "…"
        if not label or label.casefold() in seen:
            continue
        seen.add(label.casefold())
        labels.append(label)
        if len(labels) == caps.max_question_options:
            break
    return labels or None


def _ask_user_tool(handler: Callable[..., Any], takes_context: bool) -> _HostTool:
    """Wrap a host handler in the fixed `ask_user` tool.

    The model-visible schema contains `question` and optional `options`, which are
    cleaned by `_normalize_options`. Structured question and display metadata are
    injected fields, so only a trusted invocation can set them. The handler always
    receives all six fields as keyword arguments, after the call's `ToolContext`
    when `takes_context` (from `_handler_takes_context`) is true.
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

    fields = tuple(inspect.signature(ask_user).parameters)

    def yian_call(ctx: ToolContext[Any], payload: dict[str, Any]) -> Any:
        # Validate again for the cleaned values (`options` normalized), as `invoke` would.
        data = tool.args_schema.model_validate(payload)
        request = {name: getattr(data, name) for name in fields}
        return handler(ctx, **request) if takes_context else handler(**request)

    tool = _HostTool.from_function(
        ask_user,
        name=_ASK_USER_ID,
        yian_call=yian_call,
        description=(
            "Ask the user one short question and wait for the answer. Use only when "
            "you need a decision or information that only the user can provide. Put "
            "the context they need in one or two sentences. When there are clear "
            "choices, pass 2-4 short `options`; the user can always type their own "
            "answer."
        ),
    )
    return tool


def _handler_takes_context(handler: Callable[..., Any]) -> bool:
    """Return whether `handler`'s first parameter is a positional `ToolContext`.

    Works for functions, bound methods, callable objects and partials. A handler
    with no readable signature takes no context.

    Raises:
        `TypeError`:
            Raised when an annotation can't be resolved and the first parameter's
            annotation mentions `ToolContext`.
    """
    try:
        signature = inspect.signature(handler, eval_str=True)
    except _ANNOTATION_ERRORS:
        # Some annotation can't be evaluated, or there is no signature at all; only
        # the first parameter's annotation decides.
        try:
            first = next(iter(inspect.signature(handler).parameters.values()), None)
        except (TypeError, ValueError):
            return False  # No readable signature: called with keywords only.
        if first is None or not _is_positional(first):
            return False
        annotation = first.annotation
        if isinstance(annotation, str):
            try:
                annotation = eval(annotation, _annotation_globals(handler))  # noqa: S307
            except _ANNOTATION_ERRORS as exc:
                if "ToolContext" not in first.annotation:
                    return False
                raise TypeError(
                    f"tools['ask_user'] handler parameter {first.name!r} is annotated "
                    f"{first.annotation!r}, which can't be resolved: {exc}"
                ) from exc
        return _is_tool_context(annotation)
    first = next(iter(signature.parameters.values()), None)
    return first is not None and _is_positional(first) and _is_tool_context(first.annotation)



def _annotation_globals(handler: Callable[..., Any]) -> dict[str, Any]:
    """Return the globals `handler`'s string annotations resolve in.

    Unwraps partials, a callable object's `__call__`, bound methods and
    `functools.wraps` wrappers down to the function that defines them.
    """
    target: Any = handler
    while isinstance(target, functools.partial):
        target = target.func
    if not inspect.isroutine(target):
        target = getattr(type(target), "__call__", target)
    target = inspect.unwrap(getattr(target, "__func__", target))
    return getattr(target, "__globals__", {})


def _is_positional(param: inspect.Parameter) -> bool:
    """Return whether `param` can take the first positional argument."""
    return param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD)


def _group_tool(group: str, index: int, item: Any) -> BaseTool:
    """Return one group item as a tool named by its `name` or `__name__`."""
    where = f"tools[{group!r}][{index}]"
    if isinstance(item, BaseTool):
        return item
    if isinstance(item, FunctionTool):
        return _tool_from_function_tool(item)
    if inspect.isroutine(item):
        if item.__name__ == "<lambda>":
            raise TypeError(
                f"{where} is a lambda, so it has no tool name; define it with def "
                "or wrap it in a BaseTool"
            )
        return _tool_from_function(item.__name__, item)
    raise TypeError(
        f"{where} must be a BaseTool, a FunctionTool, a function, or a method, "
        f"got {type(item).__name__}"
    )


def _same_item(first: Any, other: Any) -> bool:
    """Return whether two group items are one tool: the same object or bound method."""
    # Each `obj.method` access builds a new bound method, but equal ones share
    # their function and instance.
    return first is other or (inspect.ismethod(first) and first == other)


def _check_ask_user_handler(handler: Callable[..., Any]) -> bool:
    """Reject an `ask_user` handler that cannot take the six keyword fields.

    A handler whose first parameter is annotated `ToolContext` takes them after it.

    Returns:
        `bool`:
            Whether the handler takes the call's `ToolContext` first.

    Raises:
        `TypeError`:
            Raised when the handler can't take the six keyword fields, or as
            `_handler_takes_context` raises.
    """
    takes_context = _handler_takes_context(handler)
    try:
        signature = inspect.signature(handler)
    except (TypeError, ValueError):
        return takes_context  # No readable signature; the handler is called as is.
    leading = (None,) if takes_context else ()
    try:
        signature.bind(
            *leading,
            question="",
            questions=None,
            answer_schema=None,
            title=None,
            node_id=None,
            options=None,
        )
    except TypeError as exc:
        form = "ctx: ToolContext, **request" if takes_context else "**request"
        raise TypeError(
            "tools['ask_user'] must accept question, questions, answer_schema, title, "
            f"node_id, and options as keywords; define it as def handler({form})"
        ) from exc
    return takes_context


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

    A function or method tool whose first parameter is annotated `ToolContext` or
    `ToolContext[T]` receives a `ToolContext` on each call: the call id, attempt,
    tool name, run id, calling node, and this set's `context`. The model never
    sees that parameter. A `FunctionTool`'s `on_invoke` always receives one.

    Args:
        tools (`Mapping[str, BaseTool | FunctionTool | Callable[..., Any] | Sequence[BaseTool | FunctionTool | Callable[..., Any]]] | None`, *optional*, defaults to `None`):
            The run's tools keyed by tool id. A function or method becomes a tool
            named by its key; its type hints give the arguments, and its
            docstring's first text block and `Args` entries give the description
            and the parameter descriptions. It must be synchronous. To bind values
            such as a session, pass them as `context`, or use a closure or a bound
            method, not `functools.partial`. A `BaseTool` or `FunctionTool` must
            already use its key as its name. A list or tuple is a group keyed by its
            group name; each item is a `BaseTool` or `FunctionTool` named by its
            `name`, or a function or method named by its `__name__` (not a lambda).
            The `ask_user` entry must be a plain callable; it is wrapped so the
            model sees `question` and `options`, and it receives `question`,
            `questions`, `answer_schema`, `title`, `node_id`, and `options` as
            keywords (define it as `def handler(**request)`, or
            `def handler(ctx: ToolContext, **request)` to receive the context
            first). `None` means the global registry. To keep the registered tools
            and add more, pass `{**registered_tools(), ...}`.
        context (`Any`, *optional*, defaults to `None`):
            The host's app object, passed to tools as `ToolContext.context`. It is
            never recorded.

    Raises:
        `TypeError`:
            `tools` is not a mapping or `None`, a key is not a non-empty string,
            an entry is not a `BaseTool`, `FunctionTool`, function, method, or
            group of them, a group item is a lambda, a function is async, an
            annotation can't be resolved, a function puts `ToolContext` anywhere
            but its first parameter, or the `ask_user` handler cannot take its six
            keyword fields.
        `ValueError`:
            A `BaseTool` or `FunctionTool` entry's name differs from its key, or
            a group tool's name is `ask_user` or `load_skill`, is a top-level key,
            or names a different tool elsewhere in the groups.

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

        def place(ctx: ToolContext[Broker], symbol: str) -> str:
            "Place one order."
            return ctx.context.submit(symbol, idempotency_key=ctx.call_id)

        run_tools = ToolSet({"place": place}, context=broker).for_run("run-1")
        run_tools.invoke("place", {"symbol": "X"}, node="buy")  # call_id "run-1/buy"
        ```
    """

    __slots__ = ("_tools", "_groups", "_run_id", "_context")

    def __init__(
        self,
        tools: Mapping[str, _Entry | Sequence[_Entry]] | None = None,
        *,
        context: Any = None,
    ) -> None:
        self._tools: MappingProxyType[str, BaseTool] | None = None
        self._groups: MappingProxyType[str, tuple[str, ...]] = MappingProxyType({})
        self._run_id = ""
        self._context = context
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
                built[tool_id] = _ask_user_tool(entry, _check_ask_user_handler(entry))
            elif isinstance(entry, BaseTool):
                if entry.name != tool_id:
                    raise ValueError(
                        f"tool {tool_id!r} is a BaseTool named {entry.name!r}; "
                        "its name must match its key"
                    )
                built[tool_id] = entry
            elif isinstance(entry, FunctionTool):
                if entry.name != tool_id:
                    raise ValueError(
                        f"tool {tool_id!r} is a FunctionTool named {entry.name!r}; "
                        "its name must match its key"
                    )
                built[tool_id] = _tool_from_function_tool(entry)
            elif inspect.isroutine(entry):
                built[tool_id] = _tool_from_function(tool_id, entry)
            elif isinstance(entry, (list, tuple)):
                group_items[tool_id] = [
                    (item, _group_tool(tool_id, index, item)) for index, item in enumerate(entry)
                ]
            else:
                raise TypeError(
                    f"tools[{tool_id!r}] must be a BaseTool, a FunctionTool, a function, a "
                    f"method, or a list of them, got {type(entry).__name__}; bind values in "
                    "a def or lambda, not functools.partial"
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

    def for_run(self, run_id: str) -> ToolSet:
        """Return a copy bound to one run, sharing this set's tools and context.

        Args:
            run_id (`str`):
                The run's id, passed to tools as `ToolContext.run_id` and the base
                of each call's default `call_id`.

        Returns:
            `ToolSet`:
                A new tool set; this one stays as it was.
        """
        bound = type(self).__new__(type(self))
        bound._tools = self._tools
        bound._groups = self._groups
        bound._run_id = run_id
        bound._context = self._context
        return bound

    @property
    def run_id(self) -> str:
        """The bound run's id, or `""` when `for_run` has not bound one."""
        return self._run_id

    @property
    def context(self) -> Any:
        """The host's app object passed to tools as `ToolContext.context`."""
        return self._context

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

    def invoke(
        self,
        tool_id: str,
        inputs: dict[str, Any],
        *,
        node: str = "",
        call_id: str | None = None,
        attempt: int = 1,
    ) -> Any:
        """Validate and invoke one tool through its model-visible public schema.

        Unknown keys are rejected before Pydantic validation because tool schemas
        may otherwise ignore them. Exceptions from validation and the tool
        propagate unchanged.

        Args:
            tool_id (`str`):
                The id of the tool to invoke.
            inputs (`dict[str, Any]`):
                Public arguments keyed by parameter name.
            node (`str`, *optional*, defaults to `""`):
                The expanded id of the calling node, passed as `ToolContext.node`.
            call_id (`str`, *optional*, defaults to `None`):
                The call's id. `None` uses `"<run_id>/<node>"` on a run-bound set
                given a `node`, and a fresh random hex id otherwise.
            attempt (`int`, *optional*, defaults to `1`):
                The attempt number of this call, starting at 1.

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
        return self._call(tool, tool_id, payload, node=node, call_id=call_id, attempt=attempt)

    def invoke_trusted(
        self,
        tool_id: str,
        inputs: dict[str, Any],
        *,
        node: str = "",
        call_id: str | None = None,
        attempt: int = 1,
    ) -> Any:
        """Invoke one tool through its full schema, including host-injected fields.

        Args:
            tool_id (`str`):
                The id of the tool to invoke.
            inputs (`dict[str, Any]`):
                Public and injected arguments keyed by parameter name.
            node (`str`, *optional*, defaults to `""`):
                The expanded id of the calling node, passed as `ToolContext.node`.
            call_id (`str`, *optional*, defaults to `None`):
                The call's id. `None` uses `"<run_id>/<node>"` on a run-bound set
                given a `node`, and a fresh random hex id otherwise.
            attempt (`int`, *optional*, defaults to `1`):
                The attempt number of this call, starting at 1.

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
        return self._call(tool, tool_id, payload, node=node, call_id=call_id, attempt=attempt)

    def _call(
        self,
        tool: BaseTool,
        tool_id: str,
        payload: dict[str, Any],
        *,
        node: str,
        call_id: str | None,
        attempt: int,
    ) -> Any:
        """Call a validated tool: a yian-built tool with its context, any other through `invoke`.

        Host tools run directly, not inside LangChain's `invoke`, so LangChain
        callbacks and tracing don't fire for them.
        """
        if not isinstance(tool, _HostTool):
            return tool.invoke(payload)
        if call_id is None:
            # Without a run and a node nothing makes two calls the same call, so
            # each gets its own id.
            call_id = f"{self._run_id}/{node}" if self._run_id and node else uuid4().hex
        ctx = ToolContext(
            call_id=call_id,
            attempt=attempt,
            tool_name=tool_id,
            run_id=self._run_id,
            node=node,
            context=self._context,
        )
        return tool.yian_call(ctx, payload)


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


def _validate_invocation(
    tool_id: str,
    tool: BaseTool,
    inputs: dict[str, Any],
    *,
    trusted: bool,
) -> None:
    """Reject extra keys, then validate a supported Pydantic argument schema.

    A `FunctionTool`'s JSON Schema is checked by `_validate_json_arguments`; any
    other tool with a JSON Schema is rejected.
    """
    schema = tool.args_schema if trusted else tool.tool_call_schema
    if isinstance(schema, dict):
        if isinstance(tool, _HostTool):
            _validate_json_arguments(tool_id, schema, inputs)
            return
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


def _validate_json_arguments(
    tool_id: str, schema: Mapping[str, Any], inputs: dict[str, Any]
) -> None:
    """Check call arguments against a `FunctionTool`'s JSON Schema.

    Unknown keys are rejected unless `additionalProperties` is `true` or a schema,
    in which case extra values are checked against that schema. Missing required
    keys are rejected. Each value is checked against the yian type its JSON Schema
    translates to; a schema with no translation is not checked. An integral float
    is accepted wherever that type expects an integer, including list items and
    record fields; the value itself is passed on as given.

    Raises:
        `ValueError`:
            Raised on an unknown or missing key.
        `TypeCheckError`:
            Raised when a value doesn't fit its property's translated type.
    """
    properties = schema.get("properties")
    if not isinstance(properties, Mapping):
        properties = {}
    extra = schema.get("additionalProperties")
    open_extra = extra is True or isinstance(extra, Mapping)
    unsupported = sorted(set(inputs) - set(properties))
    if unsupported and not open_extra:
        raise ValueError(
            f"tool {tool_id!r} received unsupported argument(s) {unsupported}; "
            f"allowed: {sorted(properties)}"
        )
    required = schema.get("required")
    missing = [
        name
        for name in (required if isinstance(required, list) else [])
        if isinstance(name, str) and name not in inputs
    ]
    if missing:
        raise ValueError(f"tool {tool_id!r} is missing required argument(s) {missing}")
    for name, value in inputs.items():
        prop = properties.get(name, extra if isinstance(extra, Mapping) else None)
        if prop is None:
            continue
        typ = _input_type_from_json_schema(prop, root=schema)
        if typ is None:
            continue
        try:
            build_value_as(typ, _integral_floats_as_ints(typ, value))
        except TypeCheckError as exc:
            raise TypeCheckError(
                f"tool {tool_id!r} argument {name!r}: {exc}", path=(name, *exc.path)
            ) from exc


def _integral_floats_as_ints(typ: Type, value: Any) -> Any:
    """Return `value` with each integral float in an integer position made an `int`.

    JSON decoders and models may send `3.0` for an integer, which JSON Schema
    accepts; yian's integer type does not. Lists and records are copied, not changed.
    """
    if typ.kind is ValueKind.INTEGER:
        return int(value) if type(value) is float and value.is_integer() else value
    if typ.element is not None and isinstance(value, list):
        return [_integral_floats_as_ints(typ.element, item) for item in value]
    if typ.fields is not None and isinstance(value, dict):
        return {
            key: _integral_floats_as_ints(typ.fields[key], item) if key in typ.fields else item
            for key, item in value.items()
        }
    return value


def invoke_tool(tool_id: str, inputs: dict[str, Any]) -> Any:
    """Validate and invoke one registered tool through its model-visible public schema.

    Unknown keys are rejected before Pydantic validation because tool schemas may
    otherwise ignore them. Exceptions from validation and the tool propagate to the
    node boundary unchanged. A tool that takes a `ToolContext` gets one outside any
    run: `run_id` and `node` are `""` and `call_id` is a fresh random hex id.
    """
    return ToolSet().invoke(tool_id, inputs)


def invoke_tool_trusted(tool_id: str, inputs: dict[str, Any]) -> Any:
    """Invoke one registered tool through its full schema, including host-injected fields.

    A tool that takes a `ToolContext` gets one outside any run, as with `invoke_tool`.
    """
    return ToolSet().invoke_trusted(tool_id, inputs)
