"""Host tools given as a name, a description, a JSON Schema and a call.

``FunctionTool`` describes a tool without a Python signature: the host supplies
the argument JSON Schema and an ``on_invoke(ctx, args)`` callable. ``function_tool``
builds one from a plain function (``tools.schema.function_schema``), optionally
renaming it or replacing its description. ``ToolSet`` and ``register_tool`` accept
either form.

Knows about: pydantic, ``tools.context``, ``tools.schema``, and the Python
standard library.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, overload

from pydantic.errors import PydanticInvalidForJsonSchema

from yian.tools.context import ToolContext
from yian.tools.schema import FuncSchema, function_schema


@dataclass(frozen=True)
class FunctionTool:
    """A host tool given by its name, description, argument JSON Schema and call.

    `ToolSet` checks each call's arguments against `params_json_schema` before
    calling `on_invoke`: unknown keys (unless `additionalProperties` is `true` or
    a schema), missing required keys, and values whose JSON Schema yian can
    translate to a type. A property yian can't translate is passed unchecked.

    Args:
        name (`str`):
            The tool name. A `ToolSet` key or `register_tool` id must equal it.
        description (`str`):
            The description the model sees.
        params_json_schema (`dict[str, Any]`):
            The arguments' JSON Schema; its `type` must be `"object"`.
        on_invoke (`Callable[[ToolContext[Any], dict[str, Any]], Any]`):
            A synchronous callable that receives the call's `ToolContext` and a
            deep copy of the checked arguments as a plain `dict`, and returns the
            result. Changing that copy doesn't change the caller's values.

    Raises:
        `TypeError`:
            `name` or `description` is not a string, `params_json_schema` is not a
            dict, or `on_invoke` is not callable or is async.
        `ValueError`:
            `name` is empty, or `params_json_schema`'s `type` is not `"object"`.

    Example:
        ```python
        def echo(ctx: ToolContext[Any], args: dict[str, Any]) -> str:
            return args["text"]

        tool = FunctionTool(
            name="echo",
            description="Echo the text.",
            params_json_schema={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
            on_invoke=echo,
        )
        ToolSet({"echo": tool}).invoke("echo", {"text": "hi"})  # "hi"
        ```
    """

    name: str
    description: str
    params_json_schema: dict[str, Any]
    on_invoke: Callable[[ToolContext[Any], dict[str, Any]], Any]
    # Set by `function_tool`, so a tool built from a function validates with the
    # function's pydantic model and reports its return annotation. It is not an init
    # field, so `dataclasses.replace()` drops it: the copy is a plain JSON-Schema tool,
    # which stays correct when the replace changes `on_invoke` or the schema.
    _schema: FuncSchema | None = field(default=None, init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise TypeError(f"FunctionTool name must be a string, got {type(self.name).__name__}")
        if not self.name:
            raise ValueError("FunctionTool name must not be empty")
        if not isinstance(self.description, str):
            raise TypeError(
                f"FunctionTool {self.name!r} description must be a string, "
                f"got {type(self.description).__name__}"
            )
        if not isinstance(self.params_json_schema, dict):
            raise TypeError(
                f"FunctionTool {self.name!r} params_json_schema must be a dict, "
                f"got {type(self.params_json_schema).__name__}"
            )
        if self.params_json_schema.get("type") != "object":
            raise ValueError(
                f"FunctionTool {self.name!r} params_json_schema must have type 'object'"
            )
        if not callable(self.on_invoke):
            raise TypeError(f"FunctionTool {self.name!r} on_invoke must be callable")
        if inspect.iscoroutinefunction(self.on_invoke) or inspect.iscoroutinefunction(
            getattr(self.on_invoke, "__call__", None)
        ):
            raise TypeError(
                f"FunctionTool {self.name!r} on_invoke is async; yian calls tools "
                "synchronously, so define it with def"
            )


@overload
def function_tool(
    func: Callable[..., Any],
    *,
    name_override: str | None = None,
    description_override: str | None = None,
) -> FunctionTool: ...


@overload
def function_tool(
    func: None = None,
    *,
    name_override: str | None = None,
    description_override: str | None = None,
) -> Callable[[Callable[..., Any]], FunctionTool]: ...


def function_tool(
    func: Callable[..., Any] | None = None,
    *,
    name_override: str | None = None,
    description_override: str | None = None,
) -> FunctionTool | Callable[[Callable[..., Any]], FunctionTool]:
    """Build a `FunctionTool` from a function's signature, type hints and docstring.

    Use it as a call, `@function_tool`, or `@function_tool(name_override=...)`. The
    tool validates its arguments with the pydantic model built from the function
    (lax coercion, as for a plain function tool) and passes the call's
    `ToolContext` when the function's first parameter is annotated `ToolContext`.

    Args:
        func (`Callable[..., Any]`, *optional*, defaults to `None`):
            A synchronous function or bound method. `None` returns a decorator.
        name_override (`str`, *optional*, defaults to `None`):
            The tool name. `None` uses `func.__name__`.
        description_override (`str`, *optional*, defaults to `None`):
            The tool description. `None` uses the docstring's first text block,
            and the tool name when there is none.

    Returns:
        `FunctionTool | Callable[[Callable[..., Any]], FunctionTool]`:
            The tool, or a decorator that builds it when `func` is `None`.

    Raises:
        `TypeError`:
            Raised as `function_schema` raises (async function, unresolved
            annotation, misplaced `ToolContext`), or when an argument type has no
            JSON Schema.

    Example:
        ```python
        @function_tool(name_override="buy", description_override="Buy shares.")
        def place(ctx: ToolContext[Broker], symbol: str, qty: int = 1) -> str:
            return ctx.context.submit(symbol, qty, idempotency_key=ctx.call_id)

        ToolSet({"buy": place}, context=broker)
        ```
    """
    if func is None:

        def decorator(fn: Callable[..., Any]) -> FunctionTool:
            return function_tool(
                fn, name_override=name_override, description_override=description_override
            )

        return decorator

    schema = function_schema(
        func, name_override=name_override, description_override=description_override
    )
    try:
        params_json_schema = schema.params_json_schema
    except PydanticInvalidForJsonSchema as exc:
        raise TypeError(
            f"tool {schema.name!r} has an argument type with no JSON Schema: {exc}"
        ) from exc

    def on_invoke(ctx: ToolContext[Any], args: dict[str, Any]) -> Any:
        data = schema.params_model.model_validate(args)
        call_args, call_kwargs = schema.to_call_args(data)
        if schema.takes_context:
            return func(ctx, *call_args, **call_kwargs)
        return func(*call_args, **call_kwargs)

    tool = FunctionTool(
        name=schema.name,
        description=schema.description or schema.name,
        params_json_schema=params_json_schema,
        on_invoke=on_invoke,
    )
    object.__setattr__(tool, "_schema", schema)
    return tool
