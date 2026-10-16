"""Build a tool's argument model and description from a plain Python function.

This ports the OpenAI Agents SDK's ``function_schema`` in its non-strict mode:
parameter types come from the resolved type hints, the description and
per-parameter descriptions from the docstring (``tools.docstrings``), an
``Annotated`` string describes a parameter the docstring doesn't, ``Field``
defaults and ``Annotated`` ``Field`` constraints are kept, and ``*args`` /
``**kwargs`` become a list field and an object field. A first parameter
annotated ``ToolContext`` (``tools.context``) is left out of the model; yian
passes it when it calls the tool.

Knows about: pydantic (v1 only to reject its models), ``tools.context``,
``tools.docstrings``, and the Python standard library.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from typing import Annotated, Any, get_args, get_origin, get_type_hints

from pydantic import BaseModel, ConfigDict, Field, create_model
from pydantic.fields import FieldInfo
from pydantic.v1 import BaseModel as BaseModelV1

from yian.tools.context import ToolContext
from yian.tools.docstrings import parse_docstring


@dataclass(frozen=True)
class FuncSchema:
    """A tool function's name, description, argument model and call shape.

    Args:
        name (`str`):
            The tool name: the override, or the function's `__name__`.
        description (`str` or `None`):
            The override, or the docstring's first text block. `None` when neither exists.
        params_model (`type[BaseModel]`):
            The model that validates the tool's arguments, named after the tool.
            It excludes the context parameter.
        signature (`inspect.Signature`):
            The function's signature (a bound method's excludes `self`).
        takes_context (`bool`):
            Whether the first parameter receives a `ToolContext`.
        return_annotation (`Any`):
            The resolved return annotation, with `Annotated` metadata, or
            `inspect.Signature.empty` when there is none.
    """

    name: str
    description: str | None
    params_model: type[BaseModel]
    signature: inspect.Signature
    takes_context: bool
    return_annotation: Any

    @cached_property
    def params_json_schema(self) -> dict[str, Any]:
        """The JSON Schema of `params_model`, built on first read.

        It is lazy because a model with an arbitrary class field validates but has
        no JSON Schema; building it raises pydantic's `PydanticInvalidForJsonSchema`.
        """
        return self.params_model.model_json_schema()

    def to_call_args(self, data: BaseModel) -> tuple[list[Any], dict[str, Any]]:
        """Split validated arguments into the positional and keyword arguments of a call.

        The context parameter is skipped; the caller passes it first.

        Args:
            data (`BaseModel`):
                An instance of `params_model`.

        Returns:
            `tuple[list[Any], dict[str, Any]]`:
                The positional arguments and the keyword arguments.

        Raises:
            `ValueError`:
                Raised when the `**kwargs` field has a key that names one of the
                function's own keyword-bindable parameters.
        """
        positional: list[Any] = []
        keyword: dict[str, Any] = {}
        seen_var_positional = False
        # Instance storage first, so pydantic attributes such as `model_extra` don't
        # shadow a parameter of the same name; `model_dump()` would turn nested models
        # into dicts.
        values = object.__getattribute__(data, "__dict__")
        for index, (name, param) in enumerate(self.signature.parameters.items()):
            if self.takes_context and index == 0:
                continue
            value = values[name] if name in values else getattr(data, name, None)
            if param.kind == param.VAR_POSITIONAL:
                positional.extend(value or [])
                seen_var_positional = True
            elif param.kind == param.VAR_KEYWORD:
                extra = value or {}
                self._raise_on_var_keyword_collisions(name, extra)
                keyword.update(extra)
            elif param.kind in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD):
                if seen_var_positional:
                    keyword[name] = value
                else:
                    positional.append(value)
            else:
                keyword[name] = value
        return positional, keyword

    def _raise_on_var_keyword_collisions(self, var_name: str, values: dict[str, Any]) -> None:
        """Reject `**kwargs` keys that name a parameter the call already binds by name.

        `**kwargs` is splatted last, so such a key would either replace the value
        the model supplied for that parameter or fail with "got multiple values".
        Positional-only parameters are not reserved: `f(1, a=2)` routes `a` into
        `**kwargs` for `def f(a, /, **kw)`.
        """
        reserved = {
            name
            for name, param in self.signature.parameters.items()
            if param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY)
        }
        conflicts = sorted(reserved.intersection(values))
        if conflicts:
            listed = ", ".join(repr(name) for name in conflicts)
            verb = "is" if len(conflicts) == 1 else "are"
            raise ValueError(
                f"invalid arguments for tool {self.name!r}: {listed} {verb} both a named "
                f"parameter and a key in {var_name!r}; pass each argument once, as a named "
                "parameter"
            )


def function_schema(
    func: Callable[..., Any],
    *,
    name_override: str | None = None,
    description_override: str | None = None,
) -> FuncSchema:
    """Build the schema of a tool function from its signature, type hints and docstring.

    Args:
        func (`Callable[..., Any]`):
            A synchronous function or bound method. Every annotation must resolve.
            Its first parameter may be annotated `ToolContext` or `ToolContext[T]`
            (or a subclass); no other parameter may mention `ToolContext`.
        name_override (`str`, *optional*, defaults to `None`):
            The tool name. `None` uses `func.__name__`, which must then exist.
        description_override (`str`, *optional*, defaults to `None`):
            The tool description. `None` uses the docstring's first text block.

    Returns:
        `FuncSchema`:
            The name, description, argument model and call shape.

    Raises:
        `TypeError`:
            Raised when `func` has no name, is async, an annotation can't be
            resolved, `ToolContext` appears anywhere other than as the first
            positional parameter's own type, a parameter is annotated with a
            pydantic v1 model, `*args` is a fixed-length tuple, or pydantic can't
            build the argument model or drops a parameter from it (for example a
            `_x` or `model_config` parameter).

    Example:
        ```python
        def place(ctx: ToolContext[Broker], symbol: str, qty: int = 1) -> str:
            ...

        schema = function_schema(place, description_override="Place an order.")
        assert schema.takes_context
        assert schema.params_json_schema["required"] == ["symbol"]
        ```
    """
    name = name_override or getattr(func, "__name__", None)
    if not name:
        raise TypeError(f"tool callable {func!r} has no __name__; pass name_override")
    # Callable objects such as functools.partial have no `__name__`; messages use the tool name.
    func_name = getattr(func, "__name__", None) or name
    if inspect.iscoroutinefunction(func):
        raise TypeError(
            f"tool function {func_name!r} is async; yian calls tools synchronously, "
            "so define it with def"
        )
    doc = parse_docstring(inspect.getdoc(func))
    param_descs = dict(doc.params)

    try:
        hints = get_type_hints(func, include_extras=True)
    except Exception as exc:
        raise TypeError(
            f"tool function {func_name!r} has an annotation that can't be resolved: {exc}"
        ) from exc
    stripped: dict[str, Any] = {}
    for param_name, hint in hints.items():
        if param_name == "return":
            continue
        base, metadata = _strip_annotated(hint)
        stripped[param_name] = base
        annotated_desc = next((item for item in metadata if isinstance(item, str)), None)
        if annotated_desc is not None:
            param_descs.setdefault(param_name, annotated_desc)

    sig = inspect.signature(func)
    params = list(sig.parameters.values())
    takes_context = False
    if params and _is_tool_context(stripped.get(params[0].name)):
        if params[0].kind not in (params[0].POSITIONAL_ONLY, params[0].POSITIONAL_OR_KEYWORD):
            raise TypeError(
                f"tool function {func_name!r} declares ToolContext parameter "
                f"{params[0].name!r} as {params[0].kind.description}; it must be a "
                "positional first parameter"
            )
        takes_context = True
    for position, param in enumerate(params, start=1):
        if takes_context and position == 1:
            continue
        hint = hints.get(param.name)
        if _is_tool_context(stripped.get(param.name)):
            raise TypeError(
                f"tool function {func_name!r} declares ToolContext parameter {param.name!r} "
                f"at position {position}; it must be the first parameter"
            )
        if _mentions_tool_context(hint):
            raise TypeError(
                f"tool function {func_name!r} parameter {param.name!r} wraps ToolContext in "
                f"{hint!r}; only a first parameter annotated ToolContext or ToolContext[T] "
                "receives the context"
            )
        if _mentions_pydantic_v1(hint):
            # Pydantic v2 validates a v1 model field by calling its v2-style hooks, so
            # every call would fail with an unrelated TypeError.
            raise TypeError(
                f"tool {name!r} parameter {param.name!r} is annotated with a pydantic v1 "
                f"model ({hint!r}); use a pydantic v2 model, or wrap the tool as a "
                "LangChain BaseTool with a v1 args_schema"
            )

    fields: dict[str, Any] = {}
    config = ConfigDict(arbitrary_types_allowed=True)
    for param in (params[1:] if takes_context else params):
        fields[param.name] = _field(func_name, param, hints, stripped, param_descs.get(param.name))
        if param.kind == param.VAR_KEYWORD:
            # Flat keyword arguments would otherwise be dropped before the call.
            config["extra"] = "forbid"

    try:
        model = create_model(name, __config__=config, **fields)
    except Exception as exc:
        raise TypeError(
            f"tool function {func_name!r} can't become tool {name!r}: {exc}"
        ) from exc
    # Pydantic takes some names (`model_config`) as model settings rather than fields;
    # a parameter that vanished from the model would never receive the model's value.
    for field_name in fields:
        if field_name not in model.model_fields:
            raise TypeError(
                f"tool function {func_name!r} parameter {field_name!r} can't be a field of "
                f"tool {name!r}; rename the parameter"
            )

    return FuncSchema(
        name=name,
        description=description_override or doc.description,
        params_model=model,
        signature=sig,
        takes_context=takes_context,
        return_annotation=hints.get("return", sig.return_annotation),
    )


def _field(
    func_name: str,
    param: inspect.Parameter,
    hints: dict[str, Any],
    stripped: dict[str, Any],
    description: str | None,
) -> Any:
    """Return the `create_model` definition of one parameter."""
    hint = hints.get(param.name, Any)
    base = stripped.get(param.name, Any)
    if param.kind == param.VAR_POSITIONAL:
        # `*args: X` types each positional argument as X (PEP 484), so the field is a list of X.
        item = _variadic_item(base, hint)
        if get_origin(base) is tuple:
            tuple_args = getattr(base, "__args__", None)
            if tuple_args is not None and not (len(tuple_args) == 2 and tuple_args[1] is Ellipsis):
                raise TypeError(
                    f"tool function {func_name!r} annotates *{param.name} with the "
                    f"fixed-length tuple {base!r}; a variadic annotation describes each "
                    "positional argument, so use tuple[T, ...] or list[T]"
                )
            if tuple_args is None:
                # A bare `typing.Tuple` carries no element type.
                item = Any
        return (list[item], Field(default_factory=list, description=description))
    if param.kind == param.VAR_KEYWORD:
        item = _variadic_item(base, hint)
        return (dict[str, item], Field(default_factory=dict, description=description))

    # Pydantic merges the `Field`s inside `Annotated` left to right and the assigned
    # `Field` last, keeping only the attributes each one sets. So a `Field(...)` default
    # joins the annotation, and the assigned `Field` sets only a plain default and the
    # docstring (or `Annotated` string) description, which then win. The assigned value
    # is always a `Field` so pydantic rejects a `_x` name instead of making it private.
    annotation = hint
    overrides: dict[str, Any] = {}
    if isinstance(param.default, FieldInfo):
        annotation = Annotated[hint, param.default]
    elif param.default is not inspect.Parameter.empty:
        overrides["default"] = param.default
    if description is not None:
        overrides["description"] = description
    return (annotation, Field(**overrides))


def _variadic_item(base: Any, hint: Any) -> Any:
    """Return the type of each `*args` or `**kwargs` value.

    Only the `Annotated` constraints apply to each value; a description or default
    describes the whole parameter, not its items.
    """
    if get_origin(hint) is not Annotated:
        return base
    constraints = FieldInfo.from_annotation(hint).metadata
    return Annotated[(base, *constraints)] if constraints else base


def _strip_annotated(annotation: Any) -> tuple[Any, tuple[Any, ...]]:
    """Return the type under any `Annotated` layers and their metadata."""
    metadata: tuple[Any, ...] = ()
    while get_origin(annotation) is Annotated:
        args = get_args(annotation)
        annotation = args[0]
        metadata = (*metadata, *args[1:])
    return annotation, metadata


def _is_tool_context(annotation: Any) -> bool:
    """Whether `annotation` is `ToolContext`, `ToolContext[T]`, or a subclass."""
    origin = get_origin(annotation) or annotation
    return isinstance(origin, type) and issubclass(origin, ToolContext)


def _mentions_tool_context(annotation: Any) -> bool:
    """Whether `annotation` is or contains `ToolContext` anywhere in its arguments."""
    if isinstance(annotation, (list, tuple)):
        return any(_mentions_tool_context(item) for item in annotation)
    if _is_tool_context(annotation):
        return True
    return any(_mentions_tool_context(arg) for arg in get_args(annotation))


def _mentions_pydantic_v1(annotation: Any) -> bool:
    """Whether `annotation` is or contains a pydantic v1 model anywhere in its arguments."""
    if isinstance(annotation, (list, tuple)):
        return any(_mentions_pydantic_v1(item) for item in annotation)
    if isinstance(annotation, type) and issubclass(annotation, BaseModelV1):
        return True
    return any(_mentions_pydantic_v1(arg) for arg in get_args(annotation))
