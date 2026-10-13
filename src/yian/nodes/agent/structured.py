"""Structured AGENT output — derive a pydantic schema from a declared `output:` Type and
generate a value that conforms to it.

An AGENT declares `output:` as a typed `Type`. For a bare `str` the agent stays a text
producer — `type_to_schema` returns `None` and the AGENT loop takes the plain-text path. For
any constrained or richer type (a `Literal[...]` variant, record, scalar
`int`/`float`/`bool`, or list) `type_to_schema` builds a `pydantic.BaseModel` the
AGENT loop hands to `with_structured_output`, so the model emits a value that
success-record preparation can canonicalize against the effective target's declared
output type.

Scalars, unions and lists can't be a top-level pydantic model field on their own, so they are
wrapped in a single-field model (`{"value": <scalar>}` / `{"items": [<element>]}`); `_unwrap`
strips the wrapper back to the bare value after generation. A record maps one model field per
declared field and passes through as the dumped dict. A union is a plain `Union[...]` of
member models that reject unknown fields, so its JSON schema is an `anyOf` of `$ref`s with a
`const` per tag value.

Optional memory-owned turns retain raw API replies before parsing. Provider
context and newly retained deltas stay separate across self-correction attempts.

Layer: nodes — imports memory contracts, the message adapter, `typesys` and
`pydantic`; no engine/runtime imports.
"""

from __future__ import annotations

from concurrent.futures import CancelledError as FutureCancelledError
import json
from typing import Annotated, Any, Callable, List, Literal, Optional, Union
from uuid import uuid4

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model

from yian.memory.contracts import (
    MemoryContext,
    EvaluationTrace,
    ModelValueInteraction,
)
from yian.nodes.agent.memory import invoke_model
from yian.nodes.agent.live import LiveSinks, invoke_chat, invoke_structured
from yian.nodes.agent.text import record_model_text, text_of
from yian.nodes.base import _CapabilityFault, _HostStop
from yian.typesys.values import Type, ValueKind, union_tag_fields

# ValueKind -> python type for a scalar slot. DATE/DATETIME persist as ISO strings.
_SCALAR_PY: dict[ValueKind, type] = {
    ValueKind.STRING: str,
    ValueKind.INTEGER: int,
    ValueKind.NUMBER: float,
    ValueKind.BOOLEAN: bool,
    ValueKind.DATE: str,
    ValueKind.DATETIME: str,
}


def _py_type(typ: Type, *, closed: bool = False) -> Any:
    """The python annotation for a value of `typ` as a pydantic field type.

    A `STRING` with `tags` is a `Literal[...]` variant; a nested `OBJECT` with `fields`
    becomes its own submodel; a union becomes a plain `Union[...]` of closed member models
    named `Member1`, `Member2`, ... in member order; a `LIST_*` becomes
    `list[<element type>]`.

    Args:
        typ (`Type`):
            The declared type of the slot.
        closed (`bool`, *optional*, defaults to `False`):
            Whether records reject unknown fields. Union members and everything inside
            them are always closed, so a reply can't match a member by ignoring fields.

    Returns:
        `Any`:
            The pydantic annotation.

    Raises:
        `ValueError`:
            `typ` has no structured mapping (`NONE`, `FILE`), so a new ValueKind is loud.
    """
    seg = typ.kind
    if seg == ValueKind.STRING and typ.tags:
        return Literal[tuple(sorted(typ.tags))]  # type: ignore[valid-type]
    if seg in _SCALAR_PY:
        if typ.bounds is not None:
            typ.validate_shape()
            annotation = Annotated[_SCALAR_PY[seg], Field(ge=typ.bounds[0], le=typ.bounds[1])]
            return Optional[annotation] if typ.nullable else annotation
        return _SCALAR_PY[seg]
    if seg == ValueKind.OBJECT and typ.variants is not None:
        members = tuple(
            _record_model(member, name=_member_name(index), closed=True)
            for index, member in enumerate(typ.variants)
        )
        return Union[members]  # type: ignore[valid-type]
    if seg == ValueKind.OBJECT:
        return _record_model(typ, closed=closed) if typ.fields else dict
    if seg == ValueKind.LIST_ANY and typ.element is None:
        return list
    if seg.is_list():
        elem = typ.element
        if elem is not None:
            return List[_py_type(elem, closed=closed)]  # type: ignore[misc]
        scalar = seg.element_kind
        return List[_SCALAR_PY[scalar]] if scalar in _SCALAR_PY else list  # type: ignore[misc]
    raise ValueError(f"type_to_schema: no structured mapping for segment type {seg!r}")


def _member_name(index: int) -> str:
    """Name the model of the union member at `index`; error locations carry this name."""
    return f"Member{index + 1}"


def _record_model(typ: Type, *, name: str = "Record", closed: bool = False) -> type[BaseModel]:
    """Build a pydantic model from an `OBJECT` Type.

    Each `fields` entry is a model field, required iff named in `required` (others
    default to `None`); `nullable` widens a field to Optional.

    Args:
        typ (`Type`):
            A record type.
        name (`str`, *optional*, defaults to `"Record"`):
            The model name, which is the schema title.
        closed (`bool`, *optional*, defaults to `False`):
            Whether the model and the records inside it reject unknown fields.

    Returns:
        `type[BaseModel]`:
            The generated model.
    """
    required = typ.required or frozenset()
    spec: dict[str, Any] = {}
    for field_name, sub in (typ.fields or {}).items():
        ann = _py_type(sub, closed=closed)
        if field_name in required and not sub.nullable:
            spec[field_name] = (ann, ...)
        else:
            spec[field_name] = (Optional[ann], None)
    config = ConfigDict(extra="forbid") if closed else None
    return create_model(name, __config__=config, **spec)


def type_to_schema(typ: Type) -> Optional[type[BaseModel]]:
    """Derive a pydantic model for `typ`, or `None` when the agent should stay a text
    producer (a bare, untagged `str`).

    A record (`OBJECT` with `fields`) maps directly to a model. A scalar, a union or a list
    can't be a standalone model, so it is wrapped in a single-field model (`value` /
    `items`) that `_unwrap` later strips.
    """
    seg = typ.kind
    if seg == ValueKind.STRING and not typ.tags:
        return None  # unconstrained text stays on the ordinary generation path
    if seg == ValueKind.OBJECT and typ.fields:
        return _record_model(typ)
    if seg.is_list() or seg == ValueKind.LIST_ANY:
        return create_model("ListWrapper", items=(_py_type(typ), ...))
    # any other scalar (int/float/bool/date/datetime), union or freeform object -> value wrapper
    return create_model("ScalarWrapper", value=(_py_type(typ), ...))


def _unwrap(obj: BaseModel, typ: Type) -> Any:
    """Strip the single-field wrapper `type_to_schema` adds for a scalar/union/list, returning
    the bare value the write boundary expects. A record is returned as its dumped dict, and a
    union as the matching member's dumped dict."""
    data = obj.model_dump()
    seg = typ.kind
    if seg == ValueKind.OBJECT and typ.fields:
        return data
    if seg.is_list() or seg == ValueKind.LIST_ANY:
        return data["items"]
    return data["value"]


def _supports_native_structured(llm_config: Any) -> bool:
    """Whether the node's effective `(provider, model)` can use native structured output.

    Reads `provider`/`model` off the effective config (a plain dict on the run path, or an
    `LLMConfig` in some unit harnesses); an unset field falls back to the package env defaults
    (the same source `model_from_config` uses), so the gate matches the model the node actually
    built. Delegates the decision to the capability catalog.
    """
    from yian._settings import default_llm_model, default_llm_provider
    from yian.llm_clients.capabilities import supports_native_structured

    def _get(key: str) -> Any:
        if isinstance(llm_config, dict):
            return llm_config.get(key)
        return getattr(llm_config, key, None)

    provider = _get("provider") or default_llm_provider()
    model = _get("model") or default_llm_model(provider)
    connection = _get("_connection")
    if connection is not None and connection.get("method") == "subscription":
        return True
    return supports_native_structured(provider, model)


def _strip_code_fence(text: str) -> str:
    """Strip a single wrapping Markdown code fence from `text`, returning the inner text.

    Models often wrap JSON in a ```json … ``` (or bare ``` … ```) fence despite the
    "no code fences" instruction; a bare `json.loads` then fails and burns a retry. If
    `text` opens with a ``` fence line, drop that line and a matching trailing ```;
    text without a fence is returned unchanged (surrounding whitespace trimmed).
    """
    s = text.strip()
    if not s.startswith("```"):
        return s
    newline = s.find("\n")
    if newline == -1:
        return s  # a lone ``` with no body — leave it for json.loads to reject
    inner = s[newline + 1 :]
    if inner.rstrip().endswith("```"):
        inner = inner.rstrip()[:-3]
    return inner.strip()


def generate_structured(
    model: Any,
    messages: list,
    typ: Type,
    *,
    max_retries: int = 2,
    llm_config: dict | None = None,
    check_stopped: Callable[[], None] | None = None,
    evaluation_trace: EvaluationTrace | None = None,
    memory_context: MemoryContext | None = None,
    new_messages: list | None = None,
    live: LiveSinks | None = None,
) -> Any:
    """Generate a value conforming to `typ`, retrying on provider deviation up to a cap.

    Derives the pydantic schema from `typ` and picks a generation path by capability gate
    (`_supports_native_structured` on the node's effective `llm_config`):

    - **native** — bind the schema via `with_structured_output` and invoke.
    - **fallback** — the provider has no native structured output: render the JSON schema into
      the prompt, invoke for free text, then `json.loads` + `schema.model_validate` it.

    Either way a deviation (schema rejection, unparseable JSON) appends a corrective
    `HumanMessage` naming the error and retries, up to `max_retries` extra attempts
    (`max_retries + 1` total invocations). The last error is re-raised once the cap is spent.
    Session corrections expose only the type of a provider-side error; local parser
    diagnostics after a surfaced reply stay intact. Session cancellation never retries.

    Args:
        model (`Any`): Configured provider model, optionally supporting native binding.
        messages (`list`): Complete current provider context, including restored history.
        typ (`Type`): Exact declared output type used for schema and error ownership.
        max_retries (`int`): Extra correction attempts; defaults to two.
        llm_config (`dict | None`): Provider/model capability selection; defaults to
            environment settings when omitted. Existing config objects also work.
        check_stopped (`Callable | None`): Optional cooperative cancellation check.
        evaluation_trace (`EvaluationTrace | None`): Optional visible evidence sink.
        memory_context (`MemoryContext | None`): Recalled notes and optional transcript.
            An admitted transcript retains raw replies before parsing.
        new_messages (`list | None`): Not-yet-retained input delta, defaulting to all
            `messages`. Corrections are new deltas; prior invalid replies stay stored
            without being appended to the existing retry input shape.
        live (`LiveSinks | None`): Optional event sink; each attempt publishes
            model-call observers with usage. Parsing and retries are unchanged.

    Returns:
        `Any`: Parsed, unwrapped value matching the declared type.

    Raises:
        `_CapabilityFault`: Session encoding or publication fails, without retry.
        `_HostStop`: Cooperative cancellation occurs. Exhausted provider/parser errors
            propagate as their original exception, preserving schema ownership.
    """
    check = check_stopped or (lambda: None)
    memory_context = memory_context if memory_context is not None else MemoryContext()
    schema = type_to_schema(typ)
    if _supports_native_structured(llm_config or {}):
        return _generate_native(
            model,
            messages,
            typ,
            schema,
            max_retries,
            check,
            evaluation_trace,
            memory_context,
            new_messages,
            live,
        )
    return _generate_fallback(
        model,
        messages,
        typ,
        schema,
        max_retries,
        check,
        evaluation_trace,
        memory_context,
        new_messages,
        live,
    )


def _generate_native(
    model: Any,
    messages: list,
    typ: Type,
    schema: type[BaseModel],
    max_retries: int,
    check_stopped: Callable[[], None],
    evaluation_trace: EvaluationTrace | None,
    memory_context: MemoryContext,
    new_messages: list | None = None,
    live: LiveSinks | None = None,
) -> Any:
    """Native path: `with_structured_output(schema)` + capped self-correction retry."""
    msgs = list(messages)
    pending_messages = list(messages if new_messages is None else new_messages)
    last_err: Optional[Exception] = None
    for _ in range(max_retries + 1):
        check_stopped()
        reply_surfaced = False
        try:
            bound = (model.with_structured_output(schema) if memory_context.transcript is None
                     else model.with_structured_output(schema, include_raw=True))
            check_stopped()
            exchange_id = uuid4().hex
            delta, pending_messages = pending_messages, []
            response = invoke_model(
                memory_context.transcript,
                lambda: invoke_structured(live, bound, msgs, exchange_id=exchange_id),
                new_messages=delta, check_stopped=check_stopped, purpose="structured_native",
                raw_reply=_native_raw, exchange_id=exchange_id,
            )
            reply_surfaced = True
            obj = response if memory_context.transcript is None else _native_parsed(response)
            value = _unwrap(obj, typ)
            if memory_context.transcript is not None:
                check_stopped()
        except (_CapabilityFault, _HostStop):
            raise
        except Exception as err:  # provider deviated from the schema; correct and retry
            if memory_context.transcript is not None and isinstance(err, FutureCancelledError):
                raise
            check_stopped()
            last_err = err
            # Provider/binding errors can include credentials. Only local parsing
            # diagnostics from an already retained raw reply belong in corrections.
            detail = (f"provider error ({type(err).__name__})"
                      if memory_context.transcript is not None and not reply_surfaced else err)
            correction = HumanMessage(
                content=(
                    f"Your previous output was invalid: {detail}. "
                    "Respond with valid data matching the schema."
                )
            )
            msgs = msgs + [correction]
            pending_messages.append(correction)
        else:
            if evaluation_trace is not None:
                try:
                    interaction = ModelValueInteraction(value)
                except (TypeError, ValueError, RecursionError):
                    # Evidence is observational. A value the dispatcher will reject must
                    # retain its existing contract-failure classification.
                    pass
                else:
                    evaluation_trace.record_interaction(interaction)
            return value
    raise last_err  # type: ignore[misc]


def _native_raw(response):
    """Extract surfaced raw evidence; malformed carriers are capability faults."""
    if type(response) is not dict:
        raise TypeError("native structured response must be a raw/parsed/parsing_error dictionary")
    return response["raw"]


def _native_parsed(response):
    """Validate the native carrier after raw publication, then raise its exact error."""
    if set(response) != {"raw", "parsed", "parsing_error"}:
        raise _CapabilityFault(ValueError("native structured response fields are invalid"))
    error = response["parsing_error"]
    if error is not None:
        if not isinstance(error, BaseException):
            raise _CapabilityFault(TypeError("native structured parsing_error must be an exception or None"))
        raise error
    if not isinstance(response["parsed"], BaseModel):
        raise _CapabilityFault(TypeError("native structured parsed value must be a BaseModel"))
    return response["parsed"]


def _generate_fallback(
    model: Any,
    messages: list,
    typ: Type,
    schema: type[BaseModel],
    max_retries: int,
    check_stopped: Callable[[], None],
    evaluation_trace: EvaluationTrace | None,
    memory_context: MemoryContext,
    new_messages: list | None = None,
    live: LiveSinks | None = None,
) -> Any:
    """Prompt-injection path for a provider with no native structured output: ask for JSON
    matching the schema, parse + validate the free-text reply, capped self-correction retry.
    A wrapping ```json … ``` code fence is stripped before parsing (`_strip_code_fence`) so a
    fenced-but-valid reply doesn't burn a retry."""
    instruction = HumanMessage(
        content=(
            "Respond with ONLY a JSON object matching this JSON schema (no prose, no code "
            f"fences):\n{json.dumps(schema.model_json_schema())}"
        )
    )
    msgs = list(messages) + [instruction]
    pending_messages = list(messages if new_messages is None else new_messages) + [instruction]
    last_err: Optional[Exception] = None
    for _ in range(max_retries + 1):
        check_stopped()
        reply_surfaced = False
        try:
            exchange_id = uuid4().hex
            delta, pending_messages = pending_messages, []
            reply = invoke_model(
                memory_context.transcript,
                lambda: invoke_chat(
                    live, model, msgs, exchange_id=exchange_id, purpose="structured_fallback"
                ),
                new_messages=delta, check_stopped=check_stopped, purpose="structured_fallback",
                exchange_id=exchange_id,
            )
            reply_surfaced = True
            text = text_of(reply)
            check_stopped()
            record_model_text(evaluation_trace, text)
            obj = schema.model_validate(json.loads(_strip_code_fence(text)))
            value = _unwrap(obj, typ)
            if memory_context.transcript is not None:
                check_stopped()
        except (_CapabilityFault, _HostStop):
            raise
        except Exception as err:  # unparseable / non-conforming/provider-shape deviation
            if memory_context.transcript is not None and isinstance(err, FutureCancelledError):
                raise
            check_stopped()
            last_err = err
            detail = (f"provider error ({type(err).__name__})"
                      if memory_context.transcript is not None and not reply_surfaced else err)
            correction = HumanMessage(
                content=(
                    f"Your previous output was invalid: {detail}. Respond with ONLY a JSON "
                    "object matching the schema."
                )
            )
            msgs = msgs + [correction]
            pending_messages.append(correction)
        else:
            return value
    raise last_err  # type: ignore[misc]


def _is_contract_path(path: tuple[str | int, ...], typ: Type) -> bool:
    """Whether one normalized Pydantic location is valid for ``typ``."""
    if not path:
        return True
    if typ.variants is not None:
        return any(_is_contract_path(path, member) for member in typ.variants)
    head, tail = path[0], path[1:]
    if typ.kind == ValueKind.OBJECT and typ.fields:
        if type(head) is not str or head not in typ.fields:
            return False
        return _is_contract_path(tail, typ.fields[head])
    if typ.kind == ValueKind.LIST_ANY or typ.kind.is_list():
        if type(head) is not int or head < 0:
            return False
        if typ.element is None:
            return not tail
        return _is_contract_path(tail, typ.element)
    return False


def _validated_by_schema(exception: ValidationError, schema: type[BaseModel]) -> bool:
    """Whether traceback frames prove exact generated-schema validation."""
    validation_codes = (
        BaseModel.model_validate.__func__.__code__,
        BaseModel.model_validate_json.__func__.__code__,
        BaseModel.model_validate_strings.__func__.__code__,
    )
    traceback = exception.__traceback__
    while traceback is not None:
        frame = traceback.tb_frame
        if any(frame.f_code is code for code in validation_codes):
            if frame.f_locals.get("cls") is schema:
                return True
        elif frame.f_code is BaseModel.__init__.__code__:
            instance = frame.f_locals.get("self")
            if instance is not None and type(instance) is schema:
                return True
        traceback = traceback.tb_next
    return False


def _structured_validation_path(
    exception: Exception,
    expected_type: Type,
) -> tuple[str | int, ...] | None:
    """Project an owned structured-output error to a contract-relative path.

    A Pydantic error can also come from AGENT tool arguments or provider internals.
    Ownership requires the exhausted error's exact native/fallback frame, the same
    semantic type object, the frame's final-error identity, and an exact Pydantic
    validation frame for the generated schema. A native parser may preserve that
    validation error as the exhausted error's direct cause; deeper cause chains are not
    followed.
    Record locations already match contract paths; scalar/list wrapper fields
    and union member segments are removed, and at a union position the error of the
    member the reply was meant for is chosen (`_select_member_error`). ``()``
    identifies the output-contract root, while ``None`` means the exception is not
    generated structured output.
    """
    validation_error = (
        exception
        if isinstance(exception, ValidationError)
        else exception.__cause__
        if isinstance(exception.__cause__, ValidationError)
        else None
    )
    if validation_error is None:
        return None

    traceback = exception.__traceback__
    schema: type[BaseModel] | None = None
    while traceback is not None:
        frame = traceback.tb_frame
        if (
            (
                frame.f_code is _generate_native.__code__
                or frame.f_code is _generate_fallback.__code__
            )
            and frame.f_locals.get("typ") is expected_type
            and frame.f_locals.get("last_err") is exception
        ):
            candidate = frame.f_locals.get("schema")
            if (
                isinstance(candidate, type)
                and issubclass(candidate, BaseModel)
                and candidate.__name__ == validation_error.title
            ):
                schema = candidate
                break
        traceback = traceback.tb_next
    if schema is None or not _validated_by_schema(validation_error, schema):
        return None

    is_record = expected_type.kind == ValueKind.OBJECT and bool(expected_type.fields)
    is_list = expected_type.kind == ValueKind.LIST_ANY or expected_type.kind.is_list()
    wrapper = None if is_record else "items" if is_list else "value"
    details = validation_error.errors()
    if not details:
        return None
    errors: list[tuple[tuple[str | int, ...], Any]] = []
    for detail in details:
        location = detail.get("loc")
        if type(location) is not tuple or any(
            type(segment) not in (str, int) for segment in location
        ):
            return None
        if not location or wrapper is None:
            normalized = location
        elif location[0] == wrapper:
            normalized = location[1:]
        else:
            return None
        errors.append((normalized, detail.get("type")))
    path, error_type = _select_error(errors, expected_type)
    if error_type == "extra_forbidden" and path and type(path[-1]) is str:
        # A closed union member names the unknown field, as the engine's value check does.
        return path if _is_contract_path(path[:-1], expected_type) else None
    return path if _is_contract_path(path, expected_type) else None


def _select_error(
    errors: list[tuple[tuple[str | int, ...], Any]],
    typ: Type,
) -> tuple[tuple[str | int, ...], Any]:
    """Pick the error to report and drop the union member segments from its location.

    Outside unions this is the first error. At a union position see `_select_member_error`.

    Args:
        errors (`list[tuple[tuple[str | int, ...], Any]]`):
            `(location, error type)` pairs relative to `typ`, in pydantic's order. Must not
            be empty.
        typ (`Type`):
            The type at the errors' common position.

    Returns:
        `tuple[tuple[str | int, ...], Any]`:
            The contract-relative location and the error type of the chosen error.
    """
    if typ.variants is not None:
        return _select_member_error(errors, typ)
    location = errors[0][0]
    if not location:
        return errors[0]
    head = location[0]
    if typ.kind == ValueKind.OBJECT and typ.fields and type(head) is str and head in typ.fields:
        child = typ.fields[head]
    elif (typ.kind == ValueKind.LIST_ANY or typ.kind.is_list()) and type(head) is int:
        child = typ.element
    else:
        child = None
    if child is None:
        return errors[0]
    inner = [(loc[1:], kind) for loc, kind in errors if loc[:1] == (head,)]
    path, error_type = _select_error(inner, child)
    return (head,) + path, error_type


def _select_member_error(
    errors: list[tuple[tuple[str | int, ...], Any]],
    typ: Type,
) -> tuple[tuple[str | int, ...], Any]:
    """Pick the error at a union position, where each location starts with a member name.

    When every member rejects the tag field as `missing` or `literal_error`, the error is
    the tag field. Otherwise the members whose tag reported `literal_error` are dropped;
    a single remaining member reports its first error, and several (an untagged union no
    member matches) report the union's own position. This matches the engine's value check.

    Args:
        errors (`list[tuple[tuple[str | int, ...], Any]]`):
            `(location, error type)` pairs relative to the union, in pydantic's order. Must
            not be empty.
        typ (`Type`):
            The union.

    Returns:
        `tuple[tuple[str | int, ...], Any]`:
            The contract-relative location and the error type of the chosen error.
    """
    members = {_member_name(index): member for index, member in enumerate(typ.variants)}
    location, error_type = errors[0]
    if not location or type(location[0]) is not str or location[0] not in members:
        return errors[0]
    grouped: dict[str, list[tuple[tuple[str | int, ...], Any]]] = {}
    for loc, kind in errors:
        if loc and type(loc[0]) is str and loc[0] in members:
            grouped.setdefault(loc[0], []).append((loc[1:], kind))
    remaining = list(grouped)
    tag_fields = union_tag_fields(typ)
    if tag_fields:
        tag = tag_fields[0]
        tag_errors = {
            name: next((kind for loc, kind in member_errors if loc == (tag,)), None)
            for name, member_errors in grouped.items()
        }
        if len(grouped) == len(members) and all(
            kind in ("missing", "literal_error") for kind in tag_errors.values()
        ):
            return (tag,), tag_errors[location[0]]
        remaining = [name for name in grouped if tag_errors[name] != "literal_error"]
    if len(remaining) != 1:
        return (), error_type
    return _select_error(grouped[remaining[0]], members[remaining[0]])
