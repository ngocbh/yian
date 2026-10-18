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

Layer: nodes — imports memory contracts, the message adapter, `llm_clients`, `typesys`
and `pydantic`; no engine/runtime imports.
"""

from __future__ import annotations

from concurrent.futures import CancelledError as FutureCancelledError
import json
from typing import Annotated, Any, Callable, List, Literal, Optional, Union
from uuid import uuid4

from langchain_core.messages import HumanMessage
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    ValidationError,
    create_model,
)

from yian.defaults import current
from yian.llm_clients.base_client import missing_tool_call
from yian.memory.contracts import (
    MemoryContext,
    EvaluationTrace,
    ModelValueInteraction,
)
from yian.nodes.agent.memory import invoke_model
from yian.nodes.agent.live import LiveSinks, invoke_chat, invoke_structured, is_provider_auth
from yian.nodes.agent.text import record_model_text, text_of
from yian.nodes.base import _CapabilityFault, _HostStop
from yian.typesys.values import (
    Type,
    TypeCheckError,
    ValueKind,
    parse_json_int,
    reject_deep_nesting,
    reject_long_ints,
    reject_nonfinite,
    rewrite_digit_limit_text,
    union_tag_fields,
)

# ValueKind -> python type for a scalar slot. DATE/DATETIME persist as ISO strings.
_SCALAR_PY: dict[ValueKind, type] = {
    ValueKind.STRING: str,
    ValueKind.INTEGER: int,
    ValueKind.NUMBER: float,
    ValueKind.BOOLEAN: bool,
    ValueKind.DATE: str,
    ValueKind.DATETIME: str,
}

# Any non-null JSON value. Strict members keep pydantic's smart-mode union from turning
# `True` into 1, "3" into 3, or 1 into 1.0; a reply is stored as given.
_ANY_PY = Union[
    StrictBool,
    StrictInt,
    Annotated[StrictFloat, Field(allow_inf_nan=False)],
    StrictStr,
    dict[str, Any],
    list[Any],
]


def _py_type(typ: Type, *, closed: bool = False, position: str = "") -> Any:
    """The python annotation for a value of `typ` as a pydantic field type.

    A `STRING` with `tags` is a `Literal[...]` variant; a nested `OBJECT` with `fields`
    becomes its own submodel; a union becomes a plain `Union[...]` of closed member models
    in member order; a `LIST_*` becomes `list[<element type>]`; `Any` becomes a union of
    the JSON types (`_ANY_PY`). A nullable `typ` is `Optional[...]` wherever it appears.
    Nested models are named after their position (see `_member_names`), so every `$defs`
    key stays readable.

    Args:
        typ (`Type`):
            The declared type of the slot.
        closed (`bool`, *optional*, defaults to `False`):
            Whether records reject unknown fields. Union members and everything inside
            them are always closed, so a reply can't match a member by ignoring fields.
        position (`str`, *optional*, defaults to `""`):
            The slot's position name, built by `_nested`. `""` is the schema root.

    Returns:
        `Any`:
            The pydantic annotation.

    Raises:
        `ValueError`:
            `typ` has no structured mapping (`NONE`, `FILE`), so a new ValueKind is loud.
    """
    annotation = _non_null_py_type(typ, closed=closed, position=position)
    return Optional[annotation] if typ.nullable else annotation


def _non_null_py_type(typ: Type, *, closed: bool, position: str) -> Any:
    """`_py_type` without the nullable widening."""
    seg = typ.kind
    if seg == ValueKind.ANY:
        return _ANY_PY
    if seg == ValueKind.STRING and typ.tags:
        return Literal[tuple(sorted(typ.tags))]  # type: ignore[valid-type]
    if seg in _SCALAR_PY:
        if typ.bounds is not None:
            typ.validate_shape()
        return _scalar_annotation(seg, typ.bounds)
    if seg == ValueKind.OBJECT and typ.variants is not None:
        members = tuple(
            _record_model(member, position=name, closed=True)
            for name, member in zip(_member_names(typ, position), typ.variants)
        )
        return Union[members]  # type: ignore[valid-type]
    if seg == ValueKind.OBJECT:
        if not typ.fields:
            return dict
        return _record_model(typ, position=position, closed=closed)
    if seg == ValueKind.LIST_ANY and typ.element is None:
        return list
    if seg.is_list():
        elem = typ.element
        if elem is not None:
            item = _py_type(elem, closed=closed, position=_nested(position, "item"))
            return List[item]  # type: ignore[misc]
        scalar = seg.element_kind
        if scalar not in _SCALAR_PY:
            return list
        return List[_scalar_annotation(scalar, None)]  # type: ignore[misc]
    raise ValueError(f"type_to_schema: no structured mapping for segment type {seg!r}")


def _scalar_annotation(
    kind: ValueKind,
    bounds: tuple[int | float | None, int | float | None] | None,
) -> Any:
    """Return the pydantic annotation of a scalar slot of `kind`, within inclusive `bounds`.

    An open (`None`) end has no limit, so the schema omits that end's `minimum` or
    `maximum`.
    A `float` slot rejects NaN and infinity, which are not yian values; the JSON
    schema is unchanged.
    """
    py_type = _SCALAR_PY[kind]
    if kind == ValueKind.NUMBER:
        if bounds is None:
            return Annotated[float, Field(allow_inf_nan=False)]
        return Annotated[float, Field(allow_inf_nan=False, ge=bounds[0], le=bounds[1])]
    if bounds is None:
        return py_type
    return Annotated[py_type, Field(ge=bounds[0], le=bounds[1])]


def _nested(position: str, segment: str) -> str:
    """Return the position name of `segment` (a field, `item`, or a member) inside `position`."""
    return f"{position}_{segment}" if position else segment


def _member_names(typ: Type, position: str) -> tuple[str, ...]:
    """Name the member models of the union at `position`, in member order.

    A tagged union whose tag values are all identifiers names each member after its tag
    (`buy`, `sell`); otherwise the members are `Member1`, `Member2`, .... Error locations
    carry these names, so `_select_member_error` derives them the same way.

    Args:
        typ (`Type`):
            The union.
        position (`str`):
            The union's position name. `""` is the schema root.

    Returns:
        `tuple[str, ...]`:
            One model name per member.
    """
    tag_fields = union_tag_fields(typ)
    tags = [
        next(iter(member.fields[tag_fields[0]].tags)) for member in typ.variants
    ] if tag_fields else []
    if not tags or not all(tag.isidentifier() for tag in tags):
        tags = [f"Member{index + 1}" for index in range(len(typ.variants))]
    return tuple(_nested(position, tag) for tag in tags)


def _record_model(typ: Type, *, position: str = "", closed: bool = False) -> type[BaseModel]:
    """Build a pydantic model from an `OBJECT` Type.

    Each `fields` entry is a model field, required iff named in `required` (others
    default to `None`); `nullable` widens a field to Optional.

    Args:
        typ (`Type`):
            A record type.
        position (`str`, *optional*, defaults to `""`):
            The record's position name, which is the model name and schema title and
            prefixes the models inside it. `""` is the schema root, named `Record`.
        closed (`bool`, *optional*, defaults to `False`):
            Whether the model and the records inside it reject unknown fields.

    Returns:
        `type[BaseModel]`:
            The generated model.
    """
    required = typ.required or frozenset()
    spec: dict[str, Any] = {}
    for field_name, sub in (typ.fields or {}).items():
        ann = _py_type(sub, closed=closed, position=_nested(position, field_name))
        if field_name in required and not sub.nullable:
            spec[field_name] = (ann, ...)
        else:
            spec[field_name] = (Optional[ann], None)
    config = ConfigDict(extra="forbid") if closed else None
    return create_model(position or "Record", __config__=config, **spec)


def _is_bare_record(typ: Type) -> bool:
    """Whether `typ` is a non-nullable record, which maps to an unwrapped model."""
    return typ.kind == ValueKind.OBJECT and bool(typ.fields) and not typ.nullable


def type_to_schema(typ: Type) -> Optional[type[BaseModel]]:
    """Derive a pydantic model for `typ`, or `None` when the agent should stay a text
    producer (a bare, untagged `str`).

    A non-nullable record (`OBJECT` with `fields`) maps directly to a model. A scalar, a
    union, a list, or a nullable record can't be a standalone model, so it is wrapped in a
    single-field model (`value` / `items`) that `_unwrap` later strips.
    """
    seg = typ.kind
    if seg == ValueKind.STRING and not typ.tags:
        return None  # unconstrained text stays on the ordinary generation path
    if _is_bare_record(typ):
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
    if _is_bare_record(typ):
        return data
    if seg.is_list() or seg == ValueKind.LIST_ANY:
        return data["items"]
    return data["value"]


def _reply_value(data: Any, typ: Type) -> Any:
    """Return the value a parsed prompted reply carries for `typ`, before validation.

    For a wrapped type this is the wrapper field's content when the reply has it;
    otherwise the whole reply.
    """
    if _is_bare_record(typ) or type(data) is not dict:
        return data
    key = "items" if typ.kind.is_list() or typ.kind == ValueKind.LIST_ANY else "value"
    return data.get(key, data)


def _supports_native_structured(llm_config: Any) -> bool:
    """Whether the node's effective `(provider, model)` can use native structured output.

    Reads `provider`/`model` off the effective config (a plain dict on the run path, or an
    `LLMConfig` in some unit harnesses); when either is unset, both resolve as
    `model_from_config` resolves them, so the gate matches the model the node actually
    built. Delegates the decision to the capability catalog.
    """
    from yian.llm_clients.capabilities import supports_native_structured
    from yian.llm_clients.connection import resolve_model_config

    def _get(key: str) -> Any:
        if isinstance(llm_config, dict):
            return llm_config.get(key)
        return getattr(llm_config, key, None)

    provider, model = _get("provider"), _get("model")
    if not provider or not model:
        resolved = resolve_model_config(llm_config or {})
        provider, model = resolved["provider"], resolved["model"]
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
    max_retries: int | None = None,
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

    Either way a deviation (schema rejection, unparseable JSON, a NaN, infinity, or int
    over `YIAN_MAX_INT_DIGITS` digits in the reply, a native reply with no call to the
    schema's tool) appends a corrective `HumanMessage` naming the error, with Python's
    digit-limit text given as yian's, and retries, up to `max_retries` extra attempts
    (`max_retries + 1` total invocations). The last error is re-raised once the cap is
    spent. Session corrections expose only the type of a provider-side error; diagnostics
    raised once the reply surfaced stay intact. Session cancellation never retries.

    Args:
        model (`Any`): Configured provider model, optionally supporting native binding.
        messages (`list`): Complete current provider context, including restored history.
        typ (`Type`): Exact declared output type used for schema and error ownership.
        max_retries (`int | None`): Extra correction attempts; `None` uses the
            `AGENT_RETRIES` default.
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
    if max_retries is None:
        max_retries = current().agent_retries
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
    reply_surfaced = False

    def check_reply(response: Any) -> None:
        nonlocal reply_surfaced
        reply_surfaced = True
        if memory_context.transcript is not None:
            _reject_deep_carrier(response, schema, typ)
            _reject_nonfinite_carrier(response, schema.__name__)

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
                raw_reply=_native_raw, exchange_id=exchange_id, check_reply=check_reply,
            )
            if memory_context.transcript is None:
                # The tool parser returns `None` for a reply with no call to the schema's tool.
                if response is None:
                    raise missing_tool_call(schema.__name__)
                obj = response
            else:
                obj = _native_parsed(response, schema.__name__)
            value = _unwrap(obj, typ)
            reject_deep_nesting(value, "structured output")
            reject_nonfinite(value, "structured output")
            if memory_context.transcript is not None:
                check_stopped()
        except (_CapabilityFault, _HostStop):
            raise
        except Exception as err:  # provider deviated from the schema; correct and retry
            if memory_context.transcript is not None and isinstance(err, FutureCancelledError):
                raise
            if is_provider_auth(err):
                raise  # refused credentials fail every retry the same way
            check_stopped()
            last_err = err
            # Provider/binding errors can include credentials. An error raised once the
            # reply surfaced is a local diagnostic of that reply (yian's check or the
            # client's parser), so it is sent in full.
            detail = (f"provider error ({type(err).__name__})"
                      if memory_context.transcript is not None and not reply_surfaced
                      else rewrite_digit_limit_text(str(err)))
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


def _native_parsed(response: dict, tool_name: str) -> BaseModel:
    """Validate the native carrier after raw publication, then raise its exact error.

    Args:
        response (`dict`):
            The `raw`/`parsed`/`parsing_error` carrier.
        tool_name (`str`):
            The name of the schema's tool, for the error when the reply did not call it.

    Returns:
        `BaseModel`:
            The parsed value.

    Raises:
        `Exception`:
            The carrier's parsing error, or `missing_tool_call(tool_name)` when the reply
            has no parsed value and no parsing error.
        `_CapabilityFault`:
            Raised when the carrier is malformed.
    """
    error = _native_carrier_error(response, tool_name)
    if error is not None:
        raise error
    return response["parsed"]


def _native_carrier_error(response: dict, tool_name: str) -> BaseException | None:
    """Return a native carrier's parsing error, raising `_CapabilityFault` if it is malformed.

    A carrier with neither a parsed value nor a parsing error holds a reply that did not
    call the schema's tool, so its error is `missing_tool_call(tool_name)`.

    Args:
        response (`dict`):
            The `raw`/`parsed`/`parsing_error` carrier.
        tool_name (`str`):
            The name of the schema's tool.

    Returns:
        `BaseException | None`:
            The carrier's error, or `None` when it holds a parsed value.

    Raises:
        `_CapabilityFault`:
            Raised when the carrier's fields, parsing error, or parsed value are malformed.
    """
    if set(response) != {"raw", "parsed", "parsing_error"}:
        raise _CapabilityFault(ValueError("native structured response fields are invalid"))
    error = response["parsing_error"]
    if error is not None:
        if not isinstance(error, BaseException):
            raise _CapabilityFault(TypeError("native structured parsing_error must be an exception or None"))
        return error
    if response["parsed"] is None:
        return missing_tool_call(tool_name)
    if not isinstance(response["parsed"], BaseModel):
        raise _CapabilityFault(TypeError("native structured parsed value must be a BaseModel"))
    return None


def _reject_deep_carrier(response: Any, schema: type[BaseModel], typ: Type) -> None:
    """Reject a native carrier whose reply nests past the value limit, before it is stored.

    A parsed value past the limit raises `TypeCheckError`. A carrier holding a parsing
    error, or no parsed value, whose raw reply is past the limit raises the carrier's
    error, so the retry and the exhausted error's origin match a run without memory. A
    malformed carrier is left alone. The raw reply carries the value, so a reply past the
    limit may be too deep for memory, and the non-finite walk of the raw reply recurses.
    """
    if type(response) is not dict or set(response) != {"raw", "parsed", "parsing_error"}:
        return
    if response["parsing_error"] is None and response["parsed"] is not None:
        if isinstance(response["parsed"], schema):
            reject_deep_nesting(_unwrap(response["parsed"], typ), "structured output")
        return
    # The raw message wraps the value a few levels deeper, so this can also catch a value
    # just under the limit. Only a reply that already missed the schema is checked here;
    # it is retried either way, and the check only keeps memory from storing it.
    try:
        reject_deep_nesting(response.get("raw"), "model reply")
    except TypeCheckError:
        pass
    else:
        return
    # Raised outside the handler so the provider's error gains no `__context__`.
    error = _native_carrier_error(response, schema.__name__)
    if error is not None:
        raise error


def _reject_nonfinite_carrier(response: Any, tool_name: str) -> None:
    """Raise a native carrier's parsing error when its raw reply holds a NaN or infinity.

    A carrier whose raw reply is finite is left alone, so it is stored and parsed as
    before. A malformed carrier raises `_CapabilityFault`. The parsing error is the
    schema's own, so the retry and the exhausted error's origin match a run without
    memory. A carrier with no parsed value raises `missing_tool_call(tool_name)`.

    Args:
        response (`Any`): The native carrier the model call returned.
        tool_name (`str`): The schema's tool name, for a carrier with no parsed value.
    """
    if type(response) is not dict:
        return
    try:
        reject_nonfinite(response.get("raw"), "model reply")
    except TypeCheckError:
        pass
    else:
        return
    # Raised outside the handler so the provider's error gains no `__context__`.
    error = _native_carrier_error(response, tool_name)
    if error is not None:
        raise error


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
    reply_surfaced = False

    def check_reply(_reply: Any) -> None:
        nonlocal reply_surfaced
        reply_surfaced = True

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
                exchange_id=exchange_id, check_reply=check_reply,
            )
            text = text_of(reply)
            check_stopped()
            record_model_text(evaluation_trace, text)
            data = json.loads(_strip_code_fence(text), parse_int=parse_json_int)
            reply_value = _reply_value(data, typ)
            reject_deep_nesting(reply_value, "structured output")
            # A long int is a marker here, which the schema would reject without saying why.
            reject_long_ints(reply_value, "structured output")
            obj = schema.model_validate(data)
            value = _unwrap(obj, typ)
            reject_deep_nesting(value, "structured output")
            reject_nonfinite(value, "structured output")
            if memory_context.transcript is not None:
                check_stopped()
        except (_CapabilityFault, _HostStop):
            raise
        except Exception as err:  # unparseable / non-conforming/provider-shape deviation
            if memory_context.transcript is not None and isinstance(err, FutureCancelledError):
                raise
            if is_provider_auth(err):
                raise  # refused credentials fail every retry the same way
            check_stopped()
            last_err = err
            # As on the native path, an error raised once the reply surfaced is sent in full.
            detail = (f"provider error ({type(err).__name__})"
                      if memory_context.transcript is not None and not reply_surfaced
                      else rewrite_digit_limit_text(str(err)))
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
    Bare record locations already match contract paths; wrapper fields
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

    is_record = _is_bare_record(expected_type)
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
    position: str = "",
) -> tuple[tuple[str | int, ...], Any]:
    """Pick the error to report and drop the union member segments from its location.

    Outside unions this is the first error. At a union position see `_select_member_error`.
    At an `Any` position the error is that position itself: pydantic names the JSON type
    each member of the value union expects, which is not a contract path.

    Args:
        errors (`list[tuple[tuple[str | int, ...], Any]]`):
            `(location, error type)` pairs relative to `typ`, in pydantic's order. Must not
            be empty.
        typ (`Type`):
            The type at the errors' common position.
        position (`str`, *optional*, defaults to `""`):
            That position's name, as `_py_type` built it. `""` is the schema root.

    Returns:
        `tuple[tuple[str | int, ...], Any]`:
            The contract-relative location and the error type of the chosen error.
    """
    if typ.kind == ValueKind.ANY:
        return (), errors[0][1]
    if typ.variants is not None:
        return _select_member_error(errors, typ, position)
    location = errors[0][0]
    if not location:
        return errors[0]
    head = location[0]
    if typ.kind == ValueKind.OBJECT and typ.fields and type(head) is str and head in typ.fields:
        child, child_position = typ.fields[head], _nested(position, head)
    elif (typ.kind == ValueKind.LIST_ANY or typ.kind.is_list()) and type(head) is int:
        child, child_position = typ.element, _nested(position, "item")
    else:
        child = None
    if child is None:
        return errors[0]
    inner = [(loc[1:], kind) for loc, kind in errors if loc[:1] == (head,)]
    path, error_type = _select_error(inner, child, child_position)
    return (head,) + path, error_type


def _select_member_error(
    errors: list[tuple[tuple[str | int, ...], Any]],
    typ: Type,
    position: str,
) -> tuple[tuple[str | int, ...], Any]:
    """Pick the error at a union position, where each location starts with a member name.

    In a tagged union a member with any error at its tag field is ruled out; if none is
    left, the error is the tag field. A single remaining member reports its first error,
    and several (an untagged union no member matches) report the union's own position.
    Member selection matches the engine's value check; the first error inside a member
    can differ from the engine's, since pydantic reports extra fields last.

    Args:
        errors (`list[tuple[tuple[str | int, ...], Any]]`):
            `(location, error type)` pairs relative to the union, in pydantic's order. Must
            not be empty.
        typ (`Type`):
            The union.
        position (`str`):
            The union's position name, as `_py_type` built it.

    Returns:
        `tuple[tuple[str | int, ...], Any]`:
            The contract-relative location and the error type of the chosen error.
    """
    members = dict(zip(_member_names(typ, position), typ.variants))
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
            name: kind
            for name, member_errors in grouped.items()
            for loc, kind in member_errors
            if loc == (tag,)
        }
        remaining = [name for name in grouped if name not in tag_errors]
        if not remaining:
            return (tag,), tag_errors[location[0]]
    if len(remaining) != 1:
        return (), error_type
    return _select_error(grouped[remaining[0]], members[remaining[0]], remaining[0])
