"""Structured AGENT output — derive a pydantic schema from a declared `output:` Type and
generate a value that conforms to it.

An AGENT declares `output:` as a typed `Type`. For a bare `str` the agent stays a text
producer — `type_to_schema` returns `None` and the mode takes the plain-text path. For any
constrained or richer type (a `Literal[...]` variant, record, scalar
`int`/`float`/`bool`, or list) `type_to_schema` builds a `pydantic.BaseModel` the
mode hands to `with_structured_output`, so the model emits a value that
success-record preparation can canonicalize against the effective target's declared
output type.

Scalars and lists can't be a top-level pydantic model field on their own, so they are wrapped
in a single-field model (`{"value": <scalar>}` / `{"items": [<element>]}`); `_unwrap` strips
the wrapper back to the bare value after generation. A record maps one model field per
declared field and passes through as the dumped dict.

Layer: nodes — imports `typesys` (Type/ValueKind) + `pydantic`; no engine/runtime imports.
"""

from __future__ import annotations

import json
from typing import Any, Callable, List, Literal, Optional

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, ValidationError, create_model

from agent_composer.memory.contracts import (
    EvaluationTrace,
    ModelTextInteraction,
    ModelValueInteraction,
)
from agent_composer.nodes.base import _HostStop
from agent_composer.typesys.values import Type, ValueKind

# ValueKind -> python type for a scalar slot. DATE/DATETIME persist as ISO strings.
_SCALAR_PY: dict[ValueKind, type] = {
    ValueKind.STRING: str,
    ValueKind.INTEGER: int,
    ValueKind.NUMBER: float,
    ValueKind.BOOLEAN: bool,
    ValueKind.DATE: str,
    ValueKind.DATETIME: str,
}


def _py_type(typ: Type) -> Any:
    """The python annotation for a value of `typ` as a pydantic field type.

    A `STRING` with `tags` is a `Literal[...]` variant; a nested `OBJECT` with `fields`
    becomes its own submodel; a `LIST_*` becomes `list[<element type>]`. Raises `ValueError`
    for a type with no structured mapping (`NONE`, `FILE`) so a new ValueKind is loud.
    """
    seg = typ.kind
    if seg == ValueKind.STRING and typ.tags:
        return Literal[tuple(sorted(typ.tags))]  # type: ignore[valid-type]
    if seg in _SCALAR_PY:
        return _SCALAR_PY[seg]
    if seg == ValueKind.OBJECT:
        return _record_model(typ) if typ.fields else dict
    if seg == ValueKind.LIST_ANY:
        return list
    if seg.is_list():
        elem = typ.element
        if elem is not None:
            return List[_py_type(elem)]  # type: ignore[misc]
        scalar = seg.element_kind
        return List[_SCALAR_PY[scalar]] if scalar in _SCALAR_PY else list  # type: ignore[misc]
    raise ValueError(f"type_to_schema: no structured mapping for segment type {seg!r}")


def _record_model(typ: Type) -> type[BaseModel]:
    """Build a pydantic model from an `OBJECT` Type: one field per `fields` entry, required
    iff named in `required` (others default to `None`); `nullable` widens a field to Optional."""
    required = typ.required or frozenset()
    spec: dict[str, Any] = {}
    for name, sub in (typ.fields or {}).items():
        ann = _py_type(sub)
        if name in required and not sub.nullable:
            spec[name] = (ann, ...)
        else:
            spec[name] = (Optional[ann], None)
    return create_model("Record", **spec)


def type_to_schema(typ: Type) -> Optional[type[BaseModel]]:
    """Derive a pydantic model for `typ`, or `None` when the agent should stay a text
    producer (a bare, untagged `str`).

    A record (`OBJECT` with `fields`) maps directly to a model. A scalar or a list can't be a
    standalone model, so it is wrapped in a single-field model (`value` / `items`) that
    `_unwrap` later strips.
    """
    seg = typ.kind
    if seg == ValueKind.STRING and not typ.tags:
        return None  # unconstrained text stays on the ordinary generation path
    if seg == ValueKind.OBJECT and typ.fields:
        return _record_model(typ)
    if seg.is_list() or seg == ValueKind.LIST_ANY:
        return create_model("ListWrapper", items=(_py_type(typ), ...))
    # any other scalar (int/float/bool/date/datetime) or freeform object -> value wrapper
    return create_model("ScalarWrapper", value=(_py_type(typ), ...))


def _unwrap(obj: BaseModel, typ: Type) -> Any:
    """Strip the single-field wrapper `type_to_schema` adds for a scalar/list, returning the
    bare value the write boundary expects. A record is returned as its dumped dict."""
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
    from agent_composer._settings import default_llm_model, default_llm_provider
    from agent_composer.llm_clients.capabilities import supports_native_structured

    def _get(key: str) -> Any:
        if isinstance(llm_config, dict):
            return llm_config.get(key)
        return getattr(llm_config, key, None)

    provider = _get("provider") or default_llm_provider()
    model = _get("model") or default_llm_model()
    return supports_native_structured(provider, model)


def _text_of(reply: Any) -> str:
    """The text content of a model reply (a langchain message or a bare string)."""
    content = getattr(reply, "content", reply)
    return content if isinstance(content, str) else str(content)


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
    """
    check = check_stopped or (lambda: None)
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
        )
    return _generate_fallback(
        model,
        messages,
        typ,
        schema,
        max_retries,
        check,
        evaluation_trace,
    )


def _generate_native(
    model: Any,
    messages: list,
    typ: Type,
    schema: type[BaseModel],
    max_retries: int,
    check_stopped: Callable[[], None],
    evaluation_trace: EvaluationTrace | None,
) -> Any:
    """Native path: `with_structured_output(schema)` + capped self-correction retry."""
    msgs = list(messages)
    last_err: Optional[Exception] = None
    for _ in range(max_retries + 1):
        check_stopped()
        try:
            bound = model.with_structured_output(schema)
            check_stopped()
            obj = bound.invoke(msgs)
            check_stopped()
            value = _unwrap(obj, typ)
        except _HostStop:
            raise
        except Exception as err:  # provider deviated from the schema; correct and retry
            check_stopped()
            last_err = err
            msgs = msgs + [
                HumanMessage(
                    content=(
                        f"Your previous output was invalid: {err}. "
                        "Respond with valid data matching the schema."
                    )
                )
            ]
        else:
            if evaluation_trace is not None:
                try:
                    interaction = ModelValueInteraction(value)
                except (TypeError, ValueError):
                    # Evidence is observational. A value the dispatcher will reject must
                    # retain its existing contract-failure classification.
                    pass
                else:
                    evaluation_trace.record_interaction(interaction)
            return value
    raise last_err  # type: ignore[misc]


def _generate_fallback(
    model: Any,
    messages: list,
    typ: Type,
    schema: type[BaseModel],
    max_retries: int,
    check_stopped: Callable[[], None],
    evaluation_trace: EvaluationTrace | None,
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
    last_err: Optional[Exception] = None
    for _ in range(max_retries + 1):
        check_stopped()
        try:
            text = _text_of(model.invoke(msgs))
            check_stopped()
            if evaluation_trace is not None:
                evaluation_trace.record_interaction(ModelTextInteraction(text))
            obj = schema.model_validate(json.loads(_strip_code_fence(text)))
            value = _unwrap(obj, typ)
        except _HostStop:
            raise
        except Exception as err:  # unparseable / non-conforming/provider-shape deviation
            check_stopped()
            last_err = err
            msgs = msgs + [
                HumanMessage(
                    content=(
                        f"Your previous output was invalid: {err}. Respond with ONLY a JSON "
                        "object matching the schema."
                    )
                )
            ]
        else:
            return value
    raise last_err  # type: ignore[misc]


def _is_contract_path(path: tuple[str | int, ...], typ: Type) -> bool:
    """Whether one normalized Pydantic location is valid for ``typ``."""
    if not path:
        return True
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
    followed. Record locations already match contract paths; scalar/list wrapper fields
    are removed. ``()`` identifies the output-contract root, while ``None`` means the
    exception is not generated structured output.
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
    location = details[0].get("loc")
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
    return normalized if _is_contract_path(normalized, expected_type) else None
