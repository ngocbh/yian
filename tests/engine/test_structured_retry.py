"""`generate_structured` — self-correction retry up to a cap on provider deviation."""

import json

import pytest
from pydantic import ValidationError, create_model

from agent_composer.nodes.agent.structured import (
    _structured_validation_path,
    generate_structured,
)
from agent_composer.nodes.base import _HostStop
from agent_composer.typesys.values import Type, ValueKind


def test_retry_on_invalid_then_succeeds():
    typ = Type.scalar(ValueKind.INTEGER)
    calls = {"n": 0}

    class _Flaky:
        def with_structured_output(self, schema):
            class _Bound:
                def invoke(self, msgs):
                    calls["n"] += 1
                    if calls["n"] == 1:
                        raise ValueError("bad json")  # provider deviation
                    return schema.model_validate({"value": 42})

            return _Bound()

    out = generate_structured(_Flaky(), [], typ, max_retries=2)
    assert out == 42 and calls["n"] == 2


def test_retry_on_invalid_provider_shape_then_succeeds():
    """A provider object that cannot be unwrapped remains a correctable deviation."""
    typ = Type.scalar(ValueKind.INTEGER)
    calls = {"n": 0}

    class _WrongShape:
        def model_dump(self):
            return {}

    class _Flaky:
        def with_structured_output(self, schema):
            class _Bound:
                def invoke(self, msgs):
                    calls["n"] += 1
                    if calls["n"] == 1:
                        return _WrongShape()
                    return schema.model_validate({"value": 42})

            return _Bound()

    assert generate_structured(_Flaky(), [], typ, max_retries=2) == 42
    assert calls["n"] == 2


def test_stop_during_native_binding_prevents_provider_invoke():
    """Cancellation while binding a schema cannot cross into a provider call."""
    typ = Type.scalar(ValueKind.INTEGER)
    stopped = False
    invocations = []

    class _StopsWhileBinding:
        def with_structured_output(self, schema):
            nonlocal stopped
            stopped = True

            class _Bound:
                def invoke(self, messages):
                    invocations.append(list(messages))
                    return schema.model_validate({"value": 42})

            return _Bound()

    def check_stopped():
        if stopped:
            raise _HostStop

    with pytest.raises(_HostStop):
        generate_structured(
            _StopsWhileBinding(),
            [],
            typ,
            llm_config={"provider": "openai", "model": "gpt-4o"},
            check_stopped=check_stopped,
        )

    assert invocations == []


def test_retry_exhausted_raises():
    typ = Type.scalar(ValueKind.INTEGER)

    class _Always:
        def with_structured_output(self, schema):
            class _Bound:
                def invoke(self, msgs):
                    raise ValueError("nope")

            return _Bound()

    with pytest.raises(Exception):
        generate_structured(_Always(), [], typ, max_retries=2)


def _validation_error(typ, payload, *, native=True):
    """Return one retry-exhausted generated-schema validation error."""

    class _Invalid:
        def with_structured_output(self, schema):
            class _Bound:
                def invoke(self, _messages):
                    return schema.model_validate(payload)

            return _Bound()

        def invoke(self, _messages):
            return json.dumps(payload)

    config = (
        {"provider": "openai", "model": "gpt-4o"}
        if native
        else {"provider": "vllm", "model": "no-structured-sentinel"}
    )
    with pytest.raises(ValidationError) as caught:
        generate_structured(
            _Invalid(),
            [],
            typ,
            max_retries=0,
            llm_config=config,
        )
    return caught.value


def _manual_validation_error(typ, *, title=None, location=()):
    """Return an exhausted native error with a controlled title and location."""

    class _Invalid:
        def with_structured_output(self, schema):
            class _Bound:
                def invoke(self, _messages):
                    raise ValidationError.from_exception_data(
                        title or schema.__name__,
                        [
                            {
                                "type": "value_error",
                                "loc": location,
                                "input": {},
                                "ctx": {"error": ValueError("invalid")},
                            }
                        ],
                    )

            return _Bound()

    with pytest.raises(ValidationError) as caught:
        generate_structured(
            _Invalid(),
            [],
            typ,
            max_retries=0,
            llm_config={"provider": "openai", "model": "gpt-4o"},
        )
    return caught.value


@pytest.mark.parametrize("native", [True, False])
def test_structured_validation_path_keeps_record_fields(native):
    """Native and fallback record errors expose the same contract path."""
    typ = Type(
        kind=ValueKind.OBJECT,
        fields={
            "profile": Type(
                kind=ValueKind.OBJECT,
                fields={"age": Type.scalar(ValueKind.INTEGER)},
                required=frozenset({"age"}),
            )
        },
        required=frozenset({"profile"}),
    )
    error = _validation_error(typ, {"profile": {"age": "bad"}}, native=native)

    assert _structured_validation_path(error, typ) == ("profile", "age")


def test_structured_validation_path_strips_only_generated_wrappers():
    """Scalar/list wrappers disappear while actual list indices remain."""
    scalar = Type.scalar(ValueKind.INTEGER)
    scalar_error = _validation_error(scalar, {"value": "bad"})
    item = Type(
        kind=ValueKind.OBJECT,
        fields={"age": Type.scalar(ValueKind.INTEGER)},
        required=frozenset({"age"}),
    )
    listed = Type(kind=ValueKind.LIST_OBJECT, element=item)
    list_error = _validation_error(
        listed,
        {"items": [{"age": 1}, {"age": "bad"}]},
    )

    assert _structured_validation_path(scalar_error, scalar) == ()
    assert _structured_validation_path(list_error, listed) == (1, "age")


def test_structured_validation_path_uses_first_pydantic_error():
    """Multiple rejected fields follow Pydantic's stable first-error order."""
    typ = Type(
        kind=ValueKind.OBJECT,
        fields={
            "first": Type.scalar(ValueKind.INTEGER),
            "second": Type.scalar(ValueKind.INTEGER),
        },
        required=frozenset({"first", "second"}),
    )
    error = _validation_error(typ, {"first": "bad", "second": "also bad"})

    assert _structured_validation_path(error, typ) == ("first",)


def test_structured_validation_path_preserves_owned_root_location():
    """A generated model-level failure selects the contract root."""
    typ = Type.scalar(ValueKind.INTEGER)
    error = _manual_validation_error(typ)

    assert _structured_validation_path(error, typ) == ()


def test_structured_validation_path_rejects_foreign_schema_in_owned_frame():
    """Frame provenance alone cannot turn a provider validation model into output."""
    typ = Type.scalar(ValueKind.INTEGER)
    error = _manual_validation_error(typ, title="ProviderConfig", location=("value",))

    assert _structured_validation_path(error, typ) is None


def test_structured_validation_path_rejects_unowned_pydantic_errors():
    """A wrapper-shaped validation error without an owned generator frame is unrelated."""
    typ = Type.scalar(ValueKind.INTEGER)
    generated = _validation_error(typ, {"value": "bad"})
    equal_but_distinct = Type.scalar(ValueKind.INTEGER)

    with pytest.raises(ValidationError) as caught:
        create_model("ScalarWrapper", value=(int, ...)).model_validate({"value": "bad"})

    assert _structured_validation_path(caught.value, typ) is None
    assert _structured_validation_path(generated, equal_but_distinct) is None
