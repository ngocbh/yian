"""`generate_structured` — self-correction retry up to a cap on provider deviation."""

import json

import pytest
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import ValidationError, create_model

from yian.nodes.agent.structured import (
    _is_contract_path,
    _structured_validation_path,
    generate_structured,
)
from yian.nodes.base import _HostStop
from yian.typesys.values import Type, ValueKind


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


def test_literal_retries_an_invalid_tag_then_returns_bare_tag():
    typ = Type(
        kind=ValueKind.STRING,
        tags=frozenset({"positive", "neutral", "negative"}),
    )
    calls = {"n": 0}

    class _Flaky:
        def with_structured_output(self, schema):
            class _Bound:
                def invoke(self, msgs):
                    calls["n"] += 1
                    value = "The sentiment is neutral." if calls["n"] == 1 else "neutral"
                    return schema.model_validate({"value": value})

            return _Bound()

    assert generate_structured(
        _Flaky(),
        [],
        typ,
        max_retries=2,
        llm_config={"provider": "openai", "model": "gpt-4o"},
    ) == "neutral"
    assert calls["n"] == 2


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


def _validation_error(typ, payload, *, native=True, constructor=False):
    """Return one retry-exhausted generated-schema validation error."""

    class _Invalid:
        def with_structured_output(self, schema):
            class _Bound:
                def invoke(self, _messages):
                    return (
                        schema(**payload)
                        if constructor
                        else schema.model_validate(payload)
                    )

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


def _foreign_same_title_error(typ):
    """Return a provider-model error whose title collides with the output schema."""

    class _Invalid:
        def with_structured_output(self, schema):
            foreign_schema = create_model(schema.__name__, value=(int, ...))

            class _Bound:
                def invoke(self, _messages):
                    return foreign_schema.model_validate({"value": "bad"})

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


@pytest.mark.parametrize("native", [True, False])
def test_structured_validation_path_strips_nullable_record_wrapper(native):
    """A nullable record root is wrapped like a scalar, so its wrapper is stripped too."""
    typ = Type(
        kind=ValueKind.OBJECT,
        fields={"name": Type.scalar(ValueKind.STRING), "age": Type.scalar(ValueKind.INTEGER)},
        required=frozenset({"name", "age"}),
        nullable=True,
    )
    error = _validation_error(typ, {"value": {"name": "a", "age": "bad"}}, native=native)

    assert _structured_validation_path(error, typ) == ("age",)


def test_structured_validation_path_accepts_generated_constructor_frame():
    """LangChain's Pydantic constructor route proves the same exact schema."""
    typ = Type.scalar(ValueKind.INTEGER)
    error = _validation_error(
        typ,
        {"value": "bad"},
        constructor=True,
    )

    assert _structured_validation_path(error, typ) == ()


def test_structured_validation_path_accepts_direct_parser_wrapped_error():
    """Native parsers may wrap the generated-schema error exactly once."""
    typ = Type.scalar(ValueKind.INTEGER)

    class _Invalid:
        def with_structured_output(self, schema):
            parser = PydanticOutputParser(pydantic_object=schema)

            class _Bound:
                def invoke(self, _messages):
                    return parser.parse('{"value": "bad"}')

            return _Bound()

    with pytest.raises(OutputParserException) as caught:
        generate_structured(
            _Invalid(),
            [],
            typ,
            max_retries=0,
            llm_config={"provider": "google", "model": "gemini-2.0-flash"},
        )

    cause = caught.value.__cause__
    assert isinstance(cause, ValidationError)
    assert _structured_validation_path(caught.value, typ) == ()


def test_structured_validation_path_rejects_wrapped_same_title_foreign_schema():
    """A direct cause still must come from the exact generated schema."""
    typ = Type.scalar(ValueKind.INTEGER)

    class _Invalid:
        def with_structured_output(self, schema):
            foreign_schema = create_model(schema.__name__, value=(int, ...))

            class _Bound:
                def invoke(self, _messages):
                    try:
                        return foreign_schema.model_validate({"value": "bad"})
                    except ValidationError as error:
                        raise RuntimeError("parser rejected output") from error

            return _Bound()

    with pytest.raises(RuntimeError) as caught:
        generate_structured(
            _Invalid(),
            [],
            typ,
            max_retries=0,
            llm_config={"provider": "openai", "model": "gpt-4o"},
        )

    assert isinstance(caught.value.__cause__, ValidationError)
    assert _structured_validation_path(caught.value, typ) is None


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
    error = _validation_error(typ, "not-an-object")

    assert _structured_validation_path(error, typ) == ()


def test_structured_validation_path_rejects_same_title_foreign_schema():
    """A provider model named like the wrapper lacks exact schema identity."""
    typ = Type.scalar(ValueKind.INTEGER)
    error = _foreign_same_title_error(typ)

    assert _structured_validation_path(error, typ) is None


def test_contract_path_validation_rejects_foreign_location_shapes():
    """Unknown fields, scalar tails, and non-index list paths are not contract paths."""
    record = Type(
        kind=ValueKind.OBJECT,
        fields={"age": Type.scalar(ValueKind.INTEGER)},
    )
    listed = Type(kind=ValueKind.LIST_OBJECT, element=record)

    assert not _is_contract_path(("unknown",), record)
    assert not _is_contract_path(("foreign",), Type.scalar(ValueKind.INTEGER))
    assert not _is_contract_path(("age",), listed)
    assert not _is_contract_path((-1, "age"), listed)
    assert not _is_contract_path((0, "unknown"), listed)


def test_structured_validation_path_rejects_unowned_pydantic_errors():
    """A wrapper-shaped validation error without an owned generator frame is unrelated."""
    typ = Type.scalar(ValueKind.INTEGER)
    generated = _validation_error(typ, {"value": "bad"})
    equal_but_distinct = Type.scalar(ValueKind.INTEGER)

    with pytest.raises(ValidationError) as caught:
        create_model("ScalarWrapper", value=(int, ...)).model_validate({"value": "bad"})

    assert _structured_validation_path(caught.value, typ) is None
    assert _structured_validation_path(generated, equal_but_distinct) is None
