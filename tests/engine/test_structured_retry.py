"""`generate_structured` — self-correction retry up to a cap on provider deviation."""

import pytest

from agent_composer.nodes.agent.structured import generate_structured
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
