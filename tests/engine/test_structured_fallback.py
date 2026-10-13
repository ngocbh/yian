"""Structured output prompt-injection fallback when a provider lacks native support.

Driven through the REAL capability source (`supports_native_structured`) on the catalog
sentinel pair, proving the fallback path is reachable on a concrete provider/model — not via
a monkeypatch of the gate.
"""

from yian.nodes.agent.structured import generate_structured
from yian.typesys.values import Type, ValueKind


def test_no_native_support_uses_prompt_injection():
    typ = Type.scalar(ValueKind.INTEGER)

    class _NoNative:
        def invoke(self, msgs):
            text = msgs[-1].content
            assert "schema" in text.lower() or "json" in text.lower()

            class R:
                content = '{"value": 5}'

            return R()

        def with_structured_output(self, schema):
            raise AssertionError("no-native model must not use with_structured_output")

    cfg = {"provider": "vllm", "model": "no-structured-sentinel"}
    assert generate_structured(_NoNative(), [], typ, llm_config=cfg) == 5


def test_fallback_tolerates_json_code_fence():
    # Models often wrap JSON in a ```json … ``` fence despite the "no code fences"
    # instruction; the fallback strips it rather than burning a retry.
    typ = Type.scalar(ValueKind.INTEGER)
    calls = {"n": 0}

    class _Fenced:
        def invoke(self, msgs):
            calls["n"] += 1

            class R:
                content = '```json\n{"value": 5}\n```'

            return R()

    cfg = {"provider": "vllm", "model": "no-structured-sentinel"}
    assert generate_structured(_Fenced(), [], typ, llm_config=cfg) == 5
    assert calls["n"] == 1  # parsed on the first try, no corrective retry


def test_fallback_tolerates_bare_code_fence():
    # A bare ``` … ``` fence (no language tag) is stripped just the same.
    typ = Type.scalar(ValueKind.INTEGER)
    calls = {"n": 0}

    class _Fenced:
        def invoke(self, msgs):
            calls["n"] += 1

            class R:
                content = '```\n{"value": 9}\n```'

            return R()

    cfg = {"provider": "vllm", "model": "no-structured-sentinel"}
    assert generate_structured(_Fenced(), [], typ, llm_config=cfg) == 9
    assert calls["n"] == 1


def test_fallback_accepts_null_for_a_nullable_root():
    typ = Type(kind=ValueKind.INTEGER, nullable=True)
    calls = {"n": 0}

    class _Null:
        def invoke(self, msgs):
            calls["n"] += 1
            assert '"null"' in msgs[-1].content  # the schema shown to the model allows null

            class R:
                content = '{"value": null}'

            return R()

    cfg = {"provider": "vllm", "model": "no-structured-sentinel"}
    assert generate_structured(_Null(), [], typ, llm_config=cfg) is None
    assert calls["n"] == 1


def test_fallback_retries_on_unparseable_then_succeeds():
    typ = Type.scalar(ValueKind.INTEGER)
    calls = {"n": 0}

    class _Flaky:
        def invoke(self, msgs):
            calls["n"] += 1

            class R:
                content = "not json" if calls["n"] == 1 else '{"value": 7}'

            return R()

    cfg = {"provider": "vllm", "model": "no-structured-sentinel"}
    assert generate_structured(_Flaky(), [], typ, max_retries=2, llm_config=cfg) == 7
    assert calls["n"] == 2
