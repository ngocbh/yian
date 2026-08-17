"""`plain` mode — native structured output for constrained output types."""

import agent_composer.llm_clients as llm_clients
from agent_composer.nodes.agent.modes.common import AgentRunContext
from agent_composer.nodes.agent.modes.plain import plain
from agent_composer.typesys.values import Type, ValueKind


class _StructuredModel:
    def __init__(self):
        self.structured_called_with = None

    def with_structured_output(self, schema):
        self.structured_called_with = schema

        class _Bound:
            def invoke(self, msgs):
                return schema.model_validate({"name": "Ada", "score": 9})

        return _Bound()

    def invoke(self, msgs):
        raise AssertionError("should use with_structured_output for a declared type")


def test_plain_uses_structured_output_for_record(monkeypatch):
    typ = Type(
        kind=ValueKind.OBJECT,
        fields={
            "name": Type.scalar(ValueKind.STRING),
            "score": Type.scalar(ValueKind.INTEGER),
        },
        required=frozenset({"name", "score"}),
    )
    model = _StructuredModel()
    monkeypatch.setattr(llm_clients, "model_from_config", lambda config: model)
    ctx = AgentRunContext(
        node_id="a",
        prompt="x",
        llm_config={"provider": "openai", "model": "gpt-4o"},
        output_type=typ,
    )
    out = plain(ctx)
    assert out.value == {"name": "Ada", "score": 9}  # a plain dict, not a pydantic obj


def test_plain_uses_ollama_structured_output_for_literal(monkeypatch):
    typ = Type(kind=ValueKind.STRING, tags=frozenset({"positive", "neutral"}))

    class _LiteralModel:
        def __init__(self):
            self.structured_called_with = None

        def with_structured_output(self, schema):
            self.structured_called_with = schema

            class _Bound:
                def invoke(self, msgs):
                    return schema.model_validate({"value": "neutral"})

            return _Bound()

        def invoke(self, msgs):
            raise AssertionError("Literal output must use structured generation")

    model = _LiteralModel()
    monkeypatch.setattr(llm_clients, "model_from_config", lambda config: model)
    ctx = AgentRunContext(
        node_id="a",
        prompt="classify",
        llm_config={"provider": "ollama", "model": "qwen2.5:72b"},
        output_type=typ,
    )

    assert plain(ctx).value == "neutral"
    assert model.structured_called_with is not None


class _TextModel:
    def invoke(self, msgs):
        class R:
            content = "hello"

        return R()


def test_plain_bare_str_text_passthrough(monkeypatch):
    monkeypatch.setattr(llm_clients, "model_from_config", lambda config: _TextModel())
    ctx = AgentRunContext(
        node_id="a", prompt="x",
        output_type=Type.scalar(ValueKind.STRING),
    )
    assert plain(ctx).value == "hello"  # unchanged behavior
