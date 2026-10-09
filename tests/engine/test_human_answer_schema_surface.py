from pathlib import Path

from yian.compose import load_flow

_SEEDS = Path(__file__).resolve().parents[2] / "tests" / "seeds"


def test_human_input_answer_schema_type_uses_surface_name():
    loaded = load_flow((_SEEDS / "17-effects-human-wait.yaml").read_text())
    schemas = [
        n.answer_schema
        for n in loaded.compiled.nodes.values()
        if getattr(n, "answer_schema", None)
    ]
    assert schemas, "expected a HUMAN_INPUT node with an answer_schema"
    types = {e["type"] for sch in schemas for e in sch}
    assert types, "answer_schema entries carry a 'type'"
    # the LLM/host-facing surface is Python-typing, never the engine enum value
    for t in types:
        assert t in {"str", "int", "float", "bool", "object", "date", "datetime",
                     "list[str]", "list[int]", "list[float]", "list[bool]",
                     "list[object]", "list[Any]", "None"}
        assert "[integer]" not in t and "[number]" not in t and t not in {
            "string", "integer", "number", "boolean",
        }
