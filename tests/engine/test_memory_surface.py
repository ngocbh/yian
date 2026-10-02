"""The long_term_memory parser: sugar, mapping, long form, and shape errors."""

from __future__ import annotations

import pytest

from agent_composer.compose.helpers.errors import LoadDiagnostics, LoadError
from agent_composer.compose.helpers.memory import parse_memory_declarations
from agent_composer.memory.contracts import DeclSite

_SITE = DeclSite((), ("root",), None)


def _parse(value):
    return parse_memory_declarations(
        value, diagnostics=LoadDiagnostics(), source_range=None, site=_SITE
    )


def _shape(decls):
    return [(d.name, d.type, d.config, d.site) for d in decls]


def test_sugar_is_one_memory_named_after_its_type() -> None:
    assert _shape(_parse("episodic")) == [("episodic", "episodic", {}, _SITE)]


def test_mapping_and_long_form_keep_authored_order() -> None:
    value = {
        "lessons": "reflective",
        "notes": {"type": "episodic", "config": {"max_episodes": 50}},
        "plain": {"type": "vector"},
    }
    assert _shape(_parse(value)) == [
        ("lessons", "reflective", {}, _SITE),
        ("notes", "episodic", {"max_episodes": 50}, _SITE),
        ("plain", "vector", {}, _SITE),
    ]
    assert all(decl.origin is None for decl in _parse(value))


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ({}, "declares no memory"),
        (3, "type name or a mapping"),
        (["episodic"], "type name or a mapping"),
        ({"name": "reflective", "config": {"max_items": 3}}, "reserved"),
        ({"name": "reflective"}, "reserved"),
        ({"type": "episodic"}, "reserved"),
        ({"notes": {"type": "episodic", "extra": 1}}, "unknown key"),
        ({"notes": {"config": {}}}, "requires 'type'"),
        ({"notes": {"type": 3}}, "type must be a string"),
        ({"notes": {"type": "episodic", "config": []}}, "config must be a mapping"),
        ({"notes": 3}, "type name or a mapping"),
        ({3: "episodic"}, "must be a string"),
        ({"9notes": "episodic"}, "memory name"),
        ({"notes": "my.type"}, "memory type"),
        ("", "memory name"),
        ({"notes": {"type": "episodic", "config": {"path": "x.sqlite"}}}, "not allowed|forbidden"),
        ({"notes": {"type": "episodic", "config": {"max_items": 3}}}, "unknown key"),
    ],
)
def test_malformed_values_are_load_errors(value, message: str) -> None:
    with pytest.raises(LoadError, match=message):
        _parse(value)


from agent_composer.compose import load_flow


def _flow(memory_block: str) -> str:
    return (
        "id: surface\n"
        "name: surface\n"
        f"{memory_block}"
        "nodes:\n"
        "  echo:\n"
        "    kind: code\n"
        "    code: tests.engine._compose_codefns:echo\n"
        "    input: {topic: hello}\n"
        "    output: str\n"
        "output: ${echo.output}\n"
    )


def test_root_flow_declarations_are_parsed_and_stamped() -> None:
    flow = load_flow(_flow(
        "long_term_memory:\n"
        "  lessons: reflective\n"
        "  notes:\n"
        "    type: episodic\n"
        "    config: {max_episodes: 50}\n"
    ))
    assert _shape(flow.compiled.memories) == [
        ("lessons", "reflective", {}, _SITE),
        ("notes", "episodic", {"max_episodes": 50}, _SITE),
    ]
    resolver = flow.origin_resolver
    lines = [resolver.resolve(d.origin).range.start_line for d in flow.compiled.memories]
    assert lines == [4, 5]


def test_null_flow_declaration_is_the_same_as_absent() -> None:
    assert load_flow(_flow("long_term_memory: null\n")).compiled.memories == ()


@pytest.mark.parametrize(
    "block",
    ["long_term_memory: {}\n", "long_term_memory: {name: reflective, config: {}}\n"],
)
def test_flow_shape_errors_point_at_the_value(block: str) -> None:
    with pytest.raises(LoadError) as caught:
        load_flow(_flow(block))
    assert caught.value.diagnostic.frames[-1].range.start_line == 3


def test_def_flow_declarations_carry_the_def_owner() -> None:
    flow = load_flow(
        "id: outer\n"
        "name: outer\n"
        "defs:\n"
        "  child:\n"
        "    long_term_memory: {notes: episodic}\n"
        "    nodes:\n"
        "      echo:\n"
        "        kind: code\n"
        "        code: tests.engine._compose_codefns:echo\n"
        "        input: {topic: child}\n"
        "        output: str\n"
        "    output: ${echo.output}\n"
        "nodes:\n"
        "  invoke: {kind: call, call: child}\n"
        "output: ${invoke.output}\n"
    )
    child = flow.compiled.nodes["invoke"].child
    assert _shape(child.memories) == [
        ("notes", "episodic", {}, DeclSite((), ("def", "child"), None))
    ]
