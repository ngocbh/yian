"""Import scoping: a uses: alias prefixes every reachable memory key."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from agent_composer.compile.memory_layout import (
    resolve_memory_layout,
    scope_memory_declarations,
)
from agent_composer.compose import load_flow
from agent_composer.compose.helpers.errors import LoadError
from agent_composer.memory.contracts import DeclSite

_CODE = "tests.engine._compose_codefns:echo_x"


def _lib(memory: str = "{notes: episodic}", *, call: str | None = None, uses=None) -> str:
    lines = ["id: lib", "name: lib", f"long_term_memory: {memory}", "input: {x: str}", "nodes:"]
    if call:
        lines.append(f'  a: {{kind: call, call: {call}, input: {{x: "${{input.x}}"}}}}')
    else:
        lines.append(
            f'  a: {{kind: code, code: {_CODE}, input: {{x: "${{input.x}}"}}, output: str}}'
        )
    lines.append('output: {y: "${a.output}"}')
    if uses:
        lines.append("uses:")
        lines += [f"  {alias}: {target}" for alias, target in uses.items()]
    return "\n".join(lines) + "\n"


def _root(calls: dict[str, str], uses: dict[str, str], memory: str | None = None) -> str:
    lines = ["id: root", "name: root"]
    if memory is not None:
        lines.append(f"long_term_memory: {memory}")
    lines += ["input: {x: str}", "nodes:"]
    for node, target in calls.items():
        lines.append(f'  {node}: {{kind: call, call: {target}, input: {{x: "${{input.x}}"}}}}')
    lines.append('output: {y: "${input.x}"}')
    lines.append("uses:")
    lines += [f"  {alias}: {target}" for alias, target in uses.items()]
    return "\n".join(lines) + "\n"


def _keys(flow) -> list[str]:
    return [memory.key for memory in resolve_memory_layout(flow.compiled).memories]


def test_scope_rewrite_prefixes_every_reachable_declaration() -> None:
    flow = load_flow(
        "id: outer\nname: outer\n"
        "long_term_memory: {notes: episodic}\n"
        "llm_config: {provider: openai, model: fake-model}\n"
        "defs:\n"
        "  child:\n"
        "    long_term_memory: {facts: reflective}\n"
        "    nodes:\n"
        "      ask:\n"
        "        kind: agent\n"
        "        long_term_memory: {chat: session}\n"
        "        prompt: hi\n"
        "        output: str\n"
        "    output: ${ask.output}\n"
        "nodes:\n"
        "  one: {kind: call, call: child}\n"
        "  two: {kind: call, call: child}\n"
        "output: ${one.output}\n"
    )
    compiled = flow.compiled
    scope_memory_declarations(compiled, "lib")
    assert [d.site for d in compiled.memories] == [DeclSite(("lib",), ("root",), None)]
    child = compiled.nodes["one"].child
    assert [d.key for d in child.memories] == ["lib.facts"]
    assert [d.key for d in child.nodes["ask"].memories] == ["lib.chat"]
    # Whether or not the two call sites share one child object, each
    # declaration gains the alias exactly once.
    other = compiled.nodes["two"].child
    assert [d.key for d in other.memories] == ["lib.facts"]
    assert [d.key for d in other.nodes["ask"].memories] == ["lib.chat"]
    assert sorted(_keys(flow)) == ["lib.chat", "lib.facts", "lib.notes"]


def test_scope_rewrite_visits_a_shared_node_once() -> None:
    flow = load_flow(
        "id: outer\nname: outer\n"
        "llm_config: {provider: openai, model: fake-model}\n"
        "defs:\n"
        "  child:\n"
        "    nodes:\n"
        "      ask:\n"
        "        kind: agent\n"
        "        long_term_memory: {chat: session}\n"
        "        prompt: hi\n"
        "        output: str\n"
        "    output: ${ask.output}\n"
        "nodes:\n"
        "  one: {kind: call, call: child}\n"
        "  two: {kind: call, call: child}\n"
        "output: ${one.output}\n"
    )
    compiled = flow.compiled
    # Two distinct child flow objects that hold the same AGENT node object.
    first = compiled.nodes["one"].child
    second = copy.copy(first)
    second.nodes = dict(first.nodes)
    compiled.nodes["two"].child = second
    scope_memory_declarations(compiled, "lib")
    assert [d.key for d in first.nodes["ask"].memories] == ["lib.chat"]


def test_two_libraries_with_the_same_name_stay_independent(tmp_path: Path) -> None:
    (tmp_path / "liba.yaml").write_text(_lib())
    (tmp_path / "libb.yaml").write_text(_lib())
    root = _root({"a": "liba", "b": "libb"}, {"liba": "liba", "libb": "libb"}, "{notes: episodic}")
    flow = load_flow(root, search_paths=[tmp_path])
    assert _keys(flow) == ["liba.notes", "libb.notes", "notes"]


def test_one_file_under_two_aliases_gives_two_sets(tmp_path: Path) -> None:
    (tmp_path / "lib.yaml").write_text(
        _lib(
            "null",
            call="inner",
        ).replace(
            "nodes:\n",
            "defs:\n"
            "  inner:\n"
            "    input: {x: str}\n"
            "    nodes:\n"
            "      ask:\n"
            "        kind: agent\n"
            "        llm_config: {provider: openai, model: fake-model}\n"
            "        long_term_memory: {chat: session}\n"
            "        prompt: hi\n"
            "        output: str\n"
            "    output: ${ask.output}\n"
            "nodes:\n",
        )
    )
    flow = load_flow(_root({"a": "one", "b": "two"}, {"one": "lib", "two": "lib"}), search_paths=[tmp_path])
    layout = resolve_memory_layout(flow.compiled)
    assert [m.key for m in layout.memories] == ["one.chat", "two.chat"]
    assert [m.owners for m in layout.memories] == [
        (DeclSite(("one",), ("def", "inner"), "ask"),),
        (DeclSite(("two",), ("def", "inner"), "ask"),),
    ]


def test_diamond_import_gives_independent_sets(tmp_path: Path) -> None:
    (tmp_path / "d.yaml").write_text(_lib())
    (tmp_path / "b.yaml").write_text(_lib("null", call="d", uses={"d": "d"}))
    (tmp_path / "c.yaml").write_text(_lib("null", call="d", uses={"d": "d"}))
    # A uses: alias may not equal a node id, so the CALL nodes are nb and nc.
    flow = load_flow(_root({"nb": "b", "nc": "c"}, {"b": "b", "c": "c"}), search_paths=[tmp_path])
    assert _keys(flow) == ["b.d.notes", "c.d.notes"]


def test_scoping_leaves_the_cached_child_untouched(tmp_path: Path) -> None:
    (tmp_path / "lib.yaml").write_text(_lib())
    flow = load_flow(_root({"a": "x", "b": "y"}, {"x": "lib", "y": "lib"}), search_paths=[tmp_path])
    assert _keys(flow) == ["x.notes", "y.notes"]
    alone = load_flow((tmp_path / "lib.yaml").read_text(), search_paths=[tmp_path])
    assert _keys(alone) == ["notes"]
