"""Import scoping: a uses: alias prefixes every reachable memory key."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from agent_composer.compile.memory_layout import (
    resolve_memory_layout,
    scope_memory_declarations,
)
from agent_composer.compile.model import CompiledFlow
from agent_composer.compose import load_flow
from agent_composer.compose.helpers.errors import LoadError
from agent_composer.memory.contracts import DeclSite, MemoryDecl, MemorySetupError

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


def test_scoping_leaves_the_cached_child_untouched() -> None:
    lib = load_flow(_lib())
    flow = load_flow(
        _root({"a": "x", "b": "y"}, {"x": "lib", "y": "lib"}),
        child_resolver=lambda flow_id, version: lib,
    )
    assert _keys(flow) == ["x.notes", "y.notes"]
    # The resolver handed the same object to both aliases; neither rewrite reached it.
    assert [d.key for d in lib.compiled.memories] == ["notes"]


def test_library_without_memory_loads_under_a_dotted_alias(tmp_path: Path) -> None:
    (tmp_path / "lib.yaml").write_text(_lib("null"))
    flow = load_flow(_root({"a": "my.lib"}, {"my.lib": "lib"}), search_paths=[tmp_path])
    assert _keys(flow) == []


def test_malformed_imported_graph_is_a_load_error() -> None:
    lib = load_flow(_lib())
    # A node filed under a key other than its id; the memory walk rejects it.
    lib.compiled.nodes = {"b": lib.compiled.nodes["a"]}
    with pytest.raises(LoadError, match="does not match node id 'a'"):
        load_flow(_root({"n": "lib"}, {"lib": "lib"}), child_resolver=lambda flow_id, version: lib)


def _flow_with(*decls: MemoryDecl) -> CompiledFlow:
    flow = load_flow(
        "id: f\nname: f\nnodes:\n"
        "  echo: {kind: code, code: tests.engine._compose_codefns:echo, "
        "input: {topic: hi}, output: str}\n"
        "output: ${echo.output}\n"
    ).compiled
    flow.memories = decls
    return flow


def _decl(name: str, scope: tuple[str, ...] = ()) -> MemoryDecl:
    return MemoryDecl(name, "episodic", {}, DeclSite(scope, ("root",), None))


def test_scope_segment_must_match_the_name_pattern() -> None:
    with pytest.raises(MemorySetupError, match=r"alias '1lib'.*\[A-Za-z_\]"):
        resolve_memory_layout(_flow_with(_decl("notes", ("1lib",))))


def test_keys_equal_under_casefold_are_rejected() -> None:
    with pytest.raises(MemorySetupError, match="'Notes' and 'notes' differ only by case"):
        resolve_memory_layout(_flow_with(_decl("notes"), _decl("Notes")))


def test_aliases_equal_under_casefold_are_rejected() -> None:
    with pytest.raises(MemorySetupError, match="'LIB.notes' and 'lib.notes'"):
        resolve_memory_layout(
            _flow_with(_decl("notes", ("lib",)), _decl("notes", ("LIB",)))
        )


def test_dotted_alias_is_rejected_at_load(tmp_path: Path) -> None:
    (tmp_path / "lib.yaml").write_text(_lib())
    with pytest.raises(LoadError, match="'my.lib'"):
        load_flow(_root({"a": "my.lib"}, {"my.lib": "lib"}), search_paths=[tmp_path])


def test_dotted_alias_is_named_before_a_rendered_key_clash(tmp_path: Path) -> None:
    # "a.b" + notes and "a" -> "b" + notes both render as a.b.notes.
    (tmp_path / "lib.yaml").write_text(_lib())
    (tmp_path / "lib2.yaml").write_text(_lib("{notes: reflective}"))
    (tmp_path / "mid.yaml").write_text(_lib("null", call="b", uses={"b": "lib2"}))
    with pytest.raises(LoadError, match="alias 'a.b'") as caught:
        load_flow(
            _root({"n1": "a.b", "n2": "a"}, {"a.b": "lib", "a": "mid"}),
            search_paths=[tmp_path],
        )
    assert "different type or config" not in str(caught.value)


def test_imported_flow_memory_is_owned_by_a_scoped_root_site(tmp_path: Path) -> None:
    (tmp_path / "lib.yaml").write_text(_lib())
    flow = load_flow(
        _root({"a": "lib"}, {"lib": "lib"}, "{notes: episodic}"),
        search_paths=[tmp_path],
    )
    owners = {m.key: m.owners for m in resolve_memory_layout(flow.compiled).memories}
    assert owners == {
        "lib.notes": (DeclSite(("lib",), ("root",), None),),
        "notes": (DeclSite((), ("root",), None),),
    }


def test_a_clash_inside_an_import_names_the_scoped_sites(tmp_path: Path) -> None:
    (tmp_path / "lib.yaml").write_text(
        "id: lib\nname: lib\n"
        "long_term_memory: {notes: episodic}\n"
        "llm_config: {provider: openai, model: fake-model}\n"
        "input: {x: str}\n"
        "nodes:\n"
        "  ask:\n"
        "    kind: agent\n"
        "    long_term_memory: {notes: reflective}\n"
        "    prompt: hi\n"
        "    output: str\n"
        "output: ${ask.output}\n"
    )
    with pytest.raises(LoadError) as caught:
        load_flow(_root({"a": "lib"}, {"lib": "lib"}), search_paths=[tmp_path])
    diagnostic = caught.value.diagnostic
    assert diagnostic.frames[-1].label == "lib.yaml"
    assert diagnostic.message.endswith("at lib: root and lib: root AGENT 'ask'")
    assert diagnostic.notes[0] == "also declared at lib: root AGENT 'ask'"


def test_a_related_note_at_the_primary_location_is_skipped(tmp_path: Path) -> None:
    (tmp_path / "lib.yaml").write_text(_lib())
    # Both aliases map one declaration in lib.yaml, so both origins share a location.
    with pytest.raises(LoadError, match="differ only by case") as caught:
        load_flow(_root({"a": "Lib", "b": "lib"}, {"Lib": "lib", "lib": "lib"}), search_paths=[tmp_path])
    diagnostic = caught.value.diagnostic
    assert diagnostic.frames[-1].label == "lib.yaml"
    assert diagnostic.notes == ()


def test_a_related_note_repeating_an_earlier_location_is_skipped() -> None:
    from types import SimpleNamespace

    from agent_composer.compose.loader import _memory_setup_load_error
    from agent_composer.diagnostics import DiagnosticFrame, OriginRef, SourceRange

    lines = {"p": 1, "a": 2, "a2": 2, "b": 3}

    class _Resolver:
        def resolve(self, origin):
            line = lines[origin.key]
            return DiagnosticFrame("f.yaml", "x\ny\nz\n", SourceRange(line, 1, line, 2))

    exc = MemorySetupError(
        "clash",
        origin=OriginRef("p"),
        related=(OriginRef("a"), OriginRef("a2"), OriginRef("b")),
    )
    error = _memory_setup_load_error(SimpleNamespace(origin_resolver=_Resolver()), exc)
    assert error.diagnostic.notes == (
        "related declaration at f.yaml:2:1",
        "related declaration at f.yaml:3:1",
    )
