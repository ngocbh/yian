"""External callables — `uses:` surface + resolution wiring (defs-first → uses arm)."""

import pytest

from agent_composer.compose.helpers.errors import LoadError
from agent_composer.compose.loader import load_flow
from agent_composer.compose.parse.file import parse_file
from agent_composer.compose.helpers.uses import UsesRef, parse_uses_ref
from agent_composer.compose.source import SourceDocument


# A minimal valid single-node flow; each test appends its own `uses:`/`system:` sections.
_BASE = """\
id: f
name: f
input: {x: str}
nodes:
  a: {kind: code, code: tests.seeds.fns:one_line_summary, input: {x: "${input.x}"}}
output: {y: "${a.output}"}
"""


# --------------------------------------------------------------------------- #
# parse_uses_ref grammar: [<scheme>:]<path>[@<version>]
# --------------------------------------------------------------------------- #


def test_parse_local_with_version():
    assert parse_uses_ref("library/relevance@v1") == UsesRef(None, "library/relevance", "v1")


def test_parse_local_no_version():
    assert parse_uses_ref("helper") == UsesRef(None, "helper", None)


def test_parse_nested_path():
    assert parse_uses_ref("a/b/c") == UsesRef(None, "a/b/c", None)


def test_parse_hub_scheme():
    assert parse_uses_ref("hub:quant/fancy@2.0") == UsesRef("hub", "quant/fancy", "2.0")


def test_parse_hub_scheme_no_version():
    assert parse_uses_ref("hub:quant/fancy") == UsesRef("hub", "quant/fancy", None)


def test_parse_multi_at_last_wins():
    assert parse_uses_ref("a@b@c") == UsesRef(None, "a@b", "c")


def test_parse_extracts_drive_letter_as_scheme():
    # parse only EXTRACTS the scheme (lowercased); the unknown-scheme rejection is the
    # resolver arm's job, so a Windows-drive-looking ref never hits disk.
    assert parse_uses_ref("C:/x") == UsesRef("c", "/x", None)


@pytest.mark.parametrize("bad", ["", "  ", "@v1", "hub:", "hub:@v1", "name@"])
def test_malformed_rejected(bad):
    with pytest.raises(LoadError):
        parse_uses_ref(bad)


# --------------------------------------------------------------------------- #
# uses:/system: top-level sections on ComposeFile
# --------------------------------------------------------------------------- #


def test_parse_file_accepts_uses_and_system():
    text = _BASE + "uses:\n  mom: library/relevance@v1\nsystem:\n  paths: [../shared]\n"
    f = parse_file(text)
    assert f.uses == {"mom": "library/relevance@v1"}
    assert f.system == {"paths": ["../shared"]}


def test_parse_file_defaults_empty():
    f = parse_file(_BASE)
    assert f.uses == {}
    assert f.system == {}


def test_parse_file_unknown_top_level_still_rejected():
    with pytest.raises(LoadError, match="unknown top-level key"):
        parse_file(_BASE + "bogus: 1\n")


# --------------------------------------------------------------------------- #
# system: shape validation + ${system.paths} regression lock
# --------------------------------------------------------------------------- #


def test_system_section_rejects_unknown_key():
    with pytest.raises(LoadError, match="system: unknown key"):
        load_flow(_BASE + "system:\n  bogus: 1\n")


def test_system_paths_must_be_list_of_str():
    with pytest.raises(LoadError, match="system: paths"):
        load_flow(_BASE + "system:\n  paths: 5\n")


def test_system_paths_list_of_str_ok():
    # a well-shaped system: paths: + no external call -> loads clean (system.paths is inert
    # without search_paths; here it just must not error on shape).
    load_flow(_BASE + "system:\n  paths: [a, b]\n")


def test_system_paths_is_not_a_runtime_ref():
    # ${system.paths} stays a compile error via the strict ambient allow-list.
    text = _BASE.replace('"${input.x}"', '"${system.paths}"') + "system:\n  paths: [.]\n"
    with pytest.raises(LoadError):
        load_flow(text)


# --------------------------------------------------------------------------- #
# _CallableScope uses: arm + defs∩uses guard + hub:/scheme deferral
# --------------------------------------------------------------------------- #

# A child flow text (input x: str -> echoes a one-line verdict); used by fake resolvers.
_CHILD = """\
id: child
name: child
input: {x: str}
nodes:
  c: {kind: code, code: tests.seeds.fns:one_line_summary, input: {x: "${input.x}"}}
output: {y: "${c.output}"}
"""


def _with_call(alias):
    """A flow whose node `a` is `{kind: call, call: <alias>, input: {x: ...}}`."""
    return (
        "id: f\nname: f\ninput: {x: str}\n"
        f"nodes:\n  a: {{kind: call, call: {alias}, input: {{x: \"${{input.x}}\"}}}}\n"
        "output: {y: \"${a.output}\"}\n"
    )


def _fake_resolver(seen):
    def resolve(flow_id, version=None):
        seen.append((flow_id, version))
        return load_flow(_CHILD)
    return resolve


def test_uses_alias_resolves_via_external_with_str_version():
    seen = []
    text = _with_call("mom") + "uses:\n  mom: library/relevance@v1\n"
    load_flow(text, child_resolver=_fake_resolver(seen))
    assert seen == [("library/relevance", "v1")]   # alias -> (path, version:str), in the arm


def test_uses_alias_no_version():
    seen = []
    text = _with_call("mom") + "uses:\n  mom: helper\n"
    load_flow(text, child_resolver=_fake_resolver(seen))
    assert seen == [("helper", None)]


def test_hub_scheme_deferred():
    text = _with_call("m") + "uses:\n  m: hub:q/f@1\n"
    with pytest.raises(LoadError, match="marketplace not supported yet"):
        load_flow(text, child_resolver=_fake_resolver([]))


def test_unknown_scheme_rejected_at_resolve():
    text = _with_call("m") + "uses:\n  m: git:q/f@1\n"
    with pytest.raises(LoadError, match="unknown scheme"):
        load_flow(text, child_resolver=_fake_resolver([]))


def test_defs_uses_collision():
    text = (
        _with_call("dup")
        + "uses:\n  dup: child\n"
        + "defs:\n  dup:\n    input: {x: str}\n"
        "    nodes:\n      c: {kind: code, code: tests.seeds.fns:one_line_summary, input: {x: \"${x}\"}}\n"
        "    output: {y: \"${c.output}\"}\n"
    )
    with pytest.raises(LoadError, match="both defs: and uses:"):
        load_flow(text, child_resolver=_fake_resolver([]))


# --------------------------------------------------------------------------- #
# uses∩node-id guard + eager uses resolution + inline call(alias, ...)
# --------------------------------------------------------------------------- #


def test_uses_node_id_collision():
    # alias `a` collides with the node id `a`
    text = _with_call("a") + "uses:\n  a: child\n"
    with pytest.raises(LoadError, match="collides with a node id"):
        load_flow(text, child_resolver=_fake_resolver([]))


def test_uncalled_broken_alias_is_loud():
    # `dead` is declared but never called; eager resolution must still fire (hub: deferred)
    text = _BASE + "uses:\n  dead: hub:x/y@1\n"
    with pytest.raises(LoadError, match="marketplace not supported yet"):
        load_flow(text, child_resolver=_fake_resolver([]))


def test_uncalled_missing_alias_is_loud():
    # `dead` -> a local ref the fake resolver rejects; eager resolution surfaces it at load
    def reject(flow_id, version=None):
        raise LoadError(f"no such flow {flow_id!r}")
    text = _BASE + "uses:\n  dead: nope\n"
    expected = SourceDocument.parse(text).root.child("uses").child("dead").range
    with pytest.raises(LoadError) as caught:
        load_flow(text, child_resolver=reject)
    assert caught.value.range == expected
    assert caught.value.line == expected.start_line


def test_inline_call_routes_through_uses_arm():
    seen = []
    text = (
        "id: f\nname: f\ninput: {x: str}\n"
        "nodes:\n"
        "  a: {kind: code, code: tests.seeds.fns:one_line_summary, "
        "input: {x: \"call(mom, x=inputs.x)\"}}\n"
        "output: {y: \"${a.output}\"}\nuses:\n  mom: library/relevance@v1\n"
    )
    load_flow(text, child_resolver=_fake_resolver(seen))
    assert ("library/relevance", "v1") in seen


# --------------------------------------------------------------------------- #
# cross-flow type check through a uses: alias (resolver-agnostic)
# --------------------------------------------------------------------------- #


def test_e06_through_alias():
    # child expects x: int; parent binds a str through the alias -> a type mismatch at load.
    child_int = (
        "id: c\nname: c\ninput: {x: int}\n"
        "nodes:\n  c: {kind: code, code: tests.seeds.fns:const_one, input: {x: \"${input.x}\"}}\n"
        "output: {y: \"${c.output}\"}\n"
    )
    text = (
        "id: f\nname: f\ninput: {s: str}\n"
        "nodes:\n  a: {kind: call, call: mom, input: {x: \"${input.s}\"}}\n"
        "output: {y: \"${a.output}\"}\nuses:\n  mom: c@1\n"
    )
    with pytest.raises(LoadError):
        load_flow(text, child_resolver=lambda p, v=None: load_flow(child_int))


# --------------------------------------------------------------------------- #
# external-only-via-uses: a bare call: is NOT a backdoor to `external`
# --------------------------------------------------------------------------- #


def test_bare_call_without_uses_or_def_is_located_loaderror():
    """A bare `call:` to a name that is neither an in-file `defs:` nor a `uses:` alias is a
    LOCATED LoadError, even with a child_resolver present — external flows are reachable
    only through a `uses:` alias. The resolver is NEVER consulted."""
    seen: list = []
    text = _with_call("ghost")  # node `a` does `call: ghost`; NO uses: alias for it
    with pytest.raises(LoadError) as exc:
        load_flow(text, child_resolver=_fake_resolver(seen))
    assert "ghost" in str(exc.value)
    assert exc.value.line is not None        # located at the call node's line
    assert seen == []                         # the resolver was never reached
