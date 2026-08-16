"""External callables — the default LOCAL file resolver: search path, inert paths, cycle, cache."""

from collections import Counter
from pathlib import Path

import pytest

from agent_composer.compile import iter_flow_origins
from agent_composer.compile.manifest import flow_fingerprint
from agent_composer.compose import loader as compose_loader
from agent_composer.compose.helpers.errors import LoadError
from agent_composer.compose.loader import load_flow
from agent_composer.run import run_flow

_CODE = "tests.seeds.fns:one_line_summary"


def _flow(flow_id, *, call=None, uses=None, system=None, version=None):
    lines = [f"id: {flow_id}", f"name: {flow_id}"]
    if version is not None:
        lines.append(f'version: "{version}"')  # quote: a bare 2.0/yes/null parses non-str
    lines += ["input: {x: str}", "nodes:"]
    if call:
        lines.append(f'  a: {{kind: call, call: {call}, input: {{x: "${{input.x}}"}}}}')
    else:
        lines.append(f'  a: {{kind: code, code: {_CODE}, input: {{x: "${{input.x}}"}}}}')
    lines.append('output: {y: "${a.output}"}')
    if uses:
        lines.append("uses:")
        lines += [f"  {k}: {v}" for k, v in uses.items()]
    if system:
        lines.append("system:")
        lines.append("  paths: [" + ", ".join(system) + "]")
    return "\n".join(lines) + "\n"


def _write(d, filename, flow_id, **kw):
    f = d / f"{filename}.yaml"
    f.write_text(_flow(flow_id, **kw))
    return f


def _capture_external_source_access(monkeypatch):
    """Return lists recording file reads and source-document parse labels."""

    reads: list[Path] = []
    parses: list[str] = []
    original_read = Path.read_text
    original_parse = compose_loader.parse_source_document

    def track_read(path, *args, **kwargs):
        reads.append(path.resolve())
        return original_read(path, *args, **kwargs)

    def track_parse(text, label):
        parses.append(label)
        return original_parse(text, label)

    monkeypatch.setattr(Path, "read_text", track_read)
    monkeypatch.setattr(compose_loader, "parse_source_document", track_parse)
    return reads, parses


def _origin_keys(flow) -> set[str]:
    """Return the distinct opaque origin keys reachable from one compiled flow."""
    return {origin.key for origin in iter_flow_origins(flow)}


# --------------------------------------------------------------------------- #
# system.paths inert without search_paths
# --------------------------------------------------------------------------- #


def test_system_paths_inert_without_search_paths():
    # A file declaring system: paths: + a non-def call:, loaded with NO search_paths,
    # must stay loud (external=None) — never silently hit the filesystem.
    text = _flow("f", call="mom", uses={"mom": "child"}, system=["../shared"])
    with pytest.raises(LoadError, match="needs a search path or child_resolver"):
        load_flow(text)


def test_no_search_paths_no_resolver_unchanged():
    # A plain non-def call: with neither search_paths nor child_resolver is loud (today's behavior).
    text = _flow("f", call="mom", uses={"mom": "child"})
    with pytest.raises(LoadError):
        load_flow(text)


# --------------------------------------------------------------------------- #
# make_file_resolver: relative resolve, versioned filename, precedence, skip, miss
# --------------------------------------------------------------------------- #


def test_relative_resolve_sibling(tmp_path):
    _write(tmp_path, "child", "child")
    parent = _write(tmp_path, "parent", "parent", call="ch", uses={"ch": "child"})
    assert load_flow(parent.read_text(), search_paths=[tmp_path]) is not None


def test_version_guard_matches(tmp_path):
    _write(tmp_path, "child", "child", version="v1")  # file child.yaml, declares version: v1
    parent = _write(tmp_path, "p", "p", call="ch", uses={"ch": "child@v1"})
    assert load_flow(parent.read_text(), search_paths=[tmp_path]) is not None


def test_version_guard_matches_dotted_tag(tmp_path):
    _write(tmp_path, "child", "child", version="2.0")  # dotted tag not mangled
    parent = _write(tmp_path, "p", "p", call="ch", uses={"ch": "child@2.0"})
    assert load_flow(parent.read_text(), search_paths=[tmp_path]) is not None


def test_version_guard_mismatch_is_loud_and_located(tmp_path):
    _write(tmp_path, "child", "child", version="v2")
    parent = _write(tmp_path, "p", "p", call="ch", uses={"ch": "child@v1"})
    with pytest.raises(LoadError) as exc:
        load_flow(parent.read_text(), search_paths=[tmp_path])
    assert "version" in str(exc.value)
    assert exc.value.line is not None  # located at the uses: section (eager-loop wrap)


def test_version_absent_ref_skips_guard(tmp_path):
    _write(tmp_path, "child", "child", version="v9")
    parent = _write(tmp_path, "p", "p", call="ch", uses={"ch": "child"})  # no @v -> no guard
    assert load_flow(parent.read_text(), search_paths=[tmp_path]) is not None


def test_version_pinned_but_file_unversioned_is_loud(tmp_path):
    _write(tmp_path, "child", "child")  # no version: -> None
    parent = _write(tmp_path, "p", "p", call="ch", uses={"ch": "child@v1"})
    with pytest.raises(LoadError, match="version"):
        load_flow(parent.read_text(), search_paths=[tmp_path])


def test_diamond_conflicting_version_pins_is_loud(tmp_path):
    # the same child.yaml reached via two pins (@v1 ok, @v2 mismatch): the guard runs
    # per-pin over the shared diamond cache, so the cached child still rejects @v2.
    _write(tmp_path, "child", "child", version="v1")
    _write(tmp_path, "m1", "m1", call="c", uses={"c": "child@v1"})
    _write(tmp_path, "m2", "m2", call="c", uses={"c": "child@v2"})
    parent = (tmp_path / "p.yaml")
    parent.write_text(
        "id: p\nname: p\ninput: {x: str}\nnodes:\n"
        '  a: {kind: call, call: a1, input: {x: "${input.x}"}}\n'
        '  b: {kind: call, call: a2, input: {x: "${a.output}"}}\n'
        'output: {y: "${b.output}"}\nuses:\n  a1: m1\n  a2: m2\n'
    )
    with pytest.raises(LoadError, match="version"):
        load_flow(parent.read_text(), search_paths=[tmp_path])


def test_nested_path(tmp_path):
    sub = tmp_path / "library"
    sub.mkdir()
    _write(sub, "relevance", "relevance")
    parent = _write(tmp_path, "p", "p", call="m", uses={"m": "library/relevance"})
    assert load_flow(parent.read_text(), search_paths=[tmp_path]) is not None


def test_missing_ref_errors(tmp_path):
    parent = _write(tmp_path, "p", "p", call="ch", uses={"ch": "nope"})
    with pytest.raises(LoadError, match="not found on search path"):
        load_flow(parent.read_text(), search_paths=[tmp_path])


def test_malformed_external_child_is_located_load_error(tmp_path):
    (tmp_path / "child.yaml").write_text("id: [")
    parent = _write(tmp_path, "p", "p", call="ch", uses={"ch": "child"})

    with pytest.raises(LoadError, match="invalid YAML") as caught:
        load_flow(parent.read_text(), search_paths=[tmp_path])

    assert caught.value.range is not None
    assert caught.value.range.start_line == 1


def test_search_path_precedence_own_dir_first(tmp_path):
    d2 = tmp_path / "shared"
    d2.mkdir()
    _write(tmp_path, "child", "child")  # own dir
    _write(d2, "child", "child")        # also on system.paths
    parent = _write(tmp_path, "p", "p", call="ch", uses={"ch": "child"}, system=["shared"])
    assert load_flow(parent.read_text(), search_paths=[tmp_path]) is not None


def test_nonexistent_search_dir_skipped(tmp_path):
    _write(tmp_path, "child", "child")
    parent = _write(tmp_path, "p", "p", call="ch", uses={"ch": "child"}, system=["does_not_exist"])
    assert load_flow(parent.read_text(), search_paths=[tmp_path]) is not None


def test_child_own_uses_re_roots(tmp_path):
    # parent -> mid (in subdir) -> leaf (sibling of mid). mid's uses resolve at mid's dir.
    sub = tmp_path / "lib"
    sub.mkdir()
    _write(sub, "leaf", "leaf")
    _write(sub, "mid", "mid", call="l", uses={"l": "leaf"})
    parent = _write(tmp_path, "p", "p", call="m", uses={"m": "lib/mid"})
    assert load_flow(parent.read_text(), search_paths=[tmp_path]) is not None


# --------------------------------------------------------------------------- #
# cross-file cycle guard + diamond cache reuse
# --------------------------------------------------------------------------- #


def test_load_context_has_one_typed_flow_cache():
    context = compose_loader._LoadCtx()

    assert context.loading == set()
    assert context.flows == {}
    assert not hasattr(context, "cache")
    assert not hasattr(context, "documents")


def test_cross_file_cycle_rejected(tmp_path):
    # file fa calls alias `b` -> file fb; fb calls alias `back` -> file fa. (Aliases avoid
    # the fixed node id `a`.) The shared loading-set catches the fa->fb->fa cycle.
    _write(tmp_path, "fa", "fa", call="b", uses={"b": "fb"})
    _write(tmp_path, "fb", "fb", call="back", uses={"back": "fa"})
    with pytest.raises(LoadError, match="cycle across files"):
        load_flow((tmp_path / "fa.yaml").read_text(), search_paths=[tmp_path])


def test_two_aliases_of_one_file_read_and_parse_once(tmp_path, monkeypatch):
    child = _write(tmp_path, "child", "child")
    parent = tmp_path / "parent.yaml"
    parent.write_text(
        "id: parent\nname: parent\ninput: {x: str}\nnodes:\n"
        '  a: {kind: call, call: first, input: {x: "${input.x}"}}\n'
        '  b: {kind: call, call: second, input: {x: "${a.output}"}}\n'
        'output: {y: "${b.output}"}\n'
        "uses:\n  first: child\n  second: child\n"
    )
    parent_text = parent.read_text()
    reads, parses = _capture_external_source_access(monkeypatch)

    loaded = load_flow(parent_text, search_paths=[tmp_path])
    first = loaded.compiled.nodes["a"].child
    second = loaded.compiled.nodes["b"].child
    first_origins = _origin_keys(first)
    second_origins = _origin_keys(second)

    assert Counter(reads) == Counter({child.resolve(): 1})
    assert Counter(parses) == Counter({"<flow>": 1, "child.yaml": 1})
    assert first is not second
    assert first_origins.isdisjoint(second_origins)
    assert loaded.origin_resolver is not None
    for origin in iter_flow_origins(first):
        assert loaded.origin_resolver.resolve(origin).label == "child.yaml"
    for origin in iter_flow_origins(second):
        assert loaded.origin_resolver.resolve(origin).label == "child.yaml"


def test_repeated_calls_through_one_alias_reuse_one_rebased_child(
    tmp_path,
    monkeypatch,
):
    child = _write(tmp_path, "child", "child")
    parent = tmp_path / "parent.yaml"
    parent.write_text(
        "id: parent\nname: parent\ninput: {x: str}\nnodes:\n"
        '  a: {kind: call, call: child_view, input: {x: "${input.x}"}}\n'
        '  b: {kind: call, call: child_view, input: {x: "${a.output}"}}\n'
        'output: {y: "${b.output}"}\n'
        "uses:\n  child_view: child\n"
    )
    parent_text = parent.read_text()
    reads, parses = _capture_external_source_access(monkeypatch)

    loaded = load_flow(parent_text, search_paths=[tmp_path])
    first = loaded.compiled.nodes["a"].child
    second = loaded.compiled.nodes["b"].child

    assert Counter(reads) == Counter({child.resolve(): 1})
    assert Counter(parses) == Counter({"<flow>": 1, "child.yaml": 1})
    assert first is second
    assert _origin_keys(first) == _origin_keys(second)
    assert loaded.origin_resolver is not None
    for origin in iter_flow_origins(first):
        assert loaded.origin_resolver.resolve(origin).label == "child.yaml"


def test_alias_rebasing_does_not_mutate_the_canonical_path_cache(tmp_path):
    _write(tmp_path, "child", "child")
    parent = _flow(
        "parent",
        call="child_view",
        uses={"child_view": "child"},
    )
    context = compose_loader._LoadCtx()
    resolver = compose_loader.make_file_resolver([tmp_path], context)
    canonical = resolver("child")
    canonical_origins = _origin_keys(canonical.compiled)
    canonical_fingerprint = flow_fingerprint(canonical.compiled)
    canonical_resolver = canonical.origin_resolver
    assert canonical_resolver is not None
    canonical_frames = {
        origin.key: canonical_resolver.resolve(origin)
        for origin in iter_flow_origins(canonical.compiled)
    }

    loaded = load_flow(parent, child_resolver=resolver)
    embedded = loaded.compiled.nodes["a"].child

    assert resolver("child") is canonical
    assert canonical.origin_resolver is canonical_resolver
    assert embedded is not canonical.compiled
    assert _origin_keys(canonical.compiled) == canonical_origins
    assert {
        origin.key: canonical.origin_resolver.resolve(origin)
        for origin in iter_flow_origins(canonical.compiled)
    } == canonical_frames
    assert _origin_keys(embedded).isdisjoint(canonical_origins)
    assert flow_fingerprint(embedded) == canonical_fingerprint


def test_diamond_reuse_reads_and_parses_each_path_once(tmp_path, monkeypatch):
    # p -> (m1, m2) -> leaf; leaf reached twice must load once (cache), no error.
    leaf = _write(tmp_path, "leaf", "leaf")
    mid1 = _write(tmp_path, "mid1", "mid1", call="l", uses={"l": "leaf"})
    mid2 = _write(tmp_path, "mid2", "mid2", call="l", uses={"l": "leaf"})
    parent = (tmp_path / "p.yaml")
    parent.write_text(
        "id: p\nname: p\ninput: {x: str}\nnodes:\n"
        '  a: {kind: call, call: m1, input: {x: "${input.x}"}}\n'
        '  b: {kind: call, call: m2, input: {x: "${a.output}"}}\n'
        'output: {y: "${b.output}"}\nuses:\n  m1: mid1\n  m2: mid2\n'
    )
    parent_text = parent.read_text()
    reads, parses = _capture_external_source_access(monkeypatch)

    loaded = load_flow(parent_text, search_paths=[tmp_path])
    left_leaf = loaded.compiled.nodes["a"].child.nodes["a"].child
    right_leaf = loaded.compiled.nodes["b"].child.nodes["a"].child
    left_origins = _origin_keys(left_leaf)
    right_origins = _origin_keys(right_leaf)

    assert Counter(reads) == Counter(
        {
            leaf.resolve(): 1,
            mid1.resolve(): 1,
            mid2.resolve(): 1,
        }
    )
    assert Counter(parses) == Counter(
        {
            "<flow>": 1,
            "leaf.yaml": 1,
            "mid1.yaml": 1,
            "mid2.yaml": 1,
        }
    )
    assert left_leaf is not right_leaf
    assert left_origins.isdisjoint(right_origins)
    assert loaded.origin_resolver is not None
    for origin in iter_flow_origins(left_leaf):
        assert loaded.origin_resolver.resolve(origin).label == "leaf.yaml"
    for origin in iter_flow_origins(right_leaf):
        assert loaded.origin_resolver.resolve(origin).label == "leaf.yaml"


# --------------------------------------------------------------------------- #
# runnable seed 24 (uses: an external CODE-only sibling via search path)
# --------------------------------------------------------------------------- #

_SEEDS = Path(__file__).resolve().parents[1].parent / "tests" / "seeds"


def test_seed_24_runs_via_search_path():
    parent = (_SEEDS / "24-uses-external.yaml").read_text()
    loaded = load_flow(parent, search_paths=[_SEEDS])
    res = run_flow(loaded, {"rating": 0.5, "rationale": "rising"})
    assert res.status == "succeeded", res.failure
    assert "rising" in str(res.output)
