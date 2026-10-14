"""Integration tests for complete resolver installation at the Compose boundary."""

from copy import deepcopy
from dataclasses import replace

import pytest

from yian.compile import iter_flow_origins, rewrite_flow_origins
from yian.compile.manifest import flow_fingerprint
from yian.compose import ComposedFlow, load_flow
from yian.compose.helpers.errors import LoadError
from yian.compose.loader import _finalize_composed_flow
from yian.compose.source import SourceCatalog
from yian.diagnostics import OriginRef, ResolverTable
from yian.nodes.base import Node


_CHILD = """id: child
name: child
nodes:
  worker:
    kind: code
    code: return 1
    output: int
output: ${worker.output}
"""

_PARENT = """id: parent
name: parent
nodes:
  first:
    kind: call
    call: left
  second:
    kind: call
    call: right
output: ${second.output}
uses:
  left: child
  right: child
"""

_ONE_ALIAS_PARENT = """id: parent
name: parent
nodes:
  invoke:
    kind: call
    call: child_view
output: ${invoke.output}
uses:
  child_view: child
"""


def _origins(flow) -> dict[str, OriginRef]:
    """Return one canonical reachable origin per opaque key."""
    return {origin.key: OriginRef(origin.key) for origin in iter_flow_origins(flow)}


def _assert_child_view_uses_frame(error: LoadError) -> None:
    """Assert one external-child failure is owned by the importing alias site."""
    frames = error.diagnostic.frames
    assert len(frames) == 1
    assert frames[0].label == "<flow>"
    assert frames[0].range.start_line == 9
    assert frames[0].note == "uses alias 'child_view'"


def test_load_flow_installs_an_exact_resolver_and_filters_unused_defs():
    text = """id: complete-local
name: complete-local
input:
  n: int
defs:
  unused:
    kind: code
    code: return 2
    output: int
nodes:
  worker:
    kind: code
    code: return 1
    output: int
output: ${worker.output}
asserts:
  - input.n >= 0
  - worker.output >= 0
"""

    loaded = load_flow(text)
    origins = _origins(loaded.compiled)

    assert isinstance(loaded.origin_resolver, ResolverTable)
    assert {origin.key for origin in loaded.origin_resolver.origins} == set(origins)
    assert all(
        loaded.origin_resolver.resolve(origin) is not None
        for origin in origins.values()
    )
    assert not hasattr(loaded, "_local_catalog")


def test_source_free_load_has_no_resolver():
    loaded = load_flow("id: empty\nname: empty\nnodes: {}\n")

    assert _origins(loaded.compiled) == {}
    assert loaded.origin_resolver is None


def test_one_alias_reuses_one_rebased_child_for_multiple_calls():
    child = load_flow(_CHILD)
    text = _PARENT.replace("    call: right\n", "    call: left\n")

    loaded = load_flow(text, child_resolver=lambda _path, _version=None: child)
    first = loaded.compiled.nodes["first"].child
    second = loaded.compiled.nodes["second"].child

    assert first is second
    assert _origins(first) == _origins(second)
    assert {
        origin.key for origin in loaded.origin_resolver.origins
    } == set(_origins(loaded.compiled))
    assert all(
        loaded.origin_resolver.resolve(origin) is not None
        for origin in _origins(first).values()
    )


def test_two_aliases_rebase_canonical_child_without_mutating_it():
    child = load_flow(_CHILD)
    canonical_origins = _origins(child.compiled)
    canonical_fingerprint = flow_fingerprint(child.compiled)

    loaded = load_flow(
        _PARENT,
        child_resolver=lambda _path, _version=None: child,
    )
    first = loaded.compiled.nodes["first"].child
    second = loaded.compiled.nodes["second"].child
    first_origins = _origins(first)
    second_origins = _origins(second)
    child_origins = set(first_origins) | set(second_origins)
    parent_origins = set(_origins(loaded.compiled)) - child_origins

    assert first is not second
    assert set(first_origins).isdisjoint(second_origins)
    assert parent_origins
    assert _origins(child.compiled) == canonical_origins
    assert flow_fingerprint(first) == canonical_fingerprint
    assert flow_fingerprint(second) == canonical_fingerprint
    for origin in (*first_origins.values(), *second_origins.values()):
        frame = loaded.origin_resolver.resolve(origin)
        assert frame is not None
        assert frame.text == _CHILD
    for key in parent_origins:
        frame = loaded.origin_resolver.resolve(OriginRef(key))
        assert frame is not None
        assert frame.text == _PARENT


def test_delimiter_heavy_alias_rebases_opaque_child_origins():
    alias = "left/~: child"
    parent = f"""id: parent
name: parent
nodes:
  invoke:
    kind: call
    call: {alias!r}
output: ${{invoke.output}}
uses:
  {alias!r}: child
"""

    loaded = load_flow(
        parent,
        child_resolver=lambda _path, _version=None: load_flow(_CHILD),
    )
    embedded = loaded.compiled.nodes["invoke"].child

    assert loaded.origin_resolver is not None
    assert _origins(embedded)
    assert all(
        loaded.origin_resolver.resolve(origin).text == _CHILD
        for origin in _origins(embedded).values()
    )


@pytest.mark.parametrize("replacement", [None, ResolverTable()])
def test_originful_custom_child_requires_a_complete_resolver(replacement):
    child = replace(load_flow(_CHILD), origin_resolver=replacement)

    with pytest.raises(LoadError, match="origin resolver") as caught:
        load_flow(
            _ONE_ALIAS_PARENT,
            child_resolver=lambda _path, _version=None: child,
        )

    assert "alias 'child_view'" in str(caught.value)
    _assert_child_view_uses_frame(caught.value)


def test_custom_child_resolver_failure_is_wrapped_at_the_uses_boundary():
    class BrokenResolver:
        def resolve(self, _origin):
            raise RuntimeError("resolver exploded")

    child = replace(load_flow(_CHILD), origin_resolver=BrokenResolver())

    with pytest.raises(LoadError, match="resolver exploded") as caught:
        load_flow(
            _ONE_ALIAS_PARENT,
            child_resolver=lambda _path, _version=None: child,
        )

    assert "alias 'child_view'" in str(caught.value)
    _assert_child_view_uses_frame(caught.value)


def test_custom_child_resolver_descriptor_failure_is_wrapped_at_the_uses_boundary():
    class BrokenResolver:
        @property
        def resolve(self):
            raise RuntimeError("resolver getter exploded")

    child = replace(load_flow(_CHILD), origin_resolver=BrokenResolver())

    with pytest.raises(LoadError, match="resolver getter exploded") as caught:
        load_flow(
            _ONE_ALIAS_PARENT,
            child_resolver=lambda _path, _version=None: child,
        )

    assert "alias 'child_view'" in str(caught.value)
    _assert_child_view_uses_frame(caught.value)


def test_custom_child_origin_hook_failure_is_wrapped_at_the_uses_boundary():
    class BrokenOriginNode(Node):
        """Node whose custom provenance hook fails at the injected boundary."""

        def iter_local_origins(self, *, _seen_carriers=None):
            """Raise the sentinel failure before yielding local origins."""
            raise RuntimeError("origin hook exploded")

    child = deepcopy(load_flow(_CHILD))
    child.compiled.nodes["worker"] = BrokenOriginNode("worker")

    with pytest.raises(LoadError, match="origin hook exploded") as caught:
        load_flow(
            _ONE_ALIAS_PARENT,
            child_resolver=lambda _path, _version=None: child,
        )

    assert "alias 'child_view'" in str(caught.value)
    _assert_child_view_uses_frame(caught.value)


def test_custom_child_validation_preserves_exact_origin_subclasses():
    class ChildOrigin(OriginRef):
        pass

    canonical = load_flow(_CHILD)
    assert canonical.origin_resolver is not None
    compiled = rewrite_flow_origins(
        canonical.compiled,
        lambda origin: ChildOrigin(origin.key),
    )

    class ExactResolver:
        def resolve(self, origin):
            if type(origin) is not ChildOrigin:
                return None
            return canonical.origin_resolver.resolve(OriginRef(origin.key))

    child = replace(
        canonical,
        compiled=compiled,
        origin_resolver=ExactResolver(),
    )
    loaded = load_flow(
        _PARENT,
        child_resolver=lambda _path, _version=None: child,
    )

    assert loaded.origin_resolver is not None
    assert all(
        loaded.origin_resolver.resolve(origin) is not None
        for origin in iter_flow_origins(loaded.compiled)
    )


def test_external_resolver_must_return_a_composed_flow():
    with pytest.raises(LoadError, match="expected ComposedFlow") as caught:
        load_flow(
            _ONE_ALIAS_PARENT,
            child_resolver=lambda _path, _version=None: object(),
        )

    assert "alias 'child_view'" in str(caught.value)
    _assert_child_view_uses_frame(caught.value)


def test_source_free_custom_child_rejects_an_origin_resolver():
    child = replace(
        load_flow("id: empty-child\nname: empty-child\nnodes: {}\n"),
        origin_resolver=ResolverTable(),
    )

    with pytest.raises(LoadError, match="source-free child") as caught:
        load_flow(
            _ONE_ALIAS_PARENT,
            child_resolver=lambda _path, _version=None: child,
        )

    assert "alias 'child_view'" in str(caught.value)
    _assert_child_view_uses_frame(caught.value)


def test_source_free_custom_child_remains_source_free_when_embedded():
    child = load_flow("id: empty-child\nname: empty-child\nnodes: {}\n")
    parent = """id: empty-parent
name: empty-parent
nodes:
  invoke:
    kind: call
    call: child
uses:
  child: empty-child
"""

    loaded = load_flow(
        parent,
        child_resolver=lambda _path, _version=None: child,
    )
    embedded = loaded.compiled.nodes["invoke"].child

    assert _origins(embedded) == {}
    assert isinstance(loaded.origin_resolver, ResolverTable)
    assert {
        origin.key for origin in loaded.origin_resolver.origins
    } == set(_origins(loaded.compiled))


def test_resolver_metadata_is_immutable_copy_reused_and_equality_neutral():
    loaded = load_flow(_CHILD)
    copied = deepcopy(loaded)
    replaced = replace(loaded, origin_resolver=ResolverTable())

    assert copied.origin_resolver is loaded.origin_resolver
    assert copied.compiled is not loaded.compiled
    assert replaced == loaded


def test_finalization_rejects_missing_and_overlapping_routes():
    loaded = load_flow(_CHILD)
    staged = replace(loaded, origin_resolver=None)
    assert isinstance(loaded.origin_resolver, ResolverTable)

    with pytest.raises(LoadError, match="no resolver route"):
        _finalize_composed_flow(staged, SourceCatalog(), ())
    with pytest.raises(LoadError, match="multiple resolver routes"):
        _finalize_composed_flow(
            staged,
            SourceCatalog(),
            (loaded.origin_resolver, loaded.origin_resolver),
        )
