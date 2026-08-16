"""Unit tests for the `FieldRanges` bare-key guard (#101).

`node.field_ranges` is keyed by the location's `(SourceSpan.kind, key)` tuple. A bare-string
lookup can therefore never match — and because every caller falls back (`or node.range`), it
degrades to a caret on the wrong line instead of raising. That silence is what let two dead
lookups survive both review and a fully green suite when #98 moved the map from string keys to
tuples. `FieldRanges.get` refuses the wrong key shape so the mistake is loud at the first call.

The type must also survive the copies: the descriptor stamps ranges at parse, and the compose
builder's `_finish` rebuilds them onto the runtime node — which is where the lookups actually
happen. Copying into a plain `dict` anywhere along that path would silently drop the guard
exactly where it earns its keep, so `test_guard_survives_build` pins the end of that chain.
"""

import pytest

from agent_composer.compose.loader import load_flow
from agent_composer.nodes.base import FieldRanges

_FLOW = """
id: guard
name: guard
input:
  topic: str
nodes:
  n:
    kind: code
    input:
      topic: ${input.topic}
    output: int
    code: "return {'output': 1}"
output: ${n.output}
"""


def test_bare_key_lookup_raises():
    fr = FieldRanges({("field", "over"): "RANGE"})
    with pytest.raises(TypeError, match="keyed by a .kind, key. tuple"):
        fr.get("over")


def test_bare_key_error_names_the_offending_key():
    # The message must hand back a usable fix, not just complain.
    fr = FieldRanges()
    with pytest.raises(TypeError, match=r"\('field', 'over'\)"):
        fr.get("over")


def test_tuple_lookup_still_works():
    fr = FieldRanges({("field", "over"): "RANGE"})
    assert fr.get(("field", "over")) == "RANGE"
    assert fr.get(("field", "absent")) is None
    assert fr.get(("field", "absent"), "fallback") == "fallback"
    assert ("field", "over") in fr
    assert ("field", "absent") not in fr


def test_bare_key_membership_raises():
    # Compatibility range lookups use `(kind, key)` tuples, so a bare key here would silently
    # answer False — the same silent-miss bug `get` guards against.
    fr = FieldRanges({("field", "over"): "RANGE"})
    with pytest.raises(TypeError, match="keyed by a .kind, key. tuple"):
        "over" in fr  # noqa: B015


def test_copy_keeps_the_guard():
    # `dict.copy` returns a PLAIN dict, which would launder the guard away silently.
    fr = FieldRanges({("field", "over"): "RANGE"})
    dup = fr.copy()
    assert isinstance(dup, FieldRanges)
    assert dup.get(("field", "over")) == "RANGE"
    with pytest.raises(TypeError):
        dup.get("over")


def test_guard_survives_build():
    # The end of the stamp -> copy -> lookup chain: a BUILT runtime node must still carry the
    # guard, since `_finish` is where a plain-dict copy would silently drop it.
    node = load_flow(_FLOW).compiled.nodes["n"]
    assert isinstance(node.field_ranges, FieldRanges)
    with pytest.raises(TypeError):
        node.field_ranges.get("inputs")
    assert node.field_ranges.get(("input", "topic")) is not None


def test_guard_survives_build_on_the_start_boundary():
    # The StartNode takes its map through a separate constructor path (`_boundary_field_ranges`
    # -> `StartNode(field_ranges=...)`), so it needs its own pin — it is also the node whose
    # `("input", name)` lookup was one of the two dead sites.
    start = load_flow(_FLOW).compiled.nodes["__start__"]
    assert isinstance(start.field_ranges, FieldRanges)
    assert start.field_ranges.get(("input", "topic")) is not None
