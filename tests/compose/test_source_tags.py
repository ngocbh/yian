"""Tests for local YAML tags kept by the source layer."""

import copy
import inspect
import re
import sys

import pytest

from yian.compose.loader import _single_node_def_cursor
from yian.compose.parse.file import flow_nodes_cursor
from yian.compose.parse.nodes import normalize_node_signature_cursor
from yian.compose.source import SourceCursor, SourceDocument, SourceSyntaxError, TaggedNode
from yian.defaults import current
from yian.diagnostics import SourceRange


def cur(text, *path):
    c = SourceDocument.parse(text).root
    for p in path:
        c = c.child(p)
    return c


def test_tagged_scalar_value_follows_the_flow_yaml_rules():
    c = cur("a: !int 5\n", "a")
    assert c.value == 5 and c.tag == "int" and c.has_value


def test_tagged_plain_scalars_resolve_like_untagged_ones():
    tagged = cur(
        "a: !bool yes\nb: !float 1e3\nc: !str 1.5\nd: !int 0x10\ne: !x ~\nf: !bool true\n"
    )
    untagged = cur("a: yes\nb: 1e3\nc: 1.5\nd: 0x10\ne: ~\nf: true\n")
    assert tagged.value == untagged.value
    assert tagged.value == {"a": "yes", "b": 1000.0, "c": 1.5, "d": 16, "e": None, "f": True}


def test_tagged_empty_scalar_has_no_value():
    c = cur("a: !int\n", "a")
    assert c.tag == "int" and c.value is None and not c.has_value


def test_tagged_null_has_a_value():
    c = cur("a: !Optional(str) null\n", "a")
    assert c.value is None and c.has_value


def test_tagged_empty_quoted_scalar_has_a_value():
    c = cur("a: !str ''\n", "a")
    assert c.value == "" and c.tag == "str" and c.has_value


def test_quoted_tagged_scalar_stays_text():
    assert cur("a: !int '5'\n", "a").value == "5"
    assert cur('a: !int "5"\n', "a").value == "5"


def test_tagged_block_scalar_stays_text():
    c = cur("a: !str |\n  5\n", "a")
    assert c.value == "5\n" and c.tag == "str"


def test_tagged_mapping_and_sequence():
    m = cur("d: !Buy\n  conviction: 1\n", "d")
    assert m.tag == "Buy" and m.value == {"conviction": 1}
    s = cur("h: !list(message) []\n", "h")
    assert s.tag == "list(message)" and s.value == []


def test_template_scalar_with_tag():
    assert cur("q: !int ${size.output.qty}\n", "q").value == "${size.output.qty}"


def test_untagged_and_core_tags_report_none():
    assert cur("a: 5\n", "a").tag is None
    assert cur("a: 5\n", "a").has_value
    c = cur("a: !!str 5\n", "a")
    assert c.tag is None and c.value == "5"
    assert cur("a:\n", "a").has_value


def test_untagged_containers_report_none():
    root = cur("m: {x: 1}\ns: [1]\n")
    assert root.tag is None
    assert root.child("m").tag is None and root.child("s").tag is None


def test_verbatim_local_tag_is_a_local_tag():
    c = cur("a: !<!int> 5\n", "a")
    assert c.tag == "int" and c.value == 5


def test_tagged_key_constructs_like_untagged():
    root = cur("!int 1: one\n!x <<: two\n")
    assert root.value == {1: "one", "<<": "two"}


def test_tag_survives_anchor_alias_and_merge():
    assert cur("base: &b {x: !int 1}\nuse:\n  <<: *b\n", "use", "x").tag == "int"


_MERGE_SOURCE_TAG = "a tag on a `<<:` merge source checks nothing"


@pytest.mark.parametrize(
    ("text", "where"),
    [
        ("use: {<<: !Foo {x: 1}}\n", SourceRange(1, 11, 1, 22)),
        ("a: &a {x: 1}\nuse:\n  <<: [!A {y: 1}, *a]\n", SourceRange(3, 8, 3, 17)),
        ("a: &a {x: 1}\nuse:\n  <<: !L [*a]\n", SourceRange(3, 7, 3, 14)),
        ("m:\n  <<: &b !Foo {x: 1}\nn: *b\n", SourceRange(2, 7, 2, 21)),
        ("m:\n  <<: [&a !A {x: 1}]\nn: *a\n", SourceRange(2, 8, 2, 20)),
    ],
    ids=["inline", "in-list", "on-list", "anchor-then-alias", "anchor-in-list"],
)
def test_tag_on_a_merge_source_is_an_error(text, where):
    with pytest.raises(SourceSyntaxError, match=re.escape(_MERGE_SOURCE_TAG)) as info:
        SourceDocument.parse(text)
    assert info.value.range == where
    assert "tag the mapping that merges it" in str(info.value)


@pytest.mark.parametrize(
    "text",
    [
        "buy1: &b !Buy {conviction: 1}\nbuy2: !Buy {<<: *b, qty: 2}\n",
        "n: &b !Foo {x: 1}\nm:\n  <<: *b\n",
        "a: &a !A {x: 1}\nuse: {<<: [*a]}\n",
        "x-base: &b !Buy {c: 1}\nn: {<<: *b}\n",
        "a: &a {x: 1}\nl: &l !L [*a]\nuse: {<<: *l}\n",
    ],
    ids=["merge-into-tagged", "alias", "alias-in-list", "x-key-anchor", "aliased-list"],
)
def test_tagged_anchor_merged_through_an_alias_keeps_its_tag(text):
    document = SourceDocument.parse(text)
    for node in document.tagged_nodes():
        assert document._resolve_occurrence(node.path).tag == node.tag
    assert document.tagged_nodes()


def test_merge_into_a_tagged_mapping_reads_both_tags():
    root = cur("buy1: &b !Buy {conviction: 1}\nbuy2: !Buy {<<: *b, qty: 2}\n")
    assert root.child("buy2").value == {"conviction": 1, "qty": 2}
    assert root.child("buy1").tag == "Buy" and root.child("buy2").tag == "Buy"


def test_unknown_global_tag_on_a_merge_source_is_an_error():
    with pytest.raises(SourceSyntaxError, match="unknown YAML tag") as info:
        SourceDocument.parse("use: {<<: !<tag:example.com:x> {x: 1}}\n")
    assert info.value.range == SourceRange(1, 11, 1, 38)


def test_core_tag_on_a_merge_source_keeps_its_yaml_meaning():
    assert cur("use: {<<: !!map {x: 1}}\n", "use").value == {"x": 1}


def test_tag_on_anchor_is_shared_by_aliases():
    root = cur("a: &v !int 1\nb: *v\n")
    assert root.child("a").tag == "int" and root.child("b").tag == "int"


def test_space_ends_the_tag():
    c = cur("k: !Union(A, B) x\n", "k")
    assert c.tag == "Union(A," and c.value == "B) x"


def test_tag_is_value_gated_through_projections():
    c = cur("a: !int 5\n", "a")
    assert c.with_value(5).tag == "int"
    assert c.with_value("${input.a}").tag is None and c.with_value("${input.a}").has_value


def test_tag_is_value_gated_by_type():
    c = cur("a: !int 1\n", "a")
    assert c.with_value(True).tag is None
    m = cur("m: !Buy {x: 1}\n", "m")
    assert m.with_value({"x": True}).tag is None
    assert m.with_value({"x": 1}).tag == "Buy"


def test_changed_empty_tagged_scalar_has_a_value():
    c = cur("a: !int\n", "a")
    assert c.with_value(None).tag == "int" and not c.with_value(None).has_value
    assert c.with_value(3).tag is None and c.with_value(3).has_value


def test_virtual_cursor_has_no_tag_and_has_a_value():
    from yian.compose.source import SourceCursor

    v = SourceCursor.virtual(None)
    assert v.tag is None and v.has_value


def test_bare_tag_is_an_error():
    with pytest.raises(SourceSyntaxError, match="names a type") as info:
        SourceDocument.parse("a: ! x\n")
    assert info.value.range == SourceRange(1, 4, 1, 7)


def test_bare_tag_on_a_collection_is_an_error():
    with pytest.raises(SourceSyntaxError, match="names a type"):
        SourceDocument.parse("a: ! [1]\n")


def test_unknown_global_tag_is_an_error():
    with pytest.raises(SourceSyntaxError, match="unknown YAML tag") as info:
        SourceDocument.parse("%TAG ! tag:example.com:\n---\na: !int 5\n")
    assert info.value.range is not None and info.value.range.start_line == 3


def test_flow_context_tag_hint():
    with pytest.raises(SourceSyntaxError, match=re.escape("space before `,`, `}` or `]`")):
        SourceDocument.parse("m: {a: !int, b: !str}\n")


def test_bare_tag_check_does_not_shrink_nesting_depth():
    # Composing costs about two frames per nesting level. A check inside the recursive
    # compose step would make it three; with room for only two, a document at the
    # nesting limit would not parse.
    limit = current().max_yaml_nesting
    depth = limit - 1  # the root mapping is one level
    text = "a: " + "[" * depth + "1" + "]" * depth + "\n"
    previous = sys.getrecursionlimit()
    sys.setrecursionlimit(len(inspect.stack(0)) + 2 * limit + 100)
    try:
        value = SourceDocument.parse(text).root.child("a").value
    finally:
        sys.setrecursionlimit(previous)
    for _ in range(depth - 1):
        (value,) = value
    assert value == [1]


def test_unrelated_yaml_error_has_no_tag_hint():
    with pytest.raises(SourceSyntaxError) as info:
        SourceDocument.parse("m: {a: 1, b: c: d}\n")
    assert "space before" not in str(info.value)


def test_document_lists_tagged_nodes():
    d = SourceDocument.parse("a: !int 1\nb: [!str x]\n")
    assert sorted(t.tag for t in d.tagged_nodes()) == ["int", "str"]


def test_tagged_nodes_carry_range_and_path_in_document_order():
    d = SourceDocument.parse("b: [!str x]\na: !int 1\n")
    nodes = d.tagged_nodes()
    assert all(isinstance(t, TaggedNode) for t in nodes)
    assert [t.tag for t in nodes] == ["str", "int"]
    assert nodes[0].range == SourceRange(1, 5, 1, 11)
    assert d._resolve_occurrence(nodes[0].path).tag == "str"
    assert d._resolve_occurrence(nodes[1].path).value == 1


def test_tagged_nodes_list_an_aliased_node_once_at_its_anchor():
    d = SourceDocument.parse("base: &b {x: !int 1}\nuse:\n  <<: *b\n  y: *b\n")
    (only,) = d.tagged_nodes()
    assert only.tag == "int"
    assert d._resolve_occurrence(only.path).path == only.path
    assert only.path[0].constructed_key == "base"


def test_untagged_document_lists_no_tagged_nodes():
    assert SourceDocument.parse("a: 1\nb: [x]\n").tagged_nodes() == ()
    assert SourceDocument.parse("").tagged_nodes() == ()


def test_parse_leaves_the_composed_node_tags_unchanged():
    d = SourceDocument.parse("a: !int 5\nm: !Buy {x: 1}\n")
    tags = [value.tag for _key, value in d._root_node.value]
    assert tags == ["!int", "!Buy"]


_NODE_FLOW = """\
nodes:
  n:
    kind: code
    input:
      d: !Buy
        conviction: 1
      h: !list(str) []
"""


@pytest.mark.parametrize(("name", "tag"), [("d", "Buy"), ("h", "list(str)")])
def test_tagged_node_inputs_keep_tags_through_signature_projection(name, tag):
    nodes = SourceDocument.parse(_NODE_FLOW).root.child("nodes")
    entry = normalize_node_signature_cursor(nodes.child("n")).child("inputs").child(name)
    assert entry.tag == tag
    assert entry.with_value(copy.deepcopy(entry.value)).tag == tag
    assert entry.with_value({"other": 2}).tag is None


def test_single_node_def_generated_binding_has_no_tag():
    document = SourceDocument.parse("f:\n  kind: code\n  input:\n    p: !int\n")
    def_cursor = document.root.child("f")
    body = {"kind": "code", "inputs": {"p": None}}
    nodes = _single_node_def_cursor("f", body, def_cursor)
    binding = nodes.child("f").child("input").child("p")
    assert binding.value == "${input.p}"
    assert binding.tag is None and binding.has_value


def test_compact_flow_generated_binding_has_no_tag():
    document = SourceDocument.parse("id: f\nkind: code\ninput:\n  p: !int\n")
    section = {"f": {"kind": "code", "inputs": {"p": "${input.p}"}}}
    binding = flow_nodes_cursor(document, section).child("f").child("inputs").child("p")
    assert binding.value == "${input.p}"
    assert binding.tag is None and binding.has_value


def test_tagged_nodes_walk_a_shared_merge_rope_once(monkeypatch):
    from yian.compose.source import SourceCursor

    keys = 200
    users = 200
    base = "".join(f"  k{i}: !int {i}\n" for i in range(keys))
    uses = "".join(f"u{j}:\n  <<: *b\n" for j in range(users))
    document = SourceDocument.parse(f"b: &b\n{base}{uses}")
    calls = 0
    descend = SourceCursor._descend_mapping_occurrence

    def counting(self, occurrence):
        nonlocal calls
        calls += 1
        return descend(self, occurrence)

    monkeypatch.setattr(SourceCursor, "_descend_mapping_occurrence", counting)
    assert len(document.tagged_nodes()) == keys
    # One step per authored pair: the shared rope is walked once, not per user.
    assert calls <= keys + 2 * users + 1


def test_tag_definition_id_matches_the_tagged_node():
    d = SourceDocument.parse("a: !int 1\nb: [!str x, &s !int 2, *s]\nc: 3\n")
    ids = [node.definition_id for node in d.tagged_nodes()]
    b = d.root.child("b")
    cursors = [d.root.child("a"), b.child(0), b.child(1)]
    assert [c.tag_definition_id for c in cursors] == ids
    assert b.child(2).tag_definition_id == b.child(1).tag_definition_id
    assert d.root.child("c").tag_definition_id is None
    assert d.root.child("a").with_value(2).tag_definition_id is None
    assert SourceCursor.virtual(1).tag_definition_id is None


def test_scalar_text_is_the_authored_text_of_an_unchanged_scalar():
    root = SourceDocument.parse("a: !str yes\nb: 30\nc: 'q'\nd: [1]\ne: 2026-01-01\n").root
    assert root.child("a").scalar_text == "yes"
    assert root.child("b").scalar_text == "30"
    assert root.child("c").scalar_text == "q"
    assert root.child("e").scalar_text == "2026-01-01"
    assert root.child("d").scalar_text is None
    assert root.child("b").with_value(31).scalar_text is None
    assert SourceCursor.virtual(30).scalar_text is None


def test_key_text_is_the_authored_key():
    root = SourceDocument.parse("x: {true: 1, 'q': 2, 3: 4}\n").root.child("x")
    assert root.child(True).key_text == "true"
    assert root.child("q").key_text == "q"
    assert root.child(3).key_text == "3"
    assert root.key_text == "x"
    assert SourceDocument.parse("- 1\n").root.child(0).key_text is None
    assert SourceCursor.virtual({"a": 1}).child("a").key_text is None


def test_tagged_nodes_skip_picked_top_level_keys():
    d = SourceDocument.parse(
        "x-a: &k !str hidden\nx-b: !int 1\nkind: *k\nn: {m: !int 2}\n"
    )
    nodes = d.tagged_nodes(skip_top_level=lambda key: key.startswith("x-"))
    assert [node.tag for node in nodes] == ["str", "int"]
    assert nodes[0].path[0].constructed_key == "kind"
    assert len(d.tagged_nodes()) == 3


def test_tagged_nodes_mark_a_tagged_mapping_key():
    d = SourceDocument.parse("a:\n  !int q: 1\n  r: !int 2\n")
    key, value = d.tagged_nodes()
    assert key.is_key and key.path[-1].constructed_key == "q"
    assert not value.is_key


def test_tagged_nodes_spell_their_key_path_as_written():
    d = SourceDocument.parse("n:\n  !str on: 1\n  l: [0, {null: !int 3}]\n")
    key, value = d.tagged_nodes()
    assert key.key_path == ("n", "on")
    assert value.key_path == ("n", "l", 1, "null")
