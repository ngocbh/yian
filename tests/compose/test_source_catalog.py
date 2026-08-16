"""Tests for immutable, occurrence-backed Compose source catalogs."""

from copy import deepcopy

import pytest

from agent_composer.compose.source import (
    SourceCatalog,
    SourceCursor,
    SourceDocument,
)
from agent_composer.compose.source import document as source_document_module
from agent_composer.compose.source.catalog import (
    _ConstructedLine,
    _SourceCatalogBuilder,
    _SourceSite,
    _scope_origin,
)
from agent_composer.diagnostics import DiagnosticFrame, OriginRef, SourceRange


def test_site_keys_are_typed_escaped_and_deterministic():
    cursor = SourceDocument.parse("value: present\n").root.child("value")
    paths = (
        ("nodes", 1, "input"),
        ("nodes", "1", "input"),
        ("nodes", "a/b"),
        ("nodes", "a~1b"),
        ("nodes", "a", "b"),
    )

    first_builder = _SourceCatalogBuilder()
    first = [first_builder.register(path, cursor) for path in paths]
    second_builder = _SourceCatalogBuilder()
    second = [second_builder.register(path, cursor) for path in paths]

    assert first == second
    assert all(isinstance(origin, OriginRef) for origin in first)
    assert all(origin.key for origin in first if origin is not None)
    assert len({origin.key for origin in first if origin is not None}) == len(paths)


def test_scoped_origin_keys_are_domain_separated_injective_and_deterministic():
    aliases = ("child", "a/b", "a~1b", "@scope", "子")
    old_keys = ("local", "/s:root/s:node", "a/b", "a~1b", "@scope/s:uses")

    first = [
        _scope_origin(alias, OriginRef(old_key))
        for alias in aliases
        for old_key in old_keys
    ]
    second = [
        _scope_origin(alias, OriginRef(old_key))
        for alias in aliases
        for old_key in old_keys
    ]

    assert first == second
    assert len({origin.key for origin in first}) == len(aliases) * len(old_keys)
    assert all(origin.key.startswith("@scope/") for origin in first)
    assert all(not origin.key.startswith("/") for origin in first)


def test_nested_scoped_origin_treats_the_previous_key_as_one_opaque_segment():
    inner = _scope_origin("inner/alias", OriginRef("old/~key"))
    nested = _scope_origin("outer~alias", inner)

    assert nested == _scope_origin("outer~alias", OriginRef(inner.key))
    assert nested != _scope_origin("outer/inner", OriginRef("old/~key"))
    assert nested != _scope_origin("outer~alias", OriginRef("old/~key"))


def test_scoped_origin_escaping_prevents_segment_boundary_spoofing():
    assert _scope_origin(
        "a/s:origin/s:b",
        OriginRef("c"),
    ) != _scope_origin(
        "a",
        OriginRef("b/s:origin/s:c"),
    )


@pytest.mark.parametrize(
    ("alias", "origin", "error"),
    [
        ("", OriginRef("old"), ValueError),
        (None, OriginRef("old"), TypeError),
        (type("Alias", (str,), {})("child"), OriginRef("old"), TypeError),
        ("child", "old", TypeError),
    ],
)
def test_scoped_origin_rejects_invalid_aliases_and_origins(alias, origin, error):
    with pytest.raises(error):
        _scope_origin(alias, origin)


def test_scoped_origin_revalidates_origin_subclass_keys():
    class InvalidOrigin(OriginRef):
        def __post_init__(self) -> None:
            pass

    with pytest.raises(ValueError, match="origin key must be a non-empty string"):
        _scope_origin("child", InvalidOrigin(""))


@pytest.mark.parametrize(
    "segment",
    [
        True,
        1.5,
        type("StringSubclass", (str,), {})("node"),
        type("IntegerSubclass", (int,), {})(1),
    ],
)
def test_site_paths_reject_ambiguous_segment_types(segment):
    cursor = SourceDocument.parse("value: present\n").root.child("value")

    with pytest.raises(TypeError):
        _SourceCatalogBuilder().register((segment,), cursor)


def test_site_paths_must_be_nonempty_tuples():
    cursor = SourceDocument.parse("value: present\n").root.child("value")

    with pytest.raises(TypeError):
        _SourceCatalogBuilder().register(["value"], cursor)
    with pytest.raises(ValueError):
        _SourceCatalogBuilder().register((), cursor)


def test_duplicate_site_registration_is_rejected():
    root = SourceDocument.parse(
        "first: one\n"
        "second: two\n"
    ).root
    builder = _SourceCatalogBuilder()
    builder.register(("nodes", "worker", "input"), root.child("first"))

    with pytest.raises(ValueError):
        builder.register(("nodes", "worker", "input"), root.child("second"))
    with pytest.raises(ValueError):
        builder.register(
            ("nodes", "worker", "input"),
            SourceCursor.virtual("source-free"),
        )


def test_invalid_resolver_and_registration_inputs_are_rejected():
    cursor = SourceDocument.parse("value: present\n").root.child("value")
    builder = _SourceCatalogBuilder()

    with pytest.raises(TypeError):
        builder.register(("value",), "not a cursor")
    with pytest.raises(TypeError):
        builder.register(("value",), cursor, anchor="line")
    with pytest.raises(TypeError):
        builder.freeze().resolve("not an origin")


def test_value_and_key_anchors_resolve_complete_frames():
    text = "node:\n  field: value\n"
    document = SourceDocument.parse(text, label="flow.yaml")
    cursor = document.root.child("node").child("field")
    builder = _SourceCatalogBuilder()

    value_origin = builder.register(("nodes", "node", "field", "value"), cursor)
    key_origin = builder.register(
        ("nodes", "node", "field", "key"),
        cursor,
        anchor="key",
    )
    catalog = builder.freeze()

    assert isinstance(catalog, SourceCatalog)
    assert catalog.resolve(value_origin) == DiagnosticFrame(
        label="flow.yaml",
        text=text,
        range=SourceRange(2, 10, 2, 15),
    )
    assert catalog.resolve(key_origin) == DiagnosticFrame(
        label="flow.yaml",
        text=text,
        range=SourceRange(2, 3, 2, 8),
    )
    assert catalog.resolve(OriginRef("unknown")) is None


def test_duplicate_pairs_and_aliases_replay_exact_occurrences():
    text = (
        "name: first\n"
        "name: final\n"
        "shared: &shared {value: 1}\n"
        "left: *shared\n"
        "right: *shared\n"
    )
    root = SourceDocument.parse(text, label="aliases.yaml").root
    first_name, final_name = root.mapping_occurrences("name")
    left = root.child("left")
    right = root.child("right")
    builder = _SourceCatalogBuilder()

    origins = {
        "first_name": builder.register(("names", 0), first_name),
        "final_name": builder.register(("names", 1), final_name),
        "left_value": builder.register(("aliases", "left", "value"), left),
        "right_value": builder.register(("aliases", "right", "value"), right),
        "left_key": builder.register(
            ("aliases", "left", "key"), left, anchor="key"
        ),
        "right_key": builder.register(
            ("aliases", "right", "key"), right, anchor="key"
        ),
    }
    catalog = builder.freeze()

    assert catalog.resolve(origins["first_name"]).range == SourceRange(
        1, 7, 1, 12
    )
    assert catalog.resolve(origins["final_name"]).range == SourceRange(
        2, 7, 2, 12
    )
    assert catalog.resolve(origins["left_value"]).range == SourceRange(
        3, 9, 3, 27
    )
    assert catalog.resolve(origins["right_value"]).range == SourceRange(
        3, 9, 3, 27
    )
    assert catalog.resolve(origins["left_key"]).range == SourceRange(4, 1, 4, 5)
    assert catalog.resolve(origins["right_key"]).range == SourceRange(5, 1, 5, 6)


def test_projected_and_renamed_cursors_keep_physical_backing():
    renamed_text = "original: old\n"
    renamed_document = SourceDocument.parse(renamed_text, label="renamed.yaml")
    renamed = renamed_document.root.renamed({"original": "displayed"}).child(
        "displayed"
    )

    backing_text = "backing: old\n"
    backing_document = SourceDocument.parse(backing_text, label="backing.yaml")
    host_document = SourceDocument.parse("host: local\n", label="host.yaml")
    projected = host_document.root.project_mapping(
        {
            "displayed": (
                "replacement",
                backing_document.root.child("backing"),
            )
        }
    ).child("displayed")

    builder = _SourceCatalogBuilder()
    renamed_origin = builder.register(("renamed", "value"), renamed)
    projected_value_origin = builder.register(("projected", "value"), projected)
    projected_key_origin = builder.register(
        ("projected", "key"), projected, anchor="key"
    )
    catalog = builder.freeze()

    assert catalog.resolve(renamed_origin) == DiagnosticFrame(
        label="renamed.yaml",
        text=renamed_text,
        range=renamed_document.root.child("original").range,
    )
    assert catalog.resolve(projected_value_origin) == DiagnosticFrame(
        label="backing.yaml",
        text=backing_text,
        range=backing_document.root.child("backing").range,
    )
    assert catalog.resolve(projected_key_origin) == DiagnosticFrame(
        label="backing.yaml",
        text=backing_text,
        range=backing_document.root.child("backing").key_range,
    )


def test_source_free_and_unanchored_sites_are_ignored():
    document = SourceDocument.parse("items:\n  - first\n")
    builder = _SourceCatalogBuilder()

    assert builder.register(("virtual",), SourceCursor.virtual("value")) is None
    assert builder.register(
        ("root-key",), document.root, anchor="key"
    ) is None
    assert builder.register(
        ("sequence-key",), document.root.child("items").child(0), anchor="key"
    ) is None
    assert builder.register(
        ("empty-value",), SourceDocument.parse("").root
    ) is None
    reused_origin = builder.register(
        ("virtual",), document.root.child("items").child(0)
    )

    catalog = builder.freeze()
    assert catalog.resolve(reused_origin) is not None
    assert catalog.resolve(OriginRef("not-registered")) is None


def test_freeze_isolates_snapshots_and_deepcopy_reuses_catalog():
    root = SourceDocument.parse(
        "first: one\n"
        "second: two\n"
    ).root
    builder = _SourceCatalogBuilder()
    first_origin = builder.register(("first",), root.child("first"))
    first_snapshot = builder.freeze()

    second_origin = builder.register(("second",), root.child("second"))
    second_snapshot = builder.freeze()

    assert first_snapshot.resolve(first_origin) is not None
    assert first_snapshot.resolve(second_origin) is None
    assert second_snapshot.resolve(first_origin) is not None
    assert second_snapshot.resolve(second_origin) is not None
    assert deepcopy(first_snapshot) is first_snapshot


def test_catalog_membership_is_exact_and_snapshot_isolated(monkeypatch):
    root = SourceDocument.parse(
        "first: one\n"
        "second: two\n"
    ).root
    builder = _SourceCatalogBuilder()
    first_origin = builder.register(("first",), root.child("first"))
    first_snapshot = builder.freeze()
    second_origin = builder.register(("second",), root.child("second"))
    second_snapshot = builder.freeze()

    def unexpected_resolve(*_args, **_kwargs):
        raise AssertionError("membership must not replay a source occurrence")

    monkeypatch.setattr(SourceCatalog, "resolve", unexpected_resolve)

    assert first_origin in first_snapshot
    assert OriginRef(first_origin.key) in first_snapshot
    assert second_origin not in first_snapshot
    assert first_origin in second_snapshot
    assert second_origin in second_snapshot
    assert OriginRef("unknown") not in second_snapshot
    assert "not-an-origin" not in second_snapshot


def test_failed_builder_transaction_releases_registered_site_keys():
    """A retry may reuse every semantic path from a failed construction."""
    cursor = SourceDocument.parse("value: present\n").root.child("value")
    builder = _SourceCatalogBuilder()

    with pytest.raises(RuntimeError, match="construction failed"):
        with builder.transaction():
            builder.register(("value",), cursor)
            raise RuntimeError("construction failed")

    retried = builder.register(("value",), cursor)

    assert builder.freeze().resolve(retried) is not None


def test_resolution_does_not_reparse_source(monkeypatch):
    parse_count = 0
    original_init = source_document_module._DocumentLoader.__init__

    def counting_init(loader, stream):
        nonlocal parse_count
        parse_count += 1
        original_init(loader, stream)

    monkeypatch.setattr(
        source_document_module._DocumentLoader,
        "__init__",
        counting_init,
    )
    document = SourceDocument.parse("value: present\n")
    builder = _SourceCatalogBuilder()
    origin = builder.register(("value",), document.root.child("value"))
    current_frame = builder.resolve_registered(origin)
    catalog = builder.freeze()

    assert current_frame == catalog.resolve(origin)
    assert catalog.resolve(origin) is not None
    assert catalog.resolve(origin) is not None
    assert parse_count == 1


def test_builder_lookup_reads_current_sites_without_freezing(monkeypatch):
    document = SourceDocument.parse(
        "first: one\nsecond: two\n",
        label="flow.yaml",
    )
    builder = _SourceCatalogBuilder()
    first = builder.register(("first",), document.root.child("first"))

    def unexpected_freeze(_builder):
        raise AssertionError("registered-site lookup must not freeze")

    monkeypatch.setattr(_SourceCatalogBuilder, "freeze", unexpected_freeze)

    assert builder.resolve_registered(first).range == SourceRange(1, 8, 1, 11)
    assert builder.resolve_registered(OriginRef("unknown")) is None


def test_constructed_line_literal_registers_zero_based_exact_frames():
    text = "code: |-\r\n  first\r\n\r\n  second\r\n"
    document = SourceDocument.parse(text, label="literal.yaml")
    cursor = document.root.child("code")
    builder = _SourceCatalogBuilder()

    origins = [
        builder.register_constructed_line(("code", "line", index), cursor, index)
        for index in range(3)
    ]
    catalog = builder.freeze()

    assert [catalog.resolve(origin) for origin in origins] == [
        DiagnosticFrame(
            label="literal.yaml",
            text=text,
            range=SourceRange(2, 3, 2, 8),
        ),
        DiagnosticFrame(
            label="literal.yaml",
            text=text,
            range=SourceRange(3, 1, 3, 1),
        ),
        DiagnosticFrame(
            label="literal.yaml",
            text=text,
            range=SourceRange(4, 3, 4, 9),
        ),
    ]


def test_constructed_line_unchanged_projection_retains_literal_provenance():
    text = "code: |-\n  first\n  second\n"
    cursor = SourceDocument.parse(text, label="projected.yaml").root.child("code")
    unchanged = cursor.with_value(cursor.value)
    builder = _SourceCatalogBuilder()

    origin = builder.register_constructed_line(
        ("projected", "code", "line", 1),
        unchanged,
        1,
    )

    assert builder.freeze().resolve(origin) == DiagnosticFrame(
        label="projected.yaml",
        text=text,
        range=SourceRange(3, 3, 3, 9),
    )


@pytest.mark.parametrize(
    ("text", "replacement"),
    [
        ("value: >-\n  first\n  second\n", None),
        ('value: "first\\nsecond"\n', None),
        ("value: first\n  second\n", None),
        (None, "first\nsecond"),
        ("value: |-\n  first\n  second\n", "changed\nvalue"),
    ],
    ids=["folded", "quoted", "plain-multiline", "virtual", "changed"],
)
def test_constructed_line_unavailable_does_not_reserve_site(text, replacement):
    if text is None:
        cursor = SourceCursor.virtual(replacement)
    else:
        cursor = SourceDocument.parse(text).root.child("value")
        if replacement is not None:
            cursor = cursor.with_value(replacement)
    builder = _SourceCatalogBuilder()
    site_path = ("value", "line", 0)

    assert builder.register_constructed_line(site_path, cursor, 0) is None

    whole_origin = builder.register(site_path, cursor)
    if whole_origin is None:
        fallback = SourceDocument.parse("fallback: located\n").root.child("fallback")
        whole_origin = builder.register(site_path, fallback)
    assert builder.freeze().resolve(whole_origin) is not None


@pytest.mark.parametrize("index", [True, "0", 0.0, None])
def test_constructed_line_rejects_non_exact_integer_indices(index):
    cursor = SourceDocument.parse("value: |-\n  first\n  second\n").root.child(
        "value"
    )

    with pytest.raises(TypeError):
        _SourceCatalogBuilder().register_constructed_line(
            ("value", "bad-index"),
            cursor,
            index,
        )


@pytest.mark.parametrize("index", [-1, 2], ids=["negative", "out-of-range"])
def test_constructed_line_rejects_invalid_exact_integer_indices(index):
    cursor = SourceDocument.parse("value: |-\n  first\n  second\n").root.child(
        "value"
    )

    with pytest.raises(ValueError):
        _SourceCatalogBuilder().register_constructed_line(
            ("value", "bad-index"),
            cursor,
            index,
        )


def test_constructed_line_resolution_rejects_a_malformed_negative_selector():
    document = SourceDocument.parse("value: |-\n  first\n  second\n")
    cursor = document.root.child("value")
    origin = OriginRef("malformed")
    catalog = SourceCatalog(
        {
            origin.key: _SourceSite(
                document=document,
                occurrence_path=cursor.path,
                selector=_ConstructedLine(-1),
            )
        }
    )

    assert catalog.resolve(origin) is None


def test_constructed_line_duplicate_detection_crosses_selector_kinds():
    cursor = SourceDocument.parse("value: |-\n  first\n  second\n").root.child(
        "value"
    )
    folded = SourceDocument.parse("value: >-\n  first\n  second\n").root.child(
        "value"
    )
    site_path = ("value", "site")
    line_first = _SourceCatalogBuilder()
    line_first.register_constructed_line(site_path, cursor, 0)

    with pytest.raises(ValueError):
        line_first.register(site_path, cursor)
    with pytest.raises(ValueError):
        line_first.register_constructed_line(site_path, cursor, 1)
    with pytest.raises(ValueError):
        line_first.register_constructed_line(site_path, folded, 0)

    whole_first = _SourceCatalogBuilder()
    whole_first.register(site_path, cursor)
    with pytest.raises(ValueError):
        whole_first.register_constructed_line(site_path, cursor, 0)


def test_constructed_line_transactions_freeze_and_deepcopy_are_stable():
    cursor = SourceDocument.parse("value: |-\n  first\n  second\n").root.child(
        "value"
    )
    builder = _SourceCatalogBuilder()
    first_origin = builder.register_constructed_line(("line", 0), cursor, 0)
    first_snapshot = builder.freeze()
    rolled_back_origin = None

    with pytest.raises(RuntimeError, match="construction failed"):
        with builder.transaction():
            rolled_back_origin = builder.register_constructed_line(
                ("line", 1),
                cursor,
                1,
            )
            raise RuntimeError("construction failed")

    assert first_snapshot.resolve(first_origin) is not None
    assert first_snapshot.resolve(rolled_back_origin) is None

    retried_origin = builder.register_constructed_line(("line", 1), cursor, 1)
    second_snapshot = builder.freeze()

    assert retried_origin == rolled_back_origin
    assert first_snapshot.resolve(retried_origin) is None
    assert second_snapshot.resolve(first_origin) is not None
    assert second_snapshot.resolve(retried_origin) is not None
    assert deepcopy(first_snapshot) is first_snapshot
    assert deepcopy(second_snapshot) is second_snapshot


def test_constructed_line_alias_occurrences_replay_the_shared_definition():
    text = (
        "shared: &code |-\n"
        "  first\n"
        "  second\n"
        "left: *code\n"
        "right: *code\n"
    )
    root = SourceDocument.parse(text, label="aliases.yaml").root
    builder = _SourceCatalogBuilder()

    left = builder.register_constructed_line(
        ("aliases", "left", "line", 1),
        root.child("left"),
        1,
    )
    right = builder.register_constructed_line(
        ("aliases", "right", "line", 0),
        root.child("right"),
        0,
    )
    catalog = builder.freeze()

    assert catalog.resolve(left).range == SourceRange(3, 3, 3, 9)
    assert catalog.resolve(right).range == SourceRange(2, 3, 2, 8)


def test_constructed_line_keys_survive_formatting_while_frames_move():
    first_text = "wrapper:\n  code: |-\n    first\n    second\n"
    second_text = "# shifted\nwrapper:\n    code: |-\n      first\n      second\n"
    first_cursor = SourceDocument.parse(
        first_text,
        label="first.yaml",
    ).root.child("wrapper").child("code")
    second_cursor = SourceDocument.parse(
        second_text,
        label="second.yaml",
    ).root.child("wrapper").child("code")
    site_path = ("nodes", "worker", "code", "line", 1)
    first_builder = _SourceCatalogBuilder()
    second_builder = _SourceCatalogBuilder()

    first_origin = first_builder.register_constructed_line(
        site_path,
        first_cursor,
        1,
    )
    second_origin = second_builder.register_constructed_line(
        site_path,
        second_cursor,
        1,
    )

    assert first_origin == second_origin
    assert first_builder.freeze().resolve(first_origin) == DiagnosticFrame(
        label="first.yaml",
        text=first_text,
        range=SourceRange(4, 5, 4, 11),
    )
    assert second_builder.freeze().resolve(second_origin) == DiagnosticFrame(
        label="second.yaml",
        text=second_text,
        range=SourceRange(5, 7, 5, 13),
    )


def test_constructed_line_resolution_does_not_reparse_source(monkeypatch):
    parse_count = 0
    original_init = source_document_module._DocumentLoader.__init__

    def counting_init(loader, stream):
        nonlocal parse_count
        parse_count += 1
        original_init(loader, stream)

    monkeypatch.setattr(
        source_document_module._DocumentLoader,
        "__init__",
        counting_init,
    )
    document = SourceDocument.parse("value: |-\n  first\n  second\n")
    builder = _SourceCatalogBuilder()
    origin = builder.register_constructed_line(
        ("value", "line", 1),
        document.root.child("value"),
        1,
    )
    catalog = builder.freeze()

    assert catalog.resolve(origin) is not None
    assert catalog.resolve(origin) is not None
    assert parse_count == 1
