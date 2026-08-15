"""Tests for immutable, occurrence-backed Compose source catalogs."""

from copy import deepcopy

import pytest

from agent_composer.compose.source import (
    SourceCatalog,
    SourceCursor,
    SourceDocument,
)
from agent_composer.compose.source import document as source_document_module
from agent_composer.compose.source.catalog import _SourceCatalogBuilder
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
    catalog = builder.freeze()

    assert catalog.resolve(origin) is not None
    assert catalog.resolve(origin) is not None
    assert parse_count == 1
