"""Immutable YAML source ownership for the Compose front end.

This package owns one PyYAML construction pass per physical document, the
source-navigation contracts built over its retained YAML graph, and the
immutable document-local catalogs that resolve Compose-owned opaque origins by
replaying physical occurrences. Unchanged source-backed string cursors expose
whole-scalar provenance and exact constructed-line ranges for literal blocks;
catalog sites retain semantic line indices rather than coordinates. The catalog
module also owns collision-free Compose key encoding: document-local semantic paths and
package-private lexical ``uses:`` scoping that treats a child key as one opaque
segment. Exact catalog membership reads only the frozen key table. Later Compose
phases populate the package-private catalog builder; a package-private lookup may
resolve an already-registered site for a load failure without copying or freezing
the table. The loader alone consumes the final frozen snapshot, and parser objects
stay private. Isolated location-free literals may be parsed separately without
creating a document, and one plain scalar's text may be read on its own without
parsing YAML structure. Plain scalars read with YAML 1.2 core-schema rules plus
unquoted timestamps and ``<<`` merge keys, from one loader table shared by
documents and isolated literals.

Knows about:   PyYAML and the source-neutral contracts in ``diagnostics``;
               Compose imports ``SourceRange`` directly from that leaf.
Never imports: compose helpers or phases, nodes, compile, runtime, the run
               boundary, CLI modules, or provider SDKs.
"""

from yian.compose.source.common import (
    ScalarSource,
    SourceAccessError,
    SourceSyntaxError,
    TaggedNode,
)
from yian.compose.source.catalog import SourceCatalog
from yian.compose.source.cursor import SourceCursor
from yian.compose.source.document import SourceDocument
from yian.compose.source.literals import parse_yaml_literal, read_plain_scalar

__all__ = [
    "SourceAccessError",
    "SourceCatalog",
    "SourceCursor",
    "SourceDocument",
    "ScalarSource",
    "SourceSyntaxError",
    "TaggedNode",
    "parse_yaml_literal",
    "read_plain_scalar",
]
