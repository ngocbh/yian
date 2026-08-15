"""Immutable YAML source ownership for the Compose front end.

This package owns one PyYAML construction pass per physical document, the
source-navigation contracts built over its retained YAML graph, and the
immutable document-local catalogs that resolve Compose-owned opaque origins by
replaying physical occurrences. Later Compose phases may populate the
package-private catalog builder and consume its frozen snapshots, but parser
objects stay private. Isolated location-free literals may be parsed separately
without creating a document.

Knows about:   PyYAML and the source-neutral contracts in ``diagnostics``;
               ``SourceRange`` is temporarily re-exported through ``events``.
Never imports: compose helpers or phases, nodes, compile, runtime, the run
               boundary, CLI modules, or provider SDKs.
"""

from agent_composer.compose.source.common import (
    SourceAccessError,
    SourceSyntaxError,
)
from agent_composer.compose.source.catalog import SourceCatalog
from agent_composer.compose.source.cursor import SourceCursor
from agent_composer.compose.source.document import SourceDocument
from agent_composer.compose.source.literals import parse_yaml_literal

__all__ = [
    "SourceAccessError",
    "SourceCatalog",
    "SourceCursor",
    "SourceDocument",
    "SourceSyntaxError",
    "parse_yaml_literal",
]
