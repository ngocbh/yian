"""Immutable YAML source ownership for the Compose front end.

This package owns one PyYAML construction pass per physical document, the
source-navigation contracts built over its retained YAML graph, and the
immutable catalog that resolves Compose-owned opaque origins by replaying
physical occurrences. Isolated location-free literals may be parsed separately
without creating a document. Parser-specific objects stay private so later
compose phases consume stable document, cursor, and origin-resolver interfaces.

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
