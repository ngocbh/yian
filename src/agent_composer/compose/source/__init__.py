"""Immutable source documents for the Compose-YAML front end.

This package owns one PyYAML construction pass per physical document and the
source-navigation contracts built over its retained YAML graph. Isolated
location-free literals may be parsed separately without creating a document.
Parser-specific objects stay private so later compose phases consume stable
document and cursor interfaces.

Knows about:   PyYAML and the leaf ``events.SourceRange`` contract.
Never imports: compose helpers or phases, nodes, compile, runtime, the run
               boundary, CLI modules, or provider SDKs.
"""

from agent_composer.compose.source.common import (
    SourceAccessError,
    SourceSyntaxError,
)
from agent_composer.compose.source.cursor import SourceCursor
from agent_composer.compose.source.document import SourceDocument
from agent_composer.compose.source.literals import parse_yaml_literal

__all__ = [
    "SourceAccessError",
    "SourceCursor",
    "SourceDocument",
    "SourceSyntaxError",
    "parse_yaml_literal",
]
