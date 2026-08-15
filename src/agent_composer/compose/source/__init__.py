"""Immutable source documents for the Compose-YAML front end.

This package owns the single PyYAML construction pass and the source-navigation
contracts built over its retained YAML graph.  It keeps parser-specific objects
private so later compose phases consume stable document and cursor interfaces.

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

__all__ = [
    "SourceAccessError",
    "SourceCursor",
    "SourceDocument",
    "SourceSyntaxError",
]
