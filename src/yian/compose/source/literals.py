"""Parse isolated YAML literals outside source-document ownership."""

from typing import Any

import yaml

from yian.compose.source.common import SourceSyntaxError


def parse_yaml_literal(text: str) -> Any:
    """Safely parse one location-free YAML literal.

    This intentionally uses PyYAML's ordinary safe loader rather than the
    source-document loader: input defaults need a value, not a retained graph.
    """

    try:
        return yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise SourceSyntaxError(str(exc)) from exc
