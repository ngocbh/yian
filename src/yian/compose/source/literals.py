"""Parse isolated YAML literals outside source-document ownership."""

from typing import Any

import yaml

from yian.compose.source.common import SourceSyntaxError
from yian.compose.source.scalars import _FlowLoader


def parse_yaml_literal(text: str) -> Any:
    """Safely parse one location-free YAML literal.

    Plain scalars read with the flow loader's scalar rules, the same as in a
    source document, but no document graph is retained: input defaults need a
    value, not a graph.

    Args:
        text (`str`):
            The YAML literal.

    Returns:
        `Any`:
            The constructed value.

    Raises:
        `SourceSyntaxError`:
            Raised when the text is not valid YAML for the flow loader.
    """

    try:
        return yaml.load(text, Loader=_FlowLoader)
    except yaml.YAMLError as exc:
        raise SourceSyntaxError(str(exc)) from exc
