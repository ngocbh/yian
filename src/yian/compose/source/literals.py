"""Parse isolated YAML literals outside source-document ownership."""

from typing import Any

import yaml

from yian.compose.source.common import SourceSyntaxError
from yian.compose.source.scalars import _MERGE_TAG, _FlowLoader


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


def read_plain_scalar(text: str) -> Any:
    """Read one plain scalar's text with the flow loader's scalar rules.

    The text is one stripped, single-line plain scalar, never YAML structure:
    the flow loader's implicit resolver picks its tag and the loader's
    constructors build the value, so ``1e3`` is a float and ``~`` is null as in a
    source document, while ``[1]``, ``a: b``, or ``&a`` stay text.

    Args:
        text (`str`):
            The scalar's text.

    Returns:
        `Any`:
            The constructed value, or ``text`` itself when it spans lines or
            is ``<<``.

    Raises:
        `SourceSyntaxError`:
            Raised when the resolved tag cannot construct the text, such as an
            impossible date.
    """

    if "\n" in text or "\r" in text:
        return text  # one line only: the resolver's `$` would also match before "\n"
    loader = _FlowLoader("")
    tag = loader.resolve(yaml.ScalarNode, text, (True, False))
    if tag == _MERGE_TAG:
        return text  # `<<` merges only as a mapping key; as a value it is text
    try:
        return loader.construct_object(yaml.ScalarNode(tag, text))
    except (yaml.YAMLError, ValueError) as exc:
        raise SourceSyntaxError(str(exc)) from exc
