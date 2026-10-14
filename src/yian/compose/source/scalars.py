"""The flow loader's plain-scalar rules: YAML 1.2 core schema plus dates and merges.

PyYAML resolves plain scalars with YAML 1.1 rules, where ``yes``/``no``/``on``/
``off`` are booleans, ``017`` is octal, ``1_000`` is an integer, and ``1:30`` is a
sexagesimal integer. ``_FlowLoader`` keeps PyYAML as the parser and swaps only its
own implicit-resolver table and the ``int``/``float``/``bool`` constructors for the
YAML 1.2 core schema, keeping two YAML 1.1 extensions: unquoted timestamps and
``<<`` merge keys. Its own ``timestamp`` constructor makes a date-shaped plain
scalar that is not a real date or time, such as ``2026-13-45`` or an offset of
``+05:99``, a located ``ConstructorError`` that says to quote it. PyYAML's global ``SafeLoader`` tables are never modified.

Knows about:   PyYAML.
Never imports: compose helpers or phases, nodes, compile, runtime, the run
               boundary, CLI modules, or provider SDKs.
"""

from __future__ import annotations

import re
from datetime import date, datetime

import yaml
from yaml.constructor import ConstructorError

_BOOL_TAG = "tag:yaml.org,2002:bool"
_INT_TAG = "tag:yaml.org,2002:int"
_FLOAT_TAG = "tag:yaml.org,2002:float"
_NULL_TAG = "tag:yaml.org,2002:null"
_MERGE_TAG = "tag:yaml.org,2002:merge"
_TIMESTAMP_TAG = "tag:yaml.org,2002:timestamp"

_BOOL = re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$")
_INT = re.compile(r"^(?:[-+]?[0-9]+|0o[0-7]+|0x[0-9a-fA-F]+)$")
_FLOAT = re.compile(
    r"^(?:[-+]?(?:\.[0-9]+|[0-9]+(?:\.[0-9]*)?)(?:[eE][-+]?[0-9]+)?"
    r"|[-+]?\.(?:inf|Inf|INF)"
    r"|\.(?:nan|NaN|NAN))$"
)
_NULL = re.compile(r"^(?:~|null|Null|NULL|)$")
_MERGE = re.compile(r"^(?:<<)$")


def _pyyaml_timestamp_regex() -> re.Pattern[str]:
    """Return PyYAML's own implicit timestamp pattern.

    Returns:
        `re.Pattern[str]`:
            The pattern PyYAML's resolver uses for unquoted dates.

    Raises:
        `AssertionError`:
            Raised when PyYAML's resolver has no implicit timestamp entry.
    """

    for resolvers in yaml.resolver.Resolver.yaml_implicit_resolvers.values():
        for tag, regexp in resolvers:
            if tag == _TIMESTAMP_TAG:
                return regexp
    raise AssertionError("PyYAML has no implicit timestamp resolver")


class ScalarValueError(ConstructorError):
    """A plain scalar's resolved tag can't build its text; quoting keeps it as text.

    Args:
        reason (`str`):
            Why the text can't be built, without the text itself.
        mark (`yaml.Mark | None`):
            Where the scalar starts.
    """

    def __init__(self, reason: str, mark: yaml.Mark | None):
        super().__init__(None, None, f"{reason}; quote it to keep it as text", mark)
        self.reason = reason


class _FlowLoader(yaml.SafeLoader):
    """SafeLoader that reads plain scalars with YAML 1.2 core rules.

    Unquoted timestamps and ``<<`` merge keys keep their YAML 1.1 meaning.
    Explicit ``!!int``, ``!!float``, and ``!!bool`` tags accept only the forms
    the table resolves. A ``timestamp`` that can't be built is a located
    ``ConstructorError``; every other tag keeps PyYAML's safe behavior.
    """

    # An own dict, so add_implicit_resolver below never copies SafeLoader's table.
    yaml_implicit_resolvers: dict = {}

    def construct_yaml_bool(self, node: yaml.Node) -> bool:
        """Construct a core-schema boolean.

        Args:
            node (`yaml.Node`):
                Scalar node tagged ``bool``.

        Returns:
            `bool`:
                ``True`` or ``False``.

        Raises:
            `ConstructorError`:
                Raised when the text is not a core-schema boolean.
        """

        text = self._core_text(node, _BOOL, "boolean")
        return text.lower() == "true"

    def construct_yaml_int(self, node: yaml.Node) -> int:
        """Construct a core-schema integer; leading zeros stay decimal.

        Args:
            node (`yaml.Node`):
                Scalar node tagged ``int``.

        Returns:
            `int`:
                The integer value.

        Raises:
            `ConstructorError`:
                Raised when the text is not a core-schema integer or has more digits
                than Python converts.
        """

        text = self._core_text(node, _INT, "integer")
        digits, base = text, 10
        if text.startswith("0o"):
            digits, base = text[2:], 8
        elif text.startswith("0x"):
            digits, base = text[2:], 16
        try:
            return int(digits, base)
        except ValueError:
            # Python caps how many digits it converts; keep the error located.
            raise ConstructorError(
                None,
                None,
                f"an integer with {len(digits)} digits is too long",
                node.start_mark,
            ) from None

    def construct_yaml_float(self, node: yaml.Node) -> float:
        """Construct a core-schema float, including ``.inf`` and ``.nan`` forms.

        Args:
            node (`yaml.Node`):
                Scalar node tagged ``float``.

        Returns:
            `float`:
                The float value.

        Raises:
            `ConstructorError`:
                Raised when the text is not a core-schema float.
        """

        text = self._core_text(node, _FLOAT, "float")
        lowered = text.lower()
        if lowered.endswith(".inf"):
            return float("-inf") if text[0] == "-" else float("inf")
        if lowered == ".nan":
            return float("nan")
        return float(text)

    def construct_yaml_timestamp(self, node: yaml.Node) -> date | datetime:
        """Construct a date or datetime, or raise a located error when it can't be built.

        The quoting fix is given for any unquoted text, including a plain
        ``!!timestamp 2026-13-45``, since the node does not record an explicit tag.

        Args:
            node (`yaml.Node`):
                Scalar node tagged ``timestamp``.

        Returns:
            `date | datetime`:
                The date, or the datetime for text with a time.

        Raises:
            `ScalarValueError`:
                Raised when an unquoted date-shaped text is not a real date or time,
                such as ``2026-13-45`` or an offset of ``+05:99``.
            `ConstructorError`:
                Raised when an explicitly tagged text is not date-shaped, or is quoted
                and not a real date or time.
        """

        text = self.construct_scalar(node)
        match = self.timestamp_regexp.match(text)
        if match is None:
            raise ConstructorError(
                None,
                None,
                f"expected a date or timestamp, but found {text!r}",
                node.start_mark,
            )
        try:
            if match.group("tz_hour") is not None:
                # PyYAML adds the offset as a timedelta, so minutes over 59 would roll
                # into the hours; check it as typesys/temporal.py does, hours first.
                if int(match.group("tz_hour")) >= 24:
                    raise ValueError("UTC offset must be under 24:00")
                if int(match.group("tz_minute") or 0) >= 60:
                    raise ValueError("UTC offset minutes must be under 60")
            return super().construct_yaml_timestamp(node)
        except ValueError as exc:
            kind = "date" if match.group("hour") is None else "datetime"
            reason = f"not a valid {kind}: {exc}"
            if node.style is None:
                raise ScalarValueError(reason, node.start_mark) from None
            # Quoted or block text under an explicit tag: quoting can't be the fix.
            raise ConstructorError(None, None, reason, node.start_mark) from None

    def _core_text(self, node: yaml.Node, pattern: re.Pattern[str], kind: str) -> str:
        """Return a scalar's text after checking it is a core-schema form.

        Args:
            node (`yaml.Node`):
                Scalar node with an explicit or resolved tag.
            pattern (`re.Pattern[str]`):
                The core-schema pattern for the tag.
            kind (`str`):
                Name of the expected kind, used in the error.

        Returns:
            `str`:
                The scalar's text.

        Raises:
            `ConstructorError`:
                Raised when the whole text does not match ``pattern``.
        """

        text = self.construct_scalar(node)
        # fullmatch: `$` alone would accept a trailing newline from a block or
        # quoted scalar under an explicit tag.
        if not pattern.fullmatch(text):
            raise ConstructorError(
                None,
                None,
                f"expected a YAML 1.2 core {kind}, but found {text!r}",
                node.start_mark,
            )
        return text


# Inside one first-character bucket int precedes float: the float pattern also
# matches integers.
_FlowLoader.add_implicit_resolver(_BOOL_TAG, _BOOL, list("tTfF"))
_FlowLoader.add_implicit_resolver(_INT_TAG, _INT, list("-+0123456789"))
_FlowLoader.add_implicit_resolver(_FLOAT_TAG, _FLOAT, list("-+0123456789."))
# PyYAML keys the empty scalar's bucket as ''; a None key is the wildcard.
_FlowLoader.add_implicit_resolver(_NULL_TAG, _NULL, ["~", "n", "N", ""])
_FlowLoader.add_implicit_resolver(_MERGE_TAG, _MERGE, ["<"])
_FlowLoader.add_implicit_resolver(
    _TIMESTAMP_TAG, _pyyaml_timestamp_regex(), list("0123456789")
)

_FlowLoader.add_constructor(_BOOL_TAG, _FlowLoader.construct_yaml_bool)
_FlowLoader.add_constructor(_INT_TAG, _FlowLoader.construct_yaml_int)
_FlowLoader.add_constructor(_FLOAT_TAG, _FlowLoader.construct_yaml_float)
_FlowLoader.add_constructor(_TIMESTAMP_TAG, _FlowLoader.construct_yaml_timestamp)
