"""Read a tool's description and per-parameter descriptions from its docstring.

This ports the OpenAI Agents SDK's docstring rules without depending on griffe:
the style is detected the same way (google, numpy, or sphinx), a Google
parameter section with no blank line above it is normalized the same way, and
the result keeps the two parts the SDK reads from griffe's output, the first
text block and the parameters section. Leading `*`/`**` on a parameter name is
stripped so variadic parameters match their signature names.

Uses only the Python standard library.
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

_Style = Literal["google", "numpy", "sphinx"]

# griffe's admonition rule, which is also how it recognizes a Google section header.
_GOOGLE_HEADER_RE = re.compile(r"^(?P<type>\w[\s\w-]*):(\s+(?P<title>[^\s].*))?\s*$", re.IGNORECASE)
_GOOGLE_PARAM_SECTIONS = frozenset({"args", "arguments", "params", "parameters"})
_GOOGLE_NAME_RE = re.compile(r"\*{0,2}\w+")
_PARAM_SECTION_RE = re.compile(r"^(args|arguments|params|parameters):\s*$", re.IGNORECASE)
_DASH_LINE_RE = re.compile(r"^\s*-{3,}\s*$")
_SPHINX_FIELD_RE = re.compile(r"^:(?P<kind>\w+)(?P<args>[^:]*):(?P<value>.*)$")
_SPHINX_PARAM_KINDS = frozenset({"param", "parameter", "arg", "argument", "key", "keyword"})


@dataclass(frozen=True)
class DocInfo:
    """The parts of a docstring a tool schema uses.

    Args:
        description (`str` or `None`):
            The first text block, stripped. `None` when the docstring has none.
        params (`Mapping[str, str]`):
            Parameter descriptions keyed by parameter name, without leading stars.
    """

    description: str | None
    params: Mapping[str, str]


def parse_docstring(doc: str | None) -> DocInfo:
    """Parse a docstring into its description and parameter descriptions.

    Args:
        doc (`str` or `None`):
            The raw docstring, for example a function's `__doc__`. It is cleaned
            with `inspect.cleandoc` first. `None` or blank gives an empty result.

    Returns:
        `DocInfo`:
            The description and parameter descriptions, parsed in the detected style.

    Example:
        ```python
        info = parse_docstring("Add.\\n\\nArgs:\\n    a: First.")
        assert info == DocInfo("Add.", {"a": "First."})
        ```
    """
    if doc is None:
        return DocInfo(None, {})
    cleaned = inspect.cleandoc(doc)
    if not cleaned.strip():
        return DocInfo(None, {})
    style = _detect_style(cleaned)
    if style == "sphinx":
        return _parse_sphinx(cleaned.splitlines())
    if style == "numpy":
        return _parse_numpy(cleaned.splitlines())
    return _parse_google(_ensure_blank_line_before_google_sections(cleaned).splitlines())


def _detect_style(doc: str) -> _Style:
    """Score each style by its markers; ties prefer sphinx, then numpy, then google."""
    scores: dict[_Style, int] = {"sphinx": 0, "numpy": 0, "google": 0}
    for pattern in (r"^:param\s", r"^:type\s", r"^:return:", r"^:rtype:"):
        if re.search(pattern, doc, re.MULTILINE):
            scores["sphinx"] += 1
    for pattern in (r"^Parameters\s*\n\s*-{3,}", r"^Returns\s*\n\s*-{3,}", r"^Yields\s*\n\s*-{3,}"):
        if re.search(pattern, doc, re.MULTILINE):
            scores["numpy"] += 1
    for pattern in (r"^(Args|Arguments):", r"^(Returns):", r"^(Raises):"):
        if re.search(pattern, doc, re.MULTILINE):
            scores["google"] += 1
    best = max(scores.values())
    if best == 0:
        return "google"
    styles: tuple[_Style, ...] = ("sphinx", "numpy", "google")
    return next(style for style in styles if scores[style] == best)


def _ensure_blank_line_before_google_sections(doc: str) -> str:
    """Insert a blank line above a parameter section header that directly follows text.

    griffe skips a Google section header with no blank line above it, which
    would drop every parameter description, so the header gets one when an
    indented block follows it.
    """
    lines = doc.splitlines()
    output: list[str] = []
    inserted = False
    for index, line in enumerate(lines):
        if (
            index > 0
            and _PARAM_SECTION_RE.match(line)
            and output
            and output[-1].strip()
            and index + 1 < len(lines)
            and lines[index + 1].startswith((" ", "\t"))
        ):
            output.append("")
            inserted = True
        output.append(line)
    if not inserted:
        return doc
    return "\n".join(output)


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def _join(parts: list[str]) -> str:
    """Join stripped description lines with newlines, dropping outer blank lines."""
    return "\n".join(part.strip() for part in parts).strip()


def _text(lines: list[str]) -> str | None:
    return "\n".join(lines).strip() or None


def _parse_google(lines: list[str]) -> DocInfo:
    description: str | None = None
    params: dict[str, str] = {}
    seen_header = False
    index = 0
    while index < len(lines):
        line = lines[index]
        match = _GOOGLE_HEADER_RE.match(line)
        is_header = (
            match is not None
            and (index == 0 or not lines[index - 1].strip())
            and index + 1 < len(lines)
            and lines[index + 1].startswith((" ", "\t"))
        )
        if not is_header:
            index += 1
            continue
        if not seen_header:
            description = _text(lines[:index])
            seen_header = True
        end = index + 1
        while end < len(lines) and (not lines[end].strip() or lines[end].startswith((" ", "\t"))):
            end += 1
        if match["type"].strip().lower() in _GOOGLE_PARAM_SECTIONS:
            params.update(_google_params(lines[index + 1 : end]))
        index = end
    if not seen_header:
        description = _text(lines)
    return DocInfo(description, params)


def _google_params(body: list[str]) -> dict[str, str]:
    """Read `name[ (type)]: rest` entries and their deeper-indented continuation lines."""
    base = next(_indent(line) for line in body if line.strip())
    entries: list[tuple[str, list[str]]] = []
    current: list[str] | None = None
    for line in body:
        if not line.strip() or _indent(line) > base:
            if current is not None:
                current.append(line)
            continue
        current = None
        entry = _split_google_entry(line.strip())
        if entry is not None:
            current = [entry[1]]
            entries.append((entry[0], current))
    return {name: _join(parts) for name, parts in entries}


def _split_google_entry(text: str) -> tuple[str, str] | None:
    """Split `name[ (type)]: rest` into the star-stripped name and `rest`."""
    match = _GOOGLE_NAME_RE.match(text)
    if match is None:
        return None
    name = match.group().lstrip("*")
    rest = text[match.end() :].lstrip()
    if rest.startswith("("):
        # A type may itself contain colons, so the entry splits after its closing parenthesis.
        depth = 0
        for position, char in enumerate(rest):
            depth += {"(": 1, ")": -1}.get(char, 0)
            if depth == 0:
                rest = rest[position + 1 :].lstrip()
                break
        else:
            return None
    if not rest.startswith(":"):
        return None
    return name, rest[1:]


def _parse_numpy(lines: list[str]) -> DocInfo:
    headers = [
        index
        for index in range(len(lines) - 1)
        if lines[index].strip() and _DASH_LINE_RE.match(lines[index + 1])
    ]
    description = _text(lines[: headers[0]] if headers else lines)
    entries: list[tuple[str, list[str]]] = []
    for position, header in enumerate(headers):
        if lines[header].strip().lower() != "parameters":
            continue
        end = headers[position + 1] if position + 1 < len(headers) else len(lines)
        current: list[str] | None = None
        for line in lines[header + 2 : end]:
            if not line.strip() or line.startswith((" ", "\t")):
                if current is not None:
                    current.append(line)
                continue
            current = []
            entries.append((line.split(":", 1)[0].strip().lstrip("*"), current))
    return DocInfo(description, {name: _join(parts) for name, parts in entries})


def _parse_sphinx(lines: list[str]) -> DocInfo:
    first_field = next(
        (index for index, line in enumerate(lines) if line.startswith(":")), len(lines)
    )
    description = _text(lines[:first_field])
    entries: list[tuple[str, list[str]]] = []
    current: list[str] | None = None
    for line in lines[first_field:]:
        if line.startswith((" ", "\t")):
            if current is not None:
                current.append(line)
            continue
        current = None
        match = _SPHINX_FIELD_RE.match(line)
        if match is None or match["kind"] not in _SPHINX_PARAM_KINDS:
            continue
        words = match["args"].split()
        if not words:
            continue
        current = [match["value"]]
        entries.append((words[-1].lstrip("*"), current))
    return DocInfo(description, {name: _join(parts) for name, parts in entries})
