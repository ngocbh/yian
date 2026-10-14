"""The docs table of defaults and limits matches `yian.defaults.ENTRIES`."""

from __future__ import annotations

from pathlib import Path

from yian.defaults import ENTRIES, Entry

DOC = Path(__file__).resolve().parents[1] / "docs" / "defaults.md"
HEADER = ["Variable", "Group", "Default", "Valid values", "Meaning"]


def _cells(line: str) -> list[str]:
    """Return the stripped cells of one Markdown table row."""
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _table_rows(text: str) -> list[list[str]]:
    """Return the body rows of the one table whose header is `HEADER`."""
    lines = text.splitlines()
    starts = [i for i, line in enumerate(lines) if line.startswith("|") and _cells(line) == HEADER]
    assert len(starts) == 1, f"expected one defaults table, found {len(starts)}"
    rows = []
    for line in lines[starts[0] + 2 :]:
        if not line.startswith("|"):
            break
        rows.append(_cells(line))
    return rows


def _valid_values(entry: Entry) -> str:
    """Return the "Valid values" cell for `entry`."""
    if entry.maximum_from_interpreter:
        text = f"{entry.minimum:,} to the interpreter's int digit limit"
    elif entry.maximum is None:
        text = f"{entry.minimum:,} or more"
    else:
        text = f"{entry.minimum:,} to {entry.maximum:,}"
        if entry.maximum == entry.default:
            text += " (can only be lowered)"
    if entry.no_limit is not None:
        text += f", or {entry.no_limit} for no cap"
    return text


def _expected_row(entry: Entry) -> list[str]:
    """Return the table row `entry` should have."""
    return [
        f"`{entry.variable}`",
        entry.group,
        f"{entry.default:,}",
        _valid_values(entry),
        entry.description,
    ]


def test_table_lists_every_entry_in_order() -> None:
    rows = _table_rows(DOC.read_text(encoding="utf-8"))

    assert rows == [_expected_row(entry) for entry in ENTRIES]


def test_table_parser_finds_rows() -> None:
    text = "\n".join(
        [
            "| " + " | ".join(HEADER) + " |",
            "| --- | --- | --- | --- | --- |",
            "| `YIAN_X` | limit | 1 | 1 or more | X |",
            "",
            "| not | the table |",
        ]
    )

    assert _table_rows(text) == [["`YIAN_X`", "limit", "1", "1 or more", "X"]]
