from dataclasses import FrozenInstanceError
import pytest
from agent_composer.events import Diagnostic, SourceRange


def test_source_range_is_frozen_1based():
    r = SourceRange(start_line=3, start_col=5, end_line=3, end_col=9)
    assert (r.start_line, r.start_col, r.end_line, r.end_col) == (3, 5, 3, 9)
    with pytest.raises(FrozenInstanceError):
        r.start_line = 1  # frozen


def test_diagnostic_holds_primary_and_notes():
    pr = SourceRange(1, 1, 1, 4)
    d = Diagnostic(severity="error", message="boom", primary=pr, notes=())
    assert d.severity == "error" and d.primary is pr and d.notes == ()


# --------------------------------------------------------------------------- #
# covered_lines — the half-open [start, end) -> lines conversion.
#
# The end is EXCLUSIVE, so `end_line` is covered only if the span reaches real text on it. A
# YAML block construct ends at the START OF THE NEXT TOKEN — a later line holding only
# indentation — so naively marking `end_line` highlights the following construct.
#
# `_YAML` mirrors the two shapes that matter, with node `b` deliberately right after node `a`:
#
#    7     code: |-              <- `a`'s block scalar starts here
#    8       return 1            <- ...and its content ends here
#    9   b:                      <- the next node: must NOT be highlighted
# --------------------------------------------------------------------------- #

_YAML = (
    "id: t\n"                    # 1
    "name: t\n"                  # 2
    "nodes:\n"                   # 3
    "  a:\n"                     # 4
    "    kind: code\n"           # 5
    "    output: int\n"          # 6
    "    code: |-\n"             # 7
    "      return 1\n"           # 8
    "  b:\n"                     # 9
    "    kind: code\n"           # 10
)


def test_covered_lines_block_scalar_stops_before_the_next_token():
    # `a`'s `code:` block: starts at 7:11, ends at 9:1 — the start of `b:`. Column 1 means the
    # span reaches NO text on line 9, so it covers 7-8. Marking 9 would blame the next node.
    assert SourceRange(7, 11, 9, 1).covered_lines(_YAML) == [7, 8]


def test_covered_lines_block_mapping_end_on_indentation_only():
    # A block mapping ends at the next KEY's column, not column 1 — here 10:5 (`kind:` on line
    # 10, indented 4). Line 10 before col 5 is pure indentation, so it is not covered either.
    # A rule keyed on `end_col == 1` would miss this case.
    assert SourceRange(5, 5, 10, 5).covered_lines(_YAML) == [5, 6, 7, 8, 9]


def test_covered_lines_end_reaching_real_text_keeps_its_line():
    # The complement: when the span DOES reach text on `end_line`, that line is covered.
    assert SourceRange(5, 5, 6, 10).covered_lines(_YAML) == [5, 6]


def test_covered_lines_plain_scalar_is_one_line():
    assert SourceRange(5, 11, 5, 15).covered_lines(_YAML) == [5]


def test_covered_lines_zero_width_caret_still_marks_its_line():
    # start == end (the docstring's zero-width caret). The prefix before col 5 on line 5 is pure
    # indentation, which would drop to line 4 — the clamp keeps it on its own line.
    assert SourceRange(5, 5, 5, 5).covered_lines(_YAML) == [5]


def test_covered_lines_out_of_bounds_end_does_not_raise():
    # Defensive: a range past EOF reads as blank and drops the line, never IndexErrors.
    assert SourceRange(9, 3, 99, 1).covered_lines(_YAML) == list(range(9, 99))
    assert SourceRange(3, 1, 3, 1).covered_lines("") == [3]
