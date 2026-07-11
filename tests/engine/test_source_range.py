from dataclasses import FrozenInstanceError
import pytest
from agent_composer.events import SourceRange


def test_source_range_is_frozen_1based():
    r = SourceRange(start_line=3, start_col=5, end_line=3, end_col=9)
    assert (r.start_line, r.start_col, r.end_line, r.end_col) == (3, 5, 3, 9)
    with pytest.raises(FrozenInstanceError):
        r.start_line = 1  # frozen
