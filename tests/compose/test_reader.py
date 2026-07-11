from agent_composer.compose.errors import LoadError
from agent_composer.events import SourceRange


def test_loaderror_carries_range():
    r = SourceRange(2, 1, 2, 6)
    e = LoadError("bad", range=r)
    assert e.range is r
