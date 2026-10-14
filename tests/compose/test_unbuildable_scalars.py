"""An unquoted value YAML can't build is a located load error that says to quote it."""

import pytest

from yian.compose import LoadError, load_flow

_FIX = "quote it to keep it as text"


def _flow(binding: str, *, inputs: str = "") -> str:
    return (
        "id: f\nname: f\n"
        + inputs
        + "nodes:\n  use:\n    kind: code\n    code: tests.engine._compose_codefns:echo\n"
        f"    input:\n      topic: {binding}\n    output: str\n"
        "output: ${use.output}\n"
    )


def _line_of(text: str, needle: str) -> int:
    return next(i for i, line in enumerate(text.splitlines(), 1) if needle in line)


def _load_error(text: str, **kwargs) -> LoadError:
    with pytest.raises(LoadError) as caught:
        load_flow(text, **kwargs)
    return caught.value


@pytest.mark.parametrize(
    "value",
    [
        "2026-13-45",
        "2026-02-30",
        "0000-01-01",
        "2026-01-01T25:00:00",
        "[2026-13-45]",
        "2026-01-01T10:00:00+05:99",
    ],
)
def test_a_named_input_value(value):
    text = _flow(value)
    error = _load_error(text)
    assert "not a valid date" in str(error)  # also matches "not a valid datetime"
    assert _FIX in str(error)
    assert error.diagnostic.frames[-1].range.start_line == _line_of(text, "topic:")


_DATE = ("2026-13-45", "not a valid date: month must be in 1..12")
_OFFSET = ("2026-01-01T10:00:00+05:99", "not a valid datetime: UTC offset minutes must be under 60")


@pytest.mark.parametrize(("kind", "value", "reason"), [("date", *_DATE), ("datetime", *_OFFSET)])
def test_an_input_default(kind, value, reason):
    text = _flow("${input.d}", inputs=f"input:\n  d: !{kind} {value}\n")
    error = _load_error(text)
    assert f"{reason}; {_FIX}" in str(error)
    assert error.diagnostic.frames[-1].range.start_line == _line_of(text, f"d: !{kind}")


@pytest.mark.parametrize(("kind", "value", "reason"), [("date", *_DATE), ("datetime", *_OFFSET)])
def test_a_typed_binding_literal(kind, value, reason):
    text = _flow(f"!{kind} {value}")
    error = _load_error(text)
    assert f"{reason}; {_FIX}" in str(error)
    assert error.diagnostic.frames[-1].range.start_line == _line_of(text, "topic:")


def test_a_case_when():
    text = (
        "id: f\nname: f\ninput:\n  d: !date\nnodes:\n  route:\n    kind: case\n"
        "    on: ${input.d}\n    cases:\n      - when: 2026-13-45\n"
        "        then: x\n    else: x\n"
        "  x:\n    kind: code\n    output: str\n    code: |\n      return 'x'\n"
        "output: ${route.output}\n"
    )
    error = _load_error(text)
    assert "not a valid date" in str(error) and _FIX in str(error)
    assert error.diagnostic.frames[-1].range.start_line == _line_of(text, "when:")


def test_a_uses_file(tmp_path):
    (tmp_path / "child.yaml").write_text(_flow("2026-13-45").replace("id: f", "id: child"))
    parent = (
        "id: p\nname: p\nnodes:\n  a:\n    kind: call\n    call: ch\n"
        "output: ${a.output}\nuses:\n  ch: child\n"
    )
    error = _load_error(parent, search_paths=[tmp_path], source_label="parent.yaml")
    assert "not a valid date" in str(error) and _FIX in str(error)
    frames = error.diagnostic.frames
    assert [f.label for f in frames] == ["parent.yaml", "child.yaml"]
    assert frames[-1].range is not None
