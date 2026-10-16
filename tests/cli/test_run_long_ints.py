"""`yian run` rejects an int with more than 4300 digits at its inputs."""

from pathlib import Path
import re
import sys

import pytest
from typer.testing import CliRunner

from yian.cli import app
import yian.cli.run as run_cli
from yian.typesys.values import Type, ValueKind


runner = CliRunner()

D = "1" * 5000
LIMIT_MSG = "integer with 5000 digits is larger than yian allows (4300 digits)"

# CI forces Rich color, which can place style escapes inside printed values.
_ANSI_SGR = re.compile(r"\x1b\[[0-9;]*m")


@pytest.fixture(autouse=True)
def _default_int_digit_limit():
    """Run each test under Python's default int/str digit limit."""
    before = sys.get_int_max_str_digits()
    sys.set_int_max_str_digits(4300)
    try:
        yield
    finally:
        sys.set_int_max_str_digits(before)


def _unwrapped(text: str) -> str:
    """Drop styling and collapse whitespace, since Rich wraps output at the console width."""
    return " ".join(_ANSI_SGR.sub("", text).split())


def _assert_clean(text: str) -> None:
    """Check output shows no digits of the long int and no Python limit hint."""
    assert "set_int_max_str_digits" not in text
    assert re.search(r"[0-9]{600}", text) is None


def _write(tmp_path: Path, name: str, text: str) -> Path:
    """Write one file under the test directory and return its path."""
    path = tmp_path / name
    path.write_text(text)
    return path


_LIST_FLOW = """\
id: long_list
name: long_list
input:
  xs: !list(int)
nodes:
  echo:
    kind: code
    input:
      xs: ${input.xs}
    output: list(int)
    code: |
      return inputs["xs"]
output: ${echo.output}
"""

_INT_FLOW = """\
id: long_int
name: long_int
input:
  n: !int
nodes:
  echo:
    kind: code
    input:
      n: ${input.n}
    output: int
    code: |
      return inputs["n"]
output: ${echo.output}
"""


def test_input_flag_rejects_a_long_int(tmp_path: Path) -> None:
    flow = _write(tmp_path, "flow.yaml", _INT_FLOW)
    log = tmp_path / "run.jsonl"

    result = runner.invoke(app, ["run", str(flow), "-i", f"n={D}", "--log", str(log)])

    assert result.exit_code != 0
    assert "input 'n': " + LIMIT_MSG in _unwrapped(result.output)
    _assert_clean(result.output)
    assert not log.exists()


def test_inputs_file_rejects_a_long_int_at_its_path(tmp_path: Path) -> None:
    flow = _write(tmp_path, "flow.yaml", _LIST_FLOW)
    inputs = _write(tmp_path, "in.json", '{"xs": [1, ' + D + "]}")
    log = tmp_path / "run.jsonl"

    result = runner.invoke(
        app, ["run", str(flow), "--inputs", str(inputs), "--log", str(log)]
    )

    assert result.exit_code != 0
    assert "input 'xs' at [1]: " + LIMIT_MSG in _unwrapped(result.output)
    _assert_clean(result.output)
    assert not log.exists()


def test_inputs_file_holding_a_lone_long_int_is_not_an_object(tmp_path: Path) -> None:
    flow = _write(tmp_path, "flow.yaml", _INT_FLOW)
    inputs = _write(tmp_path, "in.json", D)

    result = runner.invoke(
        app, ["run", str(flow), "--inputs", str(inputs), "--log", str(tmp_path / "run.jsonl")]
    )

    assert result.exit_code != 0
    assert "must contain a JSON object, got number" in _unwrapped(result.output)
    _assert_clean(result.output)


def test_structured_prompt_rejects_a_long_int() -> None:
    with pytest.raises(run_cli._StructuredInputError) as caught:
        run_cli._parse_structured_input(
            "[1, " + D + "]", Type.scalar(ValueKind.LIST_INTEGER), "xs"
        )
    assert str(caught.value) == "input 'xs' at [1]: " + LIMIT_MSG
    _assert_clean(str(caught.value))


def test_input_flag_accepts_an_int_at_the_limit(tmp_path: Path) -> None:
    flow = _write(tmp_path, "flow.yaml", _INT_FLOW)
    nines = "9" * 4300

    result = runner.invoke(
        app,
        ["run", str(flow), "-i", f"n={nines}", "--log", str(tmp_path / "run.jsonl")],
    )

    assert result.exit_code == 0, result.output
    assert "set_int_max_str_digits" not in result.output
