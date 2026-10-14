"""CLI coverage for reading and printing ints of any size in `yian run` and `yian chat`."""

from pathlib import Path
import re
import sys

import pytest
from typer.testing import CliRunner

from yian.cli import app
from yian.cli.utils import displayable
import yian.json_codec as json_codec


runner = CliRunner()

BIG = 10**4999 + 12345  # 5000 digits, past Python's default 4300-digit limit
DIGITS = json_codec.int_text(BIG)

# CI forces Rich color, which can place style escapes inside printed values.
_ANSI_SGR = re.compile(r"\x1b\[[0-9;]*m")


@pytest.fixture(autouse=True)
def _default_int_digit_limit():
    """Run each test under Python's default int/str digit limit."""
    get = getattr(sys, "get_int_max_str_digits", None)
    if get is None:
        yield  # this Python has no limit
        return
    before = get()
    sys.set_int_max_str_digits(4300)
    try:
        yield
        after = get()
    finally:
        sys.set_int_max_str_digits(before)
    assert after == 4300, "yian code changed the global digit limit"


def _unwrapped(text: str) -> str:
    """Drop styling and line breaks, since Rich wraps long output at the console width."""
    return _ANSI_SGR.sub("", text).replace("\n", "")


def _write(tmp_path: Path, name: str, text: str) -> Path:
    """Write one file under the test directory and return its path."""
    path = tmp_path / name
    path.write_text(text)
    return path


_LIST_FLOW = """\
id: big_list
name: big_list
input:
  xs: list[int]
nodes:
  echo:
    kind: code
    input:
      xs: ${input.xs}
    output: list[int]
    code: |
      return inputs["xs"]
output: ${echo.output}
"""

_INT_FLOW = """\
id: big_int
name: big_int
input:
  n: int
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


def test_inputs_file_reads_and_prints_a_big_int_and_resume_replays_it(tmp_path: Path) -> None:
    """A big int read from `--inputs` is printed, recorded, and printed again on replay."""
    flow = _write(tmp_path, "flow.yaml", _LIST_FLOW)
    inputs = _write(tmp_path, "in.json", '{"xs": [' + DIGITS + "]}")
    log = tmp_path / "run.jsonl"

    result = runner.invoke(
        app, ["run", str(flow), "--inputs", str(inputs), "--log", str(log)]
    )

    assert result.exit_code == 0, result.output
    assert DIGITS in _unwrapped(result.stdout)

    replay = runner.invoke(app, ["run", str(flow), "--resume", str(log)])

    assert replay.exit_code == 0, replay.output
    assert DIGITS in _unwrapped(replay.stdout)


def test_input_flag_reads_a_big_int(tmp_path: Path) -> None:
    """`-i n=<digits>` reads a big int for an `int` input and prints it."""
    flow = _write(tmp_path, "flow.yaml", _INT_FLOW)

    result = runner.invoke(
        app, ["run", str(flow), "-i", f"n={DIGITS}", "--log", str(tmp_path / "run.jsonl")]
    )

    assert result.exit_code == 0, result.output
    assert DIGITS in _unwrapped(result.stdout)


def test_verbose_prints_a_big_node_output_on_stderr(tmp_path: Path) -> None:
    """`--verbose` prints each node's big-int output under its check on stderr."""
    flow = _write(tmp_path, "flow.yaml", _INT_FLOW)

    result = runner.invoke(
        app,
        [
            "run",
            str(flow),
            "-i",
            f"n={DIGITS}",
            "--verbose",
            "--log",
            str(tmp_path / "run.jsonl"),
        ],
    )

    assert result.exit_code == 0, result.output
    # Each wrapped line of node output is indented, so drop all whitespace.
    assert DIGITS in re.sub(r"\s+", "", _ANSI_SGR.sub("", result.stderr))


def test_displayable_spells_big_ints_and_keeps_other_values() -> None:
    """`displayable` swaps long ints for their digits and leaves other values alone."""
    assert DIGITS in str(displayable({"a": [BIG]}))
    assert DIGITS in repr(displayable((BIG,)))
    assert displayable(BIG).__repr__() == DIGITS
    small = {"a": [1, 2.5, "x", None, True], "b": (3,)}
    assert displayable(small) == small
    marker = object()
    assert displayable(marker) is marker
    assert displayable(True) is True
    assert displayable(7) == 7


def test_chat_prints_a_big_int_reply(tmp_path: Path) -> None:
    """A chat flow's non-text `reply` holding a big int is printed."""
    flow = _write(
        tmp_path,
        "chat.yaml",
        "id: big_chat\n"
        "name: big_chat\n"
        "input:\n"
        "  opening: str\n"
        "nodes:\n"
        "  reply:\n"
        "    kind: code\n"
        "    output: list[int]\n"
        "    code: |\n"
        "      return [10**4999 + 12345]\n"
        "output: ${reply.output}\n",
    )

    result = runner.invoke(app, ["chat", str(flow)], input="/exit\n")

    assert result.exit_code == 0, result.output
    assert DIGITS in _unwrapped(result.stdout)
