"""A run log written while yian accepted ints of any size fails clearly when read."""

from __future__ import annotations

from pathlib import Path
import re
import sys

import pytest
from typer.testing import CliRunner

from yian import load_flow, run_flow
from yian.cli import app
from yian.expr.lineage import ValueRoute
from yian.run import resume_flow
from yian.runtime.contracts import _validate_route
from yian.runtime.log import JsonlRunLog, decode_record


D = "1" * 5000
LIMIT_MSG = "integer with 5000 digits is larger than yian allows (4300 digits)"
OLD_LOG_MSG = (
    f"run-record JSON: {LIMIT_MSG}; a log written while yian accepted such ints can't be read"
)
SENTINEL = "123456789123"

# The sentinel reaches the log only through RunStarted's initial pool here: START's own
# NodeSucceeded carries it too, so only the first line is edited.
_STARTED_FLOW = """\
id: old_started
name: old_started
input:
  n: !int
nodes:
  c:
    kind: code
    output: str
    code: |
      return "done"
output: ${c.output}
"""

# The CODE node derives the sentinel from constants, so no input carries it.
_CODE_FLOW = """\
id: old_code
name: old_code
input:
  t: !str
nodes:
  c:
    kind: code
    input:
      t: ${input.t}
    output: int
    code: |
      return 123456789000 + 123
output: ${c.output}
"""

# A learnable run records `input.n + 0` as a scalar route whose result is JSON text.
_ROUTE_FLOW = """\
id: old_route
name: old_route
input:
  n: !int
nodes:
  c:
    kind: code
    input:
      x: ${(input.n + 0) % 7}
    output: int
    code: |
      return inputs["x"]
output: ${c.output}
"""


def _assert_clean(message: str) -> None:
    """Check a message shows no digits of the long int and no Python limit hint."""
    assert "set_int_max_str_digits" not in message
    assert re.search(r"[0-9]{600}", message) is None


def _old_started_log(tmp_path: Path) -> tuple[str, Path]:
    """Return a flow and an old log whose RunStarted record holds the long int."""
    path = tmp_path / "started.jsonl"
    run_flow(load_flow(_STARTED_FLOW), {"n": int(SENTINEL)}, log=path)
    lines = path.read_text().splitlines(keepends=True)
    assert '"kind":"run_started"' in lines[0] and f'"n":{SENTINEL}' in lines[0]
    lines[0] = lines[0].replace(SENTINEL, D)
    path.write_text("".join(lines))
    return _STARTED_FLOW, path


def _old_code_log(tmp_path: Path) -> tuple[str, Path]:
    """Return a flow and an old log whose CODE node NodeSucceeded holds the long int."""
    path = tmp_path / "code.jsonl"
    run_flow(load_flow(_CODE_FLOW), {"t": "a"}, log=path)
    text = path.read_text()
    recorded = f'"node_id":"c","value":{{"kind":"int","value":{SENTINEL}}}'
    assert text.count(recorded) == 1
    path.write_text(text.replace(recorded, recorded.replace(SENTINEL, D)))
    return _CODE_FLOW, path


def _old_route_log(tmp_path: Path) -> tuple[str, Path]:
    """Return a flow and an old log whose route value_json string holds the long int."""
    path = tmp_path / "route.jsonl"
    run_flow(load_flow(_ROUTE_FLOW), {"n": int(SENTINEL)}, log=path, learnable=True)
    text = path.read_text()
    # The sentinel sits inside a JSON string here, so the outer record parse never sees it.
    recorded = f'"value_json":"{SENTINEL}"'
    assert text.count(recorded) == 1
    path.write_text(text.replace(recorded, recorded.replace(SENTINEL, D)))
    return _ROUTE_FLOW, path


_OLD_LOGS = pytest.mark.parametrize(
    "make_log",
    [_old_started_log, _old_code_log, _old_route_log],
    ids=["run_started", "code_node_succeeded", "route_value_json"],
)


@_OLD_LOGS
def test_resume_fails_clearly_and_leaves_the_log(tmp_path, make_log):
    """`resume_flow` names the digit count and why the log can't be read."""
    flow, path = make_log(tmp_path)
    before = path.read_bytes()

    with pytest.raises(ValueError) as caught:
        resume_flow(load_flow(flow), log=path)

    message = str(caught.value)
    assert message.endswith(f": {OLD_LOG_MSG}")
    _assert_clean(message)
    assert path.read_bytes() == before


@_OLD_LOGS
def test_replay_read_fails_clearly(tmp_path, make_log):
    """`JsonlRunLog.resume`, the reader replay and the CLI use, fails the same way."""
    _, path = make_log(tmp_path)
    before = path.read_bytes()

    with pytest.raises(ValueError) as caught:
        JsonlRunLog.resume(path)

    message = str(caught.value)
    assert message.endswith(f": {OLD_LOG_MSG}")
    _assert_clean(message)
    assert path.read_bytes() == before


@_OLD_LOGS
def test_decode_record_fails_clearly(tmp_path, make_log):
    """The record decoder itself raises the message, unprefixed."""
    _, path = make_log(tmp_path)
    line = next(line for line in path.read_bytes().splitlines() if D.encode() in line)

    with pytest.raises(ValueError) as caught:
        decode_record(line)

    assert str(caught.value) == OLD_LOG_MSG


@_OLD_LOGS
@pytest.mark.parametrize("python_limit", [0, 6000])
def test_decode_record_fails_clearly_whatever_pythons_limit(tmp_path, make_log, python_limit):
    """With Python's own limit off or above 4300, the decoder still finds the int."""
    _, path = make_log(tmp_path)
    line = next(line for line in path.read_bytes().splitlines() if D.encode() in line)
    saved = sys.get_int_max_str_digits()
    try:
        sys.set_int_max_str_digits(python_limit)
        with pytest.raises(ValueError) as caught:
            decode_record(line)
    finally:
        sys.set_int_max_str_digits(saved)

    assert str(caught.value) == OLD_LOG_MSG


@_OLD_LOGS
def test_cli_resume_fails_clearly(tmp_path, make_log):
    """`yian run --resume` reports the message and leaves the log unchanged."""
    flow, path = make_log(tmp_path)
    flow_path = tmp_path / "flow.yaml"
    flow_path.write_text(flow)
    before = path.read_bytes()

    result = CliRunner().invoke(app, ["run", str(flow_path), "--resume", str(path), "--quiet"])

    assert result.exit_code != 0
    output = re.sub(r"\s+", "", re.sub(r"\x1b\[[0-9;]*m", "", result.output))
    assert re.sub(r"\s+", "", OLD_LOG_MSG) in output
    _assert_clean(result.output)
    assert path.read_bytes() == before


def test_route_value_json_with_a_long_int_is_rejected():
    """A route's JSON result text holding a long int fails with the old-log message."""
    route = ValueRoute(kind="literal", value_json=f"[1, {D}]")

    with pytest.raises(ValueError) as caught:
        _validate_route(route)

    assert str(caught.value) == OLD_LOG_MSG


def test_invalid_record_json_keeps_its_message():
    """Syntax errors still read as invalid run-record JSON."""
    with pytest.raises(ValueError, match="^invalid run-record JSON$"):
        decode_record(b'{"schema": 9, ')
