"""Int literals of any length load from flow YAML, run, fingerprint and replay exactly."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

from yian import load_flow, run_flow
from yian.compile.manifest import flow_fingerprint
from yian.compose import LoadError
from yian.events import NodeStarted
from yian.run import resume_flow


D = "1" * 5000  # past Python's default 4300-digit limit
ONES = (10**5000 - 1) // 9  # the int spelled by D


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


def _flow(nodes: str, inputs: str = "", output: str = "${c.output}") -> str:
    """Return a flow with the given `input:` block, `nodes:` body and output."""
    return f"id: p\nname: p\n{inputs}nodes:\n{nodes}output: {output}\n"


def _code(binding: str) -> str:
    """Return a `c` CODE node that binds `n` to `binding` and returns it."""
    return (
        "  c:\n"
        "    kind: code\n"
        "    input:\n"
        f"      n: {binding}\n"
        "    code: |\n"
        "      return inputs['n']\n"
    )


def _run(text: str, inputs: dict | None = None):
    """Load and run `text`, asserting the run succeeds."""
    result = run_flow(load_flow(text), inputs or {})
    assert result.status == "succeeded", result.failure
    return result


@pytest.mark.parametrize(
    "text",
    [
        pytest.param(_flow(_code("${input.n}"), f"input:\n  n: !int {D}\n"), id="input-default"),
        pytest.param(_flow(_code(D)), id="bare-binding"),
        pytest.param(_flow(_code(f"!int {D}")), id="typed-binding"),
        pytest.param(_flow(f"  v: !int {D}\n" + _code("${v.output}")), id="value-entry"),
    ],
)
def test_a_long_int_literal_loads_and_runs(text):
    assert _run(text).output == ONES


def test_a_long_int_inside_an_object_value_entry_loads_and_runs():
    text = _flow(f"  v: !object {{k: [{D}]}}\n" + _code("${v.output}"))

    assert _run(text).output == {"k": [ONES]}


def test_an_inline_call_passes_a_long_int_argument():
    text = f"""id: p
name: p
defs:
  child:
    input:
      n: !int
    nodes:
      c:
        kind: code
        input:
          n: ${{input.n}}
        code: |
          return inputs['n']
    output: ${{c.output}}
nodes:
  c:
    kind: code
    input:
      n: call(child, n={D})
    code: |
      return inputs['n']
output: ${{c.output}}
"""

    assert _run(text).output == ONES


def test_the_fingerprint_covers_every_digit_of_a_long_int_literal():
    text = _flow(_code(D))
    changed = _flow(_code(D[:-1] + "2"))

    assert flow_fingerprint(load_flow(text).compiled) != flow_fingerprint(
        load_flow(changed).compiled
    )


def test_a_long_int_literal_is_replayed_on_resume(tmp_path: Path):
    text = _flow(_code(D))
    path = tmp_path / "events.jsonl"
    first = run_flow(load_flow(text), {}, log=path)
    assert first.status == "succeeded", first.failure
    before = path.read_bytes()

    resumed = resume_flow(load_flow(text), log=path)

    assert resumed.status == "succeeded"
    assert resumed.output == ONES
    assert not [event for event in resumed.events if isinstance(event, NodeStarted)]
    assert path.read_bytes() == before


def _case_flow(when: str) -> str:
    """Return a flow whose CASE on an int input routes `when:` to `big`, else `other`."""
    return _flow(
        "  g:\n"
        "    kind: case\n"
        "    on: ${input.x}\n"
        "    cases:\n"
        f"      - when: {when}\n"
        "        then: big\n"
        "    else: other\n"
        "  big: !str big\n"
        "  other: !str other\n",
        "input:\n  x: !int\n",
        "${g.output}",
    )


def test_a_case_when_matches_a_long_int():
    flow = load_flow(_case_flow(D))

    assert run_flow(flow, {"x": ONES}).output == "big"
    assert run_flow(flow, {"x": 1}).output == "other"


def test_a_case_when_matches_a_negative_long_int():
    flow = load_flow(_case_flow("-" + D))

    assert run_flow(flow, {"x": -ONES}).output == "big"
    assert run_flow(flow, {"x": ONES}).output == "other"


def test_a_quoted_long_int_when_against_an_int_on_says_to_drop_the_quotes():
    with pytest.raises(LoadError) as caught:
        load_flow(_case_flow(f'"{D}"'))

    message = str(caught.value)
    assert "; drop the quotes: `when: " in message
    assert "set_int_max_str_digits" not in message
