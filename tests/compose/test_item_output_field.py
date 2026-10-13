"""A MAP item field named `output` reads like any other field (`${item.output}`)."""

import pytest

from yian.compose import load_flow
from yian.run import run_flow

_LEAF = (
    "  leaf:\n"
    "    input:\n      x: str\n"
    "    nodes:\n"
    "      up:\n"
    "        kind: code\n"
    "        input:\n          x: ${input.x}\n"
    "        output: str\n"
    "        code: return inputs['x'].upper()\n"
    "    output: ${up.output}\n"
)

_ROWS = {"rows": [{"output": "a"}, {"output": "b"}]}


def _run(text: str) -> object:
    result = run_flow(load_flow(text), _ROWS)
    assert result.status == "succeeded", result
    return result.output


@pytest.mark.parametrize(
    "item_ref, expected",
    [("${item.output}", ["A", "B"]), ("'${item.output}!'", ["A!", "B!"])],
)
def test_map_reads_an_item_field_named_output(item_ref, expected):
    text = (
        "id: p\nname: p\n"
        "typedefs:\n  Row: {output: str}\n"
        "input:\n  rows: list[Row]\n"
        "defs:\n" + _LEAF
        + "nodes:\n"
        "  m:\n"
        "    kind: map\n"
        "    over: ${input.rows}\n"
        "    call: leaf\n"
        f"    input:\n      x: {item_ref}\n"
        "output: ${m.output}\n"
    )
    assert _run(text) == expected


def test_map_inside_a_called_def_reads_an_item_field_named_output():
    text = (
        "id: p\nname: p\n"
        "typedefs:\n  Row: {output: str}\n"
        "input:\n  rows: list[Row]\n"
        "defs:\n" + _LEAF
        + "  mid:\n"
        "    input:\n      rows: list[Row]\n"
        "    nodes:\n"
        "      m:\n"
        "        kind: map\n"
        "        over: ${input.rows}\n"
        "        call: leaf\n"
        "        input:\n          x: ${item.output}\n"
        "    output: ${m.output}\n"
        "nodes:\n"
        "  c:\n"
        "    kind: call\n"
        "    call: mid\n"
        "    input:\n      rows: ${input.rows}\n"
        "output: ${c.output}\n"
    )
    assert _run(text) == ["A", "B"]


def test_inline_map_body_reads_an_item_field_named_output():
    text = (
        "id: p\nname: p\n"
        "typedefs:\n  Row: {output: str}\n"
        "input:\n  rows: list[Row]\n"
        "nodes:\n"
        "  m:\n"
        "    kind: map\n"
        "    over: ${input.rows}\n"
        "    input:\n      x: ${item.output}\n"
        "    nodes:\n"
        "      up:\n"
        "        kind: code\n"
        "        input:\n          x: ${input.x}\n"
        "        output: str\n"
        "        code: return inputs['x'].upper()\n"
        "    output: ${up.output}\n"
        "output: ${m.output}\n"
    )
    assert _run(text) == ["A", "B"]
