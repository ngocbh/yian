"""A mapping key YAML reads as non-text is a load error at the key."""

import pytest

from yian.compose.helpers.errors import LoadError
from yian.compose.loader import load_flow
from yian.diagnostics import SourceRange

# One flow with a `{site}` format field at each place a name is written. Filling
# one field with a key and the others with plain names keeps every other part valid.
_FLOW = """id: f
name: f
typedefs:
  R:
    a: int
    {field}: int
  {typedef}:
    a: int
defs:
  child:
    nodes:
      {def_node}:
        kind: code
        code: return 1
        output: int
    output: 1
input:
  q: !str
  {flow_input}: !str
nodes:
  a:
    kind: code
    code: return inputs["v"]
    input:
      v: ${{input.q}}
      {node_input}: x
      w: {{{record}: a}}
    output: str
  b:
    kind: call
    nodes:
      inner:
        kind: code
        code: return 1
        output: int
      {inline_node}:
        kind: code
        code: return 2
        output: int
    output: ${{inner.output}}
  {root_node}:
    kind: code
    code: return 3
    output: int
output:
  r: ${{a.output}}
  {flow_output}: ${{a.output}}
x-meta:
  {x_block}: v
"""

_SITES = (
    "field",
    "typedef",
    "def_node",
    "flow_input",
    "node_input",
    "record",
    "inline_node",
    "root_node",
    "flow_output",
    "x_block",
)


def _flow_with(site: str, key: str) -> str:
    """Return the flow with `key` written at `site` and plain names elsewhere."""

    names = {name: f"n_{name}" for name in _SITES}
    names["typedef"] = "T"
    names[site] = key
    return _FLOW.format(**names)


def _key_range(text: str, key: str, site: str) -> SourceRange:
    """Return the range of `key` where `_flow_with` wrote it."""

    template_line = next(
        line for line in _FLOW.splitlines() if "{" + site + "}" in line
    )
    line_number = _FLOW.splitlines().index(template_line) + 1
    column = text.splitlines()[line_number - 1].index(key) + 1
    return SourceRange(line_number, column, line_number, column + len(key))


def _load_error(text: str) -> LoadError:
    with pytest.raises(LoadError) as caught:
        load_flow(text, source_label="k.yaml")
    return caught.value


def test_the_flow_loads_with_text_names_everywhere():
    assert load_flow(_flow_with("x_block", "n_x_block")) is not None


@pytest.mark.parametrize(
    ("key", "read_as"),
    [
        ("True", "a boolean"),
        ("1", "an integer"),
        ("null", "null"),
        ("1.5", "a float"),
        ("2026-10-13", "a date"),
    ],
)
@pytest.mark.parametrize("site", _SITES)
def test_a_non_text_key_is_a_load_error_at_the_key(site, key, read_as):
    text = _flow_with(site, key)

    error = _load_error(text)

    assert error.diagnostic.message == (
        f"the key `{key}` was read by YAML as {read_as}, but names in a flow must "
        f'be text. Quote it ("{key}") or choose another name.'
    )
    assert error.diagnostic.frames[-1].range == _key_range(text, key, site)


@pytest.mark.parametrize(
    ("key", "read_as", "fix"),
    [
        ("2026-10-13 10:00:00", "a date and time", '"2026-10-13 10:00:00"'),
        ("!!binary aGVsbG8=", "binary data", '"aGVsbG8="'),
        ("0x1F", "an integer", '"0x1F"'),
    ],
)
def test_the_message_names_what_yaml_read(key, read_as, fix):
    error = _load_error(_flow_with("x_block", key))

    shown = key.removeprefix("!!binary ")
    assert error.diagnostic.message == (
        f"the key `{shown}` was read by YAML as {read_as}, but names in a flow must "
        f"be text. Quote it ({fix}) or choose another name."
    )


@pytest.mark.parametrize(
    ("entry", "read_as"),
    [("? \n  : v", "null"), ('!!binary "": v', "binary data")],
    ids=["plain", "tagged"],
)
def test_an_empty_key_asks_for_a_name(entry, read_as):
    text = _flow_with("x_block", "n_x_block").replace("  n_x_block: v", "  " + entry)

    error = _load_error(text)

    assert error.diagnostic.message == (
        f"an empty key was read by YAML as {read_as}, but names in a flow must be "
        "text. Write a name."
    )


def test_a_quoted_key_follows_the_usual_name_rules():
    assert load_flow(_flow_with("root_node", '"True"')) is not None
    assert load_flow(_flow_with("x_block", "'2026-10-13'")) is not None

    error = _load_error(_flow_with("root_node", "'1'"))

    assert "names in a flow must be text" not in error.diagnostic.message
    assert "_1" in error.diagnostic.message


@pytest.mark.parametrize("key", ["yes", "on", "off", "no"])
def test_words_yaml_1_1_read_as_booleans_are_text_keys(key):
    assert load_flow(_flow_with("root_node", key)) is not None
    assert load_flow(_flow_with("flow_input", key)) is not None


def test_case_on_needs_no_exemption():
    text = """id: f
name: f
input:
  x: !str
nodes:
  route:
    kind: case
    on: ${input.x}
    cases:
      - when: a
        then: done
    else: done
  done:
    kind: code
    code: return 1
    output: int
output: ${done.output}
"""

    assert load_flow(text) is not None


def test_a_child_flow_file_reports_its_own_key(tmp_path):
    (tmp_path / "child.yaml").write_text(
        "id: child\nname: child\nnodes:\n  a:\n    1: x\n    kind: code\n"
        "    code: return 1\n    output: int\noutput: ${a.output}\n"
    )
    parent = (
        "id: parent\nname: parent\nnodes:\n  c:\n    kind: call\n    call: kid\n"
        "output: ${c.output}\nuses:\n  kid: child\n"
    )

    with pytest.raises(LoadError) as caught:
        load_flow(parent, search_paths=[tmp_path])

    frame = caught.value.diagnostic.frames[-1]
    assert "was read by YAML as an integer" in caught.value.diagnostic.message
    assert frame.label == "child.yaml"
    assert frame.range == SourceRange(5, 5, 5, 6)


@pytest.mark.parametrize(
    ("meta", "key_range"),
    [
        ("x-meta:\n  017: a\n  17: b\n", SourceRange(4, 3, 4, 6)),
        ("x-meta: {1: a, true: b}\n", SourceRange(3, 10, 3, 11)),
    ],
    ids=["same-integer", "one-and-true"],
)
def test_a_non_text_key_is_reported_before_a_repeat_it_causes(meta, key_range):
    text = (
        "id: f\nname: f\n"
        + meta
        + "nodes:\n  a:\n    kind: code\n    code: return 1\n    output: int\n"
        "output: ${a.output}\n"
    )

    error = _load_error(text)

    assert "was read by YAML as an integer" in error.diagnostic.message
    assert error.diagnostic.frames[-1].range == key_range


def test_a_numeric_input_key_is_a_load_error_not_a_type_error():
    text = """id: f
name: f
nodes:
  a:
    kind: code
    input: {1: 2}
    code: return 1
    output: int
output: ${a.output}
"""

    error = _load_error(text)

    assert error.diagnostic.message.startswith(
        "the key `1` was read by YAML as an integer"
    )
    assert error.diagnostic.frames[-1].range == SourceRange(6, 13, 6, 14)


@pytest.mark.parametrize(
    ("where", "value"),
    [
        ("node_input", "!object {KEY: 2}"),
        ("node_input", "!list(object) [{KEY: 2}]"),
        ("node_input", '{KEY: "${input.q}"}'),
        ("node_input", "{d: !object {KEY: 2}, s: '${input.q}'}"),
        ("flow_default", "!object {KEY: 2}"),
        ("flow_default", "!list(object) [{KEY: 2}]"),
    ],
    ids=[
        "tagged",
        "tagged-list",
        "nested-template",
        "nested-tag",
        "tagged-default",
        "tagged-list-default",
    ],
)
def test_a_non_text_key_in_a_tagged_or_nested_value_is_the_same_error(where, value):
    written = value.replace("KEY", "1")
    if where == "node_input":
        line = f"      v: {written}"
        text = (
            "id: f\nname: f\ninput:\n  q: !str\nnodes:\n  a:\n    kind: code\n"
            f"    code: return 1\n    input:\n{line}\n    output: int\n"
            "output: ${a.output}\n"
        )
    else:
        line = f"  v: {written}"
        text = (
            f"id: f\nname: f\ninput:\n{line}\nnodes:\n  a:\n    kind: code\n"
            "    code: return 1\n    output: int\noutput: ${a.output}\n"
        )

    error = _load_error(text)

    assert error.diagnostic.message == (
        "the key `1` was read by YAML as an integer, but names in a flow must be "
        'text. Quote it ("1") or choose another name.'
    )
    line_number = text.splitlines().index(line) + 1
    column = line.index("1:") + 1
    assert error.diagnostic.frames[-1].range == SourceRange(
        line_number, column, line_number, column + 1
    )
