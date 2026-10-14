"""Each old type form is a located load error that shows the tag spelling."""

import re

import pytest

from yian.compose import LoadError, load_flow, read_flow_inputs
from yian.compose.helpers.types import read_variables


def _error_at(text: str, message: str, line: str) -> LoadError:
    """Load `text`, expecting `message` located at the line spelled `line`."""
    with pytest.raises(LoadError, match=re.escape(message)) as caught:
        load_flow(text)
    expected = text.splitlines().index(line) + 1
    assert caught.value.diagnostic.frames[-1].range.start_line == expected
    return caught.value


_NODE_HEADER = """\
id: old_forms
name: old_forms
input:
  n: !int
"""


# ---------- bracket types ----------


def test_bracket_typedef_shows_the_parenthesized_spelling():
    text = (
        "id: f\nname: f\ntypedefs:\n  Buy:\n    action: Literal[buy]\n"
        "input:\n  b: !Buy\nkind: code\ncode: return inputs['b']\noutput: Buy\n"
    )
    _error_at(text, "types use parentheses: `Literal(buy)`", "    action: Literal[buy]")


@pytest.mark.parametrize(
    ("declared", "spelled"),
    [("Optional[str]", "Optional(str)"), ("int[1, 12]", "int(1..12)"),
     ("dict[str, int]", "object")],
)
def test_bracket_node_output_shows_the_parenthesized_spelling(declared, spelled):
    text = (
        _NODE_HEADER
        + "nodes:\n  step:\n    kind: code\n    code: return None\n"
        + f"    output: {declared}\noutput: ${{step.output}}\n"
    )
    _error_at(text, f"types use parentheses: `{spelled}`", f"    output: {declared}")


# ---------- `name: T = value` bindings ----------

_BINDING_FLOW = """\
id: old_bindings
name: old_bindings
typedefs:
  Buy: {{action: Literal(buy)}}
  Sell: {{action: Literal(sell)}}
input:
  n: !int
nodes:
  size:
    kind: code
    code: "return {{'qty': inputs['n']}}"
    input:
      n: ${{input.n}}
    output: {{qty: int}}
  use:
    kind: code
    code: return inputs['{name}']
    input:
      {name}: {value}
    output: str
output: ${{use.output}}
"""


def _binding_flow(name: str, value: str) -> str:
    """Spell a flow whose `use` node binds `name` to `value`."""
    return _BINDING_FLOW.format(name=name, value=value)


def _quote_hint(name: str, text: str) -> str:
    """The advice a binding error ends with, for text meant literally."""
    return f". If you meant the text, quote it: `{name}: {text}`"


@pytest.mark.parametrize(
    ("name", "value", "spelled"),
    [
        ("qty", "int = ${size.output.qty}", "`qty: !int ${size.output.qty}`"),
        ("q", "Optional[int] = ${size.output.qty}", "`q: !Optional(int) ${size.output.qty}`"),
        (
            "d",
            "Union[Buy, Sell] = ${size.output}",
            "name `Union(Buy, Sell)` in `typedefs:` and tag with its name",
        ),
        ("s", 'str = "30"', '`s: !str "30"`'),
        ("s", "str = 30", '`s: !str "30"`'),
        ("b", "bool = yes", "`b: !bool true` (`yes` is text; write `true`)"),
        ("n", "int = 5", "`n: !int 5`"),
        ("b", "Buy = ${size.output}", "`b: !Buy ${size.output}`"),
        ("e", "int =", "`e: !int ${...}`"),
        ("t", "int = call(f, x = 1)", "`t: !int call(f, x = 1)`"),
    ],
)
def test_typed_binding_text_shows_the_tag(name, value, spelled):
    _error_at(
        _binding_flow(name, value),
        f"binding {name!r}: write the type as a tag: {spelled}"
        + _quote_hint(name, '"' + value + '"' if '"' not in value else "'" + value + "'"),
        f"      {name}: {value}",
    )


def test_typed_binding_text_that_reads_as_words_offers_quoting():
    _error_at(
        _binding_flow("note", "date = tomorrow"),
        "binding 'note': write the type as a tag: `note: !date tomorrow`"
        '. If you meant the text, quote it: `note: "date = tomorrow"`',
        "      note: date = tomorrow",
    )


def test_typed_flow_output_text_shows_the_tag():
    text = _binding_flow("v", "${input.n}").replace(
        "output: ${use.output}\n", "output:\n  r: int = ${use.output}\n"
    )
    _error_at(
        text,
        "binding 'r': write the type as a tag: `r: !int ${use.output}`"
        + _quote_hint("r", '"int = ${use.output}"'),
        "  r: int = ${use.output}",
    )


def test_typed_single_flow_output_text_shows_the_tag():
    text = _binding_flow("v", "${input.n}").replace(
        "output: ${use.output}\n", "output: Optional[int] = ${use.output}\n"
    )
    _error_at(
        text,
        "binding 'output': write the type as a tag: `output: !Optional(int) ${use.output}`"
        + _quote_hint("output", '"Optional[int] = ${use.output}"'),
        "output: Optional[int] = ${use.output}",
    )


@pytest.mark.parametrize(
    ("value", "text"),
    [("a = b", "a = b"), ('"int = 5"', "int = 5"), ("itn = 5", "itn = 5"),
     ("x = ${input.n}", "x = ${input.n}")],
)
def test_text_without_a_type_on_the_left_is_plain_text(value, text):
    from yian.run import run_flow

    flow = load_flow(_binding_flow("note", value))
    result = run_flow(flow, {"n": 3})
    assert result.status == "succeeded", result.failure
    assert result.output == text.replace("${input.n}", "3")


# ---------- flow inputs and agent variables ----------

_INPUT_FLOW = """\
id: old_inputs
name: old_inputs
typedefs:
  Message: {{role: str, content: str}}
input:
  {declaration}
kind: code
code: return 1
output: int
"""


@pytest.mark.parametrize(
    ("declaration", "message"),
    [
        ("qty: int", "flow input 'qty': write the type as a tag: `qty: !int`"),
        (
            "history: list[Message] = []",
            "flow input 'history': write the type as a tag: `history: !list(Message) []`",
        ),
        ("n: int = 3", "flow input 'n': write the type as a tag: `n: !int 3`"),
        ("o: Optional[str]", "flow input 'o': write the type as a tag: `o: !Optional(str)`"),
        (
            "cfg: {a: int}",
            "flow input 'cfg': name the record in `typedefs:` and tag the input with it, "
            "e.g. `cfg: !Cfg`",
        ),
        (
            "user_cfg: {a: int}",
            "flow input 'user_cfg': name the record in `typedefs:` and tag the input with "
            "it, e.g. `user_cfg: !UserCfg`",
        ),
        (
            "x: 5",
            "flow input 'x': an input is a tag with an optional default, "
            "e.g. `x: !int` or `x: !int 5`",
        ),
        (
            "x: two words",
            "flow input 'x': an input is a tag with an optional default, "
            "e.g. `x: !str` or `x: !str two words`",
        ),
        # Text whose leading name is no type and close to none is a value, not a type.
        (
            "topic: hello",
            "flow input 'topic': an input is a tag with an optional default, "
            "e.g. `topic: !str` or `topic: !str hello`",
        ),
        (
            "topic: hello = world",
            "flow input 'topic': an input is a tag with an optional default, "
            'e.g. `topic: !str` or `topic: !str "hello = world"`',
        ),
        (
            "x: '30'",
            "flow input 'x': an input is a tag with an optional default, "
            'e.g. `x: !str` or `x: !str "30"`',
        ),
        # A leading name that is a type keeps its tag even when an argument names no
        # type, so pasting it back reaches the tag reader's unknown-type error.
        (
            "history: list[message] = []",
            "flow input 'history': write the type as a tag: `history: !list(message) []`",
        ),
        # A leading name close to a type name is spelled as that name.
        ("x: strr", "flow input 'x': write the type as a tag: `x: !str`"),
        ("x: Strr = hi", "flow input 'x': write the type as a tag: `x: !str hi`"),
        ("x: Optinal(int)", "flow input 'x': write the type as a tag: `x: !Optional(int)`"),
        ("x: Optional(Foo)", "flow input 'x': write the type as a tag: `x: !Optional(Foo)`"),
        (
            "u: Union[A, B]",
            "flow input 'u': write the type as a tag: "
            "name `Union(A, B)` in `typedefs:` and tag with its name",
        ),
        ("s: str = 30", 'flow input \'s\': write the type as a tag: `s: !str "30"`'),
        (
            "f: bool = no",
            "flow input 'f': write the type as a tag: `f: !bool false` "
            "(`no` is text; write `false`)",
        ),
    ],
)
def test_untagged_flow_input_shows_the_tag(declaration, message):
    text = _INPUT_FLOW.format(declaration=declaration)
    _error_at(text, message, f"  {declaration}")


def test_untagged_input_in_a_definition_shows_the_tag():
    text = (
        "id: f\nname: f\ndefs:\n  child:\n    input:\n      qty: int\n"
        "    kind: code\n    code: return 1\n    output: int\n"
        "nodes:\n  c:\n    kind: call\n    call: child\n    input:\n      qty: 1\n"
        "output: ${c.output}\n"
    )
    _error_at(text, "flow input 'qty': write the type as a tag: `qty: !int`", "      qty: int")


_AGENT_FLOW = """\
id: old_variables
name: old_variables
nodes:
  ask:
    kind: agent
    prompt: Count to ${{n}}.
    variables:
      {declaration}
    output: str
output: ${{ask.output}}
"""


@pytest.mark.parametrize(
    ("declaration", "message"),
    [
        ("n: int[1, 12] = 4", "variable 'n': write the type as a tag: `n: !int(1..12) 4`"),
        ("n: int = 4", "variable 'n': write the type as a tag: `n: !int 4`"),
        ("n: int", "variable 'n': write the type as a tag: `n: !int <default>`"),
        ("n: 4", "variable 'n': a variable is a tag with a default, e.g. `n: !int 4`"),
        (
            "tone: concise",
            "variable 'tone': a variable is a tag with a default, e.g. `tone: !str concise`",
        ),
    ],
)
def test_untagged_agent_variable_shows_the_tag(declaration, message):
    text = _AGENT_FLOW.format(declaration=declaration)
    _error_at(text, message, f"      {declaration}")


@pytest.mark.parametrize("reader", [read_flow_inputs, read_variables])
def test_source_free_declarations_take_a_type_only(reader):
    with pytest.raises(
        LoadError,
        match=re.escape(
            "source-free declarations take a type only; write defaults in YAML with a tag"
        ),
    ):
        reader({"x": "int = 3"}, {})


def test_source_free_input_rejects_a_record_map():
    with pytest.raises(
        LoadError, match=re.escape("input 'x': a source-free declaration is type text")
    ):
        read_flow_inputs({"x": {"a": "int"}}, {})
