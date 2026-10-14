"""A CASE routes only: a branch names a node or calls a flow, and a CASE has no `output:`.

An untagged unquoted word names a node and an unquoted whole-value `call(...)` calls a
flow. Every other branch is a value: a tagged value (even a word naming a node), quoted
text, `null` or an empty `else:`, a number, a bool, a date, a mapping, a list, or
unquoted text holding `${`. A value branch is a located load error that shows the value
entry to write and the branch naming it, and each suggested entry loads when applied.
The entry keeps the branch's own tag, else takes the CASE's former `output:` type when a
tag spells it and the value fits it (a lone reference fits when its type does), else the
type of a lone reference or a tag guessed from the YAML value, else a `!<type>`
placeholder. A value no entry can hold (a number that is not finite, a value containing
itself, a written-out record with an empty key, a nested `call(...)`, a flow call or
malformed span in `${...}`, a span reading the case) is its own located error. An
unquoted word that names no node suggests the nearest node and the value entry for the
text, which loads too. A CASE `output:`, tagged or not, is a
load error at the `output:` that says to tag the result where it is read. A secret in a
value branch is reported as a secret literal, never echoed. `${route.output}` reads the
taken branch.
"""

import re
from typing import Any

import pytest

from yian.compose import LoadError, load_flow
from yian.run import run_flow

_INPUTS = "  cmd: !str\n"
_INPUTS_N = "  cmd: !str\n  n: !int\n"
# The value entry and branch a value-branch message suggests.
_REWRITE = re.compile(
    r"name it: `(?P<entry>[^`]+)`(?: \(.*\))? and `(?P<label>then|else): (?P<id>\w+)`$"
)
# The text entry a branch that names no node, or is broken call text, suggests.
_TEXT_ENTRY = re.compile(
    r"^case node 'route' (?P<label>then|else): .*"
    r" name it: `(?P<entry>(?P<id>\w+): [^`]+)`$"
)


def _line_of(text: str, needle: str) -> int:
    """Return the 1-based line number of the first line containing `needle`."""
    return next(
        number
        for number, line in enumerate(text.splitlines(), start=1)
        if needle in line
    )


def _load_error(text: str, message: str) -> LoadError:
    """Load `text`, expecting a load error whose whole message is `message`."""
    with pytest.raises(LoadError) as caught:
        load_flow(text)
    assert str(caught.value) == message
    return caught.value


def _error_line(error: LoadError) -> int:
    """Return the start line of the innermost frame of a load error."""
    frame = error.diagnostic.frames[-1]
    assert frame.range is not None
    return frame.range.start_line


def _flow(
    case_body: str, *, head: str = "", inputs: str = _INPUTS, extra_nodes: str = ""
) -> str:
    """Return a flow with a `route` case (indented body) and a `reply` node it reads."""
    return (
        "id: f\n"
        "name: f\n"
        f"{head}"
        "input:\n"
        f"{inputs}"
        "nodes:\n"
        "  route:\n"
        "    kind: case\n"
        f"{case_body}"
        "  reply:\n"
        "    kind: code\n"
        "    code: tests.engine._compose_codefns:echo\n"
        "    input:\n"
        "      topic: replied\n"
        "    output: str\n"
        f"{extra_nodes}"
        "output: ${route.output}\n"
    )


def _on(then: str, else_: str = "reply", *, output: str = "") -> str:
    """Return an `on: ${input.cmd}` case body with one `when: a` branch and an `else:`."""
    return (
        "    on: ${input.cmd}\n"
        f"{output}"
        "    cases:\n"
        "      - when: a\n"
        f"        then: {then}\n"
        f"    else: {else_}\n"
    )


def _rewritten(
    error: LoadError,
    *,
    head: str = "",
    inputs: str = _INPUTS,
    extra_nodes: str = "",
    type_text: str | None = None,
) -> str:
    """Return the `_on` flow with the value entry and branch an error suggests applied.

    The CASE keeps `when: a` and `reply` for its other branch, and has no `output:`.
    A text entry's branch names the entry. `type_text` fills a `!<type>` placeholder.
    """
    found = _REWRITE.search(str(error)) or _TEXT_ENTRY.search(str(error))
    assert found is not None, str(error)
    entry = found["entry"]
    if type_text is not None:
        entry = entry.replace("!<type>", f"!{type_text}")
    if found["label"] == "then":
        case_body = _on(found["id"])
    else:
        case_body = _on("reply", found["id"])
    return _flow(
        case_body, head=head, inputs=inputs, extra_nodes=f"{extra_nodes}  {entry}\n"
    )


def _taken(text: str, cmd: str = "a", **inputs: Any) -> Any:
    """Load `text` and return the flow's output; `cmd: a` takes the `then:`."""
    return run_flow(load_flow(text), {"cmd": cmd, **inputs}).output


def _value_branch(label: str, entry: str, shown: str, *, note: str = "") -> str:
    """Return the message for a value branch of `route`."""
    return (
        f"case node 'route' {label}: a branch names a node or calls a flow; put the value "
        f"in its own entry in `nodes:` and name it: `{entry}: {shown}`{note} and "
        f"`{label}: {entry}`"
    )


# --------------------------------------------------------------------------- #
# Node and call branches
# --------------------------------------------------------------------------- #


def test_case_output_over_node_branches_reads_the_taken_branch():
    text = _flow(
        _on("other"),
        extra_nodes=(
            "  other:\n"
            "    kind: code\n"
            "    code: |\n"
            "      return 'other'\n"
            "    output: str\n"
        ),
    )
    flow = load_flow(text)
    assert run_flow(flow, {"cmd": "a"}).output == "other"
    assert run_flow(flow, {"cmd": "b"}).output == "replied"


def test_case_output_over_a_value_entry_branch_reads_the_taken_branch():
    text = _flow(_on("verdict"), extra_nodes='  verdict: !date "2026-01-02"\n')
    flow = load_flow(text)
    assert run_flow(flow, {"cmd": "a"}).output == "2026-01-02"
    assert run_flow(flow, {"cmd": "b"}).output == "replied"


@pytest.mark.parametrize("node_id", ["__value_0", "__case_x"])
def test_former_reserved_prefixes_name_authored_nodes(node_id):
    text = _flow(
        _on(node_id),
        extra_nodes=(
            f"  {node_id}:\n"
            "    kind: code\n"
            "    code: |\n"
            "      return 'own'\n"
            "    output: str\n"
        ),
    )
    assert run_flow(load_flow(text), {"cmd": "a"}).output == "own"


# --------------------------------------------------------------------------- #
# Value branches
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("then", "shown", "needle", "value"),
    [
        ('"approve"', "!str approve", 'then: "approve"', "approve"),
        ("'approve'", "!str approve", "then: 'approve'", "approve"),
        ('"reply"', "!str reply", 'then: "reply"', "reply"),
        ('"it\'s: here"', "!str \"it's: here\"", "then: \"it's", "it's: here"),
        ('"[x"', '!str "[x"', 'then: "[x"', "[x"),
        ("|\n          two\n          lines", '!str "two\\nlines\\n"', "then: |", "two\nlines\n"),
        (">\n          one\n          line", '!str "one line\\n"', "then: >", "one line\n"),
        ("3", "!int 3", "then: 3", 3),
        ("1e3", "!float 1e3", "then: 1e3", 1000.0),
        ("true", "!bool true", "then: true", True),
        ("2026-01-02", '!date "2026-01-02"', "then: 2026-01-02", "2026-01-02"),
        ("null", "!Optional(str) null", "then: null", None),
        (
            "\n          reply: ${input.cmd}\n          exited: true",
            '!object {reply: "${input.cmd}", exited: true}',
            "reply: ${input.cmd}",
            {"reply": "a", "exited": True},
        ),
        ("{a: 1, b: [2.5]}", "!object {a: 1, b: [2.5]}", "then: {a: 1", {"a": 1, "b": [2.5]}),
        ("{a: 2026-01-02}", '!object {a: "2026-01-02"}', "then: {a: 2026", {"a": "2026-01-02"}),
        ('[{"": x}]', '!list(object) [{"": x}]', 'then: [{"": x}]', [{"": "x"}]),
        ('[a, "${input.cmd}"]', '!list(str) [a, "${input.cmd}"]', "then: [a", ["a", "a"]),
        ("[2026-01-02]", '!list(date) ["2026-01-02"]', "then: [2026", ["2026-01-02"]),
        ("[[1, 2], [3]]", "!list(list(int)) [[1, 2], [3]]", "then: [[1", [[1, 2], [3]]),
        ("[null, 1]", "!list(Optional(int)) [null, 1]", "then: [null", [None, 1]),
        (
            "[[1], [null]]",
            "!list(list(Optional(int))) [[1], [null]]",
            "then: [[1]",
            [[1], [None]],
        ),
        ("[[], [1]]", "!list(list(int)) [[], [1]]", "then: [[]", [[], [1]]),
        ("[1, 2.5]", "!list(float) [1, 2.5]", "then: [1, 2.5]", [1.0, 2.5]),
        ("[{a: 1}, {b: 2}]", "!list(object) [{a: 1}, {b: 2}]", "then: [{a", [{"a": 1}, {"b": 2}]),
        ("hi ${input.cmd}", "!str hi ${input.cmd}", "then: hi", "hi a"),
        ("${input.cmd}", "!str ${input.cmd}", "then: ${input.cmd}", "a"),
        ('"call(take"', '!str "call(take"', 'then: "call(take"', "call(take"),
    ],
)
def test_a_then_value_shows_its_value_entry_and_the_entry_loads(then, shown, needle, value):
    text = _flow(_on(then))
    error = _load_error(text, _value_branch("then", "route_then_0", shown))
    assert _error_line(error) == _line_of(text, needle)
    assert _taken(_rewritten(error)) == value


@pytest.mark.parametrize(
    ("else_", "shown", "needle", "value"),
    [
        ('"declined"', "!str declined", 'else: "declined"', "declined"),
        ("null", "!Optional(str) null", "else: null", None),
        ("", "!Optional(str) null", "else:", None),
        ("1.5", "!float 1.5", "else: 1.5", 1.5),
    ],
)
def test_an_else_value_shows_its_value_entry_and_the_entry_loads(else_, shown, needle, value):
    text = _flow(_on("reply", else_))
    error = _load_error(text, _value_branch("else", "route_else", shown))
    assert _error_line(error) == _line_of(text, needle)
    assert _taken(_rewritten(error), cmd="b") == value


def test_a_value_in_a_later_then_names_its_index():
    text = _flow(
        "    on: ${input.cmd}\n"
        "    cases:\n"
        "      - when: a\n"
        "        then: reply\n"
        "      - when: b\n"
        '        then: "bee"\n'
        "    else: reply\n"
    )
    error = _load_error(text, _value_branch("then", "route_then_1", "!str bee"))
    assert _error_line(error) == _line_of(text, 'then: "bee"')


def test_a_long_mapping_is_cut_and_written_as_a_block():
    text = _flow(
        _on(
            "\n          reply: ${input.cmd}\n          exited: true\n"
            "          note: a much longer note that goes on and on"
        )
    )
    error = _load_error(
        text,
        _value_branch(
            "then",
            "route_then_0",
            '!object {reply: "${input.cmd}", exited: true, note: a much longer no…',
            note=" (write it as a block)",
        ),
    )
    assert _error_line(error) == _line_of(text, "reply: ${input.cmd}")


def test_a_suggested_entry_id_skips_ids_already_taken():
    taken = (
        "  route_then_0: !str one\n"
        "  route_then_0_2: !str two\n"
    )
    text = _flow(_on('"approve"'), extra_nodes=taken)
    error = _load_error(text, _value_branch("then", "route_then_0_3", "!str approve"))
    assert _error_line(error) == _line_of(text, 'then: "approve"')
    assert _taken(_rewritten(error, extra_nodes=taken)) == "approve"


def test_the_old_typed_text_is_tagged_with_its_type():
    text = _flow(_on("int = ${input.n}"), inputs=_INPUTS_N)
    error = _load_error(text, _value_branch("then", "route_then_0", "!int ${input.n}"))
    assert _error_line(error) == _line_of(text, "then: int =")
    assert _taken(_rewritten(error, inputs=_INPUTS_N), n=3) == 3


@pytest.mark.parametrize(
    ("then", "shown", "value"),
    [
        ("${input.n}", "!int ${input.n}", 3),
        ("${count.output}", "!int ${count.output}", 7),
    ],
)
def test_a_lone_reference_is_tagged_with_its_type(then, shown, value):
    count = "  count: !int 7\n"
    text = _flow(_on(then), inputs=_INPUTS_N, extra_nodes=count)
    error = _load_error(text, _value_branch("then", "route_then_0", shown))
    assert _error_line(error) == _line_of(text, f"then: {then}")
    assert _taken(_rewritten(error, inputs=_INPUTS_N, extra_nodes=count), n=3) == value


@pytest.mark.parametrize(
    ("then", "output", "shown", "value"),
    [
        ("${input.n}", "Optional(int)", "!Optional(int) ${input.n}", 3),
        ("${input.cmd}", "int", "!str ${input.cmd}", "a"),
        ("${input.n}", "str", "!int ${input.n}", 3),
        ("${count.output}", "str", "!int ${count.output}", 7),
    ],
)
def test_a_lone_reference_takes_the_former_output_type_only_when_it_fits(
    then, output, shown, value
):
    count = "  count: !int 7\n"
    text = _flow(
        _on(then, output=f"    output: {output}\n"), inputs=_INPUTS_N, extra_nodes=count
    )
    error = _load_error(text, _value_branch("then", "route_then_0", shown))
    assert _error_line(error) == _line_of(text, f"then: {then}")
    assert _taken(_rewritten(error, inputs=_INPUTS_N, extra_nodes=count), n=3) == value


def test_a_lone_span_of_unknown_type_asks_for_its_type():
    text = _flow(_on("${input.n + 1}"), inputs=_INPUTS_N)
    error = _load_error(
        text,
        _value_branch(
            "then",
            "route_then_0",
            "!<type> ${input.n + 1}",
            note=" (name its type for `<type>`)",
        ),
    )
    assert _error_line(error) == _line_of(text, "then: ${input.n + 1}")
    rewritten = _rewritten(error, inputs=_INPUTS_N, type_text="int")
    assert _taken(rewritten, n=3) == 4


@pytest.mark.parametrize("then", ["[1, a]", "[[1], [a]]", "[{a: 1}, 2]"])
def test_a_list_of_mixed_items_says_no_tag_holds_it(then):
    text = _flow(_on(then))
    error = _load_error(
        text,
        _value_branch(
            "then",
            "route_then_0",
            f"!<type> {then}",
            note=" (no tag holds a list of mixed items; give its items one type)",
        ),
    )
    assert _error_line(error) == _line_of(text, f"then: {then}")


@pytest.mark.parametrize("then", ["&R [*R]", "&R [1, *R]", "&R {a: *R}"])
def test_a_value_that_contains_itself_is_rejected(then):
    text = _flow(_on(then))
    error = _load_error(text, "case node 'route' then: a branch value can't contain itself")
    assert _error_line(error) == _line_of(text, f"then: {then}")


@pytest.mark.parametrize(
    ("then", "shown"),
    [
        (".inf", ".inf"),
        ("-.inf", "-.inf"),
        (".nan", ".nan"),
        ("{a: .inf}", ".inf"),
        ("[1.5, -.inf]", "-.inf"),
        ("!float .inf", ".inf"),
    ],
)
def test_a_number_that_is_not_finite_is_rejected(then, shown):
    text = _flow(_on(then))
    error = _load_error(
        text, f"case node 'route' then: numbers must be finite (got {shown})"
    )
    assert _error_line(error) == _line_of(text, f"then: {then}")


@pytest.mark.parametrize(
    ("then", "shown", "value"),
    [
        ("!int 3", "!int 3", 3),
        ("!float 1e3", "!float 1e3", 1000.0),
        ("!str approve", "!str approve", "approve"),
        ('!date "2026-01-02"', '!date "2026-01-02"', "2026-01-02"),
        ("!int ${input.n}", "!int ${input.n}", 3),
        ("!Optional(int) null", "!Optional(int) null", None),
        ("!Buy {side: buy}", "!Buy {side: buy}", {"side": "buy"}),
        ("!list(int)\n          - 1\n          - 2", "!list(int) [1, 2]", [1, 2]),
        ("!str reply", "!str reply", "reply"),
    ],
)
def test_a_tagged_value_keeps_its_tag(then, shown, value):
    head = "typedefs:\n  Buy:\n    side: str\n"
    # The branch's own tag wins over the former `output:` type, and a tagged word that
    # names a node is text, as in a value entry.
    text = _flow(_on(then, output="    output: str\n"), head=head, inputs=_INPUTS_N)
    error = _load_error(text, _value_branch("then", "route_then_0", shown))
    assert _error_line(error) == _line_of(text, f"then: {then.splitlines()[0]}")
    assert _taken(_rewritten(error, head=head, inputs=_INPUTS_N), n=3) == value


def test_a_value_branch_is_tagged_with_the_former_output_type():
    text = _flow(_on("3", output="    output: str\n"))
    error = _load_error(text, _value_branch("then", "route_then_0", '!str "3"'))
    # The value branch is reported before the CASE's `output:`.
    assert _error_line(error) == _line_of(text, "then: 3")
    assert _taken(_rewritten(error)) == "3"


def test_a_value_branch_is_spelled_to_fit_the_former_output_type():
    text = _flow(_on('"yes"', output="    output: bool\n"))
    error = _load_error(
        text,
        _value_branch(
            "then",
            "route_then_0",
            "!bool true",
            note=" (`yes` is text; write `true`)",
        ),
    )
    assert _error_line(error) == _line_of(text, 'then: "yes"')
    assert _taken(_rewritten(error)) is True


@pytest.mark.parametrize(
    ("then", "output", "shown", "value"),
    [
        ('"3"', "int", '!str "3"', "3"),
        ('"v"', "int", "!str v", "v"),
        ("null", "str", "!Optional(str) null", None),
        ("{a: 1}", "Reply", "!object {a: 1}", {"a": 1}),
        ('"approve"', "Strr", "!str approve", "approve"),
    ],
)
def test_a_value_that_does_not_fit_the_former_output_type_guesses_its_tag(
    then, output, shown, value
):
    head = "typedefs:\n  Reply:\n    reply: str\n"
    text = _flow(_on(then, output=f"    output: {output}\n"), head=head)
    error = _load_error(text, _value_branch("then", "route_then_0", shown))
    assert _error_line(error) == _line_of(text, f"then: {then}")
    assert _taken(_rewritten(error, head=head)) == value


def test_a_value_branch_is_tagged_with_the_former_output_typedef():
    head = "typedefs:\n  Reply:\n    reply: str\n    exited: bool\n"
    text = _flow(
        _on("\n          reply: ok\n          exited: true", output="    output: Reply\n"),
        head=head,
    )
    error = _load_error(
        text, _value_branch("then", "route_then_0", "!Reply {reply: ok, exited: true}")
    )
    assert _error_line(error) == _line_of(text, "reply: ok")
    assert _taken(_rewritten(error, head=head)) == {"reply": "ok", "exited": True}


def test_a_value_branch_under_an_unspellable_output_guesses_its_tag():
    text = _flow(_on('"approve"', output="    output: Union(int, str)\n"))
    error = _load_error(text, _value_branch("then", "route_then_0", "!str approve"))
    assert _error_line(error) == _line_of(text, 'then: "approve"')
    assert _taken(_rewritten(error)) == "approve"


@pytest.mark.parametrize(
    ("then", "needle"),
    [
        ('{"": x}', 'then: {"": x}'),
        ('{a: {"": 1}}', "then: {a:"),
        ('!object {"": x}', "then: !object"),
    ],
)
def test_a_mapping_with_an_empty_key_is_rejected(then, needle):
    text = _flow(_on(then))
    error = _load_error(
        text,
        "case node 'route' then: a record key can't be empty; a written-out `!object` "
        "keeps its keys as record fields",
    )
    assert _error_line(error) == _line_of(text, needle)


def test_a_quoted_call_holding_a_span_is_its_own_error():
    text = _flow(_on('"call(take, x=${input.cmd})"'))
    error = _load_error(
        text,
        "case node 'route' then: a quoted `call(...)` is text, so it can't hold `${...}` "
        "here; drop the quotes to call the flow, or reword the text",
    )
    assert _error_line(error) == _line_of(text, 'then: "call(take')


@pytest.mark.parametrize(
    ("then", "needle"),
    [
        ("{a: call(child, x=1)}", "then: {a: call"),
        ('[a, "call(child)"]', "then: [a"),
    ],
)
def test_a_call_nested_in_a_value_is_rejected(then, needle):
    text = _flow(_on(then))
    error = _load_error(
        text,
        "case node 'route' then: a `call(...)` nested in a value is not supported; write "
        "the call as the branch (`then: call(...)`) or in a node, or reword the text",
    )
    assert _error_line(error) == _line_of(text, needle)


@pytest.mark.parametrize("then", ["${route.output}", '{a: "got ${route.output}"}'])
def test_a_value_reading_its_own_case_is_rejected(then):
    text = _flow(_on(then))
    error = _load_error(text, "case node 'route' then: the value reads 'route' itself")
    assert _error_line(error) == _line_of(text, f"then: {then}")


def test_a_malformed_span_in_a_value_is_rejected():
    text = _flow(_on('{a: "${input.cmd"}'))
    error = _load_error(
        text, "case node 'route' then: unbalanced '${' in template '${input.cmd'"
    )
    assert _error_line(error) == _line_of(text, "then: {a:")


@pytest.mark.parametrize(
    ("branch", "needle"),
    [
        ("then: {access_token: sk-1}", "then:"),
        ("then:\n          access_token: sk-1", "access_token: sk-1"),
    ],
)
def test_a_secret_in_a_value_branch_is_a_secret_literal(branch, needle):
    text = _flow(
        "    on: ${input.cmd}\n"
        "    cases:\n"
        "      - when: a\n"
        f"        {branch}\n"
        "    else: reply\n"
    )
    error = _load_error(
        text,
        "node 'route': then[0] value has key 'access_token' (a secret name) with a "
        "literal value; pass the value as an input or from the host",
    )
    assert "sk-1" not in str(error)
    assert _error_line(error) == _line_of(text, needle)


def test_a_key_that_is_no_secret_name_gets_the_branch_error():
    text = _flow(_on("{token: abc}"))
    error = _load_error(text, _value_branch("then", "route_then_0", "!object {token: abc}"))
    assert _error_line(error) == _line_of(text, "then: {token")


# --------------------------------------------------------------------------- #
# Unquoted words that name no node
# --------------------------------------------------------------------------- #


def test_an_unknown_then_word_suggests_the_nearest_node_and_its_text_entry():
    text = _flow(_on("replyy"))
    error = _load_error(
        text,
        "case node 'route' then: `replyy` names no node. Did you mean `reply`? For the "
        "text, add a value entry and name it: `replyy_text: !str replyy`",
    )
    assert _error_line(error) == _line_of(text, "then: replyy")
    assert _taken(_rewritten(error)) == "replyy"


def test_an_unknown_else_word_suggests_the_nearest_node_and_its_text_entry():
    text = _flow(_on("reply", "replyy"))
    error = _load_error(
        text,
        "case node 'route' else: `replyy` names no node. Did you mean `reply`? For the "
        "text, add a value entry and name it: `replyy_text: !str replyy`",
    )
    assert _error_line(error) == _line_of(text, "else: replyy")
    assert _taken(_rewritten(error), cmd="b") == "replyy"


def test_an_unknown_sentence_gets_no_suggestion_and_an_entry_named_after_the_branch():
    text = _flow(_on("The user declined"))
    error = _load_error(
        text,
        "case node 'route' then: `The user declined` names no node. For the text, add a "
        "value entry and name it: `route_then_0: !str The user declined`",
    )
    assert _error_line(error) == _line_of(text, "then: The user declined")
    assert _taken(_rewritten(error)) == "The user declined"


@pytest.mark.parametrize("output", ["", "    output: bool\n"])
def test_an_unknown_yes_gets_the_same_text_entry(output):
    text = _flow(_on("reply", "yes", output=output))
    error = _load_error(
        text,
        "case node 'route' else: `yes` names no node. For the text, add a value entry "
        "and name it: `yes_text: !str yes`",
    )
    assert _error_line(error) == _line_of(text, "else: yes")
    assert _taken(_rewritten(error), cmd="b") == "yes"


def test_an_unknown_word_entry_id_skips_an_id_already_taken():
    taken = "  nope_text: !str taken\n"
    text = _flow(_on("nope"), extra_nodes=taken)
    error = _load_error(
        text,
        "case node 'route' then: `nope` names no node. Did you mean `nope_text`? For the "
        "text, add a value entry and name it: `nope_text_2: !str nope`",
    )
    assert _error_line(error) == _line_of(text, "then: nope")
    assert _taken(_rewritten(error, extra_nodes=taken)) == "nope"


@pytest.mark.parametrize(
    ("then", "else_", "message", "needle", "cmd", "value"),
    [
        (
            "call(child, x=1) | x",
            "reply",
            "case node 'route' then: a branch target must be a node id or a single "
            "call: text follows the closing `)`: '| x'; a call(...) must be the whole "
            "value. For the text, add a value entry and name it: "
            '`route_then_0: !str "call(child, x=1) | x"`',
            "then: call(",
            "a",
            "call(child, x=1) | x",
        ),
        (
            "reply",
            "call(child, x=1) | y",
            "case node 'route' else: a branch target must be a node id or a single "
            "call: text follows the closing `)`: '| y'; a call(...) must be the whole "
            "value. For the text, add a value entry and name it: "
            '`route_else: !str "call(child, x=1) | y"`',
            "else: call(",
            "b",
            "call(child, x=1) | y",
        ),
    ],
)
def test_unquoted_call_that_is_not_one_directive_is_rejected(
    then, else_, message, needle, cmd, value
):
    text = _flow(_on(then, else_))
    error = _load_error(text, message)
    assert _error_line(error) == _line_of(text, needle)
    assert _taken(_rewritten(error), cmd=cmd) == value


# --------------------------------------------------------------------------- #
# A CASE `output:`
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("output", "example", "needle"),
    [
        ("output: str", ", e.g. `x: !str ${route.output}`", "output: str"),
        ("output: !str", ", e.g. `x: !str ${route.output}`", "output: !str"),
        ("output: !int", ", e.g. `x: !int ${route.output}`", "output: !int"),
        ("output: Optional(str)", ", e.g. `x: !Optional(str) ${route.output}`", "output:"),
        (
            "output: Union(int, str)",
            "; name `Union(int, str)` in `typedefs:` and tag with its name",
            "output: Union",
        ),
        ("output: Strr", ", e.g. `x: !Strr ${route.output}`", "output: Strr"),
        ("output:", ", e.g. `x: !<type> ${route.output}`", "output:"),
        (
            "output:\n      a: int",
            "; name `{a: int}` in `typedefs:` and tag with its name",
            "output:",
        ),
        (
            "output:\n      a: Strr",
            "; name `{a: Strr}` in `typedefs:` and tag with its name",
            "output:",
        ),
        (
            "output:\n      a: .inf",
            "; name `{a: .inf}` in `typedefs:` and tag with its name",
            "output:",
        ),
        ("output: &a [*a]", ", e.g. `x: !<type> ${route.output}`", "output: &a"),
        ("output: &a {x: *a}", ", e.g. `x: !<type> ${route.output}`", "output: &a"),
    ],
)
def test_a_case_output_is_rejected_at_its_line(output, example, needle):
    text = _flow(_on("reply", output=f"    {output}\n"))
    error = _load_error(
        text,
        "case node 'route': a `case` routes and has no `output:`; tag its result where "
        f"you read it{example}",
    )
    assert _error_line(error) == _line_of(text, needle)
