"""A CASE routes only: a branch names a node or calls a flow, and a CASE has no `output:`.

An unquoted word names a node and an unquoted whole-value `call(...)` calls a flow.
Every other branch is a value: quoted text, `null` or an empty `else:`, a number, a bool,
a date, a mapping, a list, or unquoted text holding `${`. A value branch is a located
load error that shows the value entry to write and the branch naming it. The entry is
tagged with the CASE's former `output:` type when a tag can spell it, else with a tag
guessed from the YAML value. An unquoted word that names no node suggests the nearest
node and the value entry for the text. A CASE `output:`, tagged or not, is a load error
at the `output:` that says to tag the result where it is read. A secret in a value
branch is reported as a secret literal, never echoed. `${route.output}` reads the taken
branch.
"""

import re

import pytest

from yian.compose import LoadError, load_flow
from yian.run import run_flow


def _line_of(text: str, needle: str) -> int:
    """Return the 1-based line number of the first line containing `needle`."""
    return next(
        number
        for number, line in enumerate(text.splitlines(), start=1)
        if needle in line
    )


def _load_error(text: str, message: str) -> LoadError:
    """Load `text`, expecting a load error whose whole message is `message`."""
    with pytest.raises(LoadError, match=re.escape(message) + "$") as caught:
        load_flow(text)
    return caught.value


def _error_line(error: LoadError) -> int:
    """Return the start line of the innermost frame of a load error."""
    frame = error.diagnostic.frames[-1]
    assert frame.range is not None
    return frame.range.start_line


def _flow(case_body: str, *, head: str = "", extra_nodes: str = "") -> str:
    """Return a flow with a `route` case (indented body) and a `reply` node it reads."""
    return (
        "id: f\n"
        "name: f\n"
        f"{head}"
        "input:\n"
        "  cmd: !str\n"
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
    ("then", "shown", "needle"),
    [
        ('"approve"', "!str approve", 'then: "approve"'),
        ("'approve'", "!str approve", "then: 'approve'"),
        ('"reply"', "!str reply", 'then: "reply"'),
        ('"it\'s: here"', "!str \"it's: here\"", "then: \"it's"),
        ("|\n          two\n          lines", '!str "two\\nlines\\n"', "then: |"),
        (">\n          one\n          line", '!str "one line\\n"', "then: >"),
        ("3", "!int 3", "then: 3"),
        ("1e3", "!float 1e3", "then: 1e3"),
        ("true", "!bool true", "then: true"),
        ("2026-01-02", '!date "2026-01-02"', "then: 2026-01-02"),
        ("null", "!Optional(str) null", "then: null"),
        (
            "\n          reply: ${input.cmd}\n          exited: true",
            '!object {reply: "${input.cmd}", exited: true}',
            "reply: ${input.cmd}",
        ),
        ('[a, "${input.cmd}"]', '!list(str) [a, "${input.cmd}"]', "then: [a"),
        ("hi ${input.cmd}", "!str hi ${input.cmd}", "then: hi"),
        ("${ take(x) }", "!str ${ take(x) }", "then: ${ take"),
        ('"call(take"', '!str "call(take"', 'then: "call(take"'),
    ],
)
def test_a_then_value_shows_its_value_entry(then, shown, needle):
    text = _flow(_on(then))
    error = _load_error(text, _value_branch("then", "route_then_0", shown))
    assert _error_line(error) == _line_of(text, needle)


@pytest.mark.parametrize(
    ("else_", "shown", "needle"),
    [
        ('"declined"', "!str declined", 'else: "declined"'),
        ("null", "!Optional(str) null", "else: null"),
        ("", "!Optional(str) null", "else:"),
        ("1.5", "!float 1.5", "else: 1.5"),
    ],
)
def test_an_else_value_shows_its_value_entry(else_, shown, needle):
    text = _flow(_on("reply", else_))
    error = _load_error(text, _value_branch("else", "route_else", shown))
    assert _error_line(error) == _line_of(text, needle)


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
    _load_error(
        text,
        _value_branch(
            "then",
            "route_then_0",
            '!object {reply: "${input.cmd}", exited: true, note: a much longer no…',
            note=" (write it as a block)",
        ),
    )


def test_a_suggested_entry_id_skips_ids_already_taken():
    taken = (
        "  route_then_0: !str one\n"
        "  route_then_0_2: !str two\n"
    )
    text = _flow(_on('"approve"'), extra_nodes=taken)
    _load_error(text, _value_branch("then", "route_then_0_3", "!str approve"))


def test_the_old_typed_text_is_tagged_with_its_type():
    text = _flow(_on("int = ${input.cmd}"))
    _load_error(text, _value_branch("then", "route_then_0", "!int ${input.cmd}"))


def test_a_value_branch_is_tagged_with_the_former_output_type():
    text = _flow(_on("3", output="    output: str\n"))
    error = _load_error(text, _value_branch("then", "route_then_0", '!str "3"'))
    # The value branch is reported before the CASE's `output:`.
    assert _error_line(error) == _line_of(text, "then: 3")


def test_a_value_branch_is_tagged_with_the_former_output_typedef():
    text = _flow(
        _on("\n          reply: ok\n          exited: true", output="    output: Reply\n"),
        head="typedefs:\n  Reply:\n    reply: str\n    exited: bool\n",
    )
    _load_error(
        text, _value_branch("then", "route_then_0", "!Reply {reply: ok, exited: true}")
    )


def test_a_value_branch_under_an_unspellable_output_guesses_its_tag():
    text = _flow(_on('"approve"', output="    output: Union(int, str)\n"))
    _load_error(text, _value_branch("then", "route_then_0", "!str approve"))


def test_a_quoted_call_holding_a_span_is_its_own_error():
    text = _flow(_on('"call(take, x=${input.cmd})"'))
    _load_error(
        text,
        "case node 'route' then: a quoted `call(...)` is text, so it can't hold `${...}` "
        "here; drop the quotes to call the flow, or reword the text",
    )


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
    _load_error(text, _value_branch("then", "route_then_0", "!object {token: abc}"))


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


def test_an_unknown_else_word_suggests_the_nearest_node_and_its_text_entry():
    text = _flow(_on("reply", "replyy"))
    error = _load_error(
        text,
        "case node 'route' else: `replyy` names no node. Did you mean `reply`? For the "
        "text, add a value entry and name it: `replyy_text: !str replyy`",
    )
    assert _error_line(error) == _line_of(text, "else: replyy")


def test_an_unknown_sentence_gets_no_suggestion_and_an_entry_named_after_the_branch():
    text = _flow(_on("The user declined"))
    _load_error(
        text,
        "case node 'route' then: `The user declined` names no node. For the text, add a "
        "value entry and name it: `route_then_0: !str The user declined`",
    )


@pytest.mark.parametrize("output", ["", "    output: bool\n"])
def test_an_unknown_yes_gets_the_same_text_entry(output):
    text = _flow(_on("reply", "yes", output=output))
    error = _load_error(
        text,
        "case node 'route' else: `yes` names no node. For the text, add a value entry "
        "and name it: `yes_text: !str yes`",
    )
    assert _error_line(error) == _line_of(text, "else: yes")


def test_an_unknown_word_entry_id_skips_an_id_already_taken():
    text = _flow(_on("nope"), extra_nodes="  nope_text: !str taken\n")
    _load_error(
        text,
        "case node 'route' then: `nope` names no node. Did you mean `nope_text`? For the "
        "text, add a value entry and name it: `nope_text_2: !str nope`",
    )


@pytest.mark.parametrize(
    ("then", "else_", "label", "entry", "needle"),
    [
        ("call(child, x=1) | x", "reply", "then", "route_then_0", "then: call("),
        ("reply", "call(child, x=1) | y", "else", "route_else", "else: call("),
    ],
)
def test_unquoted_call_that_is_not_one_directive_is_rejected(
    then, else_, label, entry, needle
):
    text = _flow(_on(then, else_))
    tail = then if label == "then" else else_
    error = _load_error(
        text,
        f"case node 'route' {label}: a branch target must be a node id or a single call: "
        f"text follows the closing `)`: '{tail.split(') ')[1]}'; a call(...) must be the "
        f"whole value. For the text, add a value entry and name it: "
        f'`{entry}: !str "{tail}"`',
    )
    assert _error_line(error) == _line_of(text, needle)


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
        (
            "output:\n      a: int",
            "; name `{a: int}` in `typedefs:` and tag with its name",
            "output:",
        ),
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
