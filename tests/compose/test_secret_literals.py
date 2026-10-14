"""A literal value under a secret name is a load error in bindings and flow outputs.

Every authored value is part of a flow's identity, so a credential must not be written
in the flow file: under a clearly secret name the value must be a `${...}` (or a
`call(...)` directive in a binding position) or `null`. The error names the slot and
the fix and never echoes the value.
"""

import pytest

from yian.compile.manifest import flow_fingerprint
from yian.compose import LoadError, load_flow

_SECRET = "hunter2"

_FLOW = """\
id: f
name: f
input:
  pw: str
  k: str
  s: str
  x: str
nodes:
  fetch:
    kind: code
    input:
{inputs}
    output: str
    code: |
      return "x"
output: ${{fetch.output}}
"""


def _flow(*input_lines: str) -> str:
    """Return a one-node flow whose `fetch` node has the given input lines."""
    return _FLOW.format(inputs="\n".join(f"      {line}" for line in input_lines))


def _output_flow(*output_lines: str) -> str:
    """Return a one-node flow whose `output:` mapping has the given lines."""
    return (
        "id: f\nname: f\ninput:\n  pw: str\n"
        "nodes:\n  fetch:\n    kind: code\n    output: str\n"
        "    code: |\n      return 'x'\n"
        "output:\n" + "\n".join(f"  {line}" for line in output_lines) + "\n"
    )


def _secret_error(text: str, **kwargs) -> LoadError:
    """Load `text`, asserting a secret-name error that does not echo the value."""
    with pytest.raises(LoadError) as exc_info:
        load_flow(text, **kwargs)
    message = str(exc_info.value)
    assert "secret name" in message, message
    assert _SECRET not in message, message
    assert "or from the host" in message, message
    return exc_info.value


def _only_range(error: LoadError):
    """Return the sole frame range of a single-site load error."""
    ranges = [frame.range for frame in error.diagnostic.frames]
    assert len(ranges) == 1
    return ranges[0]


_LISTED = [
    "password",
    "passwd",
    "secret",
    "client_secret",
    "private_key",
    "access_token",
    "refresh_token",
    "api_key",
    "openai_api_key",
    "db_password",
    "aws_secret",
    "clientSecret",
    "DB-PASSWORD",
]


@pytest.mark.parametrize("name", _LISTED)
def test_literal_node_input_under_a_secret_name_is_rejected(name):
    message = str(_secret_error(_flow(f"{name}: {_SECRET}")))

    assert f"node 'fetch': input '{name}' is a secret name" in message
    assert "${input." in message


def test_the_suggested_input_ref_is_an_identifier():
    assert "${input.clientSecret}" in str(_secret_error(_flow(f"clientSecret: {_SECRET}")))
    assert "${input.db_password}" in str(_secret_error(_flow(f"DB-PASSWORD: {_SECRET}")))


@pytest.mark.parametrize("name", _LISTED)
def test_a_reference_under_a_secret_name_loads(name):
    load_flow(_flow(f"{name}: ${{input.pw}}"))


@pytest.mark.parametrize(
    "line",
    [
        "api_key: 'Key ${input.k}'",
        "api_key: 'sk-abc${input.s}'",
        "password: null",
        'password: "${input.x} = y"',
    ],
)
def test_bindings_and_null_under_a_secret_name_load(line):
    load_flow(_flow(line))


@pytest.mark.parametrize(
    "value",
    ["''", "1234", "true", "'$${input.pw}'", f"[{_SECRET}]", "{a: b}", f"'{_SECRET}${{'"],
)
def test_every_literal_form_under_a_secret_name_is_rejected(value):
    _secret_error(_flow(f"password: {value}"))


def test_a_malformed_call_under_a_secret_name_is_the_secret_error():
    _secret_error(_flow(f"password: call(g, a={_SECRET}"))


def test_a_quoted_call_under_a_secret_name_is_literal_text():
    _secret_error(_flow(f"password: 'call(g, a={_SECRET})'"))


@pytest.mark.parametrize(
    "name", ["key", "user", "auth", "token", "sig", "signature", "page_token"]
)
def test_a_literal_under_an_ordinary_name_loads_and_is_fingerprinted(name):
    first = flow_fingerprint(load_flow(_flow(f"{name}: one")).compiled)
    second = flow_fingerprint(load_flow(_flow(f"{name}: two")).compiled)

    assert first != second


def test_a_nested_secret_key_is_rejected_with_its_path():
    message = str(
        _secret_error(_flow("conn:", "  host: db", f"  password: {_SECRET}"))
    )

    assert "node 'fetch': input 'conn' has key 'password' (a secret name)" in message
    assert "${input.conn}" in message


def test_a_secret_key_inside_a_list_is_rejected_with_its_path():
    message = str(_secret_error(_flow("items:", f"  - api_key: {_SECRET}")))

    assert "input 'items' has key 'api_key'" in message
    assert "items[0].api_key" in message


def test_a_deeper_secret_key_renders_its_whole_path():
    message = str(
        _secret_error(
            _flow("conn:", "  items:", "    - host: db", f"      password: {_SECRET}")
        )
    )

    assert "conn.items[0].password" in message


def test_a_nested_reference_under_a_secret_key_loads():
    load_flow(_flow("conn:", "  password: ${input.pw}"))


def test_a_mapping_with_an_int_key_is_a_non_text_key_error():
    with pytest.raises(LoadError, match="the key `1` was read by YAML as an integer"):
        load_flow(_flow("conn:", "  1: x"))


def test_a_numeric_text_mapping_key_on_the_path_is_shown_as_a_key():
    message = str(_secret_error(_flow("conn:", "  '1':", f"    password: {_SECRET}")))

    assert "conn.1.password" in message


def test_a_recursive_alias_under_an_ordinary_name_does_not_crash():
    try:
        load_flow(_flow("conn: &x", "  host: db", "  self: *x"))
    except LoadError as exc:
        assert "secret name" not in str(exc)


def test_a_recursive_alias_holding_a_secret_key_is_rejected():
    _secret_error(_flow("conn: &x", f"  password: {_SECRET}", "  self: *x"))


@pytest.mark.parametrize("type_text", ["int", "str"])
def test_a_typed_literal_under_a_secret_name_is_rejected(type_text):
    _secret_error(_flow(f"password: {type_text} = {_SECRET}"))


def test_a_typed_literal_with_a_bad_type_is_the_secret_error():
    _secret_error(_flow(f"password: foo = {_SECRET}"))


def test_a_typed_reference_under_a_secret_name_loads():
    load_flow(_flow("password: str = ${input.pw}"))


def test_a_typed_collection_holding_a_secret_key_is_rejected():
    message = str(_secret_error(_flow(f'conn: dict[str, str] = {{"password":"{_SECRET}"}}')))

    assert "input 'conn' has key 'password'" in message


def test_quoted_typed_looking_text_under_an_ordinary_name_loads():
    load_flow(_flow(f"conn: 'dict = {{password: {_SECRET}}}'"))


@pytest.mark.parametrize(
    "over",
    [f"    over:\n      - api_key: {_SECRET}\n", f"    over:\n      api_key: {_SECRET}\n"],
    ids=["list", "mapping"],
)
def test_a_literal_map_over_holding_a_secret_key_is_rejected(over):
    text = (
        "id: p\nname: p\n"
        "defs:\n"
        "  leaf:\n"
        "    input:\n      x: object\n"
        "    nodes:\n"
        "      up:\n"
        "        kind: code\n"
        "        input:\n          x: ${input.x}\n"
        "        output: str\n"
        "        code: return 'x'\n"
        "    output: ${up.output}\n"
        "nodes:\n"
        "  m:\n"
        "    kind: map\n"
        + over
        + "    call: leaf\n"
        "    input:\n      x: ${item}\n"
        "output: ${m.output}\n"
    )
    message = str(_secret_error(text))

    assert "node 'm': field 'over' has key 'api_key'" in message


@pytest.mark.parametrize(
    "line", [f"password: {_SECRET}", f"password: str = {_SECRET}"]
)
def test_a_literal_flow_output_under_a_secret_name_is_rejected(line):
    message = str(_secret_error(_output_flow(line)))

    assert "flow output 'password' is a secret name" in message
    assert "${input.password}" in message


def test_a_reference_flow_output_under_a_secret_name_loads():
    load_flow(_output_flow("password: ${input.pw}"))


def test_the_node_input_error_is_located_at_its_key_line():
    text = _flow("keep: ${input.x}", f"password: {_SECRET}")
    line = text.splitlines().index(f"      password: {_SECRET}") + 1

    assert _only_range(_secret_error(text)).start_line == line


def test_the_flow_output_error_is_located_at_its_key_line():
    text = _output_flow("keep: ${input.pw}", f"password: {_SECRET}")
    line = text.splitlines().index(f"  password: {_SECRET}") + 1

    assert _only_range(_secret_error(text)).start_line == line


def test_a_uses_library_with_a_literal_secret_fails_when_imported(tmp_path):
    (tmp_path / "lib.yaml").write_text(_flow(f"password: {_SECRET}"))
    parent = (
        "id: parent\nname: parent\n"
        "uses:\n  lib: lib\n"
        "nodes:\n"
        "  run:\n"
        "    kind: call\n"
        "    call: lib\n"
        "    input:\n"
        "      pw: ${input.pw}\n      k: ${input.pw}\n"
        "      s: ${input.pw}\n      x: ${input.pw}\n"
        "input:\n  pw: str\n"
        "output: ${run.output}\n"
    )

    _secret_error(parent, search_paths=[tmp_path])


_CASE_FLOW = """\
id: f
name: f
input:
  pw: str
  score: float
nodes:
  pick:
    kind: case
    output: dict[str, str]
    cases:
      - when: input.score >= 0.5
        then:
          password: {then}
    else:
      password: {else_}
output: ${{pick.output}}
"""


def _case_flow(then: str = "${input.pw}", else_: str = "${input.pw}") -> str:
    """Return a flow whose typed `pick` case has mapping value branches."""
    return _CASE_FLOW.format(then=then, else_=else_)


def test_a_literal_secret_in_a_case_then_value_is_rejected():
    text = _case_flow(then=_SECRET)
    error = _secret_error(text)
    message = str(error)

    assert (
        "node 'pick': then[0] value has key 'password' (a secret name) with a literal "
        "value; pass the value as an input or from the host"
    ) in message
    # The branch value's own range: a block mapping starts at its first key.
    line = text.splitlines().index(f"          password: {_SECRET}") + 1
    assert _only_range(error).start_line == line


def test_a_literal_secret_in_a_case_else_value_is_rejected():
    text = _case_flow(else_=_SECRET)
    error = _secret_error(text)

    assert "node 'pick': else value has key 'password'" in str(error)
    line = text.splitlines().index(f"      password: {_SECRET}") + 1
    assert _only_range(error).start_line == line


def test_a_reference_in_a_case_value_under_a_secret_key_loads():
    load_flow(_case_flow())


_G_DEF = """\
defs:
  g:
    input:
      password: Optional[str] = null
      api_key: Optional[str] = null
      x: Optional[int] = null
    nodes:
      run:
        kind: code
        output: str
        code: |
          return "x"
    output: ${run.output}
"""


def _call_flow(binding: str) -> str:
    """Return a flow whose `b` node binds input `x` to `binding`."""
    return (
        "id: f\nname: f\ninput:\n  pw: str\n" + _G_DEF + "nodes:\n  b:\n    kind: code\n"
        f"    input:\n      x: {binding}\n    output: str\n"
        "    code: |\n      return 'x'\n"
        "output: ${b.output}\n"
    )


def test_a_literal_secret_call_argument_is_rejected():
    message = str(_secret_error(_call_flow(f"call(g, password='{_SECRET}')")))

    assert (
        "node 'b': call(...) argument 'password' is a secret name, so its value can't "
        "be written in the flow file; pass it as an input (${input.password}) or from "
        "the host"
    ) in message
    assert "__call_" not in message
    assert "quote" not in message


@pytest.mark.parametrize(
    "binding",
    [
        "call(g, password=${input.pw})",
        "call(g, password='${input.pw}')",
        "call(g, password=null)",
        "call(g, x=1, password=call(g, x=2))",
    ],
)
def test_a_binding_call_argument_under_a_secret_name_loads(binding):
    load_flow(_call_flow(binding))


@pytest.mark.parametrize(
    "binding",
    [
        f"call(g, password={_SECRET} x=1)",
        f"call(g, x=1 password={_SECRET})",
        f"call(g, x=call(g, password={_SECRET}))",
        f"call(g, password=1, password={_SECRET})",
    ],
)
def test_malformed_calls_with_a_literal_secret_argument_do_not_echo_it(binding):
    message = str(_secret_error(_call_flow(binding)))

    assert "call(...) argument 'password' is a secret name" in message


def test_a_secret_argument_inside_a_secret_named_call_is_the_argument_error():
    text = _flow(f"password: call(g, password='{_SECRET}')").replace(
        "nodes:\n", _G_DEF + "nodes:\n", 1
    )
    message = str(_secret_error(text))

    assert "node 'fetch': call(...) argument 'password' is a secret name" in message


def test_a_literal_secret_call_argument_in_a_flow_output_is_rejected():
    text = (
        "id: f\nname: f\n" + _G_DEF + "nodes:\n  fetch:\n    kind: code\n"
        "    output: str\n    code: |\n      return 'x'\n"
        f"output:\n  r: call(g, api_key='{_SECRET}')\n"
    )
    message = str(_secret_error(text))

    assert "flow outputs: call(...) argument 'api_key' is a secret name" in message


def test_a_literal_secret_in_a_case_call_target_has_a_clean_label():
    text = (
        "id: f\nname: f\ninput:\n  score: float\n" + _G_DEF + "nodes:\n"
        "  c:\n    kind: case\n    cases:\n"
        '      - when: "${input.score} >= 0.5"\n'
        f"        then: call(g, password='{_SECRET}')\n"
        "    else: call(g, x=1)\n"
        "output: ${c.output}\n"
    )
    message = str(_secret_error(text))

    assert "case node 'c' then: call(...) argument 'password' is a secret name" in message
    assert "::" not in message


@pytest.mark.parametrize(
    "binding",
    [
        f"call(g, password=call(g, x=1) {_SECRET})",
        f"call(g, password=call(g x=1, y={_SECRET}))",
    ],
)
def test_a_nested_call_under_a_secret_keyword_that_does_not_parse_is_rejected(binding):
    message = str(_secret_error(_call_flow(binding)))

    assert "call(...) argument 'password' is a secret name" in message


def _node_assert_flow(assertion: str) -> str:
    """Return a flow whose `b` node has one node-local assert."""
    return (
        "id: f\nname: f\ninput:\n  pw: str\n" + _G_DEF + "nodes:\n  b:\n    kind: code\n"
        "    output: str\n    code: |\n      return 'x'\n"
        f"    asserts:\n      - {assertion}\n"
        "output: ${b.output}\n"
    )


@pytest.mark.parametrize(
    "assertion", [f"call(g, password={_SECRET})", f"call(g, password='{_SECRET}')"]
)
def test_a_literal_secret_call_argument_in_a_node_assert_is_rejected(assertion):
    message = str(_secret_error(_node_assert_flow(assertion)))

    assert "node 'b' assert: call(...) argument 'password' is a secret name" in message


def test_a_malformed_call_in_a_node_assert_with_a_secret_keyword_is_not_echoed():
    with pytest.raises(LoadError) as exc_info:
        load_flow(_node_assert_flow(f"call(g password={_SECRET})"))
    message = str(exc_info.value)

    assert "an inline call is not allowed in a node-local assert" in message
    assert _SECRET not in message, message


def _syntax_error(text: str) -> str:
    """Load `text`, asserting a load error that is not the secret-name error and does
    not echo the secret."""
    with pytest.raises(LoadError) as exc_info:
        load_flow(text)
    message = str(exc_info.value)
    assert "secret name" not in message, message
    assert _SECRET not in message, message
    assert "quote the whole value" not in message, message
    return message


@pytest.mark.parametrize(
    ("binding", "problem"),
    [
        (
            f"call(g password={_SECRET})",
            "a call(...) must start with a flow id (a bare name) and a comma",
        ),
        (f"call(g, 30 password={_SECRET})", "needs one name before `=`"),
        (f"call(g, my password={_SECRET})", "needs one name before `=`"),
        (f"call(g, password={_SECRET}", "`call(` is never closed"),
        (f"call(g, x=1) password={_SECRET}", "text follows the closing `)`"),
        (f"call(g, password=call(g, x=1 {_SECRET})", "`call(` is never closed"),
    ],
)
def test_a_malformed_call_with_a_secret_keyword_does_not_quote_its_text(
    binding, problem
):
    assert problem in _syntax_error(_call_flow(binding))


def test_a_malformed_case_call_with_a_secret_keyword_does_not_quote_its_text():
    text = (
        "id: f\nname: f\ninput:\n  score: float\n" + _G_DEF + "nodes:\n"
        "  c:\n    kind: case\n    cases:\n"
        '      - when: "${input.score} >= 0.5"\n'
        f"        then: call(g password={_SECRET})\n"
        "    else: call(g, x=1)\n"
        "output: ${c.output}\n"
    )
    message = _syntax_error(text)

    assert "a branch target must be a node id or a single call" in message
    assert "quote it" not in message


def test_a_missing_comma_before_a_secret_binding_is_the_two_keywords_error():
    message = _syntax_error(_call_flow("call(g, x=1 password=${input.pw})"))

    assert "keyword 'x' value looks like two keywords; add the missing comma" in message


def _input_flow(*declarations: str) -> str:
    """Return a one-node flow whose `input:` has the given declaration lines."""
    return (
        "id: f\nname: f\ninput:\n"
        + "\n".join(f"  {line}" for line in declarations)
        + "\nnodes:\n  fetch:\n    kind: code\n    output: str\n"
        "    code: |\n      return 'x'\n"
        "output: ${fetch.output}\n"
    )


def _default_error(text: str) -> str:
    """Load `text`, asserting a secret default error that does not echo the value."""
    with pytest.raises(LoadError) as exc_info:
        load_flow(text)
    message = str(exc_info.value)
    assert "secret name" in message, message
    assert _SECRET not in message, message
    return message


@pytest.mark.parametrize(
    "declaration",
    [
        f"password: str = '{_SECRET}'",
        f"password: str = {_SECRET}",
        f"password: \"str = '{_SECRET}\"",
        "password: int = 1234",
        "password: str = ''",
        f"api_key: list[str] = [{_SECRET}]",
    ],
)
def test_a_secret_named_input_default_is_rejected(declaration):
    name = declaration.split(":", 1)[0]
    message = _default_error(_input_flow(declaration))

    assert (
        f"input '{name}': a secret name can't have a default value in the flow file; "
        "drop the default and pass the value when the flow runs"
    ) in message


@pytest.mark.parametrize(
    "declaration",
    [
        "api_key: Optional[str] = null",
        "api_key: Optional[str] = ~",
        "primary_password: str",
        "password: Optional[str]",
        "password: str = ${input.other}",
    ],
)
def test_a_secret_named_input_without_a_literal_default_loads(declaration):
    load_flow(_input_flow(declaration))


def test_a_nested_secret_key_in_an_input_default_is_rejected():
    message = _default_error(
        _input_flow(f"conn: 'dict[str, str] = {{\"password\": \"{_SECRET}\"}}'")
    )

    assert "input 'conn': default has key 'password' (a secret name)" in message
    assert "${input.conn}" in message


def test_the_secret_default_error_is_located_at_its_declaration():
    text = _input_flow("keep: str", f"password: str = {_SECRET}")
    with pytest.raises(LoadError) as exc_info:
        load_flow(text)
    line = text.splitlines().index(f"  password: str = {_SECRET}") + 1

    assert exc_info.value.diagnostic.frames[-1].range.start_line == line


def test_a_secret_named_def_input_default_is_rejected():
    text = (
        "id: f\nname: f\n"
        "defs:\n  g:\n    input:\n"
        f"      password: str = {_SECRET}\n"
        "    nodes:\n      run:\n        kind: code\n        output: str\n"
        "        code: |\n          return 'x'\n"
        "    output: ${run.output}\n"
        "nodes:\n  b:\n    kind: call\n    call: g\n"
        "output: ${b.output}\n"
    )

    assert "input 'password': a secret name can't have a default" in _default_error(text)


def test_a_secret_named_agent_variable_is_rejected():
    text = (
        "id: f\nname: f\nnodes:\n  writer:\n    kind: agent\n"
        f"    variables:\n      password: str = {_SECRET}\n"
        "    prompt: hi\n    output: str\n"
        "output: ${writer.output}\n"
    )

    assert (
        "variable 'password': a secret name can't be a variable, whose value is written "
        "in the flow file; pass it as a node input (${input.password}) instead"
    ) in _default_error(text)


def _env_error(text: str) -> LoadError:
    """Load `text`, asserting a secret env error that does not echo the value."""
    with pytest.raises(LoadError) as exc_info:
        load_flow(text)
    message = str(exc_info.value)
    assert "is a secret name" in message or "(a secret name)" in message, message
    assert _SECRET not in message, message
    assert "env holds static config written in the flow file" in message, message
    assert "or have the host supply it" in message, message
    return exc_info.value


def _node_env_flow(*env_lines: str) -> str:
    """Return a one-node flow whose `fetch` node has the given `env:` lines."""
    return (
        "id: f\nname: f\nnodes:\n  fetch:\n    kind: code\n    env:\n"
        + "\n".join(f"      {line}" for line in env_lines)
        + "\n    output: str\n    code: |\n      return 'x'\n"
        "output: ${fetch.output}\n"
    )


def _flow_env_flow(*env_lines: str) -> str:
    """Return a one-node flow whose flow-level `env:` has the given lines."""
    return (
        "id: f\nname: f\nenv:\n"
        + "\n".join(f"  {line}" for line in env_lines)
        + "\nnodes:\n  fetch:\n    kind: code\n    output: str\n"
        "    code: |\n      return 'x'\n"
        "output: ${fetch.output}\n"
    )


def test_a_literal_secret_in_node_env_is_rejected():
    text = _node_env_flow("region: eu", f"DB_PASSWORD: {_SECRET}")
    error = _env_error(text)

    assert (
        "node 'fetch': env 'DB_PASSWORD' is a secret name; env holds static config "
        "written in the flow file (`${...}` is not evaluated there), so move it to the "
        "node's input: (${input.DB_PASSWORD}) or have the host supply it"
    ) in str(error)
    line = text.splitlines().index(f"      DB_PASSWORD: {_SECRET}") + 1
    assert _only_range(error).start_line == line


def test_a_literal_secret_in_flow_env_is_rejected():
    text = _flow_env_flow("region: eu", f"OPENAI_API_KEY: {_SECRET}")
    error = _env_error(text)

    assert "flow env 'OPENAI_API_KEY' is a secret name" in str(error)
    line = text.splitlines().index(f"  OPENAI_API_KEY: {_SECRET}") + 1
    assert _only_range(error).start_line == line


def test_a_nested_literal_secret_in_env_is_rejected():
    message = str(_env_error(_node_env_flow("db:", f"  password: {_SECRET}")))

    assert "node 'fetch': env 'db' has key 'password' (a secret name)" in message


def test_a_literal_secret_in_multi_node_def_env_is_rejected():
    text = (
        "id: f\nname: f\n"
        "defs:\n  g:\n    env:\n"
        f"      secret: {_SECRET}\n"
        "    nodes:\n      run:\n        kind: code\n        output: str\n"
        "        code: |\n          return 'x'\n"
        "    output: ${run.output}\n"
        "nodes:\n  b:\n    kind: call\n    call: g\n"
        "output: ${b.output}\n"
    )
    message = str(_env_error(text))

    assert "defs entry 'g': env 'secret' is a secret name" in message
    assert message.count("defs entry 'g'") == 1


def test_a_literal_secret_in_single_node_def_env_is_rejected():
    text = (
        "id: f\nname: f\n"
        "defs:\n  g:\n    kind: code\n    env:\n"
        f"      password: {_SECRET}\n"
        "    output: str\n    code: |\n      return 'x'\n"
        "nodes:\n  b:\n    kind: call\n    call: g\n"
        "output: ${b.output}\n"
    )

    assert "env 'password' is a secret name" in str(_env_error(text))


def test_a_literal_secret_in_compact_flow_env_is_rejected():
    text = (
        "id: f\nname: f\nkind: code\nenv:\n"
        f"  password: {_SECRET}\n"
        "output: str\ncode: |\n  return 'x'\n"
    )

    assert "env 'password' is a secret name" in str(_env_error(text))


@pytest.mark.parametrize(
    "lines",
    [
        ("region: eu", "max_tool_iterations: 5"),
        ("API_KEY: null",),
        ("API_KEY: ${input.k}",),
    ],
)
def test_env_without_a_literal_secret_loads(lines):
    load_flow(_node_env_flow(*lines))
    load_flow(_flow_env_flow(*lines))
