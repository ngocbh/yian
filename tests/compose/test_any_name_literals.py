"""A literal under any name is an ordinary value: a name like `password`, `client_secret`
or `OPENAI_API_KEY` loads in every slot, and the literal is part of the manifest and
fingerprint.
"""

import json

import pytest

from yian.compile.manifest import flow_fingerprint, flow_manifest
from yian.compose import LoadError, load_flow
from yian.run import run_flow


def _load(text: str):
    """Load one flow file text."""
    return load_flow(text)


def _assert_identity(text_a: str, text_b: str, literal_a: str) -> None:
    """Assert two flows that differ only in a literal have different fingerprints and
    that `literal_a` is in the first flow's manifest."""
    flow_a = _load(text_a).compiled
    flow_b = _load(text_b).compiled

    assert flow_fingerprint(flow_a) != flow_fingerprint(flow_b)
    assert literal_a in json.dumps(flow_manifest(flow_a))


def _node_env_flow(env_line: str) -> str:
    """Return a one-node flow whose `fetch` node has one `env:` line."""
    return (
        "id: f\nname: f\nnodes:\n  fetch:\n    kind: code\n    env:\n"
        f"      {env_line}\n"
        "    output: str\n    code: |\n      return 'x'\n"
        "output: ${fetch.output}\n"
    )


@pytest.mark.parametrize(
    "env_line",
    [
        "DB_PASSWORD: {v}",
        "password: {v}",
        "OPENAI_API_KEY: {v}",
        "db: {{password: {v}}}",
    ],
)
def test_a_node_env_literal_under_a_credential_name_loads(env_line):
    _assert_identity(
        _node_env_flow(env_line.format(v="hunter2")),
        _node_env_flow(env_line.format(v="hunter3")),
        "hunter2",
    )


def test_a_flow_env_literal_under_a_credential_name_loads():
    def flow(value: str) -> str:
        return (
            f"id: f\nname: f\nenv:\n  OPENAI_API_KEY: {value}\n"
            "nodes:\n  fetch:\n    kind: code\n    output: str\n"
            "    code: |\n      return 'x'\n"
            "output: ${fetch.output}\n"
        )

    _assert_identity(flow("hunter2"), flow("hunter3"), "hunter2")


_MULTI_NODE_DEF_ENV = (
    "id: f\nname: f\n"
    "defs:\n  g:\n    env:\n"
    "      secret: {v}\n"
    "    nodes:\n      run:\n        kind: code\n        output: str\n"
    "        code: |\n          return 'x'\n"
    "    output: ${{run.output}}\n"
    "nodes:\n  b:\n    kind: call\n    call: g\n"
    "output: ${{b.output}}\n"
)

_SINGLE_NODE_DEF_ENV = (
    "id: f\nname: f\n"
    "defs:\n  g:\n    kind: code\n    env:\n"
    "      password: {v}\n"
    "    output: str\n    code: |\n      return 'x'\n"
    "nodes:\n  b:\n    kind: call\n    call: g\n"
    "output: ${{b.output}}\n"
)

_COMPACT_FLOW_ENV = (
    "id: f\nname: f\nkind: code\nenv:\n"
    "  password: {v}\n"
    "output: str\ncode: |\n  return 'x'\n"
)


@pytest.mark.parametrize(
    "template",
    [_MULTI_NODE_DEF_ENV, _SINGLE_NODE_DEF_ENV, _COMPACT_FLOW_ENV],
    ids=["multi-node-def", "single-node-def", "compact-flow"],
)
def test_a_def_env_literal_under_a_credential_name_loads(template):
    _assert_identity(
        template.format(v="hunter2"), template.format(v="hunter3"), "hunter2"
    )


def _input_flow(declaration: str) -> str:
    """Return a one-node flow whose `input:` has one declaration line."""
    return (
        f"id: f\nname: f\ninput:\n  {declaration}\n"
        "nodes:\n  fetch:\n    kind: code\n    output: str\n"
        "    code: |\n      return 'x'\n"
        "output: ${fetch.output}\n"
    )


@pytest.mark.parametrize(
    "declaration", ["password: !str {v}", "conn: !object {{password: {v}}}"]
)
def test_an_input_default_under_a_credential_name_loads(declaration):
    _assert_identity(
        _input_flow(declaration.format(v="hunter2")),
        _input_flow(declaration.format(v="hunter3")),
        "hunter2",
    )


def test_a_def_input_default_under_a_credential_name_loads():
    def flow(value: str) -> str:
        return (
            "id: f\nname: f\n"
            "defs:\n  g:\n    input:\n"
            f"      password: !str {value}\n"
            "    nodes:\n      run:\n        kind: code\n        output: str\n"
            "        code: |\n          return 'x'\n"
            "    output: ${run.output}\n"
            "nodes:\n  b:\n    kind: call\n    call: g\n"
            "output: ${b.output}\n"
        )

    _assert_identity(flow("hunter2"), flow("hunter3"), "hunter2")


def test_an_agent_variable_under_a_credential_name_loads():
    def flow(value: str) -> str:
        return (
            "id: f\nname: f\nnodes:\n  writer:\n    kind: agent\n"
            f"    variables:\n      password: !str {value}\n"
            "    prompt: hi\n    output: str\n"
            "output: ${writer.output}\n"
        )

    _assert_identity(flow("hunter2"), flow("hunter3"), "hunter2")


def _node_input_flow(*input_lines: str) -> str:
    """Return a one-node flow whose `fetch` node has the given input lines."""
    return (
        "id: f\nname: f\nnodes:\n  fetch:\n    kind: code\n    input:\n"
        + "\n".join(f"      {line}" for line in input_lines)
        + "\n    output: str\n    code: |\n      return 'x'\n"
        "output: ${fetch.output}\n"
    )


@pytest.mark.parametrize(
    "name",
    [
        "password",
        "passwd",
        "secret",
        "client_secret",
        "private_key",
        "access_token",
        "refresh_token",
        "api_key",
        "OPENAI_API_KEY",
        "db_password",
        "clientSecret",
    ],
)
def test_a_node_input_literal_under_a_credential_name_is_identity(name):
    _assert_identity(
        _node_input_flow(f"{name}: hunter2"),
        _node_input_flow(f"{name}: hunter3"),
        "hunter2",
    )


def test_a_nested_key_under_a_credential_name_loads():
    _assert_identity(
        _node_input_flow("conn: {items: [{password: hunter2}]}"),
        _node_input_flow("conn: {items: [{password: hunter3}]}"),
        "hunter2",
    )


def test_a_typed_binding_under_a_credential_name_loads():
    _assert_identity(
        _node_input_flow("password: !str hunter2"),
        _node_input_flow("password: !str hunter3"),
        "hunter2",
    )


def test_a_map_over_literal_under_a_credential_name_loads():
    def flow(value: str) -> str:
        return (
            "id: p\nname: p\n"
            "defs:\n"
            "  leaf:\n"
            "    input:\n      x: !object\n"
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
            f"    over: [{{password: {value}}}, {{password: b}}]\n"
            "    call: leaf\n"
            "    input:\n      x: ${item}\n"
            "output: ${m.output}\n"
        )

    _assert_identity(flow("hunter2"), flow("hunter3"), "hunter2")


def test_a_flow_output_under_a_credential_name_is_identity():
    def flow(value: str) -> str:
        return (
            "id: f\nname: f\nnodes:\n  step:\n    kind: code\n    output: str\n"
            "    code: |\n      return 'x'\n"
            f"output:\n  step: ${{step.output}}\n  password: {value}\n"
        )

    _assert_identity(flow("hunter2"), flow("hunter3"), "hunter2")


def test_a_value_entry_under_a_credential_name_is_identity():
    def flow(value: str) -> str:
        return (
            f"id: f\nname: f\nnodes:\n  api_key: !str {value}\n"
            "output: ${api_key.output}\n"
        )

    _assert_identity(flow("hunter2"), flow("hunter3"), "hunter2")
    assert run_flow(_load(flow("hunter2")), {}).output == "hunter2"


_G_DEF = (
    "defs:\n"
    "  g:\n"
    "    input:\n"
    "      password: !Optional(str) null\n"
    "    nodes:\n"
    "      run:\n"
    "        kind: code\n"
    "        output: str\n"
    "        code: |\n"
    "          return 'x'\n"
    "    output: ${run.output}\n"
)


def _call_flow(binding: str) -> str:
    """Return a flow whose `n` node binds input `x` to `binding`, with def `g`."""
    return (
        "id: f\nname: f\n" + _G_DEF + "nodes:\n  n:\n    kind: code\n"
        f"    input:\n      x: {binding}\n    output: str\n"
        "    code: |\n      return 'x'\n"
        "output: ${n.output}\n"
    )


def test_a_call_argument_under_a_credential_name_is_identity():
    _assert_identity(
        _call_flow("call(g, password=hunter2)"),
        _call_flow("call(g, password=hunter3)"),
        "hunter2",
    )


def test_a_value_entry_call_argument_under_a_credential_name_loads():
    _load(
        "id: f\nname: f\n"
        "defs:\n"
        "  g:\n"
        "    input:\n"
        "      api_key: !str\n"
        "    nodes:\n"
        "      r: !str ok\n"
        "    output: ${r.output}\n"
        "nodes:\n"
        '  x: !str call(g, api_key="abc")\n'
        "output: ${x.output}\n"
    )


def test_a_case_call_target_with_a_credential_keyword_loads():
    def flow(value: str) -> str:
        return (
            "id: f\nname: f\ninput:\n  score: !float\n" + _G_DEF + "nodes:\n"
            "  c:\n    kind: case\n    cases:\n"
            "      - when: input.score >= 0.5\n"
            f"        then: call(g, password={value})\n"
            "    else: call(g)\n"
            "output: ${c.output}\n"
        )

    _assert_identity(flow("hunter2"), flow("hunter3"), "hunter2")


def test_an_inline_call_flow_output_with_a_credential_keyword_loads():
    def flow(value: str) -> str:
        return (
            "id: f\nname: f\n" + _G_DEF + "nodes:\n  step:\n    kind: code\n"
            "    output: str\n    code: |\n      return 'x'\n"
            f"output:\n  r: call(g, password={value})\n"
        )

    _assert_identity(flow("hunter2"), flow("hunter3"), "hunter2")


def test_a_malformed_call_with_a_credential_keyword_quotes_its_text():
    with pytest.raises(LoadError) as caught:
        _load(_call_flow("call(g, password=a b=1)"))

    assert str(caught.value) == (
        "n: call(g, ...) keyword 'password' value 'a b=1' looks like two "
        "keywords; add the missing comma, or quote the value. If you meant the text, "
        'quote the whole value: `"call(g, password=a b=1)"`'
    )


def test_an_inline_call_in_a_node_assert_shows_the_assert_under_any_keyword():
    text = (
        "id: f\nname: f\n" + _G_DEF + "nodes:\n  b:\n    kind: code\n"
        "    output: str\n    code: |\n      return 'x'\n"
        "    asserts:\n      - call(g, password=x)\n"
        "output: ${b.output}\n"
    )
    with pytest.raises(LoadError) as caught:
        _load(text)

    assert "assert 'call(g, password=x)'" in str(caught.value)
