"""A literal under any name is an ordinary value: a name like `password`, `client_secret`
or `OPENAI_API_KEY` loads in every slot, and the literal is part of the manifest and
fingerprint.
"""

import json

import pytest

from yian.compile.manifest import flow_fingerprint, flow_manifest
from yian.compose import load_flow


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
