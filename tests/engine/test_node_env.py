"""Per-node `env:` — parse + build-time merge (flow default + node override, node wins).

`env:` is a static config mapping the compiler bakes onto `node.env` at build. It is
engine-opaque (only a node's own `run()` reads known keys). The merge is `{**flow_env,
**node_env}` (node wins), flow-local (a child flow does not inherit the parent's env).
"""

import pytest

from yian.compose import load_flow
from yian.compose.helpers.errors import LoadError


def _nodes(text):
    return load_flow(text).compiled.nodes


def test_node_env_overrides_flow_env():
    nodes = _nodes(
        """
id: f
name: f
env:
  max_tool_iterations: 50
  shared: from_flow
nodes:
  a:
    kind: agent
    prompt: hi
    env:
      max_tool_iterations: 300
  b:
    kind: agent
    depends_on: [a]
    prompt: hi
output: ${b.output}
"""
    )
    # node override wins on its key, still inherits the flow-only key
    assert nodes["a"].env == {"max_tool_iterations": 300, "shared": "from_flow"}
    # no node env -> pure flow default
    assert nodes["b"].env == {"max_tool_iterations": 50, "shared": "from_flow"}


def test_no_env_is_empty_dict():
    nodes = _nodes(
        """
id: f
name: f
nodes:
  a: {kind: agent, prompt: hi}
output: ${a.output}
"""
    )
    assert nodes["a"].env == {}


def test_env_must_be_string_keyed_mapping():
    with pytest.raises(LoadError, match="env"):
        load_flow(
            """
id: f
name: f
nodes:
  a:
    kind: agent
    prompt: hi
    env: [1, 2, 3]
output: ${a.output}
"""
        )


_DEF_ENV_FLOW = """
id: f
name: f
env:
  shared: from_parent
  parent_only: yes_parent
defs:
  child:
    env:
      shared: from_def
      max_tool_iterations: 7
    nodes:
      a:
        kind: agent
        prompt: hi
      b:
        kind: agent
        depends_on: [a]
        prompt: hi
        env:
          max_tool_iterations: 9
    output: ${b.output}
nodes:
  go:
    kind: call
    call: child
output: ${go.output}
"""


def test_def_child_has_its_own_flow_env():
    child_nodes = load_flow(_DEF_ENV_FLOW).compiled.nodes["go"].child.nodes
    # the def's env is the default for its nodes; the parent's env does not reach them
    assert child_nodes["a"].env == {"shared": "from_def", "max_tool_iterations": 7}
    # a def node's own env still wins on its key
    assert child_nodes["b"].env == {"shared": "from_def", "max_tool_iterations": 9}


def test_def_env_must_be_string_keyed_mapping():
    with pytest.raises(LoadError, match=r"defs entry 'child': `env:` must be a mapping"):
        load_flow(
            """
id: f
name: f
defs:
  child:
    env: [1, 2, 3]
    nodes:
      a: {kind: agent, prompt: hi}
    output: ${a.output}
nodes:
  go: {kind: call, call: child}
output: ${go.output}
"""
        )
