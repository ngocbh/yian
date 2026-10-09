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
    mode: plain
    prompt: hi
    env:
      max_tool_iterations: 300
  b:
    kind: agent
    mode: plain
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
  a: {kind: agent, mode: plain, prompt: hi}
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
    mode: plain
    prompt: hi
    env: [1, 2, 3]
output: ${a.output}
"""
        )
