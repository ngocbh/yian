"""OUTPUT_RESOLVER / COLLECTOR are fully retired (the splice replaced them).

The START_ID..END_ID splice made both packages dead producers. Retiring them deletes the
packages + the NodeKind members + exports, and RE-KEYS the eval_node pool-scoped post-assert
branch from OUTPUT_RESOLVER to END_ID (so a spliced child END_ID's re-namespaced
`${<callsite>/X.output}` post asserts still fire pool-scoped). This pins the deletion + the
re-keyed firing.
"""

import importlib

import pytest

from agent_composer.events import NodeFailed, NodeSucceeded
from agent_composer.nodes.base import NodeKind
from agent_composer.nodes.end import EndNode
from agent_composer.typesys.pool import VariablePool
from tests.engine._fakes import drive, stamp_reads


def test_node_kind_has_no_output_resolver_or_collector():
    assert not hasattr(NodeKind, "OUTPUT_RESOLVER")
    assert not hasattr(NodeKind, "COLLECTOR")


def test_output_resolver_and_collector_packages_are_gone():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("agent_composer.nodes.output_resolver")
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("agent_composer.nodes.collector")


def test_child_end_pool_scoped_post_assert_fires_through_the_rekeyed_end_branch():
    # the deleted test_clone_child::test_output_resolver_violated_pool_scoped_post_assert_fails
    # body, retargeted onto an END_ID node: a namespaced `${<callsite>/X.output}` post-assert must
    # still fire POOL-scoped through the re-keyed NodeKind.END branch.
    end = EndNode.record("each/__end__", output_names=["out"])
    end.post_asserts = ["${each/tail.output} != ''"]
    assert end.kind == NodeKind.END
    pool = VariablePool()

    pool.set("each/tail", "ok")
    # END_ID binds its `out` param from its wiring (a real flow wires each output param to
    # `${producer.output}`); derive that wiring from the stamped reads so `out` resolves
    # POOL-scoped (an EMPTY wiring would leave it ABSENT and `_run_impl` would KeyError). The
    # namespaced post-assert still fires POOL-scoped through the re-keyed NodeKind.END branch.
    ok = list(drive(stamp_reads(end, {"out": "${each/tail.output}"}), pool))
    assert any(isinstance(e, NodeSucceeded) for e in ok)

    bad = EndNode.record("each/__end__", output_names=["out"])
    bad.post_asserts = ["${each/tail.output} != ''"]
    pool.set("each/tail", "")
    failed = list(drive(stamp_reads(bad, {"out": "${each/tail.output}"}), pool))
    assert any(isinstance(e, NodeFailed) and "post-assert" in e.error for e in failed)
