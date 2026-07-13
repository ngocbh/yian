"""OUTPUT_RESOLVER / COLLECTOR are fully retired (the splice replaced them).

The START_ID..END_ID splice made both packages dead producers. Retiring them deletes the
packages + the NodeKind members + exports. A spliced child END's re-namespaced
`${<callsite>/X.output}` post-assert fires RECORD-scoped: the ref is lowered to a synth param
bound into the record. This pins the deletion + the record-scoped firing.
"""

import importlib

import pytest

from agent_composer.events import NodeFailed, NodeSucceeded
from agent_composer.nodes.base import NodeKind, ParamDecl
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


def _end_with_tail_assert(node_id="each/__end__"):
    # A spliced child END whose post-assert reads a SYNTH PARAM (`__assert_tail`) bound into the
    # record from `${each/tail.output}` — the production shape after ref lowering.
    end = EndNode.record(node_id, output_names=["out"])
    end.post_asserts = ["${__assert_tail} != ''"]
    end.params.append(ParamDecl(name="__assert_tail"))
    return stamp_reads(
        end,
        {"out": "${each/tail.output}", "__assert_tail": "${each/tail.output:-null}"},
    )


def test_child_end_record_scoped_post_assert_fires_via_synth_param():
    # a re-namespaced `${<callsite>/X.output}` post-assert fires RECORD-scoped: the ref is lowered
    # to a synth param bound into the record from the pool (no pool-fallback firing path).
    assert _end_with_tail_assert().kind == NodeKind.END
    pool = VariablePool()

    pool.set("each/tail", "ok")
    ok = list(drive(_end_with_tail_assert(), pool))
    assert any(isinstance(e, NodeSucceeded) for e in ok)

    pool.set("each/tail", "")
    failed = list(drive(_end_with_tail_assert(), pool))
    assert any(isinstance(e, NodeFailed) and "post-assert" in e.error for e in failed)
