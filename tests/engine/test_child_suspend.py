"""Nested generic-Pause compatibility reachable inside a ``call`` child.

Authorable HUMAN_INPUT and WAIT block in-process. These checkpoint-only tests load a
HUMAN_INPUT-shaped flow and replace its gates with the explicit artificial Pause fake,
proving dormant deliver-as-Output compatibility through REF, MAP, and nested growth.
"""

from agent_composer.compose import load_flow
from agent_composer.run import resume_command, resume_flow, run_flow
from tests.engine._fakes import replace_human_inputs_with_artificial_pauses


def _load_with_artificial_pauses(source: str):
    """Load ``source`` and swap authored human gates only for checkpoint testing."""
    loaded = load_flow(source)
    replace_human_inputs_with_artificial_pauses(loaded.compiled)
    return loaded

CHILD_EFFECT = """
id: parent
name: parent
input:
  settle_at: date
defs:
  sub:
    input:
      settle_at: date
    nodes:
      w:
        kind: human_input
        prompt: "continue?"
        output: Optional[str]
    output: ${w.output}
nodes:
  call_sub:
    kind: call
    call: sub
    input:
      settle_at: ${input.settle_at}
output: ${call_sub.output}
"""


def test_ref_child_with_effect_loads_pauses_namespaced_and_resumes():
    loaded = _load_with_artificial_pauses(CHILD_EFFECT)
    res = run_flow(loaded, {"settle_at": "2099-01-01"})
    assert res.status == "paused"
    reason = res.pause_reasons[0]
    # the parked artificial leaf lives under the NAMESPACED id in the live graph
    assert reason.node_id == "call_sub/w"                  # callsite "call_sub" / child node "w"
    cmd = resume_command(loaded, reason, None)
    assert cmd.node_id == "call_sub/w"                     # the live namespaced id rides through
    done = resume_flow(loaded, engine=res.engine, commands=[cmd])
    assert done.status == "succeeded"


def test_child_suspend_resumes_through_namespaced_end():
    # after a nested pause resumes, the parent's call value flows from the NAMESPACED
    # child END_ID (the alias filler) — the splice's alias substitution writes the spawner value
    # from the cloned Flow's end_id == ns(callsite, child.end_id).
    loaded = _load_with_artificial_pauses(CHILD_EFFECT)
    res = run_flow(loaded, {"settle_at": "2099-01-01"})
    assert res.status == "paused"
    done = resume_flow(loaded, engine=res.engine,
                       commands=[resume_command(loaded, res.pause_reasons[0], None)])
    assert done.status == "succeeded"
    eng = res.engine
    # the namespaced child END_ID is the redirect filler for the spawner (no __out resolver):
    # its baked `commit_as` names the spawner id.
    assert eng.flow.nodes["call_sub/__end__"].commit_as == "call_sub"
    assert not any("/__out" in nid for nid in eng.flow.nodes)


# A mapped child whose test-only artificial leaf pauses under a per-element namespace.
MAP_CHILD_EFFECT = """
id: parent
name: parent
input:
  settle_ats: list[date]
defs:
  sub:
    input:
      settle_at: date
    nodes:
      w:
        kind: human_input
        prompt: "continue?"
        output: Optional[str]
      done:
        kind: code
        depends_on: [w]
        input:
          settle_at: ${input.settle_at}
        output: str
        code: tests.seeds.fns:confirm_action
    output: ${done.output}
nodes:
  map_sub:
    kind: map
    over: ${input.settle_ats}
    call: sub
    input:
      settle_at: ${item}
output: ${map_sub.output}
"""


def test_map_child_with_effect_loads_pauses_namespaced_and_resumes():
    loaded = _load_with_artificial_pauses(MAP_CHILD_EFFECT)
    res = run_flow(loaded, {"settle_ats": ["2099-01-01"]})
    assert res.status == "paused"
    reason = res.pause_reasons[0]
    # single-element `over` -> the per-element artificial pause is "map_sub#0/w"
    assert reason.node_id == "map_sub#0/w"
    cmd = resume_command(loaded, reason, None)
    assert cmd.node_id == "map_sub#0/w"
    done = resume_flow(loaded, engine=res.engine, commands=[cmd])
    assert done.status == "succeeded"


# A def two levels down: parent -> outer -> inner, the effect lives in `inner`. The
# transitive expansion namespaces the artificial pause doubly: call_outer/call_inner/w.
NESTED_GRANDCHILD_EFFECT = """
id: parent
name: parent
input:
  settle_at: date
defs:
  inner:
    input:
      settle_at: date
    nodes:
      w:
        kind: human_input
        prompt: "continue?"
        output: Optional[str]
    output: ${w.output}
  outer:
    input:
      settle_at: date
    nodes:
      call_inner:
        kind: call
        call: inner
        input:
          settle_at: ${input.settle_at}
    output: ${call_inner.output}
nodes:
  call_outer:
    kind: call
    call: outer
    input:
      settle_at: ${input.settle_at}
output: ${call_outer.output}
"""


def test_nested_grandchild_effect_loads_pauses_doubly_namespaced_and_resumes():
    loaded = _load_with_artificial_pauses(NESTED_GRANDCHILD_EFFECT)
    res = run_flow(loaded, {"settle_at": "2099-01-01"})
    assert res.status == "paused"
    reason = res.pause_reasons[0]
    # parent -> call_outer -> call_inner -> the artificial pause is doubly namespaced
    assert reason.node_id == "call_outer/call_inner/w"
    cmd = resume_command(loaded, reason, None)
    assert cmd.node_id == "call_outer/call_inner/w"
    done = resume_flow(loaded, engine=res.engine, commands=[cmd])
    assert done.status == "succeeded"
