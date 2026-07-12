"""Nested suspension proof: a HUMAN_INPUT/WAIT effect reachable inside a `call` child.

Earlier the loader rejected a HUMAN_INPUT/WAIT reachable inside a `call` child at load
(`reject_child_effects`), and the runtime `CallNode` raised as a backstop. The engine now
grows the graph via `Grow` and resumes in-memory, so a child effect LOADS, RUNS to a
pause on its NAMESPACED leaf id, and RESUMES to a terminal via the deliver-as-Output path
(`resume_command` -> DeliverAnswerCommand on the live namespaced id). These 3 flows (REF
child / MAP child / nested grandchild) are the end-to-end proof the whole graph-growth +
deliver-as-Output design composes.
"""

from agent_composer.compose import load_flow
from agent_composer.run import resume_command, resume_flow, run_flow

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
        kind: wait
        until: ${input.settle_at}
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
    loaded = load_flow(CHILD_EFFECT)                       # no LoadError now
    res = run_flow(loaded, {"settle_at": "2099-01-01"})
    assert res.status == "paused"
    reason = res.pause_reasons[0]
    # the parked WAIT lives under the NAMESPACED id in the LIVE graph (the resume seam)
    assert reason.node_id == "call_sub/w"                  # callsite "call_sub" / child node "w"
    cmd = resume_command(loaded, reason, None)             # WAIT release = deliver-as-Output(None)
    assert cmd.node_id == "call_sub/w"                     # the live namespaced id rides through
    done = resume_flow(loaded, engine=res.engine, commands=[cmd])
    assert done.status == "succeeded"


def test_child_suspend_resumes_through_namespaced_end():
    # after a nested pause resumes, the parent's call value flows from the NAMESPACED
    # child END_ID (the alias filler) — the splice's alias substitution writes the spawner value
    # from the cloned Flow's end_id == ns(callsite, child.end_id).
    loaded = load_flow(CHILD_EFFECT)
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


# A MAPPED call (`kind: map` + `over:` -> a `MapNode`) whose def contains a suspending effect: the
# child now expands per element and parks on the namespaced per-element WAIT.
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
        kind: wait
        until: ${input.settle_at}
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
    loaded = load_flow(MAP_CHILD_EFFECT)                   # no LoadError now
    res = run_flow(loaded, {"settle_ats": ["2099-01-01"]})
    assert res.status == "paused"
    reason = res.pause_reasons[0]
    # single-element `over` -> the per-element callsite is "map_sub#0"; the WAIT is "map_sub#0/w"
    assert reason.node_id == "map_sub#0/w"
    cmd = resume_command(loaded, reason, None)
    assert cmd.node_id == "map_sub#0/w"
    done = resume_flow(loaded, engine=res.engine, commands=[cmd])
    assert done.status == "succeeded"


# A def two levels down: parent -> outer -> inner, the effect lives in `inner`. The
# transitive expansion now namespaces the WAIT doubly: call_outer/call_inner/w.
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
        kind: wait
        until: ${input.settle_at}
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
    loaded = load_flow(NESTED_GRANDCHILD_EFFECT)          # no LoadError now
    res = run_flow(loaded, {"settle_at": "2099-01-01"})
    assert res.status == "paused"
    reason = res.pause_reasons[0]
    # parent -> call_outer expands -> call_inner expands -> the WAIT is doubly namespaced
    assert reason.node_id == "call_outer/call_inner/w"
    cmd = resume_command(loaded, reason, None)
    assert cmd.node_id == "call_outer/call_inner/w"
    done = resume_flow(loaded, engine=res.engine, commands=[cmd])
    assert done.status == "succeeded"
