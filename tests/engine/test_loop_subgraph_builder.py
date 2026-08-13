"""The pure LOOP CONTINUE builder `loop_continue_subgraph` — pins body + next-driver shape.

`loop_continue_subgraph(child, origin, carried, k, driver)` splices body_k (NO commit_as — its
Output feeds the next driver by plain wiring) + the fresh `L~(k+1)` driver + the producer edge
`body_k.END -> L~(k+1)`. Bodies are keyed on ORIGIN (not the running driver id), so live `run` and
durable reconstruction builds the SAME `L#k/…` namespace.
"""
from agent_composer.compile.expand import loop_continue_subgraph, map_callsite, ns
from agent_composer.compile.model import END_ID, Flow
from agent_composer.nodes.loop import LoopNode

from tests.engine.test_expand import _child_flow


def _driver(origin, k, child):
    d = LoopNode(f"{origin}~{k}", flow_id="f", child=child, predicate_kind="while",
                 predicate="${x}", max_iters=10, iteration=k, origin_id=origin)
    d.params = child.nodes[child.start_id].params
    return d


def test_continue_subgraph_body_keyed_on_origin_not_driver():
    child = _child_flow()
    driver = _driver("lp", 1, child)                     # origin=lp; body must key on lp, not lp~0
    sg = loop_continue_subgraph(child, origin="lp", carried={"x": 1}, k=0, driver=driver)
    assert isinstance(sg, Flow)
    callsite = map_callsite("lp", 0)                      # "lp#0" (origin-keyed, NOT lp~0#0)
    assert sg.start_id == ns(callsite, child.start_id)
    assert sg.end_id == ns(callsite, END_ID)
    assert all(nid.startswith(callsite + "/") or nid == "lp~1" for nid in sg.nodes)


def test_continue_body_end_has_no_commit_as():
    child = _child_flow()
    sg = loop_continue_subgraph(child, "lp", {"x": 1}, 0, _driver("lp", 1, child))
    assert sg.nodes[ns(map_callsite("lp", 0), END_ID)].commit_as is None


def test_continue_splices_next_driver_and_edge():
    child = _child_flow()
    driver = _driver("lp", 1, child)
    sg = loop_continue_subgraph(child, "lp", {"x": 1}, 0, driver)
    assert "lp~1" in sg.nodes
    body_end = ns(map_callsite("lp", 0), END_ID)
    assert any(e.from_ == body_end and e.to == "lp~1" for e in sg.edges)
    assert sg.wiring["lp~1"] == {"x": f"${{{body_end}.output.x}}"}


def test_continue_keys_body_on_origin_when_driver_id_differs():
    # A running driver `lp~2` with origin `lp` must still splice bodies under `lp#k`, never `lp~2#k`.
    child = _child_flow()
    driver = _driver("lp", 3, child)
    sg = loop_continue_subgraph(child, origin="lp", carried={"x": 5}, k=2, driver=driver)
    assert ns(map_callsite("lp", 2), child.start_id) in sg.nodes
    assert not any(nid.startswith("lp~2#") for nid in sg.nodes)
    assert "lp~3" in sg.nodes
