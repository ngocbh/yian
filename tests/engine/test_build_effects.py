from agent_composer.compose.build import build_leaf_node
from agent_composer.compose.parse.file import HumanInputDescriptor, WaitDescriptor
from agent_composer.nodes.human_input import HumanInputNode
from agent_composer.nodes.wait import WaitNode


def test_build_human_input_node():
    desc = HumanInputDescriptor(id="approve", prompt="ok? ${a}",
                                inputs={"a": "${p.output}"}, outputs="str")
    node, wiring = build_leaf_node(desc, {})
    assert isinstance(node, HumanInputNode)
    assert node.output_type is not None         # str enforced at write boundary
    assert [p.name for p in node.params] == ["a"]     # the node-side signature
    assert wiring == {"a": "${p.output}"}            # the flow-owned source


def test_build_wait_node():
    node, wiring = build_leaf_node(WaitDescriptor(id="settle", until="${input.settle_at}"), {})
    assert isinstance(node, WaitNode)
    assert node.is_timed is True
    assert wiring == {"until": "${input.settle_at}"}  # the `until` source rides the wiring
    assert node.output_type is None


def test_wait_until_outputs_ref_makes_data_edge():
    # a wait whose `until` reads ${X.output} must order after X
    from agent_composer.compose.parse.file import WaitDescriptor, CodeDescriptor
    from agent_composer.compose.build import build_leaf_node
    from agent_composer.compose.edges.data import infer_data_edges
    descs = {
        "clock": CodeDescriptor(id="clock", code="m:f", outputs="datetime"),
        "settle": WaitDescriptor(id="settle", until="${clock.output}"),
    }
    flow_wiring = {nid: build_leaf_node(d, {})[1] for nid, d in descs.items()}
    edges = infer_data_edges(descs, flow_wiring)
    assert any(e.from_ == "clock" and e.to == "settle" for e in edges)
