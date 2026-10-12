"""`AgentNode._ctx` projects the node's semantic output contract."""

from yian.diagnostics import OriginRef
from yian.nodes.agent.node import AgentNode
from yian.nodes.contracts import ContractOrigins, ValueContract
from yian.typesys.values import Type, ValueKind


def test_ctx_carries_the_output_contract_type_only():
    n = AgentNode("a", prompt="hi")
    contract = ValueContract(
        Type.scalar(ValueKind.INTEGER),
        ContractOrigins(default=OriginRef("nodes/a/output")),
    )
    n.output_contract = contract

    ctx = n._ctx(prompt="hi")

    assert ctx.output_type is contract.type
    assert ctx.output_type is not contract
