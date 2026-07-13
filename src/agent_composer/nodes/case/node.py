"""CASE — deterministic conditional routing.

Strict (a pure function of its declared inputs): the node declares `inputs` (bound
via `from:` like any leaf), and each case's `when:` interpolates only those inputs
as bare `${name}`, evaluated against the **bound input record**, not the pool.
Routing is purely deterministic: the engine makes no LLM decision here. To branch
on a judgment, an upstream node produces a variable (e.g. a classifier AGENT
writing a label, or a CODE node a number) which the CASE binds as an input and
the case compares. Cases are tried in order; the first truthy one wins. If none
match, the reserved `"default"` handle is taken.

The node writes no state — it returns the selected handle on its result, and the
engine's `_branch` does the skip-flood.
"""

from dataclasses import dataclass
from typing import Optional

from agent_composer.expr.expressions import evaluate_when_record
from agent_composer.nodes.base import Node, NodeKind, Route

DEFAULT_HANDLE = "default"


@dataclass
class Case:
    """One labeled branch routed by a `when` expression."""

    handle: str  # matches the outgoing edge's source_handle
    when: Optional[str] = None


class CaseNode(Node):
    """
    Deterministic conditional routing — no LLM decision.

    Cases are tried in order; the first whose `when:` is truthy (evaluated against the bound
    input record) wins and its handle is returned on the result. If none match, the reserved
    `"default"` handle is taken. The node writes no value — the engine's `_branch` does the
    skip-flood off the chosen handle.

    Args:
        node_id (`str`):
            The node's unique id.
        cases (`list[Case]`):
            The labeled branches, tried in order; each must carry a `when` expression.
        title (`str`, *optional*, defaults to `None`):
            Display title.
    """

    kind = NodeKind.CASE

    def __init__(
        self,
        node_id: str,
        cases: list[Case],
        *,
        title: Optional[str] = None,
    ) -> None:
        super().__init__(node_id, title=title)
        self.cases = cases

    def _run_impl(self, inputs: dict) -> Route:
        # Strict CASE: route on the bound input record only — each `when:`
        # interpolates this node's declared inputs as bare `${name}`, not the pool.
        # Routing-only: returns a Route carrying the chosen case handle (no value).
        for case in self.cases:
            if case.when is None:
                raise ValueError(f"node {self.id!r} case {case.handle!r} has no `when` expression")
            if evaluate_when_record(case.when, inputs):
                return Route(case.handle)
        return Route(DEFAULT_HANDLE)
