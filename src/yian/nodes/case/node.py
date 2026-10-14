"""CASE — deterministic conditional routing.

Strict (a pure function of its declared inputs): the node declares `inputs` (bound
via `from:` like any leaf), and each case's `when:` interpolates only those inputs
as bare `${name}`, evaluated against the **bound input record**, not the pool.
Routing is purely deterministic: the engine makes no LLM decision here. To branch
on a judgment, an upstream node produces a variable (e.g. a classifier AGENT
writing a label, or a CODE node a number) which the CASE binds as an input and
the case compares. Cases are tried in order; the first truthy one wins. If none
match, the reserved else handle (`ELSE_HANDLE`) is taken.

The node writes no state — it returns the selected handle on its result, and the
engine's `_branch` does the skip-flood.
"""

from dataclasses import dataclass
import re
from typing import Callable, Iterator, Optional

from yian.diagnostics import OriginRef
from yian.expr.expressions import evaluate_when_record
from yian.nodes.base import ELSE_HANDLE, Node, NodeKind, Route
from yian.nodes.contracts import _map_origin, _validate_optional_origin


@dataclass
class Case:
    """One ordered branch predicate and its labeled route.

    ``origin`` is the optional source-neutral identity of this exact ``when``
    occurrence. Duplicate expression text remains distinct by list position.
    """

    handle: str  # matches the outgoing edge's source_handle
    when: Optional[str] = None
    origin: OriginRef | None = None

    def __post_init__(self) -> None:
        """Require source-neutral predicate provenance and a non-else handle."""
        _validate_optional_origin(self.origin, "CASE predicate origin")
        if self.handle == ELSE_HANDLE:
            raise ValueError(
                f"CASE handle {self.handle!r} is reserved for the else route; name the "
                "branch by its target node id"
            )


class CaseNode(Node):
    """
    Deterministic conditional routing — no LLM decision.

    Cases are tried in order; the first whose `when:` is truthy (evaluated against the bound
    input record) wins and its handle is returned on the result. If none match, the reserved
    else handle (`ELSE_HANDLE`) is taken. The node writes no value — the engine's `_branch`
    does the skip-flood off the chosen handle.

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

    def validate_execution_setup(self) -> None:
        """Require every compiled branch to carry its predicate expression."""
        super().validate_execution_setup()
        for case in self.cases:
            _validate_optional_origin(case.origin, "CASE predicate origin")
            if case.handle == ELSE_HANDLE:
                raise ValueError(
                    f"node {self.id!r} case handle {case.handle!r} is reserved for the "
                    "else route; name the branch by its target node id"
                )
            if case.when is None:
                raise ValueError(
                    f"node {self.id!r} case {case.handle!r} has no `when` expression"
                )
            if not isinstance(case.when, str):
                raise TypeError(
                    f"node {self.id!r} case {case.handle!r} `when` must be a string"
                )

    def iter_local_origins(
        self,
        *,
        _seen_carriers: Optional[set[int]] = None,
    ) -> Iterator[OriginRef]:
        """Yield common origins followed by ordered predicate origins."""
        seen = _seen_carriers if _seen_carriers is not None else set()
        yield from super().iter_local_origins(_seen_carriers=seen)
        for case in self.cases:
            identity = id(case)
            if identity in seen:
                continue
            seen.add(identity)
            _validate_optional_origin(case.origin, "CASE predicate origin")
            if case.origin is not None:
                yield case.origin

    def rewrite_local_origins(
        self,
        mapper: Callable[[OriginRef], OriginRef],
        *,
        _seen_carriers: Optional[set[int]] = None,
    ) -> None:
        """Rewrite common and ordered predicate origins on a copied CASE."""
        seen = _seen_carriers if _seen_carriers is not None else set()
        super().rewrite_local_origins(mapper, _seen_carriers=seen)
        for case in self.cases:
            identity = id(case)
            if identity in seen:
                continue
            seen.add(identity)
            _validate_optional_origin(case.origin, "CASE predicate origin")
            case.origin = _map_origin(case.origin, mapper)

    def exception_origin(self, exception: Exception) -> OriginRef | None:
        """Return the exact ordered predicate whose evaluation raised.

        The original exception and traceback remain unchanged. Selection uses
        the identity of this node's ``forward`` frame and its active ``Case``;
        duplicate predicate text therefore remains independently locatable.
        Failures outside predicate evaluation, and source-free predicates, fall
        back to the authored node origin.
        """
        active_case: Case | None = None
        traceback = exception.__traceback__
        while traceback is not None:
            frame = traceback.tb_frame
            if (
                frame.f_code is CaseNode.forward.__code__
                and frame.f_locals.get("self") is self
            ):
                candidate = frame.f_locals.get("case")
                if any(candidate is case for case in self.cases):
                    active_case = candidate
            traceback = traceback.tb_next

        if active_case is None:
            return super().exception_origin(exception)
        _validate_optional_origin(active_case.origin, "CASE predicate origin")
        return (
            active_case.origin
            if active_case.origin is not None
            else super().exception_origin(exception)
        )

    def forward(self, inputs: dict) -> Route:
        # Strict CASE: route on the bound input record only — each `when:`
        # interpolates this node's declared inputs as bare `${name}`, not the pool.
        # Routing-only: returns a Route carrying the chosen case handle (no value).
        for case in self.cases:
            if evaluate_when_record(case.when, inputs, self.types):
                return Route(case.handle)
        return Route(ELSE_HANDLE)

    def backward_input_names(self) -> tuple[str, ...]:
        """Exclude lowered CASE controls from ordinary declared input targets.

        Returns:
            `tuple[str, ...]`: Declared names except generated `__on` and `__rN`
                controls, preserving directly declared ordinary inputs in order.
        """
        return tuple(
            name for name in super().backward_input_names()
            if name != "__on" and re.fullmatch(r"__r[0-9]+", name) is None
        )
