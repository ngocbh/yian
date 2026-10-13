"""The skip model: which nodes of one flow level CASE routing may skip, and together.

A node is skipped when the runtime join rule marks it dead: it has control edges (from CASEs)
and every one is skipped, or a required data group has every producer skipped, or a required
ordering edge (`depends_on`) has its producer skipped. A control edge is skipped when its CASE
is skipped or picked another branch. A CASE that runs picks one of its branches, or no branch
at all unless it is exhaustive (`exhaustive_cases`). CASE conditions are treated as
independent, so the model enumerates every combination of the choices of the CASEs that are
ancestors of the queried nodes.

The enumeration is bounded by `_WORLD_LIMIT` assignments (complete or cut short) per query. Past
the bound each query falls back to the answer that is safe for its caller: `may_all_skip`
says True (the output gets a null default it may not need) and `may_skip_while_running` says
False (nothing is reported). A level whose edges have a cycle has no skippable nodes here; the cycle check
reports the cycle.

Knows about: compiled edges, CASE descriptors and output contracts. Never imports: runtime.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import AbstractSet, Iterable, Optional

from yian.compile.model import Edge
from yian.compose.parse.file import CaseDescriptor, NodeDescriptor
from yian.compose.validate.case_exhaustive import case_on_exhaustive
from yian.compose.validate.narrowing import Narrowing
from yian.nodes.contracts import OutputContract

_WORLD_LIMIT = 10_000


@dataclass
class _Join:
    """The incoming edges of one node that can make it skip."""

    control: list[tuple[str, str]] = field(default_factory=list)  # (case, handle)
    # input group -> (optional, producers); a group takes its first edge's `optional`, like
    # the runtime join.
    groups: dict[Optional[str], tuple[bool, list[str]]] = field(default_factory=dict)
    required_order: list[str] = field(default_factory=list)


class SkipModel:
    """Which nodes of one flow level may be skipped by its CASE routing.

    Args:
        node_ids (`Iterable[str]`):
            The level's node ids, before the START/END boundary is synthesized. Any other id
            (START, a flow input) is a producer that never skips.
        edges (`Iterable[Edge]`):
            The level's edges. An edge into a node outside `node_ids` is ignored.
        exhaustive (`AbstractSet[str]`):
            Ids of the CASEs that always take a branch when they run.
        limit (`int`, *optional*, defaults to `_WORLD_LIMIT`):
            Most CASE assignments one query examines before it falls back.

    Example:
        >>> edges = [Edge(id="t", from_="c", to="a", source_handle="then"),
        ...          Edge(id="e", from_="c", to="b", source_handle="else")]
        >>> SkipModel(["c", "a", "b"], edges, {"c"}).may_all_skip({"a", "b"})
        False
    """

    def __init__(
        self,
        node_ids: Iterable[str],
        edges: Iterable[Edge],
        exhaustive: AbstractSet[str],
        *,
        limit: int = _WORLD_LIMIT,
    ) -> None:
        self._nodes = list(dict.fromkeys(node_ids))
        self._known = set(self._nodes)
        self._limit = limit
        self._joins = {n: _Join() for n in self._nodes}
        self._parents: dict[str, set[str]] = {n: set() for n in self._nodes}
        handles: dict[str, dict[str, None]] = {}
        self._control_targets: dict[str, set[str]] = {}
        for edge in edges:
            if edge.to not in self._known:
                continue
            join = self._joins[edge.to]
            if edge.from_ in self._known:
                self._parents[edge.to].add(edge.from_)
            if edge.source_handle is not None:
                join.control.append((edge.from_, edge.source_handle))
                if edge.from_ in self._known:
                    handles.setdefault(edge.from_, {})[edge.source_handle] = None
                    self._control_targets.setdefault(edge.from_, set()).add(edge.to)
            elif edge.ordering:
                if not edge.optional:
                    join.required_order.append(edge.from_)
            else:
                optional, producers = join.groups.setdefault(
                    edge.input_group, (edge.optional, [])
                )
                producers.append(edge.from_)
        # A CASE's choices: each branch, then "no branch" (None) unless it is exhaustive.
        self._choices: dict[str, tuple[Optional[str], ...]] = {
            case: (*names, *(() if case in exhaustive else (None,)))
            for case, names in handles.items()
        }
        self._order = self._topological_order()

    def may_all_skip(self, producers: Iterable[str]) -> bool:
        """Return whether some CASE routing skips every node in `producers`.

        Args:
            producers (`Iterable[str]`):
                Node ids. An id outside the level never skips.

        Returns:
            `bool`:
                True when one assignment of CASE choices skips them all, or when the search
                hits its bound. False for an empty set or a cyclic level.
        """
        targets = set(producers)
        if not targets or not targets <= self._known or self._order is None:
            return False
        return self._search(targets, None, on_limit=True)

    def may_skip_while_running(
        self, producers: Iterable[str], consumer: str | None
    ) -> bool:
        """Return whether some CASE routing skips every node in `producers` but runs `consumer`.

        Args:
            producers (`Iterable[str]`):
                Node ids. An id outside the level never skips.
            consumer (`str | None`):
                The node that must run. `None` (or an id outside the level) is a consumer that
                always runs, such as the flow's END.

        Returns:
            `bool`:
                True when one assignment of CASE choices does so. False for an empty set, a
                cyclic level, or when the search hits its bound.
        """
        targets = set(producers)
        if not targets or not targets <= self._known or self._order is None:
            return False
        if consumer not in self._known:
            consumer = None
        return self._search(targets, consumer, on_limit=False)

    def _topological_order(self) -> list[str] | None:
        """Return the level's nodes in topological order, or None when the edges cycle."""
        indegree = {n: len(self._parents[n]) for n in self._nodes}
        children: dict[str, list[str]] = {n: [] for n in self._nodes}
        for node, parents in self._parents.items():
            for parent in parents:
                children[parent].append(node)
        ready = deque(n for n in self._nodes if not indegree[n])
        order = []
        while ready:
            node = ready.popleft()
            order.append(node)
            for child in children[node]:
                indegree[child] -= 1
                if not indegree[child]:
                    ready.append(child)
        return order if len(order) == len(self._nodes) else None

    def _search(
        self, targets: set[str], consumer: str | None, *, on_limit: bool
    ) -> bool:
        """Depth-first search for a CASE assignment that skips `targets` and runs `consumer`.

        Args:
            targets (`set[str]`):
                Node ids in the level that must all be skipped.
            consumer (`str | None`):
                A node id in the level that must run, or `None` for no such node.
            on_limit (`bool`):
                The answer when the search examines `limit` assignments without finding one.

        Returns:
            `bool`:
                Whether such an assignment exists, or `on_limit` past the bound.
        """
        queried = targets | ({consumer} if consumer is not None else set())
        closure = set(queried)
        pending = list(queried)
        while pending:
            for parent in self._parents[pending.pop()]:
                if parent not in closure:
                    closure.add(parent)
                    pending.append(parent)
        nodes = [n for n in self._order if n in closure]
        branching = {
            case
            for case in self._choices
            if case in closure and self._control_targets[case] & closure
        }
        stack: list[tuple[int, frozenset[str], dict[str, Optional[str]]]] = [
            (0, frozenset(), {})
        ]
        worlds = 0
        while stack:
            if worlds >= self._limit:
                return on_limit
            index, done, choices = stack.pop()
            skipped = set(done)
            for node in nodes[index:]:
                index += 1
                dead = self._skips(node, skipped, choices)
                if dead:
                    skipped.add(node)
                if (node in targets and not dead) or (node == consumer and dead):
                    worlds += 1
                    break
                if node in branching and not dead:
                    first, *rest = self._choices[node]
                    for choice in reversed(rest):
                        stack.append((index, frozenset(skipped), {**choices, node: choice}))
                    choices = {**choices, node: first}
            else:
                return True
        return False

    def _skips(
        self, node: str, skipped: set[str], choices: dict[str, Optional[str]]
    ) -> bool:
        """Return whether `node` is skipped, given its skipped ancestors and CASE choices.

        Args:
            node (`str`):
                A node id in the level.
            skipped (`set[str]`):
                The skipped nodes among the ones before `node` in topological order.
            choices (`dict[str, Optional[str]]`):
                The branch each running CASE before `node` picked; `None` is no branch.

        Returns:
            `bool`:
                Whether the join rule marks `node` dead.
        """
        join = self._joins[node]
        if join.control and all(
            case in skipped or (case in self._choices and choices.get(case) != handle)
            for case, handle in join.control
        ):
            return True
        if any(
            not optional and all(p in skipped for p in producers)
            for optional, producers in join.groups.values()
        ):
            return True
        return any(p in skipped for p in join.required_order)


def exhaustive_cases(
    descriptors: dict[str, NodeDescriptor],
    producers: dict[str, OutputContract],
    narrowing: Narrowing | None = None,
) -> set[str]:
    """Return the ids of the CASE descriptors that always take a branch when they run.

    Args:
        descriptors (`dict[str, NodeDescriptor]`):
            The level's node id -> descriptor. Non-CASE descriptors are ignored.
        producers (`dict[str, OutputContract]`):
            Producer node id -> output contract.
        narrowing (`Narrowing | None`, *optional*, defaults to `None`):
            Per-node narrowed views; each CASE resolves its `on:` against its own view.

    Returns:
        `set[str]`:
            The exhaustive CASE ids (see `case_on_exhaustive`).
    """
    return {
        desc.id
        for desc in descriptors.values()
        if isinstance(desc, CaseDescriptor)
        and case_on_exhaustive(
            desc,
            narrowing.for_node(desc.id)[0] if narrowing is not None else producers,
        )
    }
