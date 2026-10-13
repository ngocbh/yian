"""The skip model: which nodes of one flow level CASE routing may skip, and together.

A node is skipped when the runtime join rule marks it dead: it has control edges (from CASEs)
and every one is skipped, or a required data group has every producer skipped, or a required
ordering edge (`depends_on`) has its producer skipped. A control edge is skipped when its CASE
is skipped or picked another branch. A data or ordering edge from a CASE is skipped when the
CASE is skipped or picked a branch other than `else:` or "no branch". A CASE that runs picks
one of its branches, or no branch at all unless it is exhaustive. CASE conditions are treated
as independent, so the model enumerates every combination of the choices of the CASEs that
can skip an ancestor of the queried nodes.

The enumeration is bounded by `_WORLD_LIMIT` assignments (complete or cut short) per query.
Past the bound each query falls back to the answer that is safe for its caller:
`may_all_skip` says True (the output gets a null default it may not need) and
`may_skip_while_running` says False (nothing is reported). A level whose edges have a cycle
has no skippable nodes here; the cycle check reports the cycle.

`null_skipped_outputs` gives a flow output that reads only nodes that may all be skipped the
explicit null default, so it returns null instead of leaving END unreachable.

Knows about: compiled edges, flow outputs, the CASE default handle, and binding producers.
Never imports: runtime.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Set
from dataclasses import dataclass, field, replace

from yian.compile.model import Edge, FlowOutput
from yian.compose.edges.producers import _binding_producers
from yian.compose.helpers.output_contracts import direct_reference_path
from yian.expr import binding_co_skips
from yian.nodes.case import DEFAULT_HANDLE
from yian.nodes.contracts import binding_source

_WORLD_LIMIT = 10_000


@dataclass
class _Join:
    """The incoming edges of one node that can make it skip."""

    control: list[tuple[str, str]] = field(default_factory=list)  # (case, handle)
    # input group -> (optional, producers); a group takes its first edge's `optional`, like
    # the runtime join.
    groups: dict[str | None, tuple[bool, list[str]]] = field(default_factory=dict)
    required_order: list[str] = field(default_factory=list)

    def skip_sources(self) -> set[str]:
        """Return the producers of the edges that can make this node skip."""
        sources = {case for case, _ in self.control} | set(self.required_order)
        for optional, producers in self.groups.values():
            if not optional:
                sources.update(producers)
        return sources


class SkipModel:
    """Which nodes of one flow level may be skipped by its CASE routing.

    Args:
        node_ids (`Iterable[str]`):
            The level's node ids, before the START/END boundary is synthesized. Any other id
            (START, a flow input) is a producer that never skips.
        edges (`Iterable[Edge]`):
            The level's edges. An edge into a node outside `node_ids` is ignored.
        exhaustive (`Set[str]`):
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
        exhaustive: Set[str],
        *,
        limit: int = _WORLD_LIMIT,
    ) -> None:
        self._nodes = list(dict.fromkeys(node_ids))
        self._known = set(self._nodes)
        self._limit = limit
        self._joins = {n: _Join() for n in self._nodes}
        graph_parents: dict[str, set[str]] = {n: set() for n in self._nodes}
        handles: dict[str, dict[str, None]] = {}
        for edge in edges:
            if edge.to not in self._known:
                continue
            join = self._joins[edge.to]
            if edge.from_ in self._known:
                graph_parents[edge.to].add(edge.from_)
            if edge.source_handle is not None:
                join.control.append((edge.from_, edge.source_handle))
                if edge.from_ in self._known:
                    handles.setdefault(edge.from_, {})[edge.source_handle] = None
            elif edge.ordering:
                if not edge.optional:
                    join.required_order.append(edge.from_)
            else:
                join.groups.setdefault(edge.input_group, (edge.optional, []))[1].append(
                    edge.from_
                )
        # A CASE's choices: each branch, then "no branch" (None) unless it is exhaustive.
        self._choices: dict[str, tuple[str | None, ...]] = {
            case: (*names, *(() if case in exhaustive else (None,)))
            for case, names in handles.items()
        }
        # Only these edges can skip a node, so only they decide which CASEs a query branches on.
        self._parents = {
            n: self._joins[n].skip_sources() & self._known for n in self._nodes
        }
        self._skip_targets: dict[str, set[str]] = {case: set() for case in self._choices}
        for node, parents in self._parents.items():
            for parent in parents & self._choices.keys():
                self._skip_targets[parent].add(node)
        self._order = self._topological_order(graph_parents)

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

    def _topological_order(self, parents: dict[str, set[str]]) -> list[str] | None:
        """Return the level's nodes in topological order, or None when the edges cycle.

        Args:
            parents (`dict[str, set[str]]`):
                Node id -> the level's nodes with an edge into it.

        Returns:
            `list[str] | None`:
                The nodes, each after all of its parents, or None for a cyclic level.
        """
        indegree = {n: len(parents[n]) for n in self._nodes}
        children: dict[str, list[str]] = {n: [] for n in self._nodes}
        for node, sources in parents.items():
            for parent in sources:
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
            for case, reached in self._skip_targets.items()
            if case in closure and reached & closure
        }
        stack: list[tuple[int, frozenset[str], dict[str, str | None]]] = [
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
        self, node: str, skipped: set[str], choices: dict[str, str | None]
    ) -> bool:
        """Return whether `node` is skipped, given its skipped ancestors and CASE choices.

        Args:
            node (`str`):
                A node id in the level.
            skipped (`set[str]`):
                The skipped nodes among the ones before `node` in topological order.
            choices (`dict[str, str | None]`):
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
            not optional and all(self._edge_skipped(p, skipped, choices) for p in producers)
            for optional, producers in join.groups.values()
        ):
            return True
        return any(self._edge_skipped(p, skipped, choices) for p in join.required_order)

    @staticmethod
    def _edge_skipped(
        source: str, skipped: set[str], choices: dict[str, str | None]
    ) -> bool:
        """Return whether a data or ordering edge from `source` is skipped.

        Args:
            source (`str`):
                The edge's producer.
            skipped (`set[str]`):
                The skipped nodes so far.
            choices (`dict[str, str | None]`):
                The branch each running CASE picked; `None` is no branch.

        Returns:
            `bool`:
                Whether `source` is skipped, or it is a CASE that picked a branch other than
                `else:` or no branch.
        """
        # The runtime gives a non-control edge from a CASE the default handle, so it is taken
        # only when the CASE routes `else:` or matches nothing.
        return source in skipped or choices.get(source) not in (None, DEFAULT_HANDLE)


def null_skipped_outputs(
    flow_outputs: list[FlowOutput], model: SkipModel
) -> list[FlowOutput]:
    """Give each flow output whose producers may all be skipped the explicit null default.

    Args:
        flow_outputs (`list[FlowOutput]`):
            The flow level's declared outputs.
        model (`SkipModel`):
            The skip model of the same level.

    Returns:
        `list[FlowOutput]`:
            The outputs in order. A co-skipping source `${E}` whose producers may all be
            skipped becomes `${E:-null}` when `E` is one reference, else `${(E) :- null}`; its
            origin and declared contract are kept. Every other output is returned as is.

    Example:
        >>> edges = [Edge(id="t", from_="c", to="a", source_handle="then")]
        >>> model = SkipModel(["c", "a"], edges, set())
        >>> null_skipped_outputs([FlowOutput("v", "${a.output}")], model)[0].from_
        '${a.output:-null}'
    """
    rewritten = []
    for output in flow_outputs:
        source = binding_source(output.binding)
        if (
            isinstance(source, str)
            and binding_co_skips(source)
            and model.may_all_skip(_binding_producers(output.binding))
        ):
            # A co-skipping source is exactly one whole `${...}` span.
            interior = source[2:-1]
            if direct_reference_path(source) is not None:
                new = "${" + interior + ":-null}"
            else:
                new = "${(" + interior + ") :- null}"
            output = replace(output, binding=replace(output.binding, source=new))
        rewritten.append(output)
    return rewritten
