"""The skip model: which nodes of one flow level CASE routing may skip, and together.

A node is skipped when the runtime join rule marks it dead: it has control edges (from CASEs)
and every one is skipped, or a required data group has every producer skipped, or a required
ordering edge (`depends_on`) has its producer skipped. A control edge is skipped when its CASE
is skipped or picked another branch. A data or ordering edge is skipped only when its producer
is skipped: one from a CASE is not a branch, so it is taken whichever branch the CASE picks. A
CASE that runs picks one of its branches, or no branch at all unless it is exhaustive. CASE conditions are treated
as independent, so the model enumerates every combination of the choices of the CASEs that
can skip an ancestor of the queried nodes. The default side of `:-` is checked as if it always
runs: `${a.output :- (b.output + 1)}` is rejected even when `b` runs whenever `a` is skipped,
which is a false positive only when `a` cannot be null while it runs.

The enumeration is bounded by `_WORLD_LIMIT` assignments (complete or cut short) per query.
Past the bound each query falls back to the answer that is safe for its caller:
`may_all_skip` says True (the output gets a null default it may not need) and
`may_skip_while_running` says False (nothing is reported). A level whose edges have a cycle
has no skippable nodes here; the cycle check reports the cycle.

`null_skipped_outputs` gives a flow output that reads only nodes that may all be skipped the
explicit null default, so it returns null instead of leaving END unreachable.
`check_skipped_arithmetic` rejects arithmetic over a value that may be null because its
producers were skipped while the reading node (or END) still runs.

Knows about: compiled edges, flow outputs, node wiring, binding producers, and expression
trees.
Never imports: runtime.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping, Sequence, Set
from dataclasses import dataclass, field, replace
from typing import Any, Optional

from lark import Token, Tree

from yian.compile.model import Edge, FlowOutput
from yian.compose.edges.producers import _binding_producers, _ref_producer
from yian.compose.helpers.errors import LoadDiagnostics
from yian.compose.helpers.output_contracts import direct_reference_parts, direct_reference_path
from yian.compose.nodes.base import BuildResult
from yian.compose.validate.typed_bindings import ResolveOrigin, _located_error
from yian.diagnostics import OriginRef, SourceRange
from yian.expr import (
    ExpressionError,
    binding_co_skips,
    expr_refs,
    parse_condition,
    scan_template,
)
from yian.expr.template import Span
from yian.nodes.contracts import Binding, as_binding, binding_source

_WORLD_LIMIT = 10_000

# The grammar's arithmetic trees: `+ - * / % **` and unary minus.
_ARITHMETIC = frozenset({"add", "sub", "mul", "div", "mod", "power", "neg"})


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
        # Only these edges can skip a node, so only they decide which nodes a query must model.
        self._parents = {
            n: self._joins[n].skip_sources() & self._known for n in self._nodes
        }
        # A CASE's choice matters only to the nodes its control edges reach; its data and
        # ordering edges are taken on every route.
        self._skip_targets: dict[str, set[str]] = {case: set() for case in self._choices}
        for node in self._nodes:
            for case, _ in self._joins[node].control:
                if case in self._skip_targets:
                    self._skip_targets[case].add(node)
        self._order = self._topological_order(graph_parents)

    @property
    def nodes(self) -> frozenset[str]:
        """`frozenset[str]`: The level's node ids."""
        return frozenset(self._known)

    @property
    def has_cases(self) -> bool:
        """`bool`: Whether some CASE routes to a node of the level; without one nothing skips."""
        return bool(self._choices)

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
        # A data or ordering edge carries no handle, so the runtime takes it on every route of
        # a CASE that runs; it is skipped only when its producer is.
        if any(
            not optional and all(p in skipped for p in producers)
            for optional, producers in join.groups.values()
        ):
            return True
        return any(p in skipped for p in join.required_order)


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
    if not model.has_cases:
        return list(flow_outputs)
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


def check_skipped_arithmetic(
    flow_wiring: Mapping[str, Mapping[str, Any]],
    flow_outputs: list[FlowOutput],
    model: SkipModel,
    build_results: Mapping[str, BuildResult],
    *,
    value_mode: bool = False,
    outputs_range: Optional[SourceRange] = None,
    resolve_origin: ResolveOrigin | None = None,
    diagnostics: LoadDiagnostics | None = None,
    searched_whens: Mapping[
        str, Sequence[tuple[int, Any, OriginRef | None, SourceRange | None]]
    ]
    | None = None,
) -> None:
    """Reject arithmetic over a value that may be null because CASE routing skipped it.

    An operand of `+ - * / % **` or unary `-` is null from a skip when it is a reference to a
    node's output, or a `|` / `:-` group of them, and every producer of one of its null
    alternatives is skipped. That is an error when some routing skips them while the binding's
    node still runs; a flow output's END always runs. Any other operand (a call, a nested
    operator, a literal) is never null from a skip here. A searched CASE's authored `when:`
    is checked the same way with the CASE as the reader, since a reference under `:-`, `|`
    or `:?` there does not skip the CASE with its producer.

    Args:
        flow_wiring (`Mapping[str, Mapping[str, Any]]`):
            Each node's `param -> source` wiring, a CASE's `on:` and `when:` sources included.
        flow_outputs (`list[FlowOutput]`):
            The flow level's outputs, after `null_skipped_outputs`.
        model (`SkipModel`):
            The skip model of the same level.
        build_results (`Mapping[str, BuildResult]`):
            The level's built nodes keyed by id, for the binding labels and fallback ranges.
        value_mode (`bool`, *optional*, defaults to `False`):
            Whether the flow has a single `output:` value, named `output` in errors.
        outputs_range (`SourceRange | None`, *optional*, defaults to `None`):
            The authored `output:` range, the fallback location of an output error.
        resolve_origin (`ResolveOrigin | None`, *optional*, defaults to `None`):
            Resolves a binding's registered origin to its source frame, the preferred location.
        diagnostics (`LoadDiagnostics | None`, *optional*, defaults to `None`):
            Factory for located load errors.
        searched_whens (`Mapping[str, Sequence[tuple[int, Any, OriginRef | None, SourceRange | None]]] | None`, *optional*, defaults to `None`):
            Each searched CASE's id mapped to one `(case index, authored when, origin,
            range)` per case: the origin is the preferred location, the range the fallback.
            A `when:` is checked like a binding, without following `and`/`or`: arithmetic
            behind a null guard is reported too.

    Raises:
        `LoadError`:
            At the first such binding or `when:`, naming it, the operand and its producers,
            and asking for a `:-` default.
    """
    if not model.has_cases:
        return
    diagnostics = diagnostics or LoadDiagnostics()
    known = model.nodes
    for node_id, wiring in flow_wiring.items():
        result = build_results.get(node_id)
        reserved = result.node.reserved_wiring_keys() if result is not None else set()
        for param, binding in wiring.items():
            found = _skipped_operand(binding, model, node_id, known)
            if found is None:
                continue
            label, fallback = f"input {param!r}", None
            if result is not None:
                if param in reserved:
                    # A reserved key (`over`, `until`) is a kind field, not an input.
                    label = f"{result.node.kind.value} {param}"
                    own = result.field_ranges.get(("field", param))
                else:
                    label = result.node.wiring_ref_label(param)
                    own = result.field_ranges.get(("input", param))
                fallback = (
                    own or result.field_ranges.get(("field", "inputs")) or result.node_range
                )
            # A synthetic param's label may already end in a colon (`on:`).
            raise _located_error(
                f"node {node_id!r} {label.rstrip(':')}: {_arithmetic_message(binding, *found)}",
                as_binding(binding),
                fallback,
                resolve_origin,
                diagnostics,
            )
    for node_id, whens in (searched_whens or {}).items():
        for index, when, origin, when_range in whens:
            if not isinstance(when, str):
                continue
            try:
                tree = parse_condition(when)
            except ExpressionError:
                continue  # the CASE build reports a malformed condition
            found = _skipped_operand_in(tree, model, node_id, known)
            if found is None:
                continue
            raise _located_error(
                f"node {node_id!r} cases[{index}].when: "
                f"{_arithmetic_text(when.strip(), *found)}",
                Binding(when, origin=origin),
                when_range,
                resolve_origin,
                diagnostics,
            )
    for output in flow_outputs:
        found = _skipped_operand(output.binding, model, None, known)
        if found is None:
            continue
        label = "output" if value_mode else f"output {output.name!r}"
        raise _located_error(
            f"{label}: {_arithmetic_message(output.binding, *found)}",
            output.binding,
            outputs_range,
            resolve_origin,
            diagnostics,
        )


def _skipped_operand(
    binding: Any, model: SkipModel, consumer: str | None, known: Set[str]
) -> tuple[Tree | Token, list[str]] | None:
    """Return the first arithmetic operand of `binding` that may be null from a skip.

    Args:
        binding (`Any`):
            A wiring or output binding; only a string source has expressions.
        model (`SkipModel`):
            The level's skip model.
        consumer (`str | None`):
            The node that evaluates the binding, or `None` for END.
        known (`Set[str]`):
            The level's node ids.

    Returns:
        `tuple[Tree | Token, list[str]] | None`:
            The operand and the producers whose skip makes it null (in source order), or
            `None` when no operand may be null while `consumer` runs.
    """
    source = binding_source(binding)
    if not isinstance(source, str):
        return None
    try:
        segments = scan_template(source)
    except ExpressionError:
        return None  # reference validation reports a malformed source
    for segment in segments:
        if not isinstance(segment, Span):
            continue
        found = _skipped_operand_in(segment.tree, model, consumer, known)
        if found is not None:
            return found
    return None


def _skipped_operand_in(
    expression: Tree | Token, model: SkipModel, consumer: str | None, known: Set[str]
) -> tuple[Tree | Token, list[str]] | None:
    """Return the first arithmetic operand of a parsed expression that may be null from a skip.

    Args:
        expression (`Tree | Token`):
            A parsed expression.
        model (`SkipModel`):
            The level's skip model.
        consumer (`str | None`):
            The node that evaluates the expression, or `None` for END.
        known (`Set[str]`):
            The level's node ids.

    Returns:
        `tuple[Tree | Token, list[str]] | None`:
            The operand and the producers whose skip makes it null (in source order), or
            `None` when no operand may be null while `consumer` runs.
    """
    if not isinstance(expression, Tree):
        return None  # a lone token has no operator
    for tree in expression.iter_subtrees_topdown():
        if tree.data not in _ARITHMETIC:
            continue
        for operand in tree.children:
            for group in _null_groups(operand, known):
                if group and model.may_skip_while_running(group, consumer):
                    producers = (_ref_producer(ref) for ref in expr_refs(operand))
                    return operand, [p for p in dict.fromkeys(producers) if p in group]
    return None


def _null_groups(operand: Tree | Token, known: Set[str]) -> list[frozenset[str]]:
    """Return the producer sets whose joint skip makes `operand` null.

    Args:
        operand (`Tree | Token`):
            One parsed operand of an arithmetic tree.
        known (`Set[str]`):
            The level's node ids; a reference to anything else (an input, `item`, `system`)
            is never skipped.

    Returns:
        `list[frozenset[str]]`:
            The operand is null when every producer of one set is skipped. A `null` literal
            gives one empty set (always null); an operand that is never null from a skip
            gives no sets.
    """
    if isinstance(operand, Token):
        return [frozenset()] if operand.type == "NULL" else []
    if operand.data == "refcall":
        parts = direct_reference_parts(operand)
        return _reference_groups(".".join(parts), known) if parts is not None else []
    if operand.data in ("coalesce", "default_expr"):
        # Null only when every alternative is null: one set from each, unioned.
        combined = [frozenset()]
        for child in operand.children:
            groups = _null_groups(child, known)
            combined = list(dict.fromkeys(a | b for a in combined for b in groups))
        return combined
    return []


def _reference_groups(path: str, known: Set[str]) -> list[frozenset[str]]:
    """Return `[{producer}]` for a reference to a level node's output, else `[]`.

    Args:
        path (`str`):
            A dotted reference path.
        known (`Set[str]`):
            The level's node ids.

    Returns:
        `list[frozenset[str]]`:
            The one producer set that makes the reference null when skipped, or none.
    """
    producer = _ref_producer(path)
    return [frozenset({producer})] if producer in known else []


def _arithmetic_message(
    binding: Any, operand: Tree | Token, producers: list[str]
) -> str:
    """Return the error text for arithmetic on an operand that may be null from a skip.

    Args:
        binding (`Any`):
            The binding whose source does the arithmetic.
        operand (`Tree | Token`):
            The operand that may be null.
        producers (`list[str]`):
            The nodes whose skip makes it null.

    Returns:
        `str`:
            The message, without the binding's name.
    """
    return _arithmetic_text(binding_source(binding), operand, producers)


def _arithmetic_text(source: str, operand: Tree | Token, producers: list[str]) -> str:
    """Return the error text for arithmetic in `source` on an operand that may be null.

    Args:
        source (`str`):
            The authored text that does the arithmetic, shown as written.
        operand (`Tree | Token`):
            The operand that may be null.
        producers (`list[str]`):
            The nodes whose skip makes it null.

    Returns:
        `str`:
            The message, without the reader's name.
    """
    text = _operand_text(operand)
    hint = _default_hint(operand)
    quoted = [repr(p) for p in producers]
    if len(quoted) == 1:
        who = f"node {quoted[0]} is"
    else:
        who = f"nodes {', '.join(quoted[:-1])} and {quoted[-1]} are"
    return (
        f"{source} does arithmetic on {text}, which is null when {who} "
        f"skipped; give it a default with :-, e.g. {hint}"
    )


def _default_hint(operand: Tree | Token) -> str:
    """Return the operand with a `:- <default>` placeholder, replacing a trailing `:- null`.

    Args:
        operand (`Tree | Token`):
            The operand that may be null.

    Returns:
        `str`:
            Parenthesized text such as `(order.output.qty :- <default>)`.
    """
    if (
        isinstance(operand, Tree)
        and operand.data == "default_expr"
        and isinstance(operand.children[-1], Token)
        and operand.children[-1].type == "NULL"
    ):
        kept = [_operand_text(child) for child in operand.children[:-1]]
        return "(" + " :- ".join([*kept, "<default>"]) + ")"
    return f"({_operand_text(operand)} :- <default>)"


def _operand_text(operand: Tree | Token) -> str:
    """Return source text for an operand that may be null: a reference or a `|`/`:-` group.

    Args:
        operand (`Tree | Token`):
            A reference, a wrapped reference, `null`, or a coalesce or default of them.

    Returns:
        `str`:
            The operand's text; a group is parenthesized.
    """
    if isinstance(operand, Token):
        return str(operand)
    parts = direct_reference_parts(operand)
    if parts is not None:
        return ".".join(parts)
    joiner = " | " if operand.data == "coalesce" else " :- "
    return "(" + joiner.join(_operand_text(child) for child in operand.children) + ")"
