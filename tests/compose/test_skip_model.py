"""The skip model: which nodes of a flow level CASE routing may skip, and together."""

from itertools import count

from yian.compile.model import Edge
from yian.compose.parse.file import CaseDescriptor
from yian.compose.validate.case_exhaustive import case_is_exhaustive, exhaustive_cases
from yian.compose.validate.skips import SkipModel
from yian.nodes.base import ELSE_HANDLE
from yian.nodes.contracts import ValueContract
from yian.typesys.values import Type, ValueKind

_ids = count()


def ctl(case: str, handle: str, target: str) -> Edge:
    """A CASE control edge `case -[handle]-> target`."""
    return Edge(id=f"e{next(_ids)}", from_=case, to=target, source_handle=handle)


def data(src: str, dst: str, group: str = "x", *, optional: bool = False) -> Edge:
    """A data edge feeding input group `group` of `dst`."""
    return Edge(
        id=f"e{next(_ids)}", from_=src, to=dst, input_group=group, optional=optional
    )


def order(src: str, dst: str, *, optional: bool = False) -> Edge:
    """An ordering edge: `depends_on` when required, `runs_after` when optional."""
    return Edge(
        id=f"e{next(_ids)}", from_=src, to=dst, ordering=True, optional=optional
    )


def model(nodes: str, edges: list[Edge], exhaustive: set[str] = frozenset(), **kw) -> SkipModel:
    """A skip model over the space-separated node ids `nodes`."""
    return SkipModel(nodes.split(), edges, exhaustive, **kw)


def route(case: str = "c") -> list[Edge]:
    """`case` routes `then -> a` and `else -> b`."""
    return [ctl(case, "then", "a"), ctl(case, "else", "b")]


def test_case_target_may_skip_and_root_may_not():
    """A CASE target may be skipped; the CASE itself, a root, may not."""
    m = model("c a b", route(), {"c"})
    assert m.may_all_skip({"a"})
    assert not m.may_all_skip({"c"})
    assert not m.may_all_skip(set())


def test_producer_outside_the_level_never_skips():
    """An id outside the level, such as START, never skips."""
    m = model("c a b", route())
    assert not m.may_all_skip({"a", "__start__"})
    assert not m.may_skip_while_running({"a", "__start__"}, None)


def test_joint_targets_skip_only_when_case_is_not_exhaustive():
    """Both targets of a CASE skip together only when it may match nothing."""
    assert not model("c a b", route(), {"c"}).may_all_skip({"a", "b"})
    assert model("c a b", route()).may_all_skip({"a", "b"})


def _nested() -> list[Edge]:
    """`outer` routes `inner`/`rejected`; `inner` routes `order`/`hold`."""
    return [
        ctl("outer", "then", "inner"),
        ctl("outer", "else", "rejected"),
        ctl("inner", "then", "order"),
        ctl("inner", "else", "hold"),
    ]


def test_nested_cases():
    """Leaves of nested exhaustive CASEs never skip together; one leaf may."""
    both = model("outer inner rejected order hold", _nested(), {"outer", "inner"})
    assert not both.may_all_skip({"order", "hold", "rejected"})
    assert both.may_all_skip({"order"})
    inner_only = model("outer inner rejected order hold", _nested(), {"inner"})
    assert inner_only.may_all_skip({"order", "hold", "rejected"})


def test_required_data_group():
    """A required group skips its node only when all producers skip; an optional one never."""
    edges = [
        ctl("c", "then", "pro"),
        ctl("c", "else", "caut"),
        data("pro", "both"),
        data("caut", "both"),
        data("pro", "one"),
        data("pro", "opt", optional=True),
    ]
    m = model("c pro caut both one opt", edges, {"c"})
    assert not m.may_all_skip({"both"})
    assert m.may_all_skip({"one"})
    assert not m.may_all_skip({"opt"})


def test_ordering_edges():
    """`depends_on` propagates a skip; `runs_after` does not."""
    edges = [*route(), order("a", "dep"), order("a", "after", optional=True)]
    m = model("c a b dep after", edges, {"c"})
    assert m.may_all_skip({"dep"})
    assert not m.may_all_skip({"after"})


def test_case_reading_a_maybe_skipped_node_may_skip_with_all_its_targets():
    """A CASE that may itself be skipped takes all its targets with it."""
    edges = [*route(), data("a", "c2"), ctl("c2", "then", "x"), ctl("c2", "else", "y")]
    m = model("c a b c2 x y", edges, {"c", "c2"})
    assert m.may_all_skip({"x", "y"})


def test_node_targeted_by_two_cases_may_skip():
    """A node two CASEs target dies when both skip their edge to it."""
    assert model("c1 c2 n", [ctl("c1", "then", "n"), ctl("c2", "then", "n")]).may_all_skip({"n"})


def test_node_targeted_by_an_exhaustive_case_may_still_skip():
    """`c2` always takes a branch, but `n` dies when `c2` picks `m` and `c1` matches nothing."""
    edges = [ctl("c1", "then", "n"), ctl("c2", "then", "n"), ctl("c2", "else", "m")]
    m = model("c1 c2 n m", edges, {"c2"})
    assert m.may_all_skip({"n"})
    assert not m.may_all_skip({"n", "m"})


def test_node_targeted_by_a_sole_branch_of_an_exhaustive_case_never_skips():
    """An exhaustive CASE with one branch always keeps that edge taken."""
    m = model("c1 c2 n", [ctl("c1", "then", "n"), ctl("c2", "then", "n")], {"c1"})
    assert not m.may_all_skip({"n"})


def test_depends_on_a_case_runs_on_every_branch():
    """An ordering edge from a CASE is not a branch: it skips only when the CASE is skipped."""
    edges = [ctl("gate", "then", "a"), ctl("gate", ELSE_HANDLE, "b"), order("gate", "after")]
    m = model("gate a b after", edges, {"gate"})
    assert not m.may_all_skip({"after"})
    assert m.may_skip_while_running({"a"}, "after")


def test_depends_on_a_skipped_case_skips():
    """An ordering edge from a CASE that an outer CASE skips co-skips its node."""
    edges = [ctl("outer", "then", "gate"), ctl("gate", "then", "a"), order("gate", "after")]
    m = model("outer gate a after", edges)
    assert m.may_all_skip({"after"})
    assert not m.may_skip_while_running({"gate"}, "after")


def test_runs_after_a_case_never_skips():
    """An optional ordering edge from a CASE never skips its node."""
    edges = [ctl("gate", "then", "a"), ctl("gate", ELSE_HANDLE, "b")]
    m = model("gate a b after", [*edges, order("gate", "after", optional=True)], {"gate"})
    assert not m.may_all_skip({"after"})


def test_required_data_from_a_case_runs_on_every_branch():
    """A required data edge from a CASE is taken on every route the CASE picks."""
    edges = [ctl("gate", "then", "a"), data("gate", "reader")]
    m = model("gate a reader", edges)
    assert not m.may_all_skip({"reader"})
    assert m.may_skip_while_running({"a"}, "reader")


def test_may_skip_while_running():
    """A consumer that co-skips on the producer never runs while it is skipped."""
    edges = [
        ctl("c", "then", "order"),
        ctl("c", "else", "hold"),
        data("order", "req"),
        data("order", "mid"),
        data("mid", "chain"),
        data("order", "opt", optional=True),
    ]
    m = model("c order hold req mid chain opt", edges, {"c"})
    assert not m.may_skip_while_running({"order"}, "req")
    assert not m.may_skip_while_running({"order"}, "chain")
    assert m.may_skip_while_running({"order"}, "opt")
    assert m.may_skip_while_running({"order"}, None)
    assert not m.may_skip_while_running({"order", "hold"}, None)


def test_cycle_reports_nothing():
    """A cyclic level reports nothing skippable."""
    edges = [ctl("c", "then", "a"), data("a", "b"), data("b", "a")]
    m = model("c a b", edges)
    assert not m.may_all_skip({"a"})
    assert not m.may_skip_while_running({"a"}, None)


def _many_cases(**kw) -> SkipModel:
    """`c` (exhaustive) routes `a`/`b`; `x` is `g`'s only route; ten more CASEs feed both.

    Each `t{i}` shares a required group of `a` and `x` with the root `r`, so the ten CASEs
    are branched on but can never skip `a` or `x`.
    """
    edges = [*route(), ctl("g", "then", "x")]
    nodes = ["r", "g", "c", "a", "b", "x"]
    for i in range(10):
        edges += [ctl(f"k{i}", "then", f"t{i}")]
        edges += [data(p, n, f"in{i}") for n in ("a", "x") for p in (f"t{i}", "r")]
        nodes += [f"k{i}", f"t{i}"]
    return SkipModel(nodes, edges, {"c"}, **kw)


def test_cases_behind_edges_that_cannot_skip_are_not_branched_on():
    """CASEs reached only by an optional group or `runs_after` add no assignments."""
    edges = [*route(), ctl("k1", "then", "t1"), ctl("k2", "then", "t2")]
    edges += [data("t1", "a", optional=True), order("t2", "b", optional=True)]
    m = model("c a b k1 t1 k2 t2", edges, {"c"}, limit=2)
    assert not m.may_all_skip({"a", "b"})


def test_world_limit_falls_back_safely():
    """Past the bound, `may_all_skip` says True and `may_skip_while_running` False."""
    exact = _many_cases()
    assert not exact.may_all_skip({"a", "b"})
    assert exact.may_skip_while_running({"x"}, None)
    bounded = _many_cases(limit=1)
    assert bounded.may_all_skip({"a", "b"})
    assert not bounded.may_skip_while_running({"x"}, None)


def _on_case(**kw) -> CaseDescriptor:
    """An `on: ${cls.output}` CASE listing tags `a` and `b`."""
    return CaseDescriptor(
        id="route",
        on="${cls.output}",
        cases=[{"when": "a", "then": "na"}, {"when": "b", "then": "nb"}],
        **kw,
    )


def _cls(typ: Type) -> dict[str, ValueContract]:
    """A producer map with `cls` of type `typ`."""
    return {"cls": ValueContract(typ)}


_AB = Type(kind=ValueKind.STRING, tags=frozenset({"a", "b"}))


def test_case_is_exhaustive():
    """`else:`, or `on:` over a non-nullable Literal with every tag listed."""
    searched = CaseDescriptor(id="s", cases=[{"when": "${cls.output} == 'a'", "then": "na"}])
    string = Type.scalar(ValueKind.STRING)
    nullable = Type(kind=ValueKind.STRING, tags=frozenset({"a", "b"}), nullable=True)
    assert case_is_exhaustive(_on_case(else_="other"), _cls(string))
    assert case_is_exhaustive(_on_case(), _cls(_AB))
    assert not case_is_exhaustive(_on_case(), _cls(nullable))
    assert not case_is_exhaustive(_on_case(), _cls(string))
    assert not case_is_exhaustive(searched, _cls(_AB))
    abc = Type(kind=ValueKind.STRING, tags=frozenset({"a", "b", "c"}))
    assert not case_is_exhaustive(_on_case(), _cls(abc))


def test_exhaustive_cases():
    """Only the exhaustive CASE descriptors are returned."""
    searched = CaseDescriptor(id="s", cases=[{"when": "${cls.output} == 'a'", "then": "na"}])
    descriptors = {"route": _on_case(), "s": searched}
    assert exhaustive_cases(descriptors, _cls(_AB)) == {"route"}


def test_nodes_and_has_cases():
    """The model exposes its node ids and whether any CASE routes within the level."""
    m = model("c a b", route())
    assert m.nodes == frozenset({"c", "a", "b"})
    assert m.has_cases
    assert not model("a b", [data("a", "b")]).has_cases


def _on_input(on: str = "${input.k}") -> CaseDescriptor:
    """An `on:` CASE over a flow input, listing tags `a` and `b`."""
    return CaseDescriptor(
        id="route", on=on, cases=[{"when": "a", "then": "na"}, {"when": "b", "then": "nb"}]
    )


def test_case_over_a_literal_input_is_exhaustive():
    """`on: ${input.k}` over a fully listed non-nullable Literal input always takes a branch."""
    nullable = Type(kind=ValueKind.STRING, tags=frozenset({"a", "b"}), nullable=True)
    abc = Type(kind=ValueKind.STRING, tags=frozenset({"a", "b", "c"}))
    record = Type(kind=ValueKind.OBJECT, fields={"kind": _AB}, required=frozenset({"kind"}))
    assert case_is_exhaustive(_on_input(), {}, {"k": ValueContract(_AB)})
    assert case_is_exhaustive(_on_input("${input.r.kind}"), {}, {"r": ValueContract(record)})
    assert not case_is_exhaustive(_on_input(), {}, {"k": ValueContract(nullable)})
    assert not case_is_exhaustive(_on_input(), {}, {"k": ValueContract(abc)})
    assert not case_is_exhaustive(_on_input(), {})
    assert exhaustive_cases({"route": _on_input()}, {}, {"k": ValueContract(_AB)}) == {"route"}
