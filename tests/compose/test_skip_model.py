"""The skip model: which nodes of a flow level CASE routing may skip, and together."""

from itertools import count

from yian.compile.model import Edge
from yian.compose.parse.file import CaseDescriptor
from yian.compose.validate.case_exhaustive import case_on_exhaustive
from yian.compose.validate.skips import SkipModel, exhaustive_cases
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
    m = model("c a b", route(), {"c"})
    assert m.may_all_skip({"a"})
    assert not m.may_all_skip({"c"})
    assert not m.may_all_skip(set())


def test_producer_outside_the_level_never_skips():
    m = model("c a b", route())
    assert not m.may_all_skip({"a", "__start__"})
    assert not m.may_skip_while_running({"a", "__start__"}, None)


def test_joint_targets_skip_only_when_case_is_not_exhaustive():
    assert not model("c a b", route(), {"c"}).may_all_skip({"a", "b"})
    assert model("c a b", route()).may_all_skip({"a", "b"})


def _nested() -> list[Edge]:
    return [
        ctl("outer", "then", "inner"),
        ctl("outer", "else", "rejected"),
        ctl("inner", "then", "order"),
        ctl("inner", "else", "hold"),
    ]


def test_nested_cases():
    both = model("outer inner rejected order hold", _nested(), {"outer", "inner"})
    assert not both.may_all_skip({"order", "hold", "rejected"})
    assert both.may_all_skip({"order"})
    inner_only = model("outer inner rejected order hold", _nested(), {"inner"})
    assert inner_only.may_all_skip({"order", "hold", "rejected"})


def test_required_data_group():
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
    edges = [*route(), order("a", "dep"), order("a", "after", optional=True)]
    m = model("c a b dep after", edges, {"c"})
    assert m.may_all_skip({"dep"})
    assert not m.may_all_skip({"after"})


def test_case_reading_a_maybe_skipped_node_may_skip_with_all_its_targets():
    edges = [*route(), data("a", "c2"), ctl("c2", "then", "x"), ctl("c2", "else", "y")]
    m = model("c a b c2 x y", edges, {"c", "c2"})
    assert m.may_all_skip({"x", "y"})


def test_two_cases_target_one_node():
    edges = [ctl("c1", "then", "n"), ctl("c2", "then", "n")]
    m = model("c1 c2 n", edges)
    assert m.may_all_skip({"n"})
    edges = [ctl("c1", "then", "n"), ctl("c2", "then", "n"), ctl("c2", "else", "m")]
    m = model("c1 c2 n m", edges, {"c2"})
    # c2 always takes a branch, and n dies only when both c1 and c2 skip their edge to it.
    assert m.may_all_skip({"n"})
    assert not m.may_all_skip({"n", "m"})
    m = model("c1 c2 n", [ctl("c1", "then", "n"), ctl("c2", "then", "n")], {"c1"})
    assert not m.may_all_skip({"n"})


def test_may_skip_while_running():
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
    edges = [ctl("c", "then", "a"), data("a", "b"), data("b", "a")]
    m = model("c a b", edges)
    assert not m.may_all_skip({"a"})
    assert not m.may_skip_while_running({"a"}, None)


def _many_cases(**kw) -> SkipModel:
    """`c` (exhaustive) routes `a`/`b`; `x` is `g`'s only route; ten unrelated CASEs feed all."""
    edges = [*route(), ctl("g", "then", "x")]
    nodes = ["g", "c", "a", "b", "x"]
    for i in range(10):
        edges += [ctl(f"k{i}", "then", f"t{i}")]
        edges += [data(f"t{i}", n, f"in{i}", optional=True) for n in ("a", "x")]
        nodes += [f"k{i}", f"t{i}"]
    return SkipModel(nodes, edges, {"c"}, **kw)


def test_world_limit_falls_back_safely():
    exact = _many_cases()
    assert not exact.may_all_skip({"a", "b"})
    assert exact.may_skip_while_running({"x"}, None)
    bounded = _many_cases(limit=1)
    assert bounded.may_all_skip({"a", "b"})
    assert not bounded.may_skip_while_running({"x"}, None)


def _on_case(**kw) -> CaseDescriptor:
    return CaseDescriptor(
        id="route",
        on="${cls.output}",
        cases=[{"when": "a", "then": "na"}, {"when": "b", "then": "nb"}],
        **kw,
    )


def _cls(typ: Type) -> dict[str, ValueContract]:
    return {"cls": ValueContract(typ)}


_AB = Type(kind=ValueKind.STRING, tags=frozenset({"a", "b"}))


def test_case_on_exhaustive():
    searched = CaseDescriptor(id="s", cases=[{"when": "${cls.output} == 'a'", "then": "na"}])
    string = Type.scalar(ValueKind.STRING)
    nullable = Type(kind=ValueKind.STRING, tags=frozenset({"a", "b"}), nullable=True)
    assert case_on_exhaustive(_on_case(else_="other"), _cls(string))
    assert case_on_exhaustive(_on_case(), _cls(_AB))
    assert not case_on_exhaustive(_on_case(), _cls(nullable))
    assert not case_on_exhaustive(_on_case(), _cls(string))
    assert not case_on_exhaustive(searched, _cls(_AB))
    abc = Type(kind=ValueKind.STRING, tags=frozenset({"a", "b", "c"}))
    assert not case_on_exhaustive(_on_case(), _cls(abc))


def test_exhaustive_cases():
    searched = CaseDescriptor(id="s", cases=[{"when": "${cls.output} == 'a'", "then": "na"}])
    descriptors = {"route": _on_case(), "s": searched}
    assert exhaustive_cases(descriptors, _cls(_AB)) == {"route"}
