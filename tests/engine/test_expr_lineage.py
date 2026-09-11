"""Feedback follows the values and selected operations actually evaluated."""

from dataclasses import FrozenInstanceError, asdict
import json
import sys

import pytest

from agent_composer.expr.builtins import TEMPLATE_FNS
from agent_composer.expr.expressions import ExpressionError, ResolveMode, eval_expr
from agent_composer.expr.grammar import parse_expr
from agent_composer.expr.lineage import (
    ResolvedValue,
    SourceRef,
    ValueRoute,
    route_feedback,
    validate_feedback,
)
from agent_composer.expr.template import eval_binding, eval_template, scan_template


class Sources:
    """Single-read references backed by distinct accepted execution identities."""

    def __init__(self, **values):
        self.values = values
        self.identities = {name: SourceRef("run", i) for i, name in enumerate(values)}
        self.reads = []

    def resolve(self, path):
        self.reads.append(path)
        head, *parts = path.split(".")
        if head not in self.values:
            return ResolvedValue(None, False)
        value = self.values[head]
        for part in parts:
            if not isinstance(value, dict) or part not in value:
                return ResolvedValue(None, False)
            value = value[part]
        root = self.identities[head]
        return ResolvedValue(
            value, True, SourceRef(root.run_id, root.execution_seq, tuple(parts))
        )

    def lookup(self, source):
        assert not source.path
        return next(
            self.values[name]
            for name, identity in self.identities.items()
            if identity == source
        )

    def evaluate(self, expression):
        return eval_expr(parse_expr(expression), self.resolve, trace=True)

    def feedback(self, traced, feedback):
        return route_feedback(traced.route, feedback, self.lookup)


def messages(routed):
    return [(c.source, c.feedback) for c in routed.contributions]


def test_direct_projection_lifts_feedback_into_source_shape():
    original = {"tags": ["one", "two"], "summary": "text"}
    root = SourceRef("run", 7)
    reads = []

    def resolve(path):
        reads.append(path)
        if path == "producer.output.tags":
            return ResolvedValue(original["tags"], True, SourceRef("run", 7, ("tags",)))
        return ResolvedValue("two", True, SourceRef("run", 7, ("tags", 1)))

    result = eval_expr(parse_expr("producer.output.tags"), resolve, trace=True)
    routed = route_feedback(result.route, [None, "Bad."], lambda source: original)
    assert result.value is original["tags"]
    assert reads == ["producer.output.tags"]
    assert [(c.source, c.feedback) for c in routed.contributions] == [
        (root, {"tags": [None, "Bad."]})
    ]
    projected = eval_expr(parse_expr("tag"), resolve, trace=True)
    routed = route_feedback(projected.route, "Bad.", lambda source: original)
    assert [(c.source, c.feedback) for c in routed.contributions] == [
        (root, {"tags": [None, "Bad."]})
    ]


def test_accepted_null_has_a_source_but_absence_does_not():
    sources = Sources(accepted=None)
    present = eval_binding("${accepted}", sources.resolve, trace=True)
    absent = eval_template(scan_template("${absent}"), sources.resolve, trace=True)
    assert present.value is absent.value is None
    assert messages(sources.feedback(present, "Supply a value.")) == [
        (sources.identities["accepted"], "Supply a value.")
    ]
    assert messages(sources.feedback(absent, "Supply a value.")) == []
    assert messages(sources.feedback(present, None)) == []
    assert messages(sources.feedback(present, "")) == [
        (sources.identities["accepted"], "")
    ]
    for name in ("accepted", "absent"):
        with pytest.raises(ExpressionError, match="unresolved"):
            eval_expr(
                parse_expr(name),
                sources.resolve,
                mode=ResolveMode.STRICT_RAISE,
                trace=True,
            )


@pytest.mark.parametrize("winning", [0, False, "", ["item"]])
def test_coalesce_and_default_drop_losing_arm_routes(winning):
    sources = Sources(empty=None, winner=winning, losing="unused")
    for expression, reads in (
        ("empty | winner | losing", ["empty", "winner"]),
        ("empty :- winner", ["empty", "winner"]),
        ("winner :- losing", ["winner"]),
        ('winner :? "Required"', ["winner"]),
    ):
        sources.reads.clear()
        result = sources.evaluate(expression)
        assert result.value == winning
        assert sources.reads == reads
        assert messages(sources.feedback(result, "Keep this.")) == [
            (sources.identities["winner"], "Keep this.")
        ]
    for expression in ('empty :- "literal"', "empty | absent | null"):
        assert (
            messages(sources.feedback(sources.evaluate(expression), "Feedback.")) == []
        )
    null_default = sources.evaluate("absent :- empty")
    assert messages(sources.feedback(null_default, "Still empty.")) == [
        (sources.identities["empty"], "Still empty.")
    ]


def test_list_constructor_preserves_nested_routes():
    sources = Sources(producer={"tags": ["one", "two"], "summary": "long"})
    result = sources.evaluate("[producer.tags, [producer.summary], producer.tags]")
    assert result.value == [["one", "two"], ["long"], ["one", "two"]]
    feedback = [[None, "Bad."], ["Too long."], "List note."]
    assert messages(sources.feedback(result, feedback)) == [
        (sources.identities["producer"], {"tags": [None, "Bad."]}),
        (sources.identities["producer"], {"summary": "Too long."}),
        (sources.identities["producer"], {"tags": "List note."}),
    ]
    assert len(messages(sources.feedback(result, "Whole list."))) == 3
    assert messages(sources.feedback(result, [None, [None], [None, None]])) == []
    with pytest.raises(ValueError, match="exactly 3"):
        sources.feedback(result, [None])
    with pytest.raises(ValueError, match="exactly 2"):
        sources.feedback(result, [["Bad."], None, None])
    record = ValueRoute.record(
        (("items", result.route), ("constant", ValueRoute.literal(1)))
    )
    assert messages(
        route_feedback(record, {"items": feedback}, sources.lookup)
    ) == messages(sources.feedback(result, feedback))
    with pytest.raises(ValueError, match="unknown field"):
        route_feedback(record, {"unknown": "Bad."}, sources.lookup)


def test_concat_splits_feedback_at_recorded_operand_lengths():
    sources = Sources(left=["a"], right=["b", "c"])
    result = sources.evaluate("left + right")
    assert result.value == ["a", "b", "c"]
    assert messages(sources.feedback(result, [None, "Bad.", None])) == [
        (sources.identities["right"], ["Bad.", None])
    ]
    assert messages(sources.feedback(result, "Too many.")) == [
        (sources.identities["left"], "Too many."),
        (sources.identities["right"], "Too many."),
    ]
    repeated = sources.evaluate("left + left")
    assert messages(sources.feedback(repeated, ["First.", "Second."])) == [
        (sources.identities["left"], ["First."]),
        (sources.identities["left"], ["Second."]),
    ]
    assert messages(sources.feedback(repeated, "Whole.")) == [
        (sources.identities["left"], "Whole."),
        (sources.identities["left"], "Whole."),
    ]
    assert messages(
        sources.feedback(sources.evaluate("[] + right + []"), [None, "Bad."])
    ) == [(sources.identities["right"], [None, "Bad."])]


@pytest.mark.parametrize("expression", ["items * count", "count * items"])
def test_repetition_preserves_copy_messages_and_shape_context(expression):
    sources = Sources(items=["same", "same"], count=2)
    result = sources.evaluate(expression)
    assert result.value == ["same"] * 4
    positional = sources.feedback(result, ["First.", None, "Second.", None])
    assert messages(positional) == [
        (sources.identities["items"], ["First.", None]),
        (sources.identities["items"], ["Second.", None]),
    ]
    assert [c.context[0].positions for c in positional.contributions] == [
        ((0,), (1,)),
        ((2,), (3,)),
    ]
    whole = sources.feedback(result, "Too many copies.")
    by_source = {c.source: c for c in whole.contributions}
    assert len(whole.contributions) == 2
    content = by_source[sources.identities["items"]]
    count = by_source[sources.identities["count"]]
    assert content.feedback == count.feedback == "Too many copies."
    assert content.context[0].role == "content"
    assert count.context[0].role == "count"
    assert (
        content.context[0].positions == count.context[0].positions == ((0, 2), (1, 3))
    )
    assert count.context[0].result == result.value
    for count_value in (0, -2):
        sources.values["count"] = count_value
        empty = sources.evaluate(expression)
        assert empty.value == []
        assert sources.feedback(empty, []).contributions == ()
        whole_empty = sources.feedback(empty, "Expected content.")
        assert len(whole_empty.contributions) == 2
        assert all(
            c.context[0].positions == ((), ()) for c in whole_empty.contributions
        )


@pytest.mark.parametrize(
    "expression, expected, names",
    [
        ("x + y", 7, ("x", "y")),
        ("x - y", -1, ("x", "y")),
        ("x * y", 12, ("x", "y")),
        ("x / y", 0.75, ("x", "y")),
        ("x % y", 3, ("x", "y")),
        ("x ** y", 81, ("x", "y")),
        ("-x", -3, ("x",)),
        ("x < y", True, ("x", "y")),
        ("x == y", False, ("x", "y")),
        ("x in items", True, ("x", "items")),
        ("x not in items", False, ("x", "items")),
        ("not x", False, ("x",)),
        ("x or absent", True, ("x",)),
        ("x and y", True, ("x", "y")),
        ("zero and absent", False, ("zero",)),
    ],
)
def test_scalar_and_interpolation_text_stays_unchanged(expression, expected, names):
    sources = Sources(x=3, y=4, items=[3, 4], zero=0)
    result = sources.evaluate(expression)
    assert result.value == expected
    assert sources.reads == list(names)
    routed = sources.feedback(result, "Original text.")
    assert messages(routed) == [
        (sources.identities[name], "Original text.") for name in names
    ]
    for contribution in routed.contributions:
        assert contribution.context[0].result == expected
        assert contribution.context[0].expression
        assert contribution.context[0].sources == tuple(
            sources.identities[name] for name in names
        )
    rendered = eval_binding("x=${x}; items=${items}", sources.resolve, trace=True)
    assert rendered.value == "x=3; items=[3, 4]"
    routed = sources.feedback(rendered, "Be concise.")
    assert messages(routed) == [
        (sources.identities["x"], "Be concise."),
        (sources.identities["items"], "Be concise."),
    ]
    assert [c.context[0].span for c in routed.contributions] == [(2, 3), (11, 17)]
    assert all(c.context[0].result == rendered.value for c in routed.contributions)


@pytest.mark.parametrize(
    "name, function, suffix, expected, feedback",
    [
        (
            "reorder",
            lambda values: [values[1], values[0]],
            "",
            ["b", "a"],
            ["Wrong.", None],
        ),
        (
            "record",
            lambda values: {"second": values[1], "first": values[0]},
            "",
            {"second": "b", "first": "a"},
            {"second": "Wrong."},
        ),
        ("count", lambda values: len(values), "", 2, "Too many."),
        (
            "upper",
            lambda values: [values[1], values[0]],
            "",
            ["b", "a"],
            [None, "Wrong."],
        ),
        (
            "projected",
            lambda values: {"result": [values[1], values[0]]},
            ".result",
            ["b", "a"],
            ["Wrong.", None],
        ),
    ],
)
def test_opaque_functions_stop_feedback_and_execute_once(
    monkeypatch, name, function, suffix, expected, feedback
):
    calls = []

    def invoke(values):
        calls.append(values)
        return function(values)

    monkeypatch.setitem(TEMPLATE_FNS, name, invoke)
    sources = Sources(items=["a", "b"])
    result = sources.evaluate(f"{name}(items){suffix}")
    assert result.value == expected
    for supplied in (feedback, "Original feedback."):
        routed = sources.feedback(result, supplied)
        assert routed.contributions == ()
        assert len(routed.stopped) == 1
        assert routed.stopped[0].feedback == supplied
        detail = routed.stopped[0].context[-1]
        assert detail.result == expected
        assert detail.expression == f"{name}(items){suffix}"
        assert detail.sources == (sources.identities["items"],)
    assert sources.reads == ["items"]
    assert calls == [["a", "b"]]
    assert sources.feedback(result, None).stopped == ()


@pytest.mark.parametrize(
    "expression, item_index", [("left + right", 1), ("right * 2", 1)]
)
def test_composed_item_route_projects_fields_and_preserves_null_presence(
    expression, item_index
):
    sources = Sources(left=[{"answer": "old"}], right=[{"answer": None}])
    over = sources.evaluate(expression)
    item = ResolvedValue(
        over.value[item_index], True, over.route.project((item_index,))
    )
    sources.reads.clear()
    result = eval_binding("${item.answer}", sources.resolve, item=item, trace=True)
    assert result.value is None
    assert sources.reads == []
    assert messages(sources.feedback(result, "Fill this.")) == [
        (sources.identities["right"], [{"answer": "Fill this."}])
    ]
    missing = eval_binding("${item.absent}", sources.resolve, item=item, trace=True)
    assert messages(sources.feedback(missing, "Fill this.")) == []
    fallback = eval_binding(
        '${item.answer :- "default"}', sources.resolve, item=item, trace=True
    )
    assert fallback.value == "default"
    assert messages(sources.feedback(fallback, "Feedback.")) == []
    local_reads = []

    def local(path):
        local_reads.append(path)
        return ResolvedValue(None, True, over.route.project((item_index, "answer")))

    local_result = eval_binding("${local}", local, trace=True)
    assert local_reads == ["local"]
    assert messages(sources.feedback(local_result, "Fill this.")) == messages(
        sources.feedback(result, "Fill this.")
    )


def test_non_json_opaque_temporary_preserves_interpolation_and_projection(monkeypatch):
    calls = []
    renders = []

    class Temporary:
        def __str__(self):
            renders.append("str")
            return "rendered"

        def __repr__(self):
            raise AssertionError("tracing must not request repr")

    def temporary():
        calls.append("temporary")
        return Temporary()

    monkeypatch.setitem(TEMPLATE_FNS, "temporary", temporary)
    untraced = eval_binding("value=${temporary()}", lambda path: None)
    assert untraced == "value=rendered"
    assert calls == ["temporary"] and renders == ["str"]
    traced = eval_binding(
        "value=${temporary()}", lambda path: ResolvedValue(None, False), trace=True
    )
    assert traced.value == untraced
    routed = route_feedback(traced.route, "Shorten.", lambda source: None)
    assert routed.contributions == ()
    detail = routed.stopped[0].context[-1]
    assert detail.result_json is None
    assert detail.result_type.endswith(".Temporary")
    assert routed.stopped[0].context[0].result == untraced
    assert calls == ["temporary", "temporary"] and renders == ["str", "str"]

    monkeypatch.setitem(
        TEMPLATE_FNS, "record", lambda: {"answer": "ok", "temporary": Temporary()}
    )
    projected = eval_binding(
        "${record().answer}", lambda path: ResolvedValue(None, False), trace=True
    )
    assert projected.value == "ok"
    stopped = route_feedback(projected.route, "Explain.", lambda source: None).stopped
    assert stopped[0].context[-1].result == "ok"
    assert renders == ["str", "str"]


def test_routes_are_immutable_serializable_and_keep_opaque_snapshots(monkeypatch):
    produced = {"tags": ["original", "other"]}
    monkeypatch.setitem(TEMPLATE_FNS, "snapshot", lambda: produced)
    result = eval_binding(
        "${snapshot()}", lambda path: ResolvedValue(None, False), trace=True
    )
    original_encoding = json.dumps(asdict(result.route))
    produced["tags"].append("later")
    stopped = route_feedback(
        result.route, {"tags": ["Bad.", None]}, lambda source: None
    ).stopped
    assert stopped[0].context[-1].result == {"tags": ["original", "other"]}
    assert json.dumps(asdict(result.route)) == original_encoding
    with pytest.raises(FrozenInstanceError):
        result.route.kind = "literal"
    assert route_feedback(result.route, {}, lambda source: None).stopped == ()
    assert (
        route_feedback(
            result.route, {"tags": [None, None]}, lambda source: None
        ).stopped
        == ()
    )
    with pytest.raises(ValueError, match="exactly 2"):
        route_feedback(result.route, {"tags": ["Too short."]}, lambda source: None)
    sources = Sources(items=["one", "two"], count=2)
    repeat = sources.evaluate("items * count")
    assert all(child.value_json is None for child in repeat.route.children)
    assert json.loads(json.dumps(asdict(repeat.route)))["count"] == 2


def test_opaque_stops_preserve_nested_repetition_and_interpolation_context(monkeypatch):
    calls = []

    def reorder(values):
        calls.append("call")
        return list(reversed(values))

    monkeypatch.setitem(TEMPLATE_FNS, "reorder", reorder)
    sources = Sources(items=["one", "two"], count=2)
    repeated = sources.evaluate("reorder(items) * count")
    routed = sources.feedback(repeated, ["First.", None, "Second.", None])
    assert routed.contributions == ()
    assert [stopped.feedback for stopped in routed.stopped] == [
        ["First.", None],
        ["Second.", None],
    ]
    assert [stopped.context[0].positions for stopped in routed.stopped] == [
        ((0,), (1,)),
        ((2,), (3,)),
    ]
    assert calls == ["call"]
    short_circuit = sources.evaluate("count or reorder(items)")
    assert sources.feedback(short_circuit, "Note.").stopped == ()
    assert calls == ["call"]
    interpolated = eval_binding("items=${reorder(items)}", sources.resolve, trace=True)
    stopped = sources.feedback(interpolated, "Note.").stopped[0]
    assert stopped.feedback == "Note."
    assert stopped.context[0].span == (6, 20)
    assert stopped.context[0].result == "items=['two', 'one']"
    assert calls == ["call", "call"]


def test_unrepresentable_temporary_does_not_prevent_the_outer_call(monkeypatch):
    calls = []

    def nested():
        calls.append("nested")
        value = "leaf"
        for _ in range(sys.getrecursionlimit() + 10):
            value = [value]
        return value

    def ignore(value):
        calls.append("ignore")
        return "ok"

    monkeypatch.setitem(TEMPLATE_FNS, "nested", nested)
    monkeypatch.setitem(TEMPLATE_FNS, "ignore", ignore)
    assert eval_binding("${ignore(nested())}", lambda path: None) == "ok"
    traced = eval_binding(
        "${ignore(nested())}", lambda path: ResolvedValue(None, False), trace=True
    )
    assert traced.value == "ok"
    stopped = route_feedback(traced.route, "Feedback.", lambda source: None).stopped
    assert stopped[0].context[-1].result == "ok"
    assert calls == ["nested", "ignore", "nested", "ignore"]


@pytest.mark.parametrize(
    "expression, expected, render_count",
    [
        ("temporary_list() * 2", "items=[temporary, temporary]", 2),
        ("temporary_list() + []", "items=[temporary]", 1),
        ("[temporary_list()] * 2", "items=[[temporary], [temporary]]", 2),
    ],
)
def test_unavailable_snapshots_keep_container_structure(
    monkeypatch, expression, expected, render_count
):
    calls = []
    renders = []

    class Temporary:
        def __repr__(self):
            renders.append("repr")
            return "temporary"

    def temporary_list():
        calls.append("call")
        return [Temporary()]

    monkeypatch.setitem(TEMPLATE_FNS, "temporary_list", temporary_list)
    result = eval_binding(
        "items=${" + expression + "}",
        lambda path: ResolvedValue(None, False),
        trace=True,
    )
    assert result.value == expected
    routed = route_feedback(result.route, "Shorten.", lambda source: None)
    assert routed.contributions == ()
    assert [stop.feedback for stop in routed.stopped] == ["Shorten."]
    assert routed.stopped[0].context[0].result == expected
    assert calls == ["call"]
    assert renders == ["repr"] * render_count


def test_unavailable_snapshot_projects_available_fields_and_composed_items(monkeypatch):
    class Temporary:
        def __repr__(self):
            raise AssertionError("snapshot capture and routing must not render")

    item_value = {"answer": "ok", "temporary": Temporary()}
    resolve = lambda path: ResolvedValue(None, False)
    plain = eval_binding("${item.answer}", resolve, item=item_value, trace=True)
    assert plain.value == "ok"
    assert (
        route_feedback(plain.route, "Shorten.", lambda source: None).contributions == ()
    )

    calls = []

    def temporary_list():
        calls.append("call")
        return [item_value]

    monkeypatch.setitem(TEMPLATE_FNS, "temporary_list", temporary_list)
    over = eval_binding("${temporary_list() * 2}", resolve, trace=True)
    item = ResolvedValue(over.value[1], True, over.route.project((1,)))
    projected = eval_binding("${item.answer}", resolve, item=item, trace=True)
    assert projected.value == "ok"
    stopped = route_feedback(projected.route, "Shorten.", lambda source: None).stopped
    assert [stop.feedback for stop in stopped] == [[{"answer": "Shorten."}]]
    assert calls == ["call"]


def test_structural_routes_retain_selection_and_position_context():
    sources = Sources(left=["a"], right=["b", "c"], record={"tags": ["x", "y"]})
    combined = sources.evaluate("left + right")
    contributions = sources.feedback(combined, "Too long.").contributions
    details = [
        next(c for c in part.context if c.role == "concat") for part in contributions
    ]
    assert [c.operand for c in details] == [0, 1]
    assert [c.span for c in details] == [(0, 1), (1, 3)]
    assert [c.positions for c in details] == [((0,),), ((1,), (2,))]
    assert all(
        c.expression == "(left + right)" and c.result == combined.value for c in details
    )
    assert details[0].result_json is details[1].result_json
    assert [part.context[-1].sources for part in contributions] == [
        (sources.identities["left"],),
        (sources.identities["right"],),
    ]
    assert all(part.context[-1].result_json is None for part in contributions)

    repeated = sources.feedback(
        sources.evaluate("left + left"), "Same text."
    ).contributions
    assert [
        next(c for c in part.context if c.role == "concat").span for part in repeated
    ] == [(0, 1), (1, 2)]
    nested = sources.evaluate("[[left], right]")
    nested_contribution = sources.feedback(nested, [["Bad."], None]).contributions[0]
    items = [c for c in nested_contribution.context if c.role == "item"]
    assert [c.operand for c in items] == [0, 0]
    assert [c.result for c in items] == [nested.value, [["a"]]]

    record = ValueRoute.record(
        (
            ("other", sources.evaluate("left").route),
            ("chosen", sources.evaluate("right").route),
        )
    )
    projected = route_feedback(
        record.project(("chosen", 1)), "Bad.", sources.lookup
    ).contributions[0]
    assert projected.feedback == [None, "Bad."]
    assert projected.context[0].role == "projection"
    assert projected.context[0].operand == ("chosen", 1)
    assert projected.context[0].result == "c"
    field = next(c for c in projected.context if c.role == "field")
    assert field.operand == "chosen"
    assert field.result == {"other": ["a"], "chosen": ["b", "c"]}
    assert field.expression

    direct = sources.feedback(
        sources.evaluate("record.tags"), [None, "Bad."]
    ).contributions[0]
    assert direct.context[-1].role == "projection"
    assert direct.context[-1].sources[0].path == ("tags",)


@pytest.mark.parametrize("suffix", [" + []", " * 2"])
def test_cyclic_temporary_keeps_outer_shape_for_composition(monkeypatch, suffix):
    calls = []

    def cyclic():
        calls.append("call")
        value = []
        value.append(value)
        return value

    monkeypatch.setitem(TEMPLATE_FNS, "cyclic", cyclic)
    expression = "items=${cyclic()" + suffix + "}"
    ordinary = eval_binding(expression, lambda path: None)
    traced = eval_binding(
        expression, lambda path: ResolvedValue(None, False), trace=True
    )
    assert traced.value == ordinary
    routed = route_feedback(traced.route, "Shorten.", lambda source: None)
    assert routed.contributions == ()
    assert [stop.feedback for stop in routed.stopped] == ["Shorten."]
    assert routed.stopped[0].context[0].result == ordinary
    snapshot = routed.stopped[0].context[-1].snapshot
    assert snapshot.kind == "list" and len(snapshot.children) == 1
    recursive = snapshot.children[0]
    assert recursive.result_type == "builtins.list"
    assert recursive.value_json is None and recursive.snapshot is None
    assert json.dumps(asdict(traced.route))
    assert calls == ["call", "call"]


@pytest.mark.parametrize("suffix", [" + []", " * 2"])
@pytest.mark.parametrize("feedback", ["Note.", None])
def test_deep_temporary_composes_with_text_or_absent_feedback(
    monkeypatch, suffix, feedback
):
    calls = []

    def deep():
        calls.append("call")
        value = 1
        for _ in range(sys.getrecursionlimit() + 10):
            value = [value]
        return value

    monkeypatch.setitem(TEMPLATE_FNS, "deep", deep)
    expression = "${deep()" + suffix + "}"
    assert isinstance(eval_binding(expression, lambda path: None), list)
    traced = eval_binding(
        expression, lambda path: ResolvedValue(None, False), trace=True
    )
    assert isinstance(traced.value, list)
    routed = route_feedback(traced.route, feedback, lambda source: None)
    assert routed.contributions == ()
    assert [stop.feedback for stop in routed.stopped] == (
        [] if feedback is None else [feedback]
    )
    assert json.dumps(asdict(traced.route))
    assert calls == ["call", "call"]


def test_none_feedback_does_not_read_source_snapshots():
    reads = []
    route = ValueRoute("source", source=SourceRef("run", 1))

    def unavailable(source):
        reads.append(source)
        raise AssertionError("None feedback needs no value snapshot")

    routed = route_feedback(route, None, unavailable)
    assert routed.contributions == routed.stopped == ()
    assert reads == []


@pytest.mark.parametrize("container", ["direct", "list", "record"])
def test_capture_does_not_call_dictionary_subclass_hooks(monkeypatch, container):
    calls = []

    class Dictionary(dict):
        def items(self):
            raise RuntimeError("items hook")

    def dictionary():
        calls.append("dictionary")
        value = Dictionary(answer=1)
        if container == "list":
            return [value]
        if container == "record":
            return {"nested": value}
        return value

    def ignore(value):
        calls.append("ignore")
        return "ok"

    monkeypatch.setitem(TEMPLATE_FNS, "dictionary", dictionary)
    monkeypatch.setitem(TEMPLATE_FNS, "ignore", ignore)
    expression = "${ignore(dictionary())}"
    assert eval_binding(expression, lambda path: None) == "ok"
    traced = eval_binding(
        expression, lambda path: ResolvedValue(None, False), trace=True
    )
    assert traced.value == "ok"
    assert (
        route_feedback(traced.route, "Note.", lambda source: None).stopped[0].feedback
        == "Note."
    )
    assert calls == ["dictionary", "ignore", "dictionary", "ignore"]


def test_repetition_capture_does_not_call_integer_conversion(monkeypatch):
    calls = []

    class Integer(int):
        def __int__(self):
            raise RuntimeError("int hook")

    def integer():
        calls.append("integer")
        return Integer(2)

    def ignore(value):
        calls.append("ignore")
        assert value == [1, 1]
        return "ok"

    monkeypatch.setitem(TEMPLATE_FNS, "integer", integer)
    monkeypatch.setitem(TEMPLATE_FNS, "ignore", ignore)
    expression = "${ignore([1] * integer())}"
    assert eval_binding(expression, lambda path: None) == "ok"
    traced = eval_binding(
        expression, lambda path: ResolvedValue(None, False), trace=True
    )
    assert traced.value == "ok"
    assert (
        route_feedback(traced.route, "Note.", lambda source: None).stopped[0].feedback
        == "Note."
    )
    assert calls == ["integer", "ignore", "integer", "ignore"]


def test_item_tracing_preserves_get_without_extra_subclass_hooks():
    reads = []

    class Item(dict):
        def __contains__(self, key):
            raise RuntimeError("contains hook")

        def get(self, key):
            reads.append(key)
            return super().get(key)

    item = Item(answer="ok")
    expression = "${item.answer}"
    assert eval_binding(expression, lambda path: None, item=item) == "ok"
    traced = eval_binding(
        expression, lambda path: ResolvedValue(None, False), item=item, trace=True
    )
    assert traced.value == "ok"
    routed = route_feedback(traced.route, "Note.", lambda source: None)
    assert routed.contributions == routed.stopped == ()
    assert reads == ["answer", "answer"]


@pytest.mark.parametrize("container, depth", [("list", 33), ("dict", 40)])
def test_json_compatible_nested_results_keep_complete_feedback_shape(
    monkeypatch, container, depth
):
    value = "leaf"
    feedback = "Explain."
    for _ in range(depth):
        value = [value] if container == "list" else {"child": value}
        feedback = [feedback] if container == "list" else {"child": feedback}
    calls = []

    def nested():
        calls.append("call")
        return value

    monkeypatch.setitem(TEMPLATE_FNS, "nested", nested)
    traced = eval_binding(
        "${nested()}", lambda path: ResolvedValue(None, False), trace=True
    )
    assert traced.value is value
    validate_feedback(feedback, traced.value)
    stopped = route_feedback(traced.route, feedback, lambda source: None).stopped
    assert len(stopped) == 1
    assert stopped[0].feedback == feedback
    assert stopped[0].context[-1].result == value
    assert stopped[0].context[-1].result_type is None
    assert calls == ["call"]
