"""Feedback follows the values and selected operations actually evaluated."""

from dataclasses import FrozenInstanceError, asdict
import json
import sys

import pytest

from yian.expr.builtins import TEMPLATE_FNS
from yian.expr.expressions import ExpressionError, ResolveMode, eval_expr
from yian.expr.grammar import parse_expr
from yian.expr.lineage import (
    ResolvedValue,
    SourceRef,
    ValueRoute,
    route_feedback,
    validate_feedback,
)
from yian.expr.template import eval_binding, eval_template, scan_template


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


@pytest.mark.parametrize("temporary", [float("inf"), float("nan")])
def test_nonfinite_temporary_keeps_a_valid_final_interpolation(temporary, monkeypatch):
    calls = []

    def score():
        calls.append(None)
        return temporary

    monkeypatch.setitem(TEMPLATE_FNS, "score", score)
    traced = eval_binding("score: ${score()}", lambda _: None, trace=True)
    assert traced.value == f"score: {temporary}"
    opaque = traced.route.children[0]
    assert opaque.kind == "opaque"
    assert opaque.value_json is None
    assert opaque.result_type == "builtins.float"
    assert calls == [None]
    stopped = route_feedback(traced.route, "Check the score.", lambda _: None).stopped
    assert stopped[0].context[-1].result is None


@pytest.mark.parametrize("temporary,result_type,snapshot_kind", [
    pytest.param("\ud800", "builtins.str", None, id="string"),
    pytest.param(["kept", "\ud800"], "builtins.list", "list", id="list"),
    pytest.param({"kept": 1, "lost": "\ud800"}, "builtins.dict", "record", id="record-value"),
    pytest.param({"\ud800": 1}, "builtins.dict", None, id="record-key"),
    pytest.param({"kept": 1, "lost": {"\ud800": 1}}, "builtins.dict", "record", id="nested-key"),
])
def test_noncanonical_unicode_temporary_preserves_comparison_and_partial_snapshot(
    temporary, result_type, snapshot_kind, monkeypatch,
):
    calls = []

    def bad_text():
        calls.append(None)
        return temporary

    monkeypatch.setitem(TEMPLATE_FNS, "bad_text", bad_text)
    traced = eval_binding("${bad_text() == bad_text()}", lambda _: None, trace=True)
    assert traced.value is True
    assert calls == [None, None]
    json.dumps(asdict(traced.route), ensure_ascii=False, allow_nan=False).encode("utf-8")
    for opaque in traced.route.children:
        assert opaque.kind == "opaque" and opaque.value_json is None
        assert opaque.result_type == result_type
        if snapshot_kind is None:
            assert opaque.snapshot is None
        else:
            assert opaque.snapshot.kind == snapshot_kind
            assert json.loads(opaque.snapshot.children[0].value_json) == (
                "kept" if snapshot_kind == "list" else 1
            )
            unavailable = opaque.snapshot.children[1]
            assert unavailable.value_json is None
            assert unavailable.result_type in ("builtins.str", "builtins.dict")
    routed = route_feedback(traced.route, "Explain the comparison.", lambda _: None)
    assert routed.contributions == ()
    assert len(routed.stopped) == 2
    assert all(stop.context[0].result is True for stop in routed.stopped)
    assert all(stop.context[-1].result is None for stop in routed.stopped)
    assert calls == [None, None]


def test_reserved_map_over_uses_the_same_capture_path():
    from yian.compile.model import Flow
    from yian.nodes.map import MapNode
    from yian.runtime.eval_node import eval_node
    from yian.typesys.pool import VariablePool

    sources = Sources(items=[])
    node = MapNode("map", flow_id="child", child=Flow(nodes=[], edges=[]))
    evaluated = eval_node(
        node, None, VariablePool(), wiring_override={"over": "${items}"},
        capture_bindings=True, source_resolver=sources.resolve,
    )
    assert evaluated.evidence is None
    assert evaluated.bindings.bound_inputs == {"over": []}
    assert evaluated.bindings.evaluated_keys == frozenset({"over"})
    assert sources.reads == ["items"]
    assert messages(route_feedback(
        evaluated.bindings.route, {"over": "Add an item."}, sources.lookup,
    )) == [(sources.identities["items"], "Add an item.")]


def test_equal_map_items_get_distinct_recorded_indices():
    from yian.compile.model import Flow
    from yian.nodes.contracts import Parameter
    from yian.nodes.map import MapNode
    from yian.runtime.eval_node import eval_node
    from yian.typesys.pool import VariablePool

    sources = Sources(left=["same", None], right=["same", None])
    node = MapNode("map", flow_id="child", child=Flow(nodes=[], edges=[]))
    node.params = [Parameter("x")]
    evaluated = eval_node(
        node, None, VariablePool(),
        wiring_override={"over": "${left + right * 2}", "x": "${item}"},
        capture_bindings=True, source_resolver=sources.resolve,
    )
    assert evaluated.outcome.seed == [{"x": "same"}, {"x": None}] * 3
    assert [index for index, _ in evaluated.bindings.items] == list(range(6))
    assert sources.reads == ["left", "right"]
    for index, item in evaluated.bindings.items:
        assert item.bound_inputs == evaluated.outcome.seed[index]
        assert item.evaluated_keys == frozenset({"x"})
        expected_source = sources.identities["left" if index < 2 else "right"]
        expected_feedback = [None, "Fix."] if index % 2 else ["Fix.", None]
        routed = route_feedback(
            item.route, {"x": "Fix."}, sources.lookup,
        )
        assert messages(routed) == [(expected_source, expected_feedback)]
        assert any(
            context.operand == (index,)
            for context in routed.contributions[0].context
        )


def test_wait_capture_retains_the_delivered_iso_value_after_normalization():
    from yian.nodes.wait import WaitNode
    from yian.runtime.eval_node import eval_node
    from yian.typesys.pool import VariablePool

    class Deadline:
        calls = 0

        def isoformat(self):
            self.calls += 1
            return "2000-01-01T00:00:00+00:00"

    deadline = Deadline()
    sources = Sources(deadline=deadline)
    evaluated = eval_node(
        WaitNode("wait"), None, VariablePool(),
        wiring_override={"until": "${deadline}"},
        capture_bindings=True, source_resolver=sources.resolve,
    )
    assert evaluated.outcome.value is None
    assert evaluated.bindings.bound_inputs == {"until": "2000-01-01T00:00:00+00:00"}
    assert deadline.calls == 1
    assert sources.reads == ["deadline"]
    assert messages(route_feedback(
        evaluated.bindings.route, {"until": "Too early."}, sources.lookup,
    )) == [(sources.identities["deadline"], "Too early.")]


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


def test_sourced_item_tracing_reads_get_once_per_step():
    reads = []

    class Item(dict):
        def __contains__(self, key):
            raise RuntimeError("contains hook")

        def get(self, key):
            reads.append(key)
            return super().get(key)

    item = ResolvedValue(Item(answer="ok"), True, SourceRef("run", 7, ("items", 0)))
    traced = eval_binding(
        "${item.answer}", lambda path: ResolvedValue(None, False), item=item, trace=True
    )
    assert traced.value == "ok"
    assert traced.route.source == SourceRef("run", 7, ("items", 0, "answer"))
    assert reads == ["answer"]


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


@pytest.mark.parametrize("form", [
    "literal", "nested_literal", "opaque", "selected", "list", "record",
    "concat", "repeat", "project_record", "project_list",
])
def test_accepted_result_shape_validates_unavailable_temporary_routes(form, monkeypatch):
    temporary = object()
    calls = []
    raw = {"file": temporary, "other": 1} if form == "project_record" else (
        [temporary, 1] if form == "project_list" else temporary
    )
    monkeypatch.setitem(TEMPLATE_FNS, "accepted_temporary", lambda: calls.append(None) or raw)
    expression = "${accepted_temporary() | null}" if form == "selected" else "${accepted_temporary()}"
    traced = eval_binding(expression, lambda _: None, trace=True)
    route = traced.route
    value = {"uri": "file:///tmp/example.txt"}
    feedback = {"uri": "Check the URI."}
    expected_stop = feedback
    if form == "literal":
        route = ValueRoute.literal(temporary)
    elif form == "nested_literal":
        route = ValueRoute.literal({"files": [temporary]})
        value, feedback = {"files": [value]}, {"files": [feedback]}
    elif form == "list":
        route = ValueRoute("list", children=(route,))
        value, feedback = [value], [feedback]
    elif form == "record":
        route = ValueRoute.record((("file", route),))
        value, feedback = {"file": value}, {"file": feedback}
    elif form == "concat":
        route = ValueRoute("concat", children=(ValueRoute("list", children=(route,)), ValueRoute.literal([])), sizes=(1, 0))
        value, feedback = [value], [feedback]
    elif form == "repeat":
        route = ValueRoute("repeat", children=(ValueRoute("list", children=(route,)), ValueRoute.literal(2)), sizes=(1,), count=2)
        value, feedback = [value, value], [feedback, None]
    elif form == "project_record":
        route = route.project(("file",))
        expected_stop = {"file": feedback}
    elif form == "project_list":
        route = route.project((0,))
        expected_stop = [feedback, None]
    routed = route_feedback(route, feedback, lambda _: None, recorded_value=value)
    assert routed.contributions == ()
    assert [item.feedback for item in routed.stopped] == (
        [] if form in ("literal", "nested_literal") else [expected_stop]
    )
    if routed.stopped:
        opaque_context = routed.stopped[0].context[-1]
        original = traced.route.children[0] if form == "selected" else traced.route
        assert opaque_context.result is None
        assert opaque_context.result_type == original.result_type
        assert opaque_context.snapshot == original.snapshot
    assert calls == [None]
    with pytest.raises(ValueError):
        route_feedback(route, feedback, lambda _: None)


@pytest.mark.parametrize("container", ["record", "list"])
@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("boundary", ["opaque", "literal"])
def test_accepted_projected_value_stops_at_a_bounded_ancestor(container, wrapped, boundary, monkeypatch):
    value = object()
    step = "nested" if container == "record" else 0
    path = (step,) * 40
    for _ in path:
        value = {"nested": value} if container == "record" else [value, "sibling"]
    calls = []
    monkeypatch.setitem(TEMPLATE_FNS, "bounded_temporary", lambda: calls.append(None) or value)
    traced = eval_binding("${bounded_temporary()}", lambda _: None, trace=True)
    boundary_route = traced.route if boundary == "opaque" else ValueRoute.literal(value)
    parent = ValueRoute.record((("outer", boundary_route),)) if wrapped else boundary_route
    selected_path = ("outer", *path) if wrapped else path
    projected = parent.project(selected_path)
    feedback = {"uri": "Review."}
    routed = route_feedback(projected, feedback, lambda _: None, recorded_value={"uri": "file:///tmp/x"})
    assert routed.contributions == ()
    if boundary == "opaque":
        assert len(routed.stopped) == 1
        stop = routed.stopped[0]
        assert stop.feedback == feedback
        assert any(detail.role == "projection" and detail.operand == selected_path for detail in stop.context)
        assert stop.context[-1].snapshot == traced.route.snapshot
    else:
        assert routed.stopped == ()
    assert calls == [None]


@pytest.mark.parametrize("operation", ["concat", "repeat"])
def test_projected_partial_container_uses_recorded_operation_positions(operation, monkeypatch):
    calls = []
    raw = [object(), object()]
    monkeypatch.setitem(TEMPLATE_FNS, "operation_temporary", lambda: calls.append(None) or raw)
    expression = "${operation_temporary() + []}" if operation == "concat" else "${operation_temporary() * 2}"
    traced = eval_binding(expression, lambda _: None, trace=True)
    index = 1 if operation == "concat" else 3
    feedback = {"uri": "Check."}
    routed = route_feedback(traced.route.project((index,)), feedback, lambda _: None,
                            recorded_value={"uri": "file:///tmp/x"})
    assert routed.contributions == ()
    assert [stop.feedback for stop in routed.stopped] == [[None, feedback]]
    context = routed.stopped[0].context
    assert any(detail.role == "projection" and detail.operand == (index,) for detail in context)
    assert any((index,) in detail.positions for detail in context)
    assert calls == [None]
    with pytest.raises(IndexError):
        route_feedback(traced.route.project((len(traced.value),)), feedback, lambda _: None,
                       recorded_value={"uri": "file:///tmp/x"})


def test_accepted_shape_keeps_full_validation_and_exact_source_failures():
    source = SourceRef("run", 1)
    route = ValueRoute("list", children=(ValueRoute("source", source=source), ValueRoute.literal(object())))
    valid = [1, {"uri": "file:///tmp/x"}]
    reads = []

    def lookup(identity):
        reads.append(identity)
        return 1

    for invalid in ([None], [None, {"unknown": None}], [None, []]):
        with pytest.raises(ValueError):
            route_feedback(route, invalid, lookup, recorded_value=valid)
    assert reads == []
    routed = route_feedback(route, ["Source.", {"uri": "Literal."}], lookup, recorded_value=valid)
    assert messages(routed) == [(source, "Source.")]
    failure = RuntimeError("accepted source lookup broke")

    def broken_lookup(_):
        raise failure

    with pytest.raises(RuntimeError) as caught:
        route_feedback(route, ["Source.", None], broken_lookup, recorded_value=valid)
    assert caught.value is failure
    with pytest.raises(KeyError):
        route_feedback(ValueRoute("source", source=SourceRef("run", 1, ("absent",))), "Review.", lambda _: {}, recorded_value="present")
    with pytest.raises(ValueError, match="unknown route kind"):
        route_feedback(ValueRoute("malformed"), "Review.", lookup, recorded_value="present")


@pytest.mark.parametrize("operation", ["direct", "concat", "repeat"])
@pytest.mark.parametrize("structured", [False, True])
def test_bounded_projected_lists_compose_before_opaque_stopping(operation, structured, monkeypatch):
    raw = {"files": [object()]}
    path = ("nested",) * 40 + ("files",)
    for _ in range(40):
        raw = {"nested": raw}
    calls = []
    monkeypatch.setitem(TEMPLATE_FNS, "bounded_files", lambda: calls.append(None) or raw)
    opaque = eval_binding("${bounded_files()}", lambda _: None, trace=True).route
    selected = opaque.project(path)
    route = selected
    value = [{"uri": "file:///tmp/x"}]
    feedback = [{"uri": "Review."}] if structured else "Review."
    if operation == "concat":
        route = ValueRoute("concat", children=(selected, ValueRoute.literal([])), sizes=(1, 0))
    elif operation == "repeat":
        route = ValueRoute("repeat", children=(selected, ValueRoute.literal(2)), sizes=(1,), count=2)
        value *= 2
        if structured:
            feedback.append(None)
    routed = route_feedback(route, feedback, lambda _: None, recorded_value=value)
    assert routed.contributions == ()
    assert [stop.feedback for stop in routed.stopped] == ([feedback[:1]] if structured else [feedback])
    assert any(detail.role == "projection" and detail.operand == path
               for detail in routed.stopped[0].context)
    assert routed.stopped[0].context[-1].snapshot == opaque.snapshot
    assert calls == [None]
    with pytest.raises(ValueError):
        route_feedback(route, [None] * (len(value) + 1), lambda _: None, recorded_value=value)
    with pytest.raises(ValueError):
        route_feedback(route, [{"unknown": None}] * len(value), lambda _: None, recorded_value=value)


@pytest.mark.parametrize("fault_type", [RuntimeError, TypeError, ValueError])
def test_unavailable_concat_child_does_not_hide_real_source_faults(fault_type):
    unavailable = ValueRoute.literal(object()).project(("files",))
    source = ValueRoute("source", source=SourceRef("run", 1))
    route = ValueRoute("concat", children=(unavailable, source), sizes=(1, 1))
    failure = fault_type("source lookup failed")

    def broken_lookup(_):
        raise failure

    with pytest.raises(fault_type) as caught:
        route_feedback(route, "Review.", broken_lookup, recorded_value=[1, 2])
    assert caught.value is failure
    with pytest.raises(TypeError):
        route_feedback(route, "Review.", lambda _: 1, recorded_value=[1, 2])


@pytest.mark.parametrize("operation", ["concat", "repeat"])
def test_projection_over_unavailable_composition_uses_recorded_bounds(operation):
    from dataclasses import replace

    opaque = replace(ValueRoute.literal(object()), kind="opaque", expression="opaque()")
    selected = opaque.project(("files",))
    route = (
        ValueRoute("concat", children=(selected, ValueRoute.literal([])), sizes=(1, 0))
        if operation == "concat" else
        ValueRoute("repeat", children=(selected, ValueRoute.literal(2)), sizes=(1,), count=2)
    )
    value = {"uri": "file:///tmp/x"}
    feedback = {"uri": "Review."}
    routed = route_feedback(route.project((0,)), feedback, lambda _: None, recorded_value=value)
    assert routed.contributions == ()
    assert [stop.feedback for stop in routed.stopped] == [feedback]
    length = 1 if operation == "concat" else 2
    with pytest.raises(IndexError):
        route_feedback(route.project((length,)), feedback, lambda _: None, recorded_value=value)


@pytest.mark.parametrize("count", [0, -1])
def test_empty_repetition_does_not_infer_its_original_content_from_the_result(count):
    sources = Sources(original="value")
    traced = sources.evaluate(f"[original] * {count}")
    assert traced.value == []
    routed = route_feedback(traced.route, "Review.", sources.lookup, recorded_value=[])
    assert messages(routed) == [(sources.identities["original"], "Review.")]
