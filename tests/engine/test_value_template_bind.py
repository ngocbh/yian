"""Value templates: an authored container whose nested strings are templates.

The carrier helpers walk and rebuild template trees (including nested typed
bindings), and the binder evaluates each nested string while a plain mapping or
list source stays a literal.
"""

import copy

import pytest

from yian.diagnostics import OriginRef
from yian.expr.lineage import ResolvedValue, SourceRef, ValueRoute, route_feedback
from yian.nodes.binding import BindingCollector, BindingError, bind_params
from yian.nodes.contracts import (
    Binding,
    ContractOrigins,
    Parameter,
    ValueContract,
    ValueTemplate,
    _rewrite_origin_carrier,
    binding_input_groups,
    iter_source_strings,
    map_source_strings,
)
from yian.runtime.contracts import _validate_route
from yian.typesys import Type, ValueKind
from yian.typesys.pool import VariablePool
from yian.typesys.types import read_typedefs, type_for


INTEGER = Type.scalar(ValueKind.INTEGER)


# ---------- carrier ----------


def test_value_template_requires_a_mapping_or_list():
    assert ValueTemplate({"a": "${n.output}"}).value == {"a": "${n.output}"}
    assert ValueTemplate(["${n.output}"]).origins == ()
    with pytest.raises(TypeError):
        ValueTemplate("${n.output}")
    with pytest.raises(TypeError):
        ValueTemplate(("${n.output}",))


def test_value_template_validates_its_origins():
    origin = OriginRef("site/path/a")
    assert ValueTemplate({"a": "x"}, ((("a",), origin),)).origins == ((("a",), origin),)
    with pytest.raises(TypeError):
        ValueTemplate({"a": "x"}, [(("a",), origin)])
    with pytest.raises(TypeError):
        ValueTemplate({"a": "x"}, ((("a",), "site"),))
    with pytest.raises(TypeError):
        ValueTemplate({"a": "x"}, ((["a"], origin),))


# ---------- iter_source_strings ----------


def test_iter_source_strings_yields_a_plain_string_at_the_root():
    assert list(iter_source_strings("${a.output}")) == [((), "${a.output}")]
    assert list(iter_source_strings(Binding("${a.output}"))) == [((), "${a.output}")]


def test_iter_source_strings_skips_literals_and_plain_containers():
    assert list(iter_source_strings(5)) == []
    assert list(iter_source_strings(None)) == []
    assert list(iter_source_strings({"a": "${x.output}"})) == []
    assert list(iter_source_strings(Binding(["${x.output}"]))) == []
    # Exactly one binding layer is unwrapped; a binding of a binding is a literal.
    assert list(iter_source_strings(Binding(Binding("${x.output}")))) == []


def test_iter_source_strings_walks_template_trees_and_nested_bindings():
    template = ValueTemplate(
        {
            "a": "${n.output}",
            "b": [1, "${m.output}", {"c": "text"}],
            "d": Binding("${d.output}", contract=ValueContract(INTEGER)),
            "e": Binding({"f": "${e.output}"}),
        }
    )

    expected = [
        (("a",), "${n.output}"),
        (("b", 1), "${m.output}"),
        (("b", 2, "c"), "text"),
        (("d",), "${d.output}"),
        (("e", "f"), "${e.output}"),
    ]
    assert list(iter_source_strings(template)) == expected
    assert list(iter_source_strings(Binding(template))) == expected


# ---------- map_source_strings ----------


def test_map_source_strings_maps_a_plain_string_and_keeps_its_binding():
    origin = OriginRef("b")
    contract = ValueContract(INTEGER)
    mapped = map_source_strings(Binding("${a}", origin, contract), str.upper)

    assert mapped == Binding("${A}", origin, contract)
    assert map_source_strings("x", str.upper) == "X"


def test_map_source_strings_leaves_literals_and_plain_containers_alone():
    plain = {"a": "${x.output}"}
    assert map_source_strings(plain, str.upper) is plain
    assert map_source_strings(7, str.upper) == 7
    binding = Binding(["${x.output}"])
    assert map_source_strings(binding, str.upper) is binding


def test_map_source_strings_rebuilds_template_carriers():
    leaf_origin = OriginRef("site/path/a")
    nested_origin = OriginRef("site/type/path/d")
    contract = ValueContract(INTEGER, ContractOrigins(default=nested_origin))
    template = ValueTemplate(
        {
            "a": "${n.output}",
            "b": [1, "${m.output}"],
            "d": Binding("${d.output}", nested_origin, contract),
        },
        ((("a",), leaf_origin),),
    )
    outer = Binding(template, OriginRef("site"), None)

    mapped = map_source_strings(outer, lambda text: text.replace("output", "out"))

    assert type(mapped) is Binding and mapped.origin == OriginRef("site")
    assert type(mapped.source) is ValueTemplate
    assert mapped.source.origins == template.origins
    assert mapped.source.value == {
        "a": "${n.out}",
        "b": [1, "${m.out}"],
        "d": Binding("${d.out}", nested_origin, contract),
    }
    # The input is not mutated.
    assert template.value["a"] == "${n.output}"


# ---------- binding_input_groups ----------


def test_binding_input_groups_for_a_string_is_the_parameter():
    assert binding_input_groups("qty", "${a.output}") == [("qty", "${a.output}")]
    assert binding_input_groups("qty", Binding("${a.output}")) == [
        ("qty", "${a.output}")
    ]
    assert binding_input_groups("qty", 5) == []
    assert binding_input_groups("qty", {"a": "${x.output}"}) == []


def test_binding_input_groups_name_each_nested_string_by_its_escaped_path():
    template = ValueTemplate(
        {
            "a": "${n.output}",
            "b": [1, "${m.output}"],
            "x/y": "${p.output}",
            "t~u": Binding("${q.output}", contract=ValueContract(INTEGER)),
        }
    )

    assert binding_input_groups("d", Binding(template)) == [
        ("d/a", "${n.output}"),
        ("d/b/1", "${m.output}"),
        ("d/x~1y", "${p.output}"),
        ("d/t~0u", "${q.output}"),
    ]


# ---------- origins ----------


def test_binding_origins_include_the_template_and_nested_bindings():
    outer = OriginRef("outer")
    leaf = OriginRef("leaf")
    nested = OriginRef("nested")
    nested_contract = OriginRef("nested/contract")
    binding = Binding(
        ValueTemplate(
            {
                "a": "${n.output}",
                "d": Binding(
                    "${d.output}",
                    nested,
                    ValueContract(INTEGER, ContractOrigins(default=nested_contract)),
                ),
            },
            ((("a",), leaf),),
        ),
        outer,
    )

    assert list(binding.iter_origins()) == [outer, leaf, nested, nested_contract]

    mapped = binding.map_origins(lambda origin: OriginRef("m:" + origin.key))
    assert list(mapped.iter_origins()) == [
        OriginRef("m:outer"),
        OriginRef("m:leaf"),
        OriginRef("m:nested"),
        OriginRef("m:nested/contract"),
    ]
    # The original carrier graph is untouched.
    assert list(binding.iter_origins()) == [outer, leaf, nested, nested_contract]


def test_rewriting_a_copied_carrier_reaches_template_and_nested_origins():
    binding = Binding(
        ValueTemplate(
            {"a": "${n.output}", "b": [Binding("${d.output}", OriginRef("nested"))]},
            ((("a",), OriginRef("leaf")),),
        ),
        OriginRef("outer"),
    )
    private = copy.deepcopy(binding)

    _rewrite_origin_carrier(private, lambda origin: OriginRef("m:" + origin.key), set())

    assert list(private.iter_origins()) == [
        OriginRef("m:outer"),
        OriginRef("m:leaf"),
        OriginRef("m:nested"),
    ]
    assert list(binding.iter_origins()) == [
        OriginRef("outer"),
        OriginRef("leaf"),
        OriginRef("nested"),
    ]


def _two_level_binding():
    """Return a template binding whose nested binding holds another one."""
    inner = Binding("${c.output}", OriginRef("inner"))
    middle = Binding({"item": inner, "n": 1}, OriginRef("middle"))
    return Binding(
        ValueTemplate({"order": middle}, ((("order", "item"), OriginRef("leaf")),)),
        OriginRef("outer"),
    )


def test_every_origin_walker_reaches_a_binding_nested_two_levels_deep():
    binding = _two_level_binding()
    keys = ["outer", "leaf", "middle", "inner"]
    rename = lambda origin: OriginRef("m:" + origin.key)  # noqa: E731

    assert [origin.key for origin in binding.iter_origins()] == keys
    mapped = binding.map_origins(rename)
    assert [origin.key for origin in mapped.iter_origins()] == ["m:" + k for k in keys]
    private = copy.deepcopy(binding)
    _rewrite_origin_carrier(private, rename, set())
    assert [origin.key for origin in private.iter_origins()] == ["m:" + k for k in keys]
    assert private == mapped


def test_a_plain_binding_with_a_mapping_source_keeps_its_contents_opaque():
    binding = Binding({"item": Binding("${c.output}", OriginRef("inner"))}, OriginRef("o"))

    assert list(binding.iter_origins()) == [OriginRef("o")]


def test_value_template_keys_must_be_strings_throughout():
    with pytest.raises(TypeError, match="keys must be strings"):
        ValueTemplate({1: "${n.output}"})
    with pytest.raises(TypeError, match="keys must be strings"):
        ValueTemplate([{"a": {2: "x"}}])
    with pytest.raises(TypeError, match="keys must be strings"):
        ValueTemplate({"a": Binding({3: "${n.output}"})})
    with pytest.raises(TypeError, match="keys must be strings"):
        ValueTemplate({"a": Binding([Binding({4: "x"})])})


# ---------- binder ----------

BUY = type_for(
    "Buy",
    read_typedefs({"Buy": {"action": "Literal(buy)", "conviction": "float"}}),
)


def _pool(**outputs):
    """Return a pool holding each keyword as that node's output."""
    pool = VariablePool()
    for node_id, value in outputs.items():
        pool.set(node_id, value)
    return pool


def test_template_strings_evaluate_with_types_and_unescape():
    template = ValueTemplate({"a": "${n.output}", "b": [1, "${m.output}"], "c": "$$x"})

    record = bind_params(
        [Parameter("d")],
        {"d": Binding(template)},
        _pool(n=3.5, m=["x", 2]),
    )

    assert record == {"d": {"a": 3.5, "b": [1, ["x", 2]], "c": "$x"}}


def test_embedded_template_strings_are_stringified():
    record = bind_params(
        [Parameter("d")],
        {"d": ValueTemplate(["n=${n.output}", "plain"])},
        _pool(n=4),
    )

    assert record == {"d": ["n=4", "plain"]}


def test_a_plain_mapping_source_stays_a_literal():
    source = {"a": "${n.output}", "b": ["${m.output}"], "c": "$$x"}

    record = bind_params([Parameter("d")], {"d": Binding(source)}, _pool(n=1, m=2))

    assert record == {"d": {"a": "${n.output}", "b": ["${m.output}"], "c": "$$x"}}
    assert bind_params([Parameter("d")], {"d": ["${n.output}"]}, _pool(n=1)) == {
        "d": ["${n.output}"]
    }


def test_nested_typed_binding_passes_a_valid_value_unchanged():
    template = ValueTemplate(
        {"order": Binding("${d.output}", contract=ValueContract(BUY)), "n": 1}
    )
    value = {"action": "buy", "conviction": 1}

    record = bind_params([Parameter("p")], {"p": template}, _pool(d=value))

    assert record == {"p": {"order": {"action": "buy", "conviction": 1}, "n": 1}}
    assert type(record["p"]["order"]["conviction"]) is int


def test_nested_typed_binding_rejects_a_missing_field_at_its_origin():
    nested_origin = OriginRef("site/type/path/order")
    template = ValueTemplate(
        {"order": Binding("${d.output}", nested_origin, ValueContract(BUY))}
    )

    with pytest.raises(BindingError, match=r"'action'") as caught:
        bind_params(
            [Parameter("p")],
            {"p": Binding(template, OriginRef("site"))},
            _pool(d={"conviction": 0.5}),
        )

    assert caught.value.failure_code == "binding.type"
    assert caught.value.origin == nested_origin
    assert str(caught.value).startswith("binding 'p.order':")


def test_nested_typed_binding_prefers_the_nearest_contract_origin():
    field_origin = OriginRef("site/type/path/b/0/conviction")
    contract = ValueContract(
        BUY, ContractOrigins(entries=((("conviction",), field_origin),))
    )
    template = ValueTemplate({"b": [Binding("${d.output}", OriginRef("n"), contract)]})

    with pytest.raises(BindingError) as caught:
        bind_params(
            [Parameter("p")],
            {"p": template},
            _pool(d={"action": "buy", "conviction": "high"}),
        )

    assert caught.value.failure_code == "binding.type"
    assert caught.value.origin == field_origin
    assert str(caught.value).startswith("binding 'p.b[0]':")


def test_nested_typed_binding_over_a_subtree_checks_the_evaluated_record():
    template = ValueTemplate(
        {
            "order": Binding(
                {"action": "buy", "conviction": "${c.output}"},
                contract=ValueContract(BUY),
            )
        }
    )

    assert bind_params([Parameter("p")], {"p": template}, _pool(c=0.7)) == {
        "p": {"order": {"action": "buy", "conviction": 0.7}}
    }
    with pytest.raises(BindingError) as caught:
        bind_params([Parameter("p")], {"p": template}, _pool(c="high"))
    assert caught.value.failure_code == "binding.type"


def test_template_evaluation_errors_point_at_the_leaf_origin():
    leaf_origin = OriginRef("site/path/a")
    template = ValueTemplate({"a": "${n.output:?}"}, ((("a",), leaf_origin),))

    with pytest.raises(BindingError) as caught:
        bind_params(
            [Parameter("p")], {"p": Binding(template, OriginRef("site"))}, _pool()
        )

    assert caught.value.failure_code == "binding.evaluate"
    assert caught.value.origin == leaf_origin
    assert str(caught.value).startswith("binding 'p.a':")


def test_a_nested_binding_origin_locates_errors_inside_it():
    nested_origin = OriginRef("site/type/path/b/1")
    template = ValueTemplate(
        {"b": [1, Binding({"c": "${n.output:?}"}, nested_origin)]}
    )

    with pytest.raises(BindingError) as caught:
        bind_params(
            [Parameter("p")], {"p": Binding(template, OriginRef("site"))}, _pool()
        )

    assert caught.value.failure_code == "binding.evaluate"
    assert caught.value.origin == nested_origin
    assert str(caught.value).startswith("binding 'p.b[1].c':")


def test_map_item_scope_reaches_template_strings():
    template = ValueTemplate({"i": "${item}", "n": 1})

    assert bind_params([Parameter("p")], {"p": template}, _pool(), item=5) == {
        "p": {"i": 5, "n": 1}
    }


def test_template_routes_record_each_part_and_route_feedback():
    template = ValueTemplate(
        {"a": "${n.output}", "b": [1, "${m.output}"], "c": "lit"}
    )
    roots = {"n": {"output": 3}, "m": {"output": "x"}}

    def resolve(path):
        node, _, field = path.partition(".")
        seq = 1 if node == "n" else 2
        return ResolvedValue(roots[node][field], True, SourceRef("run", seq, (field,)))

    collector = BindingCollector()
    delivered = bind_params(
        [Parameter("p")],
        {"p": Binding(template)},
        _pool(n=3, m="x"),
        collector=collector,
        source_resolver=resolve,
    )
    captured = collector.freeze()

    assert delivered == captured.bound_inputs == {
        "p": {"a": 3, "b": [1, "x"], "c": "lit"}
    }
    assert captured.evaluated_keys == frozenset({"p"})
    _validate_route(captured.route)
    route = captured.route.children[0]
    assert route.kind == "record" and route.names == ("a", "b", "c")
    assert route.children[1].kind == "list"
    assert route.children[1].children[0] == ValueRoute.literal(1)

    routed = route_feedback(
        captured.route,
        {"p": {"a": "Too small.", "b": [None, "Wrong."]}},
        lambda source: roots["n" if source.execution_seq == 1 else "m"],
    )
    assert [(c.source, c.feedback) for c in routed.contributions] == [
        (SourceRef("run", 1), {"output": "Too small."}),
        (SourceRef("run", 2), {"output": "Wrong."}),
    ]
