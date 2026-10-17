"""Shared helpers for `llm_config` layers that hold values (`${...}`)."""

import pytest

from yian.diagnostics import OriginRef
from yian.nodes.binding import BindingError
from yian.nodes.contracts import Binding, ContractOrigins, ValueContract, ValueTemplate
from yian.typesys import Type, ValueKind
from yian.typesys.pool import VariablePool

STR = Type.scalar(ValueKind.STRING)
OPT_STR = Type(kind=ValueKind.STRING, nullable=True)


def _typed(source, typ):
    """Return a typed binding of `source` declared as `typ`."""
    return Binding(source, contract=ValueContract(typ, ContractOrigins()))


def _record(nullable=False, **fields):
    """Return a record type whose fields are all required."""
    return Type(
        kind=ValueKind.OBJECT,
        fields=dict(fields),
        required=frozenset(fields),
        nullable=nullable,
    )


def _values():
    from yian.nodes.agent import llm_values

    return llm_values


def _keep(lower, higher):
    """Return what of `lower` still contributes under `higher`, typed by tags only."""
    values = _values()
    return values.contributing(lower, higher, values.contract_nullable)


# ---------- value_leaves ----------


def test_value_leaves_of_each_form():
    values = _values()
    tagged = _typed("${pick.output}", STR)
    block = ValueTemplate(
        {
            "model": "${input.m}",
            "provider": "openai",
            "anthropic": {
                "max_tokens": tagged,
                "thinking": {"type": "enabled", "budget_tokens": "${input.b}"},
            },
            "openai": "${input.block}",
        }
    )
    whole = Binding("${input.cfg}")

    assert values.value_leaves(block) == {
        ("model",): "${input.m}",
        ("anthropic", "max_tokens"): tagged,
        ("anthropic", "thinking", "budget_tokens"): "${input.b}",
        ("openai",): "${input.block}",
    }
    assert values.value_leaves(whole) == {(): whole}
    assert values.value_leaves({"model": "${not.a.value}"}) == {}
    assert values.value_leaves(None) == {}


def test_tagged_literals_and_escaped_text_are_not_value_leaves():
    values = _values()
    layer = ValueTemplate(
        {"model": _typed("gpt-4", STR), "provider": "a$${b}", "anthropic": "${input.a}"}
    )

    assert values.value_leaves(layer) == {("anthropic",): "${input.a}"}


# ---------- layer_refs / reads_only_inputs ----------


def test_layer_refs_and_input_only_detection():
    values = _values()
    inputs = ValueTemplate({"model": "${input.m}", "openai": {"reasoning": "x-${input.r}"}})
    mixed = ValueTemplate({"model": "${pick.output | input.m}"})
    system = Binding("${system.run_id}")

    assert values.layer_refs(inputs) == ("input.m", "input.r")
    assert values.layer_refs(mixed) == ("pick.output", "input.m")
    assert values.layer_refs({"model": "m"}) == ()
    assert values.reads_only_inputs(inputs)
    assert values.reads_only_inputs(Binding("${input}"))
    assert values.reads_only_inputs({"model": "m"})
    assert not values.reads_only_inputs(mixed)
    assert not values.reads_only_inputs(system)


# ---------- contributing (D4) ----------


def test_a_literal_shields_a_lower_value():
    lower = ValueTemplate({"model": "${pick.output}", "provider": "${input.p}"})

    kept = _keep(lower, {"model": "gpt-4"})

    assert _values().value_leaves(kept) == {("provider",): "${input.p}"}


def test_a_null_literal_does_not_shield():
    lower = ValueTemplate({"model": "${pick.output}"})

    assert _keep(lower, {"model": None}) is lower


def test_a_non_nullable_value_shields_and_a_nullable_one_does_not():
    lower = ValueTemplate({"model": "${pick.output}"})
    strict = ValueTemplate({"model": _typed("${ask.output}", STR)})
    loose = ValueTemplate({"model": _typed("${ask.output}", OPT_STR)})
    untyped = ValueTemplate({"model": "${ask.output}"})
    embedded = ValueTemplate({"model": "gpt-${ask.output}"})

    assert _values().value_leaves(_keep(lower, strict)) == {}
    assert _keep(lower, loose) is lower
    assert _keep(lower, untyped) is lower
    assert _values().value_leaves(_keep(lower, embedded)) == {}


def test_the_nullable_predicate_decides_untagged_values():
    values = _values()
    lower = ValueTemplate({"model": "${pick.output}"})
    higher = ValueTemplate({"model": "${ask.output}"})
    seen = []

    def never_null(leaf, inner):
        seen.append((leaf, inner))
        return False

    assert values.value_leaves(values.contributing(lower, higher, never_null)) == {}
    assert seen == [("${ask.output}", ())]


def test_shielding_follows_block_fields():
    values = _values()
    lower = ValueTemplate(
        {
            "anthropic": {
                "max_tokens": "${input.n}",
                "thinking": {"type": "enabled", "budget_tokens": "${input.b}"},
            }
        },
        (
            (("anthropic", "max_tokens"), OriginRef("site/max")),
            (("anthropic", "thinking", "budget_tokens"), OriginRef("site/budget")),
        ),
    )
    higher = {"anthropic": {"thinking": {"type": "disabled"}}}

    kept = _keep(lower, higher)

    assert values.value_leaves(kept) == {("anthropic", "max_tokens"): "${input.n}"}
    assert kept.value == {"anthropic": {"max_tokens": "${input.n}"}}
    assert kept.origins == ((("anthropic", "max_tokens"), OriginRef("site/max")),)


def test_a_whole_block_survives_a_partial_block():
    lower = ValueTemplate({"anthropic": "${input.block}"})
    higher = ValueTemplate({"anthropic": {"max_tokens": 10, "temperature": "${input.t}"}})

    assert _keep(lower, higher) is lower


def test_a_whole_record_survives_a_partial_layer():
    lower = Binding("${input.cfg}")

    assert _keep(lower, {"model": "m", "provider": "openai"}) is lower


def test_a_whole_block_shields_the_fields_its_type_declares():
    values = _values()
    lower = ValueTemplate({"anthropic": {"max_tokens": "${a.output}", "top_k": "${b.output}"}})
    block = _typed("${input.block}", _record(max_tokens=Type.scalar(ValueKind.INTEGER)))
    higher = ValueTemplate({"anthropic": block})

    kept = _keep(lower, higher)

    assert values.value_leaves(kept) == {("anthropic", "top_k"): "${b.output}"}


def test_a_whole_record_shields_by_its_declared_fields():
    values = _values()
    lower = ValueTemplate({"model": "${pick.output}", "provider": "${input.p}"})
    strict = Binding(
        "${input.cfg}", contract=ValueContract(_record(model=STR), ContractOrigins())
    )
    loose_field = Binding(
        "${input.cfg}", contract=ValueContract(_record(model=OPT_STR), ContractOrigins())
    )
    loose_record = Binding(
        "${input.cfg}",
        contract=ValueContract(_record(nullable=True, model=STR), ContractOrigins()),
    )
    optional_field = Binding(
        "${input.cfg}",
        contract=ValueContract(
            Type(kind=ValueKind.OBJECT, fields={"model": STR}, required=frozenset()),
            ContractOrigins(),
        ),
    )

    assert values.value_leaves(_keep(lower, strict)) == {("provider",): "${input.p}"}
    assert _keep(lower, loose_field) is lower
    assert _keep(lower, loose_record) is lower
    assert _keep(lower, optional_field) is lower
    assert _keep(lower, Binding("${input.cfg}")) is lower


def test_a_whole_record_shields_block_fields_through_its_type():
    values = _values()
    lower = ValueTemplate({"openai": {"reasoning": "${pick.output}"}})
    cfg = _record(openai=_record(reasoning=Type.scalar(ValueKind.OBJECT)))
    higher = _typed("${input.cfg}", cfg)

    assert values.value_leaves(_keep(lower, higher)) == {}


def test_type_nullable_walks_records_and_unions():
    values = _values()
    one = _record(kind=Type(kind=ValueKind.STRING, tags=frozenset({"a"})), model=STR)
    two = _record(kind=Type(kind=ValueKind.STRING, tags=frozenset({"b"})), model=OPT_STR)
    union = Type(kind=ValueKind.OBJECT, variants=(one, two))

    assert not values.type_nullable(one, ("model",))
    assert values.type_nullable(one, ("provider",))
    assert values.type_nullable(union, ("model",))
    assert not values.type_nullable(union, ("kind",))
    assert values.type_nullable(Type.scalar(ValueKind.OBJECT), ("model",))
    assert values.type_nullable(None, ())
    assert not values.type_nullable(Type.scalar(ValueKind.ANY), ())


# ---------- split_layers (D3) ----------


def test_split_layers_carries_a_mixed_flow_layer_whole():
    values = _values()
    node = ValueTemplate(
        {"model": "${pick.output}"}, ((("model",), OriginRef("node/model")),)
    )
    flow = ValueTemplate(
        {"provider": "anthropic", "temperature": 0.2, "anthropic": {"top_k": "${input.k}"}},
        ((("anthropic", "top_k"), OriginRef("flow/top_k")),),
    )

    carried = values.split_layers(node, flow, True, values.contract_nullable)

    assert isinstance(carried, Binding)
    template = carried.source
    assert isinstance(template, ValueTemplate)
    assert template.value == {"node": node.value, "flow": flow.value}
    assert set(template.origins) == {
        (("node", "model"), OriginRef("node/model")),
        (("flow", "anthropic", "top_k"), OriginRef("flow/top_k")),
    }


def test_split_layers_escapes_a_literal_layer():
    values = _values()
    node = ValueTemplate({"model": "${pick.output}"})

    carried = values.split_layers(node, {"provider": "a${b}$"}, True, values.contract_nullable)

    assert carried.source.value["flow"] == {"provider": "a$${b}$$"}
    assert values.resolve_layers(carried, _pool(pick="m"), **_WHERE) == (
        {"model": "m"},
        {"provider": "a${b}$"},
    )


def test_split_layers_drops_the_flow_layer_without_inherit():
    values = _values()
    node = ValueTemplate({"model": "${pick.output}"})
    flow = ValueTemplate({"provider": "${pick2.output}"})

    carried = values.split_layers(node, flow, False, values.contract_nullable)

    assert carried.source.value == {"node": node.value}
    assert values.split_layers({"model": "m"}, flow, False, values.contract_nullable) is None


def test_split_layers_drops_shielded_flow_values():
    values = _values()
    flow = ValueTemplate({"model": "${pick.output}", "provider": "openai"})

    carried = values.split_layers({"model": "m"}, flow, True, values.contract_nullable)

    assert carried is None
    kept = values.split_layers({}, flow, True, values.contract_nullable)
    assert kept.source.value == {"node": {}, "flow": flow.value}


def test_split_layers_leaves_out_a_flow_layer_the_node_wholly_overrides():
    values = _values()
    node = ValueTemplate({"model": _typed("${pick.output}", STR)})
    flow = ValueTemplate({"model": "${pick2.output}"})

    carried = values.split_layers(node, flow, True, values.contract_nullable)

    assert carried.source.value == {"node": node.value}
    assert values.split_layers(node, {}, True, values.contract_nullable).source.value == {
        "node": node.value
    }


def test_split_layers_carries_a_whole_record():
    values = _values()
    node = Binding("${input.cfg}", origin=OriginRef("node/cfg"))

    carried = values.split_layers(node, {"provider": "openai"}, True, values.contract_nullable)

    assert carried.source.value == {"node": node, "flow": {"provider": "openai"}}


# ---------- resolve_layers ----------


_WHERE = {
    "node_where": lambda field: "node 'chat' llm_config" + "".join(f".{p}" for p in field),
    "flow_where": lambda field: "flow llm_config" + "".join(f".{p}" for p in field),
}


def _pool(start=None, **outputs):
    pool = VariablePool()
    pool.set(pool.start_id, start or {})
    for node_id, value in outputs.items():
        pool.set(node_id, value)
    return pool


def test_resolve_layers_against_a_pool():
    values = _values()
    node = ValueTemplate({"model": "${pick.output}", "anthropic": {"top_k": "${input.k}"}})
    flow = ValueTemplate({"provider": "anthropic", "temperature": "${input.t}"})
    carried = values.split_layers(node, flow, True, values.contract_nullable)

    resolved = values.resolve_layers(carried, _pool({"k": 5, "t": None}, pick="claude"), **_WHERE)

    assert resolved == (
        {"model": "claude", "anthropic": {"top_k": 5}},
        {"provider": "anthropic", "temperature": None},
    )


def test_resolve_layers_raises_on_a_nested_tag_mismatch():
    values = _values()
    node = ValueTemplate({"model": _typed("${pick.output}", STR)})
    carried = values.split_layers(node, {}, True, values.contract_nullable)

    with pytest.raises(BindingError) as caught:
        values.resolve_layers(carried, _pool(pick=3), **_WHERE)

    assert caught.value.failure_code == "binding.type"
    assert str(caught.value).startswith("node 'chat' llm_config.model: "), str(caught.value)


def test_resolve_layers_names_a_flow_field_under_the_flow():
    values = _values()
    flow = ValueTemplate({"model": _typed("${pick.output}", STR)})
    carried = values.split_layers({}, flow, True, values.contract_nullable)

    with pytest.raises(BindingError) as caught:
        values.resolve_layers(carried, _pool(pick=3), **_WHERE)

    assert caught.value.failure_code == "binding.type"
    assert str(caught.value).startswith("flow llm_config.model: "), str(caught.value)


def test_a_null_whole_record_resolves_to_an_empty_layer():
    values = _values()
    node = Binding("${input.cfg}")
    carried = values.split_layers(node, {"model": "m"}, True, values.contract_nullable)

    assert values.resolve_layers(carried, _pool({"cfg": None}), **_WHERE) == ({}, {"model": "m"})
    assert values.resolve_layers(carried, _pool({"cfg": {"model": "x"}}), **_WHERE) == (
        {"model": "x"},
        {"model": "m"},
    )


def test_a_whole_record_that_is_not_a_mapping_fails():
    values = _values()
    carried = values.split_layers(Binding("${input.cfg}"), {}, True, values.contract_nullable)

    with pytest.raises(BindingError) as caught:
        values.resolve_layers(carried, _pool({"cfg": "gpt-4"}), **_WHERE)

    assert caught.value.failure_code == "contract.input"
    assert "gpt-4" not in str(caught.value)


def test_types_nullable_answers_from_the_stored_types():
    values = _values()
    tagged = Binding("${input.m}", contract=ValueContract(Type.scalar(ValueKind.STRING)))
    layer = ValueTemplate({"model": "${pick.output}", "provider": tagged})
    proven = values.types_nullable(layer, {("model",): Type.scalar(ValueKind.STRING)})
    assert proven(layer.value["model"], ()) is False
    # A leaf without a stored type answers from its tag.
    assert proven(tagged, ()) is False
    unproven = values.types_nullable(layer, {("model",): None})
    assert unproven(layer.value["model"], ()) is True


# ---------- refused_resolved_field ----------

_HOST_ONLY = "credentials and where they are sent stay with the host or yian auth"
_SUBJECT = "node 'c/a' llm_config"


def _refused(written, resolved):
    return _values().refused_resolved_field(written, resolved, _SUBJECT)


def test_a_call_time_whole_record_may_not_set_a_host_only_field():
    values = _values()
    carried = values.split_layers(Binding("${pick.output}"), {}, True, values.contract_nullable)
    record = {"provider": "azure", "model": "dep", "base_url": "https://elsewhere.example"}
    node_layer, _ = values.resolve_layers(carried, _pool(pick=record), **_WHERE)

    message = _refused(carried.source.value["node"], node_layer)

    assert message == f"{_SUBJECT}.base_url: {_HOST_ONLY}"


def test_a_call_time_flow_whole_record_may_not_set_a_host_only_field():
    values = _values()
    carried = values.split_layers(
        {"model": "m"}, Binding("${pick.output}"), True, values.contract_nullable
    )
    _, flow_layer = values.resolve_layers(carried, _pool(pick={"api_key": "sk-x"}), **_WHERE)

    message = values.refused_resolved_field(
        carried.source.value["flow"], flow_layer, "flow llm_config"
    )

    assert message == f"flow llm_config.api_key: {_HOST_ONLY}"


def test_a_host_only_field_value_is_refused_even_beside_a_literal_azure_provider():
    layer = ValueTemplate({"provider": "azure", "base_url": "${input.url}"})
    resolved = {"provider": "azure", "base_url": "https://elsewhere.example"}

    assert _refused(layer, resolved) == f"{_SUBJECT}.base_url: {_HOST_ONLY}"


def test_a_literal_base_url_passes_only_beside_a_literal_azure_provider():
    resolved = {"provider": "azure", "base_url": "https://res.openai.azure.com"}
    literal = ValueTemplate({"provider": "azure", "base_url": resolved["base_url"]})
    from_value = ValueTemplate({"provider": "${input.p}", "base_url": resolved["base_url"]})
    tagged = ValueTemplate({"provider": Binding("azure"), "base_url": resolved["base_url"]})

    assert _refused(literal, resolved) is None
    assert _refused(tagged, resolved) is None
    assert _refused(from_value, resolved).startswith(f"{_SUBJECT}: llm_config.base_url ")


def test_a_layer_resolved_from_values_may_not_set_a_recorded_connection():
    carried = Binding("${input.cfg}")
    resolved = {"model": "m", "_connection": {"method": "api_key"}}

    assert _refused(carried, resolved) == f"{_SUBJECT}._connection: internal to recorded runs"


def test_a_layer_without_refused_fields_passes():
    layer = ValueTemplate({"provider": "${input.p}", "model": "m"})

    assert _refused(layer, {"provider": "openai", "model": "m"}) is None
    assert _refused(Binding("${input.cfg}"), {}) is None
