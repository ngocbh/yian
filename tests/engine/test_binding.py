"""The bind grammar: resolve each declared param's source into a typed record.

The production `InputBinding`/`bind_inputs` are gone (the node no longer holds sources — the
flow owns them in `CompiledFlow.wiring`). These tests exercise the SAME `_resolve_source` grammar
(coalesce / `:-` default / `:?` required / escapes / type-check / deep-copy / `${item}` scope)
through `bind_params`, the engine's live read boundary. The local `InputBinding`/`bind_inputs`
adapters below phrase each case as a (`Parameter` + wiring) split, so the comprehensive grammar
coverage carries over against the live binder without reviving a legacy declaration type.
"""

from collections import namedtuple

import pytest

from agent_composer.compile.model import START_ID
from agent_composer.nodes.binding import BindingError, bind_params
from agent_composer.nodes.contracts import Parameter, ValueContract
from agent_composer.typesys import Type, ValueKind
from agent_composer.typesys.pool import VariablePool

# Test-local sink: one canonical parameter plus its flow-owned source.
_Sink = namedtuple(
    "_Sink",
    "name contract source required has_default default coerce_strings",
    defaults=(None, None, False, False, None, False),
)


def InputBinding(
    name,
    contract=None,
    source=None,
    required=False,
    has_default=False,
    default=None,
    coerce_strings=False,
):
    return _Sink(
        name,
        contract,
        source,
        required,
        has_default,
        default,
        coerce_strings,
    )


STRING = ValueContract(Type.scalar(ValueKind.STRING))
INTEGER = ValueContract(Type.scalar(ValueKind.INTEGER))
NUMBER = ValueContract(Type.scalar(ValueKind.NUMBER))
BOOLEAN = ValueContract(Type.scalar(ValueKind.BOOLEAN))
OBJECT = ValueContract(Type.scalar(ValueKind.OBJECT))
NULLABLE_INTEGER = ValueContract(Type(kind=ValueKind.INTEGER, nullable=True))


def bind_inputs(bindings, pool, *, item=None):
    """Drive the live `bind_params` from test-local parameter/source pairs.

    The canonical node-side parameters and flow-owned wiring remain separate, so
    the grammar coverage below tests the real binder contract.

    A `source=None` sink models an OMITTED input — NO wiring edge — so the presence-based
    param `default`/`required` fires. An edge that merely RESOLVES to null (e.g. `${system.w}`
    with the pool empty) is BOUND-NULL, which shadows a param default (`f(x=None)` semantics).
    Resolve-fallback / required-in-source are the SEPARATE `:-` / `:?` grammar (tested below)."""
    params = [
        Parameter(
            name=b.name,
            contract=b.contract,
            required=b.required,
            has_default=b.has_default,
            default=b.default,
            coerce_strings=b.coerce_strings,
        )
        for b in bindings
    ]
    wiring = {b.name: b.source for b in bindings if b.source is not None}
    return bind_params(params, wiring, pool, item=item)


def _pool(**system):
    p = VariablePool()
    for k, v in system.items():
        p.add_system(k, v)
    return p


def test_bind_ref_and_literal():
    pool = _pool(topics="ACME")
    pool.set("up", {"pe": 21.0})
    rec = bind_inputs(
        [
            InputBinding(name="topic", contract=STRING, source="${system.topics}"),
            InputBinding(name="pe", contract=NUMBER, source="${up.output.pe}"),
            InputBinding(name="k", contract=NUMBER, source=90),  # literal
        ],
        pool,
    )
    assert rec == {"topic": "ACME", "pe": 21.0, "k": 90.0}


def test_bind_default_applied_when_omitted():
    # The param `default` is the OMIT-default (presence-based): it fills only when the input has
    # NO wiring edge at all. (Resolve-fallback for a present-but-null source is the `:-` grammar.)
    rec = bind_inputs(
        [InputBinding(name="w", contract=INTEGER, has_default=True, default=30)],
        _pool(),
    )  # no source -> omitted
    assert rec == {"w": 30}


def test_bind_compatible_default_is_validated_without_replacement():
    """A typed default retains its authored Python payload after validation."""
    rec = bind_inputs(
        [InputBinding(name="w", contract=NUMBER, has_default=True, default=30)],
        _pool(),
    )

    assert type(rec["w"]) is int


def test_bind_present_null_shadows_param_default():
    # An edge that is PRESENT but RESOLVES to null is bound-null: it SHADOWS the param default
    # (the `f(x=None)` contract). Pins the omit-vs-bound-null distinction at the live binder.
    rec = bind_inputs(
        [
            InputBinding(
                name="w",
                contract=INTEGER,
                source="${system.w}",
                has_default=True,
                default=30,
            )
        ],
        _pool(),
    )
    assert rec == {"w": None}


def test_bind_explicit_null_default_fills_when_omitted():
    # `has_default` owns presence, so an authored nullable `default: null` is a real
    # default and remains distinct from a parameter with no default.
    rec = bind_inputs(
        [
            InputBinding(
                name="w",
                contract=NULLABLE_INTEGER,
                has_default=True,
                default=None,
            )
        ],
        _pool(),
    )
    assert rec == {"w": None}


def test_bind_omitted_no_default_is_absent():
    # A truly OMITTED optional input with NO default leaves its key ABSENT from the record (not
    # present-as-None) so the wrapper's `apply_default` can fill it (or the strict render raises
    # on unbound). This is the omit vs bound-null split — contrast `test_bind_optional_missing_is_none`,
    # where a PRESENT edge resolving to null lands as `{"t": None}`.
    assert bind_inputs([InputBinding(name="x", contract=INTEGER)], _pool()) == {}


def test_bind_required_omitted_raises():
    # Param `required` is presence-based too: an OMITTED required input (no edge) fails loud.
    with pytest.raises(BindingError):
        bind_inputs([InputBinding(name="t", contract=STRING, required=True)], _pool())


def test_bind_params_rejects_a_raw_declaration_bypass():
    """The engine read seam enforces the hard canonical Node.params shape."""
    with pytest.raises(TypeError, match="Parameter values"):
        bind_params(["x"], {}, _pool())  # type: ignore[list-item]


def test_bind_optional_missing_is_none():
    assert bind_inputs(
        [InputBinding(name="t", contract=STRING, source="${system.t}")], _pool()
    ) == {"t": None}


def test_bind_type_mismatch_raises():
    with pytest.raises(BindingError):
        bind_inputs(
            [InputBinding(name="n", contract=NUMBER, source="${system.n}")],
            _pool(n="not-a-number"),
        )


def test_bind_supplied_string_coerces_only_when_enabled():
    pool = _pool(n="30")
    with pytest.raises(BindingError):
        bind_inputs(
            [InputBinding(name="n", contract=INTEGER, source="${system.n}")],
            pool,
        )

    rec = bind_inputs(
        [
            InputBinding(
                name="n",
                contract=INTEGER,
                source="${system.n}",
                coerce_strings=True,
            )
        ],
        pool,
    )
    assert rec == {"n": 30}


def test_bind_incompatible_stored_default_is_not_coerced():
    # String coercion is caller-input policy. It must not silently rewrite the
    # already-stored default, even when that policy is enabled.
    with pytest.raises(BindingError):
        bind_inputs(
            [
                InputBinding(
                    name="n",
                    contract=INTEGER,
                    has_default=True,
                    default="30",
                    coerce_strings=True,
                )
            ],
            _pool(),
        )


def test_bind_untyped_value_is_lenient():
    # No contract means the read boundary accepts an arbitrary Python value.
    rec = bind_inputs(
        [InputBinding(name="p", source="${system.p}")], _pool(p={"k": 1})
    )
    assert rec == {"p": {"k": 1}}


def test_bind_falsy_default_applied():
    # A falsy param default (0 / False) still fills when the input is OMITTED (no edge).
    rec = bind_inputs(
        [
            InputBinding(
                name="n", contract=INTEGER, has_default=True, default=0
            ),
            InputBinding(
                name="f", contract=BOOLEAN, has_default=True, default=False
            ),
        ],
        _pool(),
    )
    assert rec == {"n": 0, "f": False}


def test_bind_isolates_object_from_pool():
    # The leaf must see ONLY its own record — mutating it must not write the pool.
    pool = _pool()
    pool.set("up", {"k": 1, "nested": {"a": 2}})
    rec = bind_inputs(
        [InputBinding(name="x", contract=OBJECT, source="${up.output}")], pool
    )
    rec["x"]["k"] = 999
    rec["x"]["nested"]["a"] = 999  # nested mutation must not leak either
    assert pool.resolve("up", ["output"]) == {"k": 1, "nested": {"a": 2}}


def test_bind_isolates_untyped_value_from_pool():
    # The untyped path must also isolate the value.
    pool = _pool(p={"k": 1})
    rec = bind_inputs([InputBinding(name="p", source="${system.p}")], pool)
    rec["p"]["k"] = 999
    assert pool.system["p"].to_object() == {"k": 1}


def test_bind_untyped_resolves_and_isolates():
    # contract=None -> resolve + deep-copy, with no type check.
    pool = _pool(topic="ACME")
    pool.set("up", {"k": 1})
    rec = bind_inputs(
        [
            InputBinding(name="topic", source="${system.topic}"),  # type defaults None
            InputBinding(name="cfg", source="${up.output}"),
            InputBinding(name="lit", source=5),
        ],
        pool,
    )
    assert rec == {"topic": "ACME", "cfg": {"k": 1}, "lit": 5}
    rec["cfg"]["k"] = 999  # mutation must not leak (deep-copied)
    assert pool.resolve("up", ["output"]) == {"k": 1}


# --- coalesce + :- default + :? required ------ #


def test_coalesce_first_non_none():
    # ${a | b} branch-join: first non-None among peers (refs).
    pool = VariablePool()
    pool.set("pro", "pro-text")
    # a present -> a wins
    assert bind_inputs(
        [InputBinding(name="t", source="${pro.output | con.output}")], pool
    ) == {"t": "pro-text"}
    # a absent -> falls through to b
    pool2 = VariablePool()
    pool2.set("con", "con-text")
    assert bind_inputs(
        [InputBinding(name="t", source="${pro.output | con.output}")], pool2
    ) == {"t": "con-text"}


def test_coalesce_nary_and_literal_last_segment():
    pool = VariablePool()
    pool.set("c", "c-text")
    # 3-way, first two absent -> third ref wins
    assert bind_inputs(
        [InputBinding(name="t", source="${a.output | b.output | c.output}")],
        pool,
    ) == {"t": "c-text"}
    # all refs absent, a quoted/number literal last segment survives
    empty = VariablePool()
    assert bind_inputs(
        [InputBinding(name="t", source='${a.output | "fallback"}')], empty
    ) == {"t": "fallback"}
    assert bind_inputs(
        [InputBinding(name="n", source="${a.output | 0.0}")], empty
    ) == {"n": 0.0}


def test_default_operator_quoted_rhs_is_literal():
    # `:-` RHS under the unified grammar is a full expression: a QUOTED string is a
    # literal, a number/null is its literal, and a BARE word is a REFERENCE (not a bare
    # string as the legacy binding grammar treated it) — so quote a literal default.
    empty = VariablePool()
    assert bind_inputs([InputBinding(name="d", source='${input.as_of:-"today"}')], empty) == {"d": "today"}
    assert bind_inputs([InputBinding(name="n", source="${input.n:-30}")], empty) == {"n": 30}
    assert bind_inputs([InputBinding(name="z", source="${input.z:-null}")], empty) == {"z": None}
    # a bare word RHS is a ref: unbound -> None (miss), no literal string fallback.
    assert bind_inputs([InputBinding(name="d", source="${input.as_of:-today}")], empty) == {"d": None}
    # present value wins over the default
    pool = VariablePool(); pool.set(START_ID, {"as_of": "2026-06-12"})
    assert bind_inputs([InputBinding(name="d", source='${input.as_of:-"today"}')], pool) == {"d": "2026-06-12"}


def test_required_operator_fails_loud_when_unbound():
    empty = VariablePool()
    with pytest.raises(BindingError, match="a topic is required"):
        bind_inputs([InputBinding(name="t", source="${input.topic:?a topic is required}")], empty)
    # present -> returns the value, no raise
    pool = VariablePool(); pool.set(START_ID, {"topic": "ACME"})
    assert bind_inputs([InputBinding(name="t", source="${input.topic:?required}")], pool) == {"t": "ACME"}


def test_plain_ref_and_literal_unchanged_by_coalesce_parser():
    # backward-compat: a plain single ref and a non-${} literal still resolve as before.
    pool = VariablePool(); pool.set("up", {"pe": 21.0})
    assert bind_inputs([InputBinding(name="pe", source="${up.output.pe}")], pool) == {"pe": 21.0}
    assert bind_inputs([InputBinding(name="k", source=90)], pool) == {"k": 90}


def test_falsy_but_present_value_wins_coalesce():
    # only None is "no value". A present 0 / False / "" / [] must WIN a coalesce,
    # not fall through. (Pins the `is not None` boundary against a `if not value` refactor.)
    for falsy in (0, False, "", []):
        pool = VariablePool()
        pool.set("a", falsy)
        pool.set("b", "b-text")
        rec = bind_inputs(
            [InputBinding(name="t", source="${a.output | b.output}")], pool
        )
        assert rec["t"] == falsy and rec["t"] != "b-text"


def test_required_does_not_raise_on_present_falsy():
    # :? fires only on None — a present False / 0 / "" is a VALUE, returned, no raise.
    for falsy in (False, 0, ""):
        pool = VariablePool(); pool.set(START_ID, {"flag": falsy})
        assert bind_inputs([InputBinding(name="f", source="${input.flag:?req}")], pool) == {"f": falsy}


def test_default_suppressed_by_present_falsy():
    # :- present-value-wins must hold for a falsy present value (0 must not become 99).
    pool = VariablePool(); pool.set(START_ID, {"n": 0})
    assert bind_inputs([InputBinding(name="n", source="${input.n:-99}")], pool) == {"n": 0}


def test_quote_aware_literal_with_pipe_preserved():
    # a quoted literal operand containing `|` is NOT split mid-token (no silent corruption).
    empty = VariablePool()
    assert bind_inputs(
        [InputBinding(name="t", source='${a.output | "x|y"}')], empty
    ) == {"t": "x|y"}
    # and a default value containing `:` inside quotes survives (first-op is the x:- one)
    assert bind_inputs(
        [InputBinding(name="d", source='${input.x:-"a:b"}')], empty
    ) == {"d": "a:b"}


def test_one_level_nested_default_resolves_two_level_raises():
    # a `${x:-${y}}` default resolves ONE nested level; two levels is a loud error
    # (use `|` for multi-way chains).
    pool = VariablePool(); pool.add_system("today", "2026-06-12")
    # as_of unbound -> falls through to the nested ${system.today}
    assert bind_inputs(
        [InputBinding(name="d", source="${input.as_of:-${system.today}}")], pool
    ) == {"d": "2026-06-12"}
    # as_of present -> wins; the nested default is not consulted
    pool.set(START_ID, {"as_of": "2026-01-01"})
    assert bind_inputs(
        [InputBinding(name="d", source="${input.as_of:-${system.today}}")], pool
    ) == {"d": "2026-01-01"}
    # two-level nesting -> loud error. The unified `WRAPPED_REF` terminal matches to the
    # FIRST `}`, so `${x:-${system.today}` grabs the inner span and the dangling `}` makes
    # the interior unparseable — still a rejected `BindingError` (use `|` for chains).
    with pytest.raises(BindingError):
        bind_inputs([InputBinding(name="d", source="${input.as_of:-${x:-${system.today}}}")], pool)


def test_item_scope_combined_with_coalesce_and_default():
    # ${item.x | b.output}: item miss -> ref peer; item present -> wins.
    pool = VariablePool(); pool.set("b", "b-text")
    assert bind_inputs([InputBinding(name="v", source="${item.x | b.output}")],
                       pool, item={"y": 1}) == {"v": "b-text"}
    assert bind_inputs([InputBinding(name="v", source="${item.x | b.output}")],
                       pool, item={"x": 7}) == {"v": 7}


def test_whole_ref_typed_embedded_stringified_dollar_escape():
    # a value that is exactly one ${...} -> the TYPED value (type preserved); a
    # ${...} embedded in text -> stringified; $$ -> a literal $. (The binding parser
    # is shared with validation — the old runtime-vs-validation literal parity test is
    # obsolete now that there is a single parser, not two duplicated copies.)
    pool = VariablePool()
    pool.set("a", 0.7)
    pool.set("b", ["ACME", "BETA"])
    # whole ${...} -> the float (not the string "0.7")
    rec = bind_inputs([InputBinding(name="x", source="${a.output}")], pool)
    assert rec == {"x": 0.7} and isinstance(rec["x"], float)
    # whole ${...} of a list -> the list (type preserved)
    assert bind_inputs(
        [InputBinding(name="x", source="${b.output}")], pool
    ) == {"x": ["ACME", "BETA"]}
    # embedded ${...} in text -> a string with the value interpolated
    assert bind_inputs(
        [InputBinding(name="x", source="pe=${a.output}")], pool
    ) == {"x": "pe=0.7"}
    # $$ -> a literal $ (no interpolation)
    assert bind_inputs(
        [InputBinding(name="x", source="cost is $$5")], pool
    ) == {"x": "cost is $5"}


def test_bind_inputs_typed_record_from_declared_inputs():
    # bind_inputs is the read boundary the engine seam (eval_node) uses to turn a node's
    # declared inputs into a typed record (was: the deleted Node._bind_inputs helper).
    bindings = [InputBinding(name="t", contract=STRING, source="${system.t}")]
    assert bind_inputs(bindings, _pool(t="X")) == {"t": "X"}


def test_bind_inputs_resolves_item_from_local_scope():
    pool = VariablePool()
    rec = bind_inputs([InputBinding(name="topic", contract=STRING, source="${item}")],
                      pool, item="ACME")
    assert rec == {"topic": "ACME"}


def test_bind_inputs_item_dotted_walk():
    rec = bind_inputs([InputBinding(name="v", source="${item.value}")],
                      VariablePool(), item={"value": 7})
    assert rec == {"v": 7}


def test_bind_inputs_item_unset_outside_map_is_none():
    # ${item} with no item provided resolves to None (the compile-time scan forbids
    # ${item} outside a MAP body, so this only guards a defensive path).
    rec = bind_inputs([InputBinding(name="x", source="${item}")], VariablePool())
    assert rec == {"x": None}


def test_bind_inputs_non_item_ref_still_uses_pool_with_item_present():
    pool = VariablePool(); pool.set(START_ID, {"as_of": "2026-01-01"})
    rec = bind_inputs([InputBinding(name="d", source="${input.as_of}")], pool, item="X")
    assert rec == {"d": "2026-01-01"}
