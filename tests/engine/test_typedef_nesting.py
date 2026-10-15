"""Tests for how `read_typedefs` walks typedef references: the walks are iterative, so a
long alias chain or a long record cycle never raises `RecursionError`, and a cycle is
reported at the typedef and field where the walk closes it.

Each typedef is also measured against `MAX_TYPE_NESTING` (100 by default), dependencies
first. Its levels (each record and list is a level, a union counts as its deepest member,
a typedef name as deep as its definition) must stay within the limit, or the error reads
`typedef 'T' field 'a': type nests deeper than 100 levels, the limit for a type (each
record and list is a level); its type 'X' adds 100` (an alias has no field and no
`; its type` tail). So must its chain, the typedefs on the longest path of names from it,
itself included: `typedef 'A': it ends a chain of 101 typedefs, each naming the next, more
than 100, the limit for a type; write its type with fewer typedefs in between`.
"""

import time

import pytest

from yian.compose import LoadError, load_flow
from yian.defaults import Defaults
from yian.typesys.types import (
    RecordDef,
    RefExpr,
    _measure_typedef,
    _parse_field,
    _typedef_parts,
    read_typedefs,
    resolve_type,
    type_for,
)
from yian.typesys.values import TypeCheckError, type_levels


def test_a_long_alias_chain_fails_on_its_length_quickly():
    raw = {"A0": "int"}
    for i in range(1, 5000):
        raw[f"A{i}"] = f"A{i - 1}"
    start = time.monotonic()
    with pytest.raises(TypeCheckError) as info:
        read_typedefs(raw)
    assert time.monotonic() - start < 5
    assert str(info.value).startswith("typedef 'A100': it ends a chain of 101 typedefs")


@pytest.mark.parametrize("n", [10, 200, 1000])
def test_record_cycle_reports_at_the_last_typedef(n):
    raw = {f"T{i}": {"a": f"T{(i + 1) % n}"} for i in range(n)}
    last = f"T{n - 1}"
    with pytest.raises(TypeCheckError) as info:
        read_typedefs(raw)
    assert str(info.value) == f"typedef {last!r} field 'a': recursive type reference: {last!r}"
    assert info.value.path == (last, "a")


def test_self_recursive_record():
    with pytest.raises(TypeCheckError) as info:
        read_typedefs({"R": {"a": "R"}})
    assert str(info.value) == "typedef 'R' field 'a': recursive type reference: 'R'"
    assert info.value.path == ("R", "a")


def test_two_record_cycle():
    with pytest.raises(TypeCheckError) as info:
        read_typedefs({"A": {"x": "B"}, "B": {"y": "A"}})
    assert str(info.value) == "typedef 'B' field 'y': recursive type reference: 'B'"
    assert info.value.path == ("B", "y")


def test_cycle_through_an_alias_and_a_record():
    with pytest.raises(TypeCheckError) as info:
        read_typedefs({"A": "B", "B": {"x": "A"}})
    assert str(info.value) == "typedef 'B' field 'x': recursive type reference: 'B'"
    assert info.value.path == ("B", "x")


def test_alias_cycle():
    with pytest.raises(TypeCheckError) as info:
        read_typedefs({"A": "B", "B": "A"})
    assert "alias cycle through" in str(info.value)
    assert info.value.path == ("A",)


def test_field_order_decides_between_a_union_error_and_a_cycle():
    with pytest.raises(TypeCheckError) as info:
        read_typedefs({
            "U1": {"x": "int"},
            "U2": {"x": "int"},
            "R": {"a": "Union(U1, U2)", "b": "R"},
        })
    assert str(info.value) == (
        "typedef 'R' field 'a': Union(U1, U2): members 'U1' and 'U2' can match the same "
        "value; add a tag field every member declares as a single-value Literal, with a "
        "different value in each member, e.g. kind: Literal(u1)"
    )
    assert info.value.path == ("R", "a")


def test_aliases_wrapped_in_optional_resolve():
    raw = {"A0": {"a": "int"}}
    for i in range(1, 20):
        raw[f"A{i}"] = "Optional(" * 99 + f"A{i - 1}" + ")" * 99
    read_typedefs(raw)


def test_aliases_wrapped_in_union_resolve():
    raw = {"A0": {"a": "int"}}
    for i in range(1, 100):
        raw[f"A{i}"] = "Union(" * 99 + f"A{i - 1}" + ")" * 99
    read_typedefs(raw)


def _flow(typedefs: dict[str, str], output: str) -> str:
    """Return a flow declaring `typedefs` whose CODE node `use` has `output: <output>`."""
    lines = "".join(f"  {name}: {defn}\n" for name, defn in typedefs.items())
    return (
        f"id: f\nname: f\ntypedefs:\n{lines}nodes:\n  use:\n    kind: code\n"
        f"    code: tests.engine._compose_codefns:echo\n    input:\n      topic: x\n"
        f"    output: {output}\noutput: ${{use.output}}\n"
    )


def _union_levels(limit: int, tagged: bool) -> str:
    """Return a flow of `limit` record levels, each above the bottom one holding an
    optional union of the level below and `S`; the bottom field nests `limit` Optionals."""
    tag = "k: Literal(r), " if tagged else ""
    deep = "Optional(" * limit + "int" + ")" * limit
    typedefs = {"S": "{k: Literal(s)}" if tagged else "{x: int}", "T0": f"{{{tag}a: '{deep}'}}"}
    for i in range(1, limit):
        typedefs[f"T{i}"] = f"{{{tag}a: 'Optional(Union(T{i - 1}, S))'}}"
    return _flow(typedefs, f"T{limit - 1}")


def _unions_over_a_deep_union(limit: int) -> str:
    """Return a flow like `_union_levels` with `limit - 1` records over a bottom field of
    `limit` nested unions of `S`, so `S` is the last of `limit` levels."""
    deep = "Union(" * limit + "S" + ")" * limit
    typedefs = {"S": "{k: Literal(s)}", "T0": f"{{k: Literal(r), a: '{deep}'}}"}
    for i in range(1, limit - 1):
        typedefs[f"T{i}"] = f"{{k: Literal(r), a: 'Optional(Union(T{i - 1}, S))'}}"
    return _flow(typedefs, f"T{limit - 2}")


def _records_over_a_deep_union(limit: int) -> str:
    """Return a flow of `limit - 1` records, each the only field of the next, over a
    bottom field of `limit` nested unions of `S`."""
    deep = "Union(" * limit + "S" + ")" * limit
    typedefs = {"S": "{x: int}", "T0": f"{{a: '{deep}'}}"}
    for i in range(1, limit - 1):
        typedefs[f"T{i}"] = f"{{a: T{i - 1}}}"
    return _flow(typedefs, f"T{limit - 2}")


@pytest.mark.parametrize("limit", [None, 125])
@pytest.mark.parametrize(
    "build",
    [
        lambda limit: _union_levels(limit, tagged=True),
        lambda limit: _union_levels(limit, tagged=False),
        _unions_over_a_deep_union,
        _records_over_a_deep_union,
    ],
    ids=["tagged-unions", "untagged-unions", "unions-over-union", "records-over-union"],
)
def test_types_at_the_limit_load(build, limit, set_defaults):
    if limit is not None:
        set_defaults(MAX_TYPE_NESTING=limit)
    load_flow(build(limit or Defaults().max_type_nesting))


def test_field_parse_follows_the_int_digit_limit(set_defaults):
    raw = {"R": {"a": "int(0..123456)"}}
    read_typedefs(raw)
    set_defaults(MAX_INT_DIGITS=3)
    with pytest.raises(TypeCheckError):
        read_typedefs(raw)


def test_field_parse_follows_the_type_nesting_limit(set_defaults):
    raw = {"R": {"a": "list(" * 6 + "int" + ")" * 6}}
    read_typedefs(raw)
    set_defaults(MAX_TYPE_NESTING=5)
    with pytest.raises(TypeCheckError):
        read_typedefs(raw)


def _records(n: int) -> dict:
    """Return `n` records, `T0: {a: int}` and each next one `{a: <previous>}`."""
    raw = {"T0": {"a": "int"}}
    for i in range(1, n):
        raw[f"T{i}"] = {"a": f"T{i - 1}"}
    return raw


def _aliases(n: int, prefix: str = "A", bottom: object = "int", wrap: str = "{}") -> dict:
    """Return `n` aliases over `bottom`, each the previous name wrapped by `wrap`."""
    raw = {f"{prefix}0": bottom}
    for i in range(1, n):
        raw[f"{prefix}{i}"] = wrap.format(f"{prefix}{i - 1}")
    return raw


def _levels_message(name: str, field: str, text: str, adds: int, limit: int = 100) -> str:
    return (
        f"typedef {name!r} field {field!r}: type nests deeper than {limit} levels, the limit "
        f"for a type (each record and list is a level); its type {text!r} adds {adds}"
    )


def _chain_message(name: str, chain: int, limit: int = 100) -> str:
    return (
        f"typedef {name!r}: it ends a chain of {chain} typedefs, each naming the next, more "
        f"than {limit}, the limit for a type; write its type with fewer typedefs in between"
    )


@pytest.mark.parametrize("n", [101, 1000])
def test_a_record_chain_fails_on_levels(n):
    with pytest.raises(TypeCheckError) as info:
        read_typedefs(_records(n))
    assert str(info.value) == _levels_message("T100", "a", "T99", 100)
    assert info.value.path == ("T100", "a")


def test_a_record_chain_at_the_limit_loads():
    read_typedefs(_records(100))


@pytest.mark.parametrize("n", [101, 1000])
def test_an_alias_chain_fails_on_its_length(n):
    with pytest.raises(TypeCheckError) as info:
        read_typedefs(_aliases(n))
    assert str(info.value) == _chain_message("A100", 101)
    assert info.value.path == ("A100",)


def test_an_alias_chain_at_the_limit_loads():
    read_typedefs(_aliases(100))


def test_an_optional_alias_chain_fails_on_its_length():
    with pytest.raises(TypeCheckError) as info:
        read_typedefs(_aliases(101, "O", wrap="Optional({})"))
    assert str(info.value) == _chain_message("O100", 101)
    assert info.value.path == ("O100",)


def test_a_list_alias_chain_over_a_scalar_fails_on_its_length():
    with pytest.raises(TypeCheckError) as info:
        read_typedefs(_aliases(101, "L", wrap="list({})"))
    assert str(info.value) == _chain_message("L100", 101)
    assert info.value.path == ("L100",)


def test_a_list_alias_chain_over_a_record_fails_on_levels():
    with pytest.raises(TypeCheckError) as info:
        read_typedefs(_aliases(101, "L", bottom={"a": "int"}, wrap="list({})"))
    assert str(info.value) == (
        "typedef 'L100': type nests deeper than 100 levels, the limit for a type (each "
        "record and list is a level)"
    )
    assert info.value.path == ("L100",)


def test_a_list_over_a_deep_record_fails_on_levels():
    raw = {**_records(100), "T": "list(T99)"}
    with pytest.raises(TypeCheckError) as info:
        read_typedefs(raw)
    assert str(info.value).startswith("typedef 'T': type nests deeper than 100 levels")
    assert info.value.path == ("T",)


def test_a_record_field_holding_a_long_alias_chain_fails_on_the_chain():
    raw = {**_aliases(100), "R": {"x": "int", "a": "A99"}}
    with pytest.raises(TypeCheckError) as info:
        read_typedefs(raw)
    assert str(info.value) == (
        "typedef 'R' field 'a': it ends a chain of 101 typedefs, each naming the next, more "
        "than 100, the limit for a type; write its type with fewer typedefs in between"
    )
    assert info.value.path == ("R", "a")


def test_a_lower_limit_bounds_records_and_aliases(set_defaults):
    set_defaults(MAX_TYPE_NESTING=5)
    read_typedefs(_records(5))
    read_typedefs(_aliases(5))
    with pytest.raises(TypeCheckError) as info:
        read_typedefs(_records(6))
    assert str(info.value) == _levels_message("T5", "a", "T4", 5, limit=5)
    with pytest.raises(TypeCheckError) as info:
        read_typedefs(_aliases(6))
    assert str(info.value) == _chain_message("A5", 6, limit=5)


def _flow_lines(raw: dict) -> dict[str, str]:
    """Render each typedef of `raw` as the flow-style YAML `_flow` writes."""
    return {
        name: "{" + ", ".join(f"{f}: {t}" for f, t in defn.items()) + "}"
        if isinstance(defn, dict) else defn
        for name, defn in raw.items()
    }


def _causes(exc: BaseException):
    while exc is not None:
        yield exc
        exc = exc.__cause__ or exc.__context__


@pytest.mark.parametrize(
    "raw, prefix, column",
    [
        (_records(1000), "bad typedefs: typedef 'T100' field 'a'", len("  T100: {a: ") + 1),
        (_records(300), "bad typedefs: typedef 'T100' field 'a'", len("  T100: {a: ") + 1),
        (_aliases(1000), "bad typedefs: typedef 'A100'", len("  ") + 1),
    ],
    ids=["1000-records", "300-records", "1000-aliases"],
)
def test_a_flow_with_a_long_chain_is_a_located_load_error(raw, prefix, column):
    last = list(raw)[-1]
    with pytest.raises(LoadError) as info:
        load_flow(_flow(_flow_lines(raw), last))
    assert str(info.value).startswith(prefix)
    # Line 4 holds the first typedef, so the 101st is on line 104.
    span = info.value.diagnostic.frames[-1].range
    assert (span.start_line, span.start_col) == (104, column)
    assert not any(isinstance(exc, RecursionError) for exc in _causes(info.value))


def _diamond(n: int) -> dict:
    """Return `n` records, each holding the previous one in two fields."""
    raw = {"D0": {"x": "int"}}
    for i in range(1, n):
        raw[f"D{i}"] = {"x": f"D{i - 1}", "y": f"D{i - 1}"}
    return raw


def _measure_all(registry: dict) -> dict[str, int]:
    """Measure the typedefs of `registry`, listed dependencies first; return their levels."""
    levels: dict[str, int] = {}
    chain: dict[str, int] = {}
    for name, defn in registry.items():
        parts = [
            (field, _parse_field(t) if isinstance(t, str) else t)
            for field, t in _typedef_parts(defn)
        ]
        _measure_typedef(name, defn, parts, levels, chain)
    return levels


def test_measuring_a_diamond_is_memoized():
    start = time.monotonic()
    # Built by hand: `read_typedefs` would resolve the diamond, expanding 2**60 records.
    registry = {name: RecordDef(fields=fields) for name, fields in _diamond(61).items()}
    levels = _measure_all(registry)
    assert time.monotonic() - start < 5
    assert levels["D60"] == 61


def test_a_flow_with_a_diamond_loads():
    raw = _diamond(12)
    load_flow(_flow(_flow_lines(raw), "D11"))


def test_an_empty_record_and_an_enum_load():
    registry = read_typedefs({"E": {}, "V": "Literal(a)"})
    assert _measure_all(registry) == {"E": 1, "V": 0}


def test_measured_levels_match_the_resolved_type():
    raw = {
        "E": {},
        "V": "Literal(a, b)",
        "B": "int(1..5)",
        "R": {"a": "int", "b": "list(str)", "c": "Optional(E)"},
        "S": {"k": "Literal(s)", "r": "R"},
        "U": "Union(R, S)",
        "W": {"u": "U", "t": "topics", "d": "dict(str, int)", "o": "object", "v": "V"},
        "L": "list(list(W))",
        "M": {"l": "Optional(L)", "b": "B", "e": "list(E)"},
    }
    registry = read_typedefs(raw)
    measured = _measure_all(registry)
    for name in raw:
        assert measured[name] == type_levels(resolve_type(RefExpr(name), registry)), name


_EXPANDED = "nests deeper than 100 levels once its typedef names are expanded"


def _wrapped(name: str, lists: int = 100) -> str:
    """Return `name` wrapped in `lists` nested `list(...)`s."""
    return "list(" * lists + name + ")" * lists


def test_type_for_bounds_a_type_with_its_typedefs_expanded():
    registry = read_typedefs(_records(100))
    with pytest.raises(TypeCheckError) as info:
        type_for(_wrapped("T99"), registry)
    assert _EXPANDED in str(info.value)


def test_type_for_at_the_limit_resolves():
    registry = read_typedefs(_records(99))
    assert type_levels(type_for(_wrapped("T98", 1), registry)) == 100
    type_for(_wrapped("int"), {})


def _line_of(text: str, needle: str) -> int:
    """Return the 1-based line of the first `needle` in `text`."""
    return text[: text.index(needle)].count("\n") + 1


def _node_input_tag_flow(typedefs: dict[str, str], tag: str) -> str:
    """Return a flow whose CODE node passes `topic` an empty list tagged `!<tag>`."""
    return _flow(typedefs, "T0").replace("      topic: x\n", f"      topic: !{tag} []\n")


def _flow_input_tag_flow(typedefs: dict[str, str], tag: str) -> str:
    """Return a flow declaring a flow input `t` with the tag `!<tag>`."""
    return _flow(typedefs, "T0").replace("name: f\n", f"name: f\ninput:\n  t: !{tag}\n", 1)


@pytest.mark.parametrize(
    "build, needle",
    [
        (lambda typedefs, t: _flow(typedefs, f"'{t}'"), "    output: 'list("),
        (_node_input_tag_flow, "      topic: !list("),
        (_flow_input_tag_flow, "  t: !list("),
    ],
    ids=["output", "node-input-tag", "flow-input-tag"],
)
def test_a_flow_wrapping_a_deep_typedef_is_a_located_load_error(build, needle):
    text = build(_flow_lines(_records(100)), _wrapped("T99"))
    with pytest.raises(LoadError) as info:
        load_flow(text)
    assert info.value.diagnostic.frames[-1].range.start_line == _line_of(text, needle)
    assert not any(isinstance(exc, RecursionError) for exc in _causes(info.value))
