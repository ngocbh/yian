"""What a condition proves at each of its reads: checks, guards, and literals.

A **check** is a test a condition makes on one value: `isinstance(<ref>, T)`, a tag
comparison `<ref>.<field> == "<text>"` (`!=` is the same check, negated), or a null check
`<ref> != null` (`==` is the same check, negated). A **guard** is a
small boolean formula over checks in negation normal form: `Holds(check, value)` says the
check evaluated to `value`; `AllOf` and `AnyOf` combine guards; `NOTHING` proves nothing,
and `NEVER` never holds.

`and` and `or` short-circuit, so in `A and B` the operand `B` only runs once `A` is true,
and in `A or B` once `A` is false; `not` flips what a check proves. `proves` says what holds
when a condition evaluates to a value, `guarded_refs` says what holds whenever each reference
is read, and `literals_in_context` says what holds whenever each check is evaluated. This
module is syntax only: what a guard means for a value's type is `narrowing_facts`'s job.

A new check form adds a frozen dataclass with a `base` path, adds it to `Check`, and
handles it in every function that dispatches on the check's kind; each of those raises
`unknown_check` for a kind it doesn't know, so a form left out fails loudly:

- here: its recognition in `check_in` and its source in `check_text`;
- `narrowing_facts.check_fact`: the facts it proves, true and false;
- `narrowing_facts.unresolved_type`: whether it names a type that does not resolve;
- `narrowing_facts._declared_impossible`: when its base's type alone decides it, and
  `narrowing_facts._never_value`: the value such a reported check never has;
- `validate.asserts._narrows`: whether it can narrow an assert's root.

`proves(<and chain>, False)` repeats each prefix of the chain, so a guard grows
quadratically with the length of one chain. An operand that makes no check proves `NOTHING`:
it drops out of a true `and` chain (a false `or` chain), and it makes a false `and` chain (a
true `or` chain) prove `NOTHING`, since it may be the operand that decided the chain. So guards
of authored conditions stay small.

Knows about: `expr` parse trees and `direct_reference_parts`.
Never imports: types, `compose.validate`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Collection, Optional, TypeAlias

from lark import Token, Tree

from yian.compose.helpers.output_contracts import direct_reference_parts
from yian.expr import isinstance_subject, isinstance_type_names
from yian.expr.expressions import ISINSTANCE

__all__ = [
    "NEVER",
    "NOTHING",
    "AllOf",
    "AnyOf",
    "Check",
    "CheckLiteral",
    "Guard",
    "Holds",
    "IsinstanceCheck",
    "NotNullCheck",
    "TagCheck",
    "all_of",
    "any_of",
    "check_in",
    "check_text",
    "conjuncts",
    "falsified",
    "guard_checks",
    "guarded_refs",
    "literal_text",
    "literals_in_context",
    "masked",
    "proves",
    "unknown_check",
]


@dataclass(frozen=True)
class IsinstanceCheck:
    """An `isinstance(<base>, <type_name>)` check on a plain reference.

    Args:
        base (`str`):
            The dotted path of the tested reference, such as `"d.output"`.
        type_name (`str`):
            The type name the check names.
    """

    base: str
    type_name: str


@dataclass(frozen=True)
class TagCheck:
    """A comparison of a field of a plain reference with a text literal.

    `<base>.<field> == "<value>"` and `"<value>" == <base>.<field>` are this check;
    `!=` is the same check, negated.

    Args:
        base (`str`):
            The dotted path of the value whose field is compared, such as `"v"`.
        field (`str`):
            The compared field's name.
        value (`str`):
            The text, decoded as the evaluator decodes a string literal (the quotes
            dropped, nothing unescaped).
    """

    base: str
    field: str
    value: str


@dataclass(frozen=True)
class NotNullCheck:
    """A comparison of a plain reference with `null`: the check that its value is not null.

    `<base> != null` and `null != <base>` are this check, true when the value is not
    null; `==` is the same check, negated. `none` spells `null` too.

    Args:
        base (`str`):
            The dotted path of the compared reference, such as `"input.a.b"`.
    """

    base: str


Check: TypeAlias = IsinstanceCheck | TagCheck | NotNullCheck
"""One check a condition can make on a value."""


def unknown_check(check: object) -> TypeError:
    """Return the error a dispatch on a check's kind raises for a kind it doesn't handle.

    Args:
        check (`object`):
            The check.

    Returns:
        `TypeError`:
            The error, naming the check's class.
    """
    return TypeError(f"unhandled check kind {type(check).__name__}")


@dataclass(frozen=True)
class Holds:
    """The guard that `check` evaluated to `value`.

    Args:
        check (`Check`):
            The check.
        value (`bool`):
            What it evaluated to.
    """

    check: Check
    value: bool


@dataclass(frozen=True)
class AllOf:
    """The guard that every part holds. Build one with `all_of`.

    Args:
        parts (`tuple[Guard, ...]`):
            The parts; `()` is `NOTHING`, which proves nothing.
    """

    parts: tuple["Guard", ...]


@dataclass(frozen=True)
class AnyOf:
    """The guard that at least one part holds. Build one with `any_of`.

    Args:
        parts (`tuple[Guard, ...]`):
            The parts, at least two when built by `any_of`.
    """

    parts: tuple["Guard", ...]


Guard: TypeAlias = Holds | AllOf | AnyOf
"""A boolean formula over checks, in negation normal form."""

NOTHING = AllOf(())
"""The guard that proves nothing."""

NEVER = AnyOf(())
"""The guard that never holds, such as what a check that is never true proves when true."""


def all_of(*parts: Guard) -> Guard:
    """Return the guard that every part holds.

    Nested `AllOf`s are flattened and `NOTHING` parts dropped; one remaining part is
    returned as is. A `NEVER` part makes the whole guard `NEVER`.

    Args:
        *parts (`Guard`):
            The parts.

    Returns:
        `Guard`:
            `NEVER` when a part is `NEVER`, `NOTHING` for no parts, the lone part for one,
            else an `AllOf`.

    Example:
        `all_of(a, NOTHING, all_of(b, c))` is `AllOf((a, b, c))`.
    """
    flat: list[Guard] = []
    for part in parts:
        if part == NEVER:
            return NEVER
        flat.extend(part.parts if isinstance(part, AllOf) else (part,))
    if len(flat) == 1:
        return flat[0]
    return AllOf(tuple(flat))


def any_of(*parts: Guard) -> Guard:
    """Return the guard that at least one part holds.

    Nested `AnyOf`s are flattened (so `NEVER` parts drop out); a part that proves nothing
    makes the whole guard prove nothing. One part is returned as is.

    Args:
        *parts (`Guard`):
            The parts.

    Returns:
        `Guard`:
            `NOTHING` when a part is `NOTHING`, `NEVER` when no part remains, the lone
            part for one, else an `AnyOf`.

    Example:
        `any_of(a, any_of(b, c))` is `AnyOf((a, b, c))`; `any_of(a, NOTHING)` is
        `NOTHING`.
    """
    flat: list[Guard] = []
    for part in parts:
        if part == NOTHING:
            return NOTHING
        flat.extend(part.parts if isinstance(part, AnyOf) else (part,))
    if len(flat) == 1:
        return flat[0]
    return AnyOf(tuple(flat))


def _operands(node: Tree) -> list[Tree | Token]:
    """The operands of an `and_expr` or `or_expr`, without the operator tokens."""
    return [
        child
        for child in node.children
        if not (isinstance(child, Token) and child.type in ("AND", "OR"))
    ]


def _is_text(node: Tree | Token) -> bool:
    """Whether `node` is a string literal token."""
    return isinstance(node, Token) and node.type == "STRING"


def _is_null(node: Tree | Token) -> bool:
    """Whether `node` is the `null` (or `none`) literal token."""
    return isinstance(node, Token) and node.type == "NULL"


def check_in(tree: Tree | Token) -> Optional[tuple[Check, bool]]:
    """Return the check a sub-expression is, and the value it has when the sub-expression is true.

    Args:
        tree (`Tree | Token`):
            A parsed sub-expression of a condition.

    Returns:
        `tuple[Check, bool] | None`:
            `(IsinstanceCheck, True)` for `isinstance(<plain ref>, T)`;
            `(TagCheck, True)` for `<plain ref>.<field> == "<text>"` either way round, and
            `(TagCheck, False)` for `!=`; `(NotNullCheck, True)` for `<plain ref> != null`
            either way round, and `(NotNullCheck, False)` for `==`; `None` for anything
            else (a number or bool operand, a call, a tag comparison on a reference
            without a field, `not`, ...).

    Example:
        `check_in(parse_condition("v.action != 'buy'"))` is
        `(TagCheck("v", "action", "buy"), False)`, and
        `check_in(parse_condition("v != null"))` is `(NotNullCheck("v"), True)`.
    """
    subject = isinstance_subject(tree)
    if subject is not None:
        parts = direct_reference_parts(subject)
        if parts is None:
            return None
        return IsinstanceCheck(".".join(parts), isinstance_type_names(tree)[0]), True
    if not isinstance(tree, Tree) or tree.data != "compare":
        return None
    left, op, right = tree.children
    if str(op) not in ("==", "!="):
        return None
    if _is_null(left):
        left, right = right, left
    if _is_null(right):
        parts = direct_reference_parts(left)
        if parts is None:
            return None
        return NotNullCheck(".".join(parts)), str(op) == "!="
    if _is_text(left):
        left, right = right, left
    if not _is_text(right):
        return None
    parts = direct_reference_parts(left)
    if parts is None or len(parts) < 2:
        return None
    # The evaluator decodes a string literal by dropping its quotes, with no unescaping.
    value = str(right)[1:-1]
    return TagCheck(".".join(parts[:-1]), parts[-1], value), str(op) == "=="


def proves(tree: Tree | Token, value: bool) -> Guard:
    """Return what holds when a condition evaluates to `value`.

    A check gives `Holds`; `not` proves what its operand proves for the other value;
    an `and` chain is true when every operand is, and false when some operand is false
    after the ones before it were true (an `or` chain the other way round); a false
    `and` chain with an operand that proves nothing when false proves `NOTHING`, since
    that operand may have decided it. Anything else (a comparison that is no check,
    `:-`, a call, a literal) proves `NOTHING`.

    Args:
        tree (`Tree | Token`):
            A parsed condition or sub-expression.
        value (`bool`):
            The value it evaluates to.

    Returns:
        `Guard`:
            What holds then.

    Example:
        `proves(parse_condition("isinstance(v, Buy) or v.action == 'sell'"), False)` is
        `AllOf((Holds(IsinstanceCheck("v", "Buy"), False),
        Holds(TagCheck("v", "action", "sell"), False)))`.
    """
    found = check_in(tree)
    if found is not None:
        check, positive = found
        return Holds(check, value == positive)
    if not isinstance(tree, Tree):
        return NOTHING
    if tree.data == _NEVER_HAS:
        return NEVER if value == (str(tree.children[0]) == "true") else NOTHING
    if tree.data == "negate":
        return proves(tree.children[-1], not value)
    if tree.data not in ("and_expr", "or_expr"):
        return NOTHING
    operands = _operands(tree)
    # An `and` is decided by its first false operand, an `or` by its first true one.
    deciding = tree.data == "or_expr"
    if value != deciding:
        return all_of(*(proves(operand, value) for operand in operands))
    decided = [proves(operand, deciding) for operand in operands]
    # An operand that proves nothing may be the one that decided the chain.
    if NOTHING in decided:
        return NOTHING
    branches: list[Guard] = []
    passed: list[Guard] = []
    for operand, guard in zip(operands, decided):
        branches.append(all_of(*passed, guard))
        passed.append(proves(operand, not deciding))
    return any_of(*branches)


def _value_args(refcall: Tree) -> Optional[list[Tree]]:
    """The `arg` subtrees whose values a call evaluates, or `None` for a reference."""
    call_suffix = next(
        (c for c in refcall.children if isinstance(c, Tree) and c.data == "call_suffix"),
        None,
    )
    if call_suffix is None:
        return None
    args = call_suffix.children
    # `isinstance(x, T)` evaluates only `x`; `T` is a type name.
    return args[:1] if str(refcall.children[0]) == ISINSTANCE else args


def _operand_contexts(node: Tree, context: Guard) -> list[tuple[Tree | Token, Guard]]:
    """Each operand of an `and_expr`/`or_expr` with what holds whenever it is evaluated."""
    # The value an operand must have for the next one to run.
    continues = node.data == "and_expr"
    contexts = []
    passed: list[Guard] = [context]
    for operand in _operands(node):
        contexts.append((operand, all_of(*passed)))
        passed.append(proves(operand, continues))
    return contexts


def guarded_refs(tree: Tree | Token) -> list[tuple[str, Guard]]:
    """Return each reference occurrence in a condition with the guard that holds when it is read.

    The `i`-th operand of an `and` chain is evaluated only once the operands before it
    were true, and of an `or` chain once they were false; every other node (`not`,
    `:-`, `|`, a call, a comparison) evaluates its parts under its own guard. So a check
    after a read never guards it. The occurrences are those of `expr_refs` and
    `condition_reads`, in the same order: plain references, builtin-call argument values,
    and an `isinstance` call's subject.

    Args:
        tree (`Tree | Token`):
            A condition parsed with `parse_condition`, in any spelling.

    Returns:
        `list[tuple[str, Guard]]`:
            One `(reference path, guard)` pair per occurrence.

    Example:
        For `not isinstance(r, Done) or r.answer == 'ok'`, the pairs are
        `("r", NOTHING)` and `("r.answer", Holds(IsinstanceCheck("r", "Done"), True))`.
    """
    reads: list[tuple[str, Guard]] = []

    def walk(node: Tree | Token, context: Guard) -> None:
        if isinstance(node, Token):
            return
        if node.data == "refcall":
            args = _value_args(node)
            if args is None:
                segments = [str(child.children[0]) for child in node.children[1:]]
                reads.append((".".join([str(node.children[0]), *segments]), context))
                return
            for arg in args:
                # A keyword argument's value is its last child.
                walk(arg.children[-1], context)
            return
        if node.data in ("and_expr", "or_expr"):
            for operand, operand_context in _operand_contexts(node, context):
                walk(operand, operand_context)
            return
        for child in node.children:
            walk(child, context)

    walk(tree, NOTHING)
    return reads


@dataclass(frozen=True)
class CheckLiteral:
    """A check, or `not` applied to a check, in a boolean position of a condition.

    Args:
        context (`Guard`):
            What holds whenever the literal is evaluated.
        check (`Check`):
            The check.
        value (`bool`):
            The value the check has when the literal is true.
        negated (`bool`):
            Whether the literal is `not` applied to the check (after `not not` cancels).
        node (`Tree | Token | None`, *optional*, defaults to `None`):
            The check's sub-tree in the parsed condition, for `masked`. Not compared.
    """

    context: Guard
    check: Check
    value: bool
    negated: bool
    node: Tree | Token | None = field(default=None, compare=False, repr=False)


def literals_in_context(tree: Tree | Token) -> list[CheckLiteral]:
    """Return each literal in a condition's boolean positions with what holds when it is evaluated.

    Boolean positions are the condition itself, the operands of `and`, `or` and `not`,
    and the first operand of `:-` and `:?`. A check anywhere else (a comparison operand,
    a call argument, a list item, an operand of `|`) is no literal. Contexts follow the
    rules of `guarded_refs`.

    Args:
        tree (`Tree | Token`):
            A condition parsed with `parse_condition`, in any spelling.

    Returns:
        `list[CheckLiteral]`:
            The literals in source order.

    Example:
        For `isinstance(v, Buy) and v.action == 'hold'`, the second literal is
        `CheckLiteral(Holds(IsinstanceCheck("v", "Buy"), True), TagCheck("v", "action",
        "hold"), True, False)`.
    """
    literals: list[CheckLiteral] = []

    def walk(node: Tree | Token, context: Guard, negated: bool) -> None:
        found = check_in(node)
        if found is not None:
            check, positive = found
            literals.append(CheckLiteral(context, check, positive != negated, negated, node))
            return
        if not isinstance(node, Tree):
            return
        if node.data == "negate":
            walk(node.children[-1], context, not negated)
        elif node.data in ("and_expr", "or_expr"):
            for operand, operand_context in _operand_contexts(node, context):
                walk(operand, operand_context, False)
        elif node.data in ("default_expr", "required_expr"):
            walk(node.children[0], context, False)

    walk(tree, NOTHING, False)
    return literals


_MASKED = "masked_check"
"""The data of a node `masked` wraps a check in; no rule here reads it as a check."""


def masked(tree: Tree | Token, nodes: Collection[Tree | Token]) -> Tree | Token:
    """Return a copy of a condition in which the given sub-trees prove nothing.

    Each sub-tree in `nodes` (matched by identity, such as a `CheckLiteral.node`) is
    wrapped in a node no rule reads as a check, so `proves` gives `NOTHING` for it while
    `guarded_refs` still finds the same references in the same order.

    Args:
        tree (`Tree | Token`):
            A condition parsed with `parse_condition`.
        nodes (`Collection[Tree | Token]`):
            Sub-trees of `tree` to mask.

    Returns:
        `Tree | Token`:
            The copy; `tree` itself is left unchanged.

    Example:
        In `masked(tree, [literal.node])` for the literal `isinstance(v, Buy)` of
        `isinstance(v, Buy) and v.qty > 0`, the read `v.qty` is guarded by `NOTHING`.
    """
    ids = {id(node) for node in nodes}

    def copy(node: Tree | Token) -> Tree | Token:
        if id(node) in ids:
            return Tree(_MASKED, [node])
        if isinstance(node, Token):
            return node
        return Tree(node.data, [copy(child) for child in node.children])

    return copy(tree)


_NEVER_HAS = "never_has"
"""The data of a node `falsified` wraps a check in; its first child is the value the check
never has."""


def falsified(
    tree: Tree | Token, never: Collection[tuple[Tree | Token, bool]]
) -> Tree | Token:
    """Return a copy of a condition in which the given checks never have the given values.

    Each `(sub-tree, value)` pair (the sub-tree matched by identity, such as a
    `CheckLiteral.node`) wraps that check so `proves` gives `NEVER` for `value` and
    `NOTHING` for the other value, while `guarded_refs` still finds the same references in
    the same order. A reference whose guard is then `NEVER` is read only when one of those
    checks has its impossible value.

    Args:
        tree (`Tree | Token`):
            A condition parsed with `parse_condition`.
        never (`Collection[tuple[Tree | Token, bool]]`):
            Check sub-trees of `tree` and the value each never has.

    Returns:
        `Tree | Token`:
            The copy; `tree` itself is left unchanged.

    Example:
        In `falsified(tree, [(literal.node, True)])` for the literal `isinstance(v, Text)`
        of `isinstance(v, Text) and v.qty > 0`, the read `v.qty` is guarded by `NEVER`.
    """
    values = {id(node): value for node, value in never}

    def copy(node: Tree | Token) -> Tree | Token:
        if id(node) in values:
            return Tree(_NEVER_HAS, [Token("NEVER_VALUE", str(values[id(node)]).lower()), node])
        if isinstance(node, Token):
            return node
        return Tree(node.data, [copy(child) for child in node.children])

    return copy(tree)


def _quoted(text: str) -> str:
    """`text` as a string literal, in double quotes unless it holds one."""
    return f"'{text}'" if '"' in text else f'"{text}"'


def check_text(check: Check, value: bool = True) -> str:
    """Return the canonical source of a check that evaluates to `value` when true.

    Args:
        check (`Check`):
            The check.
        value (`bool`, *optional*, defaults to `True`):
            For a `TagCheck`, `False` spells the comparison with `!=`; for a
            `NotNullCheck`, with `==`. An `IsinstanceCheck` is always spelled positively.

    Returns:
        `str`:
            Such as `isinstance(d.output, Buy)`, `v.action != "buy"` or `v == null`.
    """
    if isinstance(check, IsinstanceCheck):
        return f"{ISINSTANCE}({check.base}, {check.type_name})"
    if isinstance(check, TagCheck):
        op = "==" if value else "!="
        return f"{check.base}.{check.field} {op} {_quoted(check.value)}"
    if isinstance(check, NotNullCheck):
        op = "!=" if value else "=="
        return f"{check.base} {op} null"
    raise unknown_check(check)


def literal_text(literal: CheckLiteral) -> str:
    """Return the canonical source of a literal.

    Args:
        literal (`CheckLiteral`):
            A literal from `literals_in_context`.

    Returns:
        `str`:
            The check's canonical source, after `not ` when the literal is negated:
            `not isinstance(v, Buy)`, `v.action != "buy"`, `not v.action == "buy"`.
    """
    positive = literal.value != literal.negated
    text = check_text(literal.check, positive)
    return f"not {text}" if literal.negated else text


def guard_checks(guard: Guard) -> tuple[Check, ...]:
    """Return every check in a guard.

    Args:
        guard (`Guard`):
            The guard.

    Returns:
        `tuple[Check, ...]`:
            The checks in first-seen order, without duplicates.
    """
    found: list[Check] = []

    def walk(node: Guard) -> None:
        if isinstance(node, Holds):
            if node.check not in found:
                found.append(node.check)
            return
        for part in node.parts:
            walk(part)

    walk(guard)
    return tuple(found)


def conjuncts(guard: Guard) -> tuple[Guard, ...]:
    """Return the parts of a top-level `AllOf`, else the guard itself.

    Args:
        guard (`Guard`):
            The guard.

    Returns:
        `tuple[Guard, ...]`:
            `guard.parts` for an `AllOf` (`()` for `NOTHING`), else `(guard,)`.
    """
    return guard.parts if isinstance(guard, AllOf) else (guard,)
