"""Lower a tagged `nodes:` entry (a value entry) to a CODE node with the entry's id.

`total: !int ${a.output + b.output}` and `exit_result: !object {exited: true}` are
values, not nodes. Each lowers at load to a `ValueEntryDescriptor`: a CODE node whose
one input, named after the entry, holds the entry's value typed by its tag, and whose
code returns that input. The typed-binding machinery therefore checks the value at
load and at runtime, wires its edges, and locates its errors at the entry, as for any
tagged node input. The tag only checks the value; it never fills in a field.

The node's output contract is its tag, except that a bare `!object` over a written-out
mapping keeps the record the mapping spells, so reading a field it doesn't have is a
load error. `!object ${x.output}`, a value from elsewhere, stays `object`.

Two values are load errors at the entry: a number that is not finite, which no run can
hold, and an empty key in the record a bare `!object` keeps, since no record field can
be named "" and dropping the record would quietly stop checking its reads.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from yian.compose.helpers.errors import LoadDiagnostics
from yian.compose.helpers.output_contracts import literal_contract
from yian.compose.helpers.tags import path_label, read_tagged_value, suggest_tag
from yian.compose.nodes.base import BuildContext
from yian.compose.nodes.code import CodeDescriptor
from yian.compose.source import SourceCursor
from yian.nodes.contracts import OutputContract, ValueContract, ValuePath, ValueTemplate
from yian.typesys.types import TypeRegistry
from yian.typesys.values import Type, ValueKind, shown_text

if TYPE_CHECKING:
    from yian.compose.source.catalog import _SourceCatalogBuilder

# Why a written-out `!object` can't hold a mapping with an empty key.
EMPTY_RECORD_KEY = (
    "a record key can't be empty; a written-out `!object` keeps its keys as record "
    "fields"
)


def non_finite_text(number: float) -> str:
    """Spell a number that is not finite as YAML does: `.inf`, `-.inf`, or `.nan`."""
    if math.isnan(number):
        return ".nan"
    return ".inf" if number > 0 else "-.inf"


def non_finite_reason(number: float, text: str | None = None) -> str:
    """Return why a number that is not finite can't be a value.

    Args:
        number (`float`): `inf`, `-inf`, or `nan`.
        text (`str | None`, *optional*, defaults to `None`): The authored spelling,
            shown (cut to `YIAN_MAX_SHOWN_VALUE_CHARS`) with the spelling YAML reads
            when the two differ.

    Returns:
        `str`: Such as `numbers must be finite (got .inf)`, or with a differing
            `text`, `numbers must be finite (got 1e400, which YAML reads as .inf)`.
    """
    canonical = non_finite_text(number)
    if text is None or text == canonical:
        return f"numbers must be finite (got {canonical})"
    return (
        f"numbers must be finite (got {shown_text(text)}, which YAML reads as "
        f"{canonical})"
    )


def infers_record(tag: Type, value: Any) -> bool:
    """Report whether a value entry keeps the record its written-out mapping spells.

    Args:
        tag (`Type`): The entry's tag type.
        value (`Any`): The entry's value, unwrapped from any `ValueTemplate`.

    Returns:
        `bool`: True when the tag is a bare, non-optional `object` and the value is a
            mapping.
    """
    return (
        tag.kind is ValueKind.OBJECT
        and tag.fields is None
        and not tag.nullable
        and type(value) is dict
    )


@dataclass(frozen=True, kw_only=True)
class ValueEntryDescriptor(CodeDescriptor):
    """A value entry lowered to a CODE node returning its one typed input.

    Attributes:
        tag (`ValueContract`): The entry's tag as a contract; the node's input type.
    """

    tag: ValueContract

    def __post_init__(self) -> None:
        """Reject node-only fields, which a value entry can't author.

        Raises:
            `ValueError`: The descriptor carries `asserts`, `depends_on`,
                `runs_after`, or `env`.
        """
        super().__post_init__()
        for name in ("asserts", "depends_on", "runs_after", "env"):
            if getattr(self, name):
                raise ValueError(f"value entry {self.id!r} can't carry {name!r}")

    def output_contract(self, ctx: BuildContext) -> OutputContract:
        """Return the node's output contract: the tag, or the written-out record.

        A bare `!object` over a written-out mapping keeps the record its value
        spells, so a read of a field it lacks fails at load: a literal field has its
        literal's type, a `${...}` field its producer's (opaque when unproven), an
        untagged nested mapping is inferred the same way, and a nested tag keeps its
        declared type. A nested untagged mapping with an unproven field, or a list
        whose items don't join, is opaque as a whole, since a partial record can't
        nest. Any other entry's contract is its tag.

        Args:
            ctx (`BuildContext`): Supplies `binding_contract_reader`, which types a
                value template (a value with a `${...}` span or a nested tag) from
                its producers.

        Returns:
            `OutputContract`: The inferred exact or partial record, or the tag's
                contract when the value isn't inferred or can't be typed (no reader,
                or a programmatic mapping with a non-text key).

        Raises:
            `LoadError`: A span reads the entry's own output, directly or through
                other inferred `!object` entries (a node build dependency cycle).
        """
        if not self._infers_record():
            return self.tag
        value = self.inputs[self.id]
        if isinstance(value, ValueTemplate):
            if ctx.binding_contract_reader is None:
                return self.tag
            contract = ctx.binding_contract_reader(
                self.id, value, self.field_ranges.get(("input", self.id))
            )
        else:
            contract = literal_contract(value)
        return self.tag if contract is None else contract

    def _infers_record(self) -> bool:
        """Report whether the tag is a bare `object` over a written-out mapping."""
        value = self.inputs[self.id]
        if isinstance(value, ValueTemplate):
            value = value.value
        return infers_record(self.tag.type, value)


def read_value_entry(
    node_id: str,
    cursor: SourceCursor,
    *,
    registry: TypeRegistry,
    owner: tuple[str, ...],
    catalog_builder: "_SourceCatalogBuilder",
    diagnostics: LoadDiagnostics,
) -> ValueEntryDescriptor:
    """Read one tagged `nodes:` entry into its CODE node descriptor.

    The value is read like a tagged node input named after the entry: a span-free
    literal is checked against its tag now, nested tags become typed bindings, and
    a value holding a `${...}` span is kept as a template the binder resolves. An
    error raised while the tag and value are read names the entry `'<id>':` (or
    `'<id>.<path>':` for a nested value); a tag with nothing after it names it
    `value entry '<id>'`, as later checks of the whole value do.

    Args:
        node_id (`str`): The entry id, also the node id and its input's name.
        cursor (`SourceCursor`): The entry's tagged value.
        registry (`TypeRegistry`): The flow's typedefs the tags resolve against.
        owner (`tuple[str, ...]`): Complete lexical owner path for catalog sites.
        catalog_builder (`_SourceCatalogBuilder`): File-owned origin catalog builder,
            whose `consumed_tags` ledger records each tag read.
        diagnostics (`LoadDiagnostics`): Factory for located load errors.

    Returns:
        `ValueEntryDescriptor`: The node, with `inputs = {node_id: value}` typed by
            the tag and code returning that input.

    Raises:
        `LoadError`: A tag is malformed or names no type, a space splits a tag from
            its arguments, nothing follows the entry's tag or a nested tag, a
            template has a non-string record key, a nested string starts with
            `call(`, a span-free tagged literal does not fit its tag, the value
            can't be walked (a recursive alias), a number in it is not finite, or
            the record a bare `!object` keeps has an empty key.
    """
    slot = read_tagged_value(
        cursor,
        registry,
        label=node_id,
        site_path=(*owner, "node", node_id, "input", node_id),
        catalog_builder=catalog_builder,
        diagnostics=diagnostics,
        ledger=catalog_builder.consumed_tags,
    )
    assert slot.contract is not None
    if not slot.has_value:
        spelled = suggest_tag(
            node_id, slot.contract.type, "...", registry=registry, type_text=cursor.tag
        )
        raise diagnostics.error(
            f"value entry {node_id!r} needs a value after its tag, e.g. {spelled}",
            range=cursor.range or cursor.key_range,
        )
    _reject_unholdable(
        node_id,
        cursor,
        record=infers_record(slot.contract.type, cursor.value),
        diagnostics=diagnostics,
    )
    value_range = cursor.range
    field_ranges = (
        {("input", node_id): value_range, ("field", "inputs"): value_range}
        if value_range is not None
        else {}
    )
    return ValueEntryDescriptor(
        id=node_id,
        node_name=node_id,
        code=f"return inputs[{node_id!r}]",
        inputs={node_id: slot.value},
        input_types={node_id: slot.contract},
        tag=slot.contract,
        range=cursor.key_range or value_range,
        field_ranges=field_ranges,
        field_labels={f"input {node_id!r}": f"value entry {node_id!r}"},
    )


def _reject_unholdable(
    node_id: str,
    cursor: SourceCursor,
    *,
    record: bool,
    diagnostics: LoadDiagnostics,
    path: ValuePath = (),
) -> None:
    """Reject a number that is not finite, or an empty key in a kept record.

    Args:
        node_id (`str`): The entry id.
        cursor (`SourceCursor`): The entry's value, or a part of it, which does not
            contain itself.
        record (`bool`): Whether a mapping here is kept as a record: the written-out
            mapping of a bare `!object` entry, or a mapping or list inside it outside
            any nested tag.
        diagnostics (`LoadDiagnostics`): Factory for located load errors.
        path (`ValuePath`, *optional*, defaults to `()`): Where `cursor` sits in the
            entry's value.

    Raises:
        `LoadError`: At the number or the mapping, named `'<id>'` or
            `'<id>.<path>'`.
    """
    value = cursor.value
    if isinstance(value, float) and not math.isfinite(value):
        raise diagnostics.error(
            f"{path_label(node_id, path)!r}: {non_finite_reason(value)}",
            range=cursor.range,
        )
    if type(value) is dict:
        if record and "" in value:
            raise diagnostics.error(
                f"{path_label(node_id, path)!r}: {EMPTY_RECORD_KEY}", range=cursor.range
            )
        keys: list[Any] = list(value)
    elif type(value) is list:
        keys = list(range(len(value)))
    else:
        return
    for key in keys:
        child = cursor.child(key)
        _reject_unholdable(
            node_id,
            child,
            record=record and child.tag is None,
            diagnostics=diagnostics,
            path=(*path, key),
        )
