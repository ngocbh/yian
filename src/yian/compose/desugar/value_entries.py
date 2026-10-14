"""Lower a tagged `nodes:` entry (a value entry) to a CODE node with the entry's id.

`total: !int ${a.output + b.output}` and `exit_result: !object {exited: true}` are
values, not nodes. Each lowers at load to a `ValueEntryDescriptor`: a CODE node whose
one input, named after the entry, holds the entry's value typed by its tag, and whose
code returns that input. The typed-binding machinery therefore checks the value at
load and at runtime, wires its edges, and locates its errors at the entry, as for any
tagged node input. The tag only checks the value; it never fills in a field.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from yian.compose.helpers.errors import LoadDiagnostics
from yian.compose.helpers.tags import read_tagged_value, suggest_tag
from yian.compose.nodes.base import BuildContext
from yian.compose.nodes.code import CodeDescriptor
from yian.compose.source import SourceCursor
from yian.nodes.contracts import ValueContract
from yian.typesys.types import TypeRegistry

if TYPE_CHECKING:
    from yian.compose.source.catalog import _SourceCatalogBuilder


@dataclass(frozen=True, kw_only=True)
class ValueEntryDescriptor(CodeDescriptor):
    """A value entry lowered to a CODE node returning its one typed input.

    Attributes:
        tag (`ValueContract`): The entry's tag as a contract; the node's input type
            and its output contract.
    """

    tag: ValueContract

    def __post_init__(self) -> None:
        """Reject node-only fields, which a value entry can't author."""
        super().__post_init__()
        for name in ("asserts", "depends_on", "runs_after", "env"):
            if getattr(self, name):
                raise ValueError(f"value entry {self.id!r} can't carry {name!r}")

    def with_source_origins(
        self,
        cursor: SourceCursor,
        *,
        site_path: tuple[str | int, ...],
        catalog_builder: "_SourceCatalogBuilder",
    ) -> "ValueEntryDescriptor":
        """Return this descriptor unchanged: a value entry has no `code:` field to locate.

        Args:
            cursor (`SourceCursor`): The entry's value.
            site_path (`tuple[str | int, ...]`): Complete semantic node path.
            catalog_builder (`_SourceCatalogBuilder`): File-owned origin registry.

        Returns:
            `ValueEntryDescriptor`: This descriptor.
        """
        return self

    def output_contract(self, ctx: BuildContext) -> Optional[ValueContract]:
        """Return the entry's tag as the node's output contract.

        Args:
            ctx (`BuildContext`): Unused; a value entry's type is its tag.

        Returns:
            `ValueContract | None`: The tag's contract.
        """
        return self.tag


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
    a value holding a `${...}` span is kept as a template the binder resolves. Load
    messages about the whole value name it `value entry '<id>'`.

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
        `LoadError`: The tag is malformed or names no type, nothing follows it, or a
            tagged literal does not fit its tag.
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
