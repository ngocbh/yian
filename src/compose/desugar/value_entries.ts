/**
 * Lower a tagged `nodes:` entry (a value entry) to a CODE node with the entry's id.
 *
 * `total: !int ${a.output + b.output}` and `exit_result: !object {exited: true}` are values, not
 * nodes. Each lowers at load to a `ValueEntryDescriptor`: a CODE node whose one input, named after
 * the entry, holds the entry's value typed by its tag, and whose code is the built-in identity
 * `{builtin: "identity", field: <id>}`, which returns that input. The typed-binding machinery
 * therefore checks the value at load and at runtime, wires its edges, and locates its errors at
 * the entry, as for any tagged node input. The tag only checks the value; it never fills in a
 * field.
 *
 * The node's output contract is its tag, except that a bare `!object` over a written-out mapping
 * keeps the record the mapping spells, so reading a field it doesn't have is a load error.
 * `!object ${x.output}`, a value from elsewhere, stays `object`.
 *
 * An empty key in the record a bare `!object` keeps is a load error at the entry, since no record
 * field can be named "" and dropping the record would quietly stop checking its reads. A number
 * that is not finite needs no check here: infinity is a value, and the YAML reader refuses `.nan`
 * and a float too large to read.
 *
 * Knows about: diagnostics, the compose helpers (errors, output contracts, tags), the CODE
 * descriptor, the compose source, node contracts, and typesys.
 */

import { reprText } from "../../diagnostics.ts";
import { type OutputContract, type ValueContract, type ValuePath, ValueTemplate } from "../../nodes/contracts.ts";
import { YianRangeError } from "../../typesys/errors.ts";
import type { TypeRegistry } from "../../typesys/types.ts";
import { isList, isRecord, type Type, ValueKind } from "../../typesys/values.ts";
import type { LoadDiagnostics } from "../helpers/errors.ts";
import { literalContract } from "../../compile/output_contracts.ts";
import { pathLabel, readTaggedValue, suggestTag } from "../helpers/tags.ts";
import { type BuildContext, type NodeDescriptor, siteKey } from "../nodes/base.ts";
import { CodeDescriptor, type CodeDescriptorFields } from "../nodes/code.ts";
import type { SourceCatalogBuilder } from "../source/catalog.ts";
import type { SourceCursor } from "../source/index.ts";

/** Why a written-out `!object` can't hold a mapping with an empty key. */
const EMPTY_RECORD_KEY =
  "a record key can't be empty; a written-out `!object` keeps its keys as record fields";

/**
 * Report whether a value entry keeps the record its written-out mapping spells.
 *
 * @param tag The entry's tag type.
 * @param value The entry's value, unwrapped from any `ValueTemplate`.
 * @returns True when the tag is a bare, non-optional `object` and the value is a record.
 */
function infersRecord(tag: Type, value: unknown): boolean {
  return tag.kind === ValueKind.OBJECT && tag.fields === undefined && !tag.nullable && isRecord(value);
}

/** The fields `ValueEntryDescriptor`'s constructor takes: the CODE's and the entry's tag. */
export interface ValueEntryDescriptorFields extends CodeDescriptorFields {
  tag: ValueContract;
}

/** A value entry lowered to a CODE node returning its one typed input. */
export class ValueEntryDescriptor extends CodeDescriptor {
  /** The entry's tag as a contract; the node's input type. */
  readonly tag: ValueContract;

  /**
   * Create a descriptor from `fields`, rejecting node-only fields, which a value entry can't
   * author.
   *
   * @throws RangeError when the descriptor carries `asserts`, `dependsOn`, `runsAfter`, or `env`.
   */
  constructor(fields: ValueEntryDescriptorFields) {
    super(fields);
    this.tag = fields.tag;
    // Keyed by the authored names, which the message shows.
    const carried = {
      asserts: this.asserts.length,
      depends_on: this.dependsOn.length,
      runs_after: this.runsAfter.length,
      env: Object.keys(this.env).length,
    };
    for (const [name, count] of Object.entries(carried)) {
      if (count > 0) throw new YianRangeError(`value entry ${reprText(this.id)} can't carry ${reprText(name)}`);
    }
  }

  /**
   * Return the node's output contract: the tag, or the written-out record.
   *
   * A bare `!object` over a written-out mapping keeps the record its value spells, so a read of a
   * field it lacks fails at load: a literal field has its literal's type, a `${...}` field its
   * producer's (opaque when unproven), an untagged nested mapping is inferred the same way, and a
   * nested tag keeps its declared type. A nested untagged mapping with an unproven field, or a list
   * whose items don't join, is opaque as a whole, since a partial record can't nest. Any other
   * entry's contract is its tag.
   *
   * @param ctx Supplies `bindingContractReader`, which types a value template (a value with a
   *   `${...}` span or a nested tag) from its producers.
   * @returns The inferred exact or partial record, or the tag's contract when the value isn't
   *   inferred or can't be typed (no reader).
   * @throws LoadError when a span reads the entry's own output, directly or through other inferred
   *   `!object` entries (a node build dependency cycle).
   */
  override outputContract(ctx: BuildContext): OutputContract {
    const value = this.inputs[this.id];
    if (!infersRecord(this.tag.type, value instanceof ValueTemplate ? value.value : value)) return this.tag;
    let contract: OutputContract | null;
    if (value instanceof ValueTemplate) {
      if (ctx.bindingContractReader === null) return this.tag;
      contract = ctx.bindingContractReader(this.id, value, this.fieldRanges.get(siteKey("input", this.id)) ?? null);
    } else {
      contract = literalContract(value);
    }
    return contract ?? this.tag;
  }
}

/**
 * Read one tagged `nodes:` entry into its CODE node descriptor.
 *
 * The value is read like a tagged node input named after the entry: a span-free literal is
 * checked against its tag now, nested tags become typed bindings, and a value holding a `${...}`
 * span is kept as a template the binder resolves. An error raised while the tag and value are read
 * names the entry `'<id>':` (or `'<id>.<path>':` for a nested value); a tag with nothing after it
 * names it `value entry '<id>'`, as later checks of the whole value do.
 *
 * @param nodeId The entry id, also the node id and its input's name.
 * @param cursor The entry's tagged value.
 * @param options.registry The flow's typedefs the tags resolve against.
 * @param options.owner Complete lexical owner path for catalog sites.
 * @param options.catalogBuilder File-owned origin catalog builder, whose `consumedTags` ledger
 *   records each tag read.
 * @param options.diagnostics Factory for located load errors.
 * @returns The node, with `inputs = {[nodeId]: value}` typed by the tag and the built-in identity
 *   code returning that input.
 * @throws LoadError when a tag is malformed or names no type, a space splits a tag from its
 *   arguments, nothing follows the entry's tag or a nested tag, a nested string starts with
 *   `call(`, a span-free tagged literal does not fit its tag, the value can't be walked (a
 *   recursive alias), or the record a bare `!object` keeps has an empty key.
 */
export function readValueEntry(
  nodeId: string,
  cursor: SourceCursor,
  options: {
    registry: TypeRegistry;
    owner: readonly string[];
    catalogBuilder: SourceCatalogBuilder;
    diagnostics: LoadDiagnostics;
  },
): ValueEntryDescriptor {
  const { registry, owner, catalogBuilder, diagnostics } = options;
  const slot = readTaggedValue(cursor, registry, {
    label: nodeId,
    sitePath: [...owner, "node", nodeId, "input", nodeId],
    catalogBuilder,
    diagnostics,
    ledger: catalogBuilder.consumedTags,
  });
  // The entry is tagged, so the slot always has its contract.
  const contract = slot.contract!;
  if (!slot.hasValue) {
    const spelled = suggestTag(nodeId, contract.type, "...", { registry, typeText: cursor.tag });
    throw diagnostics.error(`value entry ${reprText(nodeId)} needs a value after its tag, e.g. ${spelled}`, {
      range: cursor.range ?? cursor.keyRange,
    });
  }
  if (infersRecord(contract.type, cursor.value)) rejectEmptyRecordKeys(nodeId, cursor, { diagnostics });
  const valueRange = cursor.range;
  const fieldRanges =
    valueRange !== null
      ? new Map([
          [siteKey("input", nodeId), valueRange],
          [siteKey("field", "inputs"), valueRange],
        ])
      : new Map();
  return new ValueEntryDescriptor({
    id: nodeId,
    nodeName: nodeId,
    code: { builtin: "identity", field: nodeId },
    inputs: { [nodeId]: slot.value },
    inputTypes: new Map([[nodeId, contract]]),
    tag: contract,
    range: cursor.keyRange ?? valueRange,
    fieldRanges,
    fieldLabels: new Map([[`input ${reprText(nodeId)}`, `value entry ${reprText(nodeId)}`]]),
  });
}

/**
 * Return whether `descriptor` is a value entry.
 *
 * @param descriptor Any node descriptor.
 */
export function isValueEntry(descriptor: NodeDescriptor): descriptor is ValueEntryDescriptor {
  return descriptor instanceof ValueEntryDescriptor;
}

/**
 * Reject an empty key in a kept record.
 *
 * @param nodeId The entry id.
 * @param cursor The kept record, or a part of it outside any nested tag, which does not contain
 *   itself.
 * @param options.diagnostics Factory for located load errors.
 * @param options.path Where `cursor` sits in the entry's value; defaults to the value itself.
 * @throws LoadError at the mapping, named `'<id>'` or `'<id>.<path>'`.
 */
function rejectEmptyRecordKeys(
  nodeId: string,
  cursor: SourceCursor,
  options: { diagnostics: LoadDiagnostics; path?: ValuePath },
): void {
  const { diagnostics, path = [] } = options;
  const value = cursor.value;
  let keys: (string | number)[];
  if (isRecord(value)) {
    if (Object.hasOwn(value, "")) {
      throw diagnostics.error(`${reprText(pathLabel(nodeId, path))}: ${EMPTY_RECORD_KEY}`, { range: cursor.range });
    }
    keys = Object.keys(value);
  } else if (isList(value)) {
    keys = value.map((_item, index) => index);
  } else {
    return;
  }
  for (const key of keys) {
    const child = cursor.child(key);
    if (child.tag === null) rejectEmptyRecordKeys(nodeId, child, { diagnostics, path: [...path, key] });
  }
}
