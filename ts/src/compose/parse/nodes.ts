/**
 * Graph-stage: parse the `nodes:` mapping into typed per-kind descriptors.
 *
 * The first cross-node assembly pass. Consumes a source cursor over the flow's `nodes:` mapping
 * and classifies each entry by its YAML tag: a tagged entry is a value
 * (`total: !int ${a.output + b.output}`) handed to the caller's `readValue` reader, and an
 * untagged mapping with `kind:` is a node body dispatched to its kind's `parse` via the `KINDS`
 * table. Any other entry is a located error that says to add `kind:` for a node or a tag for a
 * value.
 *
 * Only the generic, kind-independent pre-checks live here (node id shape and reserved ids, entry
 * classification, the `input:`/`output:` to plural back-map, `kind:` resolution); each kind's own
 * `parse` owns its field legality and its descriptor-local `SourceRange` recording (each field
 * keyed `siteKey("field", name)`). Two kind-blind enrichments run after every kind parse: each
 * `input:` binding's own range is stamped into the descriptor's `fieldRanges` under
 * `siteKey("input", name)`, and each `asserts:` item range is appended positionally to its
 * `assertRanges`. This metadata stays inside Compose; `_finish` copies node and field locations
 * to the `BuildResult` for validation.
 *
 * Knows about: diagnostics, compose source, helpers and node kinds, compile model and expr.
 * Never imports: the runtime, the run boundary, the CLI or provider SDKs.
 */

import { END_ID, START_ID } from "../../compile/model.ts";
import { compareCodeUnits, reprText } from "../../diagnostics.ts";
import { EXPRESSION_KEYWORDS } from "../../expr/grammar.ts";
import { isRecord } from "../../typesys/values.ts";
import { LoadDiagnostics } from "../helpers/errors.ts";
import { checkWrittenValue } from "../helpers/value_nesting.ts";
import { type NodeDescriptor, notTextError, siteKey } from "../nodes/base.ts";
import { KINDS, REMOVED_KINDS } from "../nodes/index.ts";
import { SourceAccessError, type SourceCursor } from "../source/index.ts";

// Singular source keys are projected onto the descriptors' plural field names while keeping
// their original child sources.
const INPUT_OUTPUT_REMAP: ReadonlyMap<string, string> = new Map([
  ["input", "inputs"],
  ["output", "outputs"],
]);

// Explicit ASCII classes, matching the expression `NAME` terminal: `${...}` can't reference a
// non-ASCII letter (`café`).
const IDENTIFIER = /^[A-Za-z_][A-Za-z0-9_]*$/;
const NON_IDENTIFIER_CHAR = /[^A-Za-z0-9_]/gu;

/** The identifier rule, as node id messages state it. */
export const IDENTIFIER_RULE = "ASCII letters, digits, and _, not starting with a digit";

/** Whether `name` is an identifier `${...}` can reference: text matching the ASCII rule. */
export function isIdentifier(name: unknown): boolean {
  return typeof name === "string" && IDENTIFIER.test(name);
}

/**
 * Spell `name` as an identifier, for a rename hint: each code point outside `[A-Za-z0-9_]`
 * becomes `_`, and a `_` is prefixed before a leading digit; `""` for an empty name.
 */
export function identifierSuggestion(name: string): string {
  const suggestion = name.replace(NON_IDENTIFIER_CHAR, "_");
  return /^[0-9]/.test(suggestion) ? `_${suggestion}` : suggestion;
}

/** The message for a `nodes:` entry that is neither a node body nor a tagged value. */
function nodeOrValue(nodeId: string): string {
  return (
    `node ${reprText(nodeId)}: an entry in \`nodes:\` is a node or a value; add \`kind:\` for a ` +
    "node, or a tag (`!object`, `!Buy`, ...) for a value"
  );
}

/** Reads one tagged `nodes:` entry, given its id and cursor, into its descriptor. */
export type ValueReader = (nodeId: string, cursor: SourceCursor) => NodeDescriptor;

/**
 * Parse one `nodes:` source cursor into typed per-kind descriptors.
 *
 * The source is physical for a full graph, source-aligned for compact or def projections, or
 * virtual for a programmatic or test mapping. A tagged entry is a value and is read by
 * `readValue`; every other entry must be a node body with `kind:`.
 *
 * @param nodesCursor Cursor over the `nodes:` mapping; a non-mapping value parses to no nodes.
 * @param options.readValue Reads one tagged entry into its descriptor; may be omitted only for a
 *   mapping with no tagged entry.
 * @returns The descriptors keyed by entry id, in source order, each anchored on its entry's key
 *   line whenever an authored source exists.
 * @throws LoadError at the entry for a bad or reserved id, an untagged entry that is not a node
 *   body, or a node body that fails its kind's parse; at the `kind:` value for a non-text or
 *   unknown kind.
 * @throws TypeError when a tagged entry is found and `readValue` is absent.
 */
export function parseNodes(
  nodesCursor: SourceCursor,
  options: { readValue?: ValueReader } = {},
): Map<string, NodeDescriptor> {
  const diagnostics = LoadDiagnostics.fromCursor(nodesCursor);
  const out = new Map<string, NodeDescriptor>();
  try {
    const nodes = nodesCursor._valueView();
    if (!isRecord(nodes)) return out;
    for (const nodeId of Object.keys(nodes)) {
      let desc = parseNode(nodeId, nodesCursor.child(nodeId), options.readValue);
      // Anchor the node-level range on the node's key, not its body: the anchor a cycle, an
      // unresolved ref, or a namespaced runtime failure boxes.
      const header = nodesCursor.keyRangeOf(nodeId);
      if (header !== null && desc.range !== null) desc = desc.replace({ range: header });
      out.set(nodeId, desc);
    }
  } catch (error) {
    if (!(error instanceof SourceAccessError)) throw error;
    throw diagnostics.error(error.message, { range: error.range, cause: error });
  }
  return out;
}

/**
 * Read one `nodes:` entry into its descriptor: a tagged entry goes to `readValue`; an untagged
 * one must be a node body with `kind:` and is dispatched to its kind's `parse`.
 */
function parseNode(
  nodeId: string,
  cursor: SourceCursor,
  readValue: ValueReader | undefined,
): NodeDescriptor {
  const diagnostics = LoadDiagnostics.fromCursor(cursor);
  rejectReservedNodeId(nodeId, cursor);
  // A tag makes the entry a value even when its mapping has a `kind:` key.
  if (cursor.tag !== null) {
    if (readValue === undefined) {
      throw new TypeError(
        `node ${reprText(nodeId)} is a tagged value entry; parse_nodes needs read_value to read it`,
      );
    }
    return readValue(nodeId, cursor);
  }
  if (!isRecord(cursor._valueView())) rejectUntaggedValue(nodeId, cursor, diagnostics);
  const normalized = normalizeNodeSignatureCursor(cursor);

  const kind = normalized.get("kind");
  if (kind === null) {
    throw diagnostics.error(nodeOrValue(nodeId), {
      range: normalized.range ?? normalized.keyRange,
    });
  }
  const allowed = [...KINDS.keys()].sort(compareCodeUnits).join(", ");
  if (typeof kind !== "string") {
    throw diagnostics.error(
      `node ${reprText(nodeId)}: ${notTextError("kind", kind, `write one of ${allowed}`)}`,
      { range: normalized.rangeOf("kind") },
    );
  }
  const removed = REMOVED_KINDS.get(kind);
  if (removed !== undefined) {
    throw diagnostics.error(
      `node ${reprText(nodeId)}: kind ${reprText(kind)} was removed; ${removed}`,
      { range: normalized.rangeOf("kind") },
    );
  }
  const cls = KINDS.get(kind);
  if (cls === undefined) {
    throw diagnostics.error(
      `node ${reprText(nodeId)}: unknown kind ${reprText(kind)} (allowed: ${allowed})`,
      { range: normalized.rangeOf("kind") },
    );
  }
  const desc = cls.parse(nodeId, normalized);
  // Compose-only enrichments, read from the normalized cursor (`input:` is `inputs` there):
  // each `input:` binding's own range, keyed apart from a like-named field, and each `asserts:`
  // item's range by position, since two identical asserts still need distinct carets. Both
  // no-op for a source-free cursor.
  for (const [name, range] of normalized.mappingValueRanges("inputs")) {
    desc.fieldRanges.set(siteKey("input", name), range);
  }
  desc.assertRanges.push(...normalized.itemRangeList("asserts"));
  return desc;
}

/**
 * Reject a node id that isn't a referenceable identifier or collides with a reserved name, at
 * the node body: an expression keyword, the synthesized boundary or a resolver head; `#`, `/`
 * and `.` get their own reserved-separator message.
 *
 * @throws LoadError for a bad id.
 */
function rejectReservedNodeId(nodeId: string, cursor: SourceCursor): void {
  const diagnostics = LoadDiagnostics.fromCursor(cursor);
  const range = cursor.range;
  const id = reprText(nodeId);
  if (nodeId.includes("#") || nodeId.includes("/") || nodeId.includes(".")) {
    throw diagnostics.error(
      `node id ${id} uses a reserved separator ('#' and '/' are reserved for runtime graph ` +
        "expansion; '.' is reserved for reference paths)",
      { range },
    );
  }
  if (!isIdentifier(nodeId)) {
    const reference = nodeId ? ` so \${${nodeId}.output} can reference it` : "";
    const suggestion = identifierSuggestion(nodeId);
    const hint = suggestion ? `; try ${reprText(suggestion)}` : "";
    throw diagnostics.error(
      `node id ${id} must be an identifier (${IDENTIFIER_RULE})${reference}${hint}`,
      { range },
    );
  }
  if (nodeId === START_ID || nodeId === END_ID) {
    throw diagnostics.error(
      `node id ${id} is reserved for the synthesized START_ID/END_ID boundary; authors write ` +
        "`input:`/`output:`, not a boundary node",
      { range },
    );
  }
  if (EXPRESSION_KEYWORDS.has(nodeId)) {
    throw diagnostics.error(
      `node id ${id} is an expression keyword, so \${${nodeId}.output} can't reference it; ` +
        "pick a different node id",
      { range },
    );
  }
  // The four singular head literals keep the node-first ref `${<node>.output.k}` unambiguous;
  // the plural forms catch migration typos.
  if (["input", "output", "system", "item"].includes(nodeId)) {
    throw diagnostics.error(
      `node id ${id} is reserved (used as \`\${${nodeId}.X}\` as a resolver head — pick a ` +
        "different node id)",
      { range },
    );
  }
  if (nodeId === "inputs" || nodeId === "outputs") {
    throw diagnostics.error(
      `node id ${id} is reserved (the plural head is retired — rename to ` +
        `\`${nodeId.slice(0, -1)}\` or pick a different node id)`,
      { range },
    );
  }
}

/**
 * Reject an untagged scalar, list or null entry with the node-or-value message, at the entry.
 *
 * @throws LoadError always; one nested deeper than a yian value may reports the nesting limit
 *   instead.
 */
function rejectUntaggedValue(
  nodeId: string,
  cursor: SourceCursor,
  diagnostics: LoadDiagnostics,
): never {
  const range = cursor.range ?? cursor.keyRange;
  // First: a value past the nesting limit couldn't load as a tagged entry either.
  checkWrittenValue(cursor.value, `value entry ${reprText(nodeId)}`, range, diagnostics);
  throw diagnostics.error(nodeOrValue(nodeId), { range });
}

/**
 * Project authored node-signature keys onto descriptor field names: `input:` to `inputs` and
 * `output:` to `outputs`, keeping each child's source.
 *
 * The projection is the single crossing from the singular Compose surface to the plural carrier
 * fields; later phases that correlate a descriptor with its source reuse this cursor so collision
 * and source-selection rules can't drift from parsing.
 *
 * @throws SourceAccessError when `cursor` is not a traversable mapping, or when both spellings of
 *   one key are written.
 */
export function normalizeNodeSignatureCursor(cursor: SourceCursor): SourceCursor {
  return cursor.renamed(INPUT_OUTPUT_REMAP);
}
