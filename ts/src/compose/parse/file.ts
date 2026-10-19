/**
 * The section parser: one owned source document to a strict `ComposeFile`.
 *
 * It inspects the document's root cursor, copies its semantic value into a `ComposeFile`, and
 * throws a located `LoadError` on any violation. Four surface rules shape it:
 *
 * - **Top-level sections only** (`id`/`name`/`description?`/`version?`/`input`/`nodes`/`output`/
 *   `asserts`/`typedefs`/`defs`/`uses`/`system`/`llm_config`/`env`/`long_term_memory`): a typo'd
 *   top-level key (`nodez:`) is loud rather than silently dropped. The schema is checked by hand,
 *   field by field in this order, and the first located failure throws; a missing required key
 *   throws only when no other field fails.
 * - **`x-*` extension keys** at the top level are stripped first: custom metadata and YAML-anchor
 *   holders the engine ignores.
 * - **YAML anchors** (`&` / `*` / `<<:` merge) are expanded by the document's one construction
 *   pass before the schema sees them.
 * - **Every mapping key is text**, anywhere in the file (`x-*` blocks included): a key YAML reads
 *   as a boolean, number or null fails at the key first.
 *
 * A compact flow (a top-level node `kind:` and no `nodes:`) is desugared into the one-node form
 * first. `nodes` stays a raw mapping here; `compose/parse/nodes.ts` builds the descriptors.
 *
 * Knows about: diagnostics, compose source, helpers and node kinds, typesys values.
 * Never imports: the runtime, the run boundary, the CLI or provider SDKs.
 */

import { compareCodeUnits, reprText, type SourceRange } from "../../diagnostics.ts";
import { isList, isRecord, type Value, type ValueRecord } from "../../typesys/values.ts";
import { LoadDiagnostics } from "../helpers/errors.ts";
import { isExtensionKey } from "../helpers/source.ts";
import { yamlReadAs } from "../helpers/yaml_words.ts";
import { notTextError, valueTypeError } from "../nodes/base.ts";
import { REMOVED_KINDS } from "../nodes/index.ts";
import {
  SourceAccessError,
  SourceCursor,
  SourceDocument,
  SourceSyntaxError,
} from "../source/index.ts";
import { IDENTIFIER_RULE, identifierSuggestion, isIdentifier } from "./nodes.ts";

/** The retired plural section keys, each with its replacement. */
const LEGACY_KEYS = [
  ["inputs", "input"],
  ["outputs", "output"],
] as const;

/** The sections holding node or def bodies: the key, its entry noun, and its plural noun. */
const NESTED_SECTIONS = [
  ["nodes", "node", "nodes"],
  ["defs", "defs entry", "defs"],
] as const;

/** The authored spellings of the top-level keys. */
const TOP_SURFACE_KEYS: readonly string[] = [
  "id",
  "name",
  "description",
  "version",
  "input",
  "nodes",
  "output",
  "asserts",
  "typedefs",
  "defs",
  "uses",
  "system",
  "llm_config",
  "env",
  "long_term_memory",
];

/**
 * The top-level keys a flow may hold. The plural carrier names are absent, since
 * `rejectLegacyTopKeys` rejects them before this check.
 */
const ALLOWED_TOP: ReadonlySet<string> = new Set(TOP_SURFACE_KEYS);

/** The allowed top-level keys as an unknown-key message lists them. */
const ALLOWED_TOP_MESSAGE = [...TOP_SURFACE_KEYS].sort(compareCodeUnits).join(", ");

/**
 * The flow-metadata keys a compact flow keeps at the flow level on desugar; every other key (the
 * node `kind:` and its logic fields) becomes the single node's body. `input:` and `output:` get
 * special treatment (see `desugarCompact`).
 */
const COMPACT_FLOW_KEYS: readonly string[] = ["id", "name", "description", "version", "typedefs"];

/** The kinds a compact flow may use: the value-producing leaf kinds. */
const COMPACT_KINDS: ReadonlySet<string> = new Set([
  // case/call/map reference other nodes or callables that a one-node flow has none of, so an
  // author who needs them writes `nodes:`.
  "agent",
  "code",
  "tool",
  "human_input",
]);

/** The fields of a `ComposeFile`; an absent optional field takes its default. */
interface ComposeFileFields {
  id: string;
  name: string;
  description?: string | null;
  version?: string | null;
  inputs?: ValueRecord;
  nodes: ValueRecord;
  outputs?: Value;
  asserts?: string[];
  typedefs?: ValueRecord;
  defs?: ValueRecord;
  uses?: Record<string, string>;
  system?: ValueRecord;
  llmConfig?: Value;
  env?: ValueRecord;
  longTermMemory?: Value;
}

/**
 * A parsed Compose flow: its top-level sections, before node descriptors.
 *
 * `inputs`, `nodes`, `typedefs` and `defs` stay raw mappings and `outputs` a raw value (a
 * `${...}` binding or a name-to-binding mapping); later phases read each section into its typed
 * form. `defs` holds in-file callables, each an authored child flow
 * `{input?, nodes, output?, asserts?}`. The authored `input:`/`output:` keys are carried as
 * `inputs`/`outputs`, at the top level and inside node and def bodies.
 */
export class ComposeFile {
  readonly id: string;
  readonly name: string;
  readonly description: string | null;
  /** Opaque version tag (`v1`, `1.2.0`, ...); null for an unversioned flow. */
  readonly version: string | null;
  readonly inputs: ValueRecord;
  readonly nodes: ValueRecord;
  readonly outputs: Value;
  readonly asserts: string[];
  readonly typedefs: ValueRecord;
  readonly defs: ValueRecord;
  readonly uses: Record<string, string>;
  readonly system: ValueRecord;
  /**
   * Flow-level model-selection defaults, the cascade's flow layer; any shape, since a whole
   * record may be one `${...}`, and the loader checks it.
   */
  readonly llmConfig: Value;
  /**
   * Flow-level static per-node config defaults, merged under each node's own `env:` in this
   * flow; a called child flow doesn't inherit it.
   */
  readonly env: ValueRecord;
  /** The raw `long_term_memory:` occurrence, for the shared located policy parser. */
  readonly longTermMemory: Value;

  /** Create a flow from `fields`, defaulting the absent optional ones. */
  constructor(fields: ComposeFileFields) {
    this.id = fields.id;
    this.name = fields.name;
    this.description = fields.description ?? null;
    this.version = fields.version ?? null;
    this.inputs = fields.inputs ?? {};
    this.nodes = fields.nodes;
    this.outputs = fields.outputs ?? null;
    this.asserts = fields.asserts ?? [];
    this.typedefs = fields.typedefs ?? {};
    this.defs = fields.defs ?? {};
    this.uses = fields.uses ?? {};
    this.system = fields.system ?? {};
    // An authored `llm_config: null` stays null; only an absent one defaults.
    this.llmConfig = fields.llmConfig === undefined ? {} : fields.llmConfig;
    this.env = fields.env ?? {};
    this.longTermMemory = fields.longTermMemory ?? null;
  }
}

/**
 * Reject the retired top-level spellings `inputs:`, `outputs:` and `memory_policy:`, naming the
 * replacement, at the key.
 *
 * Node and def bodies are checked by `rejectNestedLegacyKeys`.
 */
function rejectLegacyTopKeys(
  body: ValueRecord,
  root: SourceCursor,
  diagnostics: LoadDiagnostics,
): void {
  for (const [legacy, replacement] of LEGACY_KEYS) {
    if (Object.hasOwn(body, legacy)) {
      throw diagnostics.error(
        `unknown top-level key ${reprText(legacy)} — flows use \`${replacement}:\` ` +
          "(rename the section)",
        { range: root.keyRangeOf(legacy) },
      );
    }
  }
  if (Object.hasOwn(body, "memory_policy")) {
    throw diagnostics.error(
      "unknown top-level key 'memory_policy' — `memory_policy:` was renamed to " +
        "`long_term_memory:`",
      { range: root.keyRangeOf("memory_policy") },
    );
  }
}

/**
 * Reject the retired `inputs:`/`outputs:` in every authored node and def body, at the first one.
 *
 * Walks the same `nodes:`/`defs:` nesting that `backMapToPlural` projects. The check reads the
 * authored source because the projection would merge `inputs:` with a renamed `input:` and keep
 * only one of them. A tagged `nodes:` entry is a value, not a node body, so it is skipped.
 *
 * @throws LoadError at the first `inputs:` or `outputs:` key in a nested body.
 * @throws SourceAccessError when a nested section can't be traversed.
 */
function rejectNestedLegacyKeys(cursor: SourceCursor, diagnostics: LoadDiagnostics): void {
  for (const [section, noun, pluralNoun] of NESTED_SECTIONS) {
    if (!cursor.has(section)) continue;
    const sectionCursor = cursor.child(section);
    const entries = sectionCursor._valueView();
    if (!isRecord(entries)) continue;
    for (const [name, entry] of Object.entries(entries)) {
      if (!isRecord(entry)) continue;
      const entryCursor = sectionCursor.child(name);
      // A tagged `nodes:` entry is a value, whose keys mean nothing to yian.
      if (section === "nodes" && entryCursor.tag !== null) continue;
      for (const [legacy, replacement] of LEGACY_KEYS) {
        if (Object.hasOwn(entry, legacy)) {
          throw diagnostics.error(
            `${noun} ${reprText(name)}: unknown key ${reprText(legacy)} — ${pluralNoun} use ` +
              `\`${replacement}:\` (rename the key)`,
            { range: entryCursor.keyRangeOf(legacy) },
          );
        }
      }
      rejectNestedLegacyKeys(entryCursor, diagnostics);
    }
  }
}

/**
 * Return `body` with its `input`/`output` keys renamed to the carrier names `inputs`/`outputs`,
 * recursively through `nodes:` and `defs:` entries; a non-mapping is returned as it is.
 *
 * @param active The records on the current path; a recursive alias back to one of them is kept
 *   as it is, so the walk ends. A tagged `nodes:` entry can alias back to itself, since
 *   `rejectNestedLegacyKeys` skips it and so never fails at the alias; Python recursed here until
 *   its generic `except` turned the overflow into an unlocated error, and TS has no such `except`.
 */
function backMapToPlural(body: Value, active: Set<ValueRecord> = new Set()): Value {
  if (!isRecord(body) || active.has(body)) return body;
  active.add(body);
  // A spread defines a `__proto__` key as data.
  const out: ValueRecord = { ...body };
  for (const [plural, singular] of LEGACY_KEYS) {
    if (Object.hasOwn(out, singular)) {
      const value = out[singular]!;
      delete out[singular];
      out[plural] = value;
    }
  }
  for (const [section] of NESTED_SECTIONS) {
    const entries = Object.hasOwn(out, section) ? out[section] : null;
    if (isRecord(entries)) {
      out[section] = Object.fromEntries(
        Object.entries(entries).map(([key, entry]) => [key, backMapToPlural(entry, active)]),
      );
    }
  }
  active.delete(body);
  return out;
}

/**
 * Whether `value` is null, false, zero, empty text or an empty container: a compact `input:` that
 * declares no params.
 */
function isEmptyValue(value: Value): boolean {
  if (isList(value)) return value.length === 0;
  if (isRecord(value)) return Object.keys(value).length === 0;
  return value === null || value === false || value === 0n || value === 0 || value === "";
}

/**
 * Desugar a compact single-node flow into the canonical one-node form.
 *
 * - The flow `id:` names the single node, so it must be an identifier.
 * - The flow `input:` is the node's signature (`name: TYPE`), auto-wired into the node by name
 *   (`p: ${input.p}`).
 * - The flow `output:` is the node's output type, re-exported as the flow output
 *   (`output: ${<id>.output}`).
 * - Everything else (the `kind:` and its logic fields, such as `prompt:`, `asserts:`,
 *   `llm_config:` or `long_term_memory:`) is the node body.
 *
 * @returns The canonical body, with a `nodes:` mapping, for the schema check.
 * @throws LoadError for a `kind:` YAML read as non-text (`null` included), a removed kind, a
 *   non-leaf kind, a flow `id:` YAML read as non-text, a missing, null, empty or non-identifier
 *   flow `id:`, or an `input:` that isn't a mapping.
 */
function desugarCompact(body: ValueRecord, root: SourceCursor): ValueRecord {
  const diagnostics = LoadDiagnostics.fromCursor(root);
  const allowed = [...COMPACT_KINDS].sort(compareCodeUnits).join(", ");
  const kind = body["kind"]!;
  if (typeof kind !== "string") {
    throw diagnostics.error(
      `compact (single-node) flow: ${notTextError("kind", kind, `write one of ${allowed}`)}`,
      { range: root.rangeOf("kind") },
    );
  }
  const removed = REMOVED_KINDS.get(kind);
  if (removed !== undefined) {
    throw diagnostics.error(
      `compact (single-node) flow: kind ${reprText(kind)} was removed; ${removed}`,
      { range: root.rangeOf("kind") },
    );
  }
  if (!COMPACT_KINDS.has(kind)) {
    throw diagnostics.error(
      `compact (single-node) flow: kind ${reprText(kind)} is not allowed inline ` +
        `(allowed: ${allowed}); case/call/map reference other nodes a one-node flow has none ` +
        "of — write a full `nodes:` map instead",
      { range: root.rangeOf("kind") },
    );
  }
  const flowId = Object.hasOwn(body, "id") ? body["id"]! : null;
  if (flowId !== null && typeof flowId !== "string") {
    throw diagnostics.error(`compact (single-node) flow: ${notTextError("id", flowId)}`, {
      range: root.rangeOf("id"),
    });
  }
  if (!flowId) {
    throw diagnostics.error(
      "compact (single-node) flow: a top-level `id:` is required — it names the single node",
      { range: root.has("id") ? root.rangeOf("id") : root.range },
    );
  }
  if (!isIdentifier(flowId)) {
    throw diagnostics.error(
      `compact flow id ${reprText(flowId)} also names its node and must be an identifier ` +
        `(${IDENTIFIER_RULE}); use ${reprText(identifierSuggestion(flowId))}, or the full ` +
        "`nodes:` form to keep a hyphenated flow id",
      { range: root.rangeOf("id") },
    );
  }
  const input = Object.hasOwn(body, "input") ? body["input"]! : null;
  if (!isRecord(input) && !isEmptyValue(input)) {
    throw diagnostics.error(
      "compact (single-node) flow: `input:` must be a param map (`name: !TYPE`, e.g. " +
        "`topic: !str`)",
      { range: root.rangeOf("input") },
    );
  }
  const params: ValueRecord = isRecord(input) ? input : {};

  // The node body is the kind and its logic fields. The flow inputs are auto-wired by name, and
  // the flow `output:` is carried through as the node's output type.
  const nodeBody: ValueRecord = Object.fromEntries(
    Object.entries(body).filter(
      ([key]) => !COMPACT_FLOW_KEYS.includes(key) && key !== "input" && key !== "output",
    ),
  );
  nodeBody["input"] = Object.fromEntries(Object.keys(params).map((p) => [p, `\${input.${p}}`]));
  if (Object.hasOwn(body, "output")) nodeBody["output"] = body["output"]!;

  const canonical: ValueRecord = Object.fromEntries(
    COMPACT_FLOW_KEYS.filter((key) => Object.hasOwn(body, key)).map((key) => [key, body[key]!]),
  );
  canonical["input"] = params;
  canonical["nodes"] = Object.fromEntries([[flowId, nodeBody]]);
  canonical["output"] = `\${${flowId}.output}`; // re-export the single node's output
  return canonical;
}

/**
 * Return the range a schema failure at `path` (authored keys and indexes from the root) points
 * to: the value there, else the root's range when the path can't be replayed or has no source.
 */
function schemaRange(root: SourceCursor, path: readonly (string | number)[]): SourceRange | null {
  let cursor = root;
  try {
    for (const step of path) cursor = cursor.child(step);
  } catch (error) {
    if (!(error instanceof SourceAccessError)) throw error;
    return root.range;
  }
  return cursor.range ?? root.range;
}

/**
 * Check `body`, back-mapped to the carrier names, against the flow schema, field by field in
 * the schema's order, and return it as a `ComposeFile`.
 *
 * @throws LoadError at the first failure that has a location: a text field YAML read as
 *   non-text, or a section of the wrong container type, at its value. A missing required field
 *   has no location, so it is reported, at the root, only when every other field passes.
 */
function readComposeFile(
  body: ValueRecord,
  root: SourceCursor,
  diagnostics: LoadDiagnostics,
): ComposeFile {
  const fields = backMapToPlural(body) as ValueRecord;
  const fail = (message: string, path: readonly (string | number)[]) =>
    diagnostics.error(message, { range: schemaRange(root, path) });
  const get = (key: string): Value | undefined =>
    Object.hasOwn(fields, key) ? fields[key]! : undefined;

  // Python's schema collects every failure and locates the first one that replays; a missing
  // key never replays, so a later located failure wins over it.
  let missing: string | null = null;
  const required = (key: string): Value | undefined => {
    const value = get(key);
    if (value === undefined) missing ??= key;
    return value;
  };
  /** Check that `value`, at `path`, is text, naming it `label`; an absent value passes. */
  const text = (
    label: string,
    value: Value | undefined,
    path: readonly (string | number)[] = [label],
  ): string | undefined => {
    if (value !== undefined && typeof value !== "string") {
      throw fail(notTextError(label, value), path);
    }
    return value;
  };
  const optionalText = (key: string): string | null => {
    const value = get(key) ?? null;
    return value === null ? null : text(key, value)!;
  };
  /** Read the mapping at carrier `key`, authored as `label`. */
  const mapping = (key: string, label = key, isRequired = false): ValueRecord | undefined => {
    const value = isRequired ? required(key) : get(key);
    if (value !== undefined && !isRecord(value)) {
      throw fail(valueTypeError(label, "a mapping", value, "write a mapping"), [label]);
    }
    return value;
  };

  const id = text("id", required("id"));
  const name = text("name", required("name"));
  const description = optionalText("description");
  const version = optionalText("version");
  const inputs = mapping("inputs", "input");
  const nodes = mapping("nodes", "nodes", true);
  const outputs = get("outputs");
  const asserts = get("asserts");
  if (asserts !== undefined) {
    if (!isList(asserts)) {
      throw fail(valueTypeError("asserts", "a list", asserts, "write a list"), ["asserts"]);
    }
    asserts.forEach((item, index) => text(`asserts[${index}]`, item, ["asserts", index]));
  }
  const typedefs = mapping("typedefs");
  const defs = mapping("defs");
  const uses = mapping("uses");
  for (const [key, ref] of Object.entries(uses ?? {})) text(`uses.${key}`, ref, ["uses", key]);
  const system = mapping("system");
  const llmConfig = get("llm_config");
  const env = mapping("env");
  const longTermMemory = get("long_term_memory");
  if (missing !== null) throw fail(`missing required top-level key ${reprText(missing)}`, []);

  return new ComposeFile({
    id: id!,
    name: name!,
    description,
    version,
    inputs,
    nodes: nodes!,
    outputs,
    asserts: asserts as string[] | undefined,
    typedefs,
    defs,
    uses: uses as Record<string, string> | undefined,
    system,
    llmConfig,
    env,
    longTermMemory,
  });
}

/** Explain that a key YAML read as non-text can't name anything, and the fix. */
function nonTextKeyMessage(key: Value, text: string): string {
  const readAs = yamlReadAs(key);
  if (!text) {
    return (
      `an empty key was read by YAML as ${readAs}, but names in a flow must be text. ` +
      "Write a name."
    );
  }
  // A JSON string is a valid YAML double-quoted scalar for any text; escaping every code unit
  // past `~` spells it as Python's `json.dumps` does.
  const quoted = JSON.stringify(text).replace(
    /[\u007f-\uffff]/g,
    (unit) => `\\u${unit.charCodeAt(0).toString(16).padStart(4, "0")}`,
  );
  return (
    `the key \`${text}\` was read by YAML as ${readAs}, but names in a flow must be text. ` +
    `Quote it (${quoted}) or choose another name.`
  );
}

/**
 * Parse one owned source document once, rejecting bad syntax, a key YAML read as non-text, and
 * a repeated key.
 *
 * @param text The flow file's YAML text.
 * @param label The name that identifies the file in diagnostics.
 * @throws LoadError at the problem for invalid YAML, at the key for a key YAML read as non-text,
 *   and at the second key for a repeated key.
 */
export function parseSourceDocument(text: string, label: string): SourceDocument {
  let document: SourceDocument;
  try {
    document = SourceDocument.parse(text, label);
  } catch (error) {
    if (!(error instanceof SourceSyntaxError)) throw error;
    throw new LoadDiagnostics(label, text).error(`invalid YAML: ${error.message}`, {
      range: error.range,
      cause: error,
    });
  }
  if (document.nonTextKey !== null) {
    const { key, text: keyText, range } = document.nonTextKey;
    throw new LoadDiagnostics(label, text).error(nonTextKeyMessage(key, keyText), { range });
  }
  if (document.repeatedKey !== null) {
    const { key, first, second } = document.repeatedKey;
    throw new LoadDiagnostics(label, text).error(
      `duplicate key ${reprText(key)} (first at line ${first.startLine}); a YAML mapping's keys ` +
        "must be unique, so remove or rename one",
      { range: second },
    );
  }
  return document;
}

/**
 * Return the authored `nodes:` cursor, or a source-aligned projection of a compact flow's one
 * synthesized node.
 *
 * A full flow owns a physical `nodes:` mapping. A compact flow's node header is aligned with the
 * authored `id:` key, its unchanged logic fields and its output contract keep their top-level
 * sources, and its generated input bindings stay virtual.
 *
 * @param document The flow's document.
 * @param nodesSection The parsed `nodes` section (`ComposeFile.nodes`).
 */
export function flowNodesCursor(document: SourceDocument, nodesSection: ValueRecord): SourceCursor {
  const root = document.root;
  if (root.has("nodes")) return root.child("nodes");
  const ids = Object.keys(nodesSection);
  if (ids.length !== 1) return SourceCursor.virtual(nodesSection);

  const nodeId = ids[0]!;
  const nodeBody = nodesSection[nodeId]!;
  if (!isRecord(nodeBody)) return SourceCursor.virtual(nodesSection);

  const entries = new Map<string, readonly [Value, SourceCursor | null]>();
  for (const [key, value] of Object.entries(nodeBody)) {
    let source: SourceCursor | null = null;
    if (key === "outputs" && root.has("output")) source = root.child("output");
    else if (key !== "inputs" && key !== "outputs" && root.has(key)) source = root.child(key);
    entries.set(key, [value, source]);
  }

  const header = root.has("id") ? root.child("id") : root;
  const bodyCursor = header.projectMapping(entries);
  return root.projectMapping(new Map([[nodeId, [nodeBody, bodyCursor]]]));
}

/**
 * Parse a flow into a strict `ComposeFile`; a text is parsed as a document labeled `<flow>`.
 *
 * @throws LoadError for malformed YAML, a non-mapping document, a retired `inputs:`/`outputs:`/
 *   `memory_policy:` key, a bad compact flow, an unknown top-level key (after stripping `x-*`),
 *   or a schema violation. A text field YAML read as non-text (flow `id:`/`name:`/
 *   `description:`/`version:`, a `uses:` ref, a flow `asserts:` item) names the field.
 */
export function parseFile(source: string | SourceDocument): ComposeFile {
  const document = typeof source === "string" ? parseSourceDocument(source, "<flow>") : source;
  const root = document.root;
  const diagnostics = LoadDiagnostics.fromDocument(document);
  const raw = root._valueView();
  if (!isRecord(raw)) {
    throw diagnostics.error("a flow must be a YAML mapping at the top level", {
      range: root.range,
    });
  }

  let body: ValueRecord = Object.fromEntries(
    Object.keys(raw)
      .filter((key) => !isExtensionKey(key))
      .map((key) => [key, root.get(key)]),
  );
  rejectLegacyTopKeys(body, root, diagnostics);
  try {
    rejectNestedLegacyKeys(root, diagnostics);
  } catch (error) {
    if (!(error instanceof SourceAccessError)) throw error;
    throw diagnostics.error(error.message, { range: error.range, cause: error });
  }

  // A compact flow is desugared before the schema check, since `nodes:` is required and its
  // inline node fields would be unknown top-level keys.
  if (Object.hasOwn(body, "kind") && !Object.hasOwn(body, "nodes")) {
    body = desugarCompact(body, root);
  }

  for (const key of Object.keys(body)) {
    if (!ALLOWED_TOP.has(key)) {
      throw diagnostics.error(
        `unknown top-level key ${reprText(key)} (allowed: ${ALLOWED_TOP_MESSAGE})`,
        { range: root.keyRangeOf(key) },
      );
    }
  }
  return readComposeFile(body, root, diagnostics);
}
