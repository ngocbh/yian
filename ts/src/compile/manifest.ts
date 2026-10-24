/**
 * Canonical representation-neutral manifests for authored flows and built fragments.
 *
 * The manifest is deliberately explicit: every execution-relevant IR field is named here, and
 * every known diagnostic or cache field is named as ignored. A new instance field therefore fails
 * loudly until its identity policy is decided, instead of silently changing program semantics
 * without changing the fingerprint.
 *
 * Every authored value, including a literal under any name, is part of the manifest: the manifest
 * never redacts or guesses at secrets.
 *
 * A manifest is plain JSON for `canonicalJson` (RFC 8785): every value is a typed wrapper
 * (`{"type": "int", "value": "1"}`, a float as its IEEE-754 big-endian hex), and only counts are
 * raw JSON ints. The fingerprint is SHA-256 over that text.
 *
 * Knows about:   diagnostics, nodes, memory contracts, typesys and compile/model.
 * Never imports: compose, runtime, the run boundary, the CLI, or provider SDKs.
 */

import { createHash } from "node:crypto";

import { compareCodeUnits, OriginRef, reprText } from "../diagnostics.ts";
import { DeclSite, MemoryDecl, MemoryOccurrenceSegment, MemoryScope } from "../memory/contracts.ts";
import { AgentNode, authoredPrompt } from "../nodes/agent/node.ts";
import { Node, NodeKind } from "../nodes/base.ts";
import { CallNode } from "../nodes/call/node.ts";
import { Case, CaseNode } from "../nodes/case/node.ts";
import { classifyCodeSource, CodeNode } from "../nodes/code/node.ts";
import {
  Assertion,
  Binding,
  ContractOrigins,
  ExecutionScope,
  isExactly,
  Parameter,
  PartialRecordContract,
  ValueContract,
  ValueTemplate,
} from "../nodes/contracts.ts";
import { EndNode } from "../nodes/end/node.ts";
import { HumanInputNode } from "../nodes/human_input/node.ts";
import { LoopNode } from "../nodes/loop/node.ts";
import { MapNode } from "../nodes/map/node.ts";
import { StartNode } from "../nodes/start/node.ts";
import { ToolNode } from "../nodes/tool/node.ts";
import { WaitNode } from "../nodes/wait/node.ts";
import { canonicalJson, type JsonObject, type JsonValue } from "../typesys/json.ts";
import {
  canonicalUnionOrder,
  FileRef,
  isList,
  isRecord,
  jsTypeName,
  Type,
  valueRepr,
  writeAsText,
} from "../typesys/values.ts";
import { CompiledFlow, Edge, Flow, FlowOutput, validateGrowthMetadata } from "./model.ts";

// Fields common to every concrete Node. Most are emitted below, `kind` among them; title is emitted
// only for HUMAN_INPUT because other node kinds use it as display metadata, and types only when
// non-empty.
// Keeping both sets explicit is the fail-loud field policy.
const COMMON_NODE_FIELDS = [
  "id",
  "kind",
  "title",
  "outputContract",
  "params",
  "preAsserts",
  "postAsserts",
  "commitAs",
  "originId",
  "env",
  "types",
];
// The traits are fixed by the kind (Python's are class variables, outside its census), so they are
// never identity of their own.
const IGNORED_NODE_FIELDS = [
  "origin",
  "originContext",
  "memoryScope",
  "executionScope",
  "wiringLabels",
  "bindingLabels",
  "isSpawner",
  "commitsInputRecord",
  "bindsPerItem",
  "itemSourceKey",
  "growDepthDelta",
  "isLoop",
  "needsMemoryContext",
  "needsSkills",
  "needsEvaluationTrace",
  "needsRenderCapture",
  "needsTools",
  "needsLiveEvents",
  "needsSteps",
  "needsSteering",
  "needsModels",
  "derived",
];

// A kind owns both the canonical fields that affect its behavior and known noncanonical fields
// that are derived caches or render-only metadata. Unknown fields are never serialized by guess.
const KIND_FIELDS: Readonly<Record<string, readonly string[]>> = {
  [NodeKind.START]: [],
  [NodeKind.END]: ["mode", "names", "n"],
  [NodeKind.AGENT]: [
    "prompt",
    "systemPrompt",
    "tools",
    "ownLlmConfig",
    "llmInherit",
    "retries",
    "memories",
    "variables",
    "skills",
    "steerable",
  ],
  [NodeKind.CODE]: ["code"],
  [NodeKind.TOOL]: ["toolId", "retries"],
  [NodeKind.CASE]: ["cases"],
  [NodeKind.HUMAN_INPUT]: ["prompt", "answerSchema", "questions", "questionsInput"],
  // WAIT's execution identity is its required flow-owned `until` wiring; the node owns no
  // additional constructor state.
  [NodeKind.WAIT]: [],
  [NodeKind.CALL]: ["flowId", "flowVersion", "child"],
  [NodeKind.MAP]: ["flowId", "flowVersion", "child"],
  [NodeKind.LOOP]: [
    "flowId",
    "flowVersion",
    "child",
    "predicateKind",
    "predicate",
    "times",
    "maxIters",
    "iteration",
    "returnedFields",
  ],
};
const IGNORED_KIND_FIELDS: Readonly<Record<string, readonly string[]>> = {
  [NodeKind.START]: [],
  [NodeKind.END]: [],
  // `llmConfig` is always run-resolved; `ownLlmConfig` is the authored identity source.
  // `llmConfigLabel` is render-only. `wiresLlmConfig` and `llmValueTypes` are derived from the
  // layers; the reserved wiring they decide is manifested with the wiring. `callTime` and
  // `hostCredentials` are run-resolved like `llmConfig`.
  [NodeKind.AGENT]: [
    "llmConfig",
    "llmConfigLabel",
    "wiresLlmConfig",
    "llmValueTypes",
    "callTime",
    "hostCredentials",
  ],
  // `mode` is derived from `code`; CODE origins are diagnostic metadata resolved only by the
  // authoring front end; `baseDir` is where a reference resolves, not what it names; `sourceFile`
  // only labels a failing body's stack frames.
  [NodeKind.CODE]: ["mode", "baseDir", "sourceFile", "codeOrigin", "codeLineOrigins"],
  // `inputTypes` is derived from wiring and producer contracts; the run-start tool check reads it.
  // `answerSchema` is not part of a TOOL's identity yet: no loaded flow sets it.
  [NodeKind.TOOL]: ["inputTypes", "answerSchema"],
  [NodeKind.CASE]: [],
  [NodeKind.HUMAN_INPUT]: [],
  [NodeKind.WAIT]: [],
  // `childInputs` duplicates the recursive child's START signature.
  [NodeKind.CALL]: ["childInputs"],
  // `parallel` is retained by the authoring model but runtime concurrency uses the worker count.
  [NodeKind.MAP]: ["childInputs", "parallel"],
  [NodeKind.LOOP]: ["childInputs"],
};
const KIND_CLASSES: Readonly<Record<string, abstract new (...args: never[]) => Node>> = {
  [NodeKind.START]: StartNode,
  [NodeKind.END]: EndNode,
  [NodeKind.AGENT]: AgentNode,
  [NodeKind.CODE]: CodeNode,
  [NodeKind.TOOL]: ToolNode,
  [NodeKind.CASE]: CaseNode,
  [NodeKind.HUMAN_INPUT]: HumanInputNode,
  [NodeKind.WAIT]: WaitNode,
  [NodeKind.CALL]: CallNode,
  [NodeKind.MAP]: MapNode,
  [NodeKind.LOOP]: LoopNode,
};
const NODE_KINDS: ReadonlySet<unknown> = new Set(Object.values(NodeKind));
// Each kind's full field policy, built once. Every field is constructor-owned, so a node missing
// one is not a complete canonical node.
const NODE_FIELDS: ReadonlyMap<string, ReadonlySet<string>> = new Map(
  Object.keys(KIND_FIELDS).map((kind) => [
    kind,
    new Set([...COMMON_NODE_FIELDS, ...IGNORED_NODE_FIELDS, ...KIND_FIELDS[kind]!, ...IGNORED_KIND_FIELDS[kind]!]),
  ]),
);

const BUILTIN_CODE_FIELDS = new Set(["builtin", "field"]);
const EDGE_FIELDS = new Set(["id", "from", "to", "sourceHandle", "inputGroup", "optional", "ordering"]);
const FLOW_OUTPUT_FIELDS = new Set(["name", "binding"]);
// Canonical fields, then ignored ones.
const BINDING_FIELDS = new Set(["source", "contract", "origin"]);
const VALUE_TEMPLATE_FIELDS = new Set(["value", "origins"]);
const ASSERTION_FIELDS = new Set(["expression", "display", "origin"]);
const PARAMETER_FIELDS = new Set([
  "name",
  "contract",
  "required",
  "hasDefault",
  "default",
  "coerceStrings",
  "origin",
  "defaultOrigin",
]);
const VALUE_CONTRACT_FIELDS = new Set(["type", "origins"]);
const PARTIAL_RECORD_CONTRACT_FIELDS = new Set(["fields"]);
const CONTRACT_ORIGINS_IGNORED_FIELDS = new Set(["default", "entries"]);
const TYPE_FIELDS = new Set(["kind", "fields", "required", "tags", "element", "nullable", "bounds", "variants"]);
const MEMORY_DECL_FIELDS = new Set(["name", "type", "config", "site", "origin"]);
const CASE_FIELDS = new Set(["handle", "when", "origin"]);
const FLOW_FIELDS = ["nodes", "edges", "wiring", "startId", "endId", "inputConnections", "childBoundaries"];
const FLOW_FIELD_SET = new Set(FLOW_FIELDS);
const COMPILED_FLOW_FIELD_SET = new Set([
  ...FLOW_FIELDS,
  "outputs",
  "flowLlmConfig",
  // Derived from the layer and the flow's contracts; not identity.
  "flowLlmValueTypes",
  "flowId",
  "memories",
  "incomingEdges",
  "outgoingEdges",
]);

/** Recursion stacks used to reject cycles while allowing shared immutable aliases. */
class ManifestContext {
  readonly flows = new Set<object>();
  readonly values = new Set<object>();
  readonly types = new Set<object>();
}

/** Name `value`'s type for a message, `null` for `null` (Python's `NoneType`). */
function typeName(value: unknown): string {
  return value === null ? "null" : jsTypeName(value);
}

/**
 * Run `body` with `value` on the ancestor stack `stack`, so a shared alias encodes twice and only
 * a cycle is refused.
 *
 * @throws RangeError `message` when `value` is already an ancestor.
 */
function within<T>(stack: Set<object>, value: object, message: string, body: () => T): T {
  if (stack.has(value)) throw new RangeError(message);
  stack.add(value);
  try {
    return body();
  } finally {
    stack.delete(value);
  }
}

/**
 * Require an object's own enumerable fields to match an explicit manifest policy.
 *
 * @param label Names the unknown fields in the message.
 * @throws RangeError naming the unknown fields, else the missing ones.
 */
function checkFields(
  value: object,
  expected: ReadonlySet<string>,
  path: string,
  label = "noncanonical field(s)",
): void {
  const actual = new Set(Object.keys(value));
  const unknown = [...actual].filter((name) => !expected.has(name)).sort(compareCodeUnits);
  const missing = [...expected].filter((name) => !actual.has(name)).sort(compareCodeUnits);
  if (unknown.length > 0) throw new RangeError(`${path}: unknown ${label}: ${writeAsText(unknown)}`);
  if (missing.length > 0) throw new RangeError(`${path}: missing canonical field(s): ${writeAsText(missing)}`);
}

/**
 * Check one canonical binding, whose source is identity-bearing data.
 *
 * The declared contract is canonical too, but callers project it separately so an untyped binding
 * has a source-only encoding.
 *
 * @throws TypeError when `value` isn't a `Binding`.
 * @throws RangeError when its fields aren't the policy's.
 */
function checkedBinding(value: unknown, path: string): Binding {
  if (!(value instanceof Binding)) throw new TypeError(`${path}: expected Binding, got ${typeName(value)}`);
  checkFields(value, BINDING_FIELDS, path);
  return value;
}

/** Return a binding's canonical declared type, or `undefined` when untyped. */
function bindingTypeManifest(value: Binding, ctx: ManifestContext, path: string): JsonValue | undefined {
  if (value.contract === null) return undefined;
  return valueContractManifest(value.contract, ctx, `${path}.contract`).type;
}

/**
 * Check direct wiring carriers, returning the `[parameter, binding]` entries in wiring order.
 *
 * @throws TypeError when a key isn't a string or a value isn't a `Binding`.
 */
function bindingManifestEntries(value: ReadonlyMap<unknown, unknown>, path: string): [string, Binding][] {
  for (const key of value.keys()) {
    if (typeof key !== "string") throw new TypeError(`${path}: mapping keys must be strings, got ${typeName(key)}`);
  }
  return Array.from(value, ([key, item], index): [string, Binding] => [
    key as string,
    checkedBinding(item, `${path}.values[${index}]`),
  ]);
}

/**
 * Project one canonical assertion to behavior and presentation identity.
 *
 * @throws TypeError when `value` isn't exactly an `Assertion`.
 */
function assertionManifest(value: unknown, ctx: ManifestContext, path: string): JsonObject {
  if (!isExactly(value, Assertion)) throw new TypeError(`${path}: expected Assertion, got ${typeName(value)}`);
  checkFields(value, ASSERTION_FIELDS, path);
  return {
    expression: canonicalValue(value.expression, ctx, `${path}.expression`),
    display: canonicalValue(value.display, ctx, `${path}.display`),
  };
}

/**
 * Apply the common plus per-kind strict field policy to one concrete node.
 *
 * @throws RangeError when `kind` isn't a `NodeKind`, the node's class isn't exactly its kind's, or
 *   a field is unknown or missing.
 */
function checkNodeFields(node: Node, path: string): void {
  const kind = node.kind;
  if (!NODE_KINDS.has(kind)) throw new RangeError(`${path}: node kind must be a NodeKind`);
  const expectedClass = KIND_CLASSES[kind]!;
  if (!isExactly(node, expectedClass)) {
    throw new RangeError(`${path}: node kind ${reprText(kind)} requires ${expectedClass.name}, got ${typeName(node)}`);
  }
  checkFields(node, NODE_FIELDS.get(kind)!, path, `noncanonical execution field(s) for ${kind}`);
}

/**
 * Canonicalize a record as sorted typed key/value entries.
 *
 * @param value The authored record, such as `env:` or memory config.
 * @param ctx Recursion stacks for cycle detection.
 * @param path Location used in error messages.
 * @returns The `{"entries": [...], "type": "mapping"}` record.
 * @throws TypeError when `value` isn't a plain record.
 */
function sortedMapping(value: unknown, ctx: ManifestContext, path: string): JsonObject {
  if (!isRecord(value)) throw new TypeError(`${path}: expected a record, got ${typeName(value)}`);
  return mappingManifest(Object.entries(value), value, ctx, path, canonicalValue);
}

/**
 * Canonicalize one authored `llm_config` layer as written.
 *
 * A literal layer encodes as a sorted mapping, a layer holding values as its `ValueTemplate`, and
 * a whole record written as one value as its `Binding`.
 *
 * @throws TypeError when a literal layer isn't a plain record.
 */
function llmLayerManifest(layer: unknown, ctx: ManifestContext, path: string): JsonValue {
  if (layer instanceof Binding) return templateTreeManifest(layer, ctx, path);
  if (layer instanceof ValueTemplate) return canonicalValue(layer, ctx, path);
  return sortedMapping(layer, ctx, path);
}

/** Canonicalize an ordered array with `encode`, preserving element order. */
function canonicalSequence(
  value: readonly unknown[],
  ctx: ManifestContext,
  path: string,
  encode: (item: unknown, ctx: ManifestContext, path: string) => JsonValue = canonicalValue,
): JsonValue[] {
  return within(ctx.values, value, `${path}: cyclic value is not manifestable`, () =>
    // `Array.from` visits holes as `undefined`, so a hole is refused with its path.
    Array.from(value, (item, index) => encode(item, ctx, `${path}[${index}]`)),
  );
}

/**
 * Canonicalize string-keyed entries as typed key/value entries sorted by key.
 *
 * TS keys are strings, so the key alone orders the entries.
 *
 * @param entries The entries of a record or a `Map`.
 * @param owner The container, put on the cycle stack while its values encode.
 * @param encode Encodes each value: `canonicalValue`, or `templateTreeManifest` inside a tree.
 */
function mappingManifest(
  entries: Iterable<readonly [string, unknown]>,
  owner: object,
  ctx: ManifestContext,
  path: string,
  encode: (item: unknown, ctx: ManifestContext, path: string) => JsonValue,
): JsonObject {
  return within(ctx.values, owner, `${path}: cyclic value is not manifestable`, () => {
    const encoded = Array.from(entries, ([key, item], index): [string, JsonObject] => [
      key,
      { key: { type: "str", value: key }, value: encode(item, ctx, `${path}.values[${index}]`) },
    ]);
    encoded.sort(([a], [b]) => compareCodeUnits(a, b));
    return { entries: encoded.map(([, entry]) => entry), type: "mapping" };
  });
}

/** Encode a value template's tree; its origins are diagnostic-only. */
function valueTemplateManifest(value: ValueTemplate, ctx: ManifestContext, path: string): JsonObject {
  checkFields(value, VALUE_TEMPLATE_FIELDS, path);
  return { tree: templateTreeManifest(value.value, ctx, `${path}.value`), type: "value_template" };
}

/**
 * Encode one part of a template tree, where a nested `Binding` may appear.
 *
 * Containers encode as `canonicalValue` does; a nested binding encodes its source and declared
 * type, and every other leaf encodes as a plain value.
 */
function templateTreeManifest(value: unknown, ctx: ManifestContext, path: string): JsonValue {
  if (value instanceof Binding) {
    checkFields(value, BINDING_FIELDS, path);
    const contract =
      value.contract === null ? null : valueContractManifest(value.contract, ctx, `${path}.contract`).type;
    return { contract, source: templateTreeManifest(value.source, ctx, `${path}.source`), type: "binding" };
  }
  if (isList(value)) return { items: canonicalSequence(value, ctx, path, templateTreeManifest), type: "list" };
  if (isRecord(value)) return mappingManifest(Object.entries(value), value, ctx, path, templateTreeManifest);
  return canonicalValue(value, ctx, path);
}

/**
 * Return a count (a node count or a seed-path item index) as a raw JSON int: `null` where
 * `nullable` allows it (LOOP `maxIters`), else a safe integer.
 *
 * @throws TypeError for anything else (a float, infinity, a bigint).
 */
function count(value: unknown, path: string, nullable: false): number;
function count(value: unknown, path: string, nullable: boolean): number | null;
function count(value: unknown, path: string, nullable: boolean): number | null {
  if (nullable && value === null) return null;
  if (typeof value !== "number" || !Number.isSafeInteger(value)) {
    throw new TypeError(`${path}: expected a safe integer, got ${valueRepr(value)}`);
  }
  // `-0 + 0` is `0`: negative zero never reaches a fingerprint.
  return value + 0;
}

/** Return `value`'s IEEE-754 binary64 bits, big-endian, as 16 hex digits (Python's `struct.pack("!d")`). */
function floatHex(value: number): string {
  const view = new DataView(new ArrayBuffer(8));
  view.setFloat64(0, value);
  return view.getBigUint64(0).toString(16).padStart(16, "0");
}

/**
 * Encode one value as a deterministic, type-discriminated JSON record.
 *
 * A `ValueTemplate` encodes as `value_template` around its tree, so no literal shares a template's
 * encoding. A float is its IEEE-754 hex (`-0` as `0`), and a `FileRef` its `{uri, mime, name}`
 * mapping.
 *
 * @throws TypeError for NaN (V8 and JavaScriptCore write a NaN's bits differently), a bigint the
 *   runtime can't stringify, or a value that isn't a yian value (`undefined`, a `Map`, a class
 *   instance, a function).
 * @throws RangeError for a cyclic container.
 */
function canonicalValue(value: unknown, ctx: ManifestContext, path: string): JsonValue {
  if (value instanceof ValueTemplate) return valueTemplateManifest(value, ctx, path);
  if (value === null) return { type: "null" };
  if (typeof value === "boolean") return { type: "bool", value };
  if (typeof value === "bigint") {
    let digits: string;
    try {
      digits = value.toString();
    } catch (error) {
      // JavaScriptCore reports running out of memory for a huge bigint as a RangeError.
      if (error instanceof RangeError) throw new TypeError(`${path}: integer is too large for this runtime`);
      throw error;
    }
    return { type: "int", value: digits };
  }
  if (typeof value === "number") {
    if (Number.isNaN(value)) throw new TypeError(`${path}: NaN is not a yian value`);
    return { type: "float", value: floatHex(value + 0) };
  }
  if (typeof value === "string") return { type: "str", value };
  if (FileRef.isFileRef(value)) {
    // A partial LOOP driver's default for an omitted `file` field encodes as its JSON form.
    const fields = { uri: value.uri, mime: value.mime, name: value.name };
    return mappingManifest(Object.entries(fields), value, ctx, path, canonicalValue);
  }
  if (isList(value)) return { items: canonicalSequence(value, ctx, path), type: "list" };
  if (isRecord(value)) return mappingManifest(Object.entries(value), value, ctx, path, canonicalValue);
  throw new TypeError(`${path}: unsupported non-JSON value of type ${jsTypeName(value)}`);
}

/**
 * Canonicalize one resolved structural Type, including nested fields, elements and members.
 *
 * An absent refinement is `null`; `bounds` and `union` appear only when present, the members in
 * canonical order so authored order is not identity.
 *
 * @throws RangeError for a cyclic type or a field policy mismatch.
 * @throws TypeError or RangeError as `validateShape` does, for a bounded or union type.
 */
function typeManifest(value: Type | null | undefined, ctx: ManifestContext, path: string): JsonValue {
  if (value === null || value === undefined) return null;
  checkFields(value, TYPE_FIELDS, path);
  return within(ctx.types, value, `${path}: cyclic Type is not manifestable`, () => {
    const fields =
      value.fields === undefined
        ? null
        : Array.from(value.fields, ([key, fieldType]) => ({
            name: key,
            type: typeManifest(fieldType, ctx, `${path}.fields.${key}`),
          }));
    const manifest: JsonObject = {
      element: typeManifest(value.element, ctx, `${path}.element`),
      fields,
      kind: value.kind,
      nullable: value.nullable,
      required: value.required === undefined ? null : [...value.required].sort(compareCodeUnits),
      tags: value.tags === undefined ? null : [...value.tags].sort(compareCodeUnits),
    };
    if (value.bounds !== undefined) {
      value.validateShape();
      // Each end is a typed value: a bound may be a float or an int past 2^53.
      manifest.bounds = value.bounds.map((end, index) => canonicalValue(end, ctx, `${path}.bounds[${index}]`));
    }
    if (value.variants !== undefined) {
      value.validateShape();
      manifest.union = canonicalUnionOrder(value.variants).map((item, index) =>
        typeManifest(item, ctx, `${path}.union[${index}]`),
      );
    }
    return manifest;
  });
}

/**
 * Project one value contract while excluding every nested origin.
 *
 * @throws TypeError when `value` isn't exactly a `ValueContract`, or its origins exactly a
 *   `ContractOrigins`.
 */
function valueContractManifest(value: unknown, ctx: ManifestContext, path: string): { type: JsonValue } {
  if (!isExactly(value, ValueContract)) throw new TypeError(`${path}: expected ValueContract, got ${typeName(value)}`);
  checkFields(value, VALUE_CONTRACT_FIELDS, path);
  if (!isExactly(value.origins, ContractOrigins)) {
    throw new TypeError(`${path}.origins: expected ContractOrigins, got ${typeName(value.origins)}`);
  }
  checkFields(value.origins, CONTRACT_ORIGINS_IGNORED_FIELDS, `${path}.origins`);
  return { type: typeManifest(value.type, ctx, `${path}.type`) };
}

/**
 * Canonicalize either output-contract arm while excluding provenance.
 *
 * @throws TypeError when `value` is exactly neither a `ValueContract` nor a
 *   `PartialRecordContract`.
 */
function outputContractManifest(value: unknown, ctx: ManifestContext, path: string): JsonValue {
  if (isExactly(value, ValueContract)) return valueContractManifest(value, ctx, path).type;
  if (!isExactly(value, PartialRecordContract)) {
    throw new TypeError(`${path}: expected ValueContract or PartialRecordContract, got ${typeName(value)}`);
  }
  checkFields(value, PARTIAL_RECORD_CONTRACT_FIELDS, path);
  value.validateShape();
  return {
    partial_record: value.fields.map(([name, contract]) => ({
      name,
      type: contract === null ? null : valueContractManifest(contract, ctx, `${path}.fields.${name}`).type,
    })),
  };
}

/**
 * Project one exact canonical node parameter to source-neutral identity.
 *
 * @throws TypeError when `value` isn't exactly a `Parameter`.
 */
function parameterManifest(value: unknown, ctx: ManifestContext, path: string): JsonObject {
  if (!isExactly(value, Parameter)) throw new TypeError(`${path}: expected Parameter, got ${typeName(value)}`);
  checkFields(value, PARAMETER_FIELDS, path);
  const contract = value.contract === null ? null : valueContractManifest(value.contract, ctx, `${path}.contract`);
  return {
    coerce_strings: value.coerceStrings,
    contract,
    default: canonicalValue(value.default, ctx, `${path}.default`),
    has_default: value.hasDefault,
    name: value.name,
    required: value.required,
  };
}

/**
 * Project one authored memory declaration while excluding its origin.
 *
 * @throws TypeError when `value` isn't exactly a `MemoryDecl`, its site exactly a `DeclSite`, or
 *   its origin an `OriginRef` or `null`.
 * @throws RangeError when its name or type isn't a non-empty string, or its site or origin fails
 *   its constructor's checks.
 */
function memoryDeclManifest(value: unknown, ctx: ManifestContext, path: string): JsonObject {
  if (!isExactly(value, MemoryDecl)) throw new TypeError(`${path}: expected exact MemoryDecl, got ${typeName(value)}`);
  checkFields(value, MEMORY_DECL_FIELDS, path);
  const site: unknown = value.site;
  if (!isExactly(site, DeclSite)) throw new TypeError(`${path}.site: expected exact DeclSite`);
  // Rebuilding replays the frozen carrier's checks on a forged site.
  new DeclSite(site.scope, site.owner, site.node);
  for (const label of ["name", "type"] as const) {
    const item: unknown = value[label];
    if (typeof item !== "string" || !item) throw new RangeError(`${path}.${label}: expected a non-empty exact string`);
  }
  const origin: unknown = value.origin;
  if (origin !== null) {
    if (!(origin instanceof OriginRef)) throw new TypeError(`${path}.origin: expected OriginRef or null`);
    new OriginRef(origin.key);
  }
  return {
    config: sortedMapping(value.config, ctx, `${path}.config`),
    name: value.name,
    site: { node: site.node, owner: [...site.owner], scope: [...site.scope] },
    type: value.type,
  };
}

/**
 * Project authored declarations in authored order.
 *
 * @throws TypeError when `values` isn't an array.
 */
function memoriesManifest(values: unknown, ctx: ManifestContext, path: string): JsonObject[] {
  if (!isList(values)) throw new TypeError(`${path}: expected an array of MemoryDecl values`);
  return values.map((value, index) => memoryDeclManifest(value, ctx, `${path}[${index}]`));
}

/**
 * Canonicalize one ordered CASE arm; its origin is diagnostic-only.
 *
 * @throws TypeError when `value` isn't exactly a `Case`, or its origin isn't an `OriginRef` or
 *   `null`.
 * @throws RangeError when its fields aren't the policy's.
 */
function caseManifest(value: unknown, ctx: ManifestContext, path: string): JsonObject {
  if (!isExactly(value, Case)) throw new TypeError(`${path}: expected Case, got ${typeName(value)}`);
  checkFields(value, CASE_FIELDS, path);
  const origin: unknown = value.origin;
  if (origin !== null && !(origin instanceof OriginRef)) {
    throw new TypeError(`${path}.origin: expected OriginRef or null`);
  }
  return { handle: value.handle, when: canonicalValue(value.when, ctx, `${path}.when`) };
}

/**
 * Project one dynamic memory occurrence segment.
 *
 * @throws TypeError or RangeError when `segment` isn't exactly a `MemoryOccurrenceSegment` or
 *   fails its constructor's checks.
 */
function memoryOccurrenceTopology(segment: unknown, path: string): JsonObject {
  if (!isExactly(segment, MemoryOccurrenceSegment)) {
    throw new TypeError(`${path}: expected exact MemoryOccurrenceSegment, got ${typeName(segment)}`);
  }
  // Rebuilding replays the frozen carrier's checks on a forged segment.
  new MemoryOccurrenceSegment(segment.kind, segment.staticPath, segment.index);
  return {
    index: count(segment.index, `${path}.index`, false),
    kind: segment.kind,
    static_path: [...segment.staticPath],
  };
}

/**
 * Project the memory topology needed for deterministic growth identity: the node's path, its
 * memory keys and its occurrence path, without configs or origins.
 *
 * @throws TypeError or RangeError when `value` isn't exactly a `MemoryScope` or fails its
 *   constructor's checks.
 */
function memoryScopeTopology(value: unknown, path: string): JsonObject {
  if (!isExactly(value, MemoryScope)) {
    throw new TypeError(`${path}: expected exact MemoryScope, got ${typeName(value)}`);
  }
  // Rebuilding replays the frozen carrier's checks on a forged scope.
  new MemoryScope(value.staticPath, { memories: value.memories, occurrencePath: value.occurrencePath });
  return {
    memories: [...value.memories],
    occurrence_path: value.occurrencePath.map((segment, index) =>
      memoryOccurrenceTopology(segment, `${path}.occurrencePath[${index}]`),
    ),
    static_path: [...value.staticPath],
  };
}

/**
 * Project a node's execution location for growth identity.
 *
 * @throws TypeError when `value` isn't exactly an `ExecutionScope`; TypeError or RangeError when
 *   it fails its constructor's checks.
 */
function executionScopeTopology(value: unknown, path: string): JsonObject {
  if (!isExactly(value, ExecutionScope)) throw new TypeError(`${path} must be an ExecutionScope`);
  // Rebuilding replays the frozen carrier's checks on a forged scope.
  new ExecutionScope(value.staticPath, {
    flowPath: value.flowPath,
    occurrencePath: value.occurrencePath,
    boundary: value.boundary,
  });
  return {
    static_path: [...value.staticPath],
    flow_path: [...value.flowPath],
    occurrence_path: value.occurrencePath.map((position, index) => ({
      kind: position.kind,
      static_path: [...position.staticPath],
      index: count(position.index, `${path}.occurrencePath[${index}].index`, true),
    })),
    boundary: value.boundary,
  };
}

/**
 * Describe a CODE node's `code`: `{"inline": body}` or `{"ref": "./x.ts:fn"}` for text, or the
 * built-in form `{"builtin": "identity", "field": name}`.
 *
 * @throws TypeError when `code` is neither text nor a built-in identity form.
 * @throws RangeError when a built-in form's keys aren't exactly `builtin` and `field`, or its
 *   `field` is empty.
 */
function codeManifest(code: unknown, path: string): JsonObject {
  // Text is classified from itself, not the stored `mode`, which is a derived cache.
  if (typeof code === "string") return classifyCodeSource(code) === "reference" ? { ref: code } : { inline: code };
  const message = `${path}: expected code text or a built-in operation, got ${typeName(code)}`;
  if (!isRecord(code)) throw new TypeError(message);
  checkFields(code, BUILTIN_CODE_FIELDS, path);
  if (code.builtin !== "identity" || typeof code.field !== "string") throw new TypeError(message);
  if (code.field === "") throw new RangeError(`${path}: built-in field must be non-empty`);
  return { builtin: "identity", field: code.field };
}

/**
 * Canonicalize one concrete node under its exhaustive per-kind field policy.
 *
 * @param topologyOnly Whether this is the expansion topology, which adds the node's memory and
 *   execution scopes and encodes baked child flows in topology mode too.
 * @throws TypeError or RangeError when a field breaks the policy or holds a value that isn't
 *   manifestable.
 */
function nodeManifest(node: Node, ctx: ManifestContext, path: string, topologyOnly: boolean): JsonObject {
  checkNodeFields(node, path);
  let params: JsonValue = null;
  const declared: unknown = node.params;
  if (declared !== null) {
    if (!isList(declared)) throw new TypeError(`${path}.params: expected array, got ${typeName(declared)}`);
    params = declared.map((param, index) => parameterManifest(param, ctx, `${path}.params[${index}]`));
  }
  const preAsserts: unknown = node.preAsserts;
  if (!isList(preAsserts)) throw new TypeError(`${path}.preAsserts: expected array, got ${typeName(preAsserts)}`);
  const postAsserts: unknown = node.postAsserts;
  if (!isList(postAsserts)) throw new TypeError(`${path}.postAsserts: expected array, got ${typeName(postAsserts)}`);
  const outputType =
    node.outputContract === null ? null : outputContractManifest(node.outputContract, ctx, `${path}.outputContract`);
  const manifest: JsonObject = {
    commit_as: canonicalValue(node.commitAs, ctx, `${path}.commitAs`),
    env: sortedMapping(node.env, ctx, `${path}.env`),
    id: node.id,
    kind: node.kind,
    origin_id: canonicalValue(node.originId, ctx, `${path}.originId`),
    // The contract's diagnostic origins stay outside program identity.
    output_type: outputType,
    params,
    post_asserts: Array.from(postAsserts, (assertion, index) =>
      assertionManifest(assertion, ctx, `${path}.postAsserts[${index}]`),
    ),
    pre_asserts: Array.from(preAsserts, (assertion, index) =>
      assertionManifest(assertion, ctx, `${path}.preAsserts[${index}]`),
    ),
  };

  // Emitted only when non-empty, so a node without `isinstance` or a `T(...)` constructor doesn't
  // carry the key (the shape follows Python's manifest).
  const types: unknown = node.types;
  if (!isExactly(types, Map)) throw new TypeError(`${path}.types: expected Map, got ${typeName(types)}`);
  const typeEntries = sortedStringKeys(types, `${path}.types`).map((name) => {
    const typ: unknown = types.get(name);
    if (!isExactly(typ, Type)) throw new TypeError(`${path}.types.${name}: expected Type, got ${typeName(typ)}`);
    return { name, type: typeManifest(typ, ctx, `${path}.types.${name}`) };
  });
  if (typeEntries.length > 0) manifest.types = typeEntries;

  switch (node.kind) {
    case NodeKind.START:
      break;
    case NodeKind.END: {
      const end = node as EndNode;
      const list = end.mode === "list";
      // A value terminal encodes as a record END, as Python's manifest does.
      manifest.mode = list ? "list" : "record";
      manifest.names = list ? null : canonicalSequence(end.names, ctx, `${path}.names`);
      if (end.mode === "record" && end.names.length <= 1) {
        // A zero- or one-field record would otherwise encode like a value END with the same
        // names, though it returns a record. `keyed` separates them; it is left out elsewhere, so
        // it doesn't change the identity of value and multi-field ENDs.
        manifest.keyed = true;
      }
      manifest.n = list ? count(end.n, `${path}.n`, false) : null;
      break;
    }
    case NodeKind.AGENT: {
      const agent = node as AgentNode;
      const variables: unknown = agent.variables;
      if (!isList(variables)) throw new TypeError(`${path}.variables: expected array, got ${typeName(variables)}`);
      if (variables.length > 0) {
        manifest.variables = variables.map((variable, index) =>
          parameterManifest(variable, ctx, `${path}.variables[${index}]`),
        );
      }
      // Left out when empty so flows without skills keep their identity. Grant order never changes
      // behavior (the listing is sorted), so it isn't identity either.
      const skills: unknown = agent.skills;
      if (skills === "all") {
        manifest.skills = "all";
      } else {
        if (!isList(skills)) throw new TypeError(`${path}.skills: expected "all" or an array, got ${typeName(skills)}`);
        skills.forEach((skill, index) => {
          if (typeof skill !== "string") {
            throw new TypeError(`${path}.skills[${index}]: expected a string, got ${typeName(skill)}`);
          }
        });
        if (skills.length > 0) manifest.skills = [...(skills as string[])].sort(compareCodeUnits);
      }
      // Left out when false so flows that never opt in keep their identity.
      if (agent.steerable) manifest.steerable = true;
      // Left out when unset so flows on the default system prompt keep their identity.
      if (agent.systemPrompt !== null) manifest.system_prompt = agent.systemPrompt;
      const tools: unknown = agent.tools;
      if (!isList(tools)) throw new TypeError(`${path}.tools: expected array, got ${typeName(tools)}`);
      const memories = memoriesManifest(agent.memories, ctx, `${path}.memories`);
      manifest.llm_inherit = agent.llmInherit;
      // A list prompt as authored; a text prompt stays text, so text-prompt flows keep their identity.
      manifest.prompt = typeof agent.prompt === "string" ? agent.prompt : authoredPrompt(agent.prompt);
      manifest.retries = count(agent.retries, `${path}.retries`, false);
      manifest.tools = canonicalSequence(tools, ctx, `${path}.tools`);
      manifest.own_llm_config = llmLayerManifest(agent.ownLlmConfig, ctx, `${path}.ownLlmConfig`);
      // Left out when empty so memory-free flows keep their identity.
      if (!topologyOnly && memories.length > 0) manifest.memories = memories;
      break;
    }
    case NodeKind.CODE:
      manifest.code = codeManifest((node as CodeNode).code, `${path}.code`);
      break;
    case NodeKind.TOOL: {
      const tool = node as ToolNode;
      manifest.tool_id = tool.toolId;
      const retries = count(tool.retries, `${path}.retries`, false);
      // Left out at zero, so retries don't change the identity of flows that don't set them (the
      // shape follows Python's manifest).
      if (retries !== 0) manifest.retries = retries;
      break;
    }
    case NodeKind.CASE: {
      const cases: unknown = (node as CaseNode).cases;
      if (!isList(cases)) throw new TypeError(`${path}.cases: expected array, got ${typeName(cases)}`);
      manifest.cases = cases.map((branch, index) => caseManifest(branch, ctx, `${path}.cases[${index}]`));
      break;
    }
    case NodeKind.HUMAN_INPUT: {
      const human = node as HumanInputNode;
      manifest.answer_schema = canonicalValue(human.answerSchema, ctx, `${path}.answerSchema`);
      manifest.prompt = canonicalValue(human.prompt, ctx, `${path}.prompt`);
      manifest.questions = canonicalValue(human.questions, ctx, `${path}.questions`);
      manifest.questions_input = canonicalValue(human.questionsInput, ctx, `${path}.questionsInput`);
      manifest.title = canonicalValue(human.title, ctx, `${path}.title`);
      break;
    }
    case NodeKind.WAIT:
      // WAIT owns no constructor state; its absolute deadline is canonicalized through the flow's
      // ordinary wiring manifest.
      break;
    case NodeKind.CALL:
    case NodeKind.MAP:
    case NodeKind.LOOP: {
      const spawner = node as CallNode | MapNode | LoopNode;
      manifest.child = flowManifestOf(spawner.child, ctx, `${path}.child`, topologyOnly);
      manifest.flow_id = spawner.flowId;
      manifest.flow_version = canonicalValue(spawner.flowVersion, ctx, `${path}.flowVersion`);
      if (spawner instanceof LoopNode) {
        manifest.iteration = count(spawner.iteration, `${path}.iteration`, false);
        manifest.max_iters = count(spawner.maxIters, `${path}.maxIters`, true);
        manifest.predicate = canonicalValue(spawner.predicate, ctx, `${path}.predicate`);
        manifest.predicate_kind = spawner.predicateKind;
        // `times` is a typed value, as in Python's manifest; it passes `count` first, so a forged
        // float is refused rather than written as a float.
        const times = spawner.times === null ? null : BigInt(count(spawner.times, `${path}.times`, false));
        manifest.times = canonicalValue(times, ctx, `${path}.times`);
        // Left out for a total loop so the feature doesn't change the identity of flows that don't
        // use it (the shape follows Python's manifest).
        if (spawner.returnedFields !== null) manifest.returned_fields = [...spawner.returnedFields];
      }
      break;
    }
    default:
      throw new RangeError(`${path}: unsupported node kind ${reprText(node.kind)}`);
  }

  // Derived memory and execution state is never authored program identity. A dynamic fragment's
  // identity includes the scope's root-relative path, key stack and occurrence path, so replay
  // rejects a structurally different occurrence without persisting configs or source origins.
  if (topologyOnly && node.memoryScope !== null) {
    manifest.memory_scope = memoryScopeTopology(node.memoryScope, `${path}.memoryScope`);
  }
  if (topologyOnly && node.executionScope !== null) {
    manifest.execution_scope = executionScopeTopology(node.executionScope, `${path}.executionScope`);
  }
  return manifest;
}

/** Canonicalize one edge; callers preserve the flow's semantic edge order. */
function edgeManifest(edge: unknown, ctx: ManifestContext, path: string): JsonObject {
  if (!isExactly(edge, Edge)) throw new TypeError(`${path}: expected Edge, got ${typeName(edge)}`);
  checkFields(edge, EDGE_FIELDS, path);
  return {
    from: edge.from,
    id: edge.id,
    input_group: canonicalValue(edge.inputGroup, ctx, `${path}.inputGroup`),
    optional: edge.optional,
    ordering: edge.ordering,
    source_handle: canonicalValue(edge.sourceHandle, ctx, `${path}.sourceHandle`),
    to: edge.to,
  };
}

/** Canonicalize one declared flow output; callers preserve declaration order. */
function outputManifest(output: unknown, ctx: ManifestContext, path: string): JsonObject {
  if (!isExactly(output, FlowOutput)) throw new TypeError(`${path}: expected FlowOutput, got ${typeName(output)}`);
  checkFields(output, FLOW_OUTPUT_FIELDS, path);
  const binding = checkedBinding(output.binding, `${path}.binding`);
  const manifest: JsonObject = {
    from: canonicalValue(binding.source, ctx, `${path}.binding.source`),
    name: output.name,
  };
  // Only a typed output adds a key, so typing doesn't change the identity of untyped outputs.
  const outputType = bindingTypeManifest(binding, ctx, `${path}.binding`);
  if (outputType !== undefined) manifest.type = outputType;
  return manifest;
}

/** Require `value` to be a `Map` with string keys, returning its keys sorted. */
function sortedStringKeys(value: unknown, path: string): string[] {
  if (!(value instanceof Map)) throw new TypeError(`${path}: expected Map, got ${typeName(value)}`);
  const keys = [...(value as Map<unknown, unknown>).keys()];
  for (const key of keys) {
    if (typeof key !== "string") throw new TypeError(`${path}: keys must be strings, got ${typeName(key)}`);
  }
  return (keys as string[]).sort(compareCodeUnits);
}

/**
 * Canonicalize a Flow recursively, omitting CompiledFlow-only root fields in topology mode.
 *
 * @throws TypeError or RangeError when the graph holds a noncanonical class, field or value, or a
 *   child-flow cycle.
 */
function flowManifestOf(flow: unknown, ctx: ManifestContext, path: string, topologyOnly: boolean): JsonObject {
  if (!(flow instanceof Flow)) throw new TypeError(`${path}: expected Flow, got ${typeName(flow)}`);
  return within(ctx.flows, flow, `${path}: recursive child-flow cycle is not manifestable`, () => {
    let expected: ReadonlySet<string>;
    if (isExactly(flow, CompiledFlow)) expected = COMPILED_FLOW_FIELD_SET;
    else if (isExactly(flow, Flow)) expected = FLOW_FIELD_SET;
    else throw new TypeError(`${path}: unsupported Flow subclass ${typeName(flow)}`);
    checkFields(flow, expected, path);
    validateGrowthMetadata(flow);

    const nodes = flow.nodes as Map<string, unknown>;
    const nodeEntries = sortedStringKeys(nodes, `${path}.nodes`).map((nodeId): [string, JsonObject] => {
      const node = nodes.get(nodeId);
      if (!(node instanceof Node)) throw new TypeError(`${path}.nodes.${nodeId}: expected Node, got ${typeName(node)}`);
      return [nodeId, nodeManifest(node, ctx, `${path}.nodes.${nodeId}`, topologyOnly)];
    });

    const wiring = flow.wiring as Map<string, unknown>;
    const wiringEntries: [string, JsonObject][] = [];
    const bindingTypes: [string, JsonObject][] = [];
    for (const nodeId of sortedStringKeys(wiring, `${path}.wiring`)) {
      const nodeWiring = wiring.get(nodeId);
      if (!(nodeWiring instanceof Map)) {
        throw new TypeError(`${path}.wiring.${nodeId}: expected Map, got ${typeName(nodeWiring)}`);
      }
      const bindings = bindingManifestEntries(nodeWiring, `${path}.wiring.${nodeId}`);
      const sources = bindings.map(([param, binding]): [string, unknown] => [param, binding.source]);
      wiringEntries.push([
        nodeId,
        mappingManifest(sources, nodeWiring, ctx, `${path}.wiring.${nodeId}`, canonicalValue),
      ]);
      const nodeTypes: [string, JsonValue][] = [];
      for (const [param, binding] of bindings.sort(([a], [b]) => compareCodeUnits(a, b))) {
        const paramType = bindingTypeManifest(binding, ctx, `${path}.wiring.${nodeId}.${param}`);
        if (paramType !== undefined) nodeTypes.push([param, paramType]);
      }
      if (nodeTypes.length > 0) bindingTypes.push([nodeId, Object.fromEntries(nodeTypes)]);
    }

    const edges: unknown = flow.edges;
    if (!isList(edges)) throw new TypeError(`${path}.edges: expected array, got ${typeName(edges)}`);
    // Built with `Object.fromEntries`, so a node id such as `__proto__` stays an own key.
    const manifest: JsonObject = {
      edges: Array.from(edges, (edge, index) => edgeManifest(edge, ctx, `${path}.edges[${index}]`)),
      end_id: flow.endId,
      nodes: Object.fromEntries(nodeEntries),
      start_id: flow.startId,
      wiring: Object.fromEntries(wiringEntries),
    };
    // Declared binding types live beside the wiring rather than inside it, and only when present,
    // so typing doesn't change the identity of untyped flows (the shape follows Python's manifest).
    if (bindingTypes.length > 0) manifest.binding_types = Object.fromEntries(bindingTypes);
    if (topologyOnly && flow.inputConnections.length > 0) {
      manifest.input_connections = flow.inputConnections
        .map((connection, index) => {
          const itemPath = `${path}.inputConnections[${index}].seedPath`;
          return {
            child_start_id: connection.childStartId,
            input_name: connection.inputName,
            // An item index is a count, so `-0` is written as `0`.
            seed_path: connection.seedPath.map((item, step) =>
              typeof item === "number" ? count(item, `${itemPath}[${step}]`, false) : item,
            ),
          };
        })
        .sort(
          (a, b) =>
            compareCodeUnits(a.child_start_id, b.child_start_id) || compareCodeUnits(a.input_name, b.input_name),
        );
    }
    if (topologyOnly && flow.childBoundaries.length > 0) {
      manifest.child_boundaries = [...flow.childBoundaries]
        .sort((a, b) => compareCodeUnits(a.startId, b.startId) || compareCodeUnits(a.endId, b.endId))
        .map((boundary) => ({ start_id: boundary.startId, end_id: boundary.endId }));
    }
    if (flow instanceof CompiledFlow) {
      const flowId: unknown = flow.flowId;
      if (flowId !== null && (typeof flowId !== "string" || !flowId)) {
        throw new RangeError(`${path}.flowId: expected a non-empty exact string or null`);
      }
      const memories = memoriesManifest(flow.memories, ctx, `${path}.memories`);
      if (!topologyOnly) {
        if (flowId !== null) manifest.flow_id = flowId;
        // Omitted when empty, so memories don't change the identity of memory-free flows.
        if (memories.length > 0) manifest.memories = memories;
        manifest.flow_llm_config = llmLayerManifest(flow.flowLlmConfig, ctx, `${path}.flowLlmConfig`);
        const outputs: unknown = flow.outputs;
        if (!isList(outputs)) throw new TypeError(`${path}.outputs: expected array, got ${typeName(outputs)}`);
        manifest.outputs = Array.from(outputs, (output, index) =>
          outputManifest(output, ctx, `${path}.outputs[${index}]`),
        );
      }
    }
    return manifest;
  });
}

/** Return lowercase SHA-256 over a manifest's RFC 8785 text. */
function digest(value: JsonValue): string {
  return createHash("sha256").update(canonicalJson(value), "utf8").digest("hex");
}

/**
 * Return the canonical recursive build manifest for a representation-neutral flow.
 *
 * @param flow A compiled authored flow or a shared `Flow` topology carrier.
 * @returns Plain JSON with mapping entries sorted and semantic list order preserved; object keys
 *   are sorted by `canonicalJson`.
 * @throws TypeError when the graph holds a noncanonical class or an unsupported value.
 * @throws RangeError when a node or flow has an unknown field or a recursive object cycle.
 */
export function flowManifest(flow: Flow): JsonObject {
  return flowManifestOf(flow, new ManifestContext(), "flow", false);
}

/** Return lowercase SHA-256 over the canonical recursive authored-flow manifest. */
export function flowFingerprint(flow: Flow): string {
  return digest(flowManifest(flow));
}

/**
 * Return SHA-256 over a built fragment's canonical topology manifest.
 *
 * The topology identity holds the entry and terminal ids, sorted node and wiring maps, every
 * build-relevant node field (including recursive baked children), edges in their semantic order,
 * and the fragment's seed connections and child boundaries. CompiledFlow-only root outputs, LLM
 * layers, flow id and memories are intentionally not part of a dynamic fragment digest.
 *
 * @param fragment The self-contained Flow a deterministic growth builder returned.
 * @returns A lowercase 64-character SHA-256 hexadecimal digest.
 */
export function expansionDigest(fragment: Flow): string {
  return digest(flowManifestOf(fragment, new ManifestContext(), "expansion", true));
}
