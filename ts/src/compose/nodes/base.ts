/**
 * The base descriptor contract shared by every node kind: the parse-time analogue of
 * `nodes/base.ts`.
 *
 * A `compose/nodes/<kind>.ts` module owns a *descriptor*: the validated, located shape a `kind:`
 * node body parses into before it is built into a runtime `Node`. Every per-kind descriptor
 * extends `NodeDescriptor`, which holds the fields common to all nodes, the template `parse`
 * every kind shares, and the build-side derivations over the authored `input:` and `output:`.
 * Each kind opts into the `inputs` and `outputs` carriers through its own `allowed` set, so a
 * kind without those authored fields keeps rejecting them. `build` reads one frozen
 * `BuildContext` and returns a frozen `BuildResult`, which alone keeps the YAML ranges.
 *
 * It also owns the shared messages for a field whose value YAML read as the wrong type (text or
 * otherwise), for a list or record holding a `${...}` span where the field binds whole, and for
 * an `input:` that is not a mapping, so every kind reports those mistakes in the same words.
 */

import type { Edge } from "../../compile/model.ts";
import {
  compareCodeUnits,
  type DiagnosticFrame,
  type OriginRef,
  reprText,
  type SourceRange,
} from "../../diagnostics.ts";
import { ExpressionError } from "../../expr/expressions.ts";
import { callSuffixOf, type ExprNode, Tree, wholeSpanInterior } from "../../expr/grammar.ts";
import { scanTemplate, Span, type TemplateSegment, templateRefs } from "../../expr/template.ts";
import type { Node } from "../../nodes/base.ts";
import {
  bindingSource,
  isExactly,
  type OutputContract,
  Parameter,
  validateOptionalOrigin,
  ValueContract,
} from "../../nodes/contracts.ts";
import { pythonTypeName, type TypeRegistry } from "../../typesys/types.ts";
import { isRecord, valueRepr, type ValueRecord } from "../../typesys/values.ts";
import { LoadDiagnostics, LoadError } from "../helpers/errors.ts";
import { hasSpan } from "../helpers/tags.ts";
import { readType } from "../helpers/types.ts";
import { checkEnvValues } from "../helpers/value_nesting.ts";
import { yamlReadAs } from "../helpers/yaml_words.ts";
import type { SourceCatalogBuilder } from "../source/catalog.ts";
import type { SourceCursor } from "../source/index.ts";

/**
 * Return the key of a descriptor's `fieldRanges` entry for the site named `name`: a field, an
 * `input:` binding or an `llm_config` field.
 *
 * Example: `siteKey("field", "inputs")` is `"field:inputs"`.
 */
export function siteKey(kind: "field" | "input" | "llm_config", name: string): string {
  // One text key, since a Map compares a [kind, name] pair by identity, not by value.
  return `${kind}:${name}`;
}

/**
 * Describe a non-mapping `input:` value and how to write it as a binding mapping.
 *
 * @param kind The node kind; it picks the message's binding noun and fix verb.
 * @param value The authored `input:` value, which is not a mapping.
 * @returns What `input:` must be, what it got, and a fix after a `;`. A lone reference `${R}`
 *   gets the example `<field>: ${R.<field>}`.
 */
export function inputShapeProblem(kind: string, value: unknown): string {
  const [noun, verb] =
    kind === "loop"
      ? ["carried fields", "list each carried field"]
      : kind === "call" || kind === "map"
        ? ["the child's arguments", "bind each argument"]
        : ["parameter names to bindings", "bind each parameter"];
  const path = directReferencePath(value);
  let got: string;
  let fix: string;
  if (path !== null) {
    got = `the reference '\${${path}}'`;
    fix = `${verb}, such as \`<field>: \${${path}.<field>}\``;
  } else if (typeof value === "string" && wholeSpanInterior(value) !== null) {
    got = `the expression '${value}'`;
    fix = `${verb}, such as \`<field>: \${...}\``;
  } else {
    got = valueRepr(value);
    if ((Array.isArray(value) || typeof value === "string") && value.length === 0) {
      fix = "write `input: {}` or remove `input:`";
    } else if (typeof value === "string" && hasSpan(value)) {
      fix = `${verb}, such as \`<field>: \${...}\``;
    } else {
      fix = `${verb}, such as \`<field>: ...\``;
    }
  }
  return `\`input:\` must be a mapping of ${noun} (got ${got}); ${fix}`;
}

/**
 * Reject a list or record holding a `${...}` span in a field that binds whole.
 *
 * Such a field's container is a literal, so a span inside it would stay text.
 *
 * @param kind The node kind, for the message.
 * @param nodeId The node id, for the message.
 * @param key The authored field name.
 * @param value The field's authored value.
 * @param range Where the error points.
 * @param diagnostics Factory for located load errors.
 * @throws LoadError when `value` is a list or record with a span inside it.
 */
export function rejectContainerSpans(
  kind: string,
  nodeId: string,
  key: string,
  value: unknown,
  range: SourceRange | null,
  diagnostics: LoadDiagnostics,
): void {
  if ((Array.isArray(value) || isRecord(value)) && holdsSpan(value)) {
    throw diagnostics.error(
      `${kind} node ${reprText(nodeId)} ${key}: a value holding \${...} can't be a list or ` +
        "mapping; write the list as an expression, e.g. `${[a.output, b.output]}`",
      { range },
    );
  }
}

/**
 * Return the load error message for a field whose value YAML read as the wrong type.
 *
 * @param field The field's path as authored, such as `over` or `questions[0].header`.
 * @param needed What the field takes, with its article, such as `text` or `a list`.
 * @param value The field's value as YAML read it; `null` is reported as empty.
 * @param fix What the author should write instead.
 * @returns The message, naming what YAML read the value as and the fix.
 *
 * Example: `valueTypeError("over", "a list", 5n, "write a list")` is
 * ``"`over:` must be a list, but YAML read it as an integer; write a list"``;
 * `valueTypeError("over", "a list", null, "write a list")` is
 * ``"`over:` must be a list, but it is empty; write a list"``.
 */
export function valueTypeError(field: string, needed: string, value: unknown, fix: string): string {
  if (value === null) return `\`${field}:\` must be ${needed}, but it is empty; ${fix}`;
  return `\`${field}:\` must be ${needed}, but YAML read it as ${yamlReadAs(value)}; ${fix}`;
}

/**
 * Return the load error message for a text field whose value YAML read as non-text.
 *
 * @param field The field's path as authored, such as `prompt` or `questions[0].header`.
 * @param value The field's value as YAML read it.
 * @param fix What the author should write instead. `null` uses "write the text" for an empty
 *   value and "put it in quotes" otherwise.
 * @returns The message, naming what YAML read the value as and the fix.
 *
 * Example: `notTextError("prompt", 5n)` is
 * ``"`prompt:` must be text, but YAML read it as an integer; put it in quotes"``;
 * `notTextError("questions[0].header", null)` is
 * ``"`questions[0].header:` must be text, but it is empty; write the text"``.
 */
export function notTextError(field: string, value: unknown, fix: string | null = null): string {
  fix ??= value === null ? "write the text" : "put it in quotes";
  return valueTypeError(field, "text", value, fix);
}

/**
 * Reject a text or expression field whose value YAML read as non-text.
 *
 * @param kind The node kind, for the message.
 * @param nodeId The node id, for the message.
 * @param key The authored field name.
 * @param value The field's authored value.
 * @param range Where the error points.
 * @param diagnostics Factory for located load errors.
 * @param options.required Whether the field needs a value: `false` (the default) lets `null`
 *   through as an absent field; `true` rejects it as empty.
 * @param options.fix What the author should write instead; `null` uses `notTextError`'s default.
 * @throws LoadError when `value` is neither text nor `null`, or is `null` while `required` is set.
 */
export function rejectNonText(
  kind: string,
  nodeId: string,
  key: string,
  value: unknown,
  range: SourceRange | null,
  diagnostics: LoadDiagnostics,
  options: { required?: boolean; fix?: string | null } = {},
): void {
  const { required = false, fix = null } = options;
  if (typeof value !== "string" && (value !== null || required)) {
    throw diagnostics.error(
      `node ${reprText(nodeId)} (kind=${kind}): ${notTextError(key, value, fix)}`,
      { range },
    );
  }
}

/** Resolves a child flow by id and version for call, map and loop. */
export type ChildResolver = (flowId: string, version: string | null) => unknown;

/** Presents an origin registered before the file's catalog is final. */
export type OriginPresenter = (origin: OriginRef) => DiagnosticFrame | null;

/** Compiles one node's authored `output:` declaration at its build position. */
export type OutputContractReader = (nodeId: string, outputSpec: unknown) => ValueContract | null;

/** Compiles one node's local variable declarations at their build position. */
export type VariableDeclarationReader = (nodeId: string, declarations: unknown) => Parameter[] | null;

/**
 * Resolves one parent binding's output contract from the flow inputs and the producers it
 * references, given a diagnostic label, the binding and the binding's range.
 */
export type BindingContractReader = (
  label: string,
  binding: unknown,
  range: SourceRange | null,
) => OutputContract | null;

/** The inputs a node `build` reads; frozen. */
export class BuildContext {
  /** The flow's named types, for reading declared types. */
  readonly registry: TypeRegistry;
  /** Resolves a child flow by id and version for call, map and loop; `null` for leaves. */
  readonly resolver: ChildResolver | null;
  /** The opaque file name an inline CODE body reports in its errors. */
  readonly sourceFile: string;
  /** Presents an origin registered before the file's catalog is final. */
  readonly resolveOrigin: OriginPresenter | null;
  /** Factory for located load errors. */
  readonly diagnostics: LoadDiagnostics;
  /** Compiles one authored `output:` declaration at its build position. */
  readonly outputContractReader: OutputContractReader | null;
  /** Compiles one node's local variable declarations at their build position. */
  readonly variableDeclarationReader: VariableDeclarationReader | null;
  /** Resolves one parent binding's output contract; LOOP reads it to type a carried field its body omits. */
  readonly bindingContractReader: BindingContractReader | null;
  /** The lexical flow path of the flow being assembled; AGENT stamps it into memory sites. */
  readonly owner: readonly string[];

  /**
   * Create a context. Only `registry` is required; the readers and `resolver` default to
   * `null`, `sourceFile` to `"<inline-code>"`, `diagnostics` to an unlocated context and
   * `owner` to `["root"]`, which is copied.
   */
  constructor(options: {
    registry: TypeRegistry;
    resolver?: ChildResolver | null;
    sourceFile?: string;
    resolveOrigin?: OriginPresenter | null;
    diagnostics?: LoadDiagnostics;
    outputContractReader?: OutputContractReader | null;
    variableDeclarationReader?: VariableDeclarationReader | null;
    bindingContractReader?: BindingContractReader | null;
    owner?: readonly string[];
  }) {
    const {
      registry,
      resolver = null,
      sourceFile = "<inline-code>",
      resolveOrigin = null,
      diagnostics = new LoadDiagnostics(),
      outputContractReader = null,
      variableDeclarationReader = null,
      bindingContractReader = null,
      owner = ["root"],
    } = options;
    this.registry = registry;
    this.resolver = resolver;
    this.sourceFile = sourceFile;
    this.resolveOrigin = resolveOrigin;
    this.diagnostics = diagnostics;
    this.outputContractReader = outputContractReader;
    this.variableDeclarationReader = variableDeclarationReader;
    this.bindingContractReader = bindingContractReader;
    this.owner = Object.freeze([...owner]);
    Object.freeze(this);
  }
}

/**
 * A built node plus the Compose-owned assembly metadata; frozen.
 *
 * `nodeRange`, `fieldRanges` and `fieldLabels` exist only while Compose validates the assembled
 * graph, and are dropped before the compiled flow is built.
 */
export class BuildResult {
  /** The runtime node. */
  readonly node: Node;
  /** The flow-owned wiring: parameter name to its source. */
  readonly wiring: Map<string, unknown>;
  /** The edges the node owns; empty except for a CASE. */
  readonly edges: readonly Edge[];
  /** The node body's range. */
  readonly nodeRange: SourceRange | null;
  /** Each authored site's range, keyed by `siteKey`. */
  readonly fieldRanges: Map<string, SourceRange>;
  /**
   * Site label (a field name, or an input's `input 'x'` label) to the full author-facing
   * phrase that names the site in diagnostics.
   */
  readonly fieldLabels: Map<string, string>;

  /** Create a result; `edges` defaults to empty and is copied, the metadata to empty or `null`. */
  constructor(options: {
    node: Node;
    wiring: Map<string, unknown>;
    edges?: readonly Edge[];
    nodeRange?: SourceRange | null;
    fieldRanges?: Map<string, SourceRange>;
    fieldLabels?: Map<string, string>;
  }) {
    const { node, wiring, edges = [], nodeRange = null, fieldRanges = new Map(), fieldLabels = new Map() } =
      options;
    this.node = node;
    this.wiring = wiring;
    this.edges = Object.freeze([...edges]);
    this.nodeRange = nodeRange;
    this.fieldRanges = fieldRanges;
    this.fieldLabels = fieldLabels;
    Object.freeze(this);
  }

  /**
   * Return the author-facing name of one site of this node in a diagnostic.
   *
   * @param nodeId The id this node is built under.
   * @param site A site label: a field name (`prompt`) or an input's label (`input 'x'`).
   * @returns The site's `fieldLabels` phrase when a desugar set one, else
   *   `node '<nodeId>' <site>`.
   */
  siteName(nodeId: string, site: string): string {
    return this.fieldLabels.get(site) || `node ${reprText(nodeId)} ${site}`;
  }
}

/** The fields a descriptor's constructor takes: every field but `id` is optional. */
export interface NodeDescriptorFields {
  id: string;
  nodeName?: string | null;
  dependsOn?: readonly string[];
  runsAfter?: readonly string[];
  env?: Readonly<Record<string, unknown>>;
  /** `null` reads as empty, as an authored empty `input:` parses. */
  inputs?: Readonly<Record<string, unknown>> | null;
  outputs?: unknown;
  inputTypes?: Map<string, ValueContract>;
  host?: string | null;
  fieldLabels?: Map<string, string>;
  origin?: OriginRef | null;
  range?: SourceRange | null;
  fieldRanges?: Map<string, SourceRange>;
  assertRanges?: (SourceRange | null)[];
}

/**
 * The base every `compose/nodes/<kind>.ts` descriptor extends: the fields common to all nodes.
 *
 * The statics are the per-kind parse contract each kind overrides, e.g.
 * `static override readonly allowed = this.allow("prompt")`.
 *
 * Descriptors are readonly but not frozen: the base can't freeze before the subclass assigns its
 * fields. Each constructor takes one options object keyed by field names, and each class assigns
 * its own fields after `super(fields)`; `replace` rebuilds through the kind's constructor. A kind
 * must not redeclare a base field, since an ES2022 class field resets it to `undefined` after
 * `super()`; to narrow a base field's type, write `declare readonly x: T`.
 */
export class NodeDescriptor {
  /**
   * The kind's `kind:` discriminator; empty on the base. A kind another kind extends declares it
   * `: string`, since a readonly literal would type it as that one text.
   */
  static readonly kind: string = "";
  /**
   * The complete set of body keys this kind accepts: the commons and `kind` here, which each
   * kind extends with `allow`. `inputs` and `outputs` are the carriers for the authored
   * `input:` and `output:`.
   */
  static readonly allowed: ReadonlySet<string> = new Set(["kind", "node_name", "depends_on", "runs_after", "env"]);
  /** The body keys this kind requires; empty on the base, extended with `require`. */
  static readonly required: ReadonlySet<string> = new Set();
  /**
   * Appended to the unknown-key error for an authored `when:`, since no kind takes a top-level
   * condition and the bare error would leave the author guessing where it goes.
   */
  static readonly whenHint: string =
    "; to run a node only when a condition holds, route to it from a `case` node " +
    "(`cases: [{when: ..., then: <node>}]`)";
  /** The descriptor fields carrying `${...}` binding text, for the kind-blind ref walk. */
  static readonly bindingFields: readonly string[] = [];
  /**
   * Body key to descriptor field name, where a key collides with a reserved word (`case`'s
   * `else` is `else_`); other keys take their camelCase name.
   */
  static readonly keywordRemap: ReadonlyMap<string, string> = new Map();

  /** The node's key in the `nodes:` map. */
  readonly id: string;
  /** The display title. */
  readonly nodeName: string | null;
  /** Run-ordering sources that co-skip this node when they skip. */
  readonly dependsOn: readonly string[];
  /** Run-ordering sources that gate this node without co-skipping it. */
  readonly runsAfter: readonly string[];
  /** Static per-node config, merged under the flow `env:` at build. */
  readonly env: Readonly<Record<string, unknown>>;
  /** The authored `input:` bindings: parameter name to binding. */
  readonly inputs: Readonly<Record<string, unknown>>;
  /** The authored `output:` declaration; `null` when absent. */
  readonly outputs: unknown;
  /** The declared type of each typed `input:` binding; Compose fills it at load. */
  readonly inputTypes: Map<string, ValueContract>;
  /**
   * The id of the node a desugar pass lifted this node out of, or `null`; the lifted node runs
   * only when its host can.
   */
  readonly host: string | null;
  /** Site label to the author-facing phrase naming it, set on a desugared node. */
  readonly fieldLabels: Map<string, string>;
  /** The opaque authored node identity. */
  readonly origin: OriginRef | null;
  /** The node body's range. */
  readonly range: SourceRange | null;
  /** Each authored site's range, keyed by `siteKey`; `_finish` copies it to the build result. */
  readonly fieldRanges: Map<string, SourceRange>;
  /**
   * One range per authored assert, in order (`null` where none exists). Kept out of
   * `fieldRanges` because two identical asserts still need distinct carets.
   */
  readonly assertRanges: (SourceRange | null)[];

  /**
   * Create a descriptor from `fields`; absent fields take empty or `null` values.
   *
   * @throws TypeError when `origin` isn't an `OriginRef` or `null`.
   */
  constructor(fields: NodeDescriptorFields) {
    validateOptionalOrigin(fields.origin ?? null, "node descriptor origin");
    this.id = fields.id;
    this.nodeName = fields.nodeName ?? null;
    this.dependsOn = fields.dependsOn ?? [];
    this.runsAfter = fields.runsAfter ?? [];
    this.env = fields.env ?? {};
    this.inputs = fields.inputs ?? {};
    this.outputs = fields.outputs ?? null;
    this.inputTypes = fields.inputTypes ?? new Map();
    this.host = fields.host ?? null;
    this.fieldLabels = fields.fieldLabels ?? new Map();
    this.origin = fields.origin ?? null;
    this.range = fields.range ?? null;
    this.fieldRanges = fields.fieldRanges ?? new Map();
    this.assertRanges = fields.assertRanges ?? [];
  }

  /** The kind's `kind:` discriminator: the class's static `kind`. */
  get kind(): string {
    return (this.constructor as typeof NodeDescriptor).kind;
  }

  /** Return this class's `allowed` set with `keys` added: how a kind lists its own body keys. */
  static allow(...keys: string[]): ReadonlySet<string> {
    return new Set([...this.allowed, ...keys]);
  }

  /** Return this class's `required` set with `keys` added: how a kind lists its required keys. */
  static require(...keys: string[]): ReadonlySet<string> {
    return new Set([...this.required, ...keys]);
  }

  /**
   * Parse one node body into this kind's descriptor: the template every kind shares.
   *
   * Runs `_normalizeCursor`, then `_checkKeys`, then reads each present own key (`allowed`
   * minus the base's) under its field name, recording its range under `siteKey("field", key)`,
   * then `_validateFields` and `_readCommonFields`, and constructs the descriptor.
   *
   * A kind that overrides it keeps the `this: T` signature and casts, since `super.parse`
   * returns the generic `InstanceType<T>`, which doesn't expose the kind's fields:
   *
   * ```ts
   * static override parse<T extends typeof NodeDescriptor>(
   *   this: T,
   *   nodeId: string,
   *   cursor: SourceCursor,
   * ): InstanceType<T> {
   *   const desc = super.parse(nodeId, cursor) as AgentDescriptor;
   *   // ... check or rewrite `desc` ...
   *   return desc as InstanceType<T>;
   * }
   * ```
   *
   * @param nodeId The node's id.
   * @param cursor The node body.
   * @returns A descriptor of this class.
   * @throws LoadError for an unknown or missing key, or a bad field value.
   */
  static parse<T extends typeof NodeDescriptor>(
    this: T,
    nodeId: string,
    cursor: SourceCursor,
  ): InstanceType<T> {
    const normalized = this._normalizeCursor(cursor);
    const diagnostics = LoadDiagnostics.fromCursor(normalized);
    this._checkKeys(nodeId, normalized);
    const body = normalized._valueView() as ValueRecord;
    const ownRange = normalized.range;
    const ranges = new Map<string, SourceRange>();
    const fields: Record<string, unknown> = { id: nodeId };
    for (const key of this.allowed) {
      if (NodeDescriptor.allowed.has(key) || !Object.hasOwn(body, key)) continue;
      const range = normalized.rangeOf(key);
      if (range !== null) ranges.set(siteKey("field", key), range);
      fields[this.keywordRemap.get(key) ?? camelCase(key)] = normalized.get(key);
    }
    this._validateFields(nodeId, fields, ranges, ownRange, diagnostics);
    this._readCommonFields(nodeId, normalized, fields, ranges);
    const descriptorFields = { ...fields, range: ownRange, fieldRanges: ranges } as NodeDescriptorFields;
    return new this(descriptorFields) as InstanceType<T>;
  }

  /**
   * Return the cursor `_checkKeys` inspects; unchanged by default. A kind may override it for a
   * source-aware projection or a retired-key check.
   */
  static _normalizeCursor(cursor: SourceCursor): SourceCursor {
    return cursor;
  }

  /**
   * Reject a body key outside `allowed`, at its value (a `when:` adds `whenHint`), then a missing
   * `required` key, at the body.
   *
   * @throws LoadError for the first offending key.
   */
  static _checkKeys(nodeId: string, cursor: SourceCursor): void {
    const diagnostics = LoadDiagnostics.fromCursor(cursor);
    const body = cursor._valueView() as ValueRecord;
    for (const key of Object.keys(body)) {
      if (this.allowed.has(key)) continue;
      const hint = key === "when" ? this.whenHint : "";
      // Name the authored spellings; the plural carrier names fail to load.
      const surface = new Set([...this.allowed].map((name) => SURFACE_NAMES.get(name) ?? name));
      throw diagnostics.error(
        `node ${reprText(nodeId)} (kind=${this.kind}): field ${reprText(key)} is not allowed ` +
          `(allowed: ${[...surface].sort(compareCodeUnits).join(", ")})${hint}`,
        { range: cursor.rangeOf(key) },
      );
    }
    for (const key of this.required) {
      if (!Object.hasOwn(body, key)) {
        throw diagnostics.error(
          `node ${reprText(nodeId)} (kind=${this.kind}): missing required field ${reprText(key)}`,
          { range: cursor.range },
        );
      }
    }
  }

  /**
   * Read the common `node_name`, `depends_on`, `runs_after` and `env` fields into `fields` and
   * their ranges into `ranges`, both in place.
   *
   * @throws LoadError for a non-text `node_name:`, an ordering field that isn't a list of
   *   text, or an `env:` that isn't a mapping or holds a value nested too deep.
   */
  static _readCommonFields(
    nodeId: string,
    cursor: SourceCursor,
    fields: Record<string, unknown>,
    ranges: Map<string, SourceRange>,
  ): void {
    const diagnostics = LoadDiagnostics.fromCursor(cursor);
    const body = cursor._valueView() as ValueRecord;
    for (const key of ["node_name", "depends_on", "runs_after"]) {
      if (!Object.hasOwn(body, key)) continue;
      const value = cursor.get(key);
      const range = cursor.rangeOf(key);
      if (key === "node_name") {
        rejectNonText(this.kind, nodeId, key, value, range, diagnostics);
      } else {
        this._checkOrderingField(nodeId, key, value, cursor, diagnostics);
      }
      fields[camelCase(key)] = value;
      if (range !== null) ranges.set(siteKey("field", key), range);
    }
    if (Object.hasOwn(body, "env")) {
      if (!isRecord(body["env"])) {
        throw diagnostics.error(
          `node ${reprText(nodeId)} (kind=${this.kind}): \`env:\` must be a mapping with string keys`,
          { range: cursor.rangeOf("env") },
        );
      }
      const env = cursor.mapping("env");
      checkEnvValues(env, `node ${reprText(nodeId)} `, cursor.child("env"), diagnostics);
      fields["env"] = env;
      const range = cursor.rangeOf("env");
      if (range !== null) ranges.set(siteKey("field", "env"), range);
    }
  }

  /**
   * Reject a `depends_on:` or `runs_after:` value that isn't a list of text; a bare text value
   * gets a hint to wrap it in a list, the likely intent.
   *
   * @throws LoadError at the field.
   */
  static _checkOrderingField(
    nodeId: string,
    key: string,
    value: unknown,
    cursor: SourceCursor,
    diagnostics: LoadDiagnostics,
  ): void {
    if (Array.isArray(value) && value.every((item) => typeof item === "string")) return;
    let got: string;
    if (Array.isArray(value)) got = "a list with a non-string item";
    else if (isRecord(value)) got = "a mapping";
    else if (value === null) got = "null";
    else got = `${pythonTypeName(value)} ${valueRepr(value)}`;
    let message =
      `node ${reprText(nodeId)} (kind=${this.kind}): \`${key}:\` must be a list of node ids ` +
      `(got ${got})`;
    if (typeof value === "string" && value !== "") message += `; write \`${key}: [${value}]\``;
    throw diagnostics.error(message, { range: cursor.rangeOf(key) });
  }

  /**
   * Check the kind's own fields after the template `parse` reads them; a kind overrides it and
   * calls `super`. By default a present, non-null `inputs` must be a mapping and a present
   * `asserts` a list of text.
   *
   * @param nodeId The node's id.
   * @param fields The fields read so far, by field name.
   * @param ranges The field ranges recorded so far.
   * @param ownRange The node body's range, for an error about an absent field.
   * @param diagnostics Factory for located load errors.
   * @throws LoadError for a bad `input:` or `asserts:`.
   */
  static _validateFields(
    nodeId: string,
    fields: Record<string, unknown>,
    ranges: Map<string, SourceRange>,
    ownRange: SourceRange | null,
    diagnostics: LoadDiagnostics,
  ): void {
    const inputs = fields["inputs"];
    if (inputs !== undefined && inputs !== null && !isRecord(inputs)) {
      throw diagnostics.error(
        `node ${reprText(nodeId)} (kind=${this.kind}): ${inputShapeProblem(this.kind, inputs)}`,
        { range: ranges.get(siteKey("field", "inputs")) ?? ownRange },
      );
    }
    const asserts = fields["asserts"];
    if (
      asserts !== undefined &&
      asserts !== null &&
      !(Array.isArray(asserts) && asserts.every((item) => typeof item === "string"))
    ) {
      throw diagnostics.error(
        `node ${reprText(nodeId)} (kind=${this.kind}): \`asserts:\` must be a list of strings`,
        { range: ranges.get(siteKey("field", "asserts")) ?? null },
      );
    }
  }

  /**
   * Return the declared `output:` as one contract, or `null` when there is none.
   *
   * `ctx.outputContractReader`, when set, compiles the declaration at this build position;
   * otherwise the source-free type reader does. A kind whose output is inferred may return a
   * partial record.
   *
   * @throws LoadError when the declaration isn't a valid type expression.
   * @throws TypeError when the reader returns anything but an exact `ValueContract` or `null`.
   */
  outputContract(ctx: BuildContext): OutputContract | null {
    if (this.outputs === null) return null;
    const diagnostics = ctx.diagnostics;
    try {
      if (ctx.outputContractReader !== null) {
        const contract = ctx.outputContractReader(this.id, this.outputs);
        if (contract !== null) {
          if (!isExactly(contract, ValueContract)) {
            throw new TypeError("output contract reader must return an exact ValueContract or null");
          }
          return contract;
        }
      }
      return new ValueContract(readType(this.outputs, ctx.registry, { diagnostics }));
    } catch (error) {
      if (!(error instanceof LoadError) || error.diagnostic.frames.length > 0) throw error;
      throw diagnostics.prepend(error, {
        range: this.fieldRanges.get(siteKey("field", "outputs")) ?? this.range,
        cause: error,
      });
    }
  }

  /** Return the authored `input:` names as untyped parameters; the flow owns the sources. */
  sinkParams(): Parameter[] {
    return Object.keys(this.inputs).map((name) => new Parameter(name));
  }

  /** Return the authored `input:` map as the flow-owned wiring: parameter name to source. */
  sinkWiring(): Map<string, unknown> {
    return new Map(Object.entries(this.inputs));
  }

  /**
   * Return this descriptor with kind-owned authored origins attached; the base owns none and
   * returns itself. A kind with ordered or interior operations overrides it and registers
   * those sites through the file's catalog builder.
   *
   * @param cursor The normalized authored node occurrence.
   * @param options.sitePath The node occurrence's complete semantic path.
   * @param options.catalogBuilder The file's registry that mints source-neutral origins.
   */
  withSourceOrigins(
    cursor: SourceCursor,
    options: { sitePath: readonly (string | number)[]; catalogBuilder: SourceCatalogBuilder },
  ): this {
    return this;
  }

  /**
   * Build this descriptor into a runtime node and its flow-owned wiring; each kind overrides
   * it, usually ending with `_finish`.
   *
   * @throws Error on the base, which has no build.
   */
  build(ctx: BuildContext): BuildResult {
    throw new Error(`${this.constructor.name} has no build`);
  }

  /**
   * Stamp this descriptor's `origin` onto `node` and return the build result, holding copies
   * of the ranges and field labels: the one place every build attaches source metadata.
   *
   * @throws TypeError when `origin` isn't an `OriginRef` or `null`.
   */
  _finish(node: Node, wiring: Map<string, unknown>, edges: readonly Edge[] = []): BuildResult {
    validateOptionalOrigin(this.origin, "node descriptor origin");
    node.origin = this.origin;
    return new BuildResult({
      node,
      wiring,
      edges,
      nodeRange: this.range,
      fieldRanges: new Map(this.fieldRanges),
      fieldLabels: new Map(this.fieldLabels),
    });
  }

  /**
   * Return a copy with `changes` applied, rebuilt through this class's constructor.
   *
   * `changes` is keyed by field name; the base fields are type-checked and a kind's own fields
   * pass through, so a kind's method can call `this.replace({ memories })`.
   */
  replace(changes: Partial<NodeDescriptorFields> & Readonly<Record<string, unknown>>): this {
    const cls = this.constructor as new (fields: NodeDescriptorFields) => this;
    return new cls({ ...this, ...changes });
  }
}

/** The authored spelling of each carrier key, for the unknown-key message. */
const SURFACE_NAMES: ReadonlyMap<string, string> = new Map([
  ["inputs", "input"],
  ["outputs", "output"],
]);

/** Return the camelCase field name of a snake_case YAML key: `tool_id` is `toolId`. */
function camelCase(key: string): string {
  return key.replace(/_([a-z0-9])/g, (_match, letter: string) => letter.toUpperCase());
}

/** Whether any text inside `value`, or `value` itself, holds a `${...}` span. */
function holdsSpan(value: unknown): boolean {
  if (typeof value === "string") return hasSpan(value);
  if (isRecord(value)) return Object.values(value).some(holdsSpan);
  if (Array.isArray(value)) return value.some(holdsSpan);
  return false;
}

/**
 * Return the path a binding reads when it is one lone whole-value reference.
 *
 * Python keeps this in `compose/helpers/output_contracts.py`.
 *
 * @param binding A raw binding value or a `Binding` carrier.
 * @returns The dotted reference path (`item`, `item.name`, `a.output`), or `null` for a
 *   literal, an embedded or computed template, a call, or malformed text.
 */
function directReferencePath(binding: unknown): string | null {
  const source = bindingSource(binding);
  if (typeof source !== "string") return null;
  let segments: TemplateSegment[];
  try {
    segments = scanTemplate(source);
  } catch (error) {
    if (error instanceof ExpressionError) return null;
    throw error;
  }
  const [segment] = segments;
  if (!(segments.length === 1 && segment instanceof Span)) return null;
  if (!isDirectReference(segment.tree)) return null;
  const refs = templateRefs(segments);
  return refs.length === 1 ? refs[0]! : null;
}

/** Whether a whole-value span's tree is a bare or dotted reference with no call suffix. */
function isDirectReference(tree: ExprNode): boolean {
  return tree instanceof Tree && tree.data === "refcall" && callSuffixOf(tree) === undefined;
}
