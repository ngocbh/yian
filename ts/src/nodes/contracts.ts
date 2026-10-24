/**
 * Source-neutral semantic IR shared by node and flow construction.
 *
 * These frozen carriers attach opaque origins to executable meanings without depending on
 * Compose/YAML or runtime state. Compiled wiring, node assertions, node parameters, and node
 * outputs use their canonical carriers. Execution scopes identify authored locations and
 * structured occurrences independently of memory. Seed connections and child boundary pairs
 * describe growth topology without values or runtime producer identities. Compose output
 * derivation preserves those carriers and projects exact field/value types only inside type
 * algorithms.
 */

import { compareCodeUnits, OriginRef, reprText } from "../diagnostics.ts";
import {
  buildValueAs,
  canonicalTypeLabel,
  isRecord,
  Type,
  TypeCheckError,
  TypedValue,
  type Value,
  ValueKind,
  type ValueRecord,
  unionField,
  unionTagFields,
} from "../typesys/index.ts";
import {
  fileFieldType,
  hasFile,
  jsTypeName,
  setField,
  shownText,
  typeEquals,
  valueRepr,
  writeAsText,
} from "../typesys/values.ts";

/** A source-neutral path through record fields and list indices. */
export type ValuePath = readonly (string | number)[];

/** Transforms one present origin. */
export type OriginMapper = (origin: OriginRef) => OriginRef;

/**
 * Require an optional origin field to contain only the opaque leaf value. Used by node kinds.
 *
 * @throws TypeError when `origin` is neither `null` nor an `OriginRef`.
 */
export function validateOptionalOrigin(origin: unknown, fieldName: string): asserts origin is OriginRef | null {
  if (origin !== null && !(origin instanceof OriginRef)) {
    throw new TypeError(`${fieldName} must be an OriginRef or null`);
  }
}

/**
 * Transform one present origin and reject erasure or a non-origin result. Used by node kinds.
 *
 * @returns `null` for an absent origin (the mapper isn't called), else the mapped origin.
 * @throws TypeError when `mapper` returns anything but an `OriginRef`.
 */
export function mapOrigin(origin: OriginRef | null, mapper: OriginMapper): OriginRef | null {
  if (origin === null) return null;
  const mapped: unknown = mapper(origin);
  if (!(mapped instanceof OriginRef)) throw new TypeError("origin mapper must return an OriginRef");
  return mapped;
}

/** Validate one value path made of exact string and integer segments. */
function validatePath(path: unknown, fieldName: string): asserts path is ValuePath {
  if (!Array.isArray(path)) throw new TypeError(`${fieldName} path must be an array`);
  for (const segment of path as unknown[]) {
    if (typeof segment !== "string" && !(typeof segment === "number" && Number.isInteger(segment))) {
      throw new TypeError(`${fieldName} path segment must be an exact str or int, got ${jsTypeName(segment)}`);
    }
  }
}

/**
 * Order two paths in the source-neutral canonical order `ContractOrigins` requires: strings
 * before integers, a prefix before its descendants. A comparator for `Array.prototype.sort`.
 */
export function comparePaths(a: ValuePath, b: ValuePath): number {
  for (let i = 0; i < Math.min(a.length, b.length); i++) {
    const x = a[i]!;
    const y = b[i]!;
    if (typeof x !== typeof y) return typeof x === "string" ? -1 : 1;
    const order = typeof x === "string" ? compareCodeUnits(x, y as string) : x - (y as number);
    if (order !== 0) return order;
  }
  return a.length - b.length;
}

/** Require authored names while allowing the empty root path. */
function validateAuthoredPath(path: unknown, name: string): asserts path is readonly string[] {
  if (!Array.isArray(path)) throw new TypeError(`${name} must be an array`);
  if ((path as unknown[]).some((part) => typeof part !== "string" || !part)) {
    throw new RangeError(`${name} components must be non-empty strings`);
  }
}

/** Whether `value` is an instance of exactly `cls`, not of a subclass. Used by node kinds. */
export function isExactly<T>(value: unknown, cls: abstract new (...args: never[]) => T): value is T {
  return typeof value === "object" && value !== null && Object.getPrototypeOf(value) === cls.prototype;
}

/** Whether two arrays have equal items in order, by `same`; Python's tuple `==`. */
export function sameItems<T>(
  a: readonly T[],
  b: readonly T[],
  same: (x: T, y: T) => boolean = (x, y) => x === y,
): boolean {
  return a.length === b.length && a.every((item, i) => same(item, b[i]!));
}

/** Whether two nullable values are both `null`, or both present and equal by `same`. */
export function sameOptional<T>(a: T | null, b: T | null, same: (x: T, y: T) => boolean): boolean {
  return a === null ? b === null : b !== null && same(a, b);
}

/** Copy `entries` of `(path, origin)` pairs into frozen arrays. */
function freezeEntries(
  entries: readonly (readonly [ValuePath, OriginRef])[],
): readonly (readonly [ValuePath, OriginRef])[] {
  return Object.freeze(entries.map(([path, origin]) => Object.freeze([Object.freeze([...path]), origin] as const)));
}

const POSITION_KINDS = ["call", "map_item", "loop_body", "loop_driver"] as const;

/** The kinds of structured position in an instantiated child computation. */
export type PositionKind = (typeof POSITION_KINDS)[number];

/** One structured position in an instantiated child computation. Frozen. */
export class ExecutionPosition {
  /** The authored composition operation and, for LOOP, its execution role. */
  readonly kind: PositionKind;
  /** Root-relative path of the owning spawner. */
  readonly staticPath: readonly string[];
  /**
   * Non-negative item or iteration index. CALL uses `null` because it instantiates exactly one
   * child per execution.
   */
  readonly index: number | null;

  /**
   * Validate a structured position without interpreting runtime display ids; `staticPath` is
   * copied.
   *
   * @throws RangeError for an unknown kind, an empty path component, an index on a call, or a
   *   negative index.
   * @throws TypeError when the path isn't an array, or a non-call index isn't a safe integer.
   */
  constructor(kind: PositionKind, staticPath: readonly string[], index: number | null = null) {
    if (typeof kind !== "string" || !POSITION_KINDS.includes(kind)) {
      throw new RangeError("unknown execution position kind");
    }
    validateAuthoredPath(staticPath, "static_path");
    if (kind === "call") {
      if (index !== null) throw new RangeError("a call position cannot have an index");
    } else if (typeof index !== "number" || !Number.isSafeInteger(index)) {
      throw new TypeError("execution position index must be an integer");
    } else if (index < 0) {
      throw new RangeError("execution position index must be non-negative");
    }
    this.kind = kind;
    this.staticPath = Object.freeze([...staticPath]);
    this.index = index;
    Object.freeze(this);
  }
}

/** An authored execution location independent of memory declarations. Frozen. */
export class ExecutionScope {
  /**
   * Root-relative authored node or flow path. The empty path identifies the root itself,
   * including unnamed flows.
   */
  readonly staticPath: readonly string[];
  /** Root-relative path of the containing flow; the empty path denotes the root flow. */
  readonly flowPath: readonly string[];
  /**
   * Outer-to-inner child positions. Exact dynamic identity additionally uses the run id and
   * accepted node-start sequence, supplied by the runtime.
   */
  readonly occurrencePath: readonly ExecutionPosition[];
  /** This location's role as a flow boundary, when it has one. */
  readonly boundary: "start" | "end" | null;

  /**
   * Require explicit source-neutral occurrence data; the paths are copied. `flowPath` and
   * `occurrencePath` default to empty, `boundary` to `null`.
   *
   * @throws TypeError when a path isn't an array or an occurrence isn't an `ExecutionPosition`.
   * @throws RangeError for an empty path component, an invalid position, or an unknown boundary.
   */
  constructor(
    staticPath: readonly string[],
    options: {
      flowPath?: readonly string[];
      occurrencePath?: readonly ExecutionPosition[];
      boundary?: "start" | "end" | null;
    } = {},
  ) {
    const { flowPath = [], occurrencePath = [], boundary = null } = options;
    validateAuthoredPath(staticPath, "static_path");
    validateAuthoredPath(flowPath, "flow_path");
    if (!Array.isArray(occurrencePath)) throw new TypeError("occurrence_path must be an array");
    for (const position of occurrencePath as unknown[]) {
      if (!isExactly(position, ExecutionPosition)) {
        throw new TypeError("occurrence_path items must be ExecutionPosition values");
      }
      // Rebuilding checks a position that may have bypassed the constructor.
      new ExecutionPosition(position.kind, position.staticPath, position.index);
    }
    if (boundary !== null && boundary !== "start" && boundary !== "end") {
      throw new RangeError("execution boundary must be 'start', 'end', or null");
    }
    this.staticPath = Object.freeze([...staticPath]);
    this.flowPath = Object.freeze([...flowPath]);
    this.occurrencePath = Object.freeze([...occurrencePath]);
    this.boundary = boundary;
    Object.freeze(this);
  }
}

/** Whether two positions have the same kind, index and static path. */
export function sameExecutionPosition(a: ExecutionPosition, b: ExecutionPosition): boolean {
  return a.kind === b.kind && a.index === b.index && sameItems(a.staticPath, b.staticPath);
}

/** Whether two scopes have the same static path, flow path, occurrence positions and boundary. */
export function sameExecutionScope(a: ExecutionScope, b: ExecutionScope): boolean {
  return (
    sameItems(a.staticPath, b.staticPath) &&
    sameItems(a.flowPath, b.flowPath) &&
    sameItems(a.occurrencePath, b.occurrencePath, sameExecutionPosition) &&
    a.boundary === b.boundary
  );
}

/**
 * Connect a fragment input to a materialized parent seed route. Frozen.
 *
 * The target is a child START parameter, or an input of another fragment node that is declared
 * with a default and left unwired, such as a field a LOOP body omits and the next driver keeps
 * from the carried record.
 *
 * `childStartId` names the target node, which is a child START or a node with a defaulted unwired
 * input.
 *
 * `seedPath` is a parameter name, or an explicit item index followed by a parameter name.
 * Producer identities are supplied only by accepted runtime captures; this recipe contains no
 * seed value or expression.
 */
export class SeedConnection {
  readonly childStartId: string;
  readonly inputName: string;
  readonly seedPath: readonly (string | number)[];

  /**
   * Require non-empty names and a closed materialized-seed selector; `seedPath` is copied.
   *
   * @throws RangeError for an empty or non-text id, or a `seedPath` that isn't `[name]` or
   *   `[index, name]` with a non-negative integer index.
   */
  constructor(childStartId: string, inputName: string, seedPath: readonly (string | number)[]) {
    for (const name of [childStartId, inputName]) {
      if (typeof name !== "string" || !name) {
        throw new RangeError("seed connection identifiers must be non-empty strings");
      }
    }
    if (!Array.isArray(seedPath) || (seedPath.length !== 1 && seedPath.length !== 2)) {
      throw new RangeError("seed_path must be a name or an index/name tuple");
    }
    const name: unknown = seedPath[seedPath.length - 1];
    if (typeof name !== "string" || !name) {
      throw new RangeError("seed_path must end in a non-empty parameter name");
    }
    const index: unknown = seedPath[0];
    if (seedPath.length === 2 && (typeof index !== "number" || !Number.isInteger(index) || index < 0)) {
      throw new RangeError("seed_path item index must be a non-negative integer");
    }
    this.childStartId = childStartId;
    this.inputName = inputName;
    this.seedPath = Object.freeze([...seedPath]);
    Object.freeze(this);
  }
}

/** The actual START and END of one instantiated authored child flow. Frozen. */
export class ChildFlowBoundary {
  readonly startId: string;
  readonly endId: string;

  /**
   * Require distinct ids without interpreting their spelling.
   *
   * @throws RangeError for an empty or non-text id, or equal ids.
   */
  constructor(startId: string, endId: string) {
    if ([startId, endId].some((name) => typeof name !== "string" || !name)) {
      throw new RangeError("child boundary identifiers must be non-empty strings");
    }
    if (startId === endId) throw new RangeError("child START and END must be distinct");
    this.startId = startId;
    this.endId = endId;
    Object.freeze(this);
  }
}

/**
 * One flow-owned input expression or literal, its origin, and its declared type. Frozen.
 *
 * The binding owns its declared contract: a typed binding states the type its value must have,
 * independent of the receiving node's parameter contract.
 */
export class Binding {
  /** Semantic binding payload; retained by identity. */
  readonly source: unknown;
  /** Authored binding site, when available. */
  readonly origin: OriginRef | null;
  /** Declared value contract, when typed. */
  readonly contract: ValueContract | null;

  /**
   * Bind `source`; `origin` and `contract` default to `null`.
   *
   * @throws TypeError when `origin` isn't an `OriginRef` or `contract` isn't exactly a
   *   `ValueContract`.
   */
  constructor(source: unknown, options: { origin?: OriginRef | null; contract?: ValueContract | null } = {}) {
    const { origin = null, contract = null } = options;
    validateOptionalOrigin(origin, "binding origin");
    if (contract !== null && !isExactly(contract, ValueContract)) {
      throw new TypeError("binding contract must be a ValueContract or null");
    }
    this.source = source;
    this.origin = origin;
    this.contract = contract;
    Object.freeze(this);
  }

  /**
   * Return this binding with every origin transformed by `mapper`.
   *
   * That covers its own origin, its contract's origins, and, for a `ValueTemplate` source, the
   * template's origins and nested bindings. Any other source is retained by identity.
   */
  mapOrigins(mapper: OriginMapper): Binding {
    const origin = mapOrigin(this.origin, mapper);
    const contract = this.contract !== null ? this.contract.mapOrigins(mapper) : null;
    const source = this.source instanceof ValueTemplate ? this.source.mapOrigins(mapper) : this.source;
    return new Binding(source, { origin, contract });
  }

  /**
   * Yield this binding's present origin, then its contract's origins.
   *
   * For a `ValueTemplate` source, the template's origins follow, then each nested binding's
   * origin and contract origins, in tree order.
   */
  *iterOrigins(): Generator<OriginRef> {
    yield* iterOriginCarrier(this, new Set());
  }
}

/** A template tree: a record or a list whose leaves are any values. */
type TemplateTree = Record<string, unknown> | unknown[];

/**
 * An authored record or list whose leaf strings are binding templates. Frozen.
 *
 * The binder evaluates every leaf string as a template (a whole `${...}` keeps its type, an
 * embedded one is stringified, `$$` is unescaped) and every nested typed `Binding` against its
 * own contract. A plain record or list binding source, by contrast, is always a literal.
 *
 * A nested `Binding` sits at its position without adding a path segment; its source is part of
 * the same tree (a template string or a nested container).
 */
export class ValueTemplate {
  /** The tree, a record or a list; retained by identity. */
  readonly value: TemplateTree;
  /** Authored sites of leaf strings, keyed by their path in `value`, for diagnostics. */
  readonly origins: readonly (readonly [ValuePath, OriginRef])[];

  /**
   * Require a container tree and well-formed origin entries; `origins` (default empty) is
   * copied.
   *
   * @throws TypeError when `value` isn't a plain record or an array, or an origin entry isn't a
   *   `[path, OriginRef]` pair with a valid path.
   */
  constructor(value: TemplateTree, origins: readonly (readonly [ValuePath, OriginRef])[] = []) {
    if (!isRecord(value) && !Array.isArray(value)) {
      throw new TypeError(`value template must hold a record or array, got ${jsTypeName(value)}`);
    }
    if (!Array.isArray(origins)) throw new TypeError("value template origins must be an array");
    for (const entry of origins as unknown[]) {
      if (!Array.isArray(entry) || entry.length !== 2) {
        throw new TypeError("each value template origin must be a (path, origin) tuple");
      }
      const [path, origin] = entry as unknown[];
      validatePath(path, "value template origin");
      if (!(origin instanceof OriginRef)) throw new TypeError("value template origin must be an OriginRef");
    }
    this.value = value;
    this.origins = freezeEntries(origins);
    Object.freeze(this);
  }

  /** Return the authored site registered for `path`, or `null`. */
  originAt(path: ValuePath): OriginRef | null {
    for (const [entryPath, origin] of this.origins) {
      if (sameItems(entryPath, path)) return origin;
    }
    return null;
  }

  /** Return this template with its origins and nested bindings' origins mapped. */
  mapOrigins(mapper: OriginMapper): ValueTemplate {
    return new ValueTemplate(
      mapTree(this.value, (text) => text, mapper) as TemplateTree,
      this.origins.map(([path, origin]) => [path, mapOrigin(origin, mapper) as OriginRef] as const),
    );
  }
}

/** Yield every binding nested in a template tree, each before those in its source. */
function* treeBindings(node: unknown): Generator<Binding> {
  if (node instanceof Binding) {
    yield node;
    yield* treeBindings(node.source);
  } else if (isRecord(node)) {
    for (const item of Object.values(node)) yield* treeBindings(item);
  } else if (Array.isArray(node)) {
    for (const item of node) yield* treeBindings(item);
  }
}

/** Yield a template tree's leaf strings, looking through nested bindings. */
function* treeStrings(node: unknown, path: ValuePath): Generator<[ValuePath, string]> {
  if (node instanceof Binding) {
    yield* treeStrings(node.source, path);
  } else if (typeof node === "string") {
    yield [path, node];
  } else if (isRecord(node)) {
    for (const [key, item] of Object.entries(node)) yield* treeStrings(item, [...path, key]);
  } else if (Array.isArray(node)) {
    for (const [index, item] of node.entries()) yield* treeStrings(item, [...path, index]);
  }
}

/**
 * Rebuild a template tree with `fn` applied to each leaf string.
 *
 * Nested bindings keep their contracts; with `mapper` their origins are transformed too. Rebuilt
 * records hold own data properties.
 */
function mapTree(node: unknown, fn: (text: string) => unknown, mapper: OriginMapper | null = null): unknown {
  if (node instanceof Binding) {
    return new Binding(mapTree(node.source, fn, mapper), {
      origin: mapper === null ? node.origin : mapOrigin(node.origin, mapper),
      contract: mapper === null || node.contract === null ? node.contract : node.contract.mapOrigins(mapper),
    });
  }
  if (typeof node === "string") return fn(node);
  if (isRecord(node)) {
    const rebuilt: ValueRecord = {};
    for (const [key, item] of Object.entries(node)) setField(rebuilt, key, mapTree(item, fn, mapper) as Value);
    return rebuilt;
  }
  if (Array.isArray(node)) return node.map((item: unknown) => mapTree(item, fn, mapper));
  return node;
}

/**
 * Yield the template strings of one binding source with their paths.
 *
 * Exactly one `Binding` layer is unwrapped. A plain string yields itself at the empty path; a
 * `ValueTemplate` yields each leaf string, looking through nested bindings. Any other literal,
 * including a plain record or list, yields nothing.
 *
 * @param source A binding or its raw source.
 * @returns `[path, text]` pairs in tree order.
 */
export function* iterSourceStrings(source: unknown): Generator<[ValuePath, string]> {
  const value = bindingSource(source);
  if (typeof value === "string") yield [[], value];
  else if (value instanceof ValueTemplate) yield* treeStrings(value.value, []);
}

/**
 * Rebuild one binding source with `fn` applied to each template string.
 *
 * The strings are those `iterSourceStrings` yields. Bindings, templates, origins, and contracts
 * are rebuilt around the new strings; a source with no template strings is returned unchanged.
 *
 * @param source A binding or its raw source.
 * @param fn Maps one template string to its replacement.
 * @returns The rebuilt source, of the same carrier shape.
 */
export function mapSourceStrings(source: unknown, fn: (text: string) => unknown): unknown {
  if (source instanceof Binding) {
    const value = source.source;
    if (typeof value !== "string" && !(value instanceof ValueTemplate)) return source;
    return new Binding(mapSourceStrings(value, fn), { origin: source.origin, contract: source.contract });
  }
  if (typeof source === "string") return fn(source);
  if (source instanceof ValueTemplate) {
    return new ValueTemplate(mapTree(source.value, fn) as TemplateTree, source.origins);
  }
  return source;
}

/** Escape one path segment as a JSON-pointer reference token. */
function escapeGroupSegment(segment: string | number): string {
  return String(segment).replaceAll("~", "~0").replaceAll("/", "~1");
}

/**
 * Return the input group and template string for each string in a source.
 *
 * A plain string source is one group named `param`. Each leaf string of a `ValueTemplate` is its
 * own group, named `param` followed by its path as a JSON pointer (`~` as `~0`, `/` as `~1`, list
 * indices in decimal), so group names never collide. A source with no template strings has none.
 *
 * @param param The bound parameter's name.
 * @param source The parameter's binding or raw source.
 * @returns `[group, text]` pairs in tree order.
 */
export function bindingInputGroups(param: string, source: unknown): [string, string][] {
  return [...iterSourceStrings(source)].map(([path, text]) => [
    param + path.map((part) => "/" + escapeGroupSegment(part)).join(""),
    text,
  ]);
}

/** An executable assertion and its stable author-facing presentation. Frozen. */
export class Assertion {
  /** Expression evaluated after any lowering rewrite. */
  readonly expression: string;
  /** Original text reported to the author. */
  readonly display: string;
  /** Authored assertion occurrence, when known. */
  readonly origin: OriginRef | null;

  /**
   * Build an assertion; `origin` defaults to `null`.
   *
   * @throws TypeError when `origin` isn't an `OriginRef`.
   */
  constructor(expression: string, display: string, options: { origin?: OriginRef | null } = {}) {
    const { origin = null } = options;
    validateOptionalOrigin(origin, "assertion origin");
    this.expression = expression;
    this.display = display;
    this.origin = origin;
    Object.freeze(this);
  }

  /** Return this assertion with its present origin transformed. */
  mapOrigins(mapper: OriginMapper): Assertion {
    const origin = mapOrigin(this.origin, mapper);
    return new Assertion(this.expression, this.display, { origin });
  }

  /** Yield this assertion's present origin. */
  *iterOrigins(): Generator<OriginRef> {
    yield* iterOriginCarrier(this, new Set());
  }
}

/**
 * Immutable origin index for one value contract and its nested paths. Frozen.
 *
 * `default` exclusively owns the root/fallback location. `entries` is a strictly sorted array of
 * unique, non-empty `[ValuePath, OriginRef]` pairs. String path segments sort before integer
 * segments; a prefix sorts before its descendants.
 */
export class ContractOrigins {
  /** Root/fallback contract site. */
  readonly default: OriginRef | null;
  /** Nested sites in canonical path order. */
  readonly entries: readonly (readonly [ValuePath, OriginRef])[];

  /**
   * Build the index; `default` defaults to `null` and `entries` (copied) to empty.
   *
   * @throws TypeError for a non-origin value, a malformed entry or path.
   * @throws RangeError for an empty, duplicate or out-of-order path.
   */
  constructor(
    options: { default?: OriginRef | null; entries?: readonly (readonly [ValuePath, OriginRef])[] } = {},
  ) {
    const { default: fallback = null, entries = [] } = options;
    validateOptionalOrigin(fallback, "contract default origin");
    if (!Array.isArray(entries)) throw new TypeError("contract origin entries must be an array");

    const paths: ValuePath[] = [];
    for (const entry of entries as unknown[]) {
      if (!Array.isArray(entry) || entry.length !== 2) {
        throw new TypeError("each contract origin entry must be a (path, origin) tuple");
      }
      const [path, origin] = entry as unknown[];
      validatePath(path, "contract origin entry");
      if (path.length === 0) {
        throw new RangeError("contract origin entry path cannot be empty; use default for the root");
      }
      if (!(origin instanceof OriginRef)) throw new TypeError("contract origin entry value must be an OriginRef");
      if (paths.some((seen) => sameItems(seen, path))) {
        throw new RangeError(`duplicate contract origin path: ${valueRepr(path)}`);
      }
      paths.push(path);
    }

    if (paths.some((path, i) => i > 0 && comparePaths(paths[i - 1]!, path) > 0)) {
      throw new RangeError("contract origin entries must be sorted canonically");
    }
    this.default = fallback;
    this.entries = freezeEntries(entries);
    Object.freeze(this);
  }

  /**
   * Select the exact, nearest ancestor, or default origin for `path`.
   *
   * @throws TypeError for a malformed path.
   */
  at(path: ValuePath): OriginRef | null {
    validatePath(path, "contract lookup");
    for (let length = path.length; length > 0; length--) {
      const prefix = path.slice(0, length);
      const entry = this.entries.find(([entryPath]) => sameItems(entryPath, prefix));
      if (entry !== undefined) return entry[1];
    }
    return this.default;
  }

  /** Return this index with every present origin transformed in order. */
  mapOrigins(mapper: OriginMapper): ContractOrigins {
    const fallback = mapOrigin(this.default, mapper);
    const entries = this.entries.map(([path, origin]) => [path, mapOrigin(origin, mapper) as OriginRef] as const);
    return new ContractOrigins({ default: fallback, entries });
  }

  /** Yield the default, then the nested origins in canonical path order. */
  *iterOrigins(): Generator<OriginRef> {
    yield* iterOriginCarrier(this, new Set());
  }
}

/** A semantic runtime type paired with source-neutral nested provenance. Frozen. */
export class ValueContract {
  /** Provenance-free recursive runtime type. */
  readonly type: Type;
  /** Root and nested authored contract sites. */
  readonly origins: ContractOrigins;

  /**
   * Pair `type` with `origins`, a fresh empty index by default.
   *
   * @throws TypeError when `type` isn't a `Type` or `origins` isn't a `ContractOrigins`.
   */
  constructor(type: Type, origins: ContractOrigins = new ContractOrigins()) {
    if (!(type instanceof Type)) throw new TypeError("value contract type must be a Type");
    if (!(origins instanceof ContractOrigins)) throw new TypeError("value contract origins must be ContractOrigins");
    this.type = type;
    this.origins = origins;
    Object.freeze(this);
  }

  /** Return this contract with nested origins transformed, retaining its type. */
  mapOrigins(mapper: OriginMapper): ValueContract {
    return new ValueContract(this.type, this.origins.mapOrigins(mapper));
  }

  /** Yield this contract's root and nested origins in canonical order. */
  *iterOrigins(): Generator<OriginRef> {
    yield* iterOriginCarrier(this, new Set());
  }
}

/**
 * A closed record shape whose individual field types may be unproven. Frozen.
 *
 * Every field name is part of the contract: runtime values must contain exactly this key set. A
 * present `ValueContract` enforces that field recursively; `null` keeps only the field's
 * structural presence. At least one field must be opaque, otherwise one exact record
 * `ValueContract` is canonical.
 */
export class PartialRecordContract {
  /** Ordered field names paired with optional exact value contracts. */
  readonly fields: readonly (readonly [string, ValueContract | null])[];

  /**
   * Require one unique, genuinely partial record declaration; `fields` is copied.
   *
   * @throws TypeError or RangeError as `validateShape` does.
   */
  constructor(fields: readonly (readonly [string, ValueContract | null])[]) {
    this.fields = fields;
    this.validateShape();
    this.fields = Object.freeze(fields.map(([name, contract]) => Object.freeze([name, contract] as const)));
    Object.freeze(this);
  }

  /**
   * Reject malformed or noncanonical field metadata.
   *
   * @throws TypeError when `fields` isn't an array of `[name, contract]` pairs with non-empty
   *   names and exact `ValueContract` or `null` contracts.
   * @throws RangeError for no field, a duplicate name, or no opaque field.
   */
  validateShape(): void {
    if (!Array.isArray(this.fields)) throw new TypeError("partial record fields must be an array");
    if (this.fields.length === 0) throw new RangeError("partial record contract must declare at least one field");
    const names = new Set<string>();
    let hasOpaqueField = false;
    for (const entry of this.fields as unknown[]) {
      if (!Array.isArray(entry) || entry.length !== 2) {
        throw new TypeError("partial record fields must be (name, optional contract) tuples");
      }
      const [name, contract] = entry as unknown[];
      if (typeof name !== "string" || !name) {
        throw new TypeError("partial record field names must be non-empty strings");
      }
      if (names.has(name)) throw new RangeError(`duplicate partial record field ${reprText(name)}`);
      names.add(name);
      if (contract === null) {
        hasOpaqueField = true;
      } else if (!isExactly(contract, ValueContract)) {
        throw new TypeError(
          "partial record field contracts must be exact ValueContract " +
            `values or null, got ${jsTypeName(contract)} for ${reprText(name)}`,
        );
      }
    }
    if (!hasOpaqueField) throw new RangeError("a fully typed record must use one exact ValueContract");
  }

  /** The closed field-name sequence in declaration order. */
  get names(): readonly string[] {
    return this.fields.map(([name]) => name);
  }

  /** Return this contract with every proven field origin transformed. */
  mapOrigins(mapper: OriginMapper): PartialRecordContract {
    return new PartialRecordContract(
      this.fields.map(([name, contract]) => [name, contract === null ? null : contract.mapOrigins(mapper)] as const),
    );
  }

  /** Yield proven field-contract origins in declaration order. */
  *iterOrigins(): Generator<OriginRef> {
    yield* iterOriginCarrier(this, new Set());
  }
}

/** The closed compiled output-contract sum: exact value or partial record. */
export type OutputContract = ValueContract | PartialRecordContract;

/** Whether two origins are both absent or have the same key. */
function sameOrigin(a: OriginRef | null, b: OriginRef | null): boolean {
  return a === null || b === null ? a === b : a.key === b.key;
}

/**
 * Whether two output contracts are equal, as Python's dataclass `==` compares them: `null` only
 * equals `null`; two exact `ValueContract`s by `typeEquals` and the same origins (default, then
 * each entry's path and origin key, in order); two `PartialRecordContract`s field by field, in
 * order.
 */
export function sameOutputContract(a: OutputContract | null, b: OutputContract | null): boolean {
  if (a === null || b === null) return a === b;
  if (isExactly(a, ValueContract) && isExactly(b, ValueContract)) {
    return (
      typeEquals(a.type, b.type) &&
      sameOrigin(a.origins.default, b.origins.default) &&
      sameItems(a.origins.entries, b.origins.entries, ([pathA, originA], [pathB, originB]) => {
        return sameItems(pathA, pathB) && sameOrigin(originA, originB);
      })
    );
  }
  if (isExactly(a, PartialRecordContract) && isExactly(b, PartialRecordContract)) {
    return sameItems(a.fields, b.fields, ([nameA, contractA], [nameB, contractB]) => {
      return nameA === nameB && sameOutputContract(contractA, contractB);
    });
  }
  return false;
}

/**
 * Return an exact whole-value type, validating either contract arm's shapes.
 *
 * @returns The type of a `ValueContract`, else `null`.
 * @throws TypeError when `contract` isn't an exact `ValueContract`, `PartialRecordContract` or
 *   `null`, or a contract's type isn't a `Type`.
 * @throws RangeError when a contract's type metadata or a partial record's fields are malformed.
 */
export function outputContractType(contract: OutputContract | null): Type | null {
  if (contract === null) return null;
  if (isExactly(contract, ValueContract)) {
    if (!(contract.type instanceof Type)) throw new TypeError("value contract type must be a Type");
    contract.type.validateShape();
    return contract.type;
  }
  if (isExactly(contract, PartialRecordContract)) {
    contract.validateShape();
    for (const [, fieldContract] of contract.fields) {
      if (fieldContract !== null) {
        if (!(fieldContract.type instanceof Type)) throw new TypeError("value contract type must be a Type");
        fieldContract.type.validateShape();
      }
    }
    return null;
  }
  throw new TypeError(
    "output contract must be an exact ValueContract, " +
      "PartialRecordContract, or null, " +
      `got ${jsTypeName(contract)}`,
  );
}

/**
 * Return a record of only the `fields` whose type may hold a file, none required.
 *
 * Read through `normalizeFiles`, the record turns each such field's file JSON forms into
 * `FileRef`s and leaves every other field of a value as given.
 *
 * Example: `fileFieldsType([["f", file], ["n", int]])` returns the record `{f: file}` with no
 * required field.
 *
 * @returns The record type, or `null` when no field may hold a file.
 */
export function fileFieldsType(fields: Iterable<readonly [string, Type]>): Type | null {
  const typed = new Map<string, Type>();
  for (const [name, typ] of fields) {
    if (hasFile(typ)) typed.set(name, typ);
  }
  if (typed.size === 0) return null;
  return new Type({ kind: ValueKind.OBJECT, fields: typed, required: new Set() });
}

/**
 * Return the type a value written under `contract` is read through to restore its files.
 *
 * A stored or JSON-decoded value holds each nested file as its JSON form; reading it through this
 * type with `normalizeFiles` gives a `FileRef` at each position the contract types as `file`. A
 * whole-value contract gives its type; a partial record gives `fileFieldsType` of its typed
 * fields, so its opaque fields are read as given. `null` has no file position.
 *
 * Example: for `new ValueContract(list(file))` it returns `list(file)`; for
 * `new ValueContract(int)` it returns `null`.
 *
 * @returns The reading type, or `null` when no position may hold a file.
 */
export function outputContractFileType(contract: OutputContract | null): Type | null {
  if (isExactly(contract, ValueContract)) return hasFile(contract.type) ? contract.type : null;
  if (isExactly(contract, PartialRecordContract)) {
    return fileFieldsType(
      contract.fields.flatMap(([name, field]) => (field === null ? [] : [[name, field.type] as const])),
    );
  }
  return null;
}

// Kinds a field read is not checked on: `object` is opaque unless it declares fields, and an
// `Any` value may be an object.
const UNCHECKED_FIELD_KINDS: ReadonlySet<ValueKind> = new Set([ValueKind.OBJECT, ValueKind.ANY]);

/**
 * Return the error for reading a field only some members of `union` have.
 *
 * `path` is the whole reference path, `unionPath` the path of the union value, and
 * `readUnionPath` its path within `path`. `isinstanceHint` points the fix at an `isinstance`
 * conjunct instead of a case; `tagHint` points it at a tag comparison conjunct instead, when the
 * union is tagged.
 */
function memberOnlyFieldError(
  union: Type,
  having: readonly Type[],
  fieldName: string,
  path: string,
  unionPath: string,
  isinstanceHint: boolean,
  tagHint = false,
  readUnionPath = "",
): string {
  const tag = unionTagFields(union)[0];
  let which: string;
  let fix: string;
  if (tag !== undefined) {
    const tags = having.map((member) => member.fields!.get(tag)!.tags!.values().next().value!).join(", ");
    which = `only union members with ${tag} ${tags} have`;
    fix = `read it inside a case on \${${unionPath}.${tag}}`;
  } else {
    which = `only ${having.length} of ${union.variants!.length} union members have`;
    fix = `check the member with isinstance(${unionPath}, <Type>) in a case first`;
  }
  if (tagHint && tag !== undefined) {
    const value = [...having[0]!.fields!.get(tag)!.tags!].sort(compareCodeUnits)[0]!;
    fix = `check the member first with \`${readUnionPath}.${tag} == "${value}" and ...\``;
  } else if (isinstanceHint && !tagHint) {
    fix = `check the member first with \`isinstance(${unionPath}, <Type>) and ...\``;
  }
  return `reference \${${path}} reads field ${reprText(fieldName)}, which ${which}; ${fix}`;
}

/**
 * Return the error for reading field `fieldName`, which no `file` has.
 *
 * `path` is the whole reference path without `${}`.
 */
export function fileFieldError(path: string, fieldName: string): string {
  return `reference \${${path}} reads field ${reprText(fieldName)} on a file, which has only uri, mime, and name`;
}

/**
 * Return why a reference reads a field its contract can't have, else `null`.
 *
 * An exact record is closed at every proven nested record, and a partial record is closed at its
 * root; reading below an opaque partial field is not checked. A `file` has only `uri`, `mime` and
 * `name`, typed by `fileFieldType`. Every other type but `object` and `Any` (scalars,
 * `Literal(...)`, lists, and their `Optional(...)` forms) has no fields, so reading one is an
 * error. On a union, a field every member has can be read, and the walk continues with its shared
 * type (or stops when the members' types differ); a field no member has is an unknown field, and a
 * field only some members have is an error unless `partialUnionFields` allows it.
 *
 * @param contract The exact or partial contract of the referenced value.
 * @param fields The field names the reference reads below that value. May be empty.
 * @param path The reference path without `${}`, used in the message. Its last `fields.length`
 *   segments are `fields`.
 * @param options.partialUnionFields Whether a field only some union members have can be read
 *   (default `false`). The read is then not checked further.
 * @param options.isinstanceHint Whether a member-only field error tells the author to check the
 *   member with an `isinstance(..., T) and ...` conjunct before the read rather than in a case,
 *   for a condition no case can route (a LOOP predicate or an assert). Default `false`.
 * @param options.tagHint Whether a member-only field error of a tagged union tells the author to
 *   compare the tag (`v.<tag> == "<value>" and ...`) before the read, for a condition where
 *   `isinstance` is not allowed (a CALL or LOOP node's asserts). An untagged union keeps the case
 *   hint. Wins over `isinstanceHint`. Default `false`.
 * @param options.unionRoot The pool path the referenced value is bound from, when `path` reads a
 *   node-local name. A member-only field error's fix then names the union from this path, which a
 *   case can route on, instead of from the local name. Default `null`.
 * @returns The error message, or `null` when the path can be read.
 * @throws TypeError when a field name is not a non-empty string, or `contract` is not an
 *   `OutputContract`.
 * @throws RangeError when a contract's type metadata is malformed.
 */
export function outputContractFieldError(
  contract: OutputContract,
  fields: readonly string[],
  path: string,
  options: {
    partialUnionFields?: boolean;
    isinstanceHint?: boolean;
    tagHint?: boolean;
    unionRoot?: string | null;
  } = {},
): string | null {
  const { partialUnionFields = false, isinstanceHint = false, tagHint = false, unionRoot = null } = options;
  if (!Array.isArray(fields) || (fields as unknown[]).some((field) => typeof field !== "string" || !field)) {
    throw new TypeError("output contract fields must be non-empty strings");
  }
  if (isExactly(contract, PartialRecordContract)) {
    contract.validateShape();
    if (fields.length === 0) return null;
    const [fieldName, ...remaining] = fields as [string, ...string[]];
    const entry = contract.fields.find(([name]) => name === fieldName);
    if (entry === undefined) return `reference \${${path}} reads unknown field ${reprText(fieldName)} on record`;
    const fieldContract = entry[1];
    if (fieldContract === null) return null;
    return outputContractFieldError(fieldContract, remaining, path, {
      partialUnionFields,
      isinstanceHint,
      tagHint,
      unionRoot: unionRoot === null ? null : `${unionRoot}.${fieldName}`,
    });
  }
  if (!isExactly(contract, ValueContract)) {
    outputContractType(contract);
    return null; // unreachable: the validator above always throws
  }

  let typ = outputContractType(contract)!;
  // The path segments walked so far, starting with the ones before `fields`. A negative end
  // drops from the back, as Python's slice does.
  const segments = path.split(".");
  const readWalked = segments.slice(0, segments.length - fields.length);
  const walked = unionRoot === null ? [...readWalked] : [unionRoot];
  const start = walked.length;
  for (const fieldName of fields) {
    if (typ.variants !== undefined) {
      const read = unionField(typ, fieldName);
      if (read.kind === "absent") {
        return `reference \${${path}} reads unknown field ${reprText(fieldName)} on record`;
      }
      if (read.kind === "partial") {
        if (partialUnionFields) return null;
        const readUnionPath = [...readWalked, ...walked.slice(start)].join(".");
        return memberOnlyFieldError(
          typ,
          read.members,
          fieldName,
          path,
          walked.join("."),
          isinstanceHint,
          tagHint,
          readUnionPath,
        );
      }
      if (read.type === null) return null;
      typ = read.type;
      walked.push(fieldName);
      continue;
    }
    if (typ.fields === undefined) {
      if (UNCHECKED_FIELD_KINDS.has(typ.kind)) return null;
      if (typ.kind === ValueKind.FILE) {
        const readType = fileFieldType(fieldName);
        if (readType === null) return fileFieldError(path, fieldName);
        typ = readType;
        walked.push(fieldName);
        continue;
      }
      return (
        `reference \${${path}} reads field ${reprText(fieldName)} on type ` +
        `${canonicalTypeLabel(typ)}, which has no fields`
      );
    }
    const fieldType = typ.fields.get(fieldName);
    if (fieldType === undefined) {
      return `reference \${${path}} reads unknown field ${reprText(fieldName)} on record`;
    }
    typ = fieldType;
    walked.push(fieldName);
  }
  return null;
}

/**
 * Select the most precise authored origin for one output-validation path.
 *
 * @throws TypeError for a malformed path or a non-contract `contract`.
 */
export function outputContractOrigin(contract: OutputContract | null, path: ValuePath): OriginRef | null {
  validatePath(path, "output contract lookup");
  if (contract === null) return null;
  if (isExactly(contract, ValueContract)) return contract.origins.at(path);
  if (!isExactly(contract, PartialRecordContract)) {
    outputContractType(contract);
    return null; // unreachable: the validator above always throws
  }
  const head = path[0];
  if (typeof head !== "string") return null;
  const fieldContract = contract.fields.find(([name]) => name === head)?.[1] ?? null;
  if (fieldContract === null) return null;
  return fieldContract.origins.at(path.slice(1));
}

/**
 * Validate a plain or typed value against either output-contract arm.
 *
 * @param contract The contract to enforce; `null` checks nothing.
 * @param value The plain or typed value to check.
 * @returns The plain value with each typed part normalized; an untyped part or an unchecked value
 *   is returned as given. A partial record is rebuilt with own data properties.
 * @throws TypeCheckError when the value violates the contract.
 */
export function validateOutputContract(contract: OutputContract | null, value: unknown): unknown {
  const exactType = outputContractType(contract);
  if (exactType !== null) return buildValueAs(exactType, value).toObject();
  if (value instanceof TypedValue) value = value.toObject();
  if (!isExactly(contract, PartialRecordContract)) return value;
  if (!isRecord(value)) {
    throw new TypeCheckError(`${valueRepr(value)} is not an object for a partial record contract`);
  }

  const expected = new Set(contract.names);
  const missing = contract.names.filter((name) => !Object.hasOwn(value, name));
  if (missing.length > 0) {
    throw new TypeCheckError(`record missing required fields: ${writeAsText(missing)}`, { path: [missing[0]!] });
  }
  const unknown = Object.keys(value)
    .filter((name) => !expected.has(name))
    .sort(compareCodeUnits);
  if (unknown.length > 0) {
    const shown = shownText(unknown.map((name) => JSON.stringify(name)).join(","));
    throw new TypeCheckError(`record has unknown fields: [${shown}]`, { path: [unknown[0]!] });
  }
  const checked: ValueRecord = {};
  for (const name of Object.keys(value)) setField(checked, name, value[name] as Value);
  for (const [name, fieldContract] of contract.fields) {
    if (fieldContract === null) continue;
    try {
      setField(checked, name, buildValueAs(fieldContract.type, value[name]).toObject());
    } catch (error) {
      if (!(error instanceof TypeCheckError)) throw error;
      throw new TypeCheckError(error.message, { path: [name, ...error.path], cause: error });
    }
  }
  return checked;
}

/** One direct tool call a node makes, as the run boundary checks it. Frozen. */
export class HostToolCall {
  /** The id of the tool the node invokes. */
  readonly toolId: string;
  /** The names of the inputs the node passes, in declaration order. */
  readonly inputs: readonly string[];
  /** The proven type of each input's binding; an input whose type is not proven is absent. */
  readonly inputTypes: ReadonlyMap<string, Type>;
  /** The node's declared whole-value output type, or `null` when it has none. */
  readonly outputType: Type | null;

  /** Describe one tool call; `inputs` is copied and `inputTypes` retained. */
  constructor(
    toolId: string,
    inputs: readonly string[],
    inputTypes: ReadonlyMap<string, Type>,
    outputType: Type | null,
  ) {
    this.toolId = toolId;
    this.inputs = Object.freeze([...inputs]);
    this.inputTypes = inputTypes;
    this.outputType = outputType;
    Object.freeze(this);
  }
}

/**
 * The durable steps one start of a node reads and records, from the engine. Frozen.
 *
 * The engine passes it as `caps.steps` to a node declaring `needsSteps`. A step is a JSON object
 * with a string `kind`, which the node gives meaning to; the engine appends each one to the run
 * log and keeps it until the node settles, so a resumed node reads the steps its earlier starts
 * recorded. Steps are numbered from 1 across the node's starts: `n`, the step's id within the node,
 * which `record` returns.
 *
 * Example:
 * ```ts
 * if (steps !== undefined) {
 *   const attempt = steps.starts + steps.prior.length;
 *   steps.record({ kind: "tool_attempt", event: "started", attempt: BigInt(attempt) });
 * }
 * ```
 */
export class NodeSteps {
  /**
   * Accepted starts of this node, including the current one; at least 1. A resume of an
   * unfinished start counts as a new start.
   */
  readonly starts: number;
  /**
   * Copies of the steps earlier starts recorded, in record order. Changing them does not change
   * the run.
   */
  readonly prior: readonly ValueRecord[];
  /**
   * Appends one step durably and returns its `n` once the engine has applied it. Call it from
   * the node's own evaluation, one call at a time, before the node returns; after the evaluation
   * ends it throws. Once the run stops it throws the node signal's reason and appends nothing,
   * except a step passed with `{ settled: true }` while the run has no terminal result.
   *
   * The engine can't see whether a call settled before the stop, so this is the node's
   * obligation: pass `settled: true`, and emit `ToolCallFinished` after a stop, only for a call
   * whose effect settled while the node's signal was live, recorded synchronously in that call's
   * `.then`. An error from `record` does not prove the step was not recorded: a stop can land
   * mid-commit.
   */
  readonly record: (step: ValueRecord, options?: { settled?: boolean }) => number;

  /**
   * Check the field types and that the current start is counted; `prior` is copied.
   *
   * @throws TypeError when `starts` isn't a safe integer, `prior` isn't an array of plain
   *   records, or `record` isn't a function.
   * @throws RangeError when `starts` is less than 1.
   */
  constructor(
    starts: number,
    prior: readonly ValueRecord[],
    record: (step: ValueRecord, options?: { settled?: boolean }) => number,
  ) {
    if (typeof starts !== "number" || !Number.isSafeInteger(starts)) throw new TypeError("starts must be an int");
    if (starts < 1) throw new RangeError("starts must be at least 1");
    if (!Array.isArray(prior) || !prior.every((step: unknown) => isRecord(step))) {
      throw new TypeError("prior must be an array of records");
    }
    if (typeof record !== "function") throw new TypeError("record must be callable");
    this.starts = starts;
    this.prior = Object.freeze([...prior]);
    this.record = record;
    Object.freeze(this);
  }
}

/** The optional fields of a `Parameter`. */
interface ParameterOptions {
  contract?: ValueContract | null;
  required?: boolean;
  hasDefault?: boolean;
  default?: unknown;
  coerceStrings?: boolean;
  origin?: OriginRef | null;
  defaultOrigin?: OriginRef | null;
}

/**
 * One named declared value with an optional contract and default policy. Frozen.
 *
 * Formal inputs and node-local variables share this carrier; their owning construct decides
 * whether a contract or default is required.
 *
 * `hasDefault` distinguishes an explicit `null` default from no default. `coerceStrings` is
 * run-boundary policy derived from `contract.type` by `coerceInputs`, which applies it only to
 * the top-level flow's supplied inputs; there is deliberately no parallel textual type field.
 */
export class Parameter {
  /** Declared name in the owner's local record. */
  readonly name: string;
  /** Optional semantic value contract. */
  readonly contract: ValueContract | null;
  /** Whether omission is invalid. */
  readonly required: boolean;
  /** Whether `default` is semantically present. */
  readonly hasDefault: boolean;
  /** Retained default payload; meaningful only when present. */
  readonly default: unknown;
  /** Whether the run boundary may convert supplied text. */
  readonly coerceStrings: boolean;
  /** Parameter declaration site. */
  readonly origin: OriginRef | null;
  /** Explicit default site, when present. */
  readonly defaultOrigin: OriginRef | null;

  /**
   * Declare `name`, enforcing one unambiguous representation of required/default state. The
   * flags default to `false`, the rest to `null`.
   *
   * @throws TypeError when `contract` isn't a `ValueContract` or an origin isn't an `OriginRef`.
   * @throws RangeError for a required parameter with a default, or a default or default origin
   *   without `hasDefault`.
   */
  constructor(name: string, options: ParameterOptions = {}) {
    const {
      contract = null,
      required = false,
      hasDefault = false,
      default: fallback = null,
      coerceStrings = false,
      origin = null,
      defaultOrigin = null,
    } = options;
    if (contract !== null && !(contract instanceof ValueContract)) {
      throw new TypeError("parameter contract must be a ValueContract or null");
    }
    validateOptionalOrigin(origin, "parameter origin");
    validateOptionalOrigin(defaultOrigin, "parameter default origin");
    if (required && hasDefault) throw new RangeError("a required parameter cannot have a default");
    if (!hasDefault && fallback !== null) throw new RangeError("parameter default requires has_default=true");
    if (!hasDefault && defaultOrigin !== null) {
      throw new RangeError("parameter default_origin requires has_default=true");
    }
    this.name = name;
    this.contract = contract;
    this.required = required;
    this.hasDefault = hasDefault;
    this.default = fallback;
    this.coerceStrings = coerceStrings;
    this.origin = origin;
    this.defaultOrigin = defaultOrigin;
    Object.freeze(this);
  }

  /** Transform parameter, default, and nested contract origins without copying data. */
  mapOrigins(mapper: OriginMapper): Parameter {
    const contract = this.contract !== null ? this.contract.mapOrigins(mapper) : null;
    const origin = mapOrigin(this.origin, mapper);
    const defaultOrigin = mapOrigin(this.defaultOrigin, mapper);
    return new Parameter(this.name, {
      contract,
      required: this.required,
      hasDefault: this.hasDefault,
      default: this.default,
      coerceStrings: this.coerceStrings,
      origin,
      defaultOrigin,
    });
  }

  /** Yield contract, declaration, then default origins in stable order. */
  *iterOrigins(): Generator<OriginRef> {
    yield* iterOriginCarrier(this, new Set());
  }
}

/** A carrier that holds origins. */
export type OriginCarrier =
  | Binding
  | ValueTemplate
  | Assertion
  | ContractOrigins
  | ValueContract
  | PartialRecordContract
  | Parameter;

/**
 * Yield one frozen carrier graph once, preserving its semantic origin order. Used by node kinds.
 *
 * @param seen Carriers already visited by this traversal; a carrier in it yields nothing.
 * @throws TypeError for an unknown carrier.
 */
export function* iterOriginCarrier(carrier: OriginCarrier, seen: Set<object>): Generator<OriginRef> {
  if (seen.has(carrier)) return;
  seen.add(carrier);

  if (carrier instanceof Binding) {
    if (carrier.origin !== null) yield carrier.origin;
    if (carrier.contract !== null) yield* iterOriginCarrier(carrier.contract, seen);
    if (carrier.source instanceof ValueTemplate) yield* iterOriginCarrier(carrier.source, seen);
    return;
  }
  if (carrier instanceof ValueTemplate) {
    for (const [, origin] of carrier.origins) yield origin;
    for (const binding of treeBindings(carrier.value)) yield* iterOriginCarrier(binding, seen);
    return;
  }
  if (carrier instanceof Assertion) {
    if (carrier.origin !== null) yield carrier.origin;
    return;
  }
  if (carrier instanceof ContractOrigins) {
    if (carrier.default !== null) yield carrier.default;
    for (const [, origin] of carrier.entries) yield origin;
    return;
  }
  if (carrier instanceof ValueContract) {
    yield* iterOriginCarrier(carrier.origins, seen);
    return;
  }
  if (carrier instanceof PartialRecordContract) {
    for (const [, contract] of carrier.fields) {
      if (contract !== null) yield* iterOriginCarrier(contract, seen);
    }
    return;
  }
  if (carrier instanceof Parameter) {
    if (carrier.contract !== null) yield* iterOriginCarrier(carrier.contract, seen);
    if (carrier.origin !== null) yield carrier.origin;
    if (carrier.defaultOrigin !== null) yield carrier.defaultOrigin;
    return;
  }
  throw new TypeError(`unsupported origin carrier: ${jsTypeName(carrier)}`);
}

/**
 * Return `carrier` rebuilt with every origin transformed by `mapper`. Used by node kinds.
 *
 * The carriers are frozen, so the rebuild is the private copy; the input is unchanged. Each
 * carrier is rebuilt once per `seen` (old to new), so a carrier reached twice comes back as one
 * shared object: pass one `seen` for a whole graph. A binding's source is rebuilt the same way
 * wherever the binding is reached: a `ValueTemplate` through `seen`; a record or list holding a
 * binding by rebuilding the containers around each rewritten binding; any other source, and any
 * container holding no binding, by identity. A template's tree is rebuilt the same way. The
 * carrier graph and the payloads it walks must have no cycles; compose never builds one.
 *
 * @param seen Carriers and payload containers already rewritten, each mapped to its rebuilt form
 *   (itself when unchanged).
 * @throws TypeError for an unknown carrier or a mapper result that isn't an `OriginRef`.
 */
export function rewriteOriginCarrier<C extends OriginCarrier>(
  carrier: C,
  mapper: OriginMapper,
  seen: Map<object, object>,
): C {
  const done = seen.get(carrier);
  if (done !== undefined) return done as C;

  let rebuilt: OriginCarrier;
  if (carrier instanceof Binding) {
    const origin = mapOrigin(carrier.origin, mapper);
    const contract = carrier.contract === null ? null : rewriteOriginCarrier(carrier.contract, mapper, seen);
    const source =
      carrier.source instanceof ValueTemplate
        ? rewriteOriginCarrier(carrier.source, mapper, seen)
        : rewriteTree(carrier.source, mapper, seen);
    rebuilt = new Binding(source, { origin, contract });
  } else if (carrier instanceof ValueTemplate) {
    const origins = carrier.origins.map(([path, origin]) => [path, mapOrigin(origin, mapper) as OriginRef] as const);
    rebuilt = new ValueTemplate(rewriteTree(carrier.value, mapper, seen) as TemplateTree, origins);
  } else if (carrier instanceof Assertion) {
    rebuilt = carrier.mapOrigins(mapper);
  } else if (carrier instanceof ContractOrigins) {
    rebuilt = carrier.mapOrigins(mapper);
  } else if (carrier instanceof ValueContract) {
    rebuilt = new ValueContract(carrier.type, rewriteOriginCarrier(carrier.origins, mapper, seen));
  } else if (carrier instanceof PartialRecordContract) {
    rebuilt = new PartialRecordContract(
      carrier.fields.map(
        ([name, contract]) => [name, contract === null ? null : rewriteOriginCarrier(contract, mapper, seen)] as const,
      ),
    );
  } else if (carrier instanceof Parameter) {
    const contract = carrier.contract === null ? null : rewriteOriginCarrier(carrier.contract, mapper, seen);
    rebuilt = new Parameter(carrier.name, {
      contract,
      required: carrier.required,
      hasDefault: carrier.hasDefault,
      default: carrier.default,
      coerceStrings: carrier.coerceStrings,
      origin: mapOrigin(carrier.origin, mapper),
      defaultOrigin: mapOrigin(carrier.defaultOrigin, mapper),
    });
  } else {
    throw new TypeError(`unsupported origin carrier: ${jsTypeName(carrier)}`);
  }
  seen.set(carrier, rebuilt);
  return rebuilt as C;
}

/**
 * Rebuild the records and lists of `node` around each nested binding rewritten through `seen`;
 * a container holding no binding, and any other value, is kept by identity. Each container is
 * walked once per `seen`, so a container reached twice comes back as one shared object.
 */
function rewriteTree(node: unknown, mapper: OriginMapper, seen: Map<object, object>): unknown {
  if (node instanceof Binding) return rewriteOriginCarrier(node, mapper, seen);
  if (!isRecord(node) && !Array.isArray(node)) return node;
  const done = seen.get(node);
  if (done !== undefined) return done;

  let rebuilt: object = node;
  if (Array.isArray(node)) {
    const items = node.map((item: unknown) => rewriteTree(item, mapper, seen));
    if (items.some((item, i) => item !== node[i])) rebuilt = items;
  } else {
    const entries = Object.entries(node).map(([key, item]) => [key, rewriteTree(item, mapper, seen)] as const);
    if (entries.some(([key, item]) => item !== node[key])) {
      const record: ValueRecord = {};
      for (const [key, item] of entries) setField(record, key, item as Value);
      rebuilt = record;
    }
  }
  seen.set(node, rebuilt);
  return rebuilt;
}

/** Return `value` unchanged when it is a `Binding`, otherwise wrap it origin-free. */
export function asBinding(value: unknown): Binding {
  return value instanceof Binding ? value : new Binding(value);
}

/** Unwrap exactly one `Binding` layer, or return a raw value unchanged. */
export function bindingSource(value: unknown): unknown {
  return value instanceof Binding ? value.source : value;
}
