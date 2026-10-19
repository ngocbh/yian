/**
 * Source-neutral value types and checks for injected workflow memory.
 *
 * This module holds the memory JSON snapshot (exact, finite, well-formed JSON copied into frozen
 * arrays and objects), the rejection of host-private config keys (credentials, storage locations,
 * Python import paths), and the per-declaration shape check `checkMemoryDeclaration`, with the
 * built-in types' config tables. It also holds the frozen identity values compile records:
 * declaration sites, declarations and their origin rewrite, node scopes with their occurrence
 * paths, memory specs and stacks, and the setup error. It imports only `diagnostics` and
 * `defaults`. The runtime memory, recall, learning and conversation contracts are not ported yet.
 */

import { current } from "../defaults.ts";
import { compareCodeUnits, OriginRef, reprText } from "../diagnostics.ts";

/** One JSON scalar: an int is a `bigint`, a float a finite `number`. */
export type JsonScalar = null | boolean | bigint | number | string;
/** One JSON value. */
export type JsonValue = JsonScalar | readonly JsonValue[] | JsonObject;
/** One JSON object. */
export interface JsonObject {
  readonly [key: string]: JsonValue;
}

/**
 * Return one string identifier.
 *
 * @throws TypeError when `value` isn't a string.
 * @throws RangeError when it's empty and `nonempty` is set, or holds a lone surrogate.
 */
function requireString(value: unknown, name: string, nonempty = true): string {
  if (typeof value !== "string") throw new TypeError(`${name} must be a string`);
  if (nonempty && !value) throw new RangeError(`${name} must be non-empty`);
  if (!value.isWellFormed()) throw new RangeError("Unicode surrogate is not valid canonical JSON");
  return value;
}

/** Whether `value` is a plain array (not a subclass). */
function isPlainArray(value: unknown): value is unknown[] {
  return Array.isArray(value) && Object.getPrototypeOf(value) === Array.prototype;
}

/** Whether `value` is a plain object: its prototype is `Object.prototype` or `null`. */
function isPlainObject(value: unknown): value is Record<string, unknown> {
  if (typeof value !== "object" || value === null) return false;
  const proto: unknown = Object.getPrototypeOf(value);
  return proto === Object.prototype || proto === null;
}

/**
 * Return a frozen copy of one path whose components are non-empty strings.
 *
 * @throws TypeError when `value` isn't an array or a component isn't a string.
 * @throws RangeError when the path is empty and `allowEmpty` isn't set, or a component is empty
 *   or holds a lone surrogate.
 */
function requirePath(value: unknown, name: string, allowEmpty = false): readonly string[] {
  if (!isPlainArray(value)) throw new TypeError(`${name} must be an array`);
  if (!allowEmpty && value.length === 0) throw new RangeError(`${name} must be non-empty`);
  for (const part of value) requireString(part, `${name} components`);
  return Object.freeze([...(value as string[])]);
}

/** Validate and copy one JSON value into frozen containers; see `freezeJsonObject`. */
function freezeJson(value: unknown, name: string, active: Set<object>, nesting: number, limit: number): JsonValue {
  if (value === null || typeof value === "boolean" || typeof value === "bigint") return value;
  if (typeof value === "string") return requireString(value, name, false);
  if (typeof value === "number") {
    if (!Number.isFinite(value)) throw new RangeError(`${name} JSON number must be finite`);
    return value;
  }
  const array = isPlainArray(value);
  if (!array && !isPlainObject(value)) throw new TypeError(`${name} must contain exact JSON values`);
  if (nesting >= limit) {
    throw new RangeError(
      `${name} nests deeper than ${limit} levels, the limit for a yian value (each record and list is a level)`,
    );
  }
  // Only the containers on the current path count, so a subtree reached twice is not a cycle.
  if (active.has(value)) throw new RangeError(`${name} contains a circular JSON value`);
  active.add(value);
  try {
    if (array) {
      const items: JsonValue[] = [];
      // By index, so a hole reads `undefined` and is refused.
      for (let i = 0; i < value.length; i++) {
        items.push(freezeJson(value[i], `${name}[${i}]`, active, nesting + 1, limit));
      }
      return Object.freeze(items);
    }
    if (Object.getOwnPropertySymbols(value).length > 0) {
      throw new TypeError(`${name} JSON object keys must be strings`);
    }
    const keys = Object.keys(value);
    for (const key of keys) requireString(key, `${name} JSON object key`, false);
    const entries = keys
      .sort(compareCodeUnits)
      .map((key) => [key, freezeJson(value[key], `${name}.${key}`, active, nesting + 1, limit)] as const);
    // `Object.fromEntries` defines own data properties, so a `__proto__` key stays data.
    return Object.freeze(Object.fromEntries(entries));
  } finally {
    active.delete(value);
  }
}

/**
 * Validate and snapshot one exact JSON object: `null`, booleans, `bigint` ints, finite `number`
 * floats, well-formed strings, plain arrays and plain objects with string keys, copied into frozen
 * arrays and frozen objects whose keys are inserted in code-unit order. With `limitNesting` (the
 * default, for configs) it nests at most `current().maxValueNesting` container levels.
 *
 * @throws TypeError when `value` isn't a plain object, holds a non-JSON value, or has a symbol key.
 * @throws RangeError for a non-finite number, a lone surrogate, a cycle, or nesting past the limit.
 */
function freezeJsonObject(value: unknown, name: string, options: { limitNesting?: boolean } = {}): JsonObject {
  if (!isPlainObject(value)) throw new TypeError(`${name} must be an exact JSON object`);
  const limit = (options.limitNesting ?? true) ? current().maxValueNesting : Number.POSITIVE_INFINITY;
  return freezeJson(value, name, new Set(), 0, limit) as JsonObject;
}

/** Return `vocabulary` with the predictable plural form of each term added. */
function withHighConfidencePlurals(vocabulary: readonly string[]): ReadonlySet<string> {
  const terms = new Set(vocabulary);
  for (const item of vocabulary) {
    if (item.endsWith("class")) terms.add(`${item}es`);
    else if (item.endsWith("directory")) terms.add(`${item.slice(0, -1)}ies`);
    else if (!item.endsWith("s")) terms.add(`${item}s`);
  }
  return terms;
}

const PRIVATE_CREDENTIAL_KEYS: ReadonlySet<string> = new Set([
  "api_key",
  "access_token",
  "auth",
  "auth_token",
  "authorization",
  "basic_auth",
  "backend_token",
  "bearer_token",
  "client_secret",
  "connection_string",
  "credential",
  "credentials",
  "databricks_token",
  "docker_auth_config",
  "github_token",
  "google_application_credentials",
  "hf_token",
  "id_token",
  "mysql_pwd",
  "npm_token",
  "password",
  "passphrase",
  "passwd",
  "pgpassword",
  "private_key",
  "proxy_auth",
  "pwd",
  "refresh_token",
  "secret",
  "secret_access_key",
  "secret_key",
  "session_token",
  "shared_access_key",
  "shared_access_signature",
  "sig",
  "signature",
  "slack_bot_token",
  "sslpassword",
  "token",
  "webhook_signature",
  "webhook_url",
  "x_amz_security_token",
  "x_amz_signature",
  "x_goog_signature",
]);
const PRIVATE_CREDENTIAL_SUFFIXES = withHighConfidencePlurals([
  "account_key",
  "access_key",
  "access_key_id",
  "access_token",
  "api_key",
  "api_token",
  "application_credentials",
  "auth_key",
  "auth_token",
  "authorization",
  "bearer_key",
  "bearer_token",
  "connection_string",
  "credential",
  "credentials",
  "password",
  "passphrase",
  "private_key",
  "refresh_key",
  "refresh_token",
  "secret",
  "secret_access_key",
  "secret_key",
  "secret_token",
  "session_cookie",
  "session_key",
  "session_token",
  "shared_access_key",
  "shared_access_signature",
  "subscription_key",
  "webhook_url",
]);
const PRIVATE_TENANT_KEYS: ReadonlySet<string> = new Set(["tenant_id"]);
const PRIVATE_PYTHON_KEYS: ReadonlySet<string> = new Set([
  "class",
  "class_path",
  "import",
  "import_path",
  "imports",
  "module",
  "module_path",
  "python_class",
  "python_import",
  "python_module",
  "python_path",
]);
const PRIVATE_PYTHON_SUFFIXES = withHighConfidencePlurals([
  "class_path",
  "import_path",
  "imports",
  "module_path",
  "python_path",
]);
const PRIVATE_LOCATION_ROOTS = withHighConfidencePlurals(["backend", "database", "persistence", "storage"]);
const PRIVATE_LOCATION_FIELDS = withHighConfidencePlurals([
  "connection",
  "connection_string",
  "dir",
  "directory",
  "dsn",
  "file",
  "location",
  "path",
  "root",
  "uri",
  "url",
]);
const PRIVATE_INDIRECTION_FIELDS = withHighConfidencePlurals(["dir", "directory", "file", "path"]);
const PRIVATE_WEBHOOK_FIELDS: ReadonlySet<string> = new Set(["signature", "url"]);
const PRIVATE_PYTHON_FIELDS = withHighConfidencePlurals(["class", "import", "module", "path"]);

// Single code points tested the way Python's `str.isalnum`, `isupper`, `islower` and `isdigit` do.
const ALNUM = /^[\p{L}\p{N}]$/u;
const UPPER = /^\p{Uppercase}$/u;
const LOWER = /^\p{Lowercase}$/u;
// `isdigit` is `\p{Nd}` plus Numeric_Type=Digit (superscripts, circled digits), pinned from CPython
// 3.12's `[c for c in map(chr, range(0x110000)) if c.isdigit() and not c.isdecimal()]`.
const DIGIT =
  /^[\p{Nd}\u{B2}-\u{B3}\u{B9}\u{1369}-\u{1371}\u{19DA}\u{2070}\u{2074}-\u{2079}\u{2080}-\u{2089}\u{2460}-\u{2468}\u{2474}-\u{247C}\u{2488}-\u{2490}\u{24EA}\u{24F5}-\u{24FD}\u{24FF}\u{2776}-\u{277E}\u{2780}-\u{2788}\u{278A}-\u{2792}\u{10A40}-\u{10A43}\u{10E60}-\u{10E68}\u{11052}-\u{1105A}\u{1F100}-\u{1F10A}]$/u;

/**
 * Return one code point folded as Python's `str.casefold` does, for the private-key vocabularies.
 * Upper-then-lower case differs from `casefold` only for `ı` (U+0131, kept), `ẞ` (U+1E9E, `ss`)
 * and Cherokee letters; Cherokee folds to non-ASCII either way, so the ASCII vocabularies match
 * the same keys.
 */
function casefold(ch: string): string {
  if (ch === "\u0131") return ch;
  if (ch === "\u1E9E") return "ss";
  return ch.toUpperCase().toLowerCase();
}

/** Return one case-, camel- and punctuation-neutral authored key: `apiKey` and `API-Key` give `api_key`. */
function normalizeConfigKey(value: string): string {
  // By code point, as Python indexes a `str`.
  const chars = Array.from(value);
  const normalized: string[] = [];
  for (let i = 0; i < chars.length; i++) {
    const ch = chars[i]!;
    if (!ALNUM.test(ch)) {
      if (normalized.length > 0 && normalized.at(-1) !== "_") normalized.push("_");
      continue;
    }
    const previous = i > 0 ? chars[i - 1]! : "";
    const following = i + 1 < chars.length ? chars[i + 1]! : "";
    if (
      UPPER.test(ch) &&
      normalized.length > 0 &&
      normalized.at(-1) !== "_" &&
      (LOWER.test(previous) || DIGIT.test(previous) || (UPPER.test(previous) && LOWER.test(following)))
    ) {
      normalized.push("_");
    }
    normalized.push(casefold(ch));
  }
  return normalized.join("").replace(/^_+|_+$/g, "");
}

/** Whether one normalized key equals a private key, with or without underscores. */
function matchesExactPrivateKey(value: string, vocabulary: ReadonlySet<string>): boolean {
  const compact = value.replaceAll("_", "");
  for (const item of vocabulary) {
    if (value === item || compact === item.replaceAll("_", "")) return true;
  }
  return false;
}

/** Whether one normalized key equals or ends in a private term. */
function matchesPrivateSuffix(value: string, vocabulary: ReadonlySet<string>): boolean {
  const compact = value.replaceAll("_", "");
  for (const item of vocabulary) {
    if (value === item || value.endsWith(`_${item}`) || compact.endsWith(item.replaceAll("_", ""))) return true;
  }
  return false;
}

/** Whether one normalized key names a file or path holding credentials (`api_key_file`). */
function matchesPrivateCredentialIndirection(value: string): boolean {
  const compact = value.replaceAll("_", "");
  for (const field of PRIVATE_INDIRECTION_FIELDS) {
    const suffix = `_${field}`;
    let stem: string;
    if (value.endsWith(suffix)) {
      stem = value.slice(0, -suffix.length);
    } else {
      const compactField = field.replaceAll("_", "");
      if (!compact.endsWith(compactField)) continue;
      stem = compact.slice(0, -compactField.length);
    }
    if (
      stem &&
      (matchesExactPrivateKey(stem, PRIVATE_CREDENTIAL_KEYS) || matchesPrivateSuffix(stem, PRIVATE_CREDENTIAL_SUFFIXES))
    ) {
      return true;
    }
  }
  return false;
}

/**
 * Reject recognized host-private keys anywhere in an authored memory config snapshot.
 *
 * @throws RangeError naming the first forbidden key in code-unit key order.
 */
function rejectPrivateMemoryKeys(
  value: JsonValue,
  name: string,
  locationContext = false,
  pythonContext = false,
  webhookContext = false,
): void {
  if (Array.isArray(value)) {
    value.forEach((item: JsonValue, i) =>
      rejectPrivateMemoryKeys(item, `${name}[${i}]`, locationContext, pythonContext, webhookContext),
    );
    return;
  }
  if (value === null || typeof value !== "object") return;
  const object = value as JsonObject;
  // Sorted again because enumeration puts integer-like keys first, unlike Python's sorted snapshot.
  for (const key of Object.keys(object).sort(compareCodeUnits)) {
    const normalized = normalizeConfigKey(key);
    const tokens = normalized.split("_");
    const privateLocation = locationContext || tokens.some((token) => PRIVATE_LOCATION_ROOTS.has(token));
    const privatePython = pythonContext || tokens.includes("python");
    const privateWebhook = webhookContext || tokens.includes("webhook");
    const forbidden =
      matchesExactPrivateKey(normalized, PRIVATE_CREDENTIAL_KEYS) ||
      matchesPrivateSuffix(normalized, PRIVATE_CREDENTIAL_SUFFIXES) ||
      matchesPrivateSuffix(normalized, PRIVATE_TENANT_KEYS) ||
      matchesExactPrivateKey(normalized, PRIVATE_PYTHON_KEYS) ||
      matchesPrivateSuffix(normalized, PRIVATE_PYTHON_SUFFIXES) ||
      matchesPrivateCredentialIndirection(normalized) ||
      (privateLocation && matchesPrivateSuffix(normalized, PRIVATE_LOCATION_FIELDS)) ||
      (privatePython && matchesPrivateSuffix(normalized, PRIVATE_PYTHON_FIELDS)) ||
      (privateWebhook && matchesPrivateSuffix(normalized, PRIVATE_WEBHOOK_FIELDS));
    if (forbidden) throw new RangeError(`${name} contains forbidden private key ${reprText(key)}`);
    rejectPrivateMemoryKeys(object[key]!, `${name}.${key}`, privateLocation, privatePython, privateWebhook);
  }
}

/**
 * Require one optional origin, kept by identity (an `OriginRef` subclass instance too).
 *
 * @throws TypeError when `value` is neither `null` nor an `OriginRef`.
 */
function requireOrigin(value: unknown, name: string): OriginRef | null {
  if (value !== null && !(value instanceof OriginRef)) throw new TypeError(`${name} must be an OriginRef or null`);
  return value;
}

/** Return one JSON value as compact JSON text with JS number spelling, for messages. */
function jsonText(value: JsonValue): string {
  if (value === null) return "null";
  if (typeof value === "boolean" || typeof value === "bigint" || typeof value === "number") return String(value);
  if (typeof value === "string") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map((item: JsonValue) => jsonText(item)).join(",")}]`;
  const object = value as JsonObject;
  return `{${Object.keys(object)
    .map((key) => `${JSON.stringify(key)}:${jsonText(object[key]!)}`)
    .join(",")}}`;
}

/** Return `names` as Python's repr of a list of strings: `['a', 'b']`. */
function reprList(names: readonly string[]): string {
  return `[${names.map(reprText).join(", ")}]`;
}

/** Grammar shared by memory names, memory type names and `uses:` scope segments. */
export const MEMORY_IDENTIFIER_PATTERN = /^[A-Za-z_][A-Za-z0-9_-]*$/;

const LOCATION_CONFIG_KEYS: ReadonlySet<string> = new Set(["storage", "store", "path"]);

/** Config keys each built-in memory type accepts from YAML. */
export const BUILTIN_MEMORY_CONFIG_KEYS: ReadonlyMap<string, ReadonlySet<string>> = new Map([
  ["episodic", new Set(["max_episodes", "recall_limit"])],
  ["reflective", new Set(["max_items", "recall_limit"])],
  ["session", new Set(["history_budget_chars", "recent_turns"])],
  ["profile", new Set(["max_chars", "input_budget_chars"])],
]);

/**
 * Per built-in key, in the order checked: the least integer allowed and whether null is allowed.
 *
 * These mirror the backend constructors so a bad value fails at load, not first when the host
 * builds the memories; null leaves an episodic or reflective limit to the stored or default value.
 * Null is refused for the session and profile keys, although their constructors take it as the
 * yian default; leaving a key out has that effect.
 */
const BUILTIN_MEMORY_CONFIG_RULES: ReadonlyMap<string, readonly (readonly [string, bigint, boolean])[]> = new Map([
  [
    "episodic",
    [
      ["max_episodes", 1n, true],
      ["recall_limit", 1n, true],
    ],
  ],
  [
    "reflective",
    [
      ["max_items", 1n, true],
      ["recall_limit", 1n, true],
    ],
  ],
  [
    "session",
    [
      ["history_budget_chars", 1n, false],
      ["recent_turns", 0n, false],
    ],
  ],
  [
    "profile",
    [
      ["max_chars", 1n, false],
      ["input_budget_chars", 1n, false],
    ],
  ],
]);

/**
 * Throw when a built-in type's config value breaks its backend's rule.
 *
 * @throws TypeError when a value isn't a `bigint` (or null, where allowed).
 * @throws RangeError when a value is below its minimum.
 */
function checkBuiltinConfigValues(typeName: string, config: JsonObject): void {
  for (const [key, minimum, nullable] of BUILTIN_MEMORY_CONFIG_RULES.get(typeName)!) {
    if (!Object.hasOwn(config, key) || (nullable && config[key] === null)) continue;
    const value = config[key]!;
    if (typeof value !== "bigint") {
      const expected = nullable ? "an integer or null" : "an integer";
      throw new TypeError(`${typeName} memory config ${key} must be ${expected}, got ${jsonText(value)}`);
    }
    if (value < minimum) {
      throw new RangeError(`${typeName} memory config ${key} must be an integer of at least ${minimum}, got ${value}`);
    }
  }
}

/**
 * Return a memory name or type name that matches `MEMORY_IDENTIFIER_PATTERN`.
 *
 * @throws TypeError when `value` isn't a string.
 * @throws RangeError when it doesn't match.
 */
function requireMemoryIdentifier(value: unknown, what: string): string {
  if (typeof value !== "string") throw new TypeError(`${what} must be a string`);
  if (!MEMORY_IDENTIFIER_PATTERN.test(value)) {
    throw new RangeError(`${what} ${reprText(value)} must match [A-Za-z_][A-Za-z0-9_-]*`);
  }
  return value;
}

/** Check one declaration as `checkMemoryDeclaration` does and return its frozen config snapshot. */
function checkedConfig(name: unknown, typeName: unknown, config: unknown): JsonObject {
  requireMemoryIdentifier(name, "memory name");
  const type = requireMemoryIdentifier(typeName, "memory type");
  const frozen = freezeJsonObject(config, "config");
  rejectPrivateMemoryKeys(frozen, "config");
  // The caller's key order, as Python iterates the authored dict.
  for (const key of Object.keys(config as object)) {
    if (LOCATION_CONFIG_KEYS.has(key)) {
      throw new RangeError(
        `config key ${reprText(key)} is not allowed: storage belongs to the host ` +
          "(build_memories storage=/root= or yian run --memory-root)",
      );
    }
  }
  const allowed = BUILTIN_MEMORY_CONFIG_KEYS.get(type);
  if (allowed !== undefined) {
    const extra = Object.keys(frozen)
      .filter((key) => !allowed.has(key))
      .sort(compareCodeUnits);
    if (extra.length > 0) {
      throw new RangeError(
        `${type} memory config has unknown key(s) ${reprList(extra)} ` +
          `(allowed: ${[...allowed].sort(compareCodeUnits).join(", ")})`,
      );
    }
    checkBuiltinConfigValues(type, frozen);
  }
  return frozen;
}

/**
 * Check one declaration's name, type and config without building it.
 *
 * `name` and `typeName` must match `MEMORY_IDENTIFIER_PATTERN`. `config` must be a plain JSON
 * object of finite values nesting at most `current().maxValueNesting` levels. It may not hold
 * private keys (credentials, Python import paths) anywhere, nor the storage keys `storage`,
 * `store` and `path`. A built-in type allows only the keys in `BUILTIN_MEMORY_CONFIG_KEYS`, each
 * an int in its backend's range (episodic and reflective limits may be null). Ints are `bigint`:
 * `{recall_limit: 3n}`; a `number` is a float, so `{recall_limit: 3}` is refused, as YAML `3.0` is.
 *
 * @throws TypeError when a value has the wrong type.
 * @throws RangeError when a value breaks a shape rule.
 */
export function checkMemoryDeclaration(name: unknown, typeName: unknown, config: unknown): void {
  checkedConfig(name, typeName, config);
}

/** Order two string paths: component by component in code-unit order, then a prefix first. */
function comparePaths(a: readonly string[], b: readonly string[]): number {
  for (let i = 0; i < Math.min(a.length, b.length); i++) {
    const order = compareCodeUnits(a[i]!, b[i]!);
    if (order !== 0) return order;
  }
  return a.length - b.length;
}

/** Whether `prefix` is a prefix of `path`. */
function startsWith(path: readonly string[], prefix: readonly string[]): boolean {
  return prefix.length <= path.length && prefix.every((part, i) => path[i] === part);
}

/** Throw when `keys` repeats a key. */
function requireUnique(keys: readonly string[], name: string): void {
  if (new Set(keys).size !== keys.length) throw new RangeError(`${name} must be unique`);
}

/**
 * The definition site of one memory declaration. Frozen.
 *
 * `scope` is the import-alias path from the root file (`[]` for the root file). `owner` is the
 * lexical flow path compose threads: `["root"]`, `["def", name]`, or an inline body's path.
 * `node` is the declaring AGENT id, or `null` for a flow declaration.
 */
export class DeclSite {
  readonly scope: readonly string[];
  readonly owner: readonly string[];
  readonly node: string | null;

  /**
   * Validate and copy one site.
   *
   * @throws TypeError when a path isn't an array, a component isn't a string, or `node` is
   *   neither `null` nor a string.
   * @throws RangeError when `owner` is empty, a component or `node` is empty, or a string holds
   *   a lone surrogate.
   */
  constructor(scope: readonly string[], owner: readonly string[], node: string | null = null) {
    this.scope = requirePath(scope, "scope", true);
    this.owner = requirePath(owner, "owner");
    this.node = node === null ? null : requireString(node, "node");
    Object.freeze(this);
  }

  /** Total order: scope, owner, flow site before AGENT sites, node id; 0 means equal sites. */
  static compare(a: DeclSite, b: DeclSite): number {
    return (
      comparePaths(a.scope, b.scope) ||
      comparePaths(a.owner, b.owner) ||
      Number(a.node !== null) - Number(b.node !== null) ||
      compareCodeUnits(a.node ?? "", b.node ?? "")
    );
  }
}

/** Closed set of dynamic occurrence segments. */
export const MemoryOccurrenceKind = { MAP_ITEM: "map_item", LOOP_ITERATION: "loop_iteration" } as const;
/** One dynamic occurrence segment kind. */
export type MemoryOccurrenceKind = (typeof MemoryOccurrenceKind)[keyof typeof MemoryOccurrenceKind];

const OCCURRENCE_KINDS: readonly unknown[] = Object.values(MemoryOccurrenceKind);

/** One structured MAP-item or LOOP-iteration occurrence segment. Frozen. */
export class MemoryOccurrenceSegment {
  readonly kind: MemoryOccurrenceKind;
  /** The authored path of the MAP or LOOP node. */
  readonly staticPath: readonly string[];
  /** The non-negative item or iteration index. */
  readonly index: number;

  /**
   * Validate the occurrence kind, authored path and index; `staticPath` is copied.
   *
   * @throws TypeError for an unknown kind, a path that isn't an array of strings, or an index
   *   that isn't an integer `number`.
   * @throws RangeError for an empty path or component, a lone surrogate, or a negative index.
   */
  constructor(kind: MemoryOccurrenceKind, staticPath: readonly string[], index: number) {
    if (!OCCURRENCE_KINDS.includes(kind)) throw new TypeError("kind must be a MemoryOccurrenceKind");
    this.kind = kind;
    this.staticPath = requirePath(staticPath, "static_path");
    if (typeof index !== "number" || !Number.isInteger(index)) throw new TypeError("index must be an integer");
    if (index < 0) throw new RangeError("index must be non-negative");
    this.index = index;
    Object.freeze(this);
  }
}

/** Return a frozen copy of one occurrence path whose segments nest lexically. */
function requireOccurrencePath(value: unknown): readonly MemoryOccurrenceSegment[] {
  if (!isPlainArray(value)) throw new TypeError("occurrence_path must be an array");
  let prior: readonly string[] | null = null;
  for (const segment of value) {
    if (!(segment instanceof MemoryOccurrenceSegment)) {
      throw new TypeError("occurrence_path items must be MemoryOccurrenceSegment values");
    }
    if (prior !== null && (segment.staticPath.length <= prior.length || !startsWith(segment.staticPath, prior))) {
      throw new RangeError("occurrence segments must follow lexical nesting order");
    }
    prior = segment.staticPath;
  }
  return Object.freeze([...(value as MemoryOccurrenceSegment[])]);
}

/**
 * The memory keys and occurrence path carried by one compiled node. Frozen.
 *
 * `staticPath` is the node's root-relative locator. `memories` lists the keys the node recalls and
 * learns into, outermost first. `occurrencePath` lists the MAP items and LOOP iterations around
 * the node, outermost first.
 */
export class MemoryScope {
  readonly staticPath: readonly string[];
  readonly memories: readonly string[];
  readonly occurrencePath: readonly MemoryOccurrenceSegment[];

  /**
   * Validate the locator, the unique keys and the nested occurrence path; arrays are copied.
   *
   * @throws TypeError when a path isn't an array of strings or a segment isn't a
   *   `MemoryOccurrenceSegment`.
   * @throws RangeError for an empty locator or component, a lone surrogate, a repeated key,
   *   segments out of nesting order, or an innermost segment that doesn't strictly contain the node.
   */
  constructor(
    staticPath: readonly string[],
    options: { memories?: readonly string[]; occurrencePath?: readonly MemoryOccurrenceSegment[] } = {},
  ) {
    const { memories = [], occurrencePath = [] } = options;
    this.staticPath = requirePath(staticPath, "static_path");
    this.memories = requirePath(memories, "memories", true);
    requireUnique(this.memories, "memories");
    this.occurrencePath = requireOccurrencePath(occurrencePath);
    const inner = this.occurrencePath.at(-1)?.staticPath;
    if (inner !== undefined && (inner.length >= this.staticPath.length || !startsWith(this.staticPath, inner))) {
      throw new RangeError("the innermost occurrence segment must contain the scoped node");
    }
    Object.freeze(this);
  }
}

/** One authored memory declaration: name, backend type, config and site. Frozen. */
export class MemoryDecl {
  readonly name: string;
  readonly type: string;
  /** The deep-frozen config snapshot, keys in code-unit order. */
  readonly config: JsonObject;
  readonly site: DeclSite;
  readonly origin: OriginRef | null;

  /**
   * Check one declaration as `checkMemoryDeclaration` does and keep its config snapshot; `site`
   * and `origin` are kept by identity.
   *
   * @throws TypeError for a value of the wrong type, a `site` that isn't a `DeclSite`, or an
   *   `origin` that is neither `null` nor an `OriginRef`.
   * @throws RangeError when a value breaks a shape rule.
   */
  constructor(name: string, type: string, config: unknown, site: DeclSite, origin: OriginRef | null = null) {
    this.config = checkedConfig(name, type, config);
    this.name = name;
    this.type = type;
    if (!(site instanceof DeclSite)) throw new TypeError("site must be a DeclSite");
    this.site = site;
    this.origin = requireOrigin(origin, "origin");
    Object.freeze(this);
  }

  /** The memory key: `[...site.scope, name].join(".")`. */
  get key(): string {
    return [...this.site.scope, this.name].join(".");
  }
}

/** Transforms one present origin; the same shape as the node contracts' mapper. */
type OriginMapper = (origin: OriginRef) => OriginRef;

/** Throw unless `decls` is a plain array of `MemoryDecl` values. */
function requireDecls(decls: unknown): asserts decls is readonly MemoryDecl[] {
  if (!isPlainArray(decls)) throw new TypeError("memories must be an array");
  for (const decl of decls) {
    // Exact, because the rewrite rebuilds a `MemoryDecl` and would drop a subclass.
    if (typeof decl !== "object" || decl === null || Object.getPrototypeOf(decl) !== MemoryDecl.prototype) {
      throw new TypeError("memories items must be exact MemoryDecl values");
    }
  }
}

/**
 * Yield each declaration's origin once per `seen`, in declaration order.
 *
 * @param seen Declarations already visited by this traversal; a declaration in it yields nothing.
 * @throws TypeError when `decls` isn't an array of exact `MemoryDecl` values.
 */
export function* iterMemoryDeclOrigins(decls: readonly MemoryDecl[], seen: Set<object>): Generator<OriginRef> {
  requireDecls(decls);
  for (const decl of decls) {
    if (seen.has(decl)) continue;
    seen.add(decl);
    if (decl.origin !== null) yield decl.origin;
  }
}

/**
 * Return a frozen array of `decls` rebuilt with every origin transformed by `mapper`.
 *
 * The declarations are frozen, so the rebuild is the private copy; the inputs are unchanged. The
 * input array and each declaration are rebuilt once per `seen` (old to new; a declaration without
 * an origin maps to itself), so an array or declaration reached twice comes back as one shared
 * object and the mapper runs once per declaration: pass one `seen` for a whole graph. A rebuilt
 * declaration keeps the mapper's `OriginRef` by identity, as the node contracts' rewrite does.
 *
 * @param seen Arrays and declarations already rewritten, each mapped to its rebuilt form.
 * @throws TypeError when `decls` isn't an array of exact `MemoryDecl` values, or `mapper` returns
 *   anything but an `OriginRef`.
 */
export function rewriteMemoryDeclOrigins(
  decls: readonly MemoryDecl[],
  mapper: OriginMapper,
  seen: Map<object, object>,
): readonly MemoryDecl[] {
  requireDecls(decls);
  const done = seen.get(decls);
  if (done !== undefined) return done as readonly MemoryDecl[];
  const rebuilt = Object.freeze(
    decls.map((decl) => {
      const prior = seen.get(decl);
      if (prior !== undefined) return prior as MemoryDecl;
      let next = decl;
      if (decl.origin !== null) {
        const mapped: unknown = mapper(decl.origin);
        if (!(mapped instanceof OriginRef)) throw new TypeError("origin mapper must return an OriginRef");
        next = new MemoryDecl(decl.name, decl.type, decl.config, decl.site, mapped);
      }
      seen.set(decl, next);
      return next;
    }),
  );
  seen.set(decls, rebuilt);
  return rebuilt;
}

/** One recorded memory: key, type, effective config and owning sites. Frozen. */
export class MemorySpec {
  readonly key: string;
  readonly type: string;
  /** The deep-frozen effective config snapshot, keys in code-unit order. */
  readonly config: JsonObject;
  /** The declaring sites, sorted and unique by `DeclSite.compare`. */
  readonly owners: readonly DeclSite[];
  /** True exactly when the memory has one owner and it is an AGENT site. */
  readonly conversationEligible: boolean;

  /**
   * Validate one table entry; `owners` is copied. Compiled keys and configs are taken as they
   * are: only the JSON snapshot and the private-key rule apply, not the declaration check.
   *
   * @throws TypeError for a value of the wrong type or an owner that isn't a `DeclSite`.
   * @throws RangeError for an empty key, type or owners, a bad config, owners out of order, or an
   *   eligibility flag that doesn't match the owners.
   */
  constructor(key: string, type: string, config: unknown, owners: readonly DeclSite[], conversationEligible: boolean) {
    this.key = requireString(key, "key");
    this.type = requireString(type, "type");
    this.config = freezeJsonObject(config, "config");
    rejectPrivateMemoryKeys(this.config, "config");
    if (!isPlainArray(owners)) throw new TypeError("owners must be an array");
    if (owners.length === 0) throw new RangeError("owners must be non-empty");
    if (!owners.every((owner) => owner instanceof DeclSite)) {
      throw new TypeError("owners items must be DeclSite values");
    }
    if (owners.some((owner, i) => i > 0 && DeclSite.compare(owners[i - 1]!, owner) >= 0)) {
      throw new RangeError("owners must be sorted and unique by DeclSite.compare");
    }
    this.owners = Object.freeze([...owners]);
    if (typeof conversationEligible !== "boolean") throw new TypeError("conversation_eligible must be a boolean");
    if (conversationEligible !== (owners.length === 1 && owners[0]!.node !== null)) {
      throw new RangeError("conversation_eligible must be true exactly for one AGENT owner");
    }
    this.conversationEligible = conversationEligible;
    Object.freeze(this);
  }
}

/** The ordered memory keys one static target recalls and learns into. Frozen. */
export class MemoryStack {
  readonly locator: readonly string[];
  readonly memories: readonly string[];

  /**
   * Require a non-empty locator and unique non-empty keys; arrays are copied.
   *
   * @throws TypeError when a path isn't an array of strings.
   * @throws RangeError for an empty locator or component, a lone surrogate, or a repeated key.
   */
  constructor(locator: readonly string[], memories: readonly string[]) {
    this.locator = requirePath(locator, "locator");
    this.memories = requirePath(memories, "memories", true);
    requireUnique(this.memories, "memories");
    Object.freeze(this);
  }
}

/** Invalid host memory setup, attributed to authored declarations when known. */
export class MemorySetupError extends RangeError {
  override name = "MemorySetupError";
  /** The primary declaration site. */
  readonly origin: OriginRef | null;
  /** Other sites of the same failure, in declaration-site order. */
  readonly related: readonly OriginRef[];

  /**
   * Create one setup failure with optional origins; `related` is copied.
   *
   * @throws TypeError when `message` isn't a string, `origin` is neither `null` nor an
   *   `OriginRef`, or `related` isn't an array of `OriginRef` values.
   * @throws RangeError when `message` holds a lone surrogate.
   */
  constructor(message: string, options: { origin?: OriginRef | null; related?: readonly OriginRef[] } = {}) {
    const { origin = null, related = [] } = options;
    requireString(message, "message", false);
    const checkedOrigin = requireOrigin(origin, "origin");
    if (!isPlainArray(related)) throw new TypeError("related must be an array of OriginRef values");
    for (const item of related) {
      if (item === null) throw new TypeError("related must not contain null");
      requireOrigin(item, "related origin");
    }
    super(message);
    this.origin = checkedOrigin;
    this.related = Object.freeze([...related]);
  }
}
