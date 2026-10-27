/**
 * Source-neutral value types and checks for injected workflow memory.
 *
 * This module holds the memory JSON snapshot (exact, finite, well-formed JSON copied into frozen
 * arrays and objects), the rejection of host-private config keys (credentials, storage locations,
 * Python import paths), and the per-declaration shape check `checkMemoryDeclaration`, with the
 * built-in types' config tables. It also holds the frozen identity values compile records:
 * declaration sites, declarations and their origin rewrite, node scopes with their occurrence
 * paths, memory specs and stacks, and the setup error; the host memory objects (`Memory` and its
 * run `MemorySession`), their typed failures and their recorded snapshots; and the learning result
 * carriers (`MemoryMutation` with its batch check `validateMutations`, `LearningFailure`,
 * `PolicyCommitOutcome`, `LearningResult`), frozen and holding their carriers by reference; and
 * what a learner sees of an AGENT invocation (its recalled `MemorySection` values and the visible
 * model and tool interactions), the accepted episodes (`MemoryEpisode`) and one memory's learning
 * batch (`LearnContext`, with its feedback snapshot `freezeFeedback`), whose JSON follows the yian
 * value number rule; and the value comparisons of the scope, snapshot, spec, operation failure and
 * learning carriers (`sameMemoryScope`, `sameMemorySnapshotEnvelope`, `sameMemorySpec`,
 * `sameMemoryOperationFailure`, `sameMemoryMutation`, `sameLearningFailure`,
 * `samePolicyCommitOutcome`, `sameLearningResult`). It also holds the recall carriers
 * (`RecallContext`, `MemoryContent`, `MemoryContext`, `MemoryRecallError`), and the host
 * interfaces: `Memory` and its run `MemorySession` (`recall`, `learn`, `commit`). A host interface
 * method may return a value or a promise, and yian always awaits it; the `snapshot` read stays
 * synchronous. It imports only `diagnostics` and `defaults`.
 */

import { current } from "../defaults.ts";
import { compareCodeUnits, OriginRef, reprText } from "../diagnostics.ts";

/**
 * One JSON scalar: an int is a `bigint`, a float a `number`, finite except in the interactions,
 * the episodes and the learn feedback, which may hold `±Infinity`.
 */
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

/**
 * Return one boolean.
 *
 * @throws TypeError when `value` isn't a boolean.
 */
function requireBoolean(value: unknown, name: string): boolean {
  if (typeof value !== "boolean") throw new TypeError(`${name} must be a boolean`);
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

// Local copies of nodes/contracts.ts's isExactly and sameItems: this module imports only
// diagnostics and defaults, so it can't import nodes/contracts.ts.
/** Whether `value` is an instance of exactly `cls`, not of a subclass. */
function isExactly<T>(value: unknown, cls: abstract new (...args: never[]) => T): value is T {
  return typeof value === "object" && value !== null && Object.getPrototypeOf(value) === cls.prototype;
}

/** Whether two arrays have equal items in order, by `same`. */
function sameItems<T>(a: readonly T[], b: readonly T[], same: (x: T, y: T) => boolean = (x, y) => x === y): boolean {
  return a.length === b.length && a.every((item, i) => same(item, b[i]!));
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

/** One container `freezeJson` is copying: its source, its path, and the children copied so far. */
interface FreezeFrame {
  readonly source: unknown[] | Record<string, unknown>;
  readonly path: string;
  readonly nesting: number;
  /** The object's keys in code-unit order; `null` for an array. */
  readonly keys: string[] | null;
  readonly copy: JsonValue[];
}

/**
 * Validate and copy one JSON value into frozen containers at any depth; see `freezeJsonObject`.
 * With `valueNumbers`, numbers follow the yian value rule: NaN refused, `±Infinity` kept, `-0`
 * copied as `0`. With `memo`, a container already frozen through the same memo returns its copy,
 * so a repeated subtree is shared; a memo is for unlimited nesting, since a cached copy isn't
 * measured against `limit` again.
 */
function freezeJson(
  value: unknown,
  name: string,
  limit = Number.POSITIVE_INFINITY,
  valueNumbers = false,
  memo: Map<object, JsonValue> | null = null,
): JsonValue {
  // An explicit stack, not recursion, so depth is bounded by memory rather than the call stack.
  // It visits children depth-first in index or sorted-key order, as a recursive walk would, so the
  // first error it finds is the one a recursive walk would find.
  const stack: FreezeFrame[] = [];
  // The containers on the current path, so a subtree reached twice is not a cycle.
  const active = new Set<object>();

  // Return a scalar's or a memo'd container's copy, or check a new container and push its frame.
  const enter = (item: unknown, path: string, nesting: number): JsonValue | undefined => {
    if (item === null || typeof item === "boolean" || typeof item === "bigint") return item;
    if (typeof item === "string") return requireString(item, path, false);
    if (typeof item === "number") {
      if (!valueNumbers) {
        if (!Number.isFinite(item)) throw new RangeError(`${path} JSON number must be finite`);
        return item;
      }
      if (Number.isNaN(item)) throw new RangeError(`${path} JSON number must not be NaN`);
      return Object.is(item, -0) ? 0 : item;
    }
    const array = isPlainArray(item);
    if (!array && !isPlainObject(item)) throw new TypeError(`${path} must contain exact JSON values`);
    if (nesting >= limit) {
      throw new RangeError(
        `${path} nests deeper than ${limit} levels, the limit for a yian value (each record and list is a level)`,
      );
    }
    // A memo holds only finished copies, so a container on the active path is never found here.
    const cached = memo?.get(item);
    if (cached !== undefined) return cached;
    if (active.has(item)) throw new RangeError(`${path} contains a circular JSON value`);
    let keys: string[] | null = null;
    if (!array) {
      if (Object.getOwnPropertySymbols(item).length > 0) {
        throw new TypeError(`${path} JSON object keys must be strings`);
      }
      keys = Object.keys(item);
      for (const key of keys) requireString(key, `${path} JSON object key`, false);
      keys.sort(compareCodeUnits);
    }
    active.add(item);
    stack.push({ source: item, path, nesting, keys, copy: [] });
    return undefined;
  };

  let result = enter(value, name, 0);
  while (stack.length > 0) {
    const { source, path, nesting, keys, copy } = stack.at(-1)!;
    const index = copy.length;
    // The next child, or finish the container. An array's length is read each step and its
    // children by index, so a hole reads `undefined` and is refused.
    if (index < (keys ?? (source as unknown[])).length) {
      const child =
        keys === null
          ? enter((source as unknown[])[index], `${path}[${index}]`, nesting + 1)
          : enter((source as Record<string, unknown>)[keys[index]!], `${path}.${keys[index]!}`, nesting + 1);
      if (child !== undefined) copy.push(child);
      continue;
    }
    stack.pop();
    active.delete(source);
    // `Object.fromEntries` defines own data properties, so a `__proto__` key stays data.
    result = Object.freeze(keys === null ? copy : Object.fromEntries(keys.map((key, i) => [key, copy[i]!])));
    memo?.set(source, result);
    stack.at(-1)?.copy.push(result);
  }
  return result!;
}

/**
 * Validate and snapshot one exact JSON object: `null`, booleans, `bigint` ints, finite `number`
 * floats, well-formed strings, plain arrays and plain objects with string keys, copied into frozen
 * arrays and frozen objects whose keys are inserted in code-unit order. With `limitNesting` (the
 * default, for configs) it nests at most `current().maxValueNesting` container levels. With
 * `valueNumbers` (for interaction and episode values) numbers follow the yian value rule instead:
 * NaN refused, `±Infinity` kept, `-0` copied as `0`.
 *
 * @throws TypeError when `value` isn't a plain object, holds a non-JSON value, or has a symbol key.
 * @throws RangeError for a refused number, a lone surrogate, a cycle, or nesting past the limit.
 */
export function freezeJsonObject(
  value: unknown,
  name: string,
  options: { limitNesting?: boolean; valueNumbers?: boolean } = {},
): JsonObject {
  if (!isPlainObject(value)) throw new TypeError(`${name} must be an exact JSON object`);
  const limit = (options.limitNesting ?? true) ? current().maxValueNesting : Number.POSITIVE_INFINITY;
  return freezeJson(value, name, limit, options.valueNumbers ?? false) as JsonObject;
}

/** Validate and copy one interaction or episode JSON value at any depth under the value number rule. */
function freezeEvidenceJson(value: unknown, name: string): JsonValue {
  return freezeJson(value, name, Number.POSITIVE_INFINITY, true);
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
// Keys that look like code import paths (a module, class or path to load) are private, so a memory
// never learns them.
const CODE_IMPORT_KEYS: ReadonlySet<string> = new Set([
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
const CODE_IMPORT_SUFFIXES = withHighConfidencePlurals([
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
const CODE_IMPORT_FIELDS = withHighConfidencePlurals(["class", "import", "module", "path"]);

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
  codeImportContext = false,
  webhookContext = false,
): void {
  if (Array.isArray(value)) {
    value.forEach((item: JsonValue, i) =>
      rejectPrivateMemoryKeys(item, `${name}[${i}]`, locationContext, codeImportContext, webhookContext),
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
    const privateCodeImport = codeImportContext || tokens.includes("python");
    const privateWebhook = webhookContext || tokens.includes("webhook");
    const forbidden =
      matchesExactPrivateKey(normalized, PRIVATE_CREDENTIAL_KEYS) ||
      matchesPrivateSuffix(normalized, PRIVATE_CREDENTIAL_SUFFIXES) ||
      matchesPrivateSuffix(normalized, PRIVATE_TENANT_KEYS) ||
      matchesExactPrivateKey(normalized, CODE_IMPORT_KEYS) ||
      matchesPrivateSuffix(normalized, CODE_IMPORT_SUFFIXES) ||
      matchesPrivateCredentialIndirection(normalized) ||
      (privateLocation && matchesPrivateSuffix(normalized, PRIVATE_LOCATION_FIELDS)) ||
      (privateCodeImport && matchesPrivateSuffix(normalized, CODE_IMPORT_FIELDS)) ||
      (privateWebhook && matchesPrivateSuffix(normalized, PRIVATE_WEBHOOK_FIELDS));
    if (forbidden) throw new RangeError(`${name} contains forbidden private key ${reprText(key)}`);
    rejectPrivateMemoryKeys(object[key]!, `${name}.${key}`, privateLocation, privateCodeImport, privateWebhook);
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

/** Grammar shared by memory names, memory type names and `uses:` scope segments. */
export const MEMORY_IDENTIFIER_PATTERN = /^[A-Za-z_][A-Za-z0-9_-]*$/;

const LOCATION_CONFIG_KEYS: ReadonlySet<string> = new Set(["storage", "store", "path"]);

/** Config keys each built-in memory type accepts from YAML. */
export const BUILTIN_MEMORY_CONFIG_KEYS: ReadonlyMap<string, ReadonlySet<string>> = new Map([
  ["episodic", new Set(["max_episodes", "recall_limit"])],
  ["reflective", new Set(["max_items", "recall_limit"])],
  ["profile", new Set(["max_chars", "input_budget_chars"])],
]);

/**
 * Per built-in key, in the order checked: the least integer allowed and whether null is allowed.
 *
 * These mirror the backend constructors so a bad value fails at load, not first when the host
 * builds the memories; null leaves an episodic or reflective limit to the stored or default value.
 * Null is refused for the profile keys, although their constructors take it as the yian default;
 * leaving a key out has that effect.
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
  // Chat history is a LOOP-carried value, so a memory can't hold it.
  if (type === "session") throw new RangeError("session memory was removed; carry the history as a loop field");
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
        `${type} memory config has unknown key(s) ${jsonText(extra)} ` +
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
 * @throws RangeError when a value breaks a shape rule, or the type is the removed `session`.
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

/** Whether `prefix` is a strict (shorter) prefix of `path`. */
function isStrictPrefix(path: readonly string[], prefix: readonly string[]): boolean {
  return prefix.length < path.length && prefix.every((part, i) => path[i] === part);
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

/** Closed set of authoritative episode outcomes. */
export const MemoryEpisodeStatus = { SUCCEEDED: "succeeded", FAILED: "failed" } as const;
/** One authoritative episode outcome. */
export type MemoryEpisodeStatus = (typeof MemoryEpisodeStatus)[keyof typeof MemoryEpisodeStatus];

const EPISODE_STATUSES: readonly unknown[] = Object.values(MemoryEpisodeStatus);

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
   *   that isn't a safe integer `number`.
   * @throws RangeError for an empty path or component, a lone surrogate, or a negative index.
   */
  constructor(kind: MemoryOccurrenceKind, staticPath: readonly string[], index: number) {
    if (!OCCURRENCE_KINDS.includes(kind)) throw new TypeError("kind must be a MemoryOccurrenceKind");
    this.kind = kind;
    this.staticPath = requirePath(staticPath, "static_path");
    if (!Number.isSafeInteger(index)) throw new TypeError("index must be an integer");
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
    if (prior !== null && !isStrictPrefix(segment.staticPath, prior)) {
      throw new RangeError("occurrence segments must follow lexical nesting order");
    }
    prior = segment.staticPath;
  }
  return Object.freeze([...(value as MemoryOccurrenceSegment[])]);
}

/**
 * Require the innermost occurrence segment, if any, to strictly contain `staticPath`.
 *
 * @throws RangeError when it doesn't.
 */
function requireOccurrenceContains(
  occurrencePath: readonly MemoryOccurrenceSegment[],
  staticPath: readonly string[],
): void {
  const inner = occurrencePath.at(-1)?.staticPath;
  if (inner !== undefined && !isStrictPrefix(staticPath, inner)) {
    throw new RangeError("the innermost occurrence segment must contain the scoped node");
  }
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
    requireOccurrenceContains(this.occurrencePath, this.staticPath);
    Object.freeze(this);
  }
}

/** Whether two scopes have the same static path, memories and occurrence segments (kind, path, index). */
export function sameMemoryScope(a: MemoryScope, b: MemoryScope): boolean {
  return (
    sameItems(a.staticPath, b.staticPath) &&
    sameItems(a.memories, b.memories) &&
    sameItems(
      a.occurrencePath,
      b.occurrencePath,
      (x, y) => x.kind === y.kind && x.index === y.index && sameItems(x.staticPath, y.staticPath),
    )
  );
}

/** One memory's recalled text, shown to an AGENT as its own section. Frozen. */
export class MemorySection {
  readonly memory: string;
  readonly text: string;

  /**
   * Require a non-empty memory key and a text.
   *
   * @throws TypeError when a field isn't a string.
   * @throws RangeError when `memory` is empty or a string holds a lone surrogate.
   */
  constructor(memory: string, text: string) {
    this.memory = requireString(memory, "memory");
    this.text = requireString(text, "text", false);
    Object.freeze(this);
  }
}

/** One visible free-text model response. Frozen. */
export class ModelTextInteraction {
  readonly text: string;

  /**
   * Require the visible text, without provider metadata.
   *
   * @throws TypeError when `text` isn't a string.
   * @throws RangeError when it holds a lone surrogate.
   */
  constructor(text: string) {
    this.text = requireString(text, "text", false);
    Object.freeze(this);
  }
}

/** One visible structured model response. Frozen. */
export class ModelValueInteraction {
  /** The deep-frozen response; any depth. */
  readonly value: JsonValue;

  /**
   * Validate and copy the response under the yian value number rule.
   *
   * @throws TypeError when `value` holds a non-JSON value.
   * @throws RangeError for NaN, a lone surrogate or a cycle.
   */
  constructor(value: unknown) {
    this.value = freezeEvidenceJson(value, "value");
    Object.freeze(this);
  }
}

/** One visible tool call: its name and JSON arguments. Frozen. */
export class ToolCallInteraction {
  readonly name: string;
  /** The deep-frozen arguments, keys in code-unit order; any depth. */
  readonly arguments: JsonObject;

  /**
   * Validate the name, then copy the arguments under the yian value number rule.
   *
   * @throws TypeError when `name` isn't a string, `arguments` isn't an exact JSON object or holds a
   *   non-JSON value.
   * @throws RangeError for an empty name, NaN, a lone surrogate or a cycle.
   */
  constructor(name: string, args: unknown) {
    this.name = requireString(name, "name");
    this.arguments = freezeValueObject(args, "arguments");
    Object.freeze(this);
  }
}

/** One visible tool result paired with its tool name. Frozen. */
export class ToolResultInteraction {
  readonly name: string;
  readonly text: string;

  /**
   * Require a non-empty name and the visible result text.
   *
   * @throws TypeError when a field isn't a string.
   * @throws RangeError when `name` is empty or a string holds a lone surrogate.
   */
  constructor(name: string, text: string) {
    this.name = requireString(name, "name");
    this.text = requireString(text, "text", false);
    Object.freeze(this);
  }
}

/** One visible model or tool interaction. */
export type VisibleInteraction =
  | ModelTextInteraction
  | ModelValueInteraction
  | ToolCallInteraction
  | ToolResultInteraction;

const VISIBLE_INTERACTION_CLASSES = [
  ModelTextInteraction,
  ModelValueInteraction,
  ToolCallInteraction,
  ToolResultInteraction,
] as const;

/**
 * Return a frozen copy of one interaction array; the interactions are kept by reference.
 *
 * @throws TypeError when `value` isn't an array or an item isn't exactly one of the visible
 *   interaction classes.
 */
function requireInteractions(value: unknown): readonly VisibleInteraction[] {
  if (!isPlainArray(value)) throw new TypeError("interactions must be an array");
  for (const interaction of value) {
    if (!VISIBLE_INTERACTION_CLASSES.some((cls) => isExactly<VisibleInteraction>(interaction, cls))) {
      throw new TypeError("interactions items must be VisibleInteraction values");
    }
  }
  return Object.freeze([...(value as VisibleInteraction[])]);
}

/** Copy one exact JSON object at any depth under the yian value number rule. */
function freezeValueObject(value: unknown, name: string): JsonObject {
  return freezeJsonObject(value, name, { limitNesting: false, valueNumbers: true });
}

/** Visible AGENT context from which a memory may derive a query. Frozen. */
export class RecallContext {
  readonly baseSystemPrompt: string;
  readonly prompt: string;
  /** The deep-frozen visible node inputs, keys in code-unit order; any depth. */
  readonly inputs: JsonObject;
  /** The non-empty authored path of the consuming AGENT. */
  readonly consumerStaticPath: readonly string[];

  /**
   * Validate the prompts, snapshot the inputs under the yian value number rule and copy the path,
   * in that order.
   *
   * @throws TypeError when a prompt isn't a string, `inputs` isn't an exact JSON object or holds a
   *   non-JSON value, or the path isn't an array of strings.
   * @throws RangeError for a lone surrogate, NaN, a cycle, or an empty path or component.
   */
  constructor(baseSystemPrompt: string, prompt: string, inputs: unknown, consumerStaticPath: readonly string[]) {
    this.baseSystemPrompt = requireString(baseSystemPrompt, "base_system_prompt", false);
    this.prompt = requireString(prompt, "prompt", false);
    this.inputs = freezeValueObject(inputs, "inputs");
    this.consumerStaticPath = requirePath(consumerStaticPath, "consumer_static_path");
    Object.freeze(this);
  }
}

/**
 * One reducer-accepted source-neutral execution episode. Frozen.
 *
 * Its JSON (`inputs`, `output`, `failure`) is deep-frozen, at any depth, under the yian value
 * number rule.
 */
export class MemoryEpisode {
  readonly runId: string;
  /**
   * The physical accepted node-start record, or `null`; independent of episode order and of the
   * outcome record's position. Host-created episodes may omit it, and source coordinate zero is
   * valid.
   */
  readonly executionSeq: number | null;
  readonly occurrencePath: readonly MemoryOccurrenceSegment[];
  readonly ownerStaticPath: readonly string[];
  readonly status: MemoryEpisodeStatus;
  /** The visible node inputs, or `null` when binding or safe capture was unavailable. */
  readonly inputs: JsonObject | null;
  /** The success output; `null` on a failed episode. */
  readonly output: JsonValue;
  /** The failure data, present exactly on a failed episode. */
  readonly failure: JsonObject | null;
  /** The AGENT's base system prompt; present exactly when `prompt` is. */
  readonly baseSystemPrompt: string | null;
  readonly prompt: string | null;
  readonly memorySections: readonly MemorySection[];
  readonly interactions: readonly VisibleInteraction[];

  /**
   * Validate one accepted success or failure episode and snapshot its JSON, in Python's order:
   * run id, execution sequence, occurrence path, owner path, status, outcome shape, inputs,
   * output, failure, prompts, sections, interactions. The arrays are copied and their carriers
   * kept by reference.
   *
   * @throws TypeError for a field of the wrong type: an execution sequence that isn't a safe
   *   integer `number` or `null`, an unknown status, inputs that aren't an exact JSON object or
   *   `null`, a non-JSON value, or an item that isn't exactly a segment, `MemorySection` or visible
   *   interaction.
   * @throws RangeError for an empty run id or owner path, a negative execution sequence, segments
   *   that don't nest or contain the owner, failure data on a succeeded episode (or output, or no
   *   failure, on a failed one), NaN, a cycle, a lone surrogate, or one prompt without the other.
   */
  constructor(
    runId: string,
    occurrencePath: readonly MemoryOccurrenceSegment[],
    ownerStaticPath: readonly string[],
    status: MemoryEpisodeStatus,
    inputs: unknown,
    options: {
      output?: unknown;
      failure?: unknown;
      interactions?: readonly VisibleInteraction[];
      baseSystemPrompt?: string | null;
      prompt?: string | null;
      memorySections?: readonly MemorySection[];
      executionSeq?: number | null;
    } = {},
  ) {
    const {
      output = null,
      failure = null,
      interactions = [],
      baseSystemPrompt = null,
      prompt = null,
      memorySections = [],
      executionSeq = null,
    } = options;
    this.runId = requireString(runId, "run_id");
    if (executionSeq !== null) {
      if (!Number.isSafeInteger(executionSeq)) throw new TypeError("execution_seq must be an integer or null");
      if (executionSeq < 0) throw new RangeError("execution_seq must be non-negative");
    }
    this.executionSeq = executionSeq;
    this.occurrencePath = requireOccurrencePath(occurrencePath);
    this.ownerStaticPath = requirePath(ownerStaticPath, "owner_static_path");
    requireOccurrenceContains(this.occurrencePath, this.ownerStaticPath);
    if (!EPISODE_STATUSES.includes(status)) throw new TypeError("status must be a MemoryEpisodeStatus");
    this.status = status;

    if (status === MemoryEpisodeStatus.SUCCEEDED && failure !== null) {
      throw new RangeError("a succeeded episode cannot carry failure data");
    }
    if (status === MemoryEpisodeStatus.FAILED) {
      if (failure === null) throw new RangeError("a failed episode must carry failure data");
      if (output !== null) throw new RangeError("a failed episode cannot carry output data");
    }

    if (inputs !== null && !isPlainObject(inputs)) throw new TypeError("inputs must be an exact JSON object or null");
    this.inputs = inputs === null ? null : freezeValueObject(inputs, "inputs");
    this.output = freezeEvidenceJson(output, "output");
    this.failure = failure === null ? null : freezeValueObject(failure, "failure");
    if ((baseSystemPrompt === null) !== (prompt === null)) {
      throw new RangeError("base_system_prompt and prompt must both be present or absent");
    }
    this.baseSystemPrompt =
      baseSystemPrompt === null ? null : requireString(baseSystemPrompt, "base_system_prompt", false);
    this.prompt = prompt === null ? null : requireString(prompt, "prompt", false);
    if (!isPlainArray(memorySections)) throw new TypeError("memory_sections must be an array");
    for (const section of memorySections) {
      if (!isExactly(section, MemorySection)) throw new TypeError("memory_sections items must be MemorySection values");
    }
    this.memorySections = Object.freeze([...memorySections]);
    this.interactions = requireInteractions(interactions);
    Object.freeze(this);
  }
}

/**
 * Validate and snapshot feedback items as a learning context names them: each a deep-frozen copy
 * under the yian value number rule, at any depth.
 *
 * @param memo Copies shared across calls: a container already frozen through it returns its copy,
 *   so a subtree repeated within or across the calls is one shared object.
 * @param firstIndex The context index of the first item, so errors name `feedback[firstIndex + i]`.
 * @returns A frozen array of one copy per item.
 * @throws TypeError when an item holds a non-JSON value; the message names `feedback[i]`.
 * @throws RangeError for NaN or a cycle, with a message naming `feedback[i]`, or for a lone
 *   surrogate, with the canonical JSON message.
 */
export function freezeFeedback(
  entries: Iterable<unknown>,
  memo: Map<object, JsonValue>,
  firstIndex = 0,
): readonly JsonValue[] {
  return Object.freeze(
    Array.from(entries, (item, index) =>
      freezeJson(item, `feedback[${firstIndex + index}]`, Number.POSITIVE_INFINITY, true, memo),
    ),
  );
}

/**
 * Check a learning context's run id, memory key and episodes, in that order, and return them with
 * the episodes copied into a frozen array.
 *
 * @throws TypeError when an id isn't a string, `episodes` isn't an array, or an episode isn't
 *   exactly a `MemoryEpisode`.
 * @throws RangeError when an id is empty or holds a lone surrogate, or an episode belongs to
 *   another run.
 */
function learnIdentity(
  runId: unknown,
  memory: unknown,
  episodes: unknown,
): [string, string, readonly MemoryEpisode[]] {
  const run = requireString(runId, "run_id");
  const key = requireString(memory, "memory");
  if (!isPlainArray(episodes)) throw new TypeError("episodes must be an array");
  for (const episode of episodes) {
    if (!isExactly(episode, MemoryEpisode)) throw new TypeError("episodes items must be MemoryEpisode values");
    if (episode.runId !== run) throw new RangeError("episode run_id must match the learning context");
  }
  return [run, key, Object.freeze([...(episodes as MemoryEpisode[])])];
}

/** One memory's ordered learning batch. Frozen. */
export class LearnContext {
  readonly runId: string;
  readonly memory: string;
  /** The memory's episodes, kept by reference. */
  readonly episodes: readonly MemoryEpisode[];
  /** Deep-frozen feedback copies in their reserved downstream-feedback order. */
  readonly feedback: readonly JsonValue[];

  /**
   * Validate the run id, memory key and episodes, then snapshot `feedback` with a fresh memo, so a
   * subtree repeated within it is shared.
   *
   * @throws TypeError when an id isn't a string, `episodes` or `feedback` isn't an array, an
   *   episode isn't exactly a `MemoryEpisode`, or a feedback item holds a non-JSON value.
   * @throws RangeError when an id is empty or holds a lone surrogate, an episode belongs to
   *   another run, or a feedback item holds NaN, a lone surrogate or a cycle.
   */
  constructor(runId: string, memory: string, episodes: readonly MemoryEpisode[], feedback: readonly unknown[] = []) {
    [this.runId, this.memory, this.episodes] = learnIdentity(runId, memory, episodes);
    if (!isPlainArray(feedback)) throw new TypeError("feedback must be an array");
    this.feedback = freezeFeedback(feedback, new Map());
    Object.freeze(this);
  }

  /**
   * @internal A context over feedback `freezeFeedback` already returned, stored as given; run,
   * memory and episodes checked as the constructor does. Learning's owner contexts use it.
   */
  static fromFrozen(
    runId: string,
    memory: string,
    episodes: readonly MemoryEpisode[],
    frozen: readonly JsonValue[],
  ): LearnContext {
    const [run, key, checked] = learnIdentity(runId, memory, episodes);
    // Built without the constructor, which would freeze the feedback again; fields in its order.
    const context = Object.create(LearnContext.prototype) as LearnContext;
    Object.assign(context, { runId: run, memory: key, episodes: checked, feedback: frozen });
    return Object.freeze(context);
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
    const checked = checkedConfig(name, type, config);
    this.name = name;
    this.type = type;
    this.config = checked;
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
    if (!isExactly(decl, MemoryDecl)) throw new TypeError("memories items must be exact MemoryDecl values");
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

  /**
   * Validate one table entry; `owners` is copied. Compiled keys and configs are taken as they
   * are: only the JSON snapshot and the private-key rule apply, not the declaration check.
   *
   * @throws TypeError for a value of the wrong type or an owner that isn't a `DeclSite`.
   * @throws RangeError for an empty key, type or owners, a bad config, or owners out of order.
   */
  constructor(key: string, type: string, config: unknown, owners: readonly DeclSite[]) {
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
   * Extra lines a diagnostic shows after the message, such as the other declaration sites; an own
   * array only when there are notes, so later `addNote` calls (Python's `add_note`) can append.
   */
  declare notes?: string[];

  /**
   * Create one setup failure with optional origins and notes; `related` is copied, a non-empty
   * `notes` (Python's `add_note`, given here at construction) becomes the error's own copy, and
   * `cause` (Python's `raise ... from`) is passed to `RangeError` when given.
   *
   * @throws TypeError when `message` isn't a string, `origin` is neither `null` nor an
   *   `OriginRef`, `related` isn't an array of `OriginRef` values, or `notes` isn't an array of
   *   strings.
   * @throws RangeError when `message` or a note holds a lone surrogate.
   */
  constructor(
    message: string,
    options: {
      origin?: OriginRef | null;
      related?: readonly OriginRef[];
      notes?: readonly string[];
      cause?: unknown;
    } = {},
  ) {
    const { origin = null, related = [], notes = [] } = options;
    requireString(message, "message", false);
    const checkedOrigin = requireOrigin(origin, "origin");
    if (!isPlainArray(related)) throw new TypeError("related must be an array of OriginRef values");
    for (const item of related) {
      if (item === null) throw new TypeError("related must not contain null");
      requireOrigin(item, "related origin");
    }
    if (!isPlainArray(notes)) throw new TypeError("notes must be an array of strings");
    for (const note of notes) requireString(note, "note", false);
    super(message, "cause" in options ? { cause: options.cause } : undefined);
    this.origin = checkedOrigin;
    this.related = Object.freeze([...related]);
    if (notes.length > 0) this.notes = [...notes];
  }
}

/** Ordinary recall failure attributed to a memory declaration site. */
export class MemoryRecallError extends Error {
  override name = "MemoryRecallError";
  /** The memory's declaration site. */
  readonly origin: OriginRef | null;

  /**
   * Create one recall wrapper; `cause` (Python's `raise ... from`) is passed to `Error` when given.
   *
   * @throws TypeError when `message` isn't a string or `origin` is neither `null` nor an `OriginRef`.
   * @throws RangeError when `message` holds a lone surrogate.
   */
  constructor(message: string, options: { origin?: OriginRef | null; cause?: unknown } = {}) {
    const { origin = null } = options;
    requireString(message, "message", false);
    const checkedOrigin = requireOrigin(origin, "origin");
    super(message, "cause" in options ? { cause: options.cause } : undefined);
    this.origin = checkedOrigin;
  }
}

/** One opaque public durable snapshot declared by a memory backend. Frozen. */
export class MemorySnapshotEnvelope {
  readonly memory: string;
  /** The backend's deep-frozen public export, keys in code-unit order; any depth. */
  readonly snapshot: JsonObject;

  /**
   * Snapshot `snapshot`, then check the memory key, as Python does.
   *
   * @throws TypeError when `snapshot` isn't an exact JSON object or `memory` isn't a string.
   * @throws RangeError for a bad JSON value, an empty `memory`, or a lone surrogate.
   */
  constructor(memory: string, snapshot: unknown) {
    const frozen = freezeJsonObject(snapshot, "snapshot", { limitNesting: false });
    this.memory = requireString(memory, "memory");
    this.snapshot = frozen;
    Object.freeze(this);
  }
}

/**
 * Whether two envelopes have the same memory key and equal snapshots, by value and type at each
 * node: stricter than Python's `==`, which equates `1`, `1.0` and `True`.
 */
export function sameMemorySnapshotEnvelope(a: MemorySnapshotEnvelope, b: MemorySnapshotEnvelope): boolean {
  return a.memory === b.memory && sameJson(a.snapshot, b.snapshot);
}

/**
 * Whether two specs have the same key, type, config and owners, by value and type:
 * Python's dataclass equality, with configs compared as `sameMemorySnapshotEnvelope` compares
 * snapshots and owners item by item with `DeclSite.compare`.
 */
export function sameMemorySpec(a: MemorySpec, b: MemorySpec): boolean {
  return (
    a.key === b.key &&
    a.type === b.type &&
    sameJson(a.config, b.config) &&
    sameItems(a.owners, b.owners, (x, y) => DeclSite.compare(x, y) === 0)
  );
}

/** Sanitized durable failure from one memory operation. Frozen. */
export class MemoryOperationFailure {
  readonly stage: string;
  readonly code: string;
  readonly message: string;
  readonly retryable: boolean;

  /**
   * Require non-empty string fields and a boolean retry flag.
   *
   * @throws TypeError when a field isn't a string or `retryable` isn't a boolean.
   * @throws RangeError when a string is empty or holds a lone surrogate.
   */
  constructor(stage: string, code: string, message: string, retryable: boolean) {
    this.stage = requireString(stage, "stage");
    this.code = requireString(code, "code");
    this.message = requireString(message, "message");
    this.retryable = requireBoolean(retryable, "retryable");
    Object.freeze(this);
  }
}

/** Whether two operation failures have the same stage, code, message and retry flag. */
export function sameMemoryOperationFailure(a: MemoryOperationFailure, b: MemoryOperationFailure): boolean {
  return a.stage === b.stage && a.code === b.code && a.message === b.message && a.retryable === b.retryable;
}

/** One JSON mutation a memory proposed. Frozen. */
export class MemoryMutation {
  readonly mutationId: string;
  /** The deep-frozen payload snapshot; any depth. */
  readonly payload: JsonValue;

  /**
   * Validate the id and snapshot the payload.
   *
   * @throws TypeError when `mutationId` isn't a string or `payload` holds a non-JSON value.
   * @throws RangeError for an empty `mutationId`, a lone surrogate, a non-finite number or a cycle.
   */
  constructor(mutationId: string, payload: unknown) {
    this.mutationId = requireString(mutationId, "mutation_id");
    this.payload = freezeJson(payload, "payload");
    Object.freeze(this);
  }
}

/**
 * Whether two JSON values are equal: the same type at each node (a `bigint` never equals a
 * `number`), arrays item by item, records by key set and values.
 */
function sameJson(a: JsonValue, b: JsonValue): boolean {
  if (a === null || b === null || typeof a !== "object" || typeof b !== "object") return a === b;
  const aArray = Array.isArray(a);
  if (aArray || Array.isArray(b)) {
    return aArray && Array.isArray(b) && sameItems(a as readonly JsonValue[], b as readonly JsonValue[], sameJson);
  }
  const x = a as JsonObject;
  const y = b as JsonObject;
  const keys = Object.keys(x);
  return (
    keys.length === Object.keys(y).length && keys.every((key) => Object.hasOwn(y, key) && sameJson(x[key]!, y[key]!))
  );
}

/** Whether two mutations have the same id and equal payloads. */
export function sameMemoryMutation(a: MemoryMutation, b: MemoryMutation): boolean {
  return a.mutationId === b.mutationId && sameJson(a.payload, b.payload);
}

/**
 * Return a frozen copy of one mutation batch in its order; the mutations are kept by reference.
 *
 * @throws TypeError when `mutations` isn't an array or an item isn't exactly a `MemoryMutation`.
 * @throws RangeError when two mutations share an id.
 */
export function validateMutations(mutations: unknown): readonly MemoryMutation[] {
  if (!isPlainArray(mutations)) throw new TypeError("mutations must be an array");
  for (const mutation of mutations) {
    if (!isExactly(mutation, MemoryMutation)) throw new TypeError("mutations items must be MemoryMutation values");
  }
  const checked = mutations as MemoryMutation[];
  requireUnique(checked.map((mutation) => mutation.mutationId), "mutation IDs");
  return Object.freeze([...checked]);
}

const LEARNING_OPERATIONS = ["learn", "commit"] as const;

/** The memory operation a learning failure came from. */
export type LearningOperation = (typeof LEARNING_OPERATIONS)[number];

/** One memory's learn or commit failure in a run result. Frozen. */
export class LearningFailure {
  readonly memory: string;
  readonly operation: LearningOperation;
  readonly failure: MemoryOperationFailure;

  /**
   * Require an exact attribution and one `MemoryOperationFailure`, kept by reference.
   *
   * @throws TypeError when `memory` or `operation` isn't a string or `failure` isn't exactly a
   *   `MemoryOperationFailure`.
   * @throws RangeError for an empty string, a lone surrogate, or an unknown operation.
   */
  constructor(memory: string, operation: LearningOperation, failure: MemoryOperationFailure) {
    this.memory = requireString(memory, "memory");
    requireString(operation, "operation");
    if (!LEARNING_OPERATIONS.includes(operation)) throw new RangeError("operation must be 'learn' or 'commit'");
    this.operation = operation;
    if (!isExactly(failure, MemoryOperationFailure)) throw new TypeError("failure must be a MemoryOperationFailure");
    this.failure = failure;
    Object.freeze(this);
  }
}

/** Whether two learning failures have the same memory, operation and failure. */
export function sameLearningFailure(a: LearningFailure, b: LearningFailure): boolean {
  return a.memory === b.memory && a.operation === b.operation && sameMemoryOperationFailure(a.failure, b.failure);
}

const POLICY_COMMIT_STATUSES = ["unrequested", "unknown", "succeeded", "failed"] as const;

/** One memory's commit disposition. */
export type PolicyCommitStatus = (typeof POLICY_COMMIT_STATUSES)[number];

/** One memory's confirmed or still-unresolved commit disposition. Frozen. */
export class PolicyCommitOutcome {
  readonly memory: string;
  readonly status: PolicyCommitStatus;
  /** The commit's protocol identity once requested; `null` when unrequested. */
  readonly commitId: string | null;
  /** Required only for a confirmed failed commit; its stage and retryability are the backend's. */
  readonly failure: MemoryOperationFailure | null;

  /**
   * Validate the identities and the status's optional fields; `failure` is kept by reference.
   *
   * @throws TypeError when a string field isn't a string or `failure` is neither `null` nor exactly
   *   a `MemoryOperationFailure`.
   * @throws RangeError for an empty string, a lone surrogate, an unknown status, commit data on an
   *   unrequested policy, a requested policy without `commitId`, or a failure on any status but
   *   `failed` (or none on `failed`).
   */
  constructor(
    memory: string,
    status: PolicyCommitStatus,
    options: { commitId?: string | null; failure?: MemoryOperationFailure | null } = {},
  ) {
    const { commitId = null, failure = null } = options;
    this.memory = requireString(memory, "memory");
    requireString(status, "status");
    if (!POLICY_COMMIT_STATUSES.includes(status)) throw new RangeError("invalid policy commit status");
    this.status = status;
    this.commitId = commitId === null ? null : requireString(commitId, "commit_id");
    if (failure !== null && !isExactly(failure, MemoryOperationFailure)) {
      throw new TypeError("failure must be a MemoryOperationFailure or null");
    }
    this.failure = failure;
    if (status === "unrequested") {
      if (commitId !== null || failure !== null) throw new RangeError("unrequested policy cannot carry commit data");
    } else if (commitId === null) {
      throw new RangeError("requested policy requires commit_id");
    }
    if ((status === "failed") !== (failure !== null)) {
      throw new RangeError("only failed policy commits require a failure");
    }
    Object.freeze(this);
  }
}

/** Whether two commit outcomes have the same memory, status, commit id and failure (both null or equal). */
export function samePolicyCommitOutcome(a: PolicyCommitOutcome, b: PolicyCommitOutcome): boolean {
  return (
    a.memory === b.memory &&
    a.status === b.status &&
    a.commitId === b.commitId &&
    (a.failure === null ? b.failure === null : b.failure !== null && sameMemoryOperationFailure(a.failure, b.failure))
  );
}

const LEARNING_STATUSES = ["succeeded", "failed", "incomplete"] as const;

/** How a run's learning ended. */
export type LearningStatus = (typeof LEARNING_STATUSES)[number];

/**
 * A run's learning completion, kept apart from backend retryability hints. Frozen.
 *
 * `succeeded` requires no failures and only succeeded policies; `failed` requires at least one
 * failure; `incomplete` infers nothing from the known dispositions.
 */
export class LearningResult {
  readonly status: LearningStatus;
  /** Failures unique by memory and operation, in supplied order. */
  readonly failures: readonly LearningFailure[];
  /** Commit outcomes unique by memory key, in supplied order; the run layout owns membership. */
  readonly policies: readonly PolicyCommitOutcome[];

  /**
   * Validate the supplied dispositions; the arrays are copied and their carriers kept by reference.
   *
   * @throws TypeError when `status` isn't a string, an array isn't an array, or an item isn't
   *   exactly a `LearningFailure` or `PolicyCommitOutcome`.
   * @throws RangeError for an empty or unknown status, a lone surrogate, a repeated failure
   *   attribution or memory key, or dispositions the status doesn't allow.
   */
  constructor(
    status: LearningStatus,
    options: { failures?: readonly LearningFailure[]; policies?: readonly PolicyCommitOutcome[] } = {},
  ) {
    const { failures = [], policies = [] } = options;
    requireString(status, "status");
    if (!LEARNING_STATUSES.includes(status)) {
      throw new RangeError("learning status must be 'succeeded', 'failed', or 'incomplete'");
    }
    this.status = status;
    if (!isPlainArray(failures)) throw new TypeError("learning failures must be an array");
    const attributions = new Set<string>();
    for (const failure of failures) {
      if (!isExactly(failure, LearningFailure)) throw new TypeError("failures items must be LearningFailure values");
      // The operation is "learn" or "commit", so the key can't collide across memories.
      const attribution = `${failure.operation}:${failure.memory}`;
      if (attributions.has(attribution)) throw new RangeError("learning failure attribution must be unique");
      attributions.add(attribution);
    }
    if (!isPlainArray(policies)) throw new TypeError("learning policies must be an array");
    const names = new Set<string>();
    for (const policy of policies) {
      if (!isExactly(policy, PolicyCommitOutcome)) {
        throw new TypeError("policies items must be PolicyCommitOutcome values");
      }
      if (names.has(policy.memory)) throw new RangeError("learning memory keys must be unique");
      names.add(policy.memory);
    }
    this.failures = Object.freeze([...failures]);
    this.policies = Object.freeze([...policies]);
    if (status === "succeeded" && failures.length > 0) {
      throw new RangeError("succeeded learning cannot carry failures");
    }
    if (status === "succeeded" && policies.some((policy) => policy.status !== "succeeded")) {
      throw new RangeError("succeeded learning requires succeeded policies");
    }
    if (status === "failed" && failures.length === 0) {
      throw new RangeError("failed learning requires at least one failure");
    }
    Object.freeze(this);
  }
}

/** Whether two learning results have the same status, and equal failures and policies in order. */
export function sameLearningResult(a: LearningResult, b: LearningResult): boolean {
  return (
    a.status === b.status &&
    sameItems(a.failures, b.failures, sameLearningFailure) &&
    sameItems(a.policies, b.policies, samePolicyCommitOutcome)
  );
}

/** Typed ordinary failure raised by a memory or session operation. */
export class MemoryOperationError extends Error {
  override name = "MemoryOperationError";
  readonly stage: string;
  readonly code: string;
  readonly retryable: boolean;

  /**
   * Create one stable memory or session failure; `message` is the error's own message.
   *
   * @throws TypeError when a field isn't a string or `retryable` isn't a boolean.
   * @throws RangeError when a string is empty or holds a lone surrogate.
   */
  constructor(stage: string, code: string, message: string, retryable: boolean) {
    const checkedStage = requireString(stage, "stage");
    const checkedCode = requireString(code, "code");
    const checkedMessage = requireString(message, "message");
    const checkedRetryable = requireBoolean(retryable, "retryable");
    super(checkedMessage);
    this.stage = checkedStage;
    this.code = checkedCode;
    this.retryable = checkedRetryable;
  }
}

/**
 * One memory's transient notes. Frozen.
 *
 * Return `new MemoryContent({ text: "Prefer concise answers." })` from a notes memory.
 */
export class MemoryContent {
  /** The notes; empty text contributes no visible section. */
  readonly text: string;

  /**
   * Require the notes.
   *
   * @throws TypeError when `text` isn't a string.
   * @throws RangeError when `text` holds a lone surrogate.
   */
  constructor(options: { text?: string } = {}) {
    const { text = "" } = options;
    this.text = requireString(text, "text", false);
    Object.freeze(this);
  }
}

/**
 * Ordered visible notes. Frozen.
 *
 * `new MemoryContext()` is the empty context of an invocation without memory.
 */
export class MemoryContext {
  /** The sections in outer-to-local ownership order; admission checks the scope order. */
  readonly sections: readonly MemorySection[];

  /**
   * Copy the sections array, keeping its sections by reference.
   *
   * @throws TypeError when `sections` isn't an array or an item isn't exactly a `MemorySection`.
   */
  constructor(options: { sections?: readonly MemorySection[] } = {}) {
    const { sections = [] } = options;
    if (!isPlainArray(sections)) throw new TypeError("sections must be an array");
    for (const section of sections) {
      if (!isExactly(section, MemorySection)) throw new TypeError("sections must contain MemorySection values");
    }
    this.sections = Object.freeze([...sections]);
    Object.freeze(this);
  }
}

/**
 * One run session over a host-provided memory object.
 *
 * A method may return a value or a promise, and yian always awaits it, as for tools; `snapshot` is
 * read synchronously. A session has no `with` form: its caller calls `close()` in `finally`.
 */
export interface MemorySession {
  /**
   * The backend's public durable export, persisted verbatim for run recovery: no host
   * credentials, live handles, backend paths or tenant ids.
   */
  readonly snapshot: JsonObject;
  /** Release resources held by this run session. */
  close(): void | Promise<void>;
  /**
   * Return notes for one invocation; `null` or no context asks for the whole memory, as a host
   * inspection does.
   */
  recall(context?: RecallContext | null): string | MemoryContent | Promise<string | MemoryContent>;
  /** Return this memory's proposed mutations for one batch, as an array. */
  learn(context: LearnContext): readonly MemoryMutation[] | Promise<readonly MemoryMutation[]>;
  /**
   * Atomically commit one immutable mutation batch.
   *
   * `commitId` is stable within one backward attempt, so a repeated commit of that attempt may be a
   * no-op. Every later attempt that requests commits gets a distinct id, even for an identical batch.
   */
  commit(commitId: string, mutations: readonly MemoryMutation[]): void | Promise<void>;
}

/**
 * Host-injected memory object that opens run sessions and can be reset. A method may return a
 * value or a promise, and yian always awaits it.
 */
export interface Memory {
  /** Open one run session, optionally restoring a recorded snapshot. */
  openRun(options?: { snapshot?: JsonObject | null }): MemorySession | Promise<MemorySession>;
  /** Clear current memory state according to the backend's policy. */
  reset(): void | Promise<void>;
}

/** Whether `value` has `Memory`'s methods (Python's runtime-checkable `isinstance`). */
export function isMemory(value: unknown): value is Memory {
  if (typeof value !== "object" || value === null) return false;
  const candidate = value as { openRun?: unknown; reset?: unknown };
  return typeof candidate.openRun === "function" && typeof candidate.reset === "function";
}
