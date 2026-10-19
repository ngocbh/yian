/**
 * Per-declaration shape rules, a port of `tests/engine/test_memory_decl_shape.py`. Through
 * `MemoryDecl`, as Python writes them: `test_valid_memory_names_are_accepted`,
 * `test_invalid_memory_names_are_rejected`, `test_invalid_memory_type_names_are_rejected`,
 * `test_location_keys_are_banned_for_every_type`, `test_api_key_anywhere_is_rejected`,
 * `test_builtin_types_accept_only_allowlisted_keys`, `test_custom_types_accept_any_public_config`
 * and `test_non_finite_config_is_rejected`. Through `checkMemoryDeclaration`:
 * `test_builtin_config_values_are_checked`, `test_builtin_config_rules_cover_the_allowlist`, the
 * check halves of `test_builtin_config_values_at_their_limits_load_and_build`,
 * `test_builtin_config_rules_match_the_backends` (its null half) and
 * `test_shape_check_is_public_and_exported`, and a table of the private-key rules. Also
 * `test_builtin_allowlist_is_a_static_table` and `test_setup_error_carries_related_origins`.
 *
 * Changed: ints are `bigint`; a `number` is a float, so `{recall_limit: 3}` is refused as YAML
 * `3.0` is. A value in a message is compact JSON with JS number spelling (`got [1]`, `got 1` for
 * the float `1.0`). `test_builtin_config_rules_cover_the_allowlist` can't read the private rules
 * table, so it holds each allowlisted key's minimum and checks that the minimum loads and the
 * minimum minus one is refused. `test_non_finite_config_is_rejected` adds `Infinity`. Python's
 * `ValueError` is `RangeError`. The setup error's list-for-tuple `related` row passes a `Set`
 * (TS has only arrays) and checks the error's `name`.
 *
 * Not ported: the backend halves of `test_builtin_config_values_at_their_limits_load_and_build` and
 * `test_builtin_config_rules_match_the_backends` (the built-in backends aren't ported); the
 * `__all__` half of `test_shape_check_is_public_and_exported` (no memory package facade yet).
 *
 * The private-key rows' outcomes come from Python, run on the same keys:
 *
 *   for r in rows: check_memory_declaration("n", "vector", r)
 *
 * which refused every forbidden row below with `config<path> contains forbidden private key
 * '<key>'` (for example `config.storage ... 'path'`, `config.databases[0] ... 'uri'`,
 * `config.10 ... 'token'` for `{"9": {"api_key": 1}, "10": {"token": 1}}`,
 * `config.webhook ... 'my_signature'`, `config.x²Storage ... 'path'`, `config ... 'paẞword'`) and
 * accepted `path_style`, `url`, `max_items`, `authors`, `sigma`, `storage_class`, `my_signature`
 * and `apı_key`.
 */

import { expect, test } from "bun:test";

import { OriginRef } from "../../src/diagnostics.ts";
import {
  BUILTIN_MEMORY_CONFIG_KEYS,
  checkMemoryDeclaration,
  DeclSite,
  MEMORY_IDENTIFIER_PATTERN,
  MemoryDecl,
  MemorySetupError,
} from "../../src/memory/contracts.ts";

const SITE = new DeclSite([], ["root"], null);

/** Build one declaration at the root site. */
function decl(name: unknown, typeName: unknown, config: unknown): MemoryDecl {
  return new MemoryDecl(name as string, typeName as string, config, SITE);
}

/** Assert that `fn` throws exactly `cls` with exactly `message`. */
function expectError(fn: () => unknown, cls: typeof TypeError | typeof RangeError, message: string): void {
  let caught: unknown;
  try {
    fn();
  } catch (error) {
    caught = error;
  }
  expect(caught).toBeInstanceOf(cls);
  expect((caught as Error).message).toBe(message);
}

// test_valid_memory_names_are_accepted
test.each(["notes", "_n", "a-b", "A9_x"])("memory name %p is accepted", (name) => {
  expect(decl(name, "episodic", {}).name).toBe(name);
  expect(MEMORY_IDENTIFIER_PATTERN.test(name)).toBe(true);
});

// test_invalid_memory_names_are_rejected
test.each(["", "9a", "a.b", "a b", "-a", "né"])("memory name %p is rejected", (name) => {
  expect(MEMORY_IDENTIFIER_PATTERN.test(name)).toBe(false);
  expect(() => decl(name, "episodic", {})).toThrow(RangeError);
  expect(() => decl(name, "episodic", {})).toThrow("memory name");
});

test("a memory name must be a string", () => {
  expectError(() => decl(1n, "episodic", {}), TypeError, "memory name must be a string");
  expectError(() => decl("a.b", "episodic", {}), RangeError, "memory name 'a.b' must match [A-Za-z_][A-Za-z0-9_-]*");
});

// test_invalid_memory_type_names_are_rejected
test.each(["", "my.type", "1x"])("memory type %p is rejected", (typeName) => {
  expect(() => decl("notes", typeName, {})).toThrow(RangeError);
  expect(() => decl("notes", typeName, {})).toThrow("memory type");
});

// test_location_keys_are_banned_for_every_type
test.each(["storage", "store", "path"])("location key %p is banned", (key) => {
  expectError(
    () => decl("notes", "vector", { [key]: "x" }),
    RangeError,
    `config key '${key}' is not allowed: storage belongs to the host ` +
      "(build_memories storage=/root= or yian run --memory-root)",
  );
});

test("the first location key in the caller's order is reported", () => {
  expect(() => decl("notes", "vector", { storage: 1n, path: 2n })).toThrow("config key 'storage'");
});

// test_api_key_anywhere_is_rejected
test("api_key anywhere is rejected", () => {
  expectError(
    () => decl("notes", "vector", { nested: { api_key: "x" } }),
    RangeError,
    "config.nested contains forbidden private key 'api_key'",
  );
});

test.each<[string, Record<string, unknown>, string | null]>([
  // Exact credential keys and credential suffixes, plurals included.
  ["api_key", { api_key: 1n }, "config contains forbidden private key 'api_key'"],
  ["token", { token: 1n }, "config contains forbidden private key 'token'"],
  ["password", { password: 1n }, "config contains forbidden private key 'password'"],
  ["github_token", { github_token: 1n }, "config contains forbidden private key 'github_token'"],
  ["my_api_key", { my_api_key: 1n }, "config contains forbidden private key 'my_api_key'"],
  ["api_keys", { api_keys: 1n }, "config contains forbidden private key 'api_keys'"],
  ["db_passwords", { db_passwords: 1n }, "config contains forbidden private key 'db_passwords'"],
  ["client_credentials", { client_credentials: 1n }, "config contains forbidden private key 'client_credentials'"],
  // Tenant ids.
  ["tenant_id", { tenant_id: 1n }, "config contains forbidden private key 'tenant_id'"],
  ["myTenantId", { myTenantId: 1n }, "config contains forbidden private key 'myTenantId'"],
  // Python keys and suffixes.
  ["class", { class: 1n }, "config contains forbidden private key 'class'"],
  ["module_path", { module_path: 1n }, "config contains forbidden private key 'module_path'"],
  ["plugin_imports", { plugin_imports: 1n }, "config contains forbidden private key 'plugin_imports'"],
  ["python_path", { python_path: 1n }, "config contains forbidden private key 'python_path'"],
  // Credential indirection.
  ["api_key_file", { api_key_file: 1n }, "config contains forbidden private key 'api_key_file'"],
  ["passwordPath", { passwordPath: 1n }, "config contains forbidden private key 'passwordPath'"],
  ["secret_dirs", { secret_dirs: 1n }, "config contains forbidden private key 'secret_dirs'"],
  // Location fields under a storage or backend root.
  ["storage_url", { storage_url: 1n }, "config contains forbidden private key 'storage_url'"],
  ["backend_dir", { backend_dir: 1n }, "config contains forbidden private key 'backend_dir'"],
  ["storage.path", { storage: { path: 1n } }, "config.storage contains forbidden private key 'path'"],
  ["databases[0].uri", { databases: [{ uri: 1n }] }, "config.databases[0] contains forbidden private key 'uri'"],
  // Python fields under a python token.
  ["python.path", { python: { path: 1n } }, "config.python contains forbidden private key 'path'"],
  ["python_plugin.classes", { python_plugin: { classes: 1n } }, "config.python_plugin contains forbidden private key 'classes'"],
  // Webhook fields.
  ["webhook.url", { webhook: { url: 1n } }, "config.webhook contains forbidden private key 'url'"],
  ["webhook.my_signature", { webhook: { my_signature: 1n } }, "config.webhook contains forbidden private key 'my_signature'"],
  // Case, camel and punctuation normalization.
  ["apiKey", { apiKey: 1n }, "config contains forbidden private key 'apiKey'"],
  ["API-Key", { "API-Key": 1n }, "config contains forbidden private key 'API-Key'"],
  ["APIKey", { APIKey: 1n }, "config contains forbidden private key 'APIKey'"],
  ["clientSecret", { clientSecret: 1n }, "config contains forbidden private key 'clientSecret'"],
  // Python's `isdigit` takes `²`, so `S` after it starts a `storage` token.
  ["x²Storage.path", { "x²Storage": { path: 1n } }, "config.x²Storage contains forbidden private key 'path'"],
  // Python's `casefold` gives `ss` for `ẞ` and keeps `ı`.
  ["paẞword", { "paẞword": 1n }, "config contains forbidden private key 'paẞword'"],
  // Python sorts "10" before "9"; JS enumeration would reach 'api_key' first.
  ["integer-like keys", { "9": { api_key: 1n }, "10": { token: 1n } }, "config.10 contains forbidden private key 'token'"],
  // Allowed keys.
  ["path_style", { path_style: 1n }, null],
  ["a top-level url", { url: 1n }, null],
  ["max_items", { max_items: 1n }, null],
  ["authors", { authors: 1n }, null],
  ["sigma", { sigma: 1n }, null],
  ["storage_class", { storage_class: 1n }, null],
  ["a top-level my_signature", { my_signature: 1n }, null],
  ["apı_key", { "apı_key": 1n }, null],
])("private keys: %s", (_name, config, message) => {
  if (message === null) checkMemoryDeclaration("n", "vector", config);
  else expectError(() => checkMemoryDeclaration("n", "vector", config), RangeError, message);
});

// test_builtin_allowlist_is_a_static_table
test("the built-in allowlist is a static table", () => {
  expect(new Map([...BUILTIN_MEMORY_CONFIG_KEYS].map(([k, v]) => [k, [...v].sort()]))).toEqual(
    new Map([
      ["episodic", ["max_episodes", "recall_limit"]],
      ["reflective", ["max_items", "recall_limit"]],
      ["session", ["history_budget_chars", "recent_turns"]],
      ["profile", ["input_budget_chars", "max_chars"]],
    ]),
  );
});

// test_builtin_types_accept_only_allowlisted_keys
test("built-in types accept only allowlisted keys", () => {
  decl("notes", "episodic", { max_episodes: 5n, recall_limit: 2n });
  expectError(
    () => decl("notes", "episodic", { max_items: 5n }),
    RangeError,
    "episodic memory config has unknown key(s) ['max_items'] (allowed: max_episodes, recall_limit)",
  );
  expect(() => decl("chat", "session", { budget: 1n })).toThrow("allowed: history_budget_chars, recent_turns");
  expect(() => decl("notes", "episodic", { max_items: 5n, b: 1n })).toThrow(
    "unknown key(s) ['b', 'max_items']",
  );
});

// test_builtin_config_values_are_checked
test.each<[string, Record<string, unknown>, typeof TypeError | typeof RangeError, string]>([
  ["episodic", { max_episodes: 0n }, RangeError, "max_episodes must be an integer of at least 1, got 0"],
  ["episodic", { recall_limit: -3n }, RangeError, "recall_limit must be an integer of at least 1, got -3"],
  ["reflective", { max_items: "ten" }, TypeError, 'max_items must be an integer or null, got "ten"'],
  ["reflective", { max_items: 1.5 }, TypeError, "max_items must be an integer or null, got 1.5"],
  ["reflective", { recall_limit: true }, TypeError, "recall_limit must be an integer or null, got true"],
  ["session", { recent_turns: -1n }, RangeError, "recent_turns must be an integer of at least 0, got -1"],
  ["session", { history_budget_chars: 0n }, RangeError, "history_budget_chars must be an integer of at least 1, got 0"],
  ["session", { recent_turns: null }, TypeError, "recent_turns must be an integer, got null"],
  ["profile", { max_chars: 0n }, RangeError, "max_chars must be an integer of at least 1, got 0"],
  ["profile", { input_budget_chars: null }, TypeError, "input_budget_chars must be an integer, got null"],
  ["reflective", { max_items: [1n] }, TypeError, "max_items must be an integer or null, got [1]"],
  ["reflective", { recall_limit: 3 }, TypeError, "recall_limit must be an integer or null, got 3"],
  ["reflective", { max_items: 1.0 }, TypeError, "max_items must be an integer or null, got 1"],
  // The rules table's key order decides which of two bad keys is reported.
  ["profile", { input_budget_chars: 0n, max_chars: 0n }, RangeError, "max_chars must be an integer of at least 1, got 0"],
])("%s config row %# is checked", (typeName, config, cls, message) => {
  expectError(() => checkMemoryDeclaration("notes", typeName, config), cls, `${typeName} memory config ${message}`);
});

// test_builtin_config_values_at_their_limits_load_and_build (the check half)
test.each<[string, Record<string, unknown>]>([
  ["episodic", { max_episodes: 1n, recall_limit: null }],
  ["reflective", { max_items: null, recall_limit: 1n }],
  ["session", { history_budget_chars: 1n, recent_turns: 0n }],
  ["profile", { max_chars: 1n, input_budget_chars: 1n }],
])("%s config at its limits loads", (typeName, config) => {
  checkMemoryDeclaration("notes", typeName, config);
});

/** Each built-in key's least allowed integer and whether null is allowed, as the backends take them. */
const RULES: Record<string, Record<string, [bigint, boolean]>> = {
  episodic: { max_episodes: [1n, true], recall_limit: [1n, true] },
  reflective: { max_items: [1n, true], recall_limit: [1n, true] },
  session: { history_budget_chars: [1n, false], recent_turns: [0n, false] },
  profile: { max_chars: [1n, false], input_budget_chars: [1n, false] },
};

// test_builtin_config_rules_cover_the_allowlist
test("the built-in rules cover the allowlist", () => {
  expect(new Map(Object.entries(RULES).map(([k, v]) => [k, new Set(Object.keys(v))]))).toEqual(
    new Map([...BUILTIN_MEMORY_CONFIG_KEYS].map(([k, v]) => [k, new Set(v)])),
  );
  for (const [typeName, rules] of Object.entries(RULES)) {
    for (const [key, [minimum]] of Object.entries(rules)) {
      checkMemoryDeclaration("notes", typeName, { [key]: minimum });
      expect(() => checkMemoryDeclaration("notes", typeName, { [key]: minimum - 1n })).toThrow(
        `${typeName} memory config ${key} must be an integer of at least ${minimum}, got ${minimum - 1n}`,
      );
    }
  }
});

// test_builtin_config_rules_match_the_backends (the null half)
test("an authored null is refused where the backend default isn't nullable", () => {
  for (const [typeName, rules] of Object.entries(RULES)) {
    for (const [key, [, nullable]] of Object.entries(rules)) {
      if (nullable) checkMemoryDeclaration("notes", typeName, { [key]: null });
      else expect(() => checkMemoryDeclaration("notes", typeName, { [key]: null })).toThrow(TypeError);
    }
  }
});

// test_custom_types_accept_any_public_config
test("custom types accept any public config", () => {
  expect(decl("notes", "vector", { top_k: 3n, nested: { x: 1n } }).config).toEqual({ top_k: 3n, nested: { x: 1n } });
});

// test_non_finite_config_is_rejected
test.each([Number.NaN, Number.POSITIVE_INFINITY])("non-finite config %p is rejected", (x) => {
  expectError(() => decl("notes", "vector", { x }), RangeError, "config.x JSON number must be finite");
});

// test_shape_check_is_public_and_exported (the check half)
test("the shape check is public", () => {
  checkMemoryDeclaration("notes", "episodic", { recall_limit: 2n });
  expect(() => checkMemoryDeclaration("a.b", "episodic", {})).toThrow("memory name");
});

// test_setup_error_carries_related_origins
test("a setup error carries related origins", () => {
  const first = new OriginRef("a");
  const second = new OriginRef("b");
  const related = [second];
  const error = new MemorySetupError("clash", { origin: first, related });
  related.push(first);
  expect(error.name).toBe("MemorySetupError");
  expect(error.origin).toBe(first);
  expect(error.related).toEqual([second]);
  expect(error.related[0]).toBe(second);
  expect(Object.isFrozen(error.related)).toBe(true);
  expect(new MemorySetupError("plain").related).toEqual([]);
  expect(new MemorySetupError("").origin).toBeNull();
  expectError(
    () => new MemorySetupError("bad", { related: new Set([second]) as never }),
    TypeError,
    "related must be an array of OriginRef values",
  );
  expectError(
    () => new MemorySetupError("bad", { related: [null] as never }),
    TypeError,
    "related must not contain null",
  );
  expectError(
    () => new MemorySetupError("bad", { related: ["x"] as never }),
    TypeError,
    "related origin must be an OriginRef or null",
  );
  expectError(() => new MemorySetupError(1n as never), TypeError, "message must be a string");
  expectError(
    () => new MemorySetupError("bad", { origin: "x" as never }),
    TypeError,
    "origin must be an OriginRef or null",
  );
});
