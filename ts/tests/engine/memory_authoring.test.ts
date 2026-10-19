/**
 * Authored flow memory identity in the manifest (`compile/manifest.ts`): rows of
 * `tests/engine/test_memory_authoring.py`.
 *
 * Ported: `test_python_constructors_default_and_validate_authored_memory_fields` (the
 * `CompiledFlow` half), `test_authored_identity_includes_flow_id_and_policy_but_excludes_origin`
 * (flow memories), `test_decl_config_rejects_host_private_fields`,
 * `test_policy_config_keeps_public_behavior_options`,
 * `test_manifest_keeps_every_value_of_a_forged_policy_config` and
 * `test_expansion_digest_excludes_flow_id_and_agent_memory` (the flow-id half).
 *
 * Changed:
 * - `CompiledFlow({}, [])` is `new CompiledFlow(new Map(), [])`, Python's `copy.deepcopy` is
 *   `copyGraph`, and ints in memory configs are `bigint` (`4n`), since a built-in key refuses a
 *   `number`.
 * - `test_python_constructors_default_and_validate_authored_memory_fields`: Python's str subclass
 *   is `new String("flow")`, and its list-for-tuple case has no TS form, since TS only has arrays.
 * - `test_decl_config_rejects_host_private_fields` pins the class and the exact message, with each
 *   config's key path stored next to it, where Python matches "forbidden private key".
 * - `test_manifest_keeps_every_value_of_a_forged_policy_config`: a `MemoryDecl` is frozen and keeps
 *   no private `_config`, so the row forges a `MemoryDecl` (its prototype plus own fields) whose
 *   `config` holds `api_token`, a key the constructor refuses, then swaps that config.
 *
 * Later (T3-22, with AGENT): the `agent_policy` lines of
 * `test_authored_identity_includes_flow_id_and_policy_but_excludes_origin`, the AGENT-memory half
 * of `test_expansion_digest_excludes_flow_id_and_agent_memory`, and the AGENT half of
 * `test_python_constructors_default_and_validate_authored_memory_fields`.
 *
 * Later (T3-22): `test_full_root_retains_flow_owned_policy_with_one_origin`,
 * `test_full_nested_definition_retains_flow_owned_policy_and_semantic_id`,
 * `test_compact_root_projects_policy_to_its_agent`,
 * `test_flat_agent_definition_projects_policy_to_its_agent`,
 * `test_compose_stamps_decl_sites_for_root_def_and_inline_owners`,
 * `test_non_agent_node_rejects_long_term_memory_at_the_authored_value`,
 * `test_malformed_flow_declaration_is_located_at_its_authored_occurrence`,
 * `test_retired_memory_policy_is_a_located_migration_error`,
 * `test_inline_body_rejects_flow_policy_but_allows_child_agent_policy`,
 * `test_authored_policy_config_rejects_host_private_fields` and
 * `test_compose_loaded_flow_id_changes_the_previous_memoryless_identity` (they load flows).
 */

import { expect, test } from "bun:test";

import { expansionDigest, flowFingerprint, flowManifest } from "../../src/compile/manifest.ts";
import { CompiledFlow, copyGraph } from "../../src/compile/model.ts";
import { OriginRef } from "../../src/diagnostics.ts";
import { DeclSite, MemoryDecl } from "../../src/memory/contracts.ts";

const ROOT = new DeclSite([], ["root"], null);

/** Python's private-key configs, each with the path and key the refusal names. */
const PRIVATE_CONFIG_CASES: [string, string, Record<string, unknown>][] = [
  ["config.nested", "clientSecret", { nested: { clientSecret: "private" } }],
  ["config.nested", "secretKey", { nested: { secretKey: "private" } }],
  ["config.nested", "PASSWORD", { nested: { PASSWORD: "private" } }],
  ["config.nested", "passphrase", { nested: { passphrase: "private" } }],
  ["config.nested", "passwd", { nested: { passwd: "private" } }],
  ["config.nested", "pwd", { nested: { pwd: "private" } }],
  ["config.nested", "access-token", { nested: { "access-token": "private" } }],
  ["config.nested", "refreshToken", { nested: { refreshToken: "private" } }],
  ["config.nested", "session_key", { nested: { session_key: "private" } }],
  ["config.nested", "subscriptionKey", { nested: { subscriptionKey: "private" } }],
  ["config.nested", "authToken", { nested: { authToken: "private" } }],
  ["config.nested", "bearer-key", { nested: { "bearer-key": "private" } }],
  ["config.nested", "api-key", { nested: { "api-key": "private" } }],
  ["config.nested", "apiKey", { nested: { apiKey: "private" } }],
  ["config.nested", "backend_token", { nested: { backend_token: "private" } }],
  ["config.nested", "Authorization", { nested: { Authorization: "private" } }],
  ["config.nested", "private-key", { nested: { "private-key": "private" } }],
  ["config.nested", "connectionString", { nested: { connectionString: "private" } }],
  ["config.nested", "credentials", { nested: { credentials: "private" } }],
  ["config.nested", "webhookURL", { nested: { webhookURL: "private" } }],
  ["config.nested", "webhookSignature", { nested: { webhookSignature: "private" } }],
  ["config.webhook", "signature", { webhook: { signature: "private" } }],
  ["config.nested", "tenant_id", { nested: { tenant_id: "private" } }],
  ["config.nested", "orgTenantId", { nested: { orgTenantId: "private" } }],
  ["config.nested", "pythonImport", { nested: { pythonImport: "package.policy" } }],
  ["config.nested", "pythonImports", { nested: { pythonImports: ["package.policy"] } }],
  ["config.python", "imports", { python: { imports: ["package.policy"] } }],
  ["config.nested", "modulePath", { nested: { modulePath: "package.policy" } }],
  ["config.nested", "class-path", { nested: { "class-path": "package.Policy" } }],
  ["config.nested", "secretToken", { nested: { secretToken: "private" } }],
  ["config.nested", "secretTokens", { nested: { secretTokens: ["private"] } }],
  ["config.nested", "apiKeys", { nested: { apiKeys: ["private"] } }],
  ["config.nested", "backendPaths", { nested: { backendPaths: ["/private/backend"] } }],
  ["config.nested", "pythonModules", { nested: { pythonModules: ["package.policy"] } }],
  ["config.nested", "credentialFile", { nested: { credentialFile: "/private/credentials" } }],
  ["config.nested", "apiKeyPath", { nested: { apiKeyPath: "/private/api-key" } }],
  ["config.nested", "orgtenantid", { nested: { orgtenantid: "private" } }],
  ["config.nested", "backendDir", { nested: { backendDir: "/private/backend" } }],
  ["config.nested", "storageDirectory", { nested: { storageDirectory: "/private/storage" } }],
  ["config.backend", "path", { backend: { path: "/private/backend" } }],
  ["config.backend", "directory", { backend: { directory: "/private/backend" } }],
  ["config.backend", "connection", { backend: { connection: "private-connection" } }],
  ["config.storage", "location", { storage: { location: "/private/storage" } }],
  ["config.database", "dsn", { database: { dsn: "private-dsn" } }],
  ["config.persistence", "uri", { persistence: { uri: "private://state" } }],
];

/** Assert `fn` throws exactly `cls` (not a subclass) with exactly `message`. */
function expectError(fn: () => unknown, cls: typeof Error, message: string): void {
  let caught: unknown;
  try {
    fn();
  } catch (error) {
    caught = error;
  }
  expect(caught).toBeInstanceOf(cls);
  expect((caught as Error).constructor).toBe(cls);
  expect((caught as Error).message).toBe(message);
}

/** Return a node-free compiled flow holding `memories`. */
function flowWith(...memories: MemoryDecl[]): CompiledFlow {
  return new CompiledFlow(new Map(), [], { memories });
}

test("compiled flow constructor defaults and validates authored memory fields", () => {
  const flow = new CompiledFlow(new Map(), []);

  expect(flow.flowId).toBeNull();
  expect(flow.memories).toEqual([]);

  const flowDecl = new MemoryDecl("reflective", "reflective", {}, ROOT);
  expect(new CompiledFlow(new Map(), [], { flowId: "flow", memories: [flowDecl] }).memories).toEqual([flowDecl]);
  expect(flowWith(flowDecl).memories[0]).toBe(flowDecl);
});

class DerivedDecl extends MemoryDecl {}

test.each([
  ["empty flowId", { flowId: "" }, RangeError, "flow_id must be non-empty"],
  ["number flowId", { flowId: 1 }, TypeError, "flow_id must be an exact string or null"],
  ["boolean flowId", { flowId: true }, TypeError, "flow_id must be an exact string or null"],
  ["String object flowId", { flowId: new String("flow") }, TypeError, "flow_id must be an exact string or null"],
  [
    "agent-site declaration",
    { memories: [new MemoryDecl("reflective", "reflective", {}, new DeclSite([], ["root"], "agent"))] },
    RangeError,
    "flow memories must carry flow DeclSites (node=null)",
  ],
  [
    "MemoryDecl subclass",
    { memories: [new DerivedDecl("reflective", "reflective", {}, ROOT)] },
    TypeError,
    "memories must be an array of exact MemoryDecl values",
  ],
] as const)("compiled flow constructor refuses %s", (_name, options, cls, message) => {
  const authored = options as ConstructorParameters<typeof CompiledFlow>[2];
  expectError(() => new CompiledFlow(new Map(), [], authored), cls, message);
});

test("authored identity includes flow id and policy but excludes origin", () => {
  const plain = new CompiledFlow(new Map(), []);
  const samePlain = copyGraph(plain);
  const flowNamed = new CompiledFlow(new Map(), [], { flowId: "named" });
  const flowPolicy = flowWith(new MemoryDecl("reflective", "reflective", { max_items: 4n }, ROOT));
  const flowPolicyRelocated = flowWith(
    new MemoryDecl("reflective", "reflective", { max_items: 4n }, ROOT, new OriginRef("other/source")),
  );
  const flowPolicyRenamed = flowWith(new MemoryDecl("custom", "custom", { max_items: 4n }, ROOT));
  const flowPolicyReconfigured = flowWith(new MemoryDecl("reflective", "reflective", { max_items: 5n }, ROOT));
  const flowPolicyResited = flowWith(
    new MemoryDecl("reflective", "reflective", { max_items: 4n }, new DeclSite([], ["def", "x"], null)),
  );

  const plainManifest = flowManifest(plain);
  expect("flow_id" in plainManifest).toBe(false);
  expect("memories" in plainManifest).toBe(false);
  expect(flowManifest(samePlain)).toEqual(plainManifest);
  expect(flowFingerprint(samePlain)).toBe(flowFingerprint(plain));
  expect(flowFingerprint(flowNamed)).not.toBe(flowFingerprint(plain));
  expect(flowFingerprint(flowPolicy)).not.toBe(flowFingerprint(plain));
  expect(flowFingerprint(flowPolicyRelocated)).toBe(flowFingerprint(flowPolicy));
  expect(flowFingerprint(flowPolicyRenamed)).not.toBe(flowFingerprint(flowPolicy));
  expect(flowFingerprint(flowPolicyReconfigured)).not.toBe(flowFingerprint(flowPolicy));
  expect(flowFingerprint(flowPolicyResited)).not.toBe(flowFingerprint(flowPolicy));
  expect(flowManifest(flowPolicy).memories).toEqual([
    {
      config: {
        entries: [{ key: { type: "str", value: "max_items" }, value: { type: "int", value: "4" } }],
        type: "mapping",
      },
      name: "reflective",
      site: { node: null, owner: ["root"], scope: [] },
      type: "reflective",
    },
  ]);
  expect(expansionDigest(flowPolicy)).toBe(expansionDigest(plain));
});

test.each(PRIVATE_CONFIG_CASES)("decl config rejects host private fields (%s %s)", (where, key, config) => {
  expectError(
    () => new MemoryDecl("notes", "custom", config, ROOT),
    RangeError,
    `${where} contains forbidden private key '${key}'`,
  );
});

test("policy config keeps public behavior options", () => {
  const config = {
    max_items: 20n,
    recall_limit: 5n,
    token_budget: 4096n,
    stop_token: "semantic-stop-marker",
    continuationToken: "semantic-continuation-marker",
    query_options: { strategy: "semantic", path_filter: "docs/**" },
    backend: { mode: "memory" },
    database_query_options: { timeout: 3n },
  };

  const policy = new MemoryDecl("custom", "custom", config, ROOT);

  expect(policy.config).toEqual(config);
  const rendered = JSON.stringify(flowManifest(flowWith(policy)));
  expect(rendered).toContain("semantic-stop-marker");
  expect(rendered).toContain("semantic-continuation-marker");
});

test("manifest keeps every value of a forged policy config", () => {
  const policy: MemoryDecl = Object.assign(Object.create(MemoryDecl.prototype), {
    name: "reflective",
    type: "reflective",
    config: { nested: { api_token: "first-forged-value" } },
    site: ROOT,
    origin: null,
  });
  const flow = flowWith(policy);

  const firstManifest = flowManifest(flow);
  expect(JSON.stringify(firstManifest)).toContain("first-forged-value");
  Object.assign(policy, { config: { nested: { api_token: "second-forged-value" } } });
  expect(flowManifest(flow)).not.toEqual(firstManifest);
});

test("expansion digest excludes flow id", () => {
  const plainFlow = new CompiledFlow(new Map(), []);
  const namedFlow = new CompiledFlow(new Map(), [], { flowId: "named" });

  expect(expansionDigest(namedFlow)).toBe(expansionDigest(plainFlow));
});
