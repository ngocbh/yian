/**
 * Authored flow memory identity in the manifest (`compile/manifest.ts`): rows of
 * `tests/engine/test_memory_authoring.py`.
 *
 * Ported: `test_python_constructors_default_and_validate_authored_memory_fields` (the
 * `CompiledFlow` half), `test_authored_identity_includes_flow_id_and_policy_but_excludes_origin`,
 * `test_decl_config_rejects_host_private_fields`,
 * `test_policy_config_keeps_public_behavior_options`,
 * `test_manifest_keeps_every_value_of_a_forged_policy_config` and
 * `test_expansion_digest_excludes_flow_id_and_agent_memory`.
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
 * Also ported: the AGENT half of `test_python_constructors_default_and_validate_authored_memory_fields`
 * (`memories` is an array where Python's is a tuple).
 *
 * The harvested `load-agent/engine_memory_authoring__retired_memory_policy_on_an_agent` (the
 * `ordinary-agent` row of `test_retired_memory_policy_is_a_located_migration_error`) is pinned,
 * as `parseNodes` reads it; so is the ok case
 * `load-agent/engine_memory_authoring__inline_body_agent_memory_loads`, through `assembled`.
 *
 * Ported through `loadFlow` (a CODE reference is `./fixtures/compose_codefns.ts:echo`, loaded
 * against the conformance dir): `test_full_root_retains_flow_owned_policy_with_one_origin` (2),
 * `test_full_nested_definition_retains_flow_owned_policy_and_semantic_id`,
 * `test_compact_root_projects_policy_to_its_agent`,
 * `test_flat_agent_definition_projects_policy_to_its_agent`,
 * `test_compose_stamps_decl_sites_for_root_def_and_inline_owners`,
 * `test_non_agent_node_rejects_long_term_memory_at_the_authored_value` (2),
 * `test_malformed_flow_declaration_is_located_at_its_authored_occurrence` (7), the other six rows
 * of `test_retired_memory_policy_is_a_located_migration_error`,
 * `test_inline_body_rejects_flow_policy_but_allows_child_agent_policy`,
 * `test_authored_policy_config_rejects_host_private_fields` (each config's path and key as in
 * the decl rows) and `test_compose_loaded_flow_id_changes_the_previous_memoryless_identity`. The
 * error rows pin the whole message, where Python matches a pattern; config ints are `bigint`; a
 * node without memories has no own `memories` field, as Python's `__dict__` check reads it.
 */

import { afterEach, expect, test } from "bun:test";

import { expansionDigest, flowFingerprint, flowManifest } from "../../src/compile/manifest.ts";
import { CompiledFlow, copyGraph } from "../../src/compile/model.ts";
import { forget } from "../../src/defaults.ts";
import type { LoadError } from "../../src/compose/helpers/errors.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { OriginRef, SourceRange } from "../../src/diagnostics.ts";
import { DeclSite, MemoryDecl } from "../../src/memory/contracts.ts";
import { AgentNode } from "../../src/nodes/agent/node.ts";
import type { CallNode } from "../../src/nodes/call/node.ts";
import type { JsonObject } from "../../src/typesys/json.ts";
import {
  assembled,
  type CaseRow,
  loadConformanceFlow,
  loadErrorAsync,
  parsedNodes,
  runCase,
} from "../compose/declarations.ts";

afterEach(() => forget());

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

/** A full flow named `memory-root` with `policy` inserted at the flow level. */
function fullFlow(policy: string): string {
  return (
    "id: memory-root\n" +
    "name: memory_root\n" +
    policy +
    "nodes:\n" +
    "  echo:\n" +
    "    kind: code\n" +
    "    code: ./fixtures/compose_codefns.ts:echo\n" +
    "    input: {topic: hello}\n" +
    "    output: str\n" +
    "output: ${echo.output}\n"
  );
}

/** The baked child flow of CALL `nodeId` in `flow`. */
const childOf = (flow: CompiledFlow, nodeId: string) => (flow.nodes.get(nodeId) as CallNode).child as CompiledFlow;

/** The memories of AGENT `nodeId` in `flow`. */
const agentMemories = (flow: CompiledFlow, nodeId: string) => (flow.nodes.get(nodeId) as AgentNode).memories;

/** Expect the last frame of `error` to start, at line `line` (default: its own), with `prefix`. */
function expectFrameLine(error: LoadError, prefix: string, line?: number): void {
  const frame = error.diagnostic.frames.at(-1)!;
  if (line !== undefined) expect(frame.range.startLine).toBe(line);
  expect(frame.text.split("\n")[frame.range.startLine - 1]!.trim().startsWith(prefix)).toBe(true);
}

test.each([
  ["string", "long_term_memory: reflective\n", {}, 3],
  [
    "object",
    "long_term_memory:\n  reflective:\n    type: reflective\n    config:\n      max_items: 20\n      recall_limit: 5\n",
    { max_items: 20n, recall_limit: 5n },
    4,
  ],
] as const)("a full root retains its flow-owned policy with one origin (%s)", async (_id, policy, config, line) => {
  const loaded = await loadConformanceFlow(fullFlow(policy), { sourceLabel: "memory.yaml" });

  expect(loaded.compiled.memories).toHaveLength(1);
  const decl = loaded.compiled.memories[0]!;
  expect(loaded.compiled.flowId).toBe("memory-root");
  expect(decl.constructor).toBe(MemoryDecl);
  expect([decl.name, decl.type, decl.key]).toEqual(["reflective", "reflective", "reflective"]);
  expect(decl.site).toEqual(ROOT);
  expect(decl.config).toEqual(config);
  expect(decl.origin).toBeInstanceOf(OriginRef);
  expect("memories" in loaded.compiled.nodes.get("echo")!).toBe(false);
  const frame = loaded.originResolver!.resolve(decl.origin!)!;
  expect(frame.label).toBe("memory.yaml");
  expect(frame.range.startLine).toBe(line);
});

test("a full nested definition retains its flow-owned policy and semantic id", async () => {
  const loaded = await loadConformanceFlow(`id: parent
name: parent
defs:
  child:
    long_term_memory:
      reflective: {type: reflective, config: {max_items: 8}}
    nodes:
      answer:
        kind: code
        code: ./fixtures/compose_codefns.ts:echo
        input: {topic: child}
        output: str
    output: \${answer.output}
nodes:
  invoke:
    kind: call
    call: child
output: \${invoke.output}
`);

  const child = childOf(loaded.compiled, "invoke");
  expect(child.flowId).toBe("child");
  expect(child.memories[0]!.name).toBe("reflective");
  expect(child.memories[0]!.config).toEqual({ max_items: 8n });
  expect("memories" in child.nodes.get("answer")!).toBe(false);
});

test("a compact root projects its policy to its agent", async () => {
  const loaded = await loadFlow(`id: compact_agent
name: compact_agent
long_term_memory: reflective
output: str
kind: agent
prompt: answer
`);

  expect(loaded.compiled.flowId).toBe("compact_agent");
  expect(loaded.compiled.memories).toEqual([]);
  const agent = loaded.compiled.nodes.get("compact_agent")!;
  expect(agent.constructor).toBe(AgentNode);
  expect(agentMemories(loaded.compiled, "compact_agent")[0]!.name).toBe("reflective");
});

test("a flat agent definition projects its policy to its agent", async () => {
  const loaded = await loadFlow(`id: parent
name: parent
defs:
  child:
    kind: agent
    long_term_memory: reflective
    prompt: answer
    output: str
nodes:
  invoke:
    kind: call
    call: child
output: \${invoke.output}
`);

  const child = childOf(loaded.compiled, "invoke");
  expect(child.flowId).toBe("child");
  expect(child.memories).toEqual([]);
  expect(agentMemories(child, "child")[0]!.name).toBe("reflective");
});

test("compose stamps decl sites for root, def and inline owners", async () => {
  const { compiled } = await loadFlow(`id: parent
name: parent
long_term_memory: outer
defs:
  child:
    kind: agent
    long_term_memory: inner
    prompt: answer
    output: str
nodes:
  top:
    kind: agent
    long_term_memory: local
    prompt: answer
    output: str
  invoke:
    kind: call
    call: child
  wrapper:
    kind: call
    nodes:
      answer:
        kind: agent
        long_term_memory: inline
        prompt: answer
        output: str
    output: \${answer.output}
output: \${top.output}
`);

  expect(compiled.memories.map((decl) => decl.site)).toEqual([ROOT]);
  expect(agentMemories(compiled, "top")[0]!.site).toEqual(new DeclSite([], ["root"], "top"));
  const child = childOf(compiled, "invoke");
  expect(child.memories).toEqual([]);
  expect(agentMemories(child, "child")[0]!.site).toEqual(new DeclSite([], ["def", "child"], "child"));
  const inline = childOf(compiled, "wrapper");
  expect(agentMemories(inline, "answer")[0]!.site).toEqual(
    new DeclSite([], ["root", "node", "wrapper", "inline"], "answer"),
  );
});

test.each([
  [
    "compact",
    `id: compact_code
name: compact_code
long_term_memory: reflective
output: str
kind: code
code: ./fixtures/compose_codefns.ts:echo
`,
    "node 'compact_code' (kind=code): field 'long_term_memory' is not allowed (allowed: asserts, code, " +
      "depends_on, env, input, kind, node_name, output, runs_after)",
  ],
  [
    "ordinary",
    `id: ordinary-code
name: ordinary_code
nodes:
  echo:
    kind: code
    long_term_memory: reflective
    code: ./fixtures/compose_codefns.ts:echo
    output: str
output: \${echo.output}
`,
    "node 'echo' (kind=code): field 'long_term_memory' is not allowed (allowed: asserts, code, depends_on, " +
      "env, input, kind, node_name, output, runs_after)",
  ],
])("a non-agent node rejects long_term_memory at the authored value (%s)", async (_id, source, message) => {
  const error = await loadErrorAsync(loadConformanceFlow(source));

  expect(error.message).toBe(message);
  expectFrameLine(error, "long_term_memory:");
});

test.each([
  ["long_term_memory: ''\n", "long_term_memory '': memory name '' must match [A-Za-z_][A-Za-z0-9_-]*"],
  ["long_term_memory: {}\n", "long_term_memory: {} declares no memory; remove the key or name one"],
  ["long_term_memory: {name: reflective}\n", "long_term_memory name 'name' is reserved (reserved: config, name, type)"],
  [
    "long_term_memory: {name: reflective, config: {max_items: 3}}\n",
    "long_term_memory name 'name' is reserved (reserved: config, name, type)",
  ],
  ["long_term_memory: {notes: {type: reflective, config: []}}\n", "long_term_memory 'notes' config must be a mapping"],
  [
    "long_term_memory: {notes: {type: reflective, extra: true}}\n",
    'long_term_memory \'notes\' has unknown key(s) ["extra"] (allowed: type, config)',
  ],
  [
    "long_term_memory: {notes: {type: reflective, config: {max_items: 0}}}\n",
    "long_term_memory 'notes': reflective memory config max_items must be an integer of at least 1, got 0",
  ],
])("a malformed flow declaration is located at its authored occurrence: %s", async (surface, message) => {
  const error = await loadErrorAsync(loadConformanceFlow(fullFlow(surface)));

  expect(error.message).toBe(message);
  expect(error.diagnostic.frames.at(-1)!.range.startLine).toBe(3);
});

const TOP_LEVEL_RENAMED =
  "unknown top-level key 'memory_policy' — `memory_policy:` was renamed to `long_term_memory:`";

test.each([
  ["full-root", "id: full\nname: full\nmemory_policy: reflective\nnodes: {}\n", 3, TOP_LEVEL_RENAMED],
  [
    "compact-root",
    "id: compact\nname: compact\nmemory_policy: reflective\nkind: agent\nprompt: answer\noutput: str\n",
    3,
    TOP_LEVEL_RENAMED,
  ],
  [
    "multi-node-def",
    "id: parent\nname: parent\ndefs:\n  child:\n    memory_policy: reflective\n    nodes: {}\nnodes: {}\n",
    5,
    "defs entry 'child': field `memory_policy:` was renamed to `long_term_memory:`",
  ],
  [
    "flat-agent-def",
    "id: parent\nname: parent\ndefs:\n  child:\n    kind: agent\n    memory_policy: reflective\n" +
      "    prompt: answer\n    output: str\nnodes: {}\n",
    6,
    "defs entry 'child': node field `memory_policy:` was renamed to `long_term_memory:`",
  ],
  [
    "both-old-first",
    "id: both-old-first\nname: both_old_first\nmemory_policy: reflective\nlong_term_memory: episodic\nnodes: {}\n",
    3,
    TOP_LEVEL_RENAMED,
  ],
  [
    "both-new-first",
    "id: both-new-first\nname: both_new_first\nlong_term_memory: episodic\nmemory_policy: reflective\nnodes: {}\n",
    4,
    TOP_LEVEL_RENAMED,
  ],
])("a retired memory_policy is a located migration error (%s)", async (_id, source, line, message) => {
  const error = await loadErrorAsync(loadFlow(source));

  expect(error.message).toBe(message);
  expectFrameLine(error, "memory_policy:", line);
});

test("an inline body rejects a flow policy but allows a child agent's policy", async () => {
  const rejected = `id: inline-policy
name: inline_policy
nodes:
  wrapper:
    kind: call
    long_term_memory: reflective
    nodes:
      answer:
        kind: agent
        prompt: answer
        output: str
    output: \${answer.output}
output: \${wrapper.output}
`;
  const error = await loadErrorAsync(loadFlow(rejected));
  expect(error.message).toBe(
    "node 'wrapper' (kind=call): field 'long_term_memory' is not allowed (allowed: asserts, call, depends_on, " +
      "env, input, kind, node_name, runs_after)",
  );
  expect(error.diagnostic.frames.at(-1)!.range).toEqual(new SourceRange(6, 23, 6, 33));

  const allowed = rejected
    .replace("    long_term_memory: reflective\n", "")
    .replace("        prompt: answer\n", "        long_term_memory: reflective\n        prompt: answer\n");
  const child = childOf((await loadFlow(allowed)).compiled, "wrapper");
  expect(child.memories).toEqual([]);
  expect(agentMemories(child, "answer")[0]!.name).toBe("reflective");
});

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
  const agentPolicy = new CompiledFlow(
    new Map([
      [
        "agent",
        new AgentNode("agent", {
          memories: [new MemoryDecl("reflective", "reflective", {}, new DeclSite([], ["root"], "agent"))],
        }),
      ],
    ]),
    [],
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
  expect(flowFingerprint(agentPolicy)).not.toBe(flowFingerprint(plain));
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
  const agentManifest = (flowManifest(agentPolicy).nodes as JsonObject).agent as JsonObject;
  expect(agentManifest.memories).toEqual([
    {
      config: { entries: [], type: "mapping" },
      name: "reflective",
      site: { node: "agent", owner: ["root"], scope: [] },
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

test.each(PRIVATE_CONFIG_CASES)(
  "authored policy config rejects host private fields (%s %s)",
  async (where, key, config) => {
    const source = fullFlow(
      `long_term_memory:\n  notes:\n    type: reflective\n    config: ${JSON.stringify(config)}\n`,
    );

    const error = await loadErrorAsync(loadConformanceFlow(source, { sourceLabel: "memory.yaml" }));

    expect(error.message).toBe(`long_term_memory 'notes': ${where} contains forbidden private key '${key}'`);
  },
);

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

test("expansion digest excludes flow id and agent memory", () => {
  const plainFlow = new CompiledFlow(new Map(), []);
  const namedFlow = new CompiledFlow(new Map(), [], { flowId: "named" });
  const plainAgent = new CompiledFlow(new Map([["agent", new AgentNode("agent")]]), []);
  const decl = new MemoryDecl("reflective", "reflective", { max_items: 4n }, new DeclSite([], ["root"], "agent"));
  const memoryAgent = new CompiledFlow(new Map([["agent", new AgentNode("agent", { memories: [decl] })]]), []);

  expect(expansionDigest(namedFlow)).toBe(expansionDigest(plainFlow));
  expect(expansionDigest(memoryAgent)).toBe(expansionDigest(plainAgent));
});

test("a compose-loaded flow id changes the previous memoryless identity", async () => {
  const first = (await loadFlow("id: first\nname: same\nnodes: {}\n")).compiled;
  const second = (await loadFlow("id: second\nname: same\nnodes: {}\n")).compiled;

  expect(first.flowId).toBe("first");
  expect(second.flowId).toBe("second");
  expect(flowFingerprint(first)).not.toBe(flowFingerprint(second));
});

test("the AGENT constructor defaults and validates authored memory fields", () => {
  expect(new AgentNode("agent").memories).toEqual([]);

  const flowDecl = new MemoryDecl("reflective", "reflective", {}, new DeclSite([], ["root"], null));
  const agentDecl = new MemoryDecl("reflective", "reflective", {}, new DeclSite([], ["root"], "agent"));
  expect(new AgentNode("agent", { memories: [agentDecl] }).memories[0]).toBe(agentDecl);
  expect(() => new AgentNode("agent", { memories: [flowDecl] })).toThrow(
    new RangeError("AGENT memories must carry AGENT DeclSites"),
  );

  class DerivedDecl extends MemoryDecl {}
  const derivedAgent = new DerivedDecl("reflective", "reflective", {}, new DeclSite([], ["root"], "agent"));
  expect(() => new AgentNode("agent", { memories: [derivedAgent] })).toThrow(TypeError);
  expect(() => new AgentNode("agent", { memories: [derivedAgent] })).toThrow("exact MemoryDecl");
});

test.each<CaseRow>([
  { id: "load-agent/engine_memory_authoring__retired_memory_policy_on_an_agent", call: parsedNodes },
])("parseNodes: $id", runCase);

test.each<CaseRow>([{ id: "load-agent/engine_memory_authoring__inline_body_agent_memory_loads", call: assembled }])(
  "assemble: $id",
  runCase,
);
