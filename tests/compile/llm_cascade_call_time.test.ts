/**
 * Call-time `llm_config` agents: the layers each one gets as its base, applying the resolved
 * values, and the host credentials reaching them on the private flow only.
 */

import { describe, expect, test } from "bun:test";

import {
  applyEffectiveLlmMap,
  bindHostCredentials,
  callTimeSelection,
  knownCallTimeProviders,
  type RunLLM,
  resolveRunLlm,
} from "../../src/compile/llm_cascade.ts";
import { flowFingerprint } from "../../src/compile/manifest.ts";
import { type CompiledFlow, copyGraph } from "../../src/compile/model.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { HostConnection, popHostCredentials } from "../../src/llm_clients/config.ts";
import {
  type Connection,
  ConnectionSnapshot,
  recordSelection,
  restoreSelection,
  type SelectionRecord,
} from "../../src/llm_clients/connection.ts";
import type { AgentNode } from "../../src/nodes/agent/node.ts";

const DEFAULTS = { provider: "openai", model: "default-model" };

const ROOT = `id: f
name: f
llm_config: {provider: openai, openai: {temperature: 0.5}}
nodes:
  pick: {kind: agent, prompt: Pick., output: str, llm_config: {model: pick-model}}
  chat: {kind: agent, prompt: Go., llm_config: {model: '\${pick.output}'}}
  solo:
    kind: agent
    prompt: Go.
    llm_config: {inherit: false, provider: anthropic, model: '\${pick.output}'}
output: \${chat.output}
`;

const NESTED = `id: f
name: f
llm_config: {provider: anthropic}
defs:
  child:
    input: {m: !str }
    llm_config: {anthropic: {temperature: 0.3}}
    nodes:
      a: {kind: agent, prompt: Go., llm_config: {model: '\${input.m}'}}
      b: {kind: agent, prompt: Go., llm_config: {model: b-model}}
    output: \${a.output}
nodes:
  c: {kind: call, call: child, input: {m: x}}
output: \${c.output}
`;

/** Load `text`; resolve a private copy of it under the host layer `host`. */
async function resolved(
  text: string,
  host: Record<string, unknown> = {},
): Promise<[flow: CompiledFlow, privateFlow: CompiledFlow, runLlm: RunLLM]> {
  const flow = (await loadFlow(text)).compiled;
  const privateFlow = copyGraph(flow);
  return [flow, privateFlow, resolveRunLlm(privateFlow, { ...host }, { defaults: DEFAULTS })];
}

/** Return a `RunStarted.call_time_llm` record holding `agents`. */
function callTimeRecord(
  agents: Record<string, unknown>,
  hostKeyProvider: string | null = null,
): { selection: SelectionRecord; agents: Record<string, unknown> } {
  return { selection: recordSelection(snapshotWith(), { providers: null, hostKeyProvider }), agents };
}

/** Return the AGENT at static path `path` (callsite ids, then the node id, joined with `/`). */
function agentAt(flow: CompiledFlow, path: string): AgentNode {
  const parts = path.split("/");
  let current = flow;
  for (const callsite of parts.slice(0, -1)) {
    current = (current.nodes.get(callsite) as unknown as { child: CompiledFlow }).child;
  }
  return current.nodes.get(parts.at(-1)!) as AgentNode;
}

/** Return the error `run` throws, failing the test when it returns. */
function thrown(run: () => unknown): Error {
  try {
    run();
  } catch (error) {
    return error as Error;
  }
  throw new Error("expected an error");
}

/** Expect `run` to throw an error of class `kind` whose message matches `pattern`. */
function expectThrow(run: () => unknown, kind: typeof RangeError | typeof TypeError, pattern: RegExp | string): void {
  const error = thrown(run);
  expect(error).toBeInstanceOf(kind);
  if (typeof pattern === "string") expect(error.message).toContain(pattern);
  else expect(error.message).toMatch(pattern);
}

test("a root call-time agent gets the host layer as its base", async () => {
  const [, , runLlm] = await resolved(ROOT, { model: "host-model" });
  expect(Object.keys(runLlm.effective)).toEqual(["pick"]);
  // The run's flow layer rides in the agent's own reserved binding, not in its base.
  expect(runLlm.callTime).toEqual({ chat: { model: "host-model" }, solo: {} });
});

test("a nested call-time agent gets the outer layers without its own flow layer", async () => {
  const [, , runLlm] = await resolved(NESTED, { model: "host-model" });
  expect(Object.keys(runLlm.effective)).toEqual(["c/b"]);
  expect(runLlm.callTime).toEqual({ "c/a": { provider: "anthropic", model: "host-model" } });
});

test("apply sets each call-time agent and leaves it unresolved", async () => {
  const [, , runLlm] = await resolved(NESTED, { model: "host-model" });
  const fresh = copyGraph((await loadFlow(NESTED)).compiled);
  applyEffectiveLlmMap(fresh, runLlm.effective, callTimeRecord(runLlm.callTime));
  const a = agentAt(fresh, "c/a");
  const b = agentAt(fresh, "c/b");
  expect(a.callTime!.path).toBe("c/a");
  expect(a.callTime!.base).toEqual({ provider: "anthropic", model: "host-model" });
  expect(a.callTime!.selection.snapshot).toEqual(snapshotWith());
  expect(a.llmConfig).toBeNull();
  expect(b.callTime).toBeNull();
  expect(b.llmConfig).toEqual(runLlm.effective["c/b"]!);
  // Each node holds its own copy.
  (a.callTime!.base as Record<string, unknown>)["model"] = "changed";
  expect(runLlm.callTime["c/a"]!["model"]).toBe("host-model");
});

test("the two maps must cover every agent once", async () => {
  const [flow, , runLlm] = await resolved(ROOT);
  const { effective, callTime } = runLlm;
  const withoutChat = Object.fromEntries(Object.entries(callTime).filter(([key]) => key !== "chat"));
  expectThrow(
    () => applyEffectiveLlmMap(copyGraph(flow), effective, callTimeRecord(withoutChat)),
    RangeError,
    'missing paths: ["chat"]',
  );
  expectThrow(() => applyEffectiveLlmMap(copyGraph(flow), effective), RangeError, 'missing paths: ["chat","solo"]');
  expectThrow(
    () => applyEffectiveLlmMap(copyGraph(flow), effective, callTimeRecord({ ...callTime, ghost: {} })),
    RangeError,
    'extra paths: ["ghost"]',
  );
  const overlap = { ...effective, chat: effective["pick"]! };
  expectThrow(
    () => applyEffectiveLlmMap(copyGraph(flow), overlap, callTimeRecord(callTime)),
    RangeError,
    'in both maps: ["chat"]',
  );
});

test("a call-time path must name an agent with llm_config values", async () => {
  const [flow, , runLlm] = await resolved(ROOT);
  const effective = Object.fromEntries(Object.entries(runLlm.effective).filter(([key]) => key !== "pick"));
  expectThrow(
    () => applyEffectiveLlmMap(copyGraph(flow), effective, callTimeRecord({ ...runLlm.callTime, pick: {} })),
    RangeError,
    /'pick'.*no llm_config value/,
  );
});

test("a call-time base must be a valid layer", async () => {
  const [flow, , runLlm] = await resolved(ROOT);
  const apply = (chat: unknown) => () =>
    applyEffectiveLlmMap(copyGraph(flow), runLlm.effective, callTimeRecord({ ...runLlm.callTime, chat }));
  expectThrow(apply("x"), TypeError, "'chat' must be a dict");
  expectThrow(apply({ bogus: 1n }), RangeError, /'chat'.*unknown field/);
  expectThrow(apply({ provider: "openai", api_key: "k" }), RangeError, /'chat'.*api_key/);
});

test("the host key reaches call-time agents on the private flow only", async () => {
  const host: Record<string, unknown> = {
    provider: "openai",
    model: "host-model",
    api_key: "sk-secret-call-time",
    headers: { "X-Tenant": "tenant-secret" },
  };
  const connection = popHostCredentials(host);
  const [flow, privateFlow, runLlm] = await resolved(NESTED, host);
  expect(JSON.stringify(runLlm.callTime)).not.toContain("secret");
  applyEffectiveLlmMap(privateFlow, runLlm.effective, callTimeRecord(runLlm.callTime, "openai"));
  bindHostCredentials(privateFlow, connection.apiKey!, connection.headers);

  const joined = agentAt(privateFlow, "c/a").hostCredentials;
  expect(joined).toBeInstanceOf(HostConnection);
  expect(joined!.apiKey).toBe("sk-secret-call-time");
  expect(joined!.headers).toEqual({ "X-Tenant": "tenant-secret" });
  // The key never enters the base, and a static agent gets no host credentials.
  expect(JSON.stringify(agentAt(privateFlow, "c/a").callTime!.base)).not.toContain("secret");
  expect(agentAt(privateFlow, "c/b").hostCredentials).toBeNull();
  expect(agentAt(flow, "c/a").hostCredentials).toBeNull();
  expect(agentAt(flow, "c/a").callTime).toBeNull();
});

test("bindHostCredentials without headers gives null", async () => {
  const [, privateFlow, runLlm] = await resolved(ROOT);
  applyEffectiveLlmMap(privateFlow, runLlm.effective, callTimeRecord(runLlm.callTime, "openai"));
  bindHostCredentials(privateFlow, "sk-x");
  expect(agentAt(privateFlow, "chat").hostCredentials!.headers).toBeNull();
});

test("bindHostCredentials skips a selection that names no host key", async () => {
  const [, privateFlow, runLlm] = await resolved(ROOT);
  applyEffectiveLlmMap(privateFlow, runLlm.effective, callTimeRecord(runLlm.callTime));
  bindHostCredentials(privateFlow, "sk-x");
  expect(agentAt(privateFlow, "chat").hostCredentials).toBeNull();
});

test("the fingerprint ignores the call-time attributes", async () => {
  const [flow, privateFlow, runLlm] = await resolved(ROOT);
  const before = flowFingerprint(flow);
  applyEffectiveLlmMap(privateFlow, runLlm.effective, callTimeRecord(runLlm.callTime));
  bindHostCredentials(privateFlow, "sk-x");
  expect(agentAt(privateFlow, "chat").callTime).not.toBeNull();
  expect(flowFingerprint(privateFlow)).toBe(before);
});

// The recorded connection selection.

const STORE = "/tmp/yian-auth/auth.json";
const OPENAI_STORE: Connection = {
  method: "api_key",
  source: "store",
  credential_id: "openai-login",
  endpoint: "https://api.openai.com/v1",
  store_path: STORE,
  organization: "org-saved",
};
const ANTHROPIC_STORE: Connection = {
  method: "api_key",
  source: "store",
  credential_id: "anthropic-login",
  endpoint: null,
  store_path: STORE,
};
const ENVIRONMENT: Connection = {
  method: "api_key",
  source: "environment",
  credential_id: null,
  endpoint: null,
  store_path: null,
};

/** Return the test snapshot with `overrides` applied. */
function snapshotWith(overrides: { apiEndpoint?: string } = {}): ConnectionSnapshot {
  return new ConnectionSnapshot({
    defaults: { provider: "openai", model: "gpt-saved" },
    providerModels: { openai: "gpt-saved", anthropic: "claude-saved" },
    connections: { openai: { ...OPENAI_STORE }, anthropic: { ...ANTHROPIC_STORE }, google: { ...ENVIRONMENT } },
    apiEndpoint: "https://api.openai.com/v1",
    strayModel: false,
    ...overrides,
  });
}

const HOST = new HostConnection({
  apiKey: "sk-secret-host",
  baseUrl: "https://proxy.example.com/v1",
  organization: "org-host",
  project: "proj-host",
  headers: { "X-Tenant": "tenant-secret", "X-Alpha": "alpha-secret" },
});

/** Return the record of a selection with the OpenAI host key of `HOST`. */
function hostRecord(providers: Iterable<string> | null = null, snapshot?: ConnectionSnapshot): SelectionRecord {
  return recordSelection(snapshot ?? snapshotWith(), {
    providers,
    hostKeyProvider: "openai",
    hostBaseUrl: HOST.baseUrl,
    hostSettings: HOST,
  });
}

/** Return `record` through JSON, as a run log stores it. */
function throughJson(record: SelectionRecord): Record<string, unknown> {
  return JSON.parse(JSON.stringify(record)) as Record<string, unknown>;
}

test("a recorded selection round trips and binds as the snapshot does", () => {
  const snapshot = snapshotWith();
  const restored = restoreSelection(throughJson(hostRecord(null, snapshot)));
  expect(restored.snapshot).toEqual(snapshot);
  for (const config of [
    { provider: "openai", model: "gpt-x" },
    { provider: "anthropic", model: "claude-x" },
    { provider: "ollama", model: "llama" },
  ]) {
    const expected = snapshot.bind(
      { a: config },
      { hostKeyProvider: "openai", hostBaseUrl: HOST.baseUrl, hostSettings: HOST },
    )["a"];
    expect(restored.bind(config)).toEqual(expected!);
  }
  expect(restored.bind({ provider: "openai", model: "gpt-x" })["_header_names"]).toEqual(["X-Alpha", "X-Tenant"]);
});

test("a selection without a host key binds saved connections", () => {
  const record = recordSelection(snapshotWith(), { providers: null });
  expect(record.host).toBeNull();
  const restored = restoreSelection(throughJson(record));
  expect(restored.hostKeyProvider).toBeNull();
  expect(restored.bind({ provider: "openai", model: "gpt-x" })).toEqual({
    provider: "openai",
    model: "gpt-x",
    _connection: OPENAI_STORE,
  });
});

test("no secret appears in a recorded selection", () => {
  const text = JSON.stringify(hostRecord());
  for (const secret of ["sk-secret-host", "tenant-secret", "alpha-secret"]) expect(text).not.toContain(secret);
});

type Json = Record<string, any>;

describe("a tampered selection fails to restore", () => {
  const edits: [string, (r: Json) => void][] = [
    ["openai source", (r) => (r["connections"]["openai"]["source"] = "bogus")],
    ["openai store path", (r) => (r["connections"]["openai"]["store_path"] = "relative")],
    ["ollama connection", (r) => (r["connections"]["ollama"] = { ...ENVIRONMENT })],
    ["unknown connection", (r) => (r["connections"]["nope"] = { ...ENVIRONMENT })],
    ["extra field", (r) => (r["extra"] = 1)],
    ["no stray_model", (r) => delete r["stray_model"]],
    ["string stray_model", (r) => (r["stray_model"] = "no")],
    ["ftp api_endpoint", (r) => (r["api_endpoint"] = "ftp://example.com")],
    ["unknown default provider", (r) => (r["defaults"]["provider"] = "nope")],
    ["default temperature", (r) => (r["defaults"]["temperature"] = 1)],
    ["empty model", (r) => (r["provider_models"]["openai"] = "")],
    ["unknown host provider", (r) => (r["host"]["provider"] = "nope")],
    ["http host base_url", (r) => (r["host"]["base_url"] = "http://example.com/v1")],
    ["unsorted header names", (r) => (r["host"]["header_names"] = ["X-Tenant", "X-Alpha"])],
    ["empty organization", (r) => (r["host"]["organization"] = "")],
    ["host api_key", (r) => (r["host"]["api_key"] = "sk-x")],
    ["azure host without base_url", (r) => (r["host"] = { provider: "azure" })],
  ];
  test.each(edits)("%s", (_name, edit) => {
    const record = structuredClone(hostRecord()) as Json;
    edit(record);
    let error: unknown = null;
    try {
      restoreSelection(record);
    } catch (caught) {
      error = caught;
    }
    expect(error).toBeInstanceOf(RangeError);
    expect((error as RangeError).message).toContain("recorded connection selection");
  });
});

test("the OpenAI endpoint is recorded only when OpenAI is reachable", () => {
  const snapshot = snapshotWith({ apiEndpoint: "https://gateway.example.com/v1" });
  expect(recordSelection(snapshot, { providers: new Set(["openai"]) }).api_endpoint).toBe(
    "https://gateway.example.com/v1",
  );
  const limited = recordSelection(snapshot, { providers: new Set(["anthropic"]) });
  expect(limited.api_endpoint).toBe("https://api.openai.com/v1");
  expect(JSON.stringify(limited)).not.toContain("gateway.example.com");
});

test("a selection limited to providers names only their connections", () => {
  const record = hostRecord(new Set(["anthropic"]));
  expect(record.connections).toEqual({ anthropic: ANTHROPIC_STORE });
  expect(record.provider_models).toEqual({ anthropic: "claude-saved" });
  // The default provider and the host key's provider are not reachable.
  expect(record.defaults).toEqual({});
  expect(record.host).toBeNull();
  expect(JSON.stringify(record)).not.toContain("openai-login");
  const restored = restoreSelection(record);
  expect(restored.snapshot.connections).toEqual({ anthropic: ANTHROPIC_STORE });
});

const LITERAL_PROVIDERS = `id: f
name: f
llm_config: {provider: google}
nodes:
  pick: {kind: agent, prompt: Pick., output: str, llm_config: {model: pick-model}}
  chat: {kind: agent, prompt: Go., llm_config: {provider: anthropic, model: '\${pick.output}'}}
  flowed: {kind: agent, prompt: Go., llm_config: {model: '\${pick.output}'}}
  solo:
    kind: agent
    prompt: Go.
    llm_config: {inherit: false, model: '\${pick.output}'}
output: \${chat.output}
`;

const VALUE_PROVIDER = `id: f
name: f
llm_config: {provider: google}
nodes:
  pick: {kind: agent, prompt: Pick., output: str, llm_config: {model: pick-model}}
  chat: {kind: agent, prompt: Go., llm_config: {provider: '\${pick.output}', model: m}}
output: \${chat.output}
`;

/** Return the selection run start records for the call-time agents of `text`. */
async function callTimeSelectionOf(text: string): Promise<SelectionRecord> {
  const privateFlow = copyGraph((await loadFlow(text)).compiled);
  const snapshot = snapshotWith();
  const runLlm = resolveRunLlm(privateFlow, {}, { defaults: snapshot.defaults });
  return callTimeSelection(snapshot, {
    known: knownCallTimeProviders(privateFlow, runLlm.callTime, snapshot.defaults),
    hostKeyProvider: "openai",
    hostBaseUrl: HOST.baseUrl,
    hostSettings: HOST,
  });
}

test("known providers decide which connections are recorded", async () => {
  const privateFlow = copyGraph((await loadFlow(LITERAL_PROVIDERS)).compiled);
  const runLlm = resolveRunLlm(privateFlow, {}, { defaults: snapshotWith().defaults });
  // Literal node, own-flow, and (under inherit: false) default providers.
  expect(knownCallTimeProviders(privateFlow, runLlm.callTime, snapshotWith().defaults)).toEqual(
    new Map([
      ["chat", "anthropic"],
      ["flowed", "google"],
      ["solo", "openai"],
    ]),
  );
  const record = await callTimeSelectionOf(LITERAL_PROVIDERS);
  expect(new Set(Object.keys(record.connections))).toEqual(new Set(["anthropic", "google", "openai"]));
  expect(record.host!["provider"]).toBe("openai");
});

test("only literal-provider agents record only their connections", async () => {
  const text = LITERAL_PROVIDERS.replace(
    "  solo:\n    kind: agent\n    prompt: Go.\n    llm_config: {inherit: false, model: '${pick.output}'}\n",
    "",
  );
  const record = await callTimeSelectionOf(text);
  expect(new Set(Object.keys(record.connections))).toEqual(new Set(["anthropic", "google"]));
  expect(Object.keys(record.provider_models)).toEqual(["anthropic"]);
  expect(record.defaults).toEqual({});
  expect(record.host).toBeNull();
});

test("a provider from a value records every connection", async () => {
  const privateFlow = copyGraph((await loadFlow(VALUE_PROVIDER)).compiled);
  const runLlm = resolveRunLlm(privateFlow, {}, { defaults: snapshotWith().defaults });
  expect(knownCallTimeProviders(privateFlow, runLlm.callTime, {})).toEqual(new Map([["chat", null]]));
  const record = await callTimeSelectionOf(VALUE_PROVIDER);
  expect(new Set(Object.keys(record.connections))).toEqual(new Set(["anthropic", "google", "openai"]));
  expect(record.defaults).toEqual(snapshotWith().defaults);
  expect(record.host!["provider"]).toBe("openai");
});

test("a nullable provider value above a literal is not known", async () => {
  const text = `id: f
name: f
input: {p: !Optional(str) }
llm_config: {provider: anthropic}
defs:
  child:
    input: {p: !Optional(str) }
    nodes:
      a: {kind: agent, prompt: Go., llm_config: {provider: '\${input.p}', model: m}}
    output: \${a.output}
nodes:
  c: {kind: call, call: child, input: {p: '\${input.p}'}}
output: \${c.output}
`;
  const privateFlow = copyGraph((await loadFlow(text)).compiled);
  const runLlm = resolveRunLlm(privateFlow, {}, { defaults: {} });
  expect(knownCallTimeProviders(privateFlow, runLlm.callTime, {})).toEqual(new Map([["c/a", null]]));
});

test("a call-time record needs a valid selection", async () => {
  const [flow, , runLlm] = await resolved(ROOT);
  const record = callTimeRecord(runLlm.callTime);
  Object.assign(record.selection.connections["openai"]!, { source: "bogus" });
  expectThrow(
    () => applyEffectiveLlmMap(copyGraph(flow), runLlm.effective, record),
    RangeError,
    "recorded connection selection",
  );
  expectThrow(
    () => applyEffectiveLlmMap(copyGraph(flow), runLlm.effective, runLlm.callTime),
    RangeError,
    "exactly selection and agents",
  );
  expectThrow(
    () => applyEffectiveLlmMap(copyGraph(flow), runLlm.effective, [record]),
    TypeError,
    "selection and agents",
  );
});
