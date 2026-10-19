/**
 * Mirrors `tests/test_provider_connections.py`; the factory tests go with
 * `llm_clients/factory.ts`, `test_run_records_connections` with `runFlow`. The environment table
 * test also builds and calls a model for every provider and connection source, so the factory's
 * reads are held to the same table.
 */

import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, beforeEach, describe, expect, test } from "bun:test";

import { ENTRIES, type Env } from "../src/defaults.ts";
import { collect } from "../src/llm/events.ts";
import { Message, Request, TextPart } from "../src/llm/messages.ts";
import { PROVIDER_API_KEY_ENV } from "../src/llm_clients/api_key_env.ts";
import {
  ApiKeyCredentials,
  AuthError,
  type Credentials,
  OAuthCredentials,
  type ProviderSelection,
} from "../src/llm_clients/auth/common.ts";
import { AuthStore } from "../src/llm_clients/auth/store.ts";
import {
  azureResourceEndpoint,
  HostConnection,
  LLMConfig,
  SUPPORTED_LLM_PROVIDERS,
} from "../src/llm_clients/config.ts";
import {
  type AgentConfig,
  blockProblem,
  callCredentials,
  checkModelConfig,
  type Connection,
  connectionCredentials,
  connectionSnapshot,
  ConnectionSnapshot,
  HOST_KEY_MARKER,
  headerNameDiff,
  keyIdentity,
  legacyConnectionConfig,
  legacyConnectionMap,
  refusedBlock,
  resolveModelConfig,
  splitModelConfig,
  validateConnection,
} from "../src/llm_clients/connection.ts";
import { chatModelFromConfig } from "../src/llm_clients/factory.ts";
import { requireConcreteSelection, validateLayer } from "../src/llm_clients/layers.ts";
import type { Send } from "../src/platform.ts";

const STORE: Connection = {
  method: "api_key",
  source: "store",
  credential_id: "abc",
  endpoint: null,
  store_path: "/abs/auth.json",
};
const ENVIRONMENT: Connection = {
  method: "api_key",
  source: "environment",
  credential_id: null,
  endpoint: null,
  store_path: null,
};
const OPENAI_STORE: Connection = { ...STORE, endpoint: "https://api.openai.com/v1" };
const HEADER_VALUE = "gw-token-3f9a20c1";
const MARKED = { provider: "anthropic", model: "m", [HOST_KEY_MARKER]: true, api_key: "k" };

/** Return `base` with `changes` applied. */
function withChanges(base: object, changes: Record<string, unknown>): Record<string, unknown> {
  return { ...base, ...changes };
}

/** Return the error `run` throws, failing the test if it throws none. */
function thrown(run: () => unknown): Error {
  try {
    run();
  } catch (error) {
    return error as Error;
  }
  throw new Error("expected a throw");
}

/** Return the `RangeError` `run` throws, failing the test on any other outcome. */
function rangeError(run: () => unknown): RangeError {
  const error = thrown(run);
  expect(error).toBeInstanceOf(RangeError);
  return error as RangeError;
}

/** Return the error `promise` rejects with, failing the test if it resolves. */
async function rejected(promise: Promise<unknown>): Promise<Error> {
  try {
    await promise;
  } catch (error) {
    return error as Error;
  }
  throw new Error("expected a rejection");
}

// Each test's environment: conftest's base (a private store in a fresh directory, the default
// provider and model), changed per test as Python's monkeypatch does.
let home: string;
let vars: Map<string, string>;
const env: Env = (name) => vars.get(name);

beforeEach(() => {
  home = mkdtempSync(join(tmpdir(), "yian-connections-"));
  vars = new Map([
    ["YIAN_AUTH_FILE", join(home, "auth.json")],
    ["YIAN_DEFAULT_PROVIDER", "openai"],
    ["YIAN_DEFAULT_MODEL", "gpt-5.5"],
  ]);
});

afterEach(() => {
  rmSync(home, { recursive: true, force: true });
});

/** Return the test's store. */
function authStore(): AuthStore {
  return new AuthStore(null, { env });
}

/** Return the connection a snapshot records for the saved API-key login `selection`. */
function storeConnection(selection: ProviderSelection, endpoint: string | null = null): Connection {
  return {
    method: "api_key",
    source: "store",
    credential_id: selection.credentialId,
    endpoint,
    store_path: authStore().path,
  };
}

test("connection metadata validation per provider", () => {
  expect(validateConnection(STORE, "anthropic")).toEqual(STORE);
  expect(validateConnection(ENVIRONMENT, "anthropic")).toEqual(ENVIRONMENT);
  const { endpoint: _endpoint, ...withoutEndpoint } = STORE;
  const rejected = [
    withChanges(STORE, { method: "subscription" }),
    withChanges(ENVIRONMENT, { source: "host" }),
    withChanges(ENVIRONMENT, { source: "missing" }),
    withChanges(ENVIRONMENT, { source: "legacy_environment" }),
    withChanges(STORE, { endpoint: "https://api.anthropic.com" }),
    withChanges(ENVIRONMENT, { credential_id: "abc" }),
    withChanges(STORE, { store_path: "auth.json" }),
    withChanges(STORE, { credential_id: "" }),
    withoutEndpoint,
  ];
  for (const value of rejected) rangeError(() => validateConnection(value, "anthropic"));

  const azureStore = withChanges(STORE, { endpoint: "https://r.openai.azure.com/" });
  expect(validateConnection(azureStore, "azure")).toEqual(azureStore as unknown as Connection);
  expect(validateConnection(ENVIRONMENT, "azure")).toEqual(ENVIRONMENT);
  for (const value of [
    STORE,
    withChanges(STORE, { endpoint: "http://r.openai.azure.com/" }),
    withChanges(ENVIRONMENT, { endpoint: "https://r.openai.azure.com/" }),
  ]) {
    rangeError(() => validateConnection(value, "azure"));
  }

  expect(rangeError(() => validateConnection(ENVIRONMENT, "ollama")).message).toMatch(/not used for provider ollama/);
  for (const provider of ["unknown-secret", null, ["openai"]]) {
    expect(rangeError(() => validateConnection(ENVIRONMENT, provider)).message).toMatch(
      /not used for an unsupported provider$/,
    );
  }
});

test("bind records host settings on marked agents only", () => {
  // What a snapshot reads with no saved login and no provider key in the environment.
  const snapshot = new ConnectionSnapshot({
    defaults: {},
    providerModels: {},
    connections: {
      openai: { ...ENVIRONMENT, source: "missing", endpoint: "https://api.openai.com/v1" },
      anthropic: ENVIRONMENT,
    },
  });
  const host = new HostConnection({
    apiKey: "sk-host-6d1e",
    organization: "org-abc",
    project: "proj-abc",
    headers: { "X-Trace": "t", "X-Gateway": HEADER_VALUE },
  });
  const effective = {
    o: { provider: "openai", model: "m" },
    a: { provider: "anthropic", model: "m" },
  };
  const bound = snapshot.bind(effective, { hostKeyProvider: "openai", hostSettings: host });
  expect(bound["o"]!["organization"]).toBe("org-abc");
  expect(bound["o"]!["project"]).toBe("proj-abc");
  expect(bound["o"]!["_header_names"]).toEqual(["X-Gateway", "X-Trace"]);
  expect(bound["a"]).toEqual({ provider: "anthropic", model: "m", _connection: ENVIRONMENT });
  expect(JSON.stringify(bound)).not.toContain(HEADER_VALUE);
  expect(JSON.stringify(bound)).not.toContain("sk-host-6d1e");
  expect(Object.hasOwn(bound["o"]!, "api_key")).toBe(false);
  expect(Object.hasOwn(effective.o, "organization")).toBe(false);

  const plain = snapshot.bind(effective, {
    hostKeyProvider: "openai",
    hostSettings: new HostConnection({ apiKey: "k" }),
  });
  for (const key of ["organization", "project", "_header_names"]) {
    expect(Object.hasOwn(plain["o"]!, key)).toBe(false);
  }
});

test("validate connection takes saved login settings", () => {
  const openai = {
    ...OPENAI_STORE,
    organization: "org-a",
    project: "proj-a",
    header_names: ["X-Gateway", "X-Trace"],
  };
  expect(validateConnection(openai)).toEqual(openai);
  const anthropic = { ...STORE, header_names: ["X-Gateway"] };
  expect(validateConnection(anthropic, "anthropic")).toEqual(anthropic);
  const azure = { ...STORE, endpoint: "https://r.openai.azure.com/", header_names: ["X-Gateway"] };
  expect(validateConnection(azure, "azure")).toEqual(azure);
  // Old records carry none of them.
  expect(validateConnection(OPENAI_STORE)).toEqual(OPENAI_STORE);
});

describe("validate connection refuses misplaced settings", () => {
  const rows: [string, Record<string, unknown>, string][] = [
    [
      "openai",
      { ...OPENAI_STORE, source: "environment", credential_id: "a".repeat(64), store_path: null, header_names: ["X-Gateway"] },
      "saved API-key login",
    ],
    [
      "openai",
      { ...OPENAI_STORE, source: "host", credential_id: null, store_path: null, organization: "org-a" },
      "saved API-key login",
    ],
    [
      "openai",
      {
        ...OPENAI_STORE,
        method: "subscription",
        endpoint: "https://chatgpt.com/backend-api/codex",
        header_names: ["X-Gateway"],
      },
      "saved API-key login",
    ],
    ["anthropic", { ...ENVIRONMENT, header_names: ["X-Gateway"] }, "saved API-key login"],
    ["anthropic", { ...STORE, organization: "org-a" }, "organization is only for provider openai"],
    ["google", { ...STORE, project: "proj-a" }, "project is only for provider openai"],
    ["openai", { ...OPENAI_STORE, organization: "has space" }, "connection organization"],
    ["openai", { ...OPENAI_STORE, project: null }, "connection project"],
    ["openai", { ...OPENAI_STORE, header_names: [] }, "connection header_names"],
    ["openai", { ...OPENAI_STORE, header_names: ["X-B", "X-A"] }, "connection header_names"],
    ["openai", { ...OPENAI_STORE, header_names: ["OpenAI-Project"] }, "connection header_names"],
    ["anthropic", { ...STORE, header_names: ["Authorization"] }, "connection header_names"],
    ["anthropic", { ...STORE, header_names: "X-Gateway" }, "connection header_names"],
    ["anthropic", { ...STORE, headers: { "X-Gateway": "v" } }, "metadata fields"],
  ];
  test.each(rows)("%s %#", (provider, value, message) => {
    const error = rangeError(() => validateConnection(value, provider));
    expect(error.message).toContain(message);
    expect(error.message).not.toContain("has space");
  });
});

test("header name diff names missing and unrecorded headers ignoring case", () => {
  expect(headerNameDiff(["X-Trace", "X-Gateway"], ["x-gateway", "X-TRACE"])).toBeNull();
  expect(headerNameDiff([], [])).toBeNull();
  expect(headerNameDiff(["X-Trace", "X-Gateway"], ["x-gateway", "X-Extra", "A-New"])).toBe(
    "missing X-Trace; not recorded A-New, X-Extra",
  );
  expect(headerNameDiff(["X-Trace"], [])).toBe("missing X-Trace");
});

test("an environment key's identity is its SHA-256 hex, as Python records it", () => {
  expect(keyIdentity("sk-test")).toBe("f3abf2a6cc4f00987743db5f544ba345b4899ae31f326d8ee9c4816de153c9e0");
});

test("an inherited property name is an unsupported provider", () => {
  for (const provider of ["__proto__", "constructor", "toString"]) {
    expect(rangeError(() => validateConnection(ENVIRONMENT, provider)).message).toBe(
      "Internal connection metadata is not used for an unsupported provider",
    );
  }
});

test("an OpenAI endpoint must spell its scheme and host as urlsplit reads them", () => {
  for (const endpoint of ["https:api.example", "https:///x", "ftp://api.example/v1", "https://u:p@api.example/v1"]) {
    expect(rangeError(() => validateConnection({ ...OPENAI_STORE, endpoint })).message).toBe(
      "Connection endpoint must be an HTTP URL without credentials, query, or fragment",
    );
  }
  const upper = { ...OPENAI_STORE, endpoint: "HTTPS://API.example/v1" };
  expect(validateConnection(upper)).toEqual(upper);
});

test("split and layer accept non-OpenAI metadata", () => {
  expect(splitModelConfig({ provider: "anthropic", model: "m", _connection: STORE })[1]).toEqual(STORE);
  rangeError(() => splitModelConfig({ provider: "ollama", model: "m", _connection: STORE }));

  validateLayer({ provider: "anthropic", model: "m", _connection: STORE });
  rangeError(() => validateLayer({ provider: "ollama", model: "m", _connection: STORE }));

  expect(
    rangeError(() =>
      splitModelConfig({
        provider: "azure",
        model: "deployment",
        base_url: "https://r.openai.azure.com/",
        _connection: withChanges(STORE, { endpoint: "https://other.openai.azure.com/" }),
      }),
    ).message,
  ).toMatch(/another Azure resource/);
  const [, metadata] = splitModelConfig({
    provider: "azure",
    model: "deployment",
    base_url: "https://r.openai.azure.com",
    _connection: withChanges(STORE, { endpoint: "https://r.openai.azure.com/openai/v1/" }),
  });
  expect(metadata!.endpoint).toBe("https://r.openai.azure.com/openai/v1/");
});

test("legacy adds environment connection except host key and keyless", () => {
  expect(legacyConnectionConfig({ provider: "anthropic", model: "m" })).toEqual({
    provider: "anthropic",
    model: "m",
    _connection: ENVIRONMENT,
  });
  expect(
    Object.hasOwn(
      legacyConnectionConfig({ provider: "anthropic", model: "m", [HOST_KEY_MARKER]: true }),
      "_connection",
    ),
  ).toBe(false);
  expect(Object.hasOwn(legacyConnectionConfig({ provider: "ollama", model: "m" }), "_connection")).toBe(false);
  const openai = legacyConnectionConfig({ provider: "openai", model: "m" });
  expect((openai["_connection"] as Connection).source).toBe("legacy_environment");
  expect(legacyConnectionConfig({ provider: "anthropic", model: "m", _connection: STORE })["_connection"]).toEqual(
    STORE,
  );
  // An `undefined` value is an absent key.
  const pinned = legacyConnectionConfig({ provider: "openai", model: "m", _connection: undefined });
  expect((pinned["_connection"] as Connection).source).toBe("legacy_environment");
  expect(
    legacyConnectionConfig({ provider: "anthropic", model: "m", [HOST_KEY_MARKER]: undefined })["_connection"],
  ).toEqual(ENVIRONMENT);
});

test("azure resource endpoint", () => {
  for (const url of [
    "https://r.openai.azure.com/",
    "https://r.openai.azure.com/openai",
    "https://r.openai.azure.com/openai/v1/",
  ]) {
    expect(azureResourceEndpoint(url)).toBe("https://r.openai.azure.com");
  }
});

test("split model config needs the recorded headers", () => {
  const names = ["X-Gateway", "X-Trace"];
  let error = thrown(() => splitModelConfig({ ...MARKED, _header_names: names }));
  expect(error).toBeInstanceOf(AuthError);
  expect(error.message).toContain("X-Gateway");
  expect(error.message).toContain("X-Trace");
  expect(error.message).toContain("resume_flow");
  error = thrown(() =>
    splitModelConfig({
      ...MARKED,
      _header_names: names,
      headers: { "X-Gateway": HEADER_VALUE, "X-Other": HEADER_VALUE },
    }),
  );
  expect(error).toBeInstanceOf(AuthError);
  expect(error.message).toContain("X-Trace");
  expect(error.message).toContain("X-Other");
  expect(error.message).not.toContain(HEADER_VALUE);
  error = thrown(() => splitModelConfig({ ...MARKED, headers: { "X-Gateway": HEADER_VALUE } }));
  expect(error).toBeInstanceOf(AuthError);
  expect(error.message).toContain("X-Gateway");

  // Names compare without case.
  const [validated] = splitModelConfig({
    ...MARKED,
    _header_names: names,
    headers: { "x-gateway": HEADER_VALUE, "X-TRACE": "t" },
  });
  expect(validated.headers).toEqual({ "x-gateway": HEADER_VALUE, "X-TRACE": "t" });
});

describe("split model config rejects malformed header names", () => {
  const rows: [string, unknown][] = [
    ["empty", []],
    ["unsorted", ["X-Trace", "X-Gateway"]],
    ["case-duplicate", ["X-Gateway", "x-gateway"]],
    ["not-a-token", ["bad name"]],
    ["refused", ["Authorization"]],
    ["not-a-list", "X-Gateway"],
    ["not-text", [1n]],
  ];
  test.each(rows)("%s", (_id, names) => {
    const error = rangeError(() =>
      splitModelConfig({ ...MARKED, _header_names: names, headers: { "X-Gateway": "v" } }),
    );
    expect(error.message).toContain("_header_names");
  });
});

test("split model config reads an undefined marker as absent and names a missing provider None", () => {
  const [validated] = splitModelConfig({ provider: "anthropic", model: "m", [HOST_KEY_MARKER]: undefined });
  expect(validated.provider).toBe("anthropic");
  const error = thrown(() => splitModelConfig({ model: "m", [HOST_KEY_MARKER]: true }));
  expect(error).toBeInstanceOf(AuthError);
  expect(error.message).toStartWith("This run's None API key");
});

test("snapshot connections are frozen copies", () => {
  const snapshot = new ConnectionSnapshot({ defaults: {}, providerModels: {}, connections: { anthropic: ENVIRONMENT } });
  expect(() => {
    (snapshot.connections["anthropic"] as Connection).source = "store";
  }).toThrow(TypeError);
  expect(snapshot.connections["anthropic"]!.source).toBe("environment");
  expect(ENVIRONMENT.source).toBe("environment");
});

test("split model config refuses header names without the marker", () => {
  const { [HOST_KEY_MARKER]: _marker, ...unmarked } = MARKED;
  const error = rangeError(() =>
    splitModelConfig({ ...unmarked, _header_names: ["X-Gateway"], headers: { "X-Gateway": "v" } }),
  );
  expect(error.message).toContain("_header_names");
});

test("split and resolve carry headers beside the key", () => {
  const config = new LLMConfig({
    provider: "anthropic",
    model: "m",
    api_key: "k",
    headers: { "X-Gateway": HEADER_VALUE },
  });
  const [validated] = splitModelConfig(config);
  expect(validated.headers).toEqual({ "X-Gateway": HEADER_VALUE });
  expect(validated.apiKey).toBe("k");
  let resolved = resolveModelConfig(config, { env });
  expect(resolved["headers"]).toEqual({ "X-Gateway": HEADER_VALUE });
  expect(resolved["api_key"]).toBe("k");
  resolved = resolveModelConfig(
    { ...MARKED, _header_names: ["X-Gateway"], headers: { "X-Gateway": HEADER_VALUE } },
    { env },
  );
  expect(resolved["headers"]).toEqual({ "X-Gateway": HEADER_VALUE });
  expect(Object.hasOwn(resolved, "_header_names")).toBe(false);
  // The OpenAI route resolves through the snapshot and still keeps them.
  resolved = resolveModelConfig(
    { provider: "openai", model: "m", api_key: "k", organization: "org-abc", headers: { "X-Gateway": HEADER_VALUE } },
    { env },
  );
  expect(resolved["organization"]).toBe("org-abc");
  expect(resolved["headers"]).toEqual({ "X-Gateway": HEADER_VALUE });
});

test("layers refuse recorded header names", () => {
  expect(rangeError(() => validateLayer({ provider: "anthropic", _header_names: ["X-Gateway"] })).message).toContain(
    "_header_names is internal to recorded runs",
  );
});

test("a recorded map takes settings only beside the marker", () => {
  const marked: Record<string, unknown> = {
    provider: "openai",
    model: "m",
    [HOST_KEY_MARKER]: true,
    organization: "org-abc",
    project: "proj-abc",
    _header_names: ["X-Gateway"],
  };
  requireConcreteSelection(marked, { path: "agent" });
  for (const key of ["organization", "project", "_header_names"]) {
    const unmarked = { provider: "openai", model: "m", [key]: marked[key] };
    expect(rangeError(() => requireConcreteSelection(unmarked, { path: "agent" })).message).toContain(
      `'agent' has ${key} without a host key`,
    );
  }
  for (const names of [[], ["X-B", "X-A"], ["Authorization"], "X-A"]) {
    expect(
      rangeError(() => requireConcreteSelection({ ...marked, _header_names: names }, { path: "agent" })).message,
    ).toMatch(/'agent'.*_header_names/);
  }
  expect(
    rangeError(() => requireConcreteSelection({ ...marked, organization: "has space" }, { path: "agent" })).message,
  ).toMatch(/'agent'.*organization/);
  for (const key of ["organization", "project"]) {
    const anthropic = { ...marked, provider: "anthropic", [key]: marked[key] };
    delete anthropic[key === "project" ? "organization" : "project"];
    expect(rangeError(() => requireConcreteSelection(anthropic, { path: "agent" })).message).toContain(
      `'agent': ${key} applies only to provider openai`,
    );
  }
  expect(
    rangeError(() => requireConcreteSelection({ ...marked, headers: { "X-Gateway": "v" } }, { path: "agent" })).message,
  ).toContain("headers");
});

test("a legacy connection map pins each agent and leaves its input alone", () => {
  const effective = { o: { provider: "openai", model: "m" }, a: { provider: "anthropic", model: "m" } };
  const pinned = legacyConnectionMap(effective);
  expect((pinned["o"]!["_connection"] as Connection).source).toBe("legacy_environment");
  expect(pinned["a"]!["_connection"]).toEqual(ENVIRONMENT);
  expect(effective).toEqual({ o: { provider: "openai", model: "m" }, a: { provider: "anthropic", model: "m" } });
});

describe("TS additions", () => {
  test("blockProblem moves one leading key path and keeps the rest whole", () => {
    const extra = "Extra inputs are not permitted";
    expect(blockProblem("node 'chat' llm_config", "openai", `\`openai.temperature\`: ${extra}`, " in 'sub'")).toBe(
      `node 'chat' llm_config.openai.temperature in 'sub': ${extra}`,
    );
    const several = `\`openai.temperature\`: ${extra}; \`openai.top_p\`: ${extra}`;
    expect(blockProblem("llm_config", "openai", several)).toBe(`llm_config.openai: ${several}`);
    expect(blockProblem("llm_config", "", "`openai.temperature`: bad")).toBe("llm_config: `openai.temperature`: bad");
    expect(blockProblem("llm_config", "openai", "no path here", " in 'sub'")).toBe(
      "llm_config.openai in 'sub': no path here",
    );
  });

  const subscription = {
    provider: "openai",
    model: "m",
    openai: { temperature: 0.2 },
    _connection: {
      ...OPENAI_STORE,
      method: "subscription",
      endpoint: "https://chatgpt.com/backend-api/codex",
    },
  };

  test("checkModelConfig checks the block against the API the connection reaches", () => {
    expect(rangeError(() => checkModelConfig(subscription)).message).toContain("`openai.temperature`");
    // A host key sends the same agent to the Responses API, which accepts the block.
    checkModelConfig({ ...subscription, api_key: "host-key" });
  });

  test("validateLayer ignores undefined values", () => {
    validateLayer({
      provider: "anthropic",
      model: "m",
      [HOST_KEY_MARKER]: undefined,
      _header_names: undefined,
      api_key: undefined,
      base_url: undefined,
      sandbox: undefined,
      temperature: undefined,
      modle: undefined,
      _connection: undefined,
    });
  });

  test("constructor as a layer key is unknown", () => {
    expect(rangeError(() => validateLayer({ constructor: "x" })).message).toContain(
      'llm_config has unknown field(s) ["constructor"] (allowed: ',
    );
  });

  test("refusedBlock names the block only when the block alone was refused", () => {
    expect(refusedBlock({ provider: "anthropic", model: 1n })).toBe("");
    expect(refusedBlock({ provider: "azure", model: "m" })).toBe("");
    expect(refusedBlock({ model: "m" })).toBe("");
    expect(refusedBlock(subscription)).toBe(".openai");
  });
});

test("resolve model config names stray default model", () => {
  vars.delete("YIAN_DEFAULT_PROVIDER");
  vars.set("YIAN_DEFAULT_MODEL", "m");
  expect(rangeError(() => resolveModelConfig({}, { env })).message).toContain(
    "YIAN_DEFAULT_MODEL is set but names no provider",
  );
});

test("snapshot connections per provider", async () => {
  const saved = await authStore().save("anthropic", new ApiKeyCredentials("saved-secret"), "m");

  const { connections } = connectionSnapshot({ env });

  expect(connections["anthropic"]).toEqual(storeConnection(saved));
  expect(connections["google"]).toEqual(ENVIRONMENT);
  expect(Object.hasOwn(connections, "ollama")).toBe(false);
  expect(connections["openai"]).toEqual({
    method: "api_key",
    source: "missing",
    credential_id: null,
    endpoint: "https://api.openai.com/v1",
    store_path: null,
  });
});

describe("snapshot rejects impossible saved logins", () => {
  const key = new ApiKeyCredentials("k");
  const rows: [string, Credentials | null, string | null][] = [
    ["openai", null, null],
    ["openai", key, "https://api.example/v1"],
    [
      "anthropic",
      new OAuthCredentials({ accessToken: "a", refreshToken: "r", expiresAt: Date.now() / 1000 + 3600, accountId: "acct" }),
      null,
    ],
    ["google", key, "https://example.com/"],
    ["azure", key, null],
    ["azure", null, "https://r.openai.azure.com/"],
    ["azure", key, "http://r.openai.azure.com/"],
    ["ollama", key, null],
    ["anthropic", null, null],
  ];
  test.each(rows)("%s %#", async (provider, credentials, endpoint) => {
    await authStore().save(provider, credentials, "m", { endpoint, makeDefault: false });
    expect(rangeError(() => connectionSnapshot({ env })).message).toBe(
      `The saved ${provider} login is invalid; run 'yian auth ${provider}' again`,
    );
  });
});

test("snapshot accepts keyless logins and skips unsupported ones", async () => {
  const store = authStore();
  await store.save("ollama", null, "l", { makeDefault: false });
  await store.save("vllm", null, "v", { makeDefault: false });
  await store.save("claude", new ApiKeyCredentials("k"), "c", { makeDefault: false });

  const snapshot = connectionSnapshot({ env });

  expect(snapshot.providerModels["ollama"]).toBe("l");
  expect(snapshot.connections["vllm"]).toEqual(ENVIRONMENT);
  expect(Object.hasOwn(snapshot.providerModels, "claude")).toBe(false);
  expect(Object.hasOwn(snapshot.connections, "claude")).toBe(false);

  vars.delete("YIAN_DEFAULT_PROVIDER");
  await store.setDefault("claude");
  expect(rangeError(() => connectionSnapshot({ env })).message).toContain(
    "saved default provider 'claude' is not supported",
  );
  // An empty value counts as unset.
  vars.set("YIAN_DEFAULT_PROVIDER", "");
  expect(rangeError(() => connectionSnapshot({ env })).message).toContain(
    "saved default provider 'claude' is not supported",
  );
});

test("bind records store and environment connections", async () => {
  const saved = await authStore().save("anthropic", new ApiKeyCredentials("saved-secret"), "m");
  const snapshot = connectionSnapshot({ env });
  const effective = {
    a: { provider: "anthropic", model: "m" },
    g: { provider: "google", model: "g" },
    o: { provider: "ollama", model: "l" },
  };

  const bound = snapshot.bind(effective);

  expect(bound["a"]!["_connection"]).toEqual(storeConnection(saved));
  expect(bound["g"]!["_connection"]).toEqual(ENVIRONMENT);
  expect(Object.hasOwn(bound["o"]!, "_connection")).toBe(false);
  expect(Object.hasOwn(effective.a, "_connection")).toBe(false);

  const hosted = snapshot.bind(effective, { hostKeyProvider: "anthropic" });
  expect(hosted["a"]![HOST_KEY_MARKER]).toBe(true);
  expect(Object.hasOwn(hosted["a"]!, "_connection")).toBe(false);
  expect(hosted["g"]!["_connection"]).toEqual(ENVIRONMENT);
});

test("bind azure endpoint rules", async () => {
  const effective = { z: { provider: "azure", model: "dep" } };

  let bound = connectionSnapshot({ env }).bind(effective)["z"]!;
  expect(Object.hasOwn(bound, "base_url")).toBe(false);
  expect(bound["_connection"]).toEqual(ENVIRONMENT);

  const saved = await authStore().save("azure", new ApiKeyCredentials("saved-secret"), "dep", {
    endpoint: "https://r.openai.azure.com/",
  });
  const snapshot = connectionSnapshot({ env });
  const store = storeConnection(saved, "https://r.openai.azure.com/");

  bound = snapshot.bind(effective)["z"]!;
  expect(bound["base_url"]).toBe("https://r.openai.azure.com/");
  expect(bound["_connection"]).toEqual(store);

  const same = { provider: "azure", model: "dep", base_url: "https://r.openai.azure.com/openai/v1/" };
  bound = snapshot.bind({ z: same })["z"]!;
  expect(bound["base_url"]).toBe("https://r.openai.azure.com/openai/v1/");
  expect(bound["_connection"]).toEqual(store);

  const other = { provider: "azure", model: "dep", base_url: "https://other.openai.azure.com/" };
  bound = snapshot.bind({ z: other })["z"]!;
  expect(bound["base_url"]).toBe("https://other.openai.azure.com/");
  expect(bound["_connection"]).toEqual(ENVIRONMENT);
});

test("resolve model config attaches without rereading on replay", async () => {
  // Any read of the environment, the snapshot's first step, fails the test.
  const unread: Env = (name) => {
    throw new Error(`read ${name}`);
  };
  expect(resolveModelConfig({ provider: "ollama", model: "l" }, { env: unread })).toEqual({
    provider: "ollama",
    model: "l",
  });
  const hosted = resolveModelConfig({ provider: "anthropic", model: "m", api_key: "k" }, { env: unread });
  expect(Object.hasOwn(hosted, "_connection")).toBe(false);
  const pinned = { provider: "anthropic", model: "m", _connection: STORE };
  expect(resolveModelConfig(pinned, { env: unread })).toEqual(pinned);
  const host = new ConnectionSnapshot({ defaults: {}, providerModels: {}, connections: {} }).hostConnection();
  const recorded = { provider: "openai", model: "m", api_key: "k", _connection: host };
  expect(resolveModelConfig(recorded, { env: unread })["_connection"]).toEqual(host);

  expect(resolveModelConfig({ provider: "anthropic", model: "m" }, { env })["_connection"]).toEqual(ENVIRONMENT);
  const saved = await authStore().save("anthropic", new ApiKeyCredentials("saved-secret"), "m");
  expect(resolveModelConfig({ provider: "anthropic", model: "m" }, { env })["_connection"]).toEqual(
    storeConnection(saved),
  );
  expect(resolveModelConfig({ provider: "openai", model: "m", api_key: "k" }, { env })["_connection"]).toEqual(host);
  const azure = await authStore().save("azure", new ApiKeyCredentials("saved-secret"), "dep", {
    endpoint: "https://r.openai.azure.com/",
  });
  const resolved = resolveModelConfig({ provider: "azure", model: "dep" }, { env });
  expect(resolved["base_url"]).toBe("https://r.openai.azure.com/");
  expect(resolved["_connection"]).toEqual(storeConnection(azure, "https://r.openai.azure.com/"));
});

test("connection credentials per provider", async () => {
  const saved = await authStore().save("anthropic", new ApiKeyCredentials("saved-anth-key"), "m", {
    makeDefault: false,
  });
  const pinned = storeConnection(saved);

  const credentials = await connectionCredentials(pinned, "anthropic", { env });
  expect((credentials as ApiKeyCredentials).apiKey).toBe("saved-anth-key");

  await authStore().save("anthropic", new ApiKeyCredentials("other-key"), "m", { makeDefault: false });
  let error = await rejected(connectionCredentials(pinned, "anthropic", { env }));
  expect(error).toBeInstanceOf(AuthError);
  expect(error.message).toContain("yian auth anthropic");
  await authStore().remove("anthropic");
  error = await rejected(connectionCredentials(pinned, "anthropic", { env }));
  expect(error).toBeInstanceOf(AuthError);
  expect(error.message).toContain("yian auth anthropic");
  error = await rejected(connectionCredentials(ENVIRONMENT, "anthropic", { env }));
  expect(error).toBeInstanceOf(RangeError);
  expect(error.message).toContain("read by its client");
});

test("snapshot records saved settings without values", async () => {
  const savedHeader = "saved-gw-7c2e91";
  const openai = await authStore().save(
    "openai",
    new ApiKeyCredentials("sk-saved-4b1f", { "X-Gateway": savedHeader }),
    "gpt-x",
    { organization: "org-saved", project: "proj-saved" },
  );
  const anthropic = await authStore().save(
    "anthropic",
    new ApiKeyCredentials("sk-ant-saved", { "X-Trace": "t", "X-Gateway": "g" }),
    "m",
    { makeDefault: false },
  );
  const plain = await authStore().save("google", new ApiKeyCredentials("g-saved"), "g", { makeDefault: false });

  const { connections } = connectionSnapshot({ env });

  expect(connections["openai"]).toEqual({
    ...storeConnection(openai, "https://api.openai.com/v1"),
    organization: "org-saved",
    project: "proj-saved",
    header_names: ["X-Gateway"],
  });
  expect(connections["anthropic"]).toEqual({ ...storeConnection(anthropic), header_names: ["X-Gateway", "X-Trace"] });
  expect(connections["google"]).toEqual(storeConnection(plain));
  expect(JSON.stringify(connections)).not.toContain(savedHeader);
});

describe("environment table", () => {
  const AZURE = "https://r.openai.azure.com/";
  // Design 7.4's table of the variables yian reads. Later readers (the factory) add their calls
  // to `drive` below, so this test holds them to the same table.
  const READ = new Set<string>([
    ...Object.values(PROVIDER_API_KEY_ENV).filter((name): name is string => name !== null),
    "GEMINI_API_KEY",
    "AZURE_OPENAI_AD_TOKEN",
    "OPENAI_API_VERSION",
    "OPENAI_BASE_URL",
    "ANTHROPIC_BASE_URL",
    "VLLM_BASE_URL",
    "OLLAMA_BASE_URL",
    "YIAN_DEFAULT_PROVIDER",
    "YIAN_DEFAULT_MODEL",
    "YIAN_AUTH_FILE",
    "XDG_CONFIG_HOME",
    "HOME",
    ...ENTRIES.map((entry) => entry.variable),
  ]);
  // Variables a provider SDK or another tool reads; yian sets none of their effects.
  const NEVER_READ = [
    "ANTHROPIC_AUTH_TOKEN",
    "OLLAMA_API_KEY",
    "OPENAI_CUSTOM_HEADERS",
    "ANTHROPIC_CUSTOM_HEADERS",
    "OPENAI_ORG_ID",
    "OPENAI_ORGANIZATION",
    "OPENAI_PROJECT_ID",
    "GOOGLE_GENAI_USE_VERTEXAI",
    "AZURE_OPENAI_DEPLOYMENT_NAME",
    "CODEX_HOME",
    "SSL_CERT_FILE",
    "OLLAMA_HOST",
    "GOOGLE_GEMINI_BASE_URL",
    "GOOGLE_GENAI_USE_ENTERPRISE",
    "ANTHROPIC_API_URL",
    "OPENAI_API_BASE",
    "LANGSMITH_GATEWAY",
    "LANGSMITH_GATEWAY_API_KEY",
    "LANGSMITH_API_KEY",
    "OPENAI_PROXY",
    "ANTHROPIC_PROXY",
    "LC_OUTPUT_VERSION",
    "AZURE_OPENAI_ENDPOINT",
    "OPENAI_API_TYPE",
  ];
  const REQUEST = new Request({ messages: [new Message({ role: "user", parts: [new TextPart("Hi")] })] });
  // Every request fails before the wire; only the reads up to it count.
  const fake: Send = async () => {
    throw new Error("not sent");
  };

  /** Return the logins of each kind `provider` can save. */
  function logins(provider: string): (Credentials | null)[] {
    const key = new ApiKeyCredentials("saved-key", { "X-Gateway": "saved-gw" });
    if (provider === "ollama") return [null];
    if (provider === "vllm") return [key, null];
    if (provider === "openai") {
      const expiresAt = Date.now() / 1000 + 3600;
      return [key, new OAuthCredentials({ accessToken: "a", refreshToken: "r", expiresAt, accountId: "acct" })];
    }
    return [key];
  }

  test("every variable read is in the table, and the others are never read", async () => {
    const recorded = new Set<string>();
    const recording: Env = (name) => {
      recorded.add(name);
      return env(name);
    };
    for (const name of NEVER_READ) vars.set(name, "never-read");

    /** Snapshot, resolve, list and resolve credentials for `provider`'s configs, then build and call each. */
    const drive = async (provider: string) => {
      const base: AgentConfig = provider === "azure" ? { provider, model: "m", base_url: AZURE } : { provider, model: "m" };
      connectionSnapshot({ env: recording });
      callCredentials(base, { env: recording });
      const configs = [
        resolveModelConfig(base, { env: recording }),
        resolveModelConfig({ ...base, api_key: "host-key" }, { env: recording }),
        legacyConnectionConfig(base),
      ];
      for (const config of configs) {
        callCredentials(config, { env: recording });
        if (config["_connection"] !== undefined) {
          // Only the reads count; an environment connection outside OpenAI refuses.
          await connectionCredentials(config["_connection"], provider, { env: recording }).catch(() => null);
        }
        await chatModelFromConfig(config, { env: recording, send: fake, onWarning: () => {} })
          .then((model) => collect(model.stream(REQUEST)))
          .catch(() => null);
      }
    };

    // No saved login and no key.
    for (const provider of SUPPORTED_LLM_PROVIDERS) await drive(provider);
    // Every key variable.
    for (const name of READ) if (name.endsWith("_API_KEY")) vars.set(name, `env-${name}`);
    vars.set("AZURE_OPENAI_AD_TOKEN", "env-ad-token");
    vars.set("OPENAI_BASE_URL", "https://api.example/v1");
    vars.set("ANTHROPIC_BASE_URL", "https://anthropic.example");
    vars.set("VLLM_BASE_URL", "http://gpu-box:8000/v1");
    vars.set("OLLAMA_BASE_URL", "http://user:pw@gpu-box:11434");
    vars.set("OPENAI_API_VERSION", "2025-03-01-preview");
    for (const provider of SUPPORTED_LLM_PROVIDERS) await drive(provider);
    // Each login kind, over the key variables.
    const store = new AuthStore(null, { env: recording });
    for (const provider of SUPPORTED_LLM_PROVIDERS) {
      for (const credentials of logins(provider)) {
        await store.save(provider, credentials, "m", {
          endpoint: provider === "azure" ? AZURE : null,
          makeDefault: false,
        });
        await drive(provider);
      }
    }
    // The default store path.
    vars.delete("YIAN_AUTH_FILE");
    new AuthStore(null, { env: recording });

    expect([...recorded].filter((name) => !READ.has(name))).toEqual([]);
    expect(NEVER_READ.filter((name) => recorded.has(name))).toEqual([]);
    expect([...recorded]).toEqual(
      expect.arrayContaining([
        "YIAN_AUTH_FILE",
        "YIAN_DEFAULT_PROVIDER",
        "YIAN_DEFAULT_MODEL",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_AD_TOKEN",
        "OPENAI_API_VERSION",
        "ANTHROPIC_BASE_URL",
        "VLLM_BASE_URL",
        "OLLAMA_BASE_URL",
        "XDG_CONFIG_HOME",
        "HOME",
      ]),
    );
  });
});
