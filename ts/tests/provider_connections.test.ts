/**
 * Mirrors `tests/test_provider_connections.py`; the factory tests go with
 * `llm_clients/factory.ts`, `test_run_records_connections` with `runFlow`.
 */

import { describe, expect, test } from "bun:test";

import { AuthError } from "../src/llm_clients/auth/common.ts";
import { azureResourceEndpoint, HostConnection, LLMConfig } from "../src/llm_clients/config.ts";
import {
  blockProblem,
  checkModelConfig,
  type Connection,
  ConnectionSnapshot,
  HOST_KEY_MARKER,
  headerNameDiff,
  keyIdentity,
  legacyConnectionConfig,
  legacyConnectionMap,
  refusedBlock,
  splitModelConfig,
  validateConnection,
} from "../src/llm_clients/connection.ts";

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

test("split model config refuses header names without the marker", () => {
  const { [HOST_KEY_MARKER]: _marker, ...unmarked } = MARKED;
  const error = rangeError(() =>
    splitModelConfig({ ...unmarked, _header_names: ["X-Gateway"], headers: { "X-Gateway": "v" } }),
  );
  expect(error.message).toContain("_header_names");
});

test("split carries headers beside the key", () => {
  const config = new LLMConfig({
    provider: "anthropic",
    model: "m",
    api_key: "k",
    headers: { "X-Gateway": HEADER_VALUE },
  });
  const [validated] = splitModelConfig(config);
  expect(validated.headers).toEqual({ "X-Gateway": HEADER_VALUE });
  expect(validated.apiKey).toBe("k");
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

  test("refusedBlock names the block only when the block alone was refused", () => {
    expect(refusedBlock({ provider: "anthropic", model: 1n })).toBe("");
    expect(refusedBlock({ provider: "azure", model: "m" })).toBe("");
    expect(refusedBlock({ model: "m" })).toBe("");
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
    expect(rangeError(() => checkModelConfig(subscription)).message).toContain("`openai.temperature`");
    expect(refusedBlock(subscription)).toBe(".openai");
    // A host key sends the same agent to the Responses API, which accepts the block.
    checkModelConfig({ ...subscription, api_key: "host-key" });
  });
});
