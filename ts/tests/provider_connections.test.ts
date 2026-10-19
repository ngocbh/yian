/**
 * Mirrors `tests/test_provider_connections.py`; the factory tests go with
 * `llm_clients/factory.ts`, `test_run_records_connections` with `runFlow`.
 */

import { describe, expect, test } from "bun:test";

import { HostConnection } from "../src/llm_clients/config.ts";
import {
  type Connection,
  ConnectionSnapshot,
  headerNameDiff,
  keyIdentity,
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
