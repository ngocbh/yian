/**
 * Mirrors `tests/engine/test_llm_cascade_call_time.py`: the six selection tests; the cascade
 * tests go with `compile/llm_cascade.ts`.
 */

import { describe, expect, test } from "bun:test";

import { HostConnection } from "../../src/llm_clients/config.ts";
import {
  type Connection,
  ConnectionSnapshot,
  recordSelection,
  restoreSelection,
  type SelectionRecord,
} from "../../src/llm_clients/connection.ts";

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
