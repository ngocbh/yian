/**
 * The AGENT memory adapter's compatibility descriptor, a port of
 * `test_compatibility_matches_factory_defaults_and_config_carriers` of
 * `tests/engine/test_agent_session.py`. The expected `message_format` is TS's own
 * `yian.message.v1`, not Python's `yian.langchain-message.v1`.
 *
 * TS-only: a record's `api_key` is not kept when the provider resolves from the defaults, a
 * passed `snapshot` is used, and no provider anywhere throws `RangeError`.
 *
 * Later (T6-6): every other test of the file, which needs the rest of the adapter, session memory
 * and `run_flow`.
 */

import { afterEach, beforeEach, expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import type { Env } from "../../src/defaults.ts";
import { MESSAGE_FORMAT } from "../../src/llm/messages.ts";
import { LLMConfig } from "../../src/llm_clients/config.ts";
import { ConnectionSnapshot } from "../../src/llm_clients/connection.ts";
import { compatibility } from "../../src/nodes/agent/memory.ts";

// Each test's environment: conftest's base (a private store in a fresh directory, the default
// model) with the default provider this test sets.
let home: string;
let vars: Map<string, string>;
const env: Env = (name) => vars.get(name);

beforeEach(() => {
  home = mkdtempSync(join(tmpdir(), "yian-agent-session-"));
  vars = new Map([
    ["YIAN_AUTH_FILE", join(home, "auth.json")],
    ["YIAN_DEFAULT_PROVIDER", "OpenAI"],
    ["YIAN_DEFAULT_MODEL", "gpt-5.5"],
  ]);
});

afterEach(() => {
  rmSync(home, { recursive: true, force: true });
});

const expected = { provider: "openai", message_format: MESSAGE_FORMAT };

test("compatibility matches factory defaults and config carriers", () => {
  expect(compatibility(null, { env })).toEqual(expected);
  expect(compatibility(new LLMConfig({}), { env })).toEqual(expected);
  expect(compatibility(new LLMConfig({ provider: "openai" }), { env })).toEqual(expected);
  expect(compatibility({ provider: "OpenAI", model: "other", api_key: "not-retained" }, { env })).toEqual(expected);
});

test("a record's api_key is not kept when the provider comes from the defaults", () => {
  const descriptor = compatibility({ provider: null, api_key: "not-retained" }, { env });
  expect(descriptor).toEqual(expected);
  expect(JSON.stringify(descriptor)).not.toContain("not-retained");
});

test("a passed snapshot is used", () => {
  vars = new Map([["YIAN_AUTH_FILE", join(home, "auth.json")]]);
  const snapshot = new ConnectionSnapshot({
    defaults: { provider: "anthropic", model: "claude-test" },
    providerModels: { anthropic: "claude-test" },
    connections: {},
  });
  expect(compatibility(null, { snapshot, env })).toEqual({ provider: "anthropic", message_format: MESSAGE_FORMAT });
});

test("no provider anywhere throws RangeError", () => {
  vars = new Map([["YIAN_AUTH_FILE", join(home, "auth.json")]]);
  expect(() => compatibility(null, { env })).toThrow(RangeError);
  expect(() => compatibility({ model: "m" }, { env })).toThrow(RangeError);
});
