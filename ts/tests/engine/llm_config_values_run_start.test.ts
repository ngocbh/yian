/**
 * A port of `tests/engine/test_llm_config_values_run_start.py`: run start checks the run's inputs,
 * then resolves and checks the `llm_config` values reading them.
 *
 * Ported: the other 28 tests (the parametrized groups whole). The model is a `ScriptedModel` given
 * through `serving` where Python patches `model_from_config`; Python's `fake_model` (the messages
 * of each model call) is the model's `requests`. The `run_started`, `node_succeeded` and
 * `node_failed` records are read from the run's JSONL log. Saved logins go to the run's own
 * credential store (`YIAN_AUTH_FILE` in its `env`).
 *
 * Every run's environment holds Python's conftest default provider and model.
 *
 * Changed: run-start errors are `RangeError`s (a `LayerFieldError` for a field's) where Python
 * raises `ValueError`; ints have no digit limit (decisions 4a, 23), so in the uncoercible-input row
 * the 5000-digit `window` is accepted (a run with a good `llm_config` takes it whole) and the bad
 * `llm_config` is the reported error; names and null are written in the text form (`"nope"`,
 * `["bogus"]`, `null`; decisions 7, 43); the golden `run_started` record is TS's (schema 13, with
 * `defaults`, `call_time_llm` and a wall-clock `at`, masked; TS starts its own versioned logs,
 * decisions 8, 16), and its initial pool is the run start's `pool_commit` records after it.
 *
 * Covered by `run-agent/engine_llm_config_values_run_start__a_bad_run_config_beats_a_missing_input`:
 * `test_a_missing_required_input_does_not_preempt_a_bad_llm_config`;
 * `__an_input_record_with_an_unknown_field_fails`:
 * `test_an_unknown_field_in_a_whole_record_from_an_input_fails_at_run_start`;
 * `__an_input_record_may_not_set_the_connection`:
 * `test_a_whole_record_from_an_input_may_not_set_a_recorded_connection`;
 * `__an_unknown_input_provider_names_the_field`:
 * `test_an_unknown_provider_from_an_input_fails_naming_the_field`;
 * `__an_unknown_flow_provider_names_the_flow`:
 * `test_a_flow_level_unknown_provider_from_an_input_fails_at_the_flow_field`;
 * `__a_value_that_breaks_its_tag_names_the_field`: `test_a_node_value_that_breaks_its_tag_names_the_field`;
 * `__a_missing_input_the_config_reads_raises`:
 * `test_a_missing_required_input_the_config_reads_fails_before_the_log`;
 * `__an_input_block_with_an_unknown_key_fails`:
 * `test_a_whole_block_from_an_input_that_breaks_the_schema_fails_at_run_start`;
 * `__an_input_the_config_reads_is_type_checked`:
 * `test_an_input_the_config_reads_is_checked_against_its_contract`. Each pins the class, the whole
 * message, an empty trace (no log) and no events (so no model call).
 *
 * Already ported: `test_a_non_finite_value_names_the_block_field` in `run.test.ts` ("a non-finite
 * value names the block field").
 */

import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterAll, expect, test } from "bun:test";

import { resolveRunLlm } from "../../src/compile/llm_cascade.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import type { RunEvent } from "../../src/events.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { ApiKeyCredentials, OAuthCredentials } from "../../src/llm_clients/auth/common.ts";
import { AuthStore } from "../../src/llm_clients/auth/store.ts";
import type { AgentNode } from "../../src/nodes/agent/node.ts";
import { Parameter } from "../../src/nodes/contracts.ts";
import { runFlow, type RunFlowOptions, runStartInputs } from "../../src/run.ts";
import { hostEnv } from "./mcp_helpers.ts";
import { rejected, serving, textReply } from "./run_state_fixtures.ts";

const dir = mkdtempSync(join(tmpdir(), "yian-llm-values-run-start-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));
// Python's conftest: the default provider and model.
const CONFTEST = { YIAN_DEFAULT_PROVIDER: "openai", YIAN_DEFAULT_MODEL: "gpt-5.5" };
const env = hostEnv(dir, CONFTEST);

let files = 0;

/** A fresh path in the file's temp dir. */
const fresh = (name: string) => join(dir, `${name}${files++}`);

const FLOW = `id: f
name: f
input:
  topic: !str
  window: !int 30
nodes:
  agent:
    kind: agent
    input: {topic: "\${input.topic}", window: "\${input.window}"}
    prompt: "hi \${topic} \${window}"
output: \${agent.output}
`;

const GOOD_LLM = { provider: "openai", model: "gpt-5.5" };
const BAD_LLM = { bogus: 1 };
const BAD_LLM_ERROR = /llm_config has unknown field\(s\) \["bogus"\]/;

/** A model answering every call with `ok`, served under `runEnv`; `requests` holds each call. */
function fakeModel(runEnv = env) {
  const model = new ScriptedModel(Array.from({ length: 4 }, () => textReply("ok")));
  return { model, models: serving(model, runEnv) };
}

/** Run `text` on `inputs` in the file's environment with a scripted model unless `options` gives one. */
async function run(text: string, inputs: Record<string, unknown>, options: RunFlowOptions = {}) {
  const runEnv = options.env ?? env;
  return runFlow(await loadFlow(text, { env: runEnv }), inputs, { models: fakeModel(runEnv).models, ...options, env: runEnv });
}

/** The records of the JSONL log at `path`. */
function records(path: string): { kind: string; payload: Record<string, any> }[] {
  return readFileSync(path, "utf8")
    .split("\n")
    .filter((line) => line !== "")
    .map((line) => JSON.parse(line));
}

/** The error `promise` rejects with, of class `kind`. */
async function raised(promise: Promise<unknown>, kind: typeof RangeError): Promise<Error> {
  const error = await rejected(promise);
  expect(error).toBeInstanceOf(kind);
  return error as Error;
}

test("an unknown input is reported before a bad llm config", async () => {
  const log = fresh("run.jsonl");
  const error = await raised(run(FLOW, { topic: "a", nope: 1 }, { log, llmConfig: BAD_LLM }), RangeError);
  expect(error.message).toContain('unknown input "nope"');
  expect(existsSync(log)).toBe(false);
});

test("an uncoercible input is reported before a bad llm config", async () => {
  const log = fresh("run.jsonl");
  // The name is Python's: there the 5000-digit text breaks its int limit. Here it is an int, so
  // the bad llm_config is what's reported, and with a good one the run takes the whole value.
  const error = await raised(run(FLOW, { topic: "a", window: "1".repeat(5000) }, { log, llmConfig: BAD_LLM }), RangeError);
  expect(error.message).toMatch(BAD_LLM_ERROR);
  expect(existsSync(log)).toBe(false);

  const { model, models } = fakeModel();
  const result = await run(FLOW, { topic: "a", window: "1".repeat(5000) }, { log: fresh("run.jsonl"), models, llmConfig: GOOD_LLM });
  expect(result.status).toBe("succeeded");
  expect(result.input["window"]).toBe(BigInt("1".repeat(5000)));
  expect(model.requests).toHaveLength(1);
});

test("a bad llm config alone is reported as before", async () => {
  const log = fresh("run.jsonl");
  const error = await raised(run(FLOW, { topic: "a" }, { log, llmConfig: BAD_LLM }), RangeError);
  expect(error.message).toMatch(BAD_LLM_ERROR);
  expect(existsSync(log)).toBe(false);
});

test("a missing required input stays a logged start failure", async () => {
  const log = fresh("run.jsonl");
  const { model, models } = fakeModel();
  const result = await run(FLOW, {}, { log, models, llmConfig: GOOD_LLM });
  expect(result.status).toBe("failed");
  const all = records(log);
  expect(all[0]!.kind).toBe("run_started");
  const failed = all.filter((record) => record.kind === "node_failed");
  expect(failed.map((record) => record.payload["failure"]["node_id"])).toEqual(["__start__"]);
  const failure = failed[0]!.payload["failure"];
  expect(failure["code"]).toBe("binding.evaluate");
  expect(failure["message"]).toContain("required input 'topic' is unbound");
  expect(model.requests).toEqual([]);
});

const GOLDEN_RUN_STARTED = {
  kind: "run_started",
  payload: {
    call_time_llm: null,
    defaults: {},
    effective_llm_by_static_path: {
      agent: {
        _connection: {
          credential_id: null,
          endpoint: "https://api.openai.com/v1",
          method: "api_key",
          source: "missing",
          store_path: null,
        },
        model: "gpt-5.5",
        provider: "openai",
      },
    },
    flow: { dir: "<dir>", files: { "<flow>": { refs: {}, text: FLOW } }, root: "<flow>" },
    flow_fingerprint: "3d9b9bbd0808d27babce1bded77b3e90b642d179cc9678abcae0f4da66d4bfa7",
    learnable: false,
    memories: null,
    memory_snapshots: null,
    memory_stacks: null,
    num_workers: 8,
    parent: null,
    skills: {},
    start_wiring_override: { topic: "${input.topic}" },
    tools: {},
    yian_version: "0.0.0",
  },
  run_id: "golden",
  schema: 16,
  seq: 0,
  at: "<at>",
};

test("a run with a default input records and returns as before", async () => {
  const log = fresh("run.jsonl");
  const { model, models } = fakeModel();
  const result = await run(FLOW, { topic: "ACME" }, { log, runId: "golden", models, llmConfig: GOOD_LLM });
  expect(result.status).toBe("succeeded");
  expect(result.input).toEqual({ topic: "ACME" });
  expect(result.output).toBe("ok");
  // START still fills the default: the model sees window=30.
  expect(JSON.stringify(model.requests[0])).toContain("hi ACME 30");
  const [started, ...commits] = records(log).slice(0, 5) as unknown as [Record<string, any>, ...Record<string, any>[]];
  // The run start's commits follow RunStarted: the system values by key, then the START inputs.
  const startId = "__start__";
  expect(commits.map(({ kind, payload }) => [kind, payload["key"], payload["kind"]])).toEqual([
    ["pool_commit", "system.now", "datetime"],
    ["pool_commit", "system.run_id", "str"],
    ["pool_commit", "system.today", "date"],
    ["pool_commit", startId, "object"],
  ]);
  expect(commits[1]!["payload"]).toEqual({ node_id: startId, key: "system.run_id", kind: "str", value: "golden" });
  expect(commits[3]!["payload"]).toEqual({ node_id: startId, key: startId, kind: "object", value: { topic: "ACME" } });
  expect(Number.isNaN(Date.parse(started["at"]))).toBe(false);
  started["at"] = "<at>";
  // Loaded without a folder, the flow's base folder is the current one.
  expect(started["payload"]["flow"]["dir"]).toBe(process.cwd());
  started["payload"]["flow"]["dir"] = "<dir>";
  expect(started).toEqual(GOLDEN_RUN_STARTED);
});

test("run start inputs fill valid defaults and leave bad ones to start", () => {
  const params = [
    new Parameter("topic", { required: true }),
    new Parameter("window", { hasDefault: true, default: 30n }),
    new Parameter("bad", { hasDefault: true, default: Number.NaN }),
    new Parameter("given", { hasDefault: true, default: "unused" }),
  ];
  const supplied = { given: null };
  expect(runStartInputs(params, supplied)).toEqual({ given: null, window: 30n });
  expect(supplied).toEqual({ given: null });
});

/** The `run_started` record's effective configs. */
const effective = (log: string) => records(log)[0]!.payload["effective_llm_by_static_path"];

/** The `[provider, model]` of each recorded effective config, by static path. */
function modelsOf(log: string): Record<string, [unknown, unknown]> {
  return Object.fromEntries(
    Object.entries(effective(log) as Record<string, Record<string, unknown>>).map(([path, config]) => [
      path,
      [config["provider"], config["model"]],
    ]),
  );
}

const VALUES = `id: f
name: f
input:
INPUTS
nodes:
  chat:
    kind: agent
    prompt: Go.
    llm_config: LLM
output: \${chat.output}
`;

const values = (inputs: string, llm: string) => VALUES.replace("INPUTS", inputs).replace("LLM", llm);

test("provider and model from inputs land in the run started map", async () => {
  const text = values("  p: !str\n  m: !str", "{provider: '${input.p}', model: '${input.m}'}");
  const log = fresh("run.jsonl");
  const { model, models } = fakeModel();
  const result = await run(text, { p: "anthropic", m: "claude-x" }, { log, models });
  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(modelsOf(log)).toEqual({ chat: ["anthropic", "claude-x"] });
  expect(model.requests).toHaveLength(1);
});

test("an omitted input with a default gives its default", async () => {
  const text = values("  m: !str gpt-from-default", "{model: '${input.m}'}");
  const log = fresh("run.jsonl");
  expect((await run(text, {}, { log })).status).toBe("succeeded");
  expect(modelsOf(log)).toEqual({ chat: ["openai", "gpt-from-default"] });
});

/** An environment whose credential store is a fresh `auth.json`, and that store. */
function ownStore() {
  const auth = fresh("auth");
  mkdirSync(auth, { mode: 0o700 });
  const runEnv = hostEnv(auth, CONFTEST);
  return { runEnv, store: new AuthStore(join(auth, "auth.json"), { env: runEnv }) };
}

test("an omitted optional model falls through to the saved model", async () => {
  const { runEnv, store } = ownStore();
  await store.save("anthropic", new ApiKeyCredentials("saved-key"), "claude-saved");
  const text = values("  m: !Optional(str)", "{provider: anthropic, model: '${input.m}'}");
  const log = fresh("run.jsonl");
  expect((await run(text, {}, { log, env: runEnv })).status).toBe("succeeded");
  expect(modelsOf(log)).toEqual({ chat: ["anthropic", "claude-saved"] });
  const given = fresh("given.jsonl");
  expect((await run(text, { m: "claude-given" }, { log: given, env: runEnv })).status).toBe("succeeded");
  expect(modelsOf(given)).toEqual({ chat: ["anthropic", "claude-given"] });
});

const WHOLE_RECORD = `id: f
name: f
typedefs:
  Cfg:
    provider: str
    model: str
input:
  cfg: !Cfg
llm_config: \${input.cfg}
defs:
  child:
    nodes:
      inner: {kind: agent, prompt: Inner.}
    output: \${inner.output}
nodes:
  chat:
    kind: agent
    prompt: Go.
  c:
    kind: call
    call: child
    input: {}
    depends_on: [chat]
output: \${c.output}
`;

test("a flow level whole record from an input reaches every agent", async () => {
  const log = fresh("run.jsonl");
  const cfg = { provider: "anthropic", model: "claude-whole" };
  expect((await run(WHOLE_RECORD, { cfg }, { log })).status).toBe("succeeded");
  expect(modelsOf(log)).toEqual({ chat: ["anthropic", "claude-whole"], "c/inner": ["anthropic", "claude-whole"] });
});

const LIB = `id: lib
name: lib
input: {m: !str }
llm_config: {model: '\${input.m}'}
defs:
  inner:
    nodes:
      a: {kind: agent, prompt: Go.}
    output: \${a.output}
nodes:
  c: {kind: call, call: inner, input: {}}
output: \${c.output}
`;

test("a file with defs loaded as the root resolves its input value", async () => {
  const log = fresh("run.jsonl");
  const flow = await loadFlow(LIB, { sourceLabel: "lib.yaml", env });
  expect((await runFlow(flow, { m: "gpt-lib" }, { log, env, models: fakeModel().models })).status).toBe("succeeded");
  expect(modelsOf(log)).toEqual({ "c/a": ["openai", "gpt-lib"] });
});

const PASS_DOWN = `id: f
name: f
input: {m: !str }
llm_config: {provider: anthropic, model: '\${input.m}'}
defs:
  child:
    nodes:
      a: {kind: agent, prompt: Go.}
      b: {kind: agent, prompt: Go., llm_config: {model: own-model}}
    output: \${a.output}
nodes:
  top: {kind: agent, prompt: Top.}
  c:
    kind: call
    call: child
    input: {}
    depends_on: [top]
output: \${c.output}
`;

test("a root flow input value is passed down to child agents", async () => {
  const log = fresh("run.jsonl");
  expect((await run(PASS_DOWN, { m: "claude-root" }, { log })).status).toBe("succeeded");
  expect(modelsOf(log)).toEqual({
    top: ["anthropic", "claude-root"],
    "c/a": ["anthropic", "claude-root"],
    "c/b": ["anthropic", "own-model"],
  });
});

test("a run start agent binds no reserved llm config input", async () => {
  const text = values("  m: !str", "{model: '${input.m}'}");
  const flow = await loadFlow(text, { env });
  expect(flow.compiled.wiring.get("chat")!.has("__llm_config")).toBe(true);
  const log = fresh("run.jsonl");
  const result = await runFlow(flow, { m: "gpt-learn" }, { log, env, models: fakeModel().models, learnable: true });
  expect(result.status).toBe("succeeded");
  const succeeded = records(log).filter(
    (record) => record.kind === "node_succeeded" && record.payload["node_id"] === "chat",
  );
  const capture = succeeded[0]!.payload["forward_capture"];
  expect(capture).not.toBeNull();
  expect("__llm_config" in capture["bound_inputs"]).toBe(false);
  expect(capture["input_names"]).not.toContain("__llm_config");
  expect(capture["node_data"]["effective_llm_config"]["model"]).toBe("gpt-learn");
});

const MIXED = `id: f
name: f
input: {p: !str }
llm_config: {provider: '\${input.p}', model: '\${pick.output}', openai: {temperature: 0.5}}
defs:
  child:
    nodes:
      a: {kind: agent, prompt: Go., llm_config: {model: child-model}}
    output: \${a.output}
nodes:
  pick:
    kind: agent
    prompt: Pick.
    output: str
    llm_config: {model: pick-model}
  c:
    kind: call
    call: child
    input: {}
    depends_on: [pick]
output: \${c.output}
`;

test("a mixed flow layer passes its literal and input fields and drops the rest", async () => {
  // Every agent sets its own model, so the flow's run-time model reaches none of them.
  const log = fresh("run.jsonl");
  expect((await run(MIXED, { p: "openai" }, { log })).status).toBe("succeeded");
  const recorded = effective(log);
  expect(modelsOf(log)).toEqual({ pick: ["openai", "pick-model"], "c/a": ["openai", "child-model"] });
  expect(recorded["c/a"]["openai"]).toEqual({ temperature: 0.5 });
  expect(recorded["pick"]["openai"]).toEqual({ temperature: 0.5 });
});

/** Run `text` and expect run start to refuse it matching `pattern`, with no log, event or model call. */
async function failsBeforeAnyNode(
  text: string,
  inputs: Record<string, unknown>,
  pattern: RegExp,
  runEnv = env,
): Promise<string> {
  const log = fresh("failed.jsonl");
  const events: RunEvent[] = [];
  const { model, models } = fakeModel(runEnv);
  const error = await raised(
    run(text, inputs, { log, env: runEnv, models, onEvent: (event) => void events.push(event) }),
    RangeError,
  );
  expect(error.message).toMatch(pattern);
  expect(existsSync(log)).toBe(false);
  expect(events).toEqual([]);
  expect(model.requests).toEqual([]);
  return error.message;
}

/** An environment whose credential store holds an OpenAI subscription login. */
async function subscriptionEnv() {
  const { runEnv, store } = ownStore();
  const credentials = new OAuthCredentials({
    accessToken: "access-secret",
    refreshToken: "refresh-secret",
    expiresAt: Date.now() / 1000 + 3600,
    accountId: "account",
  });
  await store.save("openai", credentials, "gpt-5.5");
  return runEnv;
}

test("a provider from an input is checked against its connection", async () => {
  const runEnv = await subscriptionEnv();
  const text = values("  p: !str", "{provider: '${input.p}', openai: {temperature: 0.2}}");
  const message = await failsBeforeAnyNode(
    text,
    { p: "openai" },
    /^node 'chat' llm_config\.openai: `openai\.temperature` is not accepted by the ChatGPT subscription endpoint/,
    runEnv,
  );
  expect(message).not.toContain("secret");
});

test("a block field from an input out of range fails at its field", async () => {
  const text = values(
    "  b: !int",
    "{provider: anthropic, model: m, anthropic: {thinking: {type: enabled, budget_tokens: '${input.b}'}}}",
  );
  await failsBeforeAnyNode(text, { b: 5 }, /^node 'chat' llm_config\.anthropic\.thinking\.budget_tokens: .*1024/);
});

test("a literal typed provider given an unknown member fails at run start", async () => {
  const text = values("  p: !Some", "{provider: '${input.p}'}").replace(
    "input:",
    "typedefs:\n  Some: Literal(openai, nope)\ninput:",
  );
  expect((await run(text, { p: "openai" })).status).toBe("succeeded");
  await failsBeforeAnyNode(text, { p: "nope" }, /^node 'chat' llm_config\.provider: not a known provider/);
});

const COMPOSER = `id: f
name: f
input: {proj: !str , m: !str , p: !str }
nodes:
  ask:
    kind: human_input
    input: {proj: '\${input.proj}'}
    adaptive_questions:
      prompt: "Q \${proj}"
      llm_config: {provider: '\${input.p}', model: '\${input.m}'}
output: \${ask.output}
`;

test("a composer value from an input resolves at run start", async () => {
  const flow = (await loadFlow(COMPOSER, { env })).compiled;
  const composer = [...flow.nodes].find(([, node]) => ((node as AgentNode).llmConfigLabel ?? null) !== null)![0];
  const resolved = resolveRunLlm(flow, {}, { defaults: {}, runInputs: { proj: "x", m: "claude-c", p: "anthropic" } });
  expect(resolved.effective[composer]!["provider"]).toBe("anthropic");
  expect(resolved.effective[composer]!["model"]).toBe("claude-c");
});

test("a composer value error names the gate", async () => {
  const site = "node 'ask' adaptive_questions\\.llm_config";
  await failsBeforeAnyNode(
    COMPOSER,
    { proj: "x", p: "openai" },
    new RegExp(`^${site}\\.model: input 'm' is required and was not given`),
  );
  await failsBeforeAnyNode(
    COMPOSER,
    { proj: "x", m: "m", p: "nope" },
    new RegExp(`^${site}\\.provider: not a known provider`),
  );
});

const HOST_ONLY = "credentials and where they are sent stay with the host or yian auth";
const HOST_ONLY_CASES: [string, unknown][] = [
  ["base_url", "https://elsewhere.example/v1"],
  ["api_key", "sk-from-input"],
  ["organization", "org-from-input"],
  ["project", "proj-from-input"],
  ["headers", { "X-From-Input": "secret-header" }],
];

/** The secret text of a host-only value: the header value for `headers`. */
const secretOf = (value: unknown) => (typeof value === "string" ? value : Object.values(value as object)[0]);

test.each(HOST_ONLY_CASES)("a node whole record from an input may not set a host only field (%s)", async (field, value) => {
  // The record also says `provider: azure`, which must not let its base_url through.
  const text = values("  cfg: !object", "'${input.cfg}'");
  const cfg = { provider: "azure", model: "dep", [field]: value };
  const message = await failsBeforeAnyNode(text, { cfg }, new RegExp(`^node 'chat' llm_config\\.${field}: ${HOST_ONLY}$`));
  expect(message).not.toContain(secretOf(value));
});

const FLOW_RECORD = `id: f
name: f
input: {cfg: !object }
llm_config: \${input.cfg}
nodes:
  chat: {kind: agent, prompt: Go.}
output: \${chat.output}
`;

test.each(HOST_ONLY_CASES)("a flow whole record from an input may not set a host only field (%s)", async (field, value) => {
  const cfg = { provider: "azure", model: "dep", [field]: value };
  const message = await failsBeforeAnyNode(FLOW_RECORD, { cfg }, new RegExp(`^flow llm_config\\.${field}: ${HOST_ONLY}$`));
  expect(message).not.toContain(secretOf(value));
});

test("a literal azure base url beside a literal azure provider still resolves", async () => {
  const text = values("  m: !str", "{provider: azure, base_url: 'https://res.openai.azure.com', model: '${input.m}'}");
  const flow = (await loadFlow(text, { env })).compiled;
  const resolved = resolveRunLlm(flow, {}, { defaults: {}, runInputs: { m: "dep" } });
  expect(resolved.effective["chat"]!["base_url"]).toBe("https://res.openai.azure.com");
  expect(resolved.effective["chat"]!["model"]).toBe("dep");
});

const INHERITED_BLOCK = `id: f
name: f
input: {p: !str }
llm_config: {provider: '\${input.p}', openai: {temperature: 0.2}}
defs:
  child:
    nodes:
      a: {kind: agent, prompt: Go.}
    output: \${a.output}
nodes:
  c: {kind: call, call: child, input: {}}
output: \${c.output}
`;

test("a child agent refused for an inherited input value names the flow layer", async () => {
  const runEnv = await subscriptionEnv();
  const message = await failsBeforeAnyNode(
    INHERITED_BLOCK,
    { p: "openai" },
    /^Agent 'c\/a' \(flow llm_config\)\.openai: `openai\.temperature` is not accepted by the ChatGPT subscription endpoint/,
    runEnv,
  );
  expect(message).not.toContain("secret");
});

const UNREAD_INPUT = `id: f
name: f
input:
  m: !str
  n: !int
nodes:
  chat:
    kind: agent
    input: {n: "\${input.n}"}
    prompt: "Go \${n}."
    llm_config: {provider: openai, model: '\${input.m}'}
output: \${chat.output}
`;

test.each([
  [{ m: "gpt-x" }, "required input 'n' is unbound"],
  [{ m: "gpt-x", n: "abc" }, "input 'n'"],
] as [Record<string, unknown>, string][])(
  "an input the config does not read stays a logged start failure (%o)",
  async (inputs, message) => {
    const log = fresh("run.jsonl");
    const { model, models } = fakeModel();
    const result = await run(UNREAD_INPUT, inputs, { log, models });
    expect(result.status).toBe("failed");
    const all = records(log);
    expect(all[0]!.kind).toBe("run_started");
    expect(modelsOf(log)).toEqual({ chat: ["openai", "gpt-x"] });
    const failed = all.filter((record) => record.kind === "node_failed");
    expect(failed.map((record) => record.payload["failure"]["node_id"])).toEqual(["__start__"]);
    expect(failed[0]!.payload["failure"]["message"]).toContain(message);
    expect(model.requests).toEqual([]);
  },
);

const PRECEDENCE = `id: f
name: f
input:
  nm: !Optional(str)
  fm: !Optional(str)
llm_config: {model: '\${input.fm}'}
nodes:
  chat:
    kind: agent
    prompt: Go.
    llm_config: {model: '\${input.nm}'}
output: \${chat.output}
`;

test.each([
  [{ nm: "node-model", fm: "flow-model" }, "node-model"],
  [{ fm: "flow-model" }, "flow-model"],
  [{ nm: null, fm: "flow-model" }, "flow-model"],
  [{}, "host-model"],
] as [Record<string, unknown>, string][])(
  "a null node value falls to the flow value then the host layer (%o)",
  async (inputs, model) => {
    const log = fresh("run.jsonl");
    const llmConfig = { provider: "openai", model: "host-model" };
    expect((await run(PRECEDENCE, inputs, { log, llmConfig })).status).toBe("succeeded");
    expect(modelsOf(log)).toEqual({ chat: ["openai", model] });
  },
);

const TAGGED = `id: f
name: f
typedefs: {D: "dict(str, str)"}
input:
  d: !D
  t: !float 1.0
llm_config: FLOW
nodes:
  chat:
    kind: agent
    prompt: Go.
    llm_config: NODE
output: \${chat.output}
`;

const tagged = (node: string, flow = "{provider: openai}") => TAGGED.replace("NODE", node).replace("FLOW", flow);

test("a flow value that breaks its tag names the flow field", async () => {
  await failsBeforeAnyNode(
    tagged("{}", "{provider: openai, model: !str '${input.d.zzz}'}"),
    { d: {} },
    /^flow llm_config\.model: null does not match declared type str$/,
  );
});

test("a composer value that breaks its tag names the gate field", async () => {
  const text = `id: f
name: f
typedefs: {D: "dict(str, str)"}
input: {d: !D }
llm_config: {provider: openai}
nodes:
  ask:
    kind: human_input
    prompt: hi
    adaptive_questions:
      prompt: Q
      llm_config: {model: !str '\${input.d.zzz}'}
output: \${ask.output}
`;
  await failsBeforeAnyNode(
    text,
    { d: {} },
    /^node 'ask' adaptive_questions\.llm_config\.model: null does not match declared type str$/,
  );
});

test("a block value out of bounds names the field once", async () => {
  await failsBeforeAnyNode(
    tagged("{model: m, openai: {temperature: '${input.t}'}}"),
    { d: {}, t: 5.0 },
    /^node 'chat' llm_config\.openai\.temperature: Input should be less than or equal to 2$/,
  );
});
