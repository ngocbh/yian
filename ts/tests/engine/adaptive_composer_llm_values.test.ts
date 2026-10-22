/**
 * A port of `tests/engine/test_adaptive_composer_llm_values.py`: `llm_config` values on an
 * adaptive gate's composer, end to end (run start, call time, errors).
 *
 * Ported: all 9 tests. The composer's model is a `ScriptedModel` given through `serving`, which
 * records the config of each model built, where Python patches `model_from_config`; it answers the
 * composer's `list(Question)` output as the `{"items": [...]}` wrapper. The `node_step` records
 * are read from the run's JSONL log. The CODE bodies are inline JavaScript.
 *
 * Changed: the run-start error is a `RangeError` where Python raises `ValueError`; a run with no
 * call-time agent writes `call_time_llm: null` in its `run_started` record, where Python leaves
 * the key out.
 *
 * `engine_adaptive_composer_llm_values__a_composer_call_time_error_names_the_gate` and
 * `__an_error_in_a_call_names_the_callsite` don't cover their tests: neither pins that no model
 * was built.
 */

import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterAll, expect, test } from "bun:test";

import { loadFlow } from "../../src/compose/loader.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { runFlow } from "../../src/run.ts";
import type { AskUserRequest } from "../../src/tools/registry.ts";
import { hostEnv } from "./mcp_helpers.ts";
import { rejected, serving, textReply } from "./run_state_fixtures.ts";

const dir = mkdtempSync(join(tmpdir(), "yian-adaptive-llm-values-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));
const env = hostEnv(dir);

const SITE = "node 'ask' adaptive_questions.llm_config";
const COMPOSER = "ask__compose";

const COMPOSED = [
  {
    question: "Which database?",
    header: "DB",
    options: [
      { label: "Postgres", description: "Relational." },
      { label: "SQLite", description: "Embedded." },
    ],
    multi_select: false,
  },
];

/** A model access whose every model composes `COMPOSED`; `configs` records each build's config. */
const composing = () => serving(new ScriptedModel([textReply(JSON.stringify({ items: COMPOSED }))]), env);

/** A flow whose `ask` gate composes its questions with `llmConfig`. */
function gate(llmConfig: string, options: { inputs?: string; typedefs?: string; pick?: string } = {}): string {
  const { inputs = "{proj: !str }", typedefs = "", pick = "" } = options;
  return `id: f
name: f
${typedefs}input: ${inputs}
nodes:
${pick}  ask:
    kind: human_input
    input: {proj: '\${input.proj}'}
    adaptive_questions:
      prompt: "Q \${proj}"
      llm_config: ${llmConfig}
output: \${ask.output}
`;
}

let runs = 0;

/** Run `text` on `inputs` with a JSONL log; return the result, the log's records and the `ask_user` requests. */
async function run(text: string, inputs: Record<string, unknown>, models: ReturnType<typeof composing>) {
  const log = join(dir, `run${runs++}.jsonl`);
  const requests: AskUserRequest[] = [];
  const result = await runFlow(await loadFlow(text, { env }), inputs, {
    log,
    env,
    models,
    tools: {
      ask_user: (request: AskUserRequest) => {
        requests.push(request);
        return { DB: "Postgres" };
      },
    },
  });
  const records = readFileSync(log, "utf8")
    .split("\n")
    .filter((line) => line !== "")
    .map((line) => JSON.parse(line));
  return { result, records, requests };
}

/** The composer's recorded steps. */
function composerSteps(records: { kind: string; payload: Record<string, any> }[]): Record<string, any>[] {
  return records
    .filter((record) => record.kind === "node_step" && record.payload["node_id"] === COMPOSER)
    .map((record) => record.payload["step"]);
}

/** The questions of a request. */
const questionsOf = (request: AskUserRequest) => request.questions as { question: string; header: string }[];

/** The `[provider, model]` of a config. */
const selection = (config: Readonly<Record<string, unknown>>) => [config["provider"], config["model"]];

test("a composer model from an input resolves at run start", async () => {
  const models = composing();
  const text = gate("{provider: openai, model: '${input.m}'}", { inputs: "{proj: !str , m: !str }" });
  const { result, records, requests } = await run(text, { proj: "x", m: "gpt-input" }, models);
  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(result.output).toEqual({ DB: "Postgres" });
  expect(questionsOf(requests[0]!)[0]!.header).toBe("DB");
  const started = records[0].payload;
  expect(selection(started["effective_llm_by_static_path"][COMPOSER])).toEqual(["openai", "gpt-input"]);
  expect(started["call_time_llm"]).toBeNull();
  expect(composerSteps(records)).toEqual([]);
  expect(models.configs.map(selection)).toEqual([["openai", "gpt-input"]]);
});

const PICK = `  pick: {kind: code, output: str, code: "return 'gpt-picked'"}\n`;

test("a composer model from an upstream node resolves at call time", async () => {
  const models = composing();
  const text = gate("{provider: openai, model: '${pick.output}'}", { pick: PICK });
  const { result, records, requests } = await run(text, { proj: "x" }, models);
  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(questionsOf(requests[0]!)[0]!.question).toBe("Which database?");
  const started = records[0].payload;
  expect(COMPOSER in started["effective_llm_by_static_path"]).toBe(false);
  expect(COMPOSER in started["call_time_llm"]["agents"]).toBe(true);
  const steps = composerSteps(records);
  expect(steps).toHaveLength(1);
  expect(selection(steps[0]!["effective_llm_config"])).toEqual(["openai", "gpt-picked"]);
  expect(models.configs.map(selection)).toEqual([["openai", "gpt-picked"]]);
});

const CFG = "typedefs:\n  Cfg: {provider: str, model: str}\n";

test("a composer whole record from an input resolves at run start", async () => {
  const models = composing();
  const text = gate("'${input.cfg}'", { inputs: "{proj: !str , cfg: !Cfg }", typedefs: CFG });
  const cfg = { provider: "anthropic", model: "claude-input" };
  const { result, records } = await run(text, { proj: "x", cfg }, models);
  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(selection(records[0].payload["effective_llm_by_static_path"][COMPOSER])).toEqual([
    "anthropic",
    "claude-input",
  ]);
  expect(models.configs.map(selection)).toEqual([["anthropic", "claude-input"]]);
});

test("a composer whole record from an upstream node resolves at call time", async () => {
  const models = composing();
  const pick = `  pick: {kind: code, output: Cfg, code: "return {provider: 'openai', model: 'whole'}"}\n`;
  const text = gate("!Cfg ${pick.output}", { typedefs: CFG, pick });
  const { result, records } = await run(text, { proj: "x" }, models);
  expect(result.status, result.failure?.message).toBe("succeeded");
  const steps = composerSteps(records);
  expect(steps).toHaveLength(1);
  expect(steps[0]!["effective_llm_config"]["model"]).toBe("whole");
  expect(models.configs.map(selection)).toEqual([["openai", "whole"]]);
});

test("a composer run start error names the gate", async () => {
  const models = composing();
  const text = gate("{provider: '${input.p}', model: m}", { inputs: "{proj: !str , p: !str }" });
  const error = await rejected(
    runFlow(await loadFlow(text, { env }), { proj: "x", p: "nope" }, { log: join(dir, `run${runs++}.jsonl`), env, models }),
  );
  expect(error).toBeInstanceOf(RangeError);
  const message = (error as RangeError).message;
  expect(message).toStartWith(`${SITE}.provider: not a known provider`);
  expect(message).not.toContain(COMPOSER);
  expect(models.configs).toEqual([]);
});

/** Run `text` on `inputs` with the composing models; expect it failed and no model built. */
async function failed(text: string, inputs: Record<string, unknown> = {}) {
  const models = composing();
  const result = await runFlow(await loadFlow(text, { env }), inputs, { env, models });
  expect(result.status).toBe("failed");
  expect(models.configs).toEqual([]);
  return result.failure!;
}

test("a composer call time error names the gate", async () => {
  const pick = `  pick: {kind: code, output: str, code: "return 'nope'"}\n`;
  const failure = await failed(gate("{provider: '${pick.output}', model: m}", { pick }), { proj: "x" });
  expect(failure.nodeId).toBe(COMPOSER);
  expect(failure.code).toBe("contract.input");
  expect(failure.message).toStartWith(`${SITE}.provider: not a known provider`);
});

test("a composer whole record error names the gate field", async () => {
  const typedefs = "typedefs:\n  Cfg: {provider: str, model: str, openai: object}\n";
  const pick =
    `  pick: {kind: code, output: Cfg, code: "return {provider: 'openai', ` +
    `model: 'm', openai: {temperature: 9.0}}"}\n`;
  const failure = await failed(gate("!Cfg ${pick.output}", { typedefs, pick }), { proj: "x" });
  expect(failure.code).toBe("contract.input");
  expect(failure.message).toStartWith(`${SITE}.openai.temperature: `);
});

test("a composer call time error in a defs call names the gate and callsite", async () => {
  const failure = await failed(`id: f
name: f
defs:
  child:
    input: {p: !str }
    nodes:
      ask:
        kind: human_input
        adaptive_questions:
          prompt: "Q"
          llm_config: {provider: '\${input.p}', model: m}
    output: \${ask.output}
nodes:
  pick: {kind: code, output: str, code: "return 'nope'"}
  sub: {kind: call, call: child, input: {p: '\${pick.output}'}}
output: \${sub.output}
`);
  expect(failure.code).toBe("contract.input");
  expect(failure.message).toStartWith(`${SITE}.provider in 'sub': not a known provider`);
});

test("a composer value that breaks its tag in a defs call names the field and callsite", async () => {
  const failure = await failed(`id: f
name: f
llm_config: {provider: openai}
defs:
  child:
    input: {m: !Optional(str) }
    nodes:
      ask:
        kind: human_input
        adaptive_questions:
          prompt: "Q"
          llm_config: {model: !str '\${input.m}'}
    output: \${ask.output}
nodes:
  pick: {kind: code, output: Optional(str), code: "return null"}
  sub: {kind: call, call: child, input: {m: '\${pick.output}'}}
output: \${sub.output}
`);
  expect(failure.code).toBe("binding.type");
  expect(failure.message, failure.message).toStartWith(`${SITE}.model in 'sub': `);
});
