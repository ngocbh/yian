/**
 * The language safety limits follow `YIAN_<NAME>`, and each maximum loads and runs: a port of
 * `tests/engine/test_limit_overrides.py`.
 *
 * A test sets a limit in the `env` it passes to `loadFlow` and `runFlow` (decision 46), where
 * Python's `set_defaults` fixture sets it for the process; a check outside a load or run (a type
 * parse, a skip model) runs inside `withEnv` with that `env`. The chain helpers are
 * `test_call_nesting_limit.py`'s (as `call_nesting_limit.test.ts` writes them), the many-CASE skip
 * model `test_skip_model.py`'s, and the memory helpers `test_value_nesting_boundaries.py`'s; the
 * CODE references are TS's (`./fixtures/compose_codefns.ts:echo`), and the inline CODE bodies
 * read the same in JS but for `_list_body`'s loop (decision 11). An AGENT's model is a
 * `ScriptedModel` (on Azure, a `chatModelFromConfig` model) given as `models`, where Python
 * patches `model_from_config`. Ints read as
 * `bigint`s (decision 4a). Every run sees only the test's credential store in its environment.
 *
 * Ported: every row but those below. Changed: the "with less stack" rows and
 * `test_agent_output_nesting_maximum_loads_under_the_deepest_import_chain` load and run at each
 * entry's maximum on the stack as it is: JS has no recursion limit to lower (Python's
 * `_with_less_stack` and `_stack_headroom`), and the maxima are measured for Node's and a Bun
 * Worker's stacks (`docs/defaults.md`). A refused load is a `LoadError` and a refused type a
 * `TypeCheckError`, as in Python. In
 * `test_agent_output_nesting_maximum_runs_with_less_stack_on_azure`, a fake `send` streams the
 * reply where Python uses `httpx.MockTransport`; every model call streams (decision 31).
 *
 * Already ported: `test_value_nesting_follows_the_variable`, in `value_nesting.test.ts`'s `value
 * nesting follows the variable`.
 *
 * Covered by `run-agent/engine_limit_overrides__a_run_input_over_the_set_value_limit_is_refused`:
 * `test_a_run_input_over_the_set_value_limit_is_refused`.
 */

import { afterAll, expect, test } from "bun:test";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { Edge } from "../../src/compile/model.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { SkipModel } from "../../src/compose/validate/skips.ts";
import { ENTRIES, withEnv } from "../../src/defaults.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { chatModelFromConfig } from "../../src/llm_clients/factory.ts";
import { buildMemories } from "../../src/memory_host.ts";
import type { Send } from "../../src/platform.ts";
import { resumeFlow, runFlow } from "../../src/run.ts";
import { TypeCheckError } from "../../src/typesys/index.ts";
import { parseType } from "../../src/typesys/types.ts";
import { linkConformanceFixtures, loadConformanceFlow, loadErrorAsync } from "../compose/declarations.ts";
import { hostEnv } from "./mcp_helpers.ts";
import { FakeMemory } from "./memory_fakes.ts";
import { serving, textReply } from "./run_state_fixtures.ts";

const dir = mkdtempSync(join(tmpdir(), "yian-limit-overrides-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));

/** The test's environment with `YIAN_<name>` set to each of `limits`. */
function limitEnv(limits: Record<string, number> = {}) {
  return hostEnv(dir, Object.fromEntries(Object.entries(limits).map(([name, value]) => [`YIAN_${name}`, String(value)])));
}

/** Return `levels` nested `upper(...)` calls around `ref`. */
const upper = (levels: number, ref: string) => "upper(".repeat(levels) + ref + ")".repeat(levels);

/** Return the largest value the entry `name` accepts from the environment. */
const maximum = (name: string) => ENTRIES.find((entry) => entry.name === name)!.maximum!;

/** Return the error `fn` throws; fails the test when it returns. */
function thrown(fn: () => unknown): Error {
  try {
    fn();
  } catch (error) {
    return error as Error;
  }
  throw new Error("expected a throw");
}

/** Return a flow whose one CODE node binds and returns `topic`. */
const bindingFlow = (binding: string) =>
  "id: f\nname: f\ninput:\n  s: !str\nnodes:\n  use:\n    kind: code\n    input:\n" +
  "      topic: ${" +
  binding +
  "}\n    output: str\n" +
  "    code: |\n      return inputs['topic']\noutput: ${use.output}\n";

/** Return a flow whose CODE node's `topic` input is written out as `topic`. */
const yamlFlow = (topic: string) =>
  "id: f\nname: f\nnodes:\n  use:\n    kind: code\n" +
  "    code: ./fixtures/compose_codefns.ts:echo\n" +
  "    input:\n      topic: " +
  topic +
  "\n    output: str\noutput: ${use.output}\n";

// ---- the chain helpers of `test_call_nesting_limit.py` ----

const LEAF =
  "    input:\n" +
  "      x: !str\n" +
  "    nodes:\n" +
  "      step:\n" +
  "        kind: code\n" +
  "        code: ./fixtures/compose_codefns.ts:echo\n" +
  "        input:\n" +
  "          topic: ${input.x}\n" +
  "        output: str\n" +
  "    output: ${step.output}\n";

/** Return a def body whose one `step` node calls `target` with `x`. */
const calling = (target: string) =>
  "    input:\n      x: !str\n    nodes:\n      step:\n        kind: call\n" +
  `        call: ${target}\n` +
  "        input:\n          x: ${input.x}\n    output: ${step.output}\n";

/**
 * Return a flow whose root `top` calls `c0` and each `ci` calls `c(i+1)`. The last def is a CODE
 * leaf echoing `x`, or calls `lastCalls` when given.
 */
function defsChain(n: number, lastCalls: string | null = null): string {
  const entries: string[] = [];
  for (let i = 0; i < n; i++) {
    const body = i < n - 1 ? calling(`c${i + 1}`) : lastCalls !== null ? calling(lastCalls) : LEAF;
    entries.push(`  c${i}:\n${body}`);
  }
  return (
    `id: chain${n}\nname: chain${n}\ninput:\n  x: !str\n` +
    "defs:\n" +
    entries.join("") +
    "nodes:\n  top:\n    kind: call\n    call: c0\n" +
    "    input:\n      x: ${input.x}\noutput: ${top.output}\n"
  );
}

const ROOT_LEAF =
  "  step:\n" +
  "    kind: code\n" +
  "    code: ./fixtures/compose_codefns.ts:echo\n" +
  "    input:\n" +
  "      topic: ${input.x}\n" +
  "    output: str\n" +
  "output: ${step.output}\n";

/** Return a flow `name` that imports `uses` without calling them; its node is a CODE leaf. */
function importing(name: string, uses: Record<string, string>): string {
  let head = `id: ${name}\nname: ${name}\ninput:\n  x: !str\n`;
  const entries = Object.entries(uses);
  if (entries.length > 0) head += "uses:\n" + entries.map(([alias, ref]) => `  ${alias}: ${ref}\n`).join("");
  return head + "nodes:\n" + ROOT_LEAF;
}

/** Return flows `u0`..`u(n-1)` by name, each a CODE leaf importing the next as `nxt`. */
function chainTexts(n: number): Record<string, string> {
  const texts: Record<string, string> = {};
  for (let i = 0; i < n; i++) texts[`u${i}`] = importing(`u${i}`, i < n - 1 ? { nxt: `u${i + 1}` } : {});
  return texts;
}

test("expression nesting follows the value limit", async () => {
  const env = limitEnv({ MAX_VALUE_NESTING: 3 });

  await loadFlow(bindingFlow(upper(3, "input.s")), { env });
  const error = await loadErrorAsync(loadFlow(bindingFlow(upper(4, "input.s")), { env }));
  expect(error.message).toContain("expression nests deeper than 3 levels");
});

test("a written out value over the set value limit fails at load", async () => {
  const env = limitEnv({ MAX_VALUE_NESTING: 3 });

  await loadConformanceFlow(yamlFlow("[[[1]]]"), { env });
  const error = await loadErrorAsync(loadConformanceFlow(yamlFlow("[[[[1]]]]"), { env }));
  expect(error.message).toContain("value nests deeper than 3 levels");
});

test("type nesting follows the variable", () => {
  withEnv(limitEnv({ MAX_TYPE_NESTING: 2 }), () => {
    parseType("list(list(int))");
    const error = thrown(() => parseType("list(list(list(int)))"));
    expect(error).toBeInstanceOf(TypeCheckError);
    expect(error.message).toContain("at most 2 levels");
  });
});

test("yaml nesting follows the variable and names the value limit", async () => {
  const env = limitEnv({ MAX_YAML_NESTING: 5, MAX_VALUE_NESTING: 4 });

  // The root, nodes, use, and input mappings are levels 1-4.
  await loadConformanceFlow(yamlFlow("[1]"), { env });
  const error = await loadErrorAsync(loadConformanceFlow(yamlFlow("[[1]]"), { env }));

  expect(error.message).toContain("this list nests deeper than 5 levels");
  expect(error.message).toContain("a yian value nests at most 4)");
  const range = error.diagnostic.frames.at(-1)!.range;
  expect([range.startLine, range.startCol]).toEqual([8, 15]);
});

test("call nesting follows the variable", async () => {
  const env = limitEnv({ MAX_CALL_NESTING: 2 });

  await loadConformanceFlow(defsChain(2), { env });
  const error = await loadErrorAsync(loadConformanceFlow(defsChain(3), { env }));

  expect(error.message).toContain("nests calls 3 levels deep, past the call nesting limit of 2");
});

// ---- the many-CASE skip model of `test_skip_model.py` ----

let ids = 0;

/** A CASE control edge `case -[handle]-> target`. */
const ctl = (caseId: string, handle: string, target: string) =>
  new Edge(`e${ids++}`, caseId, target, { sourceHandle: handle });

/** A data edge feeding input group `group` of `dst`. */
const data = (src: string, dst: string, group: string) => new Edge(`e${ids++}`, src, dst, { inputGroup: group });

/**
 * `c` (exhaustive) routes `a`/`b`; `x` is `g`'s only route; ten more CASEs feed both. Each `t{i}`
 * shares a required group of `a` and `x` with the root `r`, so the ten CASEs are branched on but
 * can never skip `a` or `x`.
 */
function manyCases(): SkipModel {
  const edges = [ctl("c", "then", "a"), ctl("c", "else", "b"), ctl("g", "then", "x")];
  const nodes = ["r", "g", "c", "a", "b", "x"];
  for (let i = 0; i < 10; i++) {
    edges.push(ctl(`k${i}`, "then", `t${i}`));
    for (const n of ["a", "x"]) for (const p of [`t${i}`, "r"]) edges.push(data(p, n, `in${i}`));
    nodes.push(`k${i}`, `t${i}`);
  }
  return new SkipModel(nodes, edges, new Set(["c"]));
}

test("skip worlds follow the variable", () => {
  expect(withEnv(limitEnv(), () => manyCases().mayAllSkip(["a", "b"]))).toBe(false);

  expect(withEnv(limitEnv({ MAX_SKIP_WORLDS: 1 }), () => manyCases().mayAllSkip(["a", "b"]))).toBe(true);
});

// ---- the maxima ----

test("value nesting maximum learns with less stack", async () => {
  const levels = maximum("MAX_VALUE_NESTING");
  const env = limitEnv({ MAX_VALUE_NESTING: levels });
  // `test_value_nesting_boundaries.py`'s `_memory_flow(_code_flow(_list_body(levels)))`.
  const text = `id: nesting
name: nesting
long_term_memory: episodic
nodes:
  deep:
    kind: code
    code: |
      let v = "leaf";
      for (let i = 0; i < ${levels}; i++) v = [v];
      return v;
output: \${deep.output}
`;
  const flow = await loadFlow(text, { env });
  await using memory = await buildMemories(flow);

  const result = await runFlow(flow, {}, { env, memory, learnable: true });
  expect(result.status, result.failure?.message).toBe("succeeded");
  const outcome = await result.backward("good");

  expect(outcome.status).toBe("succeeded");
  expect(outcome.learning!.status).toBe("succeeded");
});

for (const numWorkers of [0, 2]) {
  test(`expression nesting maximum runs with less stack: ${numWorkers}`, async () => {
    const levels = maximum("MAX_VALUE_NESTING");
    const env = limitEnv({ MAX_VALUE_NESTING: levels });

    const flow = await loadFlow(bindingFlow(upper(levels, "input.s")), { env });
    const result = await runFlow(flow, { s: "a" }, { env, numWorkers });

    expect(result.status, result.failure?.message).toBe("succeeded");
    expect(result.output).toBe("A");
  });
}

test("condition at the expression maximum runs thirty calls deep", async () => {
  const levels = maximum("MAX_VALUE_NESTING");
  const env = limitEnv({ MAX_VALUE_NESTING: levels });
  const condition = upper(levels - 1, "input.s") + " == 'A'"; // the comparison is a level
  const lines = ["id: f", "name: f", "input:", "  s: !str", "defs:"];
  for (let i = 0; i < 30; i++) {
    lines.push(`  c${i}:`, "    input:", "      s: !str", "    nodes:");
    if (i < 29) {
      lines.push(
        "      sub:", "        kind: call", `        call: c${i + 1}`,
        "        input:", "          s: ${input.s}", "    output: ${sub.output}",
      );
    } else {
      lines.push(
        "      route:", "        kind: case", "        cases:",
        `          - when: "${condition}"`, "            then: yes",
        "        else: no",
        "      yes:", "        kind: code", "        output: str", "        code: |",
        "          return 'yes'",
        "      no:", "        kind: code", "        output: str", "        code: |",
        "          return 'no'",
        "    output: ${yes.output | no.output}",
      );
    }
  }
  lines.push("nodes:", "  top:", "    kind: call", "    call: c0", "    input:", "      s: ${input.s}", "output: ${top.output}");

  const flow = await loadFlow(lines.join("\n") + "\n", { env });

  expect((await runFlow(flow, { s: "a" }, { env })).output).toBe("yes");
  expect((await runFlow(flow, { s: "b" }, { env })).output).toBe("no");
});

test("learnable run at the expression maximum logs and resumes", async () => {
  const levels = maximum("MAX_VALUE_NESTING");
  const env = limitEnv({ MAX_VALUE_NESTING: levels });
  const text =
    "id: deep-call\nname: deep_call\nlong_term_memory: test\ninput:\n  s: !str\n" +
    "defs:\n  echo:\n    input:\n      s: !str\n    nodes: {}\n    output: ${input.s}\n" +
    "nodes:\n  outer:\n    kind: call\n    call: echo\n    input:\n" +
    "      s: ${" +
    upper(levels, "input.s") +
    "}\n" +
    "output: ${outer.output}\n";
  const flow = await loadFlow(text, { env });
  const backend = new FakeMemory();
  const path = join(dir, "deep.jsonl");

  const result = await runFlow(flow, { s: "a" }, { env, log: path, learnable: true, memory: new Map([["test", backend]]) });
  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(result.output).toBe("A");
  const replayed = await resumeFlow(flow, { env, log: path, memory: new Map([["test", backend]]) });

  expect(replayed.status, replayed.failure?.message).toBe("succeeded");
  expect(replayed.output).toBe("A");
});

for (const numWorkers of [0, 2]) {
  test(`call nesting maximum runs with less stack: ${numWorkers}`, async () => {
    const levels = maximum("MAX_CALL_NESTING");
    const env = limitEnv({ MAX_CALL_NESTING: levels });

    const flow = await loadConformanceFlow(defsChain(levels), { env });
    const result = await runFlow(flow, { x: "hi" }, { env, numWorkers });

    expect(result.status, result.failure?.message).toBe("succeeded");
    expect(result.output).toBe("hi");
  });
}

test("type nesting maximum parses with less stack", () => {
  const levels = maximum("MAX_TYPE_NESTING");
  const text = "list(".repeat(levels) + "int" + ")".repeat(levels);

  withEnv(limitEnv({ MAX_TYPE_NESTING: levels }), () => {
    parseType(text);
    const error = thrown(() => parseType("list(" + text + ")"));
    expect(error).toBeInstanceOf(TypeCheckError);
    expect(error.message).toContain(`at most ${levels} levels`);
  });
});

/**
 * Return `[typedefs, reply]` for `U<levels>`, an `Optional(Union(...))` at each level: the
 * costliest shape per level for structured-output schema generation.
 */
function unionChain(levels: number): [string, unknown] {
  const lines = ["typedefs:", "  A1: {tag: Literal(a), k: int}", "  B1: {tag: Literal(b), n: int}", "  U1: Union(A1, B1)"];
  let reply: unknown = 1n;
  for (let i = 2; i <= levels; i++) {
    lines.push(`  A${i}: {tag: Literal(a), k: 'Optional(U${i - 1})'}`, `  B${i}: {tag: Literal(b), n: int}`, `  U${i}: Union(A${i}, B${i})`);
  }
  for (let i = 0; i < levels; i++) reply = { tag: "a", k: reply };
  return [lines.join("\n") + "\n", reply];
}

/**
 * Return `[flow, reply]`: an AGENT with the worst-shape output 50 calls deep. `llmConfig` holds
 * the AGENT's `llm_config:` entries, one `key: value` per line.
 */
function deepAgentFlow(levels: number, llmConfig: string): [string, unknown] {
  const [typedefs, reply] = unionChain(levels);
  const config = llmConfig.split("\n").map((line) => `          ${line}\n`).join("");
  const leaf =
    "  leaf:\n    input:\n      x: !str\n    nodes:\n      writer:\n" +
    "        kind: agent\n        prompt: hi\n        retries: 0\n" +
    `        llm_config:\n${config}        output: U${levels}\n` +
    "    output: ${writer.output}\n";
  return [defsChain(49, "leaf").replace("defs:\n", typedefs + "defs:\n" + leaf), reply];
}

/** JSON text of `value`, writing a `bigint` as its digits. */
const jsonText = (value: unknown) => JSON.stringify(value, (_key, each) => (typeof each === "bigint" ? Number(each) : each));

for (const numWorkers of [0, 2]) {
  test(`agent output nesting maximum runs with less stack prompted: ${numWorkers}`, async () => {
    const levels = maximum("MAX_AGENT_OUTPUT_NESTING");
    const env = limitEnv({ MAX_AGENT_OUTPUT_NESTING: levels });
    // This model has no native structured output, so the reply is prompted for.
    const [text, reply] = deepAgentFlow(levels, "provider: openai\nmodel: no-structured-sentinel");
    // A union root is wrapped in a `value` field.
    const model = new ScriptedModel([textReply(jsonText({ value: reply }))], {
      provider: "openai",
      model: "no-structured-sentinel",
    });

    const flow = await loadConformanceFlow(text, { env });
    const result = await runFlow(flow, { x: "hi" }, { env, numWorkers, models: serving(model, env) });

    expect(result.status, result.failure?.message).toBe("succeeded");
    expect(result.output).toEqual(reply);
    expect(model.requests).toHaveLength(1);
    expect(model.requests[0]!.outputFormat).toBeNull();
  });
}

/** One Chat Completions stream `data:` event per chunk, then `[DONE]`. */
const chatStream = (chunks: Record<string, unknown>[]) =>
  chunks.map((data) => `data: ${JSON.stringify(data)}\n\n`).join("") + "data: [DONE]\n\n";

/** A Chat Completions stream chunk carrying `delta` in its one choice. */
const chatChunk = (delta: Record<string, unknown>, finishReason: string | null = null) => ({
  id: "c",
  object: "chat.completion.chunk",
  created: 0,
  model: "m",
  choices: [{ index: 0, delta, finish_reason: finishReason }],
});

/**
 * A `send` that records each Chat Completions request body and answers with a call to its first
 * tool with `args`: Python's `_ToolCallServer`.
 */
function toolCallServer(args: unknown): { bodies: Record<string, unknown>[]; send: Send } {
  const bodies: Record<string, unknown>[] = [];
  const send: Send = async (_input, init) => {
    const body = JSON.parse(String(init.body)) as Record<string, unknown>;
    bodies.push(body);
    const tools = body["tools"] as { function: { name: string } }[];
    const call = { index: 0, id: "call_1", type: "function", function: { name: tools[0]!.function.name, arguments: jsonText(args) } };
    const text = chatStream([chatChunk({ role: "assistant", tool_calls: [call] }), chatChunk({}, "tool_calls")]);
    return new Response(text, { status: 200, headers: { "content-type": "text/event-stream" } });
  };
  return { bodies, send };
}

for (const numWorkers of [0, 2]) {
  test(`agent output nesting maximum runs with less stack on azure: ${numWorkers}`, async () => {
    const levels = maximum("MAX_AGENT_OUTPUT_NESTING");
    const env = hostEnv(dir, {
      YIAN_MAX_AGENT_OUTPUT_NESTING: String(levels),
      AZURE_OPENAI_API_KEY: "k",
      OPENAI_API_VERSION: "2025-03-01-preview",
    });
    // A `gpt-4` deployment sends structured output as a forced tool call, after the open-schema
    // check walks the schema; a `gpt-4o` one would send a strict `json_schema` instead.
    const [text, reply] = deepAgentFlow(levels, "provider: azure\nmodel: gpt-4\nbase_url: https://x.openai.azure.com/");
    const server = toolCallServer({ value: reply });

    const flow = await loadConformanceFlow(text, { env });
    const result = await runFlow(flow, { x: "hi" }, {
      env,
      numWorkers,
      models: { env, createModel: (config) => chatModelFromConfig(config, { env, send: server.send }) },
    });

    expect(result.status, result.failure?.message).toBe("succeeded");
    expect(result.output).toEqual(reply);
    expect(server.bodies).toHaveLength(1);
    expect(server.bodies[0]).toHaveProperty("tools");
  });
}

test("agent output nesting maximum loads under the deepest import chain", async () => {
  // The worst accepted load: 50 imports whose last file calls a 50-level chain, here ending in
  // an AGENT whose output schema is built at load.
  const levels = maximum("MAX_AGENT_OUTPUT_NESTING");
  const texts = chainTexts(50);
  texts["u49"] = deepAgentFlow(levels, "provider: openai\nmodel: gpt-5.5")[0];
  const files = mkdtempSync(join(dir, "chain-"));
  linkConformanceFixtures(files);
  for (const [name, text] of Object.entries(texts)) writeFileSync(join(files, `${name}.yaml`), text);

  await loadFlow(importing("root", { first: "u0" }), { searchPaths: [files], env: limitEnv() });
});
