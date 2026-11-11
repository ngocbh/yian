/** A child flow's agents may not inherit an `llm_config` value read only when a flow runs. */

import { afterEach, beforeEach, expect, test } from "bun:test";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { resolveRunLlm } from "../../src/compile/llm_cascade.ts";
import type { LoadError } from "../../src/compose/helpers/errors.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { ValueTemplate } from "../../src/nodes/contracts.ts";

import { loadErrorAsync } from "../helpers/declarations.ts";

const FIX = "pass it to the child as an input and write `${input.model}` in the child";

const DEF_CALL = `id: f
name: f
input: {m: !str }
llm_config: {model: ROOT_MODEL}
defs:
  child:
    nodes:
      a:
        kind: agent
        prompt: Go.
CHILD_LLM
    output: \${a.output}
nodes:
  pick:
    kind: agent
    prompt: Pick a model.
    output: !str
    llm_config: {model: m0}
  c:
    kind: call
    call: child
    input: {}
    depends_on: [pick]
output: \${c.output}
`;

/** Return `DEF_CALL` with the root model and the child agent's `llm_config` line filled in. */
function defCall(options: { rootModel?: string; childLlm?: string } = {}): string {
  const { rootModel = "'${pick.output}'", childLlm = "" } = options;
  return DEF_CALL.replace("ROOT_MODEL", rootModel).replace("CHILD_LLM\n", childLlm);
}

/** Return the `LoadError` loading `text` as `f.yaml` rejects with. */
function error(text: string, options: { searchPaths?: string[]; sourceLabel?: string } = {}): Promise<LoadError> {
  return loadErrorAsync(loadFlow(text, { sourceLabel: "f.yaml", ...options }));
}

/** Return the start line of the error's innermost frame. */
function line(loadError: LoadError): number {
  return loadError.diagnostic.frames.at(-1)!.range.startLine;
}

/** Return the 1-based number of the first line of `text` holding `needle`. */
function lineOf(text: string, needle: string): number {
  return text.split("\n").findIndex((textLine) => textLine.includes(needle)) + 1;
}

test("a run-time flow value a child agent would inherit fails at the flow field", async () => {
  const text = defCall();
  const caught = await error(text);
  const message = caught.diagnostic.message;
  expect(message).toStartWith("flow llm_config.model: ");
  expect(message).toContain("agent 'c/a'");
  expect(message).toContain(FIX);
  expect(line(caught)).toBe(lineOf(text, "llm_config: {model: '${pick.output}'}"));
});

const COMPOSER_IN_CHILD = `id: f
name: f
llm_config: {model: '\${pick.output}'}
nodes:
  pick:
    kind: agent
    prompt: Pick.
    output: !str
    llm_config: {model: m0}
  c:
    kind: call
    input: {}
    nodes:
      ask:
        kind: tool
        tool_id: ask
        input:
          prompt: hi
        adaptive_questions:
          prompt: Q
    output: \${ask.output}
    depends_on: [pick]
output: \${c.output}
`;

test("a composer in a child flow is named by its gate", async () => {
  const message = (await error(COMPOSER_IN_CHILD)).diagnostic.message;
  expect(message).toStartWith("flow llm_config.model: node 'ask' adaptive_questions.llm_config in 'c' would inherit ");
  expect(message).not.toContain("__compose");
  expect(message).toContain(FIX);
});

/** Return the message of the `RangeError` `run` throws, failing the test on any other outcome. */
function rangeError(run: () => unknown): string {
  try {
    run();
  } catch (caught) {
    expect(caught).toBeInstanceOf(RangeError);
    return (caught as RangeError).message;
  }
  throw new Error("expected a RangeError");
}

const CASCADE_DEFAULTS = { provider: "openai", model: "gpt-x" };

test("the cascade check names a composer by its gate", async () => {
  const flow = (await loadFlow(COMPOSER_IN_CHILD.replace("'${pick.output}'", "m1"))).compiled;
  flow.flowLlmConfig = new ValueTemplate({ model: "${pick.output}" });
  const message = rangeError(() => resolveRunLlm(flow, {}, { defaults: CASCADE_DEFAULTS }));
  expect(message).toStartWith("node 'ask' adaptive_questions.llm_config in 'c' would inherit llm_config.model ");
  expect(message).not.toContain("__compose");
});

test("a child agent that sets the field loads", async () => {
  await loadFlow(defCall({ childLlm: "        llm_config: {model: m1}\n" }));
});

test("a child agent that does not inherit loads", async () => {
  await loadFlow(defCall({ childLlm: "        llm_config: {inherit: false, model: m1}\n" }));
});

test("a root input value above a child loads", async () => {
  await loadFlow(defCall({ rootModel: "'${input.m}'" }));
});

test("a def input value above its own nested call fails", async () => {
  const text = `id: f
name: f
defs:
  inner:
    nodes:
      a: {kind: agent, prompt: Go.}
    output: \${a.output}
  outer:
    input: {m: !str }
    llm_config: {model: '\${input.m}'}
    nodes:
      i: {kind: call, call: inner, input: {}}
    output: \${i.output}
nodes:
  o:
    kind: call
    call: outer
    input: {m: x}
output: \${o.output}
`;
  const caught = await error(text);
  const message = caught.diagnostic.message;
  expect(message).toStartWith("defs entry 'outer' llm_config.model (called by node 'o'): ");
  expect(message).toContain("agent 'o/i/a'");
  expect(message).toContain(FIX);
  expect(line(caught)).toBe(lineOf(text, "${input.m}"));
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

const IMPORTER = `id: parent
name: parent
uses:
  mylib: lib
nodes:
  n:
    kind: call
    call: mylib
    input: {m: x}
output: \${n.output}
`;

let tmp: string;

beforeEach(() => {
  tmp = mkdtempSync(join(tmpdir(), "yian-llm-inherit-"));
});

afterEach(() => {
  rmSync(tmp, { recursive: true, force: true });
});

test("an imported flow's input value above its nested call fails at the import", async () => {
  writeFileSync(join(tmp, "lib.yaml"), LIB);
  const caught = await error(IMPORTER, { searchPaths: [tmp] });
  const message = caught.diagnostic.message;
  expect(message).toStartWith("llm_config.model of the flow imported through uses: 'mylib' (called by node 'n'): ");
  expect(message).toContain("agent 'n/c/a'");
  expect(message).toContain(FIX);
  const frame = caught.diagnostic.frames.at(-1)!;
  expect(frame.label).toBe("lib.yaml");
  expect(frame.range.startLine).toBe(lineOf(LIB, "${input.m}"));
  // Loaded alone, the file is the run's root, so its input is known at run start.
  await loadFlow(LIB, { sourceLabel: "lib.yaml" });
});

test("an imported file whose label matches the importer's is still named an import", async () => {
  writeFileSync(join(tmp, "lib.yaml"), LIB);
  const caught = await error(IMPORTER, { searchPaths: [tmp], sourceLabel: "lib.yaml" });
  expect(caught.diagnostic.message).toStartWith(
    "llm_config.model of the flow imported through uses: 'mylib' (called by node 'n'): ",
  );
  expect(caught.diagnostic.frames.at(-1)!.range.startLine).toBe(lineOf(LIB, "${input.m}"));
});

test("an inline call body inherits like a named child", async () => {
  const text = `id: f
name: f
llm_config: {model: '\${pick.output}'}
nodes:
  pick:
    kind: agent
    prompt: Pick a model.
    output: !str
    llm_config: {model: m0}
  c:
    kind: call
    input: {}
    nodes:
      a: {kind: agent, prompt: Go., output: !str}
    output: \${a.output}
    depends_on: [pick]
output: \${c.output}
`;
  const message = (await error(text)).diagnostic.message;
  expect(message).toStartWith("flow llm_config.model: ");
  expect(message).toContain("agent 'c/a'");
});

test("an inline loop body inherits like a named child", async () => {
  const text = `id: f
name: f
llm_config: {model: '\${pick.output}'}
nodes:
  pick:
    kind: agent
    prompt: Pick a model.
    output: !str
    llm_config: {model: m0}
  repeat:
    kind: loop
    times: 2
    input: {t: start}
    nodes:
      a:
        kind: agent
        input: {t: '\${input.t}'}
        prompt: Go \${t}.
        output: !str
    output: {t: '\${a.output}'}
    depends_on: [pick]
output: \${repeat.output}
`;
  const message = (await error(text)).diagnostic.message;
  expect(message).toStartWith("flow llm_config.model: ");
  expect(message).toContain("agent 'repeat/a'");
});

test("a flow built around compose hits the cascade check", async () => {
  const flow = (await loadFlow(defCall({ rootModel: "m0" }))).compiled;
  flow.flowLlmConfig = new ValueTemplate({ model: "${pick.output}" });
  const message = rangeError(() => resolveRunLlm(flow, {}, { defaults: CASCADE_DEFAULTS }));
  expect(message).toMatch(/node 'c\/a' would inherit llm_config\.model/);
  expect(message).toContain(FIX);
});

test("a root input value or a literal passes the cascade check", async () => {
  const flow = (await loadFlow(defCall({ rootModel: "m0" }))).compiled;
  expect(resolveRunLlm(flow, {}, { defaults: CASCADE_DEFAULTS }).effective["c/a"]!["model"]).toBe("m0");
  flow.flowLlmConfig = new ValueTemplate({ model: "${input.m}" });
  const run = resolveRunLlm(flow, {}, { defaults: CASCADE_DEFAULTS, runInputs: { m: "gpt-y" } });
  expect(run.effective["c/a"]!["model"]).toBe("gpt-y");
});

const TYPED_CHILD = `id: f
name: f
llm_config: {model: '\${pick.output}'}
defs:
  child:
    input: {model: MODEL_TAG }
    llm_config: {model: '\${input.model}'}
    nodes:
      a: {kind: agent, prompt: Go.}
    output: \${a.output}
nodes:
  pick:
    kind: agent
    prompt: Pick a model.
    output: !str
    llm_config: {model: m0}
  c:
    kind: call
    call: child
    input: {model: x}
    depends_on: [pick]
output: \${c.output}
`;

test("a child flow value that cannot be null shields the outer value", async () => {
  await loadFlow(TYPED_CHILD.replace("MODEL_TAG", "!str"));
});

test("a child flow value that may be null leaves the outer value reaching", async () => {
  const message = (await error(TYPED_CHILD.replace("MODEL_TAG", "!Optional(str)"))).diagnostic.message;
  expect(message).toStartWith("flow llm_config.model: ");
  expect(message).toContain("agent 'c/a'");
});

/**
 * Return a flow whose defs `d1` .. `d<depth - 1>` each call the next twice, under a root that calls
 * `d1` twice, so the last def is reached along 2^depth static paths; `leaf` is the last def's node.
 */
function diamond(depth: number, leaf: string, rootLlm = ""): string {
  const calls = (callee: string, pad: string) =>
    `${pad}nodes:\n${pad}  k0: {kind: call, call: ${callee}, input: {}}\n${pad}  k1: {kind: call, call: ${callee}, input: {}}\n` +
    `${pad}output: \${k0.output}\n`;
  let text = `id: f\nname: f\n${rootLlm}defs:\n`;
  for (let i = 1; i < depth; i++) {
    text += `  d${i}:\n` + (i === depth - 1 ? `    nodes:\n      a: ${leaf}\n    output: \${a.output}\n` : calls(`d${i + 1}`, "    "));
  }
  return text + calls("d1", "");
}

test("a deep diamond of defs walks each def once", async () => {
  // Walking every static path would take 2^24 steps.
  const start = performance.now();
  await loadFlow(diamond(24, "{kind: agent, prompt: Go., output: !str}"), { sourceLabel: "f.yaml" });
  expect(performance.now() - start).toBeLessThan(2000);
});

test("a deep diamond reports the first path that inherits a run-time value", async () => {
  const rootLlm = "llm_config: {model: '${pick.output}'}\n";
  const text = diamond(24, "{kind: agent, prompt: Go., output: !str}", rootLlm).replace(
    "nodes:\n  k0:",
    "nodes:\n  pick: {kind: agent, prompt: Pick., output: !str, llm_config: {model: m0}}\n  k0:",
  );
  const start = performance.now();
  const message = (await error(text)).diagnostic.message;
  expect(performance.now() - start).toBeLessThan(2000);
  expect(message).toContain(`agent '${["k0", ...Array(22).fill("k0"), "a"].join("/")}'`);
});
