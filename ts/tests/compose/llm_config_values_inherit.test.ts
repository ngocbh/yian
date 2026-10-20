/**
 * A child flow's agents may not inherit an `llm_config` value read only when a flow runs: a port
 * of `tests/compose/test_llm_config_values_inherit.py`.
 *
 * Ported: every row that only loads.
 *
 * Later (T3-22): `test_the_cascade_check_names_a_composer_by_its_gate`,
 * `test_a_flow_built_around_compose_hits_the_cascade_check` and
 * `test_a_root_input_value_or_a_literal_passes_the_cascade_check` (they call
 * `resolve_llm_cascade` and `resolve_run_llm`).
 */

import { afterEach, beforeEach, expect, test } from "bun:test";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import type { LoadError } from "../../src/compose/helpers/errors.ts";
import { loadFlow } from "../../src/compose/loader.ts";

import { loadErrorAsync } from "./declarations.ts";

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
    output: str
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
function error(text: string, options: { searchPaths?: string[] } = {}): Promise<LoadError> {
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
    output: str
    llm_config: {model: m0}
  c:
    kind: call
    input: {}
    nodes:
      ask:
        kind: human_input
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

test("an inline call body inherits like a named child", async () => {
  const text = `id: f
name: f
llm_config: {model: '\${pick.output}'}
nodes:
  pick:
    kind: agent
    prompt: Pick a model.
    output: str
    llm_config: {model: m0}
  c:
    kind: call
    input: {}
    nodes:
      a: {kind: agent, prompt: Go., output: str}
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
    output: str
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
        output: str
    output: {t: '\${a.output}'}
    depends_on: [pick]
output: \${repeat.output}
`;
  const message = (await error(text)).diagnostic.message;
  expect(message).toStartWith("flow llm_config.model: ");
  expect(message).toContain("agent 'repeat/a'");
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
    output: str
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
