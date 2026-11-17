/**
 * A name `item` given to a node's input, an agent's variable or a loop's carried field reads by
 * bare name like any other local name, as `input` and `system` do: only a MAP's `input:` bindings
 * read the MAP item as `${item}`.
 */

import { expect, test } from "bun:test";

import type { Message, Request, TextPart } from "../../src/llm/messages.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { type RunResult, runFlow } from "../../src/run.ts";
import { loadConformanceFlow } from "../helpers/declarations.ts";
import { NO_ENV, serving, textReply } from "../helpers/run_state_fixtures.ts";

const CONFIG = { provider: "anthropic", model: "claude-test" };

/** Run `source` on `inputs` with a scripted model that answers each call with `ok`. */
async function run(source: string, inputs: Record<string, unknown> = {}) {
  const model = new ScriptedModel(Array.from({ length: 4 }, () => textReply("ok")), CONFIG);
  const result = await runFlow(await loadConformanceFlow(source), inputs, {
    llmConfig: CONFIG,
    models: serving(model),
    env: NO_ENV,
  });
  return { result, requests: model.requests };
}

/** Assert that `result` succeeded, showing its failure's message when it didn't. */
function expectSucceeded(result: RunResult): void {
  expect(result.failure?.message ?? null).toBeNull();
  expect(result.status).toBe("succeeded");
}

/** The role and text of each message `request` sends. */
function sent(request: Request): [string, string][] {
  return request.messages.map((message: Message) => [
    message.role,
    message.parts.map((part) => (part as TextPart).text).join(""),
  ]);
}

test("a def's agent reads its input named item in its prompt", async () => {
  const { result, requests } = await run(
    `
id: t
name: t
input:
  x: !str
defs:
  one:
    input:
      item: !str
    nodes:
      a:
        kind: agent
        input:
          item: \${input.item}
        prompt: Say \${item}.
        output: !str
    output: \${a.output}
nodes:
  c:
    kind: call
    call: one
    input:
      item: \${input.x}
output: \${c.output}
`,
    { x: "hi" },
  );
  expectSucceeded(result);
  expect(sent(requests[0]!).at(-1)).toEqual(["user", "Say hi."]);
});

test("an agent reads an input named item in its system prompt, a list prompt and a field read", async () => {
  const { result, requests } = await run(
    `
id: t
name: t
typedefs:
  Person:
    name: str
input:
  x: !Person
nodes:
  a:
    kind: agent
    input:
      item: \${input.x}
    system_prompt: Speak of \${item.name}.
    prompt:
      - user: Say \${item.name}.
    output: !str
output: \${a.output}
`,
    { x: { name: "Ada" } },
  );
  expectSucceeded(result);
  expect(sent(requests[0]!)).toEqual([
    ["system", "Speak of Ada."],
    ["user", "Say Ada."],
  ]);
});

test("a list prompt splices an input named item", async () => {
  const { result, requests } = await run(
    `
id: t
name: t
typedefs:
  Turn:
    role: Literal(user, assistant)
    text: str
input:
  history: !list(Turn)
nodes:
  a:
    kind: agent
    input:
      item: \${input.history}
    prompt:
      - \${item}
      - user: go on
    output: !str
output: \${a.output}
`,
    { history: [{ role: "user", text: "hi" }] },
  );
  expectSucceeded(result);
  expect(sent(requests[0]!).slice(-2)).toEqual([
    ["user", "hi"],
    ["user", "go on"],
  ]);
});

test("an agent's variable named item and a single-agent flow's input named item read in the prompt", async () => {
  const variable = await run(`
id: t
name: t
nodes:
  a:
    kind: agent
    variables:
      item: !str hi
    prompt: Say \${item}.
    output: !str
output: \${a.output}
`);
  expectSucceeded(variable.result);
  expect(sent(variable.requests[0]!).at(-1)).toEqual(["user", "Say hi."]);

  const single = await run(
    `
id: t
name: t
input:
  item: !str
kind: agent
prompt: Say \${item}.
output: !str
`,
    { item: "hi" },
  );
  expectSucceeded(single.result);
  expect(sent(single.requests[0]!).at(-1)).toEqual(["user", "Say hi."]);
});

test("a map binds each item to a def input named item, which its agent reads as item", async () => {
  const { result, requests } = await run(
    `
id: t
name: t
input:
  xs: !list(str)
defs:
  one:
    input:
      item: !str
    nodes:
      a:
        kind: agent
        input:
          item: \${input.item}
        prompt: Say \${item}.
        output: !str
    output: \${a.output}
nodes:
  m:
    kind: map
    over: \${input.xs}
    call: one
    input:
      item: \${item}
output: \${m.output}
`,
    { xs: ["a", "b"] },
  );
  expectSucceeded(result);
  expect(requests.map((request) => sent(request).at(-1)![1]).sort()).toEqual(["Say a.", "Say b."]);
});

test("node asserts read an input named item, before and after the node runs", async () => {
  const flow = (check: string) => `
id: t
name: t
input:
  x: !str
nodes:
  a:
    kind: code
    input:
      item: \${input.x}
    asserts:
      - \${${check}}
    code: |
      return inputs.item + "!"
    output: !str
output: \${a.output}
`;
  for (const check of ['item == "hi"', 'output == item + "!"']) {
    const { result } = await run(flow(check), { x: "hi" });
    expectSucceeded(result);
    expect(result.output).toBe("hi!");
  }
  const { result } = await run(flow('item == "no"'), { x: "hi" });
  expect(result.status).toBe("failed");
  expect(result.failure!.code).toBe("assertion.failed");
});

test("a call node's assert reads its input named item", async () => {
  const { result } = await run(
    `
id: t
name: t
input:
  x: !str
defs:
  one:
    input:
      item: !str
    nodes:
      a:
        kind: code
        input:
          item: \${input.item}
        code: |
          return inputs.item + "!"
        output: !str
    output: \${a.output}
nodes:
  c:
    kind: call
    call: one
    input:
      item: \${input.x}
    asserts:
      - \${item == "hi" and output == "hi!"}
output: \${c.output}
`,
    { x: "hi" },
  );
  expectSucceeded(result);
  expect(result.output).toBe("hi!");
});

test("a loop's carried field named item drives its until: and while:", async () => {
  const flow = (driver: string) => `
id: t
name: t
nodes:
  l:
    kind: loop
    input:
      item: ""
    ${driver}
    max: 5
    nodes:
      step:
        kind: code
        input:
          item: \${input.item}
        output: !str
        code: |
          return inputs.item + "x"
    output:
      item: \${step.output}
output: \${l.output.item}
`;
  for (const driver of ['until: item == "xxx"', "while: len(item) < 3"]) {
    const { result } = await run(flow(driver));
    expectSucceeded(result);
    expect(result.output).toBe("xxx");
  }
});
