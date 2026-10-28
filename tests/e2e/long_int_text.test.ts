/**
 * An expression that writes an int over 4300 digits as text. Each run is resumed from its log,
 * which must end the same.
 */

import { afterAll, expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { loadFlow } from "../../src/compose/loader.ts";
import type { ModelAccess } from "../../src/llm_clients/factory.ts";
import { TextPart } from "../../src/llm/messages.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { type RunResult, advance, runFlow } from "../../src/run.ts";
import type { AskRequest, ToolEntry } from "../../src/tools/registry.ts";
import { hostEnv } from "../helpers/mcp_helpers.ts";
import { serving, textReply } from "../helpers/run_state_fixtures.ts";

const dir = mkdtempSync(join(tmpdir(), "yian-long-int-text-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));
const env = hostEnv(dir);

const POWER = 10n ** 4999n; // `10 ** input.n` with `n = 4999`
const DIGITS = String(POWER);

let runs = 0;

/** Run the flow `text` with `n = 4999`, check its log resumes to the same end, and return the result. */
async function run(
  text: string,
  options: { learnable?: boolean; tools?: Record<string, ToolEntry>; models?: ModelAccess } = {},
): Promise<RunResult> {
  const flow = await loadFlow(text, { env });
  const log = join(dir, `run-${runs++}.jsonl`);
  const result = await runFlow(flow, { n: 4999n }, { env, log, ...options });
  const resumed = await advance({ log, env, tools: options.tools ?? null });
  expect(resumed.status).toBe(result.status);
  if (result.failure !== null) expect(resumed.failure!.message).toBe(result.failure.message);
  else expect(resumed.output).toEqual(result.output);
  return result;
}

/** Assert the run succeeded with `output`. */
function assertSucceeds(result: RunResult, output: unknown): void {
  expect(result.failure).toBeNull();
  expect(result.status).toBe("succeeded");
  expect(result.output).toEqual(output);
}

/** Return a flow whose `c` CODE node binds its input `x` to `binding` and returns it. */
const bindingFlow = (binding: string) => `
id: long
name: long
input:
  n: !int
nodes:
  c:
    kind: code
    input:
      x: ${binding}
    code: |
      return inputs.x
output: \${c.output}
`;

test("a long int in a text binding fails at the binding", async () => {
  assertSucceeds(await run(bindingFlow("'v=${10 ** input.n}'")), `v=${DIGITS}`);
});

test("a long int in an agent prompt fails the node", async () => {
  const model = new ScriptedModel([textReply("ok")]);

  const result = await run(
    `
id: long
name: long
llm_config: {provider: openai, model: test-model}
input:
  n: !int
nodes:
  a:
    kind: agent
    prompt: "v=\${10 ** n}"
    input:
      n: \${input.n}
    output: str
output: \${a.output}
`,
    { models: serving(model, env) },
  );

  assertSucceeds(result, "ok");
  expect(model.requests).toHaveLength(1);
  const user = model.requests[0]!.messages.at(-1)!;
  expect(user.role).toBe("user");
  expect((user.parts[0] as TextPart).text).toBe(`v=${DIGITS}`);
});

test("a long int in a human input question fails the node", async () => {
  const requests: AskRequest[] = [];
  const askUser = (request: AskRequest) => {
    requests.push(request);
    return "yes";
  };

  const result = await run(
    `
id: long
name: long
input:
  n: !int
nodes:
  ask:
    kind: tool
    tool_id: ask
    input:
      prompt: "v=\${10 ** input.n}"
    output: str
output: \${ask.output}
`,
    { tools: { ask: askUser } },
  );

  assertSucceeds(result, "yes");
  expect(requests.map((request) => request.question)).toEqual([`v=${DIGITS}`]);
});

test.each([
  ["int", `'\${"%d" % (10 ** input.n)}'`],
  ["list", `'\${"%s" % [10 ** input.n]}'`],
])("formatting a long int with percent fails (%s)", async (_id, binding) => {
  const result = await run(bindingFlow(binding));

  expect(result.status).toBe("failed");
  expect(result.failure!.code).toBe("binding.evaluate");
  expect(result.failure!.nodeId).toBe("c");
  expect(result.failure!.message).toBe("text formatting with `%` was removed; use a template");
});

test.each([
  ["join", `'\${join([10 ** input.n, 1], ",")}'`, `${DIGITS},1`],
  ["render_as_json", "'${render_as_json([1, 10 ** input.n])}'", `[\n  1,\n  ${DIGITS}\n]`],
])("a builtin writing a long int as text fails (%s)", async (_id, binding, output) => {
  assertSucceeds(await run(bindingFlow(binding)), output);
});

test("a learnable run with a long intermediate succeeds and resumes (default)", async () => {
  const result = await run(
    `
id: long
name: long
input:
  n: !int
nodes:
  c:
    kind: code
    input:
      x: \${(10 ** input.n) % 7}
    code: |
      return inputs["x"]
output: \${c.output}
`,
    { learnable: true },
  );

  assertSucceeds(result, POWER % 7n);
});
