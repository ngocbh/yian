/** Loading a removed node kind, or the removed `tool_id:`, is a load error that names what to write instead. */
import { expect, test } from "bun:test";

import type { LoadError } from "../../src/compose/helpers/errors.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { rejected } from "../helpers/run_state_fixtures.ts";

/** The message loading a flow whose one node `n` is `body` fails with. */
async function refusal(body: string): Promise<string> {
  const error = await rejected(loadFlow(`id: f\nname: f\nnodes:\n  n:\n${body}output: \${n.output}\n`));
  return (error as LoadError).message;
}

test.each([
  [
    "human_input",
    "kind 'human_input' was removed; to ask a person, write `kind: call` with `call: ask` " +
      "(input: prompt, questions, timeout)",
  ],
  ["wait", "kind 'wait' was removed; to wait, write `kind: call` with `call: sleep` (input: until or for)"],
])("loading kind %p names its replacement", async (kind, why) => {
  expect(await refusal(`    kind: ${kind}\n`)).toContain(`node 'n': ${why}`);
});

// Each row: the node's `tool_id:` lines, and the `call:` the message shows for them.
test.each([
  ["    tool_id: search\n", "search"],
  ["    tool_id: {github: search}\n", "{github: search}"],
  ["    tool_id:\n      github: search\n", "{github: search}"],
  ["", "<tool>"],
  ["    tool_id: [search]\n", "<tool>"],
  ["    tool_id: 'web search'\n", "<tool>"],
  ["    tool_id: |\n      search\n", "<tool>"],
  ["    tool_id: {github: 'web search'}\n", "<tool>"],
])("kind: tool with %p shows the call to write", async (toolId, call) => {
  expect(await refusal(`    kind: tool\n${toolId}`)).toBe(
    `node 'n': kind 'tool' was removed; write \`kind: call\` with \`call: ${call}\``,
  );
});

test.each([
  ["call", ""],
  ["map", "    over: [1]\n"],
  ["loop", "    max: 2\n"],
])("tool_id: on a %s node shows the call: to write", async (kind, rest) => {
  expect(await refusal(`    kind: ${kind}\n    tool_id: {github: search}\n${rest}`)).toBe(
    `node 'n' (kind=${kind}): \`tool_id:\` is now \`call:\`; write \`call: {github: search}\``,
  );
});
