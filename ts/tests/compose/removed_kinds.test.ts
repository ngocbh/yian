/** Loading a removed node kind is a load error that names its replacement. TS-only. */
import { expect, test } from "bun:test";

import type { LoadError } from "../../src/compose/helpers/errors.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { rejected } from "../engine/run_state_fixtures.ts";

test.each([
  [
    "human_input",
    "kind 'human_input' was removed; ask a person with kind: tool, tool_id: ask (input: prompt, questions, timeout)",
  ],
  ["wait", "kind 'wait' was removed; wait with kind: tool, tool_id: sleep (input: until or for)"],
])("loading kind %p names its replacement", async (kind, why) => {
  const error = await rejected(loadFlow(`id: f\nname: f\nnodes:\n  n:\n    kind: ${kind}\noutput: \${n.output}\n`));
  expect((error as LoadError).message).toContain(`node 'n': ${why}`);
});
