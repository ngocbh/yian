/**
 * A bad adaptive brief is reported on the gate at the brief, beside the gate's prompt errors, in
 * full, compact and `defs:` flows.
 */

import { expect, test } from "bun:test";

import type { LoadError } from "../../src/compose/helpers/errors.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { assembled, assembleText, loadError, loadErrorAsync, runCase } from "../compose/declarations.ts";

const GATE_PROMPT = '      prompt: "Pick one for ${input.proj}"\n';

const SITE = "node 'ask' adaptive_questions.prompt: ";

/**
 * Return the adaptive-gate flow whose `ask` has input `prompt:` line `gatePrompt` and the brief
 * `brief`: the ask's `prompt:` is on line 10 and the brief on line 12; without the `prompt:`, the
 * brief moves up to line 11.
 */
const flow = (brief: string, gatePrompt = GATE_PROMPT) => `\
id: f
name: f
input: {proj: !str }
nodes:
  ask:
    kind: tool
    tool_id: ask
    input:
      proj: \${input.proj}
${gatePrompt}    adaptive_questions:
      prompt: "${brief}"
output: \${ask.output}
`;

/** Return the `LoadError` assembling flow `text` raises. */
const loadFails = (text: string) => loadError(() => assembleText(text));

/** Return the 1-based line of the error's last frame. */
const lineOf = (error: LoadError) => error.diagnostic.frames.at(-1)!.range.startLine;

const BRIEFS = [
  ["Q ${nope}", "${nope} is not a declared input or variable"],
  [
    "Q ${input.proj}",
    "${input.proj} is not a declared input or variable (a pool ref belongs in an " +
      "input `from:`, not a strict AGENT prompt)",
  ],
  ["Q ${upper(${proj})}", "write `${upper(proj)}`"],
  ["Q ${proj +}", "could not parse"],
] as const;

test.each(
  (
    [
      ["gate-prompt", GATE_PROMPT, 12],
      ["no-gate-prompt", "", 11],
    ] as const
  ).flatMap(([id, gatePrompt, briefLine]) =>
    BRIEFS.map(([brief, fragment]) => [id, brief, gatePrompt, briefLine, fragment] as const),
  ),
)("a bad brief is named and located at the brief: %s %p", (_id, brief, gatePrompt, briefLine, fragment) => {
  const error = loadFails(flow(brief, gatePrompt));
  const line = error.message.split("\n").find((ln) => ln.includes(SITE))!;

  expect(line.trim().startsWith(SITE)).toBe(true);
  expect(line.split(SITE)[1]).toContain(fragment);
  expect(error.message).not.toContain("ask__compose");
  expect(lineOf(error)).toBe(briefLine);
});

test("brief and gate prompt errors are both reported", () => {
  const error = loadFails(flow("${b}", '      prompt: "${a}"\n'));
  const lines = error.message
    .split("\n")
    .map((ln) => ln.trim())
    .filter((ln) => ln.startsWith("node 'ask' "));

  expect(lines).toEqual([
    "node 'ask' adaptive_questions.prompt: ${b} is not a declared input or variable",
    "node 'ask' input 'prompt' from: reference ${a} uses unknown namespace 'a'",
  ]);
  expect(lineOf(error)).toBe(12);
});

test("an aliased brief shares the anchor line", () => {
  const error = loadFails(`\
id: f
name: f
input: {proj: !str }
nodes:
  ask:
    kind: tool
    tool_id: ask
    input:
      proj: \${input.proj}
      prompt: &p "Q \${nope}"
    adaptive_questions:
      prompt: *p
output: \${ask.output}
`);

  expect(error.message).toContain("node 'ask' input 'prompt' from: reference ${nope} uses unknown namespace 'nope'");
  expect(error.message).toContain("node 'ask' adaptive_questions.prompt: ${nope}");
  expect(lineOf(error)).toBe(10);
});

test("a compact flow's brief error names the flow node", () => {
  const error = loadFails(`\
id: f
name: f
input: {proj: !str }
kind: tool
tool_id: ask
adaptive_questions:
  prompt: "Q \${nope}"
`);

  expect(error.message).toContain("node 'f' adaptive_questions.prompt: ${nope}");
  expect(error.message).not.toContain("f__compose");
  expect(lineOf(error)).toBe(7);
});

test("a defs body's brief error", async () => {
  const error = await loadErrorAsync(
    loadFlow(`\
id: f
name: f
input:
  topic: !str
defs:
  pick:
    input:
      topic: !str
    nodes:
      ask:
        kind: tool
        tool_id: ask
        input:
          topic: \${input.topic}
        adaptive_questions:
          prompt: "Q \${nope}"
    output: \${ask.output}
nodes:
  run:
    kind: call
    call: pick
    input:
      topic: \${input.topic}
output: \${run.output}
`),
  );

  expect(error.message.startsWith("defs entry 'pick': ")).toBe(true);
  expect(error.message).toContain("node 'ask' adaptive_questions.prompt: ${nope}");
  expect(error.message).not.toContain("ask__compose");
  expect(lineOf(error)).toBe(16);
});

test("a clean brief loads", () => {
  expect(assembleText(flow("Q ${proj}")).flow.compiled.nodes.has("ask__compose")).toBe(true);
});

// ---------- harvested conformance cases ----------

test("assembled: load-agent/engine_adaptive_brief_errors__brief_and_gate_prompt_errors_are_both_reported", () => {
  runCase({
    id: "load-agent/engine_adaptive_brief_errors__brief_and_gate_prompt_errors_are_both_reported",
    call: assembled,
    full: true,
  });
});
