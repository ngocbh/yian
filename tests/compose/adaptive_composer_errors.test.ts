/**
 * A load error in an adaptive gate's composer names the gate and points at the authored text: a
 * bad `llm_config`, typed inputs, references, and cycles through the gate.
 */

import { expect, test } from "bun:test";

import type { LoadError } from "../../src/compose/helpers/errors.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import {
  assembled,
  assembleText,
  type CaseRow,
  loadError,
  loadErrorAsync,
  runCase,
} from "../helpers/declarations.ts";

const LLM_SITE = "node 'ask' adaptive_questions.llm_config";

/** Return the adaptive-gate flow with the `llm_config:` lines `llmConfig`; its value starts on line 12. */
const flow = (llmConfig = "") => `\
id: f
name: f
input: {proj: !str }
nodes:
  ask:
    kind: tool
    tool_id: ask
    input:
      proj: \${input.proj}
    adaptive_questions:
      prompt: "Q \${proj}"
${llmConfig}output: \${ask.output}
`;

/** Return the 1-based line of the error's last frame. */
const lineOf = (error: LoadError) => error.diagnostic.frames.at(-1)!.range.startLine;

test.each([
  ["inherit", "{inherit: 'false'}", `${LLM_SITE} \`inherit:\` must be true or false, got "false"`],
  [
    "model-type",
    "{model: 5}",
    `${LLM_SITE}: \`model:\` must be text, but YAML read it as an integer; put it in quotes`,
  ],
  [
    "unknown-key",
    "{temparature: 1}",
    `${LLM_SITE}: 1 validation error for LLMConfig\ntemparature\n  Extra inputs are not permitted`,
  ],
  ["not-mapping", "5", `${LLM_SITE} must be a mapping`],
  [
    "api-key",
    "{api_key: x}",
    `${LLM_SITE}.api_key is a host credential; pass it in the run's llm_config, not in a flow file`,
  ],
])("bad llm config is named and located at the llm config: %s", (_id, config, message) => {
  const error = loadError(() => assembleText(flow(`      llm_config: ${config}\n`)));
  expect(error.message).toBe(message);
  expect(lineOf(error)).toBe(12);
});

test("block llm config error is located at the offending key", () => {
  const error = loadError(() => assembleText(flow("      llm_config:\n        model: m\n        bogus: 1\n")));
  expect(error.message).toBe(
    `${LLM_SITE}: 1 validation error for LLMConfig\nbogus\n  Extra inputs are not permitted`,
  );
  expect(lineOf(error)).toBe(14);
});

test("valid llm config loads", () => {
  expect(assembleText(flow("      llm_config: {model: m}\n")).flow.compiled.nodes.has("ask__compose")).toBe(true);
});

test("a compact flow's llm config error names the flow node", async () => {
  const error = await loadErrorAsync(
    loadFlow(`\
id: f
name: f
input: {proj: !str }
kind: tool
tool_id: ask
adaptive_questions:
  prompt: "Q \${proj}"
  llm_config: {temparature: 1}
`),
  );
  expect(error.message.startsWith("node 'f' adaptive_questions.llm_config")).toBe(true);
  expect(error.message).not.toContain("f__compose");
  expect(lineOf(error)).toBe(8);
});

/**
 * Return the flow whose gate `ask` has the `input:` lines `inputs`, the brief `brief` and the
 * input `prompt:` line `gatePrompt` (default none); its first input is on line 13.
 */
const inputFlow = (inputs: string, brief = "Q ${proj}", gatePrompt = "") => `\
id: f
name: f
input: {proj: !str }
nodes:
  up:
    kind: code
    output: !int
    code: "return 1"
  ask:
    kind: tool
    tool_id: ask
    input:
${inputs}${gatePrompt}    adaptive_questions:
      prompt: "${brief}"
output: \${ask.output}
`;

/** Return each of the error's site lines (those starting `node `) up to its first `: `. */
const sites = (error: LoadError) =>
  error.message
    .split("\n")
    .map((ln) => ln.trim())
    .filter((ln) => ln.startsWith("node "))
    .map((ln) => ln.split(": ")[0]);

test.each([
  ["flow-input", "${input.nope}", "points to unknown flow input 'nope'"],
  ["namespace", "${nope.output}", "uses unknown namespace 'nope'"],
  ["field", "${up.output.x}", "reads field 'x' on type int"],
  ["default", "${input.nope :- 'x'}", "points to unknown flow input 'nope'"],
])("a bad input is reported once on the gate: %s", (_id, source, fragment) => {
  const error = loadError(() => assembleText(inputFlow(`      proj: "${source}"\n`)));
  const lines = error.message
    .split("\n")
    .filter((ln) => ln.includes("input 'proj'"))
    .map((ln) => ln.trim());

  expect(lines).toHaveLength(1);
  expect(lines[0]!.startsWith("node 'ask' input 'proj' from: ")).toBe(true);
  expect(lines[0]).toContain(fragment);
  expect(error.message).not.toContain("ask__compose");
  expect(lineOf(error)).toBe(13);
});

test("two inputs with the same bad source are both reported", () => {
  const error = loadError(() =>
    assembleText(inputFlow("      x: ${input.nope}\n      y: ${input.nope}\n", "Q ${x} ${y}")),
  );

  expect(sites(error)).toEqual(["node 'ask' input 'x' from", "node 'ask' input 'y' from"]);
  expect(error.message).not.toContain("ask__compose");
});

test("a bad brief and a bad input are each reported once", () => {
  const error = loadError(() => assembleText(inputFlow("      proj: ${input.nope}\n", "Q ${nope}")));

  expect(sites(error)).toEqual(["node 'ask' adaptive_questions.prompt", "node 'ask' input 'proj' from"]);
  expect(error.message).not.toContain("ask__compose");
});

test("a bad input is listed before a bad gate prompt", () => {
  const error = loadError(() =>
    assembleText(inputFlow("      proj: ${input.nope}\n", "Q ${proj}", '      prompt: "${nope}"\n')),
  );

  expect(sites(error)).toEqual(["node 'ask' input 'proj' from", "node 'ask' input 'prompt' from"]);
  expect(error.message).not.toContain("ask__compose");
});

test("typed input mismatch names the gate", () => {
  const error = loadError(() => assembleText(inputFlow("      proj: !list(int) ${input.proj}\n", "Q ${proj}")));
  expect(error.message).toBe("node 'ask' input 'proj': declared list(int), but the value is str");
  expect(lineOf(error)).toBe(13);
});

test("isinstance unknown type names the gate", () => {
  const error = loadError(() => assembleText(inputFlow("      proj: ${isinstance(input.proj, Nope)}\n", "Q ${proj}")));
  expect(error.message).toBe(
    "node 'ask' input 'proj': unknown type 'Nope'; isinstance takes a typedefs: name or one of str, int, float, " +
      "bool, date, datetime, object, list, Message",
  );
  expect(lineOf(error)).toBe(13);
});

test("literal inputs still load", () => {
  const inputs = "      cfg: {a: 1}\n      ls: [1, 2]\n      n: !list(int) [1]\n";
  expect(assembleText(inputFlow(inputs, "Q ${cfg} ${ls} ${n}")).flow.compiled.nodes.has("ask__compose")).toBe(true);
  const agentFlow = `\
id: f
name: f
nodes:
  a:
    kind: agent
    input:
${inputs}    prompt: "Q \${cfg} \${ls} \${n}"
    output: !str
output: \${a.output}
`;
  expect(assembleText(agentFlow).flow.compiled.nodes.has("a")).toBe(true);
});

test("skipped operand arithmetic names the gate", () => {
  const error = loadError(() =>
    assembleText(`\
id: f
name: f
input:
  approve: !bool
nodes:
  gate:
    kind: case
    cases:
      - when: \${input.approve}
        then: order
  order:
    kind: code
    output: {qty: !int}
    code: |
      return {"qty": 1}
  ask:
    kind: tool
    tool_id: ask
    input:
      q: \${order.output.qty * 2}
    adaptive_questions:
      prompt: "Q \${q}"
output: \${ask.output}
`),
  );
  expect(error.message).toBe(
    "node 'ask' input 'q': ${order.output.qty} reads field 'qty' of order.output, which may be null " +
      "(node 'order' may be skipped while this node runs); write (order.output?.qty :- <default>)",
  );
  expect(lineOf(error)).toBe(20);
});

test("a defs body's input error names the gate", async () => {
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
          proj: \${input.nope}
        adaptive_questions:
          prompt: "Q \${proj}"
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
  const lines = error.message.split("\n").filter((ln) => ln.includes("input 'proj'"));
  expect(lines).toHaveLength(1);
  expect(lines[0]).toContain("node 'ask' input 'proj' from: ");
  expect(error.message).not.toContain("ask__compose");
  expect(lineOf(error)).toBe(14);
});

// ---------- cycles: the gate is named, not its composer, unless an author reads the composer ----------

/** Return the cycle message naming `names`. */
const cycleMessage = (names: string[]) => `flow has a cycle involving ${JSON.stringify(names)}; flows must be acyclic`;
const GENERATED = "ask__compose is generated from node 'ask'";

const SELF_READING = `\
id: f
name: f
input: {proj: !str }
nodes:
  ask:
    kind: tool
    tool_id: ask
    input:
      proj: \${ask.output}
    adaptive_questions:
      prompt: "Q \${proj}"
`;

/** Return the 1-based line of the first line of `text` that starts with `needle`. */
function lineStarting(text: string, needle: string): number {
  const index = text.split("\n").findIndex((line) => line.startsWith(needle));
  if (index < 0) throw new Error(`${needle} not in flow`);
  return index + 1;
}

/** Return the start line of each frame of `error`, in order. */
const frameLines = (error: LoadError) => error.diagnostic.frames.map((frame) => frame.range.startLine);

test("a self-reading gate's cycle names only the gate", () => {
  const text = SELF_READING + "output: ${ask.output}\n";
  const error = loadError(() => assembleText(text));
  expect(error.message).toBe(cycleMessage(["ask"]));
  expect(frameLines(error)).toEqual([lineStarting(text, "  ask:")]);
  expect(error.diagnostic.notes).toEqual(["ask depends on ask (ask.input.proj)"]);
});

/** Return the flow where `b` reads `bReads` and the gate `ask`, with `gateExtra` lines, reads `gateReads`. */
const throughB = (bReads: string, gateExtra: string, gateReads: string) => `\
id: f
name: f
input: {proj: !str }
nodes:
  b:
    kind: agent
    input:
      a: \${${bReads}}
    prompt: "x \${a}"
  ask:
    kind: tool
    tool_id: ask
${gateExtra}    input:
      proj: \${${gateReads}}
    adaptive_questions:
      prompt: "Q \${proj}"
output: \${ask.output}
`;

test("a cycle through a gate names only authored nodes", () => {
  const text = throughB("ask.output", "", "b.output");
  const error = loadError(() => assembleText(text));
  expect(error.message).toBe(cycleMessage(["ask", "b"]));
  expect(frameLines(error)).toEqual([lineStarting(text, "  b:"), lineStarting(text, "  ask:")]);
  expect(error.diagnostic.notes).toEqual(["ask depends on b (ask.input.proj)", "b depends on ask (b.input.a)"]);
});

test("an ordering cycle through a gate names only authored nodes", () => {
  const error = loadError(() => assembleText(throughB("ask.output", "    depends_on: [b]\n", "input.proj")));
  expect(error.message).toBe(cycleMessage(["ask", "b"]));
  expect(error.diagnostic.notes).toEqual(["ask depends on b (ordering)", "b depends on ask (b.input.a)"]);
});

test("a cycle through a direct composer reader keeps the id", () => {
  const error = loadError(() => assembleText(throughB("ask__compose.output", "", "b.output")));
  expect(error.message).toBe(cycleMessage(["ask__compose", "b"]));
  expect(error.diagnostic.notes).toContain("b depends on ask__compose (b.input.a)");
  expect(error.diagnostic.notes).toContain(GENERATED);
});

test("a composer read outside the cycle still names only the gate", () => {
  const text = `${SELF_READING}\
  log:
    kind: code
    input:
      q: \${ask__compose.output}
    output: !int
    code: return inputs["q"].length
output:
  a: \${ask.output}
  q: \${ask__compose.output}
`;
  const error = loadError(() => assembleText(text));
  expect(error.message).toBe(cycleMessage(["ask"]));
  expect(frameLines(error)).toEqual([lineStarting(text, "  ask:")]);
  expect(error.diagnostic.notes).toEqual(["ask depends on ask (ask.input.proj)"]);
});

test("two self-reading gates each name only the gate", () => {
  const text = `${SELF_READING}\
  ask2:
    kind: tool
    tool_id: ask
    input:
      proj: \${ask2.output}
    adaptive_questions:
      prompt: "Q \${proj}"
output:
  a: \${ask.output}
  b: \${ask2.output}
`;
  const error = loadError(() => assembleText(text));
  expect(error.message).toBe(cycleMessage(["ask", "ask2"]));
  expect(frameLines(error)).toEqual([lineStarting(text, "  ask:"), lineStarting(text, "  ask2:")]);
  expect(error.diagnostic.notes).toEqual([
    "ask depends on ask (ask.input.proj)",
    "ask2 depends on ask2 (ask2.input.proj)",
  ]);
});

test("a gate reading its own composer keeps the id", () => {
  // The composer copies the gate's `x`, so it reads itself: the cycle is its self-edge.
  const text = `\
id: f
name: f
input: {proj: !str }
nodes:
  ask:
    kind: tool
    tool_id: ask
    input:
      proj: \${input.proj}
      x: \${ask__compose.output}
    adaptive_questions:
      prompt: "Q \${proj} \${x}"
output: \${ask.output}
`;
  const error = loadError(() => assembleText(text));
  expect(error.message).toBe(cycleMessage(["ask__compose"]));
  expect(error.diagnostic.notes).toEqual(["ask__compose depends on ask__compose (ask__compose.input.x)", GENERATED]);
});

test("a defs child's cycle names only the gate", async () => {
  const text = `\
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
          proj: \${ask.output}
        adaptive_questions:
          prompt: "Q \${proj}"
    output: \${ask.output}
nodes:
  run:
    kind: call
    call: pick
    input:
      topic: \${input.topic}
output: \${run.output}
`;
  const error = await loadErrorAsync(loadFlow(text));
  expect(error.message.startsWith("defs entry 'pick': " + cycleMessage(["ask"]))).toBe(true);
  expect(error.message).not.toContain("ask__compose");
  // The defs entry's frame is its body, which starts on the line after `pick:`.
  expect(frameLines(error)).toEqual([lineStarting(text, "  pick:") + 1, lineStarting(text, "      ask:")]);
});

// ---------- harvested conformance cases ----------

test.each<CaseRow>([
  { id: "load-agent/engine_adaptive_composer_errors__api_key_in_the_composer_llm_config", call: assembled },
  { id: "load-agent/engine_adaptive_composer_errors__unknown_key_in_the_composer_llm_config", call: assembled },
  { id: "load-agent/engine_adaptive_composer_errors__valid_llm_config_loads", call: assembled },
  {
    id: "load-agent/engine_adaptive_composer_errors__bad_input_is_reported_once_on_the_gate",
    call: assembled,
    full: true,
  },
  { id: "load-agent/engine_adaptive_composer_errors__composer_reader_cycle_keeps_the_id", call: assembled, full: true },
  {
    id: "load-agent/engine_adaptive_composer_errors__cycle_through_a_gate_names_only_authored_nodes",
    call: assembled,
    full: true,
  },
])("assemble: $id", runCase);
