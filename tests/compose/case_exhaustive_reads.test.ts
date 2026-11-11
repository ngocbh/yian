/**
 * A CASE whose result is read through `${case.output}` must be exhaustive.
 *
 * `${case.output}` reads a coalesce over the CASE's branch targets, so a CASE read that way (or
 * one such a CASE routes to) must take a branch on every run: it has an `else:`, or it is an `on:`
 * form whose `on:` type is a non-nullable `Literal` or `bool` listed in full, under the narrowing
 * of the CASE branches that reach it. A routing-only CASE may match nothing and run no branch. An
 * unquoted bare word names a node, even a YAML 1.1 boolean word.
 */

import { expect, test } from "bun:test";

import type { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import type { LoadError } from "../../src/compose/helpers/errors.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { NodeSkipped } from "../../src/events.ts";
import { type RunResult, runFlow } from "../../src/run.ts";
import {
  assembled,
  assembleText,
  type CaseRow,
  loadConformanceFlow,
  loadError,
  runCase,
} from "../helpers/declarations.ts";
import { startedIds } from "../helpers/run_state_fixtures.ts";

/** Return the 1-based number of the first line of `text` holding `needle`. */
function lineOf(text: string, needle: string): number {
  return text.split("\n").findIndex((line) => line.includes(needle)) + 1;
}

/** Return the start line of a load error's innermost frame. */
function errorLine(error: LoadError): number {
  return error.diagnostic.frames.at(-1)!.range.startLine;
}

/** The start of the message for a value-producing case `caseId` that is not exhaustive. */
function exhaustive(caseId: string): string {
  return `case node '${caseId}' produces a value, so it must be exhaustive: add an \`else:\``;
}

/** Run the loaded `flow` on `inputs`, returning its output; the run must succeed. */
async function outputOf(flow: ComposedFlow, inputs: Readonly<Record<string, unknown>>): Promise<unknown> {
  const result = await runFlow(flow, inputs);
  expect(result.status, result.failure?.message).toBe("succeeded");
  return result.output;
}

/** Return the ids of the nodes a run skipped. */
const skippedIds = (result: RunResult) =>
  result.events.filter((event) => event instanceof NodeSkipped).map((event) => (event as NodeSkipped).nodeId);

/** Return a flow with a `gate` case (indented body), CODE nodes `pa` and `pb`, and `output`. */
function gateFlow(
  gate: string,
  options: { inputs?: string; output?: string; extraNodes?: string; typedefs?: string } = {},
): string {
  const { inputs = "  x: !str\n", output = "${gate.output}", extraNodes = "", typedefs = "" } = options;
  return (
    `id: f\nname: f\n${typedefs}input:\n${inputs}nodes:\n  gate:\n    kind: case\n${gate}` +
    '  pa:\n    kind: code\n    code: |\n      return "A"\n    output: str\n' +
    '  pb:\n    kind: code\n    code: |\n      return "B"\n    output: str\n' +
    `${extraNodes}output: ${output}\n`
  );
}

const CHOICE = "typedefs:\n  Choice: Literal(a, b)\n";

/** Return `cases:` routing each tag to the CODE node `p<tag>`. */
function routes(...tags: string[]): string {
  return `    cases:\n${tags.map((tag) => `      - when: ${tag}\n        then: p${tag}\n`).join("")}`;
}

test("read case over a str without else is rejected", () => {
  const text = gateFlow("    on: ${input.x}\n" + routes("a", "b"));

  const error = loadError(() => assembleText(text));

  expect(error.message).toBe(`${exhaustive("gate")} (on: \${input.x} is str, not a Literal or bool)`);
  expect(errorLine(error)).toBe(lineOf(text, "  gate:"));
});

test("read searched case without else is rejected", () => {
  const text = gateFlow("    cases:\n      - when: \"input.x == 'a'\"\n        then: pa\n");

  const error = loadError(() => assembleText(text));

  expect(error.message).toBe(`${exhaustive("gate")} (a searched case needs an else:)`);
  expect(errorLine(error)).toBe(lineOf(text, "  gate:"));
});

const PICK =
  "  pick:\n    kind: code\n    input:\n      x: ${input.x}\n" +
  '    code: |\n      return inputs["x"]\n    output: Literal(a, b)\n';

test("read case over a Literal node output listing every tag runs", async () => {
  const loaded = await loadFlow(gateFlow("    on: ${pick.output}\n" + routes("a", "b"), { extraNodes: PICK }));

  expect(await outputOf(loaded, { x: "a" })).toBe("A");
  expect(await outputOf(loaded, { x: "b" })).toBe("B");
});

test("read case over a Literal node output missing a tag is rejected", () => {
  const text = gateFlow("    on: ${pick.output}\n" + routes("a"), { extraNodes: PICK });

  const error = loadError(() => assembleText(text));

  expect(error.message).toBe(`${exhaustive("gate")} (enum tag(s) ["b"] not covered)`);
  expect(errorLine(error)).toBe(lineOf(text, "  gate:"));
});

test("read case over a Literal input listing every tag runs", async () => {
  const loaded = await loadFlow(
    gateFlow("    on: ${input.choice}\n" + routes("a", "b"), { typedefs: CHOICE, inputs: "  choice: !Choice\n" }),
  );

  expect(await outputOf(loaded, { choice: "a" })).toBe("A");
  expect(await outputOf(loaded, { choice: "b" })).toBe("B");
});

test("read case over a Literal input missing a tag is rejected", () => {
  const text = gateFlow("    on: ${input.choice}\n" + routes("a"), { typedefs: CHOICE, inputs: "  choice: !Choice\n" });

  expect(loadError(() => assembleText(text)).message).toBe(`${exhaustive("gate")} (enum tag(s) ["b"] not covered)`);
});

test("read case over an Optional Literal input is rejected", () => {
  const text = gateFlow("    on: ${input.choice}\n" + routes("a", "b"), {
    typedefs: CHOICE,
    inputs: "  choice: !Optional(Choice)\n",
  });

  expect(loadError(() => assembleText(text)).message).toBe(`${exhaustive("gate")} (\${input.choice} may be null)`);
});

test("routing-only case over a str without else still skips on no match", async () => {
  const loaded = await loadFlow(
    gateFlow("    on: ${input.x}\n" + routes("a", "b"), { output: "${pa.output | pb.output:-'none'}" }),
  );

  expect(await outputOf(loaded, { x: "a" })).toBe("A");
  const result = await runFlow(loaded, { x: "c" });
  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(result.output).toBe("none");
  expect(new Set(skippedIds(result))).toEqual(new Set(["pa", "pb"]));
  expect(new Set(startedIds(result))).toEqual(new Set(["__start__", "gate", "__end__"]));
});

test("routing-only case over a Literal input missing a tag is a load error", () => {
  const text = gateFlow("    on: ${input.choice}\n" + routes("a"), {
    typedefs: CHOICE,
    inputs: "  choice: !Choice\n",
    output: "${input.choice}",
  });

  const error = loadError(() => assembleText(text));

  expect(error.message).toBe(
    'case node \'gate\' on ${input.choice}: non-exhaustive — enum tag(s) ["b"] not covered by a case ' +
      "`when:` or an `else:`",
  );
  expect(errorLine(error)).toBe(lineOf(text, "  gate:"));
});

test("inner case of a value-producing case must be exhaustive", () => {
  const text = gateFlow("    on: ${input.x}\n    cases:\n      - when: a\n        then: pa\n    else: inner\n", {
    extraNodes: "  inner:\n    kind: case\n    on: ${input.x}\n    cases:\n      - when: b\n        then: pb\n",
  });

  const error = loadError(() => assembleText(text));

  expect(error.message).toBe(`${exhaustive("inner")} (on: \${input.x} is str, not a Literal or bool)`);
  expect(errorLine(error)).toBe(lineOf(text, "  inner:"));
});

test("case read inside text must be exhaustive", () => {
  const text = gateFlow("    on: ${input.x}\n" + routes("a", "b"), { output: "got ${gate.output}" });

  expect(loadError(() => assembleText(text)).message).toBe(
    `${exhaustive("gate")} (on: \${input.x} is str, not a Literal or bool)`,
  );
});

/** Return a flow whose `route` sends `buy` to `inner`, a read case on the same tag. */
function narrowedFlow(innerTag: string): string {
  return (
    "id: f\nname: f\n" +
    "typedefs:\n  Buy:\n    action: Literal(buy)\n    qty: int\n" +
    "  Sell:\n    action: Literal(sell)\n    price: float\n" +
    "input:\n  kind: !str\n" +
    "nodes:\n" +
    "  d:\n    kind: code\n    input:\n      kind: ${input.kind}\n    code: |\n" +
    '      if (inputs["kind"] === "buy")\n        return {action: "buy", qty: 3}\n' +
    '      return {action: "sell", price: 1.5}\n' +
    "    output: Union(Buy, Sell)\n" +
    "  route:\n    kind: case\n    on: ${d.output.action}\n    cases:\n" +
    "      - when: buy\n        then: inner\n    else: sold\n" +
    "  inner:\n    kind: case\n    on: ${d.output.action}\n    cases:\n" +
    `      - when: ${innerTag}\n        then: bought\n` +
    '  bought: !str "bought ${d.output.qty}"\n' +
    "  sold: !str sold\n" +
    "output: ${route.output}\n"
  );
}

test("narrowed read case counts only the tags that reach it", async () => {
  const loaded = await loadFlow(narrowedFlow("buy"));

  expect(await outputOf(loaded, { kind: "buy" })).toBe("bought 3");
  expect(await outputOf(loaded, { kind: "sell" })).toBe("sold");
});

test("narrowed read case listing only a tag that can't reach it is rejected at that when", () => {
  const text = narrowedFlow("sell");

  const error = loadError(() => assembleText(text));

  expect(error.message).toBe(
    "case node 'inner': `when: sell` never matches: the branches that reach this case rule it out, " +
      "so `on: ${d.output.action}` is `Literal(buy)` here; remove it",
  );
  expect(errorLine(error)).toBe(lineOf(text, "when: sell"));
});

test("unquoted yes routes to a node named yes", async () => {
  const loaded = await loadConformanceFlow(
    "id: f\nname: f\ninput:\n  cmd: !str\nnodes:\n" +
      "  route:\n    kind: case\n    on: ${input.cmd}\n    cases:\n" +
      "      - when: a\n        then: reply\n    else: yes\n" +
      "  reply:\n    kind: code\n    code: ./fixtures/compose_codefns.ts:echo\n" +
      "    input:\n      topic: replied\n    output: str\n" +
      "  yes:\n    kind: code\n    code: ./fixtures/compose_codefns.ts:echo\n" +
      "    input:\n      topic: agreed\n    output: str\n" +
      "output: ${route.output}\n",
  );

  const result = await runFlow(loaded, { cmd: "b" });
  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(startedIds(result)).toContain("yes");
  expect(startedIds(result)).not.toContain("reply");
});

// ---------- harvested conformance cases ----------

test.each<CaseRow>(
  [
    "inner_case_of_a_read_case_must_be_exhaustive",
    "read_case_over_a_literal_missing_a_tag",
    "read_searched_case_without_else",
  ].map((name) => ({ id: `load-case/engine_case_exhaustive_reads__${name}`, call: assembled })),
)("assemble: $id", runCase);
