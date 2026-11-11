/**
 * A too-deep expression is a located load error at every expression site; a long operator chain
 * loads and runs.
 *
 * Each call, list, unary operator, comparison, nested `${`, and chain of same-precedence operators
 * is one level, and an expression may nest 100. A deeper one fails the load with a message naming
 * the limit, located at the field that holds it. A chain of thousands of `+` or `*` terms is one
 * level, so it loads, runs, logs, and resumes.
 */

import { afterAll, expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { advance, runFlow } from "../../src/run.ts";
import { hostEnv } from "../helpers/mcp_helpers.ts";
import { caseText, loadConformanceFlow, loadErrorAsync } from "../helpers/declarations.ts";

const dir = mkdtempSync(join(tmpdir(), "yian-expression-nesting-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));
const env = hostEnv(dir);

const TOO_DEEP = "expression nests deeper than 100 levels";
const CHAIN = 3000;
/** The explicit timeout of a row that loads or runs a 3000-term chain. */
const SLOW = 60_000;

/** The 1-based line number of the first line of `text` containing `needle`. */
const lineOf = (text: string, needle: string) => text.split("\n").findIndex((line) => line.includes(needle)) + 1;

/** A condition of `n` nested `not`s. */
const not = (n: number) => "not ".repeat(n) + "true";

/** A number of `n` nested negations. */
const neg = (n: number) => "-(".repeat(n) + "1" + ")".repeat(n);

/** `n` nested `upper(...)` calls around `ref`. */
const upper = (n: number, ref: string) => "upper(".repeat(n) + ref + ")".repeat(n);

/** A list literal nested `n` deep. */
const list = (n: number) => "[".repeat(n) + "1" + "]".repeat(n);

/** A flow whose one CODE node binds `topic` to `binding`. */
function codeFlow(
  binding: string,
  options: { nodeExtra?: string; output?: string; tail?: string } = {},
): string {
  const { nodeExtra = "", output = "${use.output}", tail = "" } = options;
  return (
    "id: f\nname: f\ninput:\n  x: !int\n  s: !str\n" +
    "nodes:\n  use:\n    kind: code\n    input:\n      topic: " + binding + "\n" +
    "    output: str\n" + nodeExtra +
    "    code: |\n      return String(inputs.topic)\n" +
    "output: " + output + "\n" + tail
  );
}

/** A flow whose CASE `route` (fields `caseFields`) picks node `yes` or `no`. */
const routed = (caseFields: string) =>
  "id: f\nname: f\ninput:\n  x: !int\n  s: !str\nnodes:\n  route:\n    kind: case\n" +
  caseFields +
  "    else: no\n" +
  "  yes:\n    kind: code\n    output: str\n    code: |\n      return 'yes'\n" +
  "  no:\n    kind: code\n    output: str\n    code: |\n      return 'no'\n" +
  "output: ${yes.output | no.output}\n";

/** A routed flow with one searched branch on `condition`. */
const searched = (condition: string) => routed('    cases:\n      - when: "' + condition + '"\n        then: yes\n');

/** An `ask` flow whose question text holds an `n`-level span. */
const asked = (n: number) =>
  "id: f\nname: f\ninput:\n  proj: !str\nnodes:\n  ask:\n    kind: tool\n    tool_id: ask\n" +
  "    input:\n      questions:\n" +
  '        - question: "Which framework for ${' + upper(n, "input.proj") + '}?"\n' +
  "          header: Framework\n" +
  "output: ${ask.output}\n";

const SEED_05 = "load-map-call/seed_05_call_map";
// Seed 05's `uses:` child comes from the case's files.
const researchOne = caseText(SEED_05, { file: "03-research-one.yaml" }).text;

/** A seed's flow text, read from its harvested case. */
const seed = (id: string) => caseText(id).text;

const CALL_DEF =
  "defs:\n  g:\n    input:\n      d: !int\n    nodes:\n      run:\n        kind: code\n" +
  "        output: str\n        code: |\n          return 'x'\n    output: ${run.output}\n";

/** Site -> builder of [flow text, needle on the line the error is located at]. */
const SITES: Record<string, (n: number) => [string, string]> = {
  "searched when": (n) => [searched(not(n)), "- when:"],
  "searched when span": (n) => [searched("${" + not(n) + "}"), "- when:"],
  "case on": (n) => [routed('    on: "${' + neg(n) + '}"\n    cases:\n      - when: 1\n        then: yes\n'), "on:"],
  "whole binding": (n) => [codeFlow('"${' + neg(n) + '}"'), "topic:"],
  "binding in text": (n) => [codeFlow('"v=${' + neg(n) + '}"'), "topic:"],
  "flow output": (n) => [codeFlow("${input.s}", { output: '"${' + neg(n) + '}"' }), 'output: "${'],
  "flow assert": (n) => [codeFlow("${input.s}", { tail: "asserts:\n  - " + not(n) + "\n" }), "- not"],
  "node assert": (n) => [codeFlow("${input.s}", { nodeExtra: "    asserts:\n      - " + not(n) + "\n" }), "- not"],
  "loop until": (n) => [
    seed("load-loop/seed_28_refine_loop").replaceAll("until: score >= 0.8", "until: " + not(n)),
    "until: not",
  ],
  "map over": (n) => [
    seed(SEED_05).replaceAll("over: ${input.topics}", 'over: "${input.topics :- ' + list(n) + '}"'),
    'over: "${',
  ],
  "prompt span": (n) => [
    seed("load-core/seed_00_hello_agent").replaceAll("on ${topic}.", "on ${" + upper(n, "topic") + "}."),
    "prompt: |-",
  ],
  "call argument": (n) => [
    codeFlow("call(g, d=${" + neg(n) + "})").replace("nodes:\n", CALL_DEF + "nodes:\n"),
    "topic: call(",
  ],
  "typed binding": (n) => [codeFlow("!int ${" + neg(n) + "}"), "topic:"],
  "ask question": (n) => [asked(n), "- question:"],
};

test.each(Object.keys(SITES).flatMap((site) => [101, CHAIN].map((depth) => [site, depth] as const)))(
  "a too deep expression is a located load error (%s, %d)",
  async (site, depth) => {
    const [text, needle] = SITES[site]!(depth);
    const error = await loadErrorAsync(
      loadConformanceFlow(text, { childResolver: () => loadConformanceFlow(researchOne) }),
    );
    expect(error.message).toContain(TOO_DEEP);
    expect(error.diagnostic.frames.at(-1)!.range.startLine).toBe(lineOf(text, needle));
  },
  SLOW,
);

test(
  "a deep span in a case on match value is refused as a literal, not a nesting error",
  async () => {
    // A match value under `on:` is a literal, so a span in it is refused before it is parsed.
    const text = routed('    on: ${input.s}\n    cases:\n      - when: "${' + neg(CHAIN) + '}"\n        then: yes\n');
    const error = await loadErrorAsync(loadConformanceFlow(text));
    expect(error.message).toContain("an on: case's `when:` is a literal");
    expect(error.message.length).toBeLessThan(500);
    expect(error.diagnostic.frames.at(-1)!.range.startLine).toBe(lineOf(text, "- when:"));
  },
  SLOW,
);

/** `n` copies of `term` joined by ` + `. */
const sum = (n: number, term = "input.x") => Array(n).fill(term).join(" + ");

test(
  "a long sum binding loads and runs",
  async () => {
    const text =
      "id: f\nname: f\ninput:\n  x: !int\nnodes:\n  use:\n    kind: code\n    input:\n" +
      "      total: ${" + sum(CHAIN) + "}\n    output: int\n" +
      "    code: |\n      return inputs.total\noutput: ${use.output}\n";
    const result = await runFlow(await loadConformanceFlow(text), { x: 7 }, { env });
    expect(result.status).toBe("succeeded");
    expect(result.output).toBe(BigInt(CHAIN * 7));
  },
  SLOW,
);

test.each([
  [1, "yes"],
  [-1, "no"],
])(
  "a long sum condition routes (%d)",
  async (x, taken) => {
    const flow = await loadConformanceFlow(searched(sum(CHAIN) + " > 0"));
    expect((await runFlow(flow, { x, s: "a" }, { env })).output).toBe(taken);
  },
  SLOW,
);

test(
  "a long product binding runs",
  async () => {
    const text =
      "id: f\nname: f\ninput:\n  x: !int\nnodes:\n  use:\n    kind: code\n    input:\n" +
      "      total: ${input.x * " + Array(CHAIN).fill("1").join(" * ") + "}\n    output: int\n" +
      "    code: |\n      return inputs.total\noutput: ${use.output}\n";
    const result = await runFlow(await loadConformanceFlow(text), { x: 5 }, { env });
    expect(result.status).toBe("succeeded");
    expect(result.output).toBe(5n);
  },
  SLOW,
);

/** A flow passing a `terms`-term list concatenation through a CALL. */
const concatFlow = (terms: number) =>
  loadConformanceFlow(
    "id: long-concat\nname: long_concat\ninput:\n  values: !list(str)\n" +
      "defs:\n  identity:\n    input:\n      values: !list(str)\n    nodes: {}\n    output: ${input.values}\n" +
      "nodes:\n  outer:\n    kind: call\n    call: identity\n    input:\n" +
      "      values: ${" + sum(terms, "input.values") + "}\n" +
      "output: ${outer.output}\n",
  );

// Past the old evaluator's recursion limit (about 250 terms).
const LONG_CHAIN = 600;

test(
  "a run of a long list chain logs and resumes",
  async () => {
    const flow = await concatFlow(LONG_CHAIN);
    const path = join(dir, "long.jsonl");
    const result = await runFlow(flow, { values: ["v"] }, { env, log: path });
    expect(result.status).toBe("succeeded");
    expect(result.output).toEqual(Array(LONG_CHAIN).fill("v"));
    const replayed = await advance({ env, log: path });
    expect(replayed.status).toBe("succeeded");
    expect(replayed.output).toEqual(result.output);
  },
  SLOW,
);

const SUM_DEF =
  "defs:\n  add:\n    input:\n      x: !int\n    nodes:\n      use:\n        kind: code\n" +
  "        input:\n          total: ${" + sum(CHAIN) + "}\n        output: int\n" +
  "        code: |\n          return inputs.total\n    output: ${use.output}\n";

test(
  "a long chain inside a call runs",
  async () => {
    const text =
      "id: f\nname: f\ninput:\n  x: !int\n" + SUM_DEF +
      "nodes:\n  top:\n    kind: call\n    call: add\n    input:\n      x: ${input.x}\n" +
      "output: ${top.output}\n";
    const result = await runFlow(await loadConformanceFlow(text), { x: 2 }, { env });
    expect(result.status).toBe("succeeded");
    expect(result.output).toBe(BigInt(CHAIN * 2));
  },
  SLOW,
);

test(
  "a long chain inside a map body runs",
  async () => {
    const text =
      "id: f\nname: f\ninput:\n  xs: !list(int)\n" + SUM_DEF +
      "nodes:\n  each:\n    kind: map\n    over: ${input.xs}\n    call: add\n    input:\n      x: ${item}\n" +
      "output: ${each.output}\n";
    const result = await runFlow(await loadConformanceFlow(text), { xs: [1, 3] }, { env });
    expect(result.status).toBe("succeeded");
    expect(result.output).toEqual([BigInt(CHAIN), BigInt(CHAIN * 3)]);
  },
  SLOW,
);

test("a condition at the limit runs thirty calls deep", async () => {
  const condition = upper(99, "input.s") + " == 'A'"; // 99 calls and a comparison
  const lines = ["id: f", "name: f", "input:", "  s: !str", "defs:"];
  for (let i = 0; i < 30; i++) {
    lines.push(`  c${i}:`, "    input:", "      s: !str", "    nodes:");
    if (i < 29) {
      lines.push(
        "      sub:", "        kind: call", `        call: c${i + 1}`, "        input:",
        "          s: ${input.s}", "    output: ${sub.output}",
      );
    } else {
      lines.push(
        "      route:", "        kind: case", "        cases:",
        `          - when: "${condition}"`, "            then: yes", "        else: no",
        "      yes:", "        kind: code", "        output: str", "        code: |",
        "          return 'yes'",
        "      no:", "        kind: code", "        output: str", "        code: |",
        "          return 'no'",
        "    output: ${yes.output | no.output}",
      );
    }
  }
  lines.push(
    "nodes:", "  top:", "    kind: call", "    call: c0", "    input:", "      s: ${input.s}",
    "output: ${top.output}",
  );
  const flow = await loadConformanceFlow(lines.join("\n") + "\n");
  expect((await runFlow(flow, { s: "a" }, { env })).output).toBe("yes");
  expect((await runFlow(flow, { s: "b" }, { env })).output).toBe("no");
});
