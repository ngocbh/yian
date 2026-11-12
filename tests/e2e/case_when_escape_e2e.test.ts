/**
 * A searched CASE whose `when:` reads a skipped branch under `:-`, `|` and `:?`.
 *
 * `gate` routes on `input.score` to `pro` or `con` (only one runs). `gate2` is a searched CASE
 * whose `when:` reads `pro.output`, routing to `hot` or `cold`. With `score: 0.2`, `pro` is
 * skipped: a plain read skips `gate2` with it, while a read under an escape runs `gate2` and lets
 * the escape supply the value, as a binding of that escape would. The `when:` stays lazy: a later
 * case's `:?` or call never runs when an earlier case matches.
 */

import { afterAll, expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { type RunResult, runFlow } from "../../src/run.ts";
import { assembleText, loadConformanceFlow, loadError, loadErrorAsync } from "../helpers/declarations.ts";
import { hostEnv } from "../helpers/mcp_helpers.ts";
import { outputTag } from "../helpers/output_tags.ts";

const dir = mkdtempSync(join(tmpdir(), "yian-case-when-escape-e2e-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));
const env = hostEnv(dir);

const FLOW = `
id: when_escape
name: when_escape
input:
  score: !float
  topic: !str
nodes:
  gate:
    kind: case
    cases:
      - when: "input.score >= 0.5"
        then: pro
    else: con
  pro:
    kind: code
    code: ./fixtures/compose_codefns.ts:pick_pro
    input:
      topic: \${input.topic}
    output: !str
  con:
    kind: code
    code: ./fixtures/compose_codefns.ts:pick_con
    input:
      topic: \${input.topic}
    output: !str
  gate2:
    kind: case
    cases:
<CASES>
    else: cold
  hot:
    kind: code
    code: ./fixtures/compose_codefns.ts:positive
    input:
      topic: \${input.topic}
    output: !str
  cold:
    kind: code
    code: ./fixtures/compose_codefns.ts:cautious
    input:
      topic: \${input.topic}
    output: !str
output: \${gate2.output}
`;

const HOT = "pro case for ACME";
const COLD = "cautious note for ACME";

/** `FLOW` with `gate2`'s `[when, then]` cases spliced in. */
function flow(...cases: [string, string][]): string {
  const body = cases.map(([when, then]) => `      - when: ${when}\n        then: ${then}\n`).join("");
  return FLOW.replace("<CASES>\n", body);
}

/** Load and run the flow with `cases`, on a score that skips `pro` unless raised. */
async function run(cases: [string, string][], score = 0.2): Promise<RunResult> {
  return runFlow(await loadConformanceFlow(flow(...cases), { env }), { score, topic: "ACME" }, { env });
}

test("a literal coalesce runs the gate on a skipped branch", async () => {
  const out = await run([[`'(pro.output | "none") == "none"'`, "hot"]]);
  expect(out.status).toBe("succeeded");
  expect(out.output).toBe(HOT);
});

test("a required read fails the gate on a skipped branch", async () => {
  const out = await run([[`'(pro.output :? "pro must run") == "x"'`, "hot"]]);
  expect(out.status).toBe("failed");
  expect(out.failure!.code).toBe("condition.evaluate");
  expect(out.failure!.message).toBe("pro must run");
  expect(out.failure!.nodeId).toBe("gate2");
  expect(out.failure!.causeType).toBe("RequiredError");
  expect(out.failure!.trace[0]!.key).toEndWith("/s:case/i:0/s:when");
});

test("a coalesce of two branches reads the one that ran", async () => {
  const out = await run([[`'(pro.output | con.output) == "con: ACME"'`, "hot"]]);
  expect(out.status).toBe("succeeded");
  expect(out.output).toBe(HOT);
});

test("a call on a skipped branch in a later case is not evaluated", async () => {
  // `upper` of the int default would fail; passing `pro.output` bare is a load error.
  const out = await run([
    [`"input.score < 0.5"`, "hot"],
    [`'upper(pro.output :- 1) == "CON: ACME"'`, "cold"],
  ]);
  expect(out.status).toBe("succeeded");
  expect(out.output).toBe(HOT);
});

test("a default reads the branch when it ran", async () => {
  const out = await run([[`'(pro.output :- "none") == "none"'`, "hot"]], 0.9);
  expect(out.status).toBe("succeeded");
  expect(out.output).toBe(COLD);
});

test.each([
  [`"(nope.output | pro.output) == 1"`, "${nope.output}"],
  [`"(pro.output :- nope.output.x) == 1"`, "${nope.output.x}"],
])("an unknown namespace inside a co skipping escape is reported once: %s", async (when, reference) => {
  const error = await loadErrorAsync(loadConformanceFlow(flow([when, "hot"]), { env }));
  expect(error.message).toBe(
    "flow has unresolved references:\n" +
      `  node 'gate2' cases[0].when from: reference ${reference} uses unknown namespace 'nope'`,
  );
});

// ---------- nullable field reads in a searched when: ----------

/** A flow where `route` may skip `x` (`Rec`), and searched CASE `gate` routes on `when`. */
const skippableWhen = (when: string) => `id: w
name: w
typedefs:
  Rec: {k: int}
input:
  go: !bool
nodes:
  route:
    kind: case
    cases:
      - when: \${input.go}
        then: x
  x:
    kind: code
    output: !Rec
    code: "return {'k': 2}"
  gate:
    kind: case
    cases:
      - when: "${when}"
        then: hot
    else: cold
  hot:
    kind: code
    output: !str
    code: return 'hot'
  cold:
    kind: code
    output: !str
    code: return 'cold'
output: \${hot.output | cold.output}
`;

test("a field read inside an escape, of a node that may be skipped, needs ?.", () => {
  // The root is read only inside the escape, so `gate` runs while `x` is skipped; `:-` doesn't
  // catch the read of a field of null.
  const error = loadError(() => assembleText(skippableWhen("(x.output.k :- 0) > 1")));
  expect(error.diagnostic.code).toBe("expr.nullable_field_read");
  expect(error.message).toStartWith("node 'gate' cases[0].when: ");
  expect(error.message).toContain(
    "${x.output.k} reads field 'k' of x.output, which may be null (node 'x' may be skipped while 'gate' runs); " +
      "write x.output?.k, or check x.output != null first",
  );
  expect(error.message).not.toContain("__r");
});

test.each(["(x.output?.k :- 0) > 1", "x.output != null and x.output.k > 1", "x.output.k > 1"])(
  "a searched when: loads: %s",
  (when) => {
    assembleText(skippableWhen(when));
  },
);

test("an unknown field in a searched when: names the authored read", () => {
  const message = loadError(() => assembleText(skippableWhen("x.output.nosuch > 1"))).message;
  expect(message).toContain("reference ${x.output.nosuch} reads unknown field 'nosuch'");
  expect(message).not.toContain("__r");
});

// --- declared fields at run time (#1149) --- //

/** A searched CASE `pick` on `when` over CODE `x`, typed `output` (or untyped), returning `value`. */
function pickFlow(when: string, output: string | null, value: string): string {
  return `
id: pick
name: pick
typedefs:
  Rec: {k: int, opt: Optional(str)}
nodes:
  x:
    kind: code
${output !== null ? `    output: ${outputTag(output)}\n` : ""}    code: |
      return ${value}
  pick:
    kind: case
    cases:
      - when: "${when}"
        then: hit
    else: miss
  hit:
    kind: code
    code: |
      return "hit"
  miss:
    kind: code
    code: |
      return "miss"
output: \${hit.output | miss.output}
`;
}

test.each<[string, string, string | null, string, string | null, string | null]>([
  ["a declared absent field reads null through __r0", "x.output.opt == null", "Rec", "{k: 2}", "hit", null],
  ["an untyped absent key raises", "x.output.opt == null", null, "{k: 2}", null, "condition.evaluate"],
  ["a null guard keeps the read lazy", "x.output != null and x.output.k > 1", "Optional(Rec)", "null", "miss", null],
  ["?. on a null producer reads null", "x.output?.k > 1", "Optional(Rec)", "null", "miss", null],
])("%s", async (_name, when, output, value, expected, code) => {
  const result = await runFlow(await loadConformanceFlow(pickFlow(when, output, value), { env }), {}, { env });
  if (code !== null) {
    expect(result.failure?.code).toBe(code as never);
    expect(result.failure?.message).toBe("cannot read ${x.output.opt}: x.output has no field 'opt'");
  } else {
    expect(result.failure?.message).toBeUndefined();
    expect(result.output).toBe(expected);
  }
});
