/**
 * Co-skip, a `:-null` escape and a reference-coalesce join through a real CASE route.
 *
 * `pro` and `con` are the two branches (only one runs); `pro_detail` reads a plain
 * `${pro.output}`, so it co-skips with `pro`; `assemble` joins `${pro.output | con.output}` and
 * `${pro_detail.output:-null}`, which binds null when `pro_detail` co-skips.
 */

import { afterAll, expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { runFlow } from "../../src/run.ts";
import { loadConformanceFlow } from "../helpers/declarations.ts";
import { hostEnv } from "../helpers/mcp_helpers.ts";

const dir = mkdtempSync(join(tmpdir(), "yian-coskip-e2e-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));
const env = hostEnv(dir);

const FLOW = `
id: coskip
name: coskip
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
  pro_detail:
    kind: code
    code: ./fixtures/compose_codefns.ts:detail_of
    input:
      base: \${pro.output}
    output: !str
  assemble:
    kind: code
    code: ./fixtures/compose_codefns.ts:assemble_join
    input:
      claim: \${pro.output | con.output}
      detail: \${pro_detail.output:-null}
    output: !str
output: \${assemble.output}
`;

test("pro route runs detail", async () => {
  const out = await runFlow(await loadConformanceFlow(FLOW, { env }), { score: 0.9, topic: "ACME" }, { env });
  expect(out.status).toBe("succeeded");
  expect(out.output).toBe("pro: ACME|detail=detail: pro: ACME");
});
