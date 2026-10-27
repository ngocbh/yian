// A failure inside an inline CALL body is framed at the authored wrapper and at the failing line,
// never under the synthesized flow's id.

import { expect, test } from "bun:test";

import { renderRunError } from "../../src/cli/run.ts";
import type { Writer } from "../../src/cli/utils.ts";
import { INLINE_FLOW_PREFIX } from "../../src/compose/desugar/inline_bodies.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import type { CallNode } from "../../src/nodes/call/node.ts";
import { runFlow } from "../../src/run.ts";

const SOURCE = `id: inline-composite-traceback
name: inline_composite_traceback
nodes:
  wrapper:
    kind: call
    nodes:
      boom:
        kind: code
        code: |
          throw new Error("inline child boom");
          return "unreachable";
        output: str
    output: \${boom.output}
output: \${wrapper.output}
`;

test("inline call runtime failure uses authored friendly frames", async () => {
  const label = "inline-composite.yaml";
  const loaded = await loadFlow(SOURCE, { sourceLabel: label });
  const result = await runFlow(loaded, {});
  expect(result.status).toBe("failed");

  const wrapper = loaded.compiled.nodes.get("wrapper") as CallNode;
  const diagnostic = loaded.diagnose(result.failure!);
  expect(diagnostic.frames.map((frame) => frame.range.startLine)).toEqual([4, 10]);

  const chunks: string[] = [];
  const out: Writer = { write: async (text) => void chunks.push(text) };
  await renderRunError(result, loaded, false, out, () => undefined);
  const output = chunks.join("");

  const outer = `${label}:4`;
  const inner = `${label}:10`;
  expect(output).toContain("source locations:");
  expect(output).toContain(outer);
  expect(output).toContain(inner);
  expect(output.indexOf(outer)).toBeLessThan(output.indexOf(inner));
  expect(output).toContain("inline child boom");
  expect(output).not.toContain(INLINE_FLOW_PREFIX);
  expect(output).not.toContain(wrapper.flowId);
});
