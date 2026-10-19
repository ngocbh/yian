/**
 * The effect-node parser arms: a port of `tests/engine/test_parser_effects.py`, through
 * `parseNodes` over a virtual cursor.
 */

import { expect, test } from "bun:test";

import { HumanInputDescriptor } from "../../src/compose/nodes/human_input.ts";
import { WaitDescriptor } from "../../src/compose/nodes/wait.ts";
import { parseNodes } from "../../src/compose/parse/nodes.ts";
import { SourceCursor } from "../../src/compose/source/index.ts";
import { loadError } from "../compose/declarations.ts";

test("parse human_input and wait", () => {
  const nodes = parseNodes(
    SourceCursor.virtual({
      approve: { kind: "human_input", prompt: "ok? ${a}", inputs: { a: "${p.output}" }, outputs: "Approval" },
      settle: { kind: "wait", until: "${input.settle_at}" },
    }),
  );

  const approve = nodes.get("approve");
  expect(approve).toBeInstanceOf(HumanInputDescriptor);
  expect((approve as HumanInputDescriptor).prompt).toBe("ok? ${a}");
  const settle = nodes.get("settle");
  expect(settle).toBeInstanceOf(WaitDescriptor);
  expect((settle as WaitDescriptor).until).toBe("${input.settle_at}");
});

test("wait rejects inputs field", () => {
  const error = loadError(() =>
    parseNodes(SourceCursor.virtual({ w: { kind: "wait", until: "${input.x}", inputs: { a: "1" } } })),
  );

  expect(error.message).toContain("field 'inputs' is not allowed");
});

test("human_input missing prompt loud", () => {
  loadError(() => parseNodes(SourceCursor.virtual({ h: { kind: "human_input" } })));
});
