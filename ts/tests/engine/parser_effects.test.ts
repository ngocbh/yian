/**
 * The effect-node parser arms: a port of `tests/engine/test_parser_effects.py`'s WAIT rows,
 * through `parseNodes` over a virtual cursor.
 *
 * Later (T3-15, the HUMAN_INPUT reader): `test_human_input_missing_prompt_loud` and the
 * HUMAN_INPUT half of `test_parse_human_input_and_wait`.
 */

import { expect, test } from "bun:test";

import { WaitDescriptor } from "../../src/compose/nodes/wait.ts";
import { parseNodes } from "../../src/compose/parse/nodes.ts";
import { SourceCursor } from "../../src/compose/source/index.ts";
import { loadError } from "../compose/declarations.ts";

test("parse human_input and wait: the WAIT half", () => {
  const nodes = parseNodes(SourceCursor.virtual({ settle: { kind: "wait", until: "${input.settle_at}" } }));

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
