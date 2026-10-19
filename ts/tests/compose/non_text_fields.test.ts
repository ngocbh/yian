/**
 * A text or expression field YAML reads as non-text is a load error at that value: a port of
 * `tests/compose/test_non_text_fields.py`. Values are as TS YAML gives them: ints are `bigint`.
 *
 * Dropped: the date rows (TS reads an unquoted date as text) and
 * `test_not_text_detail_reads_the_first_string_error` (`not_text_detail` reads a pydantic error).
 *
 * Later (T3-14, the parse rows): the `parseFile` rows (compact kind, compact id, required id,
 * flow fields, null flow fields, null optional fields, uses ref, flow assert, missing `nodes`)
 * and the `parseNodes` rows (node name, non-text kind, null and unknown kinds, def asserts).
 * Later (T3-15): the `llm_config` rows. Later (T3-22): the `load_flow` rows.
 */

import { expect, test } from "bun:test";

import { notTextError } from "../../src/compose/nodes/base.ts";

test("notTextError names the field", () => {
  expect(notTextError("prompt", 5n)).toBe(
    "`prompt:` must be text, but YAML read it as an integer; put it in quotes",
  );
  expect(notTextError("questions[0].question", [1n])).toBe(
    "`questions[0].question:` must be text, but YAML read it as a list; put it in quotes",
  );
  expect(notTextError("questions[0].header", null)).toBe(
    "`questions[0].header:` must be text, but it is empty; write the text",
  );
});

test("notTextError takes a fix", () => {
  expect(notTextError("kind", [1n], "write one of a, b")).toBe(
    "`kind:` must be text, but YAML read it as a list; write one of a, b",
  );
  expect(notTextError("kind", null, "write one of a, b")).toBe(
    "`kind:` must be text, but it is empty; write one of a, b",
  );
});
