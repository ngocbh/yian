/**
 * The question-list type: mirrors `tests/engine/test_human_input_question_type.py`.
 *
 * Ported: `test_question_list_type_structure` and `test_question_type_drives_structured_schema`
 * (the exact schema is pinned in `type_to_schema.test.ts`).
 *
 * Changed: none.
 *
 * Later (T4-11): `test_human_input_with_any_output_stores_a_free_text_answer_as_text`.
 */

import { expect, test } from "bun:test";

import { typeToSchema } from "../../src/nodes/agent/structured.ts";
import { questionListType } from "../../src/nodes/human_input/questions.ts";
import { ValueKind } from "../../src/typesys/values.ts";

test("question list type structure", () => {
  const sh = questionListType();
  expect(sh.kind).toBe(ValueKind.LIST_OBJECT);
  const q = sh.element!;
  expect(q.kind).toBe(ValueKind.OBJECT);
  expect([...q.fields!.keys()]).toEqual(expect.arrayContaining(["question", "header", "options", "multi_select"]));
  expect([...q.required!]).toEqual(expect.arrayContaining(["question", "header"]));
  const opt = q.fields!.get("options")!;
  expect(opt.kind).toBe(ValueKind.LIST_OBJECT);
  expect([...opt.element!.fields!.keys()]).toEqual(expect.arrayContaining(["label", "description"]));
});

test("question type drives structured schema", () => {
  expect(typeToSchema(questionListType())).not.toBeNull();
});
