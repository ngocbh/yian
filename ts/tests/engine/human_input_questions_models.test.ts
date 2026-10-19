/**
 * The question specs and their parser: mirrors `tests/engine/test_human_input_questions_models.py`.
 *
 * Ported: `test_parse_questions_minimal_single_select`,
 * `test_parse_questions_multiselect_and_freetext_only` and
 * `test_parse_questions_rejects_empty_too_many_dups_badtype` (each rejection also checks the
 * message).
 *
 * Changed: `multi_select` accepts only `true`/`false`, where pydantic's lax mode also turns
 * "yes", "true", "on", 1 and 0 into booleans; a bad item reads `question[<i>] is invalid: ` and
 * the first problem found, where Python dumps the pydantic validation error.
 *
 * TS-only: one row per item message, `null` given for `options`, `multi_select` or an option's
 * `description`, and the frozen result.
 */

import { expect, test } from "bun:test";

import { parseQuestions, QuestionSpecError } from "../../src/nodes/human_input/questions.ts";

/** Expect `parseQuestions(raw)` to throw a `QuestionSpecError` reading `message`. */
function expectRejected(raw: unknown, message: string): void {
  let caught: unknown;
  try {
    parseQuestions(raw);
  } catch (error) {
    caught = error;
  }
  expect(caught).toBeInstanceOf(QuestionSpecError);
  expect((caught as QuestionSpecError).name).toBe("QuestionSpecError");
  expect((caught as QuestionSpecError).message).toBe(message);
}

test("parse questions minimal single select", () => {
  const raw = [
    {
      question: "Which framework?",
      header: "Framework",
      options: [
        { label: "React", description: "UI lib" },
        { label: "Vue", description: "progressive" },
      ],
    },
  ];
  const qs = parseQuestions(raw);
  expect(qs[0]!.header).toBe("Framework");
  expect(qs[0]!.multiSelect).toBe(false);
  expect(qs[0]!.options.map((o) => o.label)).toEqual(["React", "Vue"]);
  expect(qs[0]!.options[0]!.description).toBe("UI lib");
});

test("parse questions multiselect and freetext only", () => {
  const raw = [
    { question: "Pick areas", header: "Areas", multi_select: true, options: [{ label: "API" }, { label: "UI" }] },
    { question: "Anything else?", header: "Notes" },
  ];
  const qs = parseQuestions(raw);
  expect(qs[0]!.multiSelect).toBe(true);
  expect(qs[0]!.options[0]!.description).toBe("");
  expect(qs[1]!.options).toEqual([]);
  expect(qs).toStrictEqual([
    {
      question: "Pick areas",
      header: "Areas",
      options: [
        { label: "API", description: "" },
        { label: "UI", description: "" },
      ],
      multiSelect: true,
    },
    { question: "Anything else?", header: "Notes", options: [], multiSelect: false },
  ]);
});

test("parse questions rejects empty too many dups badtype", () => {
  expectRejected([], "expected 1..4 questions, got 0");
  expectRejected(
    [0, 1, 2, 3, 4].map((i) => ({ question: `q${i}`, header: `H${i}` })),
    "expected 1..4 questions, got 5",
  );
  expectRejected(
    [
      { question: "a", header: "D" },
      { question: "b", header: "D" },
    ],
    'question headers must be unique, got ["D","D"]',
  );
  expectRejected([{ question: "no header" }], 'question[0] is invalid: missing key(s) ["header"]');
  expectRejected("nope", "questions must be a list, got str");
});

const ITEM_ROWS: [string, unknown, string][] = [
  ["not a record", "nope", 'must be a record, got "nope"'],
  ["a list item", [1n], "must be a record, got [1]"],
  ["unknown keys, sorted", { question: "q", header: "h", zeta: 1n, alpha: 1n }, 'unknown key(s) ["alpha","zeta"]'],
  ["unknown key before missing keys", { extra: 1n }, 'unknown key(s) ["extra"]'],
  ["missing both keys, in order", {}, 'missing key(s) ["question","header"]'],
  ["missing question", { header: "h" }, 'missing key(s) ["question"]'],
  ["question not text", { question: 1n, header: 2n }, "question must be text, got 1"],
  ["header not text", { question: "q", header: null }, "header must be text, got null"],
  ["options not a list", { question: "q", header: "h", options: "a" }, 'options must be a list, got "a"'],
  ["options null", { question: "q", header: "h", options: null }, "options must be a list, got null"],
  [
    "an option not a record",
    { question: "q", header: "h", options: [{ label: "a" }, "b"] },
    'options[1] must be a record, got "b"',
  ],
  [
    "an option's unknown keys",
    { question: "q", header: "h", options: [{ label: "a", value: 1n, extra: 2n }] },
    'options[0]: unknown key(s) ["extra","value"]',
  ],
  [
    "an option's missing label",
    { question: "q", header: "h", options: [{ description: "d" }] },
    'options[0]: missing key(s) ["label"]',
  ],
  [
    "an option's label not text",
    { question: "q", header: "h", options: [{ label: 1.5 }] },
    "options[0].label must be text, got 1.5",
  ],
  [
    "an option's description null",
    { question: "q", header: "h", options: [{ label: "a", description: null }] },
    "options[0].description must be text, got null",
  ],
  [
    "multi_select null",
    { question: "q", header: "h", multi_select: null },
    "multi_select must be true or false, got null",
  ],
  [
    "multi_select yes",
    { question: "q", header: "h", multi_select: "yes" },
    'multi_select must be true or false, got "yes"',
  ],
  ["multi_select 1", { question: "q", header: "h", multi_select: 1n }, "multi_select must be true or false, got 1"],
  [
    "options checked before multi_select",
    { question: "q", header: "h", options: 1n, multi_select: 1n },
    "options must be a list, got 1",
  ],
];

for (const [name, item, detail] of ITEM_ROWS) {
  test(`item message: ${name}`, () => {
    expectRejected([{ question: "first", header: "first" }, item], `question[1] is invalid: ${detail}`);
  });
}

test("the parsed questions are frozen", () => {
  const [spec] = parseQuestions([{ question: "q", header: "h", options: [{ label: "a" }] }]);
  expect(Object.isFrozen(spec)).toBe(true);
  expect(Object.isFrozen(spec!.options)).toBe(true);
  expect(Object.isFrozen(spec!.options[0])).toBe(true);
});
