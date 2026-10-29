/**
 * `parseQuestions`, the authoring rules of an `ask`'s question list: option labels are unique
 * within a question, since the answer is the chosen label, while the lists asks already run with
 * (an empty header, label or question text, a label shared by two questions, `multi_select` on a
 * free-text question) parse as before.
 */

import { expect, test } from "bun:test";

import { parseQuestions, QuestionSpecError } from "../../src/nodes/tool/questions.ts";

/** Return what `fn` throws, or fail when it returns. */
function thrown(fn: () => unknown): unknown {
  try {
    fn();
  } catch (error) {
    return error;
  }
  throw new Error("expected a throw");
}

test("a label repeated within a question is refused, naming the question", () => {
  const error = thrown(() =>
    parseQuestions([
      { question: "Ship?", header: "ship", options: [{ label: "yes" }, { label: "no" }] },
      {
        question: "How?",
        header: "how",
        options: [
          { label: "now", description: "deploy" },
          { label: "now", description: "hold the deploy" },
        ],
      },
    ]),
  );

  expect(error).toBeInstanceOf(QuestionSpecError);
  expect((error as Error).message).toBe('question[1] is invalid: option labels must be unique, got ["now","now"]');
});

test("a label shared by two questions parses: each answer is keyed by its own header", () => {
  const questions = [
    { question: "Ship?", header: "ship", options: [{ label: "yes" }] },
    { question: "Tag?", header: "tag", options: [{ label: "yes" }] },
  ];
  expect(parseQuestions(questions).map((q) => q.options.map((option) => option.label))).toEqual([["yes"], ["yes"]]);
});

test("empty text and multi_select on a free-text question parse as before", () => {
  expect(
    parseQuestions([
      { question: "", header: "", options: [{ label: "" }, { label: "no" }] },
      { question: "Notes?", header: "notes", multi_select: true },
    ]),
  ).toEqual([
    {
      question: "",
      header: "",
      options: [
        { label: "", description: "" },
        { label: "no", description: "" },
      ],
      multiSelect: false,
    },
    { question: "Notes?", header: "notes", options: [], multiSelect: true },
  ]);
});
