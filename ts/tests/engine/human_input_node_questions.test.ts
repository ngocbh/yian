/**
 * The HUMAN_INPUT node's shell and its prompt sites: a port of
 * `tests/engine/test_human_input_node_questions.py`.
 *
 * Ported: `test_template_sites_lists_every_rendered_string` and
 * `test_template_sites_skips_absent_prompt_and_questions`.
 *
 * Changed: none.
 *
 * TS-only: `validatePrompts` names the node and the site of an unparseable span, and names the
 * given path instead of the id; it accepts parseable templates; the node's kind, title, fields
 * and `needsTools` trait read on an instance; `run` rejects as not ported yet.
 *
 * Not ported: none.
 *
 * Later (T4-11): `test_literal_questions_render_and_invoke_host`,
 * `test_questions_input_ref_resolves`, `test_bad_fed_type_raises`,
 * `test_prompt_only_invokes_host` and `test_forward_renders_exactly_the_template_sites`.
 */

import { expect, test } from "bun:test";

import { ExpressionError } from "../../src/expr/expressions.ts";
import { NodeKind } from "../../src/nodes/base.ts";
import { HumanInputNode, siteLabel, templateSites } from "../../src/nodes/human_input/node.ts";

/** Return what `validatePrompts` throws, or fail when it returns. */
function promptError(node: HumanInputNode, options: { path?: string | null } = {}): Error {
  try {
    node.validatePrompts(options);
  } catch (error) {
    expect(error).toBeInstanceOf(ExpressionError);
    return error as Error;
  }
  throw new Error("validatePrompts accepted the node");
}

test("template sites lists every rendered string", () => {
  const questions = [
    {
      question: "Q ${a}",
      header: "H",
      options: [{ label: "L ${b}", description: "D" }, "plain", { label: 3 }],
    },
    "not a mapping",
    { question: 7, header: "H2" },
  ];
  const sites = templateSites("P ${c}", questions);
  expect(sites.map(([path, text]) => [siteLabel(path), text])).toEqual([
    ["prompt", "P ${c}"],
    ["questions[0].question", "Q ${a}"],
    ["questions[0].options[0].label", "L ${b}"],
    ["questions[0].options[0].description", "D"],
  ]);
});

test("template sites skips absent prompt and questions", () => {
  expect(templateSites(null, null)).toEqual([]);
  expect(templateSites("", null)).toEqual([[["prompt"], ""]]);
});

test("validatePrompts names the node and the site of an unparseable span", () => {
  const node = new HumanInputNode("ask", { prompt: "Approve ${x}?", questions: [{ question: "Pick ${" }] });
  expect(promptError(node).message).toStartWith("node 'ask' questions[0].question: ");
  expect(promptError(node, { path: "outer/ask" }).message).toStartWith("node 'outer/ask' questions[0].question: ");
});

test("validatePrompts accepts parseable templates", () => {
  const node = new HumanInputNode("ask", {
    prompt: "Approve ${x}?",
    questions: [{ question: "Pick ${y}", options: [{ label: "${z}", description: "plain" }] }],
  });
  expect(node.validatePrompts()).toBeUndefined();
});

test("a HUMAN_INPUT node's kind, fields and traits", () => {
  const node = new HumanInputNode("ask", { prompt: "P", questionsInput: "qs", title: "Ask" });
  expect(node.kind).toBe(NodeKind.HUMAN_INPUT);
  expect(node.title).toBe("Ask");
  expect(node.prompt).toBe("P");
  expect(node.answerSchema).toEqual([]);
  expect(node.questions).toBeNull();
  expect(node.questionsInput).toBe("qs");
  expect(node.needsTools).toBe(true);
});

test("a HUMAN_INPUT forward is not ported yet", async () => {
  await expect(new HumanInputNode("ask").run({})).rejects.toThrow(
    "node 'ask' (HumanInputNode) forward is not ported yet (T4-11)",
  );
});
