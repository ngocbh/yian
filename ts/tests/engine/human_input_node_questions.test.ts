/**
 * The HUMAN_INPUT gate: a port of `tests/engine/test_human_input_node_questions.py`.
 *
 * TS handlers take `(request, ctx)` and the request is camelCase, each question a parsed spec with
 * camelCase `multiSelect`.
 *
 * Ported: `test_literal_questions_render_and_invoke_host`, `test_questions_input_ref_resolves`,
 * `test_bad_fed_type_raises`, `test_prompt_only_invokes_host`,
 * `test_template_sites_lists_every_rendered_string` and
 * `test_template_sites_skips_absent_prompt_and_questions`.
 *
 * Ported (adapted): `test_forward_renders_exactly_the_template_sites`: the question also sets
 * `multi_select: true`, which reaches the handler as `multiSelect: true`. A string option, the
 * other non-site the sites skip, can't reach the handler: `parseQuestions` refuses it.
 *
 * Changed: a literal question that isn't a record is passed to `parseQuestions`, which refuses it
 * (`QuestionSpecError`), where Python's `dict(question)` raises; both are `node.execute`, and the
 * loader refuses such a list, so neither is reachable from YAML.
 *
 * TS-only: `validatePrompts` names the node and the site of an unparseable span, and names the
 * given path instead of the id; it accepts parseable templates; the node's kind, title, fields
 * and `needsTools` trait read on an instance; a non-record literal question is refused.
 *
 * Not ported: none.
 */

import { expect, test } from "bun:test";

import { ExpressionError } from "../../src/expr/expressions.ts";
import { NodeKind, Output } from "../../src/nodes/base.ts";
import { HumanInputNode, siteLabel, templateSites } from "../../src/nodes/human_input/node.ts";
import { QuestionSpecError } from "../../src/nodes/human_input/questions.ts";
import { type AskUserRequest, ToolSet } from "../../src/tools/registry.ts";
import { rejected } from "./run_state_fixtures.ts";

/** A tool set whose `ask_user` records each request in `requests` and returns `answer`. */
function recording(requests: AskUserRequest[], answer: unknown): ToolSet {
  return new ToolSet({
    ask_user: (request) => {
      requests.push(request);
      return answer;
    },
  });
}

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

test("literal questions render and invoke host", async () => {
  const node = new HumanInputNode("ask", {
    questions: [{ question: "Pick ${x}", header: "H", options: [{ label: "A" }] }],
  });
  const requests: AskUserRequest[] = [];

  const result = await node.run({ x: "framework" }, { tools: recording(requests, { H: "A" }) });

  expect(result).toEqual(new Output({ H: "A" }));
  expect(requests[0]!.questions).toHaveLength(1);
  const q = requests[0]!.questions![0] as Record<string, unknown>;
  expect(q.question).toBe("Pick framework");
  expect(q.header).toBe("H");
});

test("questions input ref resolves", async () => {
  const node = new HumanInputNode("ask", { questionsInput: "qs" });
  const requests: AskUserRequest[] = [];

  const result = await node.run(
    { qs: [{ question: "Q", header: "H", options: [{ label: "A" }] }] },
    { tools: recording(requests, { H: "A" }) },
  );

  expect(result).toEqual(new Output({ H: "A" }));
  expect(requests[0]!.questions).toHaveLength(1);
  expect((requests[0]!.questions![0] as Record<string, unknown>).question).toBe("Q");
});

test("bad fed type raises", async () => {
  const node = new HumanInputNode("ask", { questionsInput: "qs" });

  // An empty list: parseQuestions rejects it before invoking the host.
  const error = await rejected(node.run({ qs: [] }, { tools: new ToolSet({ ask_user: () => null }) }));

  expect(error).toBeInstanceOf(QuestionSpecError);
});

test("prompt only invokes host", async () => {
  const node = new HumanInputNode("ask", { prompt: "Approve ${x}?" });
  const requests: AskUserRequest[] = [];

  const result = await node.run({ x: "deploy" }, { tools: recording(requests, "yes") });

  expect(result).toEqual(new Output("yes"));
  expect(requests[0]!.question).toBe("Approve deploy?");
  expect(requests[0]!.questions).toBeNull();
});

test("forward renders exactly the template sites", async () => {
  // Every string field carries `${m}`; `forward` must render the template sites and leave every
  // other string verbatim, so the load check and the renderer agree.
  const questions = [
    {
      question: "${m}",
      header: "${m}",
      options: [
        { label: "${m}", description: "${m}" },
        { label: "${m}", description: "${m}" },
      ],
      multi_select: true,
    },
  ];
  const node = new HumanInputNode("ask", { prompt: "${m}", questions });
  const requests: AskUserRequest[] = [];

  await node.run({ m: "X" }, { tools: recording(requests, {}) });

  const sent = requests[0]!;
  const rendered = new Set(templateSites("${m}", questions).map(([path]) => siteLabel(path)));
  const authoredPaths: (string | number)[][] = [["prompt"]];
  questions.forEach((question, i) => {
    authoredPaths.push(["questions", i, "question"], ["questions", i, "header"]);
    question.options.forEach((_option, j) => {
      authoredPaths.push(["questions", i, "options", j, "label"], ["questions", i, "options", j, "description"]);
    });
  });
  // The prompt is sent under `question`; question fields keep their authored path.
  const sentValue = (path: (string | number)[]): unknown => {
    if (path.length === 1 && path[0] === "prompt") return sent.question;
    let value: unknown = sent;
    for (const step of path) value = (value as Record<string | number, unknown>)[step];
    return value;
  };
  for (const path of authoredPaths) {
    const label = siteLabel(path);
    expect([label, sentValue(path)]).toEqual([label, rendered.has(label) ? "X" : "${m}"]);
  }
  expect((sent.questions![0] as Record<string, unknown>).multiSelect).toBe(true);
});

test("a literal question that isn't a record is refused", async () => {
  const node = new HumanInputNode("ask", { questions: ["Pick one"] });

  const error = await rejected(node.run({}, { tools: new ToolSet({ ask_user: () => null }) }));

  expect(error).toBeInstanceOf(QuestionSpecError);
});
