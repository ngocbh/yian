/**
 * HUMAN_INPUT: a blocking call to the run's `ask_user` tool.
 *
 * The node renders and validates its authored question surface, invokes the reserved tool
 * through the trusted metadata path, and returns the handler's answer as its one `Output`. The
 * worker remains occupied until the host answers or raises.
 *
 * Knows about:   nodes/base, nodes/tool/questions, diagnostics, expr, tools and typesys.
 * Never imports: compile, compose, runtime, the run boundary, the CLI, or provider SDKs.
 */

import { reprText } from "../../diagnostics.ts";
import { renderTemplateRecord } from "../../expr/template.ts";
import { ToolSet } from "../../tools/registry.ts";
import { isList, isRecord, type ValueRecord } from "../../typesys/values.ts";
import { checkNamedPrompt, hostOutput, Node, type NodeCaps, NodeKind, Output } from "../base.ts";
import { parseQuestions, type QuestionSpec } from "../tool/questions.ts";

/**
 * A structural gate that blocks its worker until a person answers.
 *
 * On its single run the node invokes the run's `ask_user` tool through the trusted path and
 * returns the answer as this node's `Output`. Tool validation and handler exceptions propagate to
 * the ordinary node-failure boundary.
 *
 * The node carries questions in one of three ways: a literal list baked in (`questions`), a
 * runtime read from a declared input param (`questionsInput`), or neither (the legacy prompt-only
 * form). At most one of the two question sources is set.
 */
export class HumanInputNode extends Node {
  readonly kind = NodeKind.HUMAN_INPUT;
  /** Asks through the run's `ask_user` tool. */
  override readonly needsTools: boolean = true;
  /** The intro text shown to the human; `null` for a questions-only node. */
  prompt: string | null;
  /** Field schema describing the expected answer. */
  answerSchema: readonly Readonly<Record<string, unknown>>[];
  /** A literal question list baked into the node; its text fields are templates. */
  questions: readonly unknown[] | null;
  /** The declared input param the questions list is read from at run time, taken as-is. */
  questionsInput: string | null;

  /**
   * Build a HUMAN_INPUT node.
   *
   * @param nodeId The node's unique id.
   * @param options.prompt The intro text to show the human, rendered against the bound input
   *   record.
   * @param options.answerSchema Field schema describing the expected answer; defaults to `[]`.
   * @param options.questions A literal question list; each question's `question` and each
   *   option's `label`/`description` are `${...}`-rendered against the bound input record.
   * @param options.questionsInput The name of a declared input param to read the questions list
   *   from at run time (not re-templated).
   * @param options.title A display title.
   */
  constructor(
    nodeId: string,
    options: {
      prompt?: string | null;
      answerSchema?: readonly Readonly<Record<string, unknown>>[] | null;
      questions?: readonly unknown[] | null;
      questionsInput?: string | null;
      title?: string | null;
    } = {},
  ) {
    const { prompt = null, answerSchema = null, questions = null, questionsInput = null, title = null } = options;
    super(nodeId, { title });
    this.prompt = prompt;
    this.answerSchema = answerSchema ?? [];
    this.questions = questions;
    this.questionsInput = questionsInput;
  }

  /**
   * Ask the run's `ask_user` handler and return its answer.
   *
   * The prompt is rendered against `inputs`; literal questions have their `question` and each
   * record option's `label`/`description` rendered, while a `questionsInput` list is read as is.
   * Either list is parsed with `parseQuestions`, and the handler receives the parsed specs. The
   * call goes through `caps.tools` (`new ToolSet()`, the global registry, when absent) with
   * `caps.signal`, and passes no attempt, so the handler's context reads attempt 1, also when a
   * resumed run asks again.
   *
   * @throws ExpressionError or RequiredError when a template doesn't render.
   * @throws QuestionSpecError when the question list is invalid, before the handler runs.
   * @throws RangeError when the run has no `ask_user` handler.
   * @throws HostOutputError when the answer isn't a yian value.
   * @throws Whatever the handler throws, unchanged.
   */
  override async forward(inputs: Record<string, unknown>, caps: NodeCaps): Promise<Output> {
    const record = inputs as ValueRecord;
    const prompt = this.prompt === null ? null : renderTemplateRecord(this.prompt, record);
    let questions: QuestionSpec[] | null = null;
    if (this.questions !== null) {
      // `header` is a key and `multi_select` a flag: both are left verbatim.
      questions = parseQuestions(this.questions.map((question) => this.renderQuestion(question, record)));
    } else if (this.questionsInput !== null) {
      // An upstream-produced list, already resolved: taken as is, no templating. Only an own key
      // counts: an omitted optional input is absent, and a name like `constructor` must not read
      // `Object.prototype`.
      const name = this.questionsInput;
      questions = parseQuestions(Object.hasOwn(inputs, name) ? inputs[name] : null);
    }
    const answer = await (caps.tools ?? new ToolSet()).invokeTrusted(
      "ask_user",
      {
        question: prompt || "",
        questions,
        answerSchema: this.answerSchema.length > 0 ? this.answerSchema : null,
        title: this.title,
        nodeId: this.id,
      },
      { node: this.id, signal: caps.signal },
    );
    return new Output(hostOutput(answer));
  }

  /**
   * Check the prompt and each literal question's text templates.
   *
   * @param options.path The static node path to name in the error instead of the node's id, used
   *   by a pre-run check over a whole flow.
   * @throws ExpressionError when a template's `${...}` span doesn't parse, calls `isinstance` or a
   *   function that isn't a builtin, or calls a builtin with arguments it doesn't take; the message
   *   names the node and the site.
   */
  override validatePrompts(options: { path?: string | null } = {}): void {
    const { path = null } = options;
    for (const [sitePath, text] of templateSites(this.prompt, this.questions)) {
      // `||`, as Python's `or`: an empty path names the node by its id.
      checkNamedPrompt(text, `node ${reprText(path || this.id)} ${siteLabel(sitePath)}`);
    }
  }

  /**
   * Render `${...}` in a literal question's `question` text and its options against `record`.
   *
   * Other keys are copied verbatim for `parseQuestions` to validate; an item that isn't a record
   * is returned as is, and `parseQuestions` refuses it.
   */
  private renderQuestion(question: unknown, record: ValueRecord): unknown {
    if (!isRecord(question)) return question;
    const rendered: Record<string, unknown> = { ...question };
    if (typeof rendered.question === "string") rendered.question = renderTemplateRecord(rendered.question, record);
    const options = rendered.options;
    if (isList(options)) rendered.options = options.map((option) => this.renderOption(option, record));
    return rendered;
  }

  /** Render `${...}` in one record option's `label`/`description` against `record`. */
  private renderOption(option: unknown, record: ValueRecord): unknown {
    if (!isRecord(option)) return option;
    const rendered: Record<string, unknown> = { ...option };
    for (const key of ["label", "description"]) {
      const value = rendered[key];
      if (typeof value === "string") rendered[key] = renderTemplateRecord(value, record);
    }
    return rendered;
  }
}

/**
 * Every string a HUMAN_INPUT gate renders as a template, as `[path, text]` pairs.
 *
 * `path` is the key path from the node body: `["prompt"]`, `["questions", i, "question"]`, or
 * `["questions", i, "options", j, "label" | "description"]`. Only string values count, matching
 * the node's `forward`: `header`, `multi_select`, string options, and an upstream
 * `questionsInput` list are not templates. A non-list `questions` and non-record items contribute
 * no sites.
 *
 * @returns The pairs in body order: the prompt, then each question's `question` followed by its
 *   options' `label` and `description`.
 */
export function templateSites(prompt: unknown, questions: unknown): [readonly (string | number)[], string][] {
  const sites: [readonly (string | number)[], string][] = [];
  if (typeof prompt === "string") sites.push([["prompt"], prompt]);
  if (!isList(questions)) return sites;
  questions.forEach((question, i) => {
    if (!isRecord(question)) return;
    if (typeof question.question === "string") sites.push([["questions", i, "question"], question.question]);
    const options = question.options;
    if (!isList(options)) return;
    options.forEach((option, j) => {
      if (!isRecord(option)) return;
      for (const key of ["label", "description"]) {
        const value = option[key];
        if (typeof value === "string") sites.push([["questions", i, "options", j, key], value]);
      }
    });
  });
  return sites;
}

/**
 * A key path written as an author reads it, such as `questions[0].options[1].label`: the string
 * steps (field names) joined with `.`, each number step (a list index) written as `[i]`.
 */
export function siteLabel(path: readonly (string | number)[]): string {
  let label = "";
  for (const step of path) {
    if (typeof step === "number") label += `[${step}]`;
    else label += label ? `.${step}` : step;
  }
  return label;
}
