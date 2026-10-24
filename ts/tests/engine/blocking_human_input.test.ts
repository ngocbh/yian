/**
 * HUMAN_INPUT invokes the run's `ask_user` tool and returns its answer: a port of
 * `tests/engine/test_blocking_human_input.py`.
 *
 * TS handlers take `(request, ctx)` and the request is camelCase (`answerSchema`, `nodeId`), each
 * question a parsed spec with camelCase `multiSelect`.
 *
 * Ported: `test_none_answer_is_an_ordinary_output`,
 * `test_questions_input_is_validated_before_trusted_invocation` and
 * `test_host_exception_propagates_unchanged` (`node.run` rejects with the same object; through the
 * engine the failure is `node.execute`).
 *
 * Ported (adapted): `test_prompt_only_returns_host_answer_with_trusted_metadata` runs through
 * `FlowEngine` (the `engine()` fixture, run id `run-123`) on START -> HUMAN_INPUT `ask` -> END, the
 * prompt's `plan` wired as a literal, and also checks the handler's context: `node` `ask`,
 * `attempt` 1, `callId` `run-123/ask`.
 *
 * Changed: `test_literal_questions_are_rendered_validated_and_injected` expects each sent question
 * with `multiSelect: false` where Python's dict has `multi_select: False`.
 *
 * TS-only, through the engine: an answer `5` commits `5n`; a `Date` answer fails
 * `contract.output`; the handler's `ctx.signal` aborts when the run stops (and, directly, follows
 * the node's `caps.signal`). Directly: an upstream question holding `options: null` is refused
 * before the handler runs, as a composer reply holding `null` is; `questionsInput` reads only an
 * own input key, so `constructor` with no such input is refused as `null` before the handler runs.
 */

import { expect, test } from "bun:test";

import { END_ID, FlowOutput, START_ID } from "../../src/compile/model.ts";
import type { Failure } from "../../src/diagnostics.ts";
import { type RunEvent, RunFailed } from "../../src/events.ts";
import { Output } from "../../src/nodes/base.ts";
import { Parameter } from "../../src/nodes/contracts.ts";
import { HumanInputNode } from "../../src/nodes/human_input/node.ts";
import { QuestionSpecError } from "../../src/nodes/tool/questions.ts";
import type { ToolContext } from "../../src/tools/context.ts";
import { type AskHandler, type AskRequest, type AskUserRequest, ToolSet } from "../../src/tools/registry.ts";
import { engine, graph, rejected } from "./run_state_fixtures.ts";

/** Distinct host failure used to prove exception identity is preserved. */
class PromptFailed extends Error {
  override name = "PromptFailed";
}

/**
 * Run START -> `node` -> END through the engine with `handler` as the run's `ask_user`; END
 * outputs `${<id>.output}`. Return the engine, its events and the error `run()` rejected with.
 */
async function runEngine(
  node: HumanInputNode,
  handler: AskHandler,
  options: { wiring?: Readonly<Record<string, Readonly<Record<string, unknown>>>>; signal?: AbortSignal } = {},
) {
  const { wiring, signal } = options;
  const flow = graph([node], [[START_ID, node.id], [node.id, END_ID]], {
    outputs: [new FlowOutput("result", `\${${node.id}.output}`)],
    ...(wiring === undefined ? {} : { wiring }),
  });
  const events: RunEvent[] = [];
  const built = engine(flow, {
    tools: new ToolSet({ ask_user: handler }),
    ...(signal === undefined ? {} : { signal }),
    onEvent: (event) => {
      events.push(event);
    },
  });
  let error: unknown;
  try {
    await built.run();
  } catch (caught) {
    error = caught;
  }
  return { built, events, error };
}

/** The run's failure: the `RunFailed` event's. */
function runFailure(events: readonly RunEvent[]): Failure {
  const last = events.at(-1);
  expect(last).toBeInstanceOf(RunFailed);
  return (last as RunFailed).failure;
}

test("prompt only returns host answer with trusted metadata", async () => {
  const requests: AskRequest[] = [];
  const contexts: ToolContext[] = [];
  const node = new HumanInputNode("ask", {
    prompt: "Approve ${plan}?",
    answerSchema: [{ name: "answer", type: "str", required: true, enum: ["approve", "revise"] }],
    title: "Plan approval",
  });
  node.params = [new Parameter("plan")];

  const { built, error } = await runEngine(
    node,
    (request, ctx) => {
      requests.push(request);
      contexts.push(ctx);
      return "approve";
    },
    { wiring: { ask: { plan: "release" } } },
  );

  expect(error).toBeUndefined();
  expect(built.state.pool.get("ask")).toBe("approve");
  expect(requests).toEqual([
    {
      question: "Approve release?",
      options: null,
      questions: null,
      answerSchema: [{ name: "answer", type: "str", required: true, enum: ["approve", "revise"] }],
      title: "Plan approval",
      nodeId: "ask",
      callId: "run-123/ask",
      deadline: null,
      extra: {},
    },
  ]);
  expect(contexts.map((ctx) => [ctx.node, ctx.attempt, ctx.callId])).toEqual([["ask", 1, "run-123/ask"]]);
});

test("a null answer is an ordinary output", async () => {
  const result = await new HumanInputNode("optional", { prompt: "Anything to add?" }).run(
    {},
    { tools: new ToolSet({ ask_user: () => null }) },
  );

  expect(result).toEqual(new Output(null));
});

test("literal questions are rendered, validated and injected", async () => {
  const requests: AskRequest[] = [];
  const node = new HumanInputNode("choose", {
    prompt: "Choose for ${project}",
    questions: [
      {
        question: "Which framework for ${project}?",
        header: "Framework",
        options: [{ label: "${candidate}", description: "Use ${candidate}" }],
      },
    ],
  });

  const result = await node.run(
    { project: "Atlas", candidate: "Vue" },
    {
      tools: new ToolSet({
        ask_user: (request) => {
          requests.push(request);
          return { Framework: "Vue" };
        },
      }),
    },
  );

  expect(result).toEqual(new Output({ Framework: "Vue" }));
  expect(requests).toEqual([
    {
      question: "Choose for Atlas",
      options: null,
      questions: [
        {
          question: "Which framework for Atlas?",
          header: "Framework",
          options: [{ label: "Vue", description: "Use Vue" }],
          multiSelect: false,
        },
      ],
      answerSchema: null,
      title: null,
      nodeId: "choose",
      // Outside a run each call gets a fresh id.
      callId: expect.any(String),
      deadline: null,
      extra: {},
    },
  ]);
});

test("questions input is validated before trusted invocation", async () => {
  const seen: AskUserRequest[] = [];
  const node = new HumanInputNode("choose", { questionsInput: "questions" });
  const questions = [{ question: "Pick", header: "Choice", options: [{ label: "A" }] }];

  const result = await node.run(
    { questions },
    {
      tools: new ToolSet({
        ask_user: (request) => {
          seen.push(request);
          return { Choice: "A" };
        },
      }),
    },
  );

  expect(result).toEqual(new Output({ Choice: "A" }));
  expect(seen[0]!.questions).toEqual([
    { question: "Pick", header: "Choice", options: [{ label: "A", description: "" }], multiSelect: false },
  ]);
});

test("host exception propagates unchanged", async () => {
  const failure = new PromptFailed("terminal closed");
  const fail = (): never => {
    throw failure;
  };

  const caught = await rejected(
    new HumanInputNode("ask", { prompt: "Question?" }).run({}, { tools: new ToolSet({ ask_user: fail }) }),
  );
  expect(caught).toBe(failure);

  const { events } = await runEngine(new HumanInputNode("ask", { prompt: "Question?" }), fail);
  const runFailed = runFailure(events);
  expect(runFailed.code).toBe("node.execute");
  expect(runFailed.causeType).toBe("PromptFailed");
});

test("an answer 5 commits 5n", async () => {
  const { built, error } = await runEngine(new HumanInputNode("ask", { prompt: "How many?" }), () => 5);

  expect(error).toBeUndefined();
  expect(built.state.pool.get("ask")).toBe(5n);
});

test("a Date answer fails contract.output", async () => {
  const { events } = await runEngine(new HumanInputNode("ask", { prompt: "When?" }), () => ({ at: new Date(0) }));

  const failure = runFailure(events);
  expect(failure.code).toBe("contract.output");
  expect(failure.message).toBe("output at at: Date is not a yian value");
});

test("the handler's signal aborts when the run stops", async () => {
  const host = new AbortController();
  const reason = new Error("stop now");
  let seen: AbortSignal | undefined;

  const { error } = await runEngine(
    new HumanInputNode("ask", { prompt: "Wait for me?" }),
    (_request, ctx) => {
      seen = ctx.signal;
      const answer = new Promise((_resolve, reject) => {
        ctx.signal.addEventListener("abort", () => reject(ctx.signal.reason), { once: true });
      });
      // The host stops the run while the handler waits.
      host.abort(reason);
      return answer;
    },
    { signal: host.signal },
  );

  expect(error).toBe(reason);
  expect(seen?.aborted).toBe(true);

  // Directly, on a tool set bound to no run, the handler's signal is the node's own.
  const controller = new AbortController();
  let direct: AbortSignal | undefined;
  await new HumanInputNode("ask", { prompt: "Wait for me?" }).run(
    {},
    {
      tools: new ToolSet({
        ask_user: (_request, ctx) => {
          direct = ctx.signal;
          return null;
        },
      }),
      signal: controller.signal,
    },
  );
  controller.abort(reason);
  expect(direct?.aborted).toBe(true);
});

test("an upstream question with options null is refused before the handler runs", async () => {
  const calls: AskUserRequest[] = [];
  const node = new HumanInputNode("ask", { questionsInput: "qs" });

  const error = await rejected(
    node.run(
      { qs: [{ question: "Q", header: "H", options: null, multi_select: null }] },
      {
        tools: new ToolSet({
          ask_user: (request) => {
            calls.push(request);
            return null;
          },
        }),
      },
    ),
  );

  expect(error).toBeInstanceOf(QuestionSpecError);
  expect((error as Error).message).toBe("question[0] is invalid: options must be a list, got null");
  expect(calls).toEqual([]);
});

test("questionsInput reads only an own input key", async () => {
  const calls: AskUserRequest[] = [];
  const node = new HumanInputNode("ask", { questionsInput: "constructor" });

  const error = await rejected(
    node.run(
      {},
      {
        tools: new ToolSet({
          ask_user: (request) => {
            calls.push(request);
            return null;
          },
        }),
      },
    ),
  );

  expect(error).toBeInstanceOf(QuestionSpecError);
  expect((error as Error).message).toBe("questions must be a list, got NoneType");
  expect(calls).toEqual([]);
});
