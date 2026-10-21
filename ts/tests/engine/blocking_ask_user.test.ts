/**
 * An AGENT keeps blocking human interaction inside one live model/tool loop: a port of
 * `tests/engine/test_blocking_ask_user.py`.
 *
 * The rows evaluate an AGENT through `evalNode` (`drive`) or `node.run`, its model a
 * `ScriptedModel` from `models` where Python patches `model_from_config`, and the `ask_user`
 * handler in the run's `ToolSet`. A host stop is the node's aborted signal: a stopped node rejects
 * with the signal's reason where Python raises `_HostStop`. The `ask_user` request is camelCase.
 *
 * Ported: `test_ask_user_answer_is_a_tool_message_in_the_same_loop`,
 * `test_model_options_reach_the_handler_and_the_answer_returns`,
 * `test_two_blocking_questions_keep_one_conversation`,
 * `test_model_tool_calls_run_in_exact_returned_order`,
 * `test_ask_user_requires_an_explicit_authored_grant`,
 * `test_ask_user_exception_propagates_from_the_live_node` and
 * `test_post_assert_fires_after_blocking_ask`.
 *
 * Changed: `test_agent_resolves_model_factory_at_call_time` checks that the node builds its model
 * through `models` when it runs, not before (the node has an `llmConfig`: TS refuses an AGENT
 * with no provider). `test_stop_during_model_call_prevents_returned_tool_side_effect`: the model
 * aborts the signal during its call, then returns the tool call. `test_stop_during_tool_resolution_prevents_model_binding`:
 * the `ToolSet`'s `resolve` aborts the signal (patched on the instance), and no request reaches
 * the model.
 * `test_stop_during_structured_attempt_prevents_retry`: the prompted structured plan (`vllm`)
 * with retries, the model aborting the signal and answering text that doesn't parse.
 * `test_structured_output_succeeds_after_blocking_ask`: the structured call after the loop (the
 * native `json_schema` plan) answers the value's JSON text, where Python's LangChain carrier
 * returns the validated model.
 *
 * Not ported: `test_agent_has_no_continuation_or_engine_llm_cap_surface` reads Python's module
 * attributes and signatures.
 */

import { afterEach, beforeEach, expect, test } from "bun:test";

import { BUILTIN, forget, install } from "../../src/defaults.ts";
import { type ChatModel, type Event, EventStream } from "../../src/llm/events.ts";
import { Message, type Request, Response, TextPart, ToolCallPart, ToolResultPart } from "../../src/llm/messages.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { AgentNode } from "../../src/nodes/agent/node.ts";
import { Output } from "../../src/nodes/base.ts";
import { Assertion, ValueContract } from "../../src/nodes/contracts.ts";
import { EvaluationFailure } from "../../src/runtime/eval_node.ts";
import { FunctionTool, tool } from "../../src/tools/function_tools.ts";
import { type AskUserHandler, type AskUserRequest, ToolSet } from "../../src/tools/registry.ts";
import type { JsonObject } from "../../src/typesys/json.ts";
import { Type, ValueKind } from "../../src/typesys/values.ts";
import { drive, rejected, serving, textReply } from "./run_state_fixtures.ts";

// Start each test from the built-in values, whatever `YIAN_*` variables the shell sets.
beforeEach(() => install(BUILTIN));
afterEach(forget);

const CONFIG = { provider: "openai", model: "gpt-5.5" };

/** One model reply calling `calls` (`[name, args, id]`). */
function callReply(...calls: [string, JsonObject, string][]): Response {
  const parts = calls.map(([name, args, id]) => new ToolCallPart({ id, name, arguments: args }));
  return new Response({ message: new Message({ role: "assistant", parts }), finishReason: "tool_calls" });
}

/** A scripted model answering `replies` under `CONFIG`. */
const scripted = (replies: Response[]) => new ScriptedModel(replies, CONFIG);

/** An AGENT `agent` with `prompt: go` over `CONFIG`, granted `tools`. */
const agent = (tools: string[] = ["ask_user"]) => new AgentNode("agent", { prompt: "go", tools, llmConfig: CONFIG });

/** Evaluate `node` over `model`, with `askUser` as the run's handler when given. */
function evaluate(node: AgentNode, model: ChatModel, askUser?: AskUserHandler, signal?: AbortSignal) {
  return drive(node, {
    models: serving(model),
    ...(askUser === undefined ? {} : { tools: new ToolSet({ ask_user: askUser }) }),
    ...(signal === undefined ? {} : { signal }),
  });
}

/** The tool results of the last message of `request`, as `[callId, content]`. */
const toolResults = (request: Request) =>
  request.messages
    .flatMap((message) => message.parts)
    .filter((part) => part instanceof ToolResultPart)
    .map((part) => [(part as ToolResultPart).callId, (part as ToolResultPart).content]);

/** The text of the first part of `message`. */
const textOf = (message: Message) => (message.parts[0] as TextPart).text;

test("agent resolves its model at call time", async () => {
  const model = scripted([textReply("late-bound")]);
  const access = serving(model);
  const node = new AgentNode("agent", { prompt: "go", llmConfig: CONFIG });
  expect(access.built).toBe(0);

  const outcome = await drive(node, { models: access });

  expect(outcome).toEqual(new Output("late-bound"));
  expect(access.built).toBe(1);
});

test("ask_user answer is a tool message in the same loop", async () => {
  const model = scripted([callReply(["ask_user", { question: "Approve?" }, "question-1"]), textReply("approved result")]);
  const prompts: AskUserRequest[] = [];

  const outcome = await evaluate(agent(), model, (request) => {
    prompts.push(request);
    return "yes";
  });

  expect(outcome).toEqual(new Output("approved result"));
  expect(model.requests[0]!.tools.map((spec) => spec.name)).toEqual(["ask_user"]);
  expect(textOf(model.requests[0]!.messages[0]!)).toContain("Ask the user only through an explicitly granted tool");
  expect(prompts).toEqual([
    { question: "Approve?", questions: null, answerSchema: null, title: null, nodeId: null, options: null },
  ]);
  const last = model.requests[1]!.messages.at(-1)!;
  expect(last.parts).toHaveLength(1);
  expect(last.parts[0]).toBeInstanceOf(ToolResultPart);
  expect((last.parts[0] as ToolResultPart).content).toBe("yes");
  expect((last.parts[0] as ToolResultPart).callId).toBe("question-1");
});

test("model options reach the handler and the answer returns", async () => {
  const model = scripted([
    callReply(["ask_user", { question: "Which?", options: '["Retry", "Give up"]' }, "question-1"]),
    textReply("done"),
  ]);
  const options: unknown[] = [];

  const outcome = await evaluate(agent(), model, (request) => {
    options.push(request.options);
    return "Retry";
  });

  expect(outcome).toEqual(new Output("done"));
  expect(options).toEqual([["Retry", "Give up"]]);
  expect(toolResults(model.requests[1]!).at(-1)).toEqual(["question-1", "Retry"]);
});

test("two blocking questions keep one conversation", async () => {
  const model = scripted([
    callReply(["ask_user", { question: "First?" }, "question-1"]),
    callReply(["ask_user", { question: "Second?" }, "question-2"]),
    textReply("complete"),
  ]);
  const answers = ["one", "two"];
  const questions: string[] = [];

  const outcome = await evaluate(agent(), model, (request) => {
    questions.push(request.question);
    return answers.shift();
  });

  expect(outcome).toEqual(new Output("complete"));
  expect(questions).toEqual(["First?", "Second?"]);
  expect(toolResults(model.requests.at(-1)!)).toEqual([
    ["question-1", "one"],
    ["question-2", "two"],
  ]);
});

test("model tool calls run in exact returned order", async () => {
  const seen: [string, unknown][] = [];
  /** A tool `name` recording its call and returning `reply`. */
  const recording = (name: string, reply: string) =>
    tool({
      name,
      description: `Record the ${name} tool call.`,
      parameters: { type: "object", properties: { value: { type: "integer" } }, required: ["value"] },
      execute: (args) => {
        seen.push([name, args.value]);
        return reply;
      },
    });
  const model = scripted([
    callReply(["second", { value: 2 }, "2"], ["ask_user", { question: "Between tools?" }, "human"], ["first", { value: 1 }, "1"]),
    textReply("done"),
  ]);
  const askUser: AskUserHandler = (request) => {
    seen.push(["ask_user", request.question]);
    return "human answer";
  };

  const outcome = await drive(agent(["first", "ask_user", "second"]), {
    models: serving(model),
    tools: new ToolSet({ first: recording("first", "one"), second: recording("second", "two"), ask_user: askUser }),
  });

  expect(outcome).toEqual(new Output("done"));
  expect(seen).toEqual([
    ["second", 2],
    ["ask_user", "Between tools?"],
    ["first", 1],
  ]);
});

test("a stop during the model call prevents the returned tool's side effect", async () => {
  const stop = new AbortController();
  const effects: unknown[] = [];
  const mutate = tool({
    name: "mutate_after_stop",
    description: "Record one externally visible effect.",
    parameters: { type: "object", properties: { value: { type: "string" } }, required: ["value"] },
    execute: (args) => {
      effects.push(args.value);
      return "ok";
    },
  });
  const requests: Request[] = [];
  // The call aborts the stop, then returns a tool call anyway.
  const model: ChatModel = {
    ...CONFIG,
    stream: (request) => {
      requests.push(request);
      const reply = callReply(["mutate_after_stop", { value: "ran" }, "late-call"]);
      return new EventStream(
        (async function* (): AsyncGenerator<Event> {
          stop.abort(new Error("stopped"));
          yield* scripted([reply]).stream(request);
        })(),
      );
    },
  };

  const error = await rejected(
    drive(agent(["mutate_after_stop"]), {
      models: serving(model),
      tools: new ToolSet({ mutate_after_stop: mutate }),
      signal: stop.signal,
    }),
  );

  expect(error).toBe(stop.signal.reason);
  expect(effects).toEqual([]);
  expect(requests).toHaveLength(1);
});

test("a stop during tool resolution prevents the model call", async () => {
  const stop = new AbortController();
  const model = scripted([textReply("unreached")]);
  const probe = new FunctionTool({
    name: "tool",
    description: "Do nothing.",
    parametersJsonSchema: { type: "object", properties: {} },
    execute: () => "ok",
  });
  // The evaluator takes only an exact `ToolSet`, so the instance's `resolve` aborts the stop.
  const tools = new ToolSet({ tool: probe });
  const resolve = tools.resolve.bind(tools);
  tools.resolve = (ids) => {
    stop.abort(new Error("stopped"));
    return resolve(ids);
  };

  const error = await rejected(drive(agent(["tool"]), { models: serving(model), tools, signal: stop.signal }));

  expect(error).toBe(stop.signal.reason);
  expect(model.requests).toEqual([]);
});

test("a stop during a structured attempt prevents the retry", async () => {
  const stop = new AbortController();
  const prompted = { provider: "vllm", model: "no-structured-sentinel" };
  const attempts: Request[] = [];
  const model: ChatModel = {
    ...prompted,
    stream: (request) => {
      attempts.push(request);
      return new EventStream(
        (async function* (): AsyncGenerator<Event> {
          stop.abort(new Error("stopped"));
          yield* new ScriptedModel([textReply("not json")], prompted).stream(request);
        })(),
      );
    },
  };
  const node = new AgentNode("agent", { prompt: "go", llmConfig: prompted, retries: 2 });
  node.outputContract = new ValueContract(
    new Type({
      kind: ValueKind.OBJECT,
      fields: new Map([["answer", Type.scalar(ValueKind.STRING)]]),
      required: new Set(["answer"]),
    }),
  );

  const error = await rejected(drive(node, { models: serving(model), signal: stop.signal }));

  expect(error).toBe(stop.signal.reason);
  expect(attempts).toHaveLength(1);
});

test("ask_user requires an explicit authored grant", async () => {
  const model = scripted([callReply(["ask_user", { question: "Secret?" }, "question-1"])]);
  const handled: AskUserRequest[] = [];

  const outcome = await evaluate(agent([]), model, (request) => {
    handled.push(request);
    return "no";
  });

  expect(outcome).toBeInstanceOf(EvaluationFailure);
  expect((outcome as EvaluationFailure).failure.message).toContain("not granted");
  expect(handled).toEqual([]);
});

test("an ask_user error propagates from the live node", async () => {
  const failure = new Error("terminal failed");
  const model = scripted([callReply(["ask_user", { question: "Continue?" }, "question-1"])]);

  const error = await rejected(
    agent().run({}, {
      models: serving(model),
      tools: new ToolSet({
        ask_user: () => {
          throw failure;
        },
      }),
    }),
  );

  expect(error).toBe(failure);
});

test("structured output succeeds after a blocking ask", async () => {
  const model = scripted([
    callReply(["ask_user", { question: "Approve?" }, "question-1"]),
    textReply("ready for structured output"),
    textReply('{"name": "Ada", "score": 9}'),
  ]);
  const node = agent();
  node.outputContract = new ValueContract(
    new Type({
      kind: ValueKind.OBJECT,
      fields: new Map([
        ["name", Type.scalar(ValueKind.STRING)],
        ["score", Type.scalar(ValueKind.INTEGER)],
      ]),
      required: new Set(["name", "score"]),
    }),
  );

  const outcome = await evaluate(node, model, () => "yes");

  expect(outcome).toEqual(new Output({ name: "Ada", score: 9n }));
  expect(model.requests).toHaveLength(3);
});

test("a post-assert fires after a blocking ask", async () => {
  const model = scripted([callReply(["ask_user", { question: "Approve?" }, "question-1"]), textReply("BAD")]);
  const node = agent();
  node.postAsserts = [new Assertion("output == 'GOOD'", "output == 'GOOD'")];

  const outcome = await evaluate(node, model, () => "yes");

  expect(outcome).toBeInstanceOf(EvaluationFailure);
  expect((outcome as EvaluationFailure).failure.causeType).toBe("NodeAssertFailed");
  expect((outcome as EvaluationFailure).failure.message.toLowerCase()).toContain("post-assert");
});
