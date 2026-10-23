/**
 * `generateStructured`: correcting an invalid structured reply under `retries:`, and the output
 * path of the last error; mirrors `tests/engine/test_structured_retry.py` and the node-level
 * halves of `tests/engine/test_agent_missing_tool_call.py`.
 *
 * Ported: `test_literal_retries_an_invalid_tag_then_returns_bare_tag`,
 * `test_a_reply_without_a_structured_result_is_corrected_and_retried` and
 * `test_replies_without_a_structured_result_fail_naming_the_tool_once_retries_are_spent` (over a
 * stub `RouteModel` taking the tool strategy), `test_structured_validation_path_keeps_record_fields`
 * (native and prompt), `test_structured_validation_path_strips_only_generated_wrappers`,
 * `test_structured_validation_path_strips_nullable_record_wrapper` (native and prompt) and
 * `test_contract_path_validation_rejects_foreign_location_shapes` (`isContractPath`). The error's
 * output path is `OutputInvalidError.path` rather than a function of the exception.
 * `test_missing_tool_call_error_reads_cleanly` is ported in `tests/llm/structured.test.ts`
 * (`MissingToolCallError`'s text).
 *
 * Changed:
 * - `test_retry_on_invalid_then_succeeds`: the invalid first attempt is an unparseable reply;
 *   Python's raised a `ValueError` from the model call, which TS does not correct (next row).
 * - Python corrects any non-auth error a model call raises; TS never corrects a provider error:
 *   transport failures are `callWithRetries`' job under `YIAN_MODEL_RETRIES`, and what escapes it
 *   propagates unchanged after one call.
 * - `test_retry_on_invalid_provider_shape_then_succeeds`: the reply that can't be unwrapped is a
 *   JSON object without the wrapper field.
 * - `test_stop_during_native_binding_prevents_provider_invoke`: there is no binding step; a stop
 *   (an aborted signal) before the call sends nothing.
 * - `test_retry_exhausted_raises`: once the corrections are spent the error is an
 *   `OutputInvalidError` with the last error's message and the last error as `cause`, not the last
 *   error itself; so the tool rows and `test_agent_missing_tool_call.py` read
 *   `OutputInvalidError` with the `MissingToolCallError` as `cause` (Python's `cause_type` is
 *   `MissingToolCallError`, its code `node.execute`; TS's code is `agent.output_invalid`).
 * - `test_structured_validation_path_preserves_owned_root_location`: a reply is always a JSON
 *   object, so the root row is a reply without the wrapper field, refused at the root `[]`.
 *
 * Not ported: pydantic frame ownership (`test_structured_validation_path_accepts_generated_constructor_frame`,
 * `..._accepts_direct_parser_wrapped_error`, `..._rejects_wrapped_same_title_foreign_schema`,
 * `..._rejects_same_title_foreign_schema`, `..._rejects_unowned_pydantic_errors`): the path comes
 * with yian's own value check error, so there is no foreign validation error to tell apart;
 * `test_structured_validation_path_uses_first_pydantic_error`: the value check stops at its first
 * error, so there is no error list to choose from; `test_tool_parser_returns_none_for_a_reply_without_a_tool_call`:
 * a LangChain parser.
 *
 * Ported through `runFlow` (`runWith`), the flow rows of `test_agent_missing_tool_call.py`:
 * `test_an_agent_retries_a_reply_without_the_tool_call` and
 * `test_an_agent_fails_naming_the_tool_once_retries_are_spent`, without their session rows (TS yian
 * has no session memory). Changed: the model is the
 * `ToolRoute` stub on OpenAI Chat (Python: an Anthropic client over a fake server), so the flow's
 * config names `gpt-5.5`; the corrected request holds the system message, the question and the
 * correction, with no assistant reply (Anthropic's body sends the system apart and merges the two
 * user messages); the failure is `agent.output_invalid` with
 * cause type `OutputInvalidError` (Python: `node.execute`, `MissingToolCallError`).
 *
 * Not ported: `test_a_generic_agent_fails_naming_the_tool_when_the_reply_has_no_result`, whose
 * subject is a LangChain `with_structured_output` binding that returns no parsed value; TS has no
 * such binding, and a reply without the tool call is the tool strategy's, which the rows above run.
 *
 * TS-only: a corrected request holds the earlier messages and the correction, not the invalid
 * reply; a stop during the call of an invalid last reply wins over the `OutputInvalidError`, and
 * one during the call of a valid reply wins over its value; a
 * refusal written in the reply, once the corrections are spent, is an `OutputInvalidError` with no
 * path; a value check message reads `structured output at <path>:`; a reply whose value nests past
 * `YIAN_MAX_VALUE_NESTING` is corrected naming the path, then fails with no path, and a value
 * exactly at the limit passes (the wrapper is not a level); the native strategy records the value
 * in the evaluation trace, the prompt strategy every reply's text; `retry` is called once per
 * structured call, so a transport retry of a later call reports that call.
 */

import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterAll, afterEach, beforeEach, describe, expect, test } from "bun:test";

import { BUILTIN, forget, install } from "../../src/defaults.ts";
import { ModelCallFinished } from "../../src/events.ts";
import { NoAuth } from "../../src/llm/auth.ts";
import { ProviderError } from "../../src/llm/errors.ts";
import type { EventStream } from "../../src/llm/events.ts";
import { RouteModel } from "../../src/llm/executor.ts";
import {
  Message,
  type Part,
  RefusalPart,
  type Request,
  Response,
  TextPart,
  ToolCallPart,
} from "../../src/llm/messages.ts";
import { OpenAIChat } from "../../src/llm/protocols/openai_chat.ts";
import { Route } from "../../src/llm/route.ts";
import { MissingToolCallError, StructuredOutputError } from "../../src/llm/structured.ts";
import { ScriptedModel, type ScriptedReply } from "../../src/llm/testing.ts";
import type { EvaluationTrace, VisibleInteraction } from "../../src/memory/contracts.ts";
import { ModelTextInteraction, ModelValueInteraction } from "../../src/memory/contracts.ts";
import {
  generateStructured,
  isContractPath,
  OutputInvalidError,
} from "../../src/nodes/agent/structured.ts";
import { AgentSteps } from "../../src/nodes/agent/context.ts";
import type { LiveSinks } from "../../src/nodes/agent/live.ts";
import { NodeSteps } from "../../src/nodes/contracts.ts";
import { virtualScheduler } from "../../src/platform.ts";
import { Type, TypeCheckError, type ValueRecord, ValueKind } from "../../src/typesys/values.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { hostEnv } from "./mcp_helpers.ts";
import { PLAIN_FLOW, runWith } from "./run_state_fixtures.ts";

beforeEach(() => install({ ...BUILTIN, modelRetries: 0 }));
afterEach(forget);

const NATIVE = { provider: "openai", model: "gpt-5.5" };
const PROMPTED = { provider: "vllm", model: "no-structured-sentinel" };
const INT = Type.scalar(ValueKind.INTEGER);
const QUESTION = user("question");

function user(text: string): Message {
  return new Message({ role: "user", parts: [new TextPart(text)] });
}

function reply(...parts: Part[]): Response {
  return new Response({ message: new Message({ role: "assistant", parts }), finishReason: "stop" });
}

/** A reply whose text is `text`. */
function text(text: string): Response {
  return reply(new TextPart(text));
}

/** A model answering `replies` in turn, native (`json_schema`) or prompted. */
function scripted(replies: ScriptedReply[], options = NATIVE): ScriptedModel {
  return new ScriptedModel(replies, options);
}

/** A `RouteModel` on OpenAI Chat (`gpt-5.5`, so the tool strategy) that replays `replies`. */
class ToolRoute extends RouteModel {
  readonly script: ScriptedModel;

  constructor(replies: ScriptedReply[]) {
    super({
      route: new Route({ provider: "openai", baseUrl: "https://api.example", path: "x", auth: new NoAuth() }),
      protocol: new OpenAIChat({ model: "gpt-5.5", provider: "openai" }),
    });
    this.script = new ScriptedModel(replies, NATIVE);
  }

  override stream(request: Request, options: { signal?: AbortSignal } = {}): EventStream {
    return this.script.stream(request, options);
  }
}

/** The error `promise` rejects with. */
async function rejection(promise: Promise<unknown>): Promise<unknown> {
  try {
    await promise;
  } catch (error) {
    return error;
  }
  throw new Error("the promise resolved");
}

/** The `OutputInvalidError` generating `typ` from one reply `payload` throws once retries are spent. */
async function exhausted(typ: Type, payload: string, options = NATIVE): Promise<OutputInvalidError> {
  const error = await rejection(generateStructured(scripted([text(payload)], options), [], typ, { retries: 0 }));
  expect(error).toBeInstanceOf(OutputInvalidError);
  return error as OutputInvalidError;
}

function record(fields: Record<string, Type>, options: { nullable?: boolean } = {}): Type {
  return new Type({
    kind: ValueKind.OBJECT,
    fields: new Map(Object.entries(fields)),
    required: new Set(Object.keys(fields)),
    nullable: options.nullable,
  });
}

test("test_retry_on_invalid_then_succeeds (an unparseable reply)", async () => {
  const model = scripted([text("bad json"), text('{"value": 42}')]);
  expect(await generateStructured(model, [], INT, { retries: 2 })).toBe(42n);
  expect(model.requests).toHaveLength(2);
});

test("test_literal_retries_an_invalid_tag_then_returns_bare_tag", async () => {
  const typ = new Type({ kind: ValueKind.STRING, tags: new Set(["positive", "neutral", "negative"]) });
  const model = scripted([text('{"value": "The sentiment is neutral."}'), text('{"value": "neutral"}')]);
  expect(await generateStructured(model, [], typ, { retries: 2 })).toBe("neutral");
  expect(model.requests).toHaveLength(2);
});

test("a provider error propagates uncorrected (Changed)", async () => {
  const failure = new ProviderError("server", "upstream failed", { status: 500 });
  const model = scripted([failure, text('{"value": 42}')]);
  expect(await rejection(generateStructured(model, [], INT, { retries: 2 }))).toBe(failure);
  expect(model.requests).toHaveLength(1);
});

test("each structured call is written as a model_call step, a transport retry as a failed one first", async () => {
  install({ ...BUILTIN, modelRetries: 1 });
  const failure = new ProviderError("server", "upstream failed", { status: 500 });
  const model = scripted([text("bad json"), failure, text('{"value": 42}')]);
  const recorded: ValueRecord[] = [];
  const steps = new AgentSteps(new NodeSteps(1, [], (step) => void recorded.push(step)), () => 0, null, (error) => error);
  const retry = { scheduler: virtualScheduler() };
  expect(await generateStructured(model, [], INT, { retries: 1, retry, steps })).toBe(42n);
  expect(recorded.map((step) => ("error" in step ? `failed ${(step["error"] as ValueRecord)["kind"]}` : "ok"))).toEqual([
    "ok",
    "failed server",
    "ok",
  ]);
  // The correction makes the second call's request differ from the first; its retry repeats it.
  const digests = recorded.map((step) => step["request_sha256"]);
  expect(digests[1]).toBe(digests[2]);
  expect(digests[0]).not.toBe(digests[1]);
  expect(model.requests).toHaveLength(3);
});

test("test_retry_on_invalid_provider_shape_then_succeeds (a reply without the wrapper field)", async () => {
  const model = scripted([text('{"wrong": 1}'), text('{"value": 42}')]);
  expect(await generateStructured(model, [], INT, { retries: 2 })).toBe(42n);
  expect(model.requests).toHaveLength(2);
});

test("test_stop_during_native_binding_prevents_provider_invoke (a stop before the call)", async () => {
  const stop = new AbortController();
  stop.abort(new Error("stopped"));
  const model = scripted([text('{"value": 42}')]);
  const error = await rejection(generateStructured(model, [], INT, { signal: stop.signal }));
  expect(error).toBe(stop.signal.reason);
  expect(model.requests).toEqual([]);
});

test("test_retry_exhausted_raises", async () => {
  const model = scripted([text("nope 1"), text("nope 2"), text("nope 3")]);
  const error = await rejection(generateStructured(model, [], INT, { retries: 2 }));
  expect(error).toBeInstanceOf(OutputInvalidError);
  const invalid = error as OutputInvalidError;
  expect(invalid.name).toBe("OutputInvalidError");
  expect(invalid.message).toBe("Invalid json output: nope 3");
  expect(invalid.cause).toBeInstanceOf(StructuredOutputError);
  expect((invalid.cause as Error).message).toBe("Invalid json output: nope 3");
  expect(invalid.path).toBeNull();
  expect(model.requests).toHaveLength(3);
});

describe("a reply without the schema's tool call", () => {
  const missing = "the reply did not call the ScalarWrapper tool; call it with the answer";

  test("test_a_reply_without_a_structured_result_is_corrected_and_retried", async () => {
    const call = new ToolCallPart({ id: "c1", name: "ScalarWrapper", arguments: { value: 42n } });
    const model = new ToolRoute([text("42"), reply(call)]);
    expect(await generateStructured(model, [QUESTION], INT, { retries: 2 })).toBe(42n);
    const requests = model.script.requests;
    expect(requests).toHaveLength(2);
    expect(requests[0]!.tools.map((tool) => tool.name)).toEqual(["ScalarWrapper"]);
    const correction = requests[1]!.messages.at(-1)!;
    expect(correction.role).toBe("user");
    expect(correction.text()).toContain("the reply did not call the ScalarWrapper tool");
  });

  test.each([0, 2])(
    "test_replies_without_a_structured_result_fail_naming_the_tool_once_retries_are_spent (retries %p)",
    async (retries) => {
      const model = new ToolRoute(Array.from({ length: retries + 1 }, () => text("7")));
      const error = await rejection(generateStructured(model, [QUESTION], INT, { retries }));
      expect(error).toBeInstanceOf(OutputInvalidError);
      expect((error as Error).message).toBe(missing);
      expect((error as Error).cause).toBeInstanceOf(MissingToolCallError);
      expect((error as OutputInvalidError).path).toBeNull();
      expect(model.script.requests).toHaveLength(retries + 1);
    },
  );
});

describe("test_structured_validation_path_keeps_record_fields", () => {
  const typ = record({ profile: record({ age: INT }) });

  test.each([
    ["native", NATIVE],
    ["prompt", PROMPTED],
  ])("%s", async (_label, options) => {
    const error = await exhausted(typ, '{"profile": {"age": "bad"}}', options);
    expect(error.path).toEqual(["profile", "age"]);
  });
});

test("test_structured_validation_path_strips_only_generated_wrappers", async () => {
  expect((await exhausted(INT, '{"value": "bad"}')).path).toEqual([]);
  const listed = new Type({ kind: ValueKind.LIST_OBJECT, element: record({ age: INT }) });
  const error = await exhausted(listed, '{"items": [{"age": 1}, {"age": "bad"}]}');
  expect(error.path).toEqual([1, "age"]);
});

describe("test_structured_validation_path_strips_nullable_record_wrapper", () => {
  const typ = record({ name: Type.scalar(ValueKind.STRING), age: INT }, { nullable: true });

  test.each([
    ["native", NATIVE],
    ["prompt", PROMPTED],
  ])("%s", async (_label, options) => {
    const error = await exhausted(typ, '{"value": {"name": "a", "age": "bad"}}', options);
    expect(error.path).toEqual(["age"]);
  });
});

test("test_structured_validation_path_preserves_owned_root_location (a reply without the wrapper field)", async () => {
  const error = await exhausted(INT, '{"other": 1}');
  expect(error.message).toBe('structured output: record missing required fields: ["value"]');
  expect(error.path).toEqual([]);
  expect(error.cause).toBeInstanceOf(TypeCheckError);
});

test("test_contract_path_validation_rejects_foreign_location_shapes", () => {
  const rec = new Type({ kind: ValueKind.OBJECT, fields: new Map([["age", INT]]) });
  const listed = new Type({ kind: ValueKind.LIST_OBJECT, element: rec });
  expect(isContractPath(["unknown"], rec)).toBe(false);
  expect(isContractPath(["foreign"], INT)).toBe(false);
  expect(isContractPath(["age"], listed)).toBe(false);
  expect(isContractPath([-1, "age"], listed)).toBe(false);
  expect(isContractPath([0, "unknown"], listed)).toBe(false);
  expect(isContractPath([0, "age"], listed)).toBe(true);
  expect(isContractPath([], INT)).toBe(true);
});

describe("TS-only", () => {
  test("a corrected request holds the earlier messages and the correction, not the invalid reply", async () => {
    const model = scripted([text('{"value": "x"}'), text('{"value": 1}')]);
    await generateStructured(model, [QUESTION], INT, { retries: 1 });
    const second = model.requests[1]!.messages;
    expect(second.map((message) => message.role)).toEqual(["user", "user"]);
    expect(second[0]!.text()).toBe("question");
    expect(second[1]!.text()).toBe(
      'Your previous output was invalid: structured output: "x" does not match declared type int. ' +
        "Respond with valid data matching the schema.",
    );
  });

  test("a stop during the call of an invalid last reply wins over the OutputInvalidError", async () => {
    const stop = new AbortController();
    // The finished event comes once the reply is complete, before it is checked.
    const live: LiveSinks = {
      nodeId: "agent",
      provider: null,
      model: null,
      emitEvent: (event) => {
        if (event instanceof ModelCallFinished) stop.abort(new Error("stopped"));
      },
      emitStream: null,
    };
    const model = scripted([text("not json")]);
    const error = await rejection(generateStructured(model, [], INT, { retries: 0, live, signal: stop.signal }));
    expect(error).toBe(stop.signal.reason);
    expect(model.requests).toHaveLength(1);
  });

  test("a stop during the call of a valid reply wins over its value", async () => {
    const stop = new AbortController();
    const live: LiveSinks = {
      nodeId: "agent",
      provider: null,
      model: null,
      emitEvent: (event) => {
        if (event instanceof ModelCallFinished) stop.abort(new Error("stopped"));
      },
      emitStream: null,
    };
    const model = scripted([text('{"value": 42}')]);
    const error = await rejection(generateStructured(model, [], INT, { retries: 0, live, signal: stop.signal }));
    expect(error).toBe(stop.signal.reason);
    expect(model.requests).toHaveLength(1);
  });

  test("a refusal written in the reply, once the corrections are spent, is an OutputInvalidError", async () => {
    const model = scripted([reply(new RefusalPart("no")), reply(new RefusalPart("still no"))]);
    const error = await rejection(generateStructured(model, [], INT, { retries: 1 }));
    expect(error).toBeInstanceOf(OutputInvalidError);
    expect((error as Error).message).toBe("the model refused: still no");
    expect((error as OutputInvalidError).path).toBeNull();
    expect(model.requests).toHaveLength(2);
  });

  test("a value check message names the path inside the output", async () => {
    const error = await exhausted(record({ profile: record({ age: INT }) }), '{"profile": {"age": "bad"}}');
    expect(error.message).toBe('structured output at profile.age: "bad" does not match declared type int');
  });

  test("a reply nesting past the limit is corrected, then fails with no path; one at the limit passes", async () => {
    const nest = (levels: number) => '{"a": '.repeat(levels) + "1" + "}".repeat(levels);
    const any = Type.scalar(ValueKind.ANY);
    const reason =
      `structured output at ${Array(100).fill("a").join(".")}: value nests deeper than 100 levels, ` +
      "the limit for a yian value (each record and list is a level)";
    const model = scripted([text(`{"value": ${nest(101)}}`), text('{"value": {"a": 1}}')]);
    expect(await generateStructured(model, [], any, { retries: 1 })).toEqual({ a: 1n });
    expect(model.requests[1]!.messages.at(-1)!.text()).toBe(
      `Your previous output was invalid: ${reason}. Respond with valid data matching the schema.`,
    );
    const error = await exhausted(any, `{"value": ${nest(101)}}`);
    expect(error.message).toBe(reason);
    expect(error.path).toBeNull();
    // The wrapper is not a level of the value.
    const atLimit = await generateStructured(scripted([text(`{"value": ${nest(100)}}`)]), [], any, { retries: 0 });
    expect(JSON.stringify(atLimit, (_key, item: unknown) => (typeof item === "bigint" ? Number(item) : item))).toBe(
      nest(100).replaceAll(" ", ""),
    );
  });

  test("the native strategy records the value, the prompt strategy every reply's text", async () => {
    const recorded: VisibleInteraction[] = [];
    const trace: EvaluationTrace = {
      recordPrompt: () => undefined,
      recordInteraction: (interaction) => void recorded.push(interaction),
    };
    await generateStructured(scripted([text('{"value": 3}')]), [], INT, { evaluationTrace: trace });
    expect(recorded).toEqual([new ModelValueInteraction(3n)]);
    recorded.length = 0;
    const prompted = scripted([text("x"), text('{"value": 4}')], PROMPTED);
    await generateStructured(prompted, [], INT, { retries: 1, evaluationTrace: trace });
    expect(recorded).toEqual([new ModelTextInteraction("x"), new ModelTextInteraction('{"value": 4}')]);
  });
});

// ---------- an AGENT whose reply skips the schema's tool, through the runner ----------

describe("an agent reply without the schema's tool call", () => {
  const missing = "the reply did not call the ScalarWrapper tool; call it with the answer";
  const dir = mkdtempSync(join(tmpdir(), "yian-structured-retry-"));
  afterAll(() => rmSync(dir, { recursive: true, force: true }));

  /** Run `PLAIN_FLOW` as an int AGENT with `retries` over `model`. */
  async function runAgent(model: ToolRoute, retries: number) {
    const text = PLAIN_FLOW.replace("{provider: openai, model: test-model}", "{provider: openai, model: gpt-5.5}").replace(
      "    output: str",
      `    retries: ${retries}\n    output: int`,
    );
    return runWith(await loadFlow(text), { question: "q" }, null, model, { env: hostEnv(dir) });
  }

  test("an agent retries a reply without the tool call", async () => {
    const call = new ToolCallPart({ id: "c1", name: "ScalarWrapper", arguments: { value: 7n } });
    const model = new ToolRoute([text("7"), reply(call)]);
    const result = await runAgent(model, 2);
    expect(result.failure).toBeNull();
    expect(result.output).toBe(7n);
    const requests = model.script.requests;
    expect(requests).toHaveLength(2);
    const messages = requests[1]!.messages;
    expect(messages.map((message) => message.role)).toEqual(["system", "user", "user"]);
    const correction = messages.at(-1)!.text();
    expect(correction).toContain(missing);
    expect(correction).not.toContain("provider error (");
  });

  test.each([0, 2])("an agent fails naming the tool once retries are spent (retries %p)", async (retries) => {
    const model = new ToolRoute(Array.from({ length: retries + 1 }, () => text("7")));
    const result = await runAgent(model, retries);
    expect(result.status).toBe("failed");
    const failure = result.failure!;
    expect([failure.code, failure.nodeId]).toEqual(["agent.output_invalid", "agent"]);
    expect(failure.message).toContain(missing);
    expect(failure.trace.length).toBeGreaterThan(0);
    expect(failure.causeType).toBe("OutputInvalidError");
    expect(model.script.requests).toHaveLength(retries + 1);
  });
});
