/**
 * `generateStructured` on the prompt strategy, for a model with no native structured output (the
 * catalog's `vllm`/`no-structured-sentinel` pair, through the real capability check): mirrors
 * `tests/engine/test_structured_fallback.py`.
 *
 * Ported: `test_no_native_support_uses_prompt_injection` (the last message asks for JSON matching
 * the schema; the request carries no tools and no output format),
 * `test_fallback_tolerates_json_code_fence`, `test_fallback_tolerates_bare_code_fence`,
 * `test_fallback_accepts_null_for_a_nullable_root` and
 * `test_fallback_retries_on_unparseable_then_succeeds`.
 *
 * Changed: the model is a `ScriptedModel` rather than a LangChain chat model whose
 * `with_structured_output` must not be called; the strategy shows in the request instead.
 */

import { afterEach, beforeEach, expect, test } from "bun:test";

import { BUILTIN, forget, install } from "../../src/defaults.ts";
import { Message, Response, TextPart } from "../../src/llm/messages.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { generateStructured } from "../../src/nodes/agent/structured.ts";
import { Type, ValueKind } from "../../src/typesys/values.ts";

beforeEach(() => install({ ...BUILTIN, modelRetries: 0 }));
afterEach(forget);

const PROMPTED = { provider: "vllm", model: "no-structured-sentinel" };
const INT = Type.scalar(ValueKind.INTEGER);

function text(text: string): Response {
  return new Response({ message: new Message({ role: "assistant", parts: [new TextPart(text)] }), finishReason: "stop" });
}

/** A prompt-strategy model answering `replies` in turn. */
function prompted(...replies: string[]): ScriptedModel {
  return new ScriptedModel(replies.map(text), PROMPTED);
}

test("test_no_native_support_uses_prompt_injection", async () => {
  const model = prompted('{"value": 5}');
  expect(await generateStructured(model, [], INT)).toBe(5n);
  const request = model.requests[0]!;
  const instruction = request.messages.at(-1)!;
  expect(instruction.role).toBe("user");
  expect(instruction.text()).toStartWith("Respond with ONLY a JSON object matching this JSON schema");
  expect(instruction.text().toLowerCase()).toContain("schema");
  expect(request.tools).toEqual([]);
  expect(request.outputFormat).toBeNull();
});

test("test_fallback_tolerates_json_code_fence", async () => {
  const model = prompted('```json\n{"value": 5}\n```');
  expect(await generateStructured(model, [], INT)).toBe(5n);
  expect(model.requests).toHaveLength(1);
});

test("test_fallback_tolerates_bare_code_fence", async () => {
  const model = prompted('```\n{"value": 9}\n```');
  expect(await generateStructured(model, [], INT)).toBe(9n);
  expect(model.requests).toHaveLength(1);
});

test("test_fallback_accepts_null_for_a_nullable_root", async () => {
  const model = prompted('{"value": null}');
  expect(await generateStructured(model, [], new Type({ kind: ValueKind.INTEGER, nullable: true }))).toBeNull();
  expect(model.requests[0]!.messages.at(-1)!.text()).toContain('"null"');
  expect(model.requests).toHaveLength(1);
});

test("test_fallback_retries_on_unparseable_then_succeeds", async () => {
  const model = prompted("not json", '{"value": 7}');
  expect(await generateStructured(model, [], INT, { retries: 2 })).toBe(7n);
  expect(model.requests).toHaveLength(2);
  expect(model.requests[1]!.messages.at(-1)!.text()).toBe(
    "Your previous output was invalid: Invalid json output: not json. Respond with ONLY a JSON object matching the schema.",
  );
});
