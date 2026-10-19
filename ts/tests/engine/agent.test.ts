/**
 * The AGENT node's helpers: mirrors `tests/engine/test_agent.py`.
 *
 * TS-only: `textOf` of a string, of a message with text, reasoning, tool-call and refusal parts
 * (only the text, joined with `"\n"`), and of a message without text; `ReplyInterrupted`'s message,
 * `reply`, `usage` (default `null`) and that it is an `Error`.
 *
 * The AgentNode construction tests come with the node: `test_direct_agent_rejects_*`,
 * `test_direct_agent_requires_an_exact_variable_contract`,
 * `test_env_max_tool_iterations_minus_one_accepted` and
 * `test_env_max_tool_iterations_rejects_bad_value`.
 *
 * Later (T5-11b), the tool loop: `test_tool_loop_round_trip`,
 * `test_tool_loop_executes_model_calls_in_returned_order`,
 * `test_agent_rejects_globally_registered_but_ungranted_tool`,
 * `test_tool_loop_rejects_model_injected_arguments`,
 * `test_tool_loop_propagates_tool_exception_as_node_failure`,
 * `test_tool_validation_error_does_not_claim_scalar_output_origin`, the cap tests
 * (`test_tool_loop_cap_trips_when_bounded`, `test_default_no_cap_runs_unbounded`,
 * `test_env_max_tool_iterations_overrides_cap`, `test_tool_loop_cap_does_not_run_the_last_replys_tools`,
 * `test_ungranted_tool_stays_node_execute`, `test_tool_loop_cap_of_one_still_answers_without_tools`)
 * and `test_tool_auth_failure_stays_node_execute`.
 *
 * Later (T5-11a): `test_no_tools_returns_plain_text`; the structured-origin tests
 * (`test_structured_failure_selects_exact_or_ancestor_output_origin`,
 * `test_structured_scalar_failure_selects_contract_root`,
 * `test_non_validation_structured_failure_keeps_node_origin`,
 * `test_source_free_structured_failure_keeps_empty_trace`); the `provider.auth` tests
 * (`test_model_auth_failure_is_provider_auth`, `test_permission_denied_is_provider_auth`,
 * `test_rate_limit_stays_node_execute`, `test_credential_resolution_auth_error_is_provider_auth`,
 * `test_structured_call_auth_failure_is_provider_auth`,
 * `test_fallback_structured_auth_failure_is_not_retried`,
 * `test_a_stop_during_a_refused_structured_call_stops_the_node`,
 * `test_provider_auth_failure_message_is_scrubbed`).
 *
 * Not ported: `test_agent_run_has_no_scratch_kwarg` (signature introspection).
 */

import { expect, test } from "bun:test";

import { TokenUsage } from "../../src/events.ts";
import { Message, ReasoningPart, RefusalPart, TextPart, ToolCallPart } from "../../src/llm/messages.ts";
import { ReplyInterrupted } from "../../src/nodes/agent/interrupts.ts";
import { textOf } from "../../src/nodes/agent/text.ts";

test("textOf returns a string as it is", () => {
  expect(textOf("hello\nworld")).toBe("hello\nworld");
  expect(textOf("")).toBe("");
});

test("textOf keeps only a message's text parts, joined with a newline", () => {
  const message = new Message({
    role: "assistant",
    parts: [
      new ReasoningPart("thinking"),
      new TextPart("first"),
      new ToolCallPart({ id: "c1", name: "search", arguments: {} }),
      new RefusalPart("no"),
      new TextPart("second"),
    ],
  });
  expect(textOf(message)).toBe("first\nsecond");
});

test("textOf of a message without text is empty", () => {
  const message = new Message({
    role: "assistant",
    parts: [new ReasoningPart("thinking"), new ToolCallPart({ id: "c1", name: "search", arguments: {} })],
  });
  expect(textOf(message)).toBe("");
});

test("ReplyInterrupted carries the partial reply and the usage so far", () => {
  const reply = new Message({ role: "assistant", parts: [new TextPart("partial")] });
  const usage = new TokenUsage({ inputTokens: 3, outputTokens: 2, totalTokens: 5 });
  const error = new ReplyInterrupted(reply, usage);
  expect(error).toBeInstanceOf(Error);
  expect(error.name).toBe("ReplyInterrupted");
  expect(error.message).toBe("model reply interrupted by the host");
  expect(error.reply).toBe(reply);
  expect(error.usage).toBe(usage);
});

test("ReplyInterrupted's usage defaults to null", () => {
  const reply = new Message({ role: "assistant", parts: [] });
  expect(new ReplyInterrupted(reply).usage).toBeNull();
});
