/** `LLMConfig` refuses a non-finite number in a block. */

import { expect, test } from "bun:test";

import { LLMConfig } from "../../src/llm_clients/config.ts";
import { ValidationError } from "../../src/llm_clients/provider_blocks.ts";

// A block takes numbers as written, so the string "inf" is refused as a non-number rather than
// read as infinity.
test.each([
  ["nan", Number.NaN, "finite"],
  ["inf", Number.POSITIVE_INFINITY, "finite"],
  ["inf-string", "inf", "valid number"],
])("llm config rejects a nonfinite temperature (%s)", (_id, temperature, message) => {
  const build = () => new LLMConfig({ openai: { temperature } });
  expect(build).toThrow(ValidationError);
  expect(build).toThrow(message);
});
