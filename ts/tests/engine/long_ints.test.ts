/**
 * A port of `tests/engine/test_long_ints.py`: ints with more than 4300 decimal digits as yian
 * values.
 *
 * Ported: `test_reject_nonfinite_rejects_a_long_int` (its `root`, `negative` and `nested` rows),
 * `test_reject_nonfinite_passes_ints_within_the_limit_and_bools`,
 * `test_value_repr_shows_a_digit_count_for_a_long_int` and
 * `test_a_long_int_run_input_is_rejected_before_the_run_starts`. Host intake (`fromHost`) is
 * `reject_nonfinite`'s counterpart, as in `nonfinite_values.test.ts`.
 *
 * Changed: ints are `bigint`s with no digit limit (decisions 4a, 4b, 23), so a 5000-digit int is
 * a value where Python refuses it: intake returns it whole; `valueRepr` shows its digits cut at
 * the shown-value cap where Python shows `<int with 5000 digits>`; a run takes it as an `int` or
 * `list(int)` input (and as `int` text) and outputs it whole, writing its log. A long int for a
 * `str` input fails the run at START's input check (`contract.input`), where Python refuses it
 * before the run starts.
 *
 * Already ported: `test_reject_nonfinite_still_reports_nan` in `nonfinite_values.test.ts` (`still
 * reports NaN`) and `test_value_repr_of_a_small_int` in `bounded_value_messages.test.ts`.
 *
 * Not ported (Python APIs with no TS counterpart, decision 4a): `int_digits`, `parse_int_text`,
 * `parse_json_int` and its `OverLimitInt` marker, `reject_long_ints`, `long_int_digits`,
 * `rewrite_digit_limit_text` and `MAX_INT_DIGITS`, so `test_int_digits_counts_decimal_digits`,
 * `test_parse_int_text_skips_leading_zeros`, `test_parse_int_text_rejects_a_long_int`,
 * `test_parse_int_text_accepts_int_grammar`, `test_parse_int_text_skips_leading_zeros_of_any_script`,
 * `test_parse_int_text_raises_int_own_error_for_invalid_text`,
 * `test_parse_int_text_rejects_a_long_int_with_python_limit_off`,
 * `test_parse_json_int_marks_a_long_token`, `test_reject_long_ints_reports_the_path`,
 * `test_reject_long_ints_ignores_nan`, `test_long_int_digits_finds_the_first_long_int`,
 * `test_rewrite_digit_limit_text_replaces_python_message`,
 * `test_rewrite_digit_limit_text_without_a_count`,
 * `test_rewrite_digit_limit_text_matches_any_host_limit`,
 * `test_rewrite_digit_limit_text_leaves_other_text` and `test_yian_max_int_digits_sets_the_limit`; the `marker` row of
 * `test_reject_nonfinite_rejects_a_long_int` (`parse_json_int`), its `key` row (a record key is
 * always a string) and its `typed-value` row (no `ListAnyValue` carrier; intake refuses a class
 * instance); the set half of `test_value_repr_shows_a_digit_count_for_a_long_int` (no sets in
 * TS); and `test_value_repr_shows_a_digit_count_with_python_limit_off` (Python's
 * `sys.set_int_max_str_digits`).
 */

import { afterAll, expect, test } from "bun:test";
import { existsSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { loadFlow } from "../../src/compose/loader.ts";
import { runFlow } from "../../src/run.ts";
import { fromHost, type Value, valueRepr } from "../../src/typesys/values.ts";
import { hostEnv } from "./mcp_helpers.ts";

const dir = mkdtempSync(join(tmpdir(), "yian-long-ints-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));
const env = hostEnv(dir);

const BIG = 10n ** 4999n + 12345n; // 5000 digits
const D = "1".repeat(5000);

test.each<[string, Value]>([
  ["root", BIG],
  ["negative", -BIG],
  ["nested", { a: [1n, BIG] }],
])("reject nonfinite rejects a long int (%s)", (_id, value) => {
  expect(fromHost(value, "input 'n'")).toEqual(value);
});

test("reject nonfinite passes ints within the limit and bools", () => {
  expect(fromHost(10n ** 4300n - 1n, "input 'n'")).toBe(10n ** 4300n - 1n);
  expect(fromHost(true, "input 'n'")).toBe(true);
});

test("value repr shows a digit count for a long int", () => {
  expect(valueRepr(BIG)).toBe(String(BIG).slice(0, 199) + "…");
  expect(valueRepr([BIG])).toBe("[" + String(BIG).slice(0, 198) + "…");
});

/** Return a flow whose `c` CODE node binds `y` to the input `declared` declares and returns it. */
const inputFlow = (declared: string) => `
id: long-input
name: long_input
input:
  ${declared}
nodes:
  c:
    kind: code
    input:
      y: \${input.${declared.split(":")[0]}}
    code: |
      return inputs.y
output: \${c.output}
`;

let runs = 0;

test.each<[string, string, Record<string, unknown>, Value]>([
  ["int", "n: !int", { n: BIG }, BIG],
  ["int-text", "n: !int", { n: D }, BigInt(D)],
  ["list", "xs: !list(int)", { xs: [1n, BIG] }, [1n, BIG]],
])("a long int run input is rejected before the run starts (%s)", async (_id, declared, inputs, output) => {
  const log = join(dir, `run-${runs++}.jsonl`);

  const result = await runFlow(await loadFlow(inputFlow(declared), { env }), inputs, { env, log });

  expect(result.failure).toBeNull();
  expect(result.status).toBe("succeeded");
  expect(result.output).toEqual(output);
  expect(existsSync(log)).toBe(true);
});

test("a long int run input is rejected before the run starts (str)", async () => {
  const result = await runFlow(await loadFlow(inputFlow("n: !str"), { env }), { n: BIG }, { env });

  expect(result.status).toBe("failed");
  expect(result.failure!.code).toBe("contract.input");
  expect(result.failure!.message).toBe(`input 'n': ${String(BIG).slice(0, 199)}… does not match declared type str`);
});
