/**
 * A flow's `output:` shape decides a child's signature: a port of
 * `tests/engine/test_flow_output_shape.py`.
 *
 * Ported: `test_child_signature_of_a_one_field_record_is_its_exact_record`,
 * `test_child_signature_of_a_value_output_is_the_value_contract`,
 * `test_child_signature_of_an_empty_record_is_the_empty_record_contract` and
 * `test_direct_child_signature_must_say_it_returns_a_record`.
 *
 * Changed: the child-signature rows build the child with `handChild` (a `step` producer typed
 * `str`), where Python loads a flow; the empty-record row compares with `sameOutputContract`
 * (Python's dataclass `==`). Python's `ValueError` is a `RangeError`.
 *
 * TS-only: `ChildSignature`'s constructor errors, which no Python test reaches: a non-contract
 * output, a non-array `declaredOutputs`, a malformed pair, an empty or duplicate name, a
 * non-contract declared output; and the form with no declared evidence, which skips the check.
 *
 * Later (T3-16, the MAP and LOOP readers):
 * `test_map_over_a_one_field_record_def_returns_a_list_of_records` (its contract half),
 * `test_map_over_a_def_without_fields_is_rejected`,
 * `test_loop_body_with_an_empty_record_output_is_rejected` and
 * `test_loop_value_body_re_exporting_an_empty_record_is_rejected`.
 *
 * Later (T3-17, T3-18, T3-22, T4): `test_flow_result_follows_the_output_shape` (T3-17, T4);
 * `test_call_over_a_one_field_record_def_returns_the_record`,
 * `test_call_over_a_flat_single_node_def_returns_the_bare_value` and
 * `test_nested_value_re_export_of_a_record_def_keeps_the_record` (T3-22, T4);
 * `test_inline_call_body_with_a_one_field_output_returns_the_record` and
 * `test_loop_body_with_a_one_field_record_output_runs` (T3-17, T4); the run half of
 * `test_map_over_a_one_field_record_def_returns_a_list_of_records`,
 * `test_root_feedback_on_a_one_field_record_reaches_its_producer`,
 * `test_an_unquoted_yes_flow_output_name_is_text` and
 * `test_a_literal_flow_output_is_returned_as_written` (T4);
 * `test_case_then_call_over_a_one_field_record_def_returns_the_record` (T3-17, T3-22, T4);
 * `test_reading_a_missing_field_of_a_record_child_is_rejected_at_load` (T3-18, T3-22); and
 * `test_a_non_string_output_name_in_a_def_is_a_load_error` (T3-22).
 *
 * Not ported here: `test_a_non_string_flow_output_name_is_a_load_error_at_its_key` is the source
 * layer's message, which `tests/compose/non_text_keys.test.ts` pins.
 */

import { expect, test } from "bun:test";

import { combineOutputContracts } from "../../src/compose/helpers/output_contracts.ts";
import { ChildSignature, childSignature } from "../../src/compose/nodes/resolver.ts";
import { isExactly, type OutputContract, sameOutputContract, ValueContract } from "../../src/nodes/contracts.ts";
import { Type, typeEquals, ValueKind } from "../../src/typesys/values.ts";
import { handChild } from "../compose/declarations.ts";

const STR = Type.scalar(ValueKind.STRING);

/** A child whose one producer `step` is typed `str`, returning `output`. */
function shapeChild(output: Parameters<typeof handChild>[0]["output"]) {
  return handChild({ producers: { step: new ValueContract(STR) }, output });
}

/** Return the error `run` throws, failing when it returns. */
function thrown(run: () => unknown): Error {
  try {
    run();
  } catch (error) {
    return error as Error;
  }
  throw new Error("expected an error");
}

/** Expect `run` to throw exactly `cls` with `message`. */
function expectError(run: () => unknown, cls: typeof Error, message: string): void {
  const error = thrown(run);
  expect(error.constructor).toBe(cls);
  expect(error.message).toBe(message);
}

test("the child signature of a one-field record is its exact record", () => {
  const sig = childSignature(shapeChild({ order: "${step.output}" }));

  expect(sig.returnsRecord).toBe(true);
  expect(isExactly(sig.output, ValueContract)).toBe(true);
  const output = sig.output as ValueContract;
  expect(output.type.kind).toBe(ValueKind.OBJECT);
  expect(output.type.fields!.size).toBe(1);
  expect(typeEquals(output.type.fields!.get("order")!, STR)).toBe(true);
});

test("the child signature of a value output is the value contract", () => {
  const sig = childSignature(shapeChild("${step.output}"));

  expect(sig.returnsRecord).toBe(false);
  expect(isExactly(sig.output, ValueContract)).toBe(true);
  expect(typeEquals((sig.output as ValueContract).type, STR)).toBe(true);
});

test("the child signature of an empty record is the empty record contract", () => {
  const sig = childSignature(shapeChild({}));

  expect(sig.returnsRecord).toBe(true);
  expect(sameOutputContract(sig.output, combineOutputContracts(new Map()))).toBe(true);
});

test("a direct child signature must say it returns a record", () => {
  const text = new ValueContract(STR);
  expectError(
    () =>
      new ChildSignature({
        inputs: [],
        output: combineOutputContracts(
          new Map([
            ["a", text],
            ["b", text],
          ]),
        ),
        declaredOutputs: [
          ["a", text],
          ["b", text],
        ],
        returnsRecord: false,
      }),
    RangeError,
    "a child that does not return a record declares at most one output",
  );
});

test("the ChildSignature constructor rejects malformed output evidence", () => {
  const text = new ValueContract(STR);
  const forged = "x" as unknown as OutputContract;

  expectError(
    () => new ChildSignature({ inputs: [], output: forged }),
    TypeError,
    "child output must be ValueContract, PartialRecordContract, or null, got string",
  );
  expectError(
    () =>
      new ChildSignature({
        inputs: [],
        output: text,
        declaredOutputs: "result" as never,
      }),
    TypeError,
    "declared child outputs must be an array",
  );
  expectError(
    () =>
      new ChildSignature({
        inputs: [],
        output: text,
        declaredOutputs: [["result"]] as never,
      }),
    TypeError,
    "declared child outputs must be (name, optional contract) tuples",
  );
  expectError(
    () => new ChildSignature({ inputs: [], output: text, declaredOutputs: [["", text]] }),
    TypeError,
    "declared child output names must be non-empty strings",
  );
  expectError(
    () =>
      new ChildSignature({
        inputs: [],
        output: text,
        declaredOutputs: [
          ["a", text],
          ["a", text],
        ],
        returnsRecord: true,
      }),
    RangeError,
    "duplicate declared child output 'a'",
  );
  expectError(
    () => new ChildSignature({ inputs: [], output: text, declaredOutputs: [["a", forged]] }),
    TypeError,
    "declared child output contracts must be ValueContract, PartialRecordContract, or null, got string for 'a'",
  );
});

test("a ChildSignature with no declared evidence skips the evidence check", () => {
  const text = new ValueContract(STR);
  const sig = new ChildSignature({ inputs: [], output: text });

  expect(sig.output).toBe(text);
  expect(sig.declaredOutputs).toEqual([]);
  expect(sig.outputNames).toEqual([]);
  expect(sig.returnsRecord).toBe(false);
  expect(Object.isFrozen(sig)).toBe(true);
  expect(Object.isFrozen(sig.inputs)).toBe(true);
});
