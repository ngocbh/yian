/**
 * AGENT local variables: mirrors `tests/engine/test_agent_variables.py`.
 *
 * Ported: `test_an_object_variable_default_holding_a_model_is_rejected` and
 * `test_declared_variable_inventory_preserves_canonical_declarations`.
 *
 * Changed: the object default holds a class instance where Python's holds a pydantic model; the
 * declarations are built by hand where Python reads them from YAML; the base `Node` is abstract,
 * so a minimal concrete node stands in for `Node("ordinary")`.
 *
 * Later (T4, T5-11a), they run the node: the retained render capture tests
 * (`test_retained_agent_captures_actual_render_locals_once`,
 * `test_retained_render_locals_are_detached_before_render_and_model_callbacks`,
 * `test_retained_agent_distinguishes_partial_and_empty_render_locals`,
 * `test_retained_agent_canonicalizes_supported_file_variable_without_changing_default`),
 * `test_render_capture_preparation_faults_escape_with_original_host_identity` and
 * `test_retained_agent_keeps_early_binding_unavailable_and_capture_off_dormant`.
 *
 * Later (T5-11a), rendering through `forward`:
 * `test_agent_renders_inputs_and_variable_defaults_without_execution_or_memory_scope`,
 * `test_agent_resolves_and_renders_once_with_repeated_literal_default_references`,
 * `test_mutable_variable_defaults_are_detached_between_invocations`,
 * `test_nullable_variable_defaults_follow_existing_strict_prompt_rules`,
 * `test_direct_variable_input_overlap_is_rejected_before_render_or_run`.
 *
 * Later (T5-11a, T3-22), through the loader and a run:
 * `test_temperature_variable_renders_independently_of_effective_model_configuration` and
 * `test_ordinary_run_keeps_variables_out_of_wiring_and_pool_and_backward_dormant`.
 */

import { expect, test } from "bun:test";

import { AgentNode } from "../../src/nodes/agent/node.ts";
import { Node, NodeKind } from "../../src/nodes/base.ts";
import { Parameter, ValueContract } from "../../src/nodes/contracts.ts";
import { Type, TypeCheckError, ValueKind } from "../../src/typesys/values.ts";

/** A minimal concrete node; the base's `declaredVariables` is the subject. */
class FuncNode extends Node {
  readonly kind = NodeKind.CODE;
}

/** A class instance, which no `object` value may hold. */
class Extras {
  readonly fields: Record<string, unknown>;

  constructor(fields: Record<string, unknown>) {
    this.fields = fields;
  }
}

test("an object variable default holding a class instance is rejected", () => {
  // An `object` holds JSON values only, so an instance inside one fails before any run.
  const variable = () =>
    new Parameter("settings", {
      contract: new ValueContract(Type.scalar(ValueKind.OBJECT)),
      hasDefault: true,
      default: { nested: new Extras({ child: new Extras({ lost: 1.0 }) }) },
    });
  expect(() => new AgentNode("writer", { prompt: "ready", variables: [variable()] })).toThrow(TypeCheckError);
  expect(() => new AgentNode("writer", { prompt: "ready", variables: [variable()] })).toThrow("is not a JSON value");
});

test("declared variable inventory preserves canonical declarations", () => {
  const declarations = [
    new Parameter("tone", {
      contract: new ValueContract(Type.scalar(ValueKind.STRING)),
      hasDefault: true,
      default: "concise",
    }),
    new Parameter("count", {
      contract: new ValueContract(new Type({ kind: ValueKind.INTEGER, bounds: [1n, 12n] })),
      hasDefault: true,
      default: 4n,
    }),
  ];
  const node = new AgentNode("writer", { variables: declarations });

  const inventory = node.declaredVariables();

  expect(Array.isArray(inventory)).toBe(true);
  expect(inventory).not.toBe(node.variables);
  expect(inventory.length).toBe(2);
  inventory.forEach((actual, index) => expect(actual).toBe(declarations[index]!));
  expect(new FuncNode("ordinary").declaredVariables()).toEqual([]);
  expect(new AgentNode("empty").declaredVariables()).toEqual([]);
});
