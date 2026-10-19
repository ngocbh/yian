/**
 * The `call` kind's parse: a port of `tests/compose/test_kind_call_parse.py`.
 *
 * Ported: `test_call_parse_matches_descriptor_and_records_ranges`,
 * `test_call_parse_requires_call`, `test_call_parse_rejects_non_string_asserts`.
 *
 * Changed: Python compares the descriptor to a `CallDescriptor` with the parse metadata and the
 * compiler-owned `child_label` and `call_site` excluded; TS compares its class and fields. The
 * requires-call and asserts rows pin the whole message and its range (the node body; the
 * `asserts:` value), where Python matches a fragment.
 *
 * The harvested cases whose error the CALL parse raises are pinned in the TS mirrors of the Python
 * files they come from: `non_text_fields.test.ts` (`call:`), `node_input_shape.test.ts`
 * (`input:`), `node_when_hint.test.ts` (`when:`) and `engine/inline_composite_bodies.test.ts`
 * (the key check).
 *
 * Pinned at the end (`assembledWith`, each def built by hand with `handChild`), the CALL build's
 * cases from files with no TS mirror yet:
 * `load-map-call/compose_call_argument_labels__an_authored_call_node_keeps_its_binding_wording`
 * (`tests/compose/test_call_argument_labels.py`) and
 * `load-core/engine_limit_overrides__call_nesting_follows_the_variable`
 * (`tests/engine/test_limit_overrides.py`), whose `c0` calls `c1`, which calls `c2`.
 */

import { afterEach, expect, test } from "bun:test";

import type { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { siteKey } from "../../src/compose/nodes/base.ts";
import { CallDescriptor } from "../../src/compose/nodes/call.ts";
import { type SourceCursor, SourceDocument } from "../../src/compose/source/index.ts";
import { forget } from "../../src/defaults.ts";
import { SourceRange } from "../../src/diagnostics.ts";
import { CallNode } from "../../src/nodes/call/node.ts";
import { ValueContract } from "../../src/nodes/contracts.ts";
import { Type, ValueKind } from "../../src/typesys/values.ts";
import { assembledWith, type CaseRow, handChild, loadError, parametersFromYaml, runCase } from "./declarations.ts";

afterEach(() => forget());

const FLOW = `nodes:
  step:
    kind: call
    call: child_flow
    inputs:
      x: \${input.x}
`;

/** Return the cursor over node `nodeId` of `text`'s `nodes:`. */
function readerFor(text: string, nodeId: string): SourceCursor {
  return SourceDocument.parse(text).root.child("nodes").child(nodeId);
}

/** Return a descriptor's fields, leaving out what Python's `compare=False` leaves out. */
function fieldsOf(desc: CallDescriptor): Record<string, unknown> {
  const {
    range: _range,
    fieldRanges: _fieldRanges,
    assertRanges: _assertRanges,
    childLabel: _childLabel,
    callSite: _callSite,
    ...fields
  } = desc;
  return fields;
}

test("call parse matches the descriptor and records ranges", () => {
  const desc = CallDescriptor.parse("step", readerFor(FLOW, "step"));

  expect(desc).toBeInstanceOf(CallDescriptor);
  expect(fieldsOf(desc)).toEqual(
    fieldsOf(new CallDescriptor({ id: "step", call: "child_flow", inputs: { x: "${input.x}" } })),
  );
  expect(desc.kind).toBe("call");
  expect(desc.range).toBeInstanceOf(SourceRange);
  expect(desc.fieldRanges.get(siteKey("field", "call"))).toBeInstanceOf(SourceRange);
  expect(desc.fieldRanges.get(siteKey("field", "inputs"))).toBeInstanceOf(SourceRange);
});

test("call parse requires call", () => {
  const flow = "nodes:\n  step:\n    kind: call\n    inputs:\n      x: ${input.x}\n";

  const cursor = readerFor(flow, "step");

  const error = loadError(() => CallDescriptor.parse("step", cursor));

  expect(error.message).toBe("node 'step' (kind=call): missing required field 'call'");
  expect(error.diagnostic.frames.at(-1)!.range).toEqual(cursor.range!);
});

test("call parse rejects non-string asserts", () => {
  const flow = "nodes:\n  step:\n    kind: call\n    call: child_flow\n    asserts:\n      - 5\n";

  const cursor = readerFor(flow, "step");

  const error = loadError(() => CallDescriptor.parse("step", cursor));

  expect(error.message).toBe("node 'step' (kind=call): `asserts:` must be a list of strings");
  expect(error.diagnostic.frames.at(-1)!.range).toEqual(cursor.rangeOf("asserts")!);
});

// ---------- harvested conformance cases ----------

/** A def built by hand: `input:` text `inputs` and the one node `id`, whose output it returns. */
function def(inputs: string, id: string, producer: ValueContract | CallNode): ComposedFlow {
  return handChild({
    inputs: parametersFromYaml(inputs, new Map()),
    producers: { [id]: producer },
    output: `\${${id}.output}`,
  });
}

/** The `input:` text of `c0`, `c1` and `c2`. */
const X_INPUT = "input:\n  x: !str\n";
/** A def over `x` whose node `step` calls `callee`, built as `child`. */
const callingDef = (callee: string, child: ComposedFlow) =>
  def(X_INPUT, "step", new CallNode("step", { flowId: callee, child: child.compiled }));
const c2 = def(X_INPUT, "step", new ValueContract(Type.scalar(ValueKind.STRING)));

test.each<CaseRow>([
  {
    id: "load-map-call/compose_call_argument_labels__an_authored_call_node_keeps_its_binding_wording",
    call: assembledWith({ child: def("input:\n  y: !int\n", "x", new ValueContract(Type.scalar(ValueKind.INTEGER))) }),
  },
  {
    id: "load-core/engine_limit_overrides__call_nesting_follows_the_variable",
    call: assembledWith({ c0: callingDef("c1", callingDef("c2", c2)) }),
  },
])("assemble: $id", runCase);
