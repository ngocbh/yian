/** The `call` kind's parse. */

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
import { assembledWith, type CaseRow, handChild, loadError, parametersFromYaml, runCase } from "../helpers/declarations.ts";

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

/** Return a descriptor's fields, leaving out the parse metadata. */
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
  ...[
    "a_missing_argument_names_the_call",
    "an_undeclared_argument_in_a_value_entry_call",
    "an_undeclared_argument_names_the_argument",
  ].map((name) => ({
    id: `load-map-call/compose_call_argument_labels__${name}`,
    call: assembledWith({ child: def("input:\n  y: !int\n", "x", new ValueContract(Type.scalar(ValueKind.INTEGER))) }),
  })),
])("assemble: $id", runCase);
