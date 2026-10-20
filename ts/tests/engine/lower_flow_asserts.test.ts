/**
 * `lowerFlowAsserts`, the pure flow-assert lowering pass: a port of
 * `tests/engine/test_lower_flow_asserts.py`. It takes a flow's `asserts:` list plus the name
 * environment (flow inputs, node ids, producer contracts) and lowers each assert into a node-local
 * assert for the synthesized START/END, plus the synth END params and wiring that make cross-node
 * post-asserts structural. These rows pin the exact rewritten forms and the synth naming scheme.
 *
 * Changed: the synth params compare with `toEqual`, as Python's `==` compares them, and the
 * wiring is a `Map`. A flow assert's text holding a lone surrogate is a located load error (Python
 * yian loads it); the TS-only row pins the message.
 */

import { expect, test } from "bun:test";

import { LoadDiagnostics, LoadError } from "../../src/compose/helpers/errors.ts";
import { lowerFlowAsserts } from "../../src/compose/validate/asserts.ts";
import { OriginRef, SourceRange } from "../../src/diagnostics.ts";
import { conditionRefs } from "../../src/expr/index.ts";
import {
  Assertion,
  ContractOrigins,
  type OutputContract,
  Parameter,
  PartialRecordContract,
  ValueContract,
} from "../../src/nodes/contracts.ts";
import { Type, ValueKind } from "../../src/typesys/values.ts";
import { loadError } from "../compose/declarations.ts";

// A small shared environment: two opaque producers and two flow inputs. Opaque producers are absent
// from the contract map and stay lenient in the Type walker.
const PRODUCERS = new Map<string, OutputContract>();
const VALID_TARGETS = new Set(["emit", "classify"]);
const FLOW_INPUTS = new Set(["age", "threshold"]);

/** Lower `asserts` (a string is its own display) in the shared environment, overridden by `options`. */
function lower(
  asserts: readonly (string | Assertion)[],
  options: {
    flowInputs?: ReadonlySet<string>;
    validTargets?: ReadonlySet<string>;
    producers?: ReadonlyMap<string, OutputContract>;
    assertRanges?: readonly (SourceRange | null)[];
    diagnostics?: LoadDiagnostics;
  } = {},
) {
  return lowerFlowAsserts(
    asserts.map((a) => (a instanceof Assertion ? a : new Assertion(a, a))),
    options.flowInputs ?? FLOW_INPUTS,
    options.validTargets ?? VALID_TARGETS,
    options.producers ?? PRODUCERS,
    options.assertRanges ?? null,
    { diagnostics: options.diagnostics ?? null },
  );
}

/** Project executable text while keeping carrier assertions explicit. */
const expressions = (assertions: readonly Assertion[]) => assertions.map((assertion) => assertion.expression);

/** A record with one required number field `score`. */
const SCORE_VIEW = new Type({
  kind: ValueKind.OBJECT,
  fields: new Map([["score", Type.scalar(ValueKind.NUMBER)]]),
  required: new Set(["score"]),
});

// ---- PRE (boundary) asserts: rewrite `input.` to `output.` (START's coerced output) ----

test("pre whole span rewrites input to output", () => {
  const r = lower(["${input.age >= 18}"]);
  expect(expressions(r.startPre)).toEqual(["${output.age >= 18}"]);
  expect(r.endPost).toEqual([]);
  expect(r.synthParams).toEqual([]);
  expect(r.synthWiring).toEqual(new Map());
});

test("pre bare spelling rewrites input to output", () => {
  const r = lower(["input.age >= 18"]);
  expect(expressions(r.startPre)).toEqual(["output.age >= 18"]);
  expect(r.synthParams).toEqual([]);
});

test.each([
  ["age >= 18", "output.age >= 18"],
  ["${age >= 18}", "${output.age >= 18}"],
])("pre unqualified flow input rewrites in every condition spelling: %s", (source, expected) => {
  const r = lower([source]);
  expect(expressions(r.startPre)).toEqual([expected]);
  expect(r.endPost).toEqual([]);
  expect(r.synthParams).toEqual([]);
  expect(r.synthWiring).toEqual(new Map());
});

test("an unqualified flow input precedes a same-named node", () => {
  const r = lower(["age >= 18", "age.output >= 18"], { validTargets: new Set(["age"]) });

  expect(expressions(r.startPre)).toEqual(["output.age >= 18"]);
  expect(expressions(r.endPost)).toEqual(["__assert_age >= 18"]);
  expect(r.synthParams).toEqual([new Parameter("__assert_age")]);
  expect(r.synthWiring).toEqual(new Map([["__assert_age", "${age.output:-null}"]]));
});

// ---- POST asserts: synth END params ----

test("post bare producer ref lowers to a synth param", () => {
  const r = lower(["emit.output.n >= 1"]);
  // The bare form rewrites without braces.
  expect(expressions(r.endPost)).toEqual(["__assert_emit.n >= 1"]);
  expect(r.synthParams).toEqual([new Parameter("__assert_emit")]);
  expect(r.synthWiring).toEqual(new Map([["__assert_emit", "${emit.output:-null}"]]));
  expect(r.startPre).toEqual([]);
});

test("post dotted field ref lowers to a synth field", () => {
  const r = lower(['classify.output.label == "x"']);
  expect(expressions(r.endPost)).toEqual(['__assert_classify.label == "x"']);
  expect(r.synthParams).toEqual([new Parameter("__assert_classify")]);
  expect(r.synthWiring).toEqual(new Map([["__assert_classify", "${classify.output:-null}"]]));
});

test("post same producer two fields share one synth", () => {
  const r = lower(["emit.output.n >= 1", "emit.output.m >= 2"]);
  expect(r.synthParams.length).toBe(1);
  expect(r.synthParams[0]!.name).toBe("__assert_emit");
  expect(expressions(r.endPost)).toEqual(["__assert_emit.n >= 1", "__assert_emit.m >= 2"]);
  expect(r.synthWiring).toEqual(new Map([["__assert_emit", "${emit.output:-null}"]]));
});

test("post mixed producer and input give two synths", () => {
  const r = lower(["emit.output.n >= input.threshold"]);
  expect(new Set(r.synthParams.map((p) => p.name))).toEqual(new Set(["__assert_emit", "__assert_input_threshold"]));
  expect(expressions(r.endPost)).toEqual(["__assert_emit.n >= __assert_input_threshold"]);
  // The producer edge defaults to null; the input edge does not (an input is always present).
  expect(r.synthWiring.get("__assert_emit")).toBe("${emit.output:-null}");
  expect(r.synthWiring.get("__assert_input_threshold")).toBe("${input.threshold}");
});

test("post mixed producer and unqualified input give two synths", () => {
  const r = lower(["emit.output.n >= age"]);

  expect(expressions(r.endPost)).toEqual(["__assert_emit.n >= __assert_input_age"]);
  expect(r.synthParams).toEqual([new Parameter("__assert_emit"), new Parameter("__assert_input_age")]);
  expect(r.synthWiring).toEqual(
    new Map([
      ["__assert_emit", "${emit.output:-null}"],
      ["__assert_input_age", "${input.age}"],
    ]),
  );
});

// ---- Errors ----

test("the system namespace is forbidden", () => {
  expect(() => lower(["system.run_id != null"])).toThrow(LoadError);
});

test("a dangling producer ref raises", () => {
  expect(() => lower(["nonexist.output.x > 0"])).toThrow(LoadError);
});

test("a dangling input ref raises", () => {
  // A `${input.X}` naming an undeclared input is a loud LoadError.
  expect(loadError(() => lower(["input.nope != []"])).message).toContain("nope");
});

test("an unknown unqualified name remains a located error", () => {
  const rng = new SourceRange(3, 5, 3, 18);
  const diagnostics = new LoadDiagnostics("flow.yaml", "first\nsecond\n    missing >= 18\n");

  const error = loadError(() => lower(["missing >= 18"], { assertRanges: [rng], diagnostics }));

  expect(error.message).toContain("unknown namespace 'missing'");
  expect(error.diagnostic.frames[0]!.range).toEqual(rng);
});

test("a dotted input name is not unqualified shorthand", () => {
  expect(loadError(() => lower(["profile.age >= 18"], { flowInputs: new Set(["profile"]) })).message).toContain(
    "unknown namespace 'profile'",
  );
});

test("an unknown field on a record producer raises", () => {
  // A typed producer is needed so the dotted walk runs (an absent producer contract stays lenient).
  const error = loadError(() =>
    lower(["synth.output.badfield >= 0"], {
      validTargets: new Set(["synth"]),
      producers: new Map([["synth", new ValueContract(SCORE_VIEW)]]),
    }),
  );
  expect(error.message).toContain("badfield");
});

test("an unknown field on a partial record producer raises", () => {
  // A partial producer is opaque by type, not by its closed root shape.
  const partial = new PartialRecordContract([
    ["score", new ValueContract(Type.scalar(ValueKind.NUMBER))],
    ["note", null],
  ]);

  const error = loadError(() =>
    lower(["synth.output.badfield >= 0"], {
      validTargets: new Set(["synth"]),
      producers: new Map([["synth", partial]]),
    }),
  );
  expect(error.message).toContain("badfield");
});

test("the type walk preserves the producer contract and origins", () => {
  // Assert validation projects `.type` without replacing the semantic carrier.
  const contract = new ValueContract(SCORE_VIEW, new ContractOrigins({ default: new OriginRef("producer-output") }));

  const lowered = lower(["synth.output.score >= 0"], {
    validTargets: new Set(["synth"]),
    producers: new Map([["synth", contract]]),
  });

  expect(expressions(lowered.endPost)).toEqual(["__assert_synth.score >= 0"]);
  expect(contract.type).toBe(SCORE_VIEW);
  expect(contract.origins.default).toEqual(new OriginRef("producer-output"));
});

test("assert lowering rejects non-contract producer map values", () => {
  // The flow-assert boundary accepts exact contract values only.
  const call = () =>
    lower(["synth.output != null"], {
      validTargets: new Set(["synth"]),
      producers: new Map([["synth", Type.scalar(ValueKind.STRING) as unknown as OutputContract]]),
    });
  expect(call).toThrow(TypeError);
  expect(call).toThrow(/producer contracts.*ValueContract/);
});

test("a malformed expression raises", () => {
  expect(() => lower(["emit.output.n >= "])).toThrow(LoadError);
});

test("a lowering error reports the display, not the rewritten expression", () => {
  // Compile diagnostics use the carrier's stable authored presentation.
  const assertion = new Assertion("missing.output > 0", "the authored business rule");

  expect(loadError(() => lower([assertion])).message).toContain("authored business rule");
});

test("a flow assert holding a lone surrogate is a located load error", () => {
  const rng = new SourceRange(3, 5, 3, 28);
  const text = "asserts:\n  - input.age >= 18\n  - \"input.age == '\\uD800'\"\n";
  const diagnostics = new LoadDiagnostics("flow.yaml", text);

  const error = loadError(() =>
    lower(["input.age >= 18", "input.age == '\uD800'"], { assertRanges: [null, rng], diagnostics }),
  );

  expect(error.message).toBe("flow field 'asserts[1]': text holding a lone surrogate is not a yian value");
  expect(error.diagnostic.frames[0]!.range).toEqual(rng);
});

// ---- Every lowered string must remain a parseable condition ----

test("all lowered strings parse", () => {
  const r = lower([
    "input.age >= 18",
    "emit.output.n >= 1",
    'classify.output.label == "x"',
    "emit.output.n >= input.threshold",
  ]);
  for (const assertion of [...r.startPre, ...r.endPost]) {
    // `conditionRefs` parses the whole value as a boolean expression; a broken ref rewrite would
    // throw here.
    conditionRefs(assertion.expression);
  }
  // The wiring sources are `${...}` refs too; they parse as well.
  for (const source of r.synthWiring.values()) conditionRefs(source as string);
});

// ---- Ranges stay in compose validation ----

test("ranges are not carried by the lowered product", () => {
  // Positional validation ranges do not cross into the lowered semantic product.
  const rngPre = new SourceRange(3, 1, 3, 20);
  const rngPost = new SourceRange(4, 1, 4, 20);
  const r = lower(["input.age >= 18", "emit.output.n >= 1"], { assertRanges: [rngPre, rngPost] });
  expect(r.startPre.length).toBe(1);
  expect(r.endPost.length).toBe(1);
  expect("preRanges" in r).toBe(false);
  expect("postRanges" in r).toBe(false);
});

test("lowering without ranges needs no padding", () => {
  // A source-free assert lowers without synthetic location state.
  const r = lower(["input.age >= 18"]);
  expect(r.startPre.length).toBe(1);
  expect(r.endPost).toEqual([]);
});

test("lowering changes only the expression and preserves occurrence metadata", () => {
  // Display and origin belong to each occurrence even when the executable text is rewritten.
  const first = new Assertion("input.age >= 18", "age must be adult", { origin: new OriginRef("asserts/0") });
  const second = new Assertion("input.age >= 18", "age must be adult", { origin: new OriginRef("asserts/1") });

  const lowered = lower([first, second]);

  expect(expressions(lowered.startPre)).toEqual(["output.age >= 18", "output.age >= 18"]);
  expect(lowered.startPre.map((a) => a.display)).toEqual(["age must be adult", "age must be adult"]);
  expect(lowered.startPre.map((a) => a.origin)).toEqual([first.origin, second.origin]);
  expect(lowered.startPre[0]).not.toBe(lowered.startPre[1]);
});
