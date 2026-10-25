/**
 * Strict type inference for parameters of generated inline composite flows: a port of
 * `tests/engine/test_inline_composite_types.py`.
 *
 * Ported: every test but `test_non_string_mapping_key_is_rejected` (see Changed), through
 * `inferInlineBindingType`, `requireProvenType`, `canonicalizeInlineType` and
 * `canonicalTypeLabel`. Where Python checks a message fragment, the row pins the whole message.
 *
 * TS-only: a typed `Binding` nested in a `ValueTemplate` infers as its declared type (the
 * docstring's rule, which no Python row checks).
 *
 * Changed: YAML ints are `bigint`s (`7n` infers `int`; a `number` infers `float`). TS values have
 * no tuples or arbitrary objects, so `test_unproven_or_unsupported_literal_is_rejected` replaces
 * its `(1, 2)` and `object()` params with one `undefined` param, whose message names the type with
 * `pythonTypeName`; `test_non_string_mapping_key_is_rejected` is dropped, as a record key is
 * always text. Lists print as JSON (`["absent"]`), and so does a record key in a literal's path
 * (`c.input.payload["items"]`, rule `messages`). The no-mutation checks compare a type with a
 * second one built the same way, where Python compares with a pickled copy.
 *
 * Pinned at the end (`assembled`, the inference errors of inline bodies):
 * `load-types/compose_nested_template_types__{an_empty_list_in_an_inline_call_template,
 * inline_call_template_list_of_mixed_types_fails}` and
 * `load-types/compose_union_narrowing__an_inline_body_target_cannot_infer_a_member_field`.
 */

import { describe, expect, test } from "bun:test";

import {
  canonicalizeInlineType,
  inferInlineBindingType,
  requireProvenType,
} from "../../src/compose/build/inline_inputs.ts";
import { LoadDiagnostics, type LoadError } from "../../src/compose/helpers/errors.ts";
import { type DiagnosticFrame, SourceRange } from "../../src/diagnostics.ts";
import { Binding, PartialRecordContract, ValueContract, ValueTemplate } from "../../src/nodes/contracts.ts";
import { typeFor } from "../../src/typesys/types.ts";
import {
  buildValueAs,
  canonicalTypeLabel,
  isListKind,
  listElementKind,
  Type,
  TypeCheckError,
  ValueKind,
} from "../../src/typesys/values.ts";
import { assembled, type CaseRow, loadError, runCase } from "../compose/declarations.ts";

const LIST_VALUE_KINDS = Object.values(ValueKind).filter(isListKind);

/** The label of a list kind with no element type, e.g. `list(str)`. */
function listLabel(kind: ValueKind): string {
  const element = listElementKind(kind);
  return `list(${element === null ? "Optional(Any)" : element})`;
}

const INT = Type.scalar(ValueKind.INTEGER);
const STR = Type.scalar(ValueKind.STRING);

/** Diagnostics over a blank 100-line `flow.yaml`, so any range in it renders. */
function diagnostics(): LoadDiagnostics {
  return new LoadDiagnostics("flow.yaml", `${" ".repeat(100)}\n`.repeat(100));
}

/** The one frame of `error`. */
function onlyFrame(error: LoadError): DiagnosticFrame {
  expect(error.diagnostic.frames.length).toBe(1);
  return error.diagnostic.frames[0]!;
}

/** Fail if inference unexpectedly asks for a producer type. */
function unusedProducerType(): Type {
  throw new Error("literal/template inference must not ask for a producer type");
}

/** Fail if inference unexpectedly asks for a MAP item type. */
function unusedItemType(): Type {
  throw new Error("literal/template inference must not ask for a MAP item type");
}

/** Infer `value` with no producer or item type available. */
function infer(
  value: unknown,
  options: { label?: string; parentInputTypes?: ReadonlyMap<string, Type> } = {},
): Type {
  return inferInlineBindingType(value, {
    parentInputTypes: options.parentInputTypes ?? new Map(),
    producerType: unusedProducerType,
    itemType: unusedItemType,
    bindingLabel: options.label ?? "each.input.value",
  });
}

/** The inference error message for binding `label` failing for `reason`. */
function message(label: string, reason: string): string {
  const name = label.split(".").at(-1)!;
  return (
    `cannot infer type for inline binding '${label}': ${reason}; declare its type with a tag ` +
    `(\`${name}: !<type> ...\`) or use an explicitly typed named \`defs:\` body`
  );
}

/** A record type with `fields` and every field required unless `required` is given. */
function record(
  fields: Record<string, Type>,
  options: { required?: readonly string[]; nullable?: boolean } = {},
): Type {
  const map = new Map(Object.entries(fields));
  return new Type({
    kind: ValueKind.OBJECT,
    fields: map,
    required: new Set(options.required ?? map.keys()),
    nullable: options.nullable ?? false,
  });
}

const NOT_SCALAR = "a whole-value expression has no supported static scalar type";

describe("inline input inference", () => {
  test.each([
    [null, ValueKind.NONE],
    [false, ValueKind.BOOLEAN],
    [7n, ValueKind.INTEGER],
    [2.5, ValueKind.NUMBER],
    ["plain text", ValueKind.STRING],
    ["$$ remains a top-level string", ValueKind.STRING],
  ])("literal scalar types: %p", (value, kind) => {
    expect(infer(value)).toEqual(Type.scalar(kind));
  });

  test.each([
    ["${null}", ValueKind.NONE],
    ["${none}", ValueKind.NONE],
    ["${true}", ValueKind.BOOLEAN],
    ["${false}", ValueKind.BOOLEAN],
    ["${42}", ValueKind.INTEGER],
    ["${2.5}", ValueKind.NUMBER],
    ['${"literal"}', ValueKind.STRING],
    ["${'literal'}", ValueKind.STRING],
  ])("lone scalar expression token types: %s", (source, kind) => {
    expect(infer(source)).toEqual(Type.scalar(kind));
  });

  test.each([
    "prefix ${producer.output}",
    "${producer.output} suffix",
    "${left.output}${right.output}",
    "prefix ${left.output} middle ${right.output} suffix",
  ])("embedded or multiple template spans are strings: %s", (source) => {
    expect(infer(source)).toEqual(STR);
  });

  test.each([
    "${[1, 2]}",
    "${1 + 2}",
    "${1 == 2}",
    "${true and false}",
    "${not true}",
    '${upper("x")}',
    "${left | right}",
    "${left :- 1}",
    '${left :? "required"}',
    "${-1}",
  ])("a lone non-scalar expression is rejected: %s", (source) => {
    const error = loadError(() => infer(source, { label: "call.input.payload" }));
    expect(error.message).toBe(message("call.input.payload", NOT_SCALAR));
  });

  test("a mapping literal is a sorted closed record", () => {
    const inferred = infer({ z: true, a: 1n, nested: { text: "literal" } });

    expect(inferred.kind).toBe(ValueKind.OBJECT);
    expect([...inferred.fields!.keys()]).toEqual(["a", "nested", "z"]);
    expect(inferred.required).toEqual(new Set(["a", "nested", "z"]));
    expect(inferred.fields).toEqual(
      new Map([
        ["a", INT],
        ["nested", record({ text: STR })],
        ["z", Type.scalar(ValueKind.BOOLEAN)],
      ]),
    );
  });

  test("an empty mapping literal is an exact empty record", () => {
    expect(infer({})).toEqual(new Type({ kind: ValueKind.OBJECT, fields: new Map(), required: new Set() }));
  });

  test.each([
    [["a", "b"], new Type({ kind: ValueKind.LIST_STRING, element: STR })],
    [[false, true], new Type({ kind: ValueKind.LIST_BOOLEAN, element: Type.scalar(ValueKind.BOOLEAN) })],
    [[1n, 2n], new Type({ kind: ValueKind.LIST_INTEGER, element: INT })],
    [[1.5, 2.5], new Type({ kind: ValueKind.LIST_NUMBER, element: Type.scalar(ValueKind.NUMBER) })],
    [[1n, 2.5], new Type({ kind: ValueKind.LIST_NUMBER, element: Type.scalar(ValueKind.NUMBER) })],
    [[null, null], new Type({ kind: ValueKind.LIST_ANY, element: Type.scalar(ValueKind.NONE) })],
    [
      [[1n], [2n]],
      new Type({ kind: ValueKind.LIST_ANY, element: new Type({ kind: ValueKind.LIST_INTEGER, element: INT }) }),
    ],
    [[{ x: 1n }, { x: 2n }], new Type({ kind: ValueKind.LIST_OBJECT, element: record({ x: INT }) })],
  ])("a non-empty literal list retains its structural element: %p", (value, expected) => {
    expect(infer(value)).toEqual(expected);
  });

  test("equivalent record elements ignore authored key order", () => {
    const inferred = infer([
      { a: 1n, b: "x" },
      { b: "y", a: 2n },
    ]);
    expect(inferred).toEqual(new Type({ kind: ValueKind.LIST_OBJECT, element: record({ a: INT, b: STR }) }));
    expect([...inferred.element!.fields!.keys()]).toEqual(["a", "b"]);
  });

  test("nested template strings scan as templates", () => {
    const inferred = infer(
      new ValueTemplate({ deep: { value: "${input.x}" }, ref: "${input.x}", items: ["n=${input.x}", "$$"] }),
      { parentInputTypes: new Map([["x", INT]]) },
    );

    expect(inferred).toEqual(
      record({
        deep: record({ value: INT }),
        ref: INT,
        items: new Type({ kind: ValueKind.LIST_STRING, element: STR }),
      }),
    );
  });

  test("a typed binding nested in a template takes its declared type", () => {
    const declared = Type.scalar(ValueKind.DATE);
    const inferred = infer(
      new ValueTemplate({ when: new Binding("${input.x}", { contract: new ValueContract(declared) }) }),
      { parentInputTypes: new Map([["x", INT]]) },
    );

    expect(inferred.fields!.get("when")).toBe(declared);
  });

  const EMPTY = "empty literal list at each.input.value has no provable element type";
  const MIXED = "literal list at each.input.value has heterogeneous element types";

  test.each([
    [[], EMPTY],
    [[1n, "one"], MIXED],
    [[null, 1n], MIXED],
    [[true, 1n], MIXED],
    [[[1n], [1.5]], MIXED],
    [[{ x: 1n }, { y: 1n }], MIXED],
    [[{ x: 1n }, { x: 1.5 }], MIXED],
    [undefined, "literal at each.input.value has unsupported type 'undefined'"],
  ])("an unproven or unsupported literal is rejected: %p", (value, reason) => {
    expect(loadError(() => infer(value)).message).toBe(message("each.input.value", reason));
  });

  test.each(["mapping", "list"])("an active literal cycle is rejected: %s", (containerKind) => {
    let value: unknown;
    let at: string;
    if (containerKind === "mapping") {
      const mapping: Record<string, unknown> = {};
      mapping["self"] = mapping;
      [value, at] = [mapping, 'each.input.value["self"]'];
    } else {
      const list: unknown[] = [];
      list.push(list);
      [value, at] = [list, "each.input.value[0]"];
    }

    expect(loadError(() => infer(value)).message).toBe(
      message("each.input.value", `cyclic literal container encountered at ${at}`),
    );
  });

  test("a repeated non-cyclic alias is allowed", () => {
    const shared = { value: 3n };
    const inferred = infer({ left: shared, right: shared });
    expect(inferred.fields!.get("left")).toEqual(inferred.fields!.get("right")!);
  });

  test("a direct parent input reference uses the declared type without callbacks", () => {
    const declared = record({ value: INT });
    expect(infer("${input.record.value}", { parentInputTypes: new Map([["record", declared]]) })).toEqual(INT);
  });

  test("a whole unrefined object reference is allowed but copied", () => {
    const declared = Type.scalar(ValueKind.OBJECT);

    const inferred = infer("${input.payload}", { parentInputTypes: new Map([["payload", declared]]) });

    expect(inferred).toEqual(declared);
    expect(inferred).not.toBe(declared);
  });

  test("a direct producer reference passes the request context and returns a copy", () => {
    const sourceRange = new SourceRange(31, 8, 31, 32);
    const outputType = record({ result: STR });
    const calls: [string, string, SourceRange | null][] = [];

    const inferred = inferInlineBindingType("${prepare.output}", {
      parentInputTypes: new Map(),
      producerType: (nodeId, bindingLabel, bindingRange) => {
        calls.push([nodeId, bindingLabel, bindingRange]);
        return outputType;
      },
      bindingLabel: "call.input.payload",
      bindingRange: sourceRange,
    });

    expect(calls).toEqual([["prepare", "call.input.payload", sourceRange]]);
    expect(inferred).toEqual(outputType);
    expect(inferred).not.toBe(outputType);
  });

  test("a direct item reference passes the request context and walks fields", () => {
    const sourceRange = new SourceRange(42, 6, 42, 18);
    const item = record({ name: STR });
    const calls: [string, SourceRange | null][] = [];

    const inferred = inferInlineBindingType("${item.name}", {
      parentInputTypes: new Map(),
      producerType: unusedProducerType,
      itemType: (bindingLabel, bindingRange) => {
        calls.push([bindingLabel, bindingRange]);
        return item;
      },
      bindingLabel: "each.input.name",
      bindingRange: sourceRange,
    });

    expect(calls).toEqual([["each.input.name", sourceRange]]);
    expect(inferred).toEqual(STR);
  });

  test("a whole item reference returns a fresh nullable type without mutation", () => {
    const sourceRange = new SourceRange(47, 6, 47, 13);
    const makeItem = () => record({ name: STR }, { nullable: true });
    const item = makeItem();
    const calls: [string, SourceRange | null][] = [];

    const inferred = inferInlineBindingType("${item}", {
      parentInputTypes: new Map(),
      producerType: unusedProducerType,
      itemType: (bindingLabel, bindingRange) => {
        calls.push([bindingLabel, bindingRange]);
        return item;
      },
      bindingLabel: "each.input.row",
      bindingRange: sourceRange,
    });

    expect(calls).toEqual([["each.input.row", sourceRange]]);
    expect(inferred).toEqual(item);
    expect(inferred).not.toBe(item);
    expect(inferred.nullable).toBe(true);
    expect(item).toEqual(makeItem());
  });

  test("a computed item expression is rejected without invoking the item provider", () => {
    let calls = 0;

    const error = loadError(() =>
      inferInlineBindingType("${item.value + 1}", {
        parentInputTypes: new Map(),
        producerType: unusedProducerType,
        itemType: () => {
          calls += 1;
          return INT;
        },
        bindingLabel: "each.input.computed",
      }),
    );

    expect(error.message).toBe(message("each.input.computed", NOT_SCALAR));
    expect(calls).toBe(0);
  });

  // Each value is wrapped in its own row, since `test.each` spreads an array row into arguments.
  test.each([
    [3n],
    ["prefix ${item.name}"],
    ["${input.record.name}"],
    [{ nested: "${item.name}" }],
    [["${item.name}"]],
  ])(
    "the item provider is demand-driven only for a direct item reference: %p",
    (value) => {
      let calls = 0;

      inferInlineBindingType(value, {
        parentInputTypes: new Map([["record", record({ name: STR })]]),
        producerType: unusedProducerType,
        itemType: () => {
          calls += 1;
          return STR;
        },
        bindingLabel: "each.input.value",
      });

      expect(calls).toBe(0);
    },
  );

  test.each([
    [false, false, false, false],
    [true, false, false, true],
    [false, true, false, true],
    [false, false, true, true],
  ])(
    "a direct record walk accumulates every nullable step without mutation: root %p, middle %p, value %p",
    (rootNullable, middleNullable, valueNullable, expectedNullable) => {
      const valueType = valueNullable ? new Type({ kind: ValueKind.INTEGER, nullable: true }) : INT;
      const makeRoot = () =>
        record(
          {
            middle: record(
              { value: valueType },
              { required: valueNullable ? [] : ["value"], nullable: middleNullable },
            ),
          },
          { nullable: rootNullable },
        );
      const root = makeRoot();

      const inferred = infer("${input.record.middle.value}", { parentInputTypes: new Map([["record", root]]) });

      expect(inferred).toEqual(new Type({ kind: ValueKind.INTEGER, nullable: expectedNullable }));
      expect(inferred).not.toBe(valueType);
      expect(root).toEqual(makeRoot());
    },
  );

  test("a nullable producer root propagates to a selected field", () => {
    const outputType = record({ value: INT }, { nullable: true });

    const inferred = inferInlineBindingType("${source.output.value}", {
      parentInputTypes: new Map(),
      producerType: () => outputType,
      bindingLabel: "call.input.value",
    });

    expect(inferred).toEqual(new Type({ kind: ValueKind.INTEGER, nullable: true }));
  });

  const PARTIAL = new PartialRecordContract([
    ["value", new ValueContract(INT)],
    ["note", null],
  ]);

  test("a known partial producer field is available to inline inference", () => {
    const inferred = inferInlineBindingType("${source.output.value}", {
      parentInputTypes: new Map(),
      producerType: () => PARTIAL,
      bindingLabel: "call.input.value",
    });

    expect(inferred).toEqual(INT);
  });

  test("inline inference rejects a field on a scalar producer", () => {
    const error = loadError(() =>
      inferInlineBindingType("${source.output.x}", {
        parentInputTypes: new Map(),
        producerType: () => new ValueContract(INT),
        bindingLabel: "call.input.value",
      }),
    );

    expect(error.message).toBe(
      message("call.input.value", "reference ${source.output.x} reads field 'x' on type int, which has no fields"),
    );
  });

  test.each(["${source.output}", "${source.output.note}"])(
    "an opaque partial producer read stays unavailable to inline inference: %s",
    (reference) => {
      const error = loadError(() =>
        inferInlineBindingType(reference, {
          parentInputTypes: new Map(),
          producerType: () => PARTIAL,
          bindingLabel: "call.input.value",
        }),
      );

      expect(error.message).toBe(
        message("call.input.value", `reference ${reference} has no statically proven exact type`),
      );
    },
  );

  test("a nullable item root propagates to a selected field", () => {
    const item = record({ value: INT }, { nullable: true });

    const inferred = inferInlineBindingType("${item.value}", {
      parentInputTypes: new Map(),
      producerType: unusedProducerType,
      itemType: () => item,
      bindingLabel: "each.input.value",
    });

    expect(inferred).toEqual(new Type({ kind: ValueKind.INTEGER, nullable: true }));
  });

  test("a zero-step direct walk always returns a fresh type", () => {
    const sourceType = new Type({ kind: ValueKind.INTEGER, nullable: true });

    const inferred = infer("${input.value}", { parentInputTypes: new Map([["value", sourceType]]) });

    expect(inferred).toEqual(sourceType);
    expect(inferred).not.toBe(sourceType);
  });

  test.each([
    ["${input}", "reference ${input} must be input.<name>[.field]"],
    ["${input.missing}", "reference ${input.missing} points to unknown parent input 'missing'"],
    ["${input.opaque.value}", "reference ${input.opaque.value} traverses 'value' without proven record fields"],
    ["${input.record.missing}", "reference ${input.record.missing} reads unknown record field 'missing'"],
    ["${item.value}", "reference ${item.value} is only typed inside an outer MAP input"],
    ["${system.now}", "reference ${system.now} has no statically proven system type"],
    ["${producer}", "reference ${producer} must select producer.output[.field] or an AGENT's producer.new_messages"],
    ["${producer.value}", "reference ${producer.value} must select producer.output[.field] or an AGENT's producer.new_messages"],
    ["${producer()}", NOT_SCALAR],
    ["${producer.output + 1}", NOT_SCALAR],
    ['${upper("x")}', NOT_SCALAR],
  ])("an unproven direct reference is a located named-def error: %s", (source, reason) => {
    const sourceRange = new SourceRange(51, 4, 51, 28);
    const inputs = new Map([
      ["opaque", Type.scalar(ValueKind.OBJECT)],
      ["record", record({ known: STR })],
    ]);

    const error = loadError(() =>
      inferInlineBindingType(source, {
        parentInputTypes: inputs,
        producerType: unusedProducerType,
        bindingLabel: "call.input.payload",
        bindingRange: sourceRange,
        diagnostics: diagnostics(),
      }),
    );

    expect(onlyFrame(error).range).toEqual(sourceRange);
    expect(error.message).toBe(message("call.input.payload", reason));
  });

  test.each(["producer", "item"])("a provider's load error propagates unchanged: %s", (provider) => {
    const sourceRange = new SourceRange(64, 3, 64, 22);
    const expected = diagnostics().error("provider could not prove a type", { range: sourceRange });
    const calls: unknown[][] = [];

    const error = loadError(() =>
      inferInlineBindingType(provider === "producer" ? "${worker.output}" : "${item}", {
        parentInputTypes: new Map(),
        producerType: (nodeId, bindingLabel, bindingRange) => {
          calls.push([nodeId, bindingLabel, bindingRange]);
          throw expected;
        },
        itemType: (bindingLabel, bindingRange) => {
          calls.push([bindingLabel, bindingRange]);
          throw expected;
        },
        bindingLabel: "each.input.payload",
        bindingRange: sourceRange,
        diagnostics: diagnostics(),
      }),
    );

    expect(error).toBe(expected);
    expect(calls).toEqual(
      provider === "producer"
        ? [["worker", "each.input.payload", sourceRange]]
        : [["each.input.payload", sourceRange]],
    );
  });

  test("an inference error keeps the binding range and hint", () => {
    const sourceRange = new SourceRange(17, 9, 17, 11);

    const error = loadError(() =>
      inferInlineBindingType([], {
        parentInputTypes: new Map(),
        producerType: unusedProducerType,
        bindingLabel: "each.input.items",
        bindingRange: sourceRange,
        diagnostics: diagnostics(),
      }),
    );

    expect(onlyFrame(error).range).toEqual(sourceRange);
    expect(error.message).toBe(
      message("each.input.items", "empty literal list at each.input.items has no provable element type"),
    );
  });

  test("a template scan error keeps the binding range and hint", () => {
    const sourceRange = new SourceRange(23, 7, 23, 21);

    const error = loadError(() =>
      inferInlineBindingType("${unterminated", {
        parentInputTypes: new Map(),
        producerType: unusedProducerType,
        bindingLabel: "call.input.query",
        bindingRange: sourceRange,
        diagnostics: diagnostics(),
      }),
    );

    expect(onlyFrame(error).range).toEqual(sourceRange);
    expect(error.message).toBe(
      message("call.input.query", "invalid binding template: unbalanced '${' in template '${unterminated'"),
    );
  });

  test.each(LIST_VALUE_KINDS)("every list kind requires a recursively proven element: %s", (kind) => {
    const proven = new Type({ kind, element: STR });

    expect(requireProvenType(proven, { bindingLabel: "call.input.items" })).toBe(proven);
    expect(loadError(() => requireProvenType(Type.scalar(kind), { bindingLabel: "call.input.items" })).message).toBe(
      message("call.input.items", `list type '${listLabel(kind)}' has no proven element type`),
    );
  });

  test("list proof recurses into a present malformed element", () => {
    const malformedElement = record({ optional_but_nonnullable: STR }, { required: [] });
    const typ = new Type({ kind: ValueKind.LIST_ANY, element: malformedElement });

    expect(loadError(() => requireProvenType(typ, { bindingLabel: "call.input.items" })).message).toBe(
      message("call.input.items", "record field 'optional_but_nonnullable' is neither required nor nullable"),
    );
  });

  test.each([
    [new Type({ kind: ValueKind.LIST_ANY, element: Type.scalar(ValueKind.NONE) }), "list(None)"],
    [
      new Type({
        kind: ValueKind.LIST_ANY,
        element: new Type({ kind: ValueKind.LIST_ANY, element: Type.scalar(ValueKind.DATE) }),
      }),
      "list(list(date))",
    ],
    [new Type({ kind: ValueKind.LIST_ANY, element: Type.scalar(ValueKind.DATETIME) }), "list(datetime)"],
    [new Type({ kind: ValueKind.LIST_ANY, element: Type.scalar(ValueKind.FILE) }), "list(file)"],
  ])("list(Any) keeps proven special and nested elements: %#", (typ, label) => {
    expect(requireProvenType(typ, { bindingLabel: "call.input.items" })).toBe(typ);
    expect(canonicalTypeLabel(typ)).toBe(label);
  });

  test("an unrefined object is proven whole but not dottable", () => {
    const opaque = Type.scalar(ValueKind.OBJECT);
    const parentInputTypes = new Map([["payload", opaque]]);

    expect(requireProvenType(opaque, { bindingLabel: "call.input.payload" })).toBe(opaque);
    expect(infer("${input.payload}", { parentInputTypes })).toEqual(opaque);
    expect(loadError(() => infer("${input.payload.value}", { parentInputTypes })).message).toBe(
      message("each.input.value", "reference ${input.payload.value} traverses 'value' without proven record fields"),
    );
  });

  test("a record rejects required names absent from its fields", () => {
    const typ = new Type({
      kind: ValueKind.OBJECT,
      fields: new Map([["present", STR]]),
      required: new Set(["present", "absent"]),
    });

    expect(loadError(() => requireProvenType(typ, { bindingLabel: "call.input.record" })).message).toBe(
      message("call.input.record", 'record requires unknown fields ["absent"]'),
    );
  });

  test("a record rejects any non-required non-nullable field", () => {
    const typ = record({ required: INT, ordinary_string: STR }, { required: ["required"] });

    expect(loadError(() => requireProvenType(typ, { bindingLabel: "loop.input.record" })).message).toBe(
      message("loop.input.record", "record field 'ordinary_string' is neither required nor nullable"),
    );
  });

  test("a record accepts required nullable, optional nullable and required None fields", () => {
    const typ = record(
      {
        required_nullable: new Type({ kind: ValueKind.STRING, nullable: true }),
        optional_nullable: new Type({ kind: ValueKind.INTEGER, nullable: true }),
        required_none: Type.scalar(ValueKind.NONE),
      },
      { required: ["required_nullable", "required_none"] },
    );

    expect(requireProvenType(typ, { bindingLabel: "call.input.record" })).toBe(typ);
  });

  test("recursive proof rejects a malformed nested record", () => {
    const root = record({ child: record({ unproven: STR }, { required: [] }) });

    expect(loadError(() => requireProvenType(root, { bindingLabel: "call.input.record" })).message).toBe(
      message("call.input.record", "record field 'unproven' is neither required nor nullable"),
    );
  });

  test("a direct reference proves the entire root before selecting a field", () => {
    const root = record({ selected: INT, malformed_sibling: Type.scalar(ValueKind.LIST_ANY) });

    const error = loadError(() =>
      infer("${input.record.selected}", { parentInputTypes: new Map([["record", root]]), label: "call.input.value" }),
    );

    expect(error.message).toBe(
      message("call.input.value", "list type 'list(Optional(Any))' has no proven element type"),
    );
  });

  test("a structural proof error keeps the source range and hint", () => {
    const sourceRange = new SourceRange(73, 5, 73, 27);

    const error = loadError(() =>
      requireProvenType(Type.scalar(ValueKind.LIST_OBJECT), {
        bindingLabel: "each.input.rows",
        bindingRange: sourceRange,
        diagnostics: diagnostics(),
      }),
    );

    expect(onlyFrame(error).range).toEqual(sourceRange);
    expect(error.message).toBe(message("each.input.rows", "list type 'list(object)' has no proven element type"));
  });

  test("canonicalization deep-copies, sorts and keeps required membership", () => {
    const requiredNullable = new Type({ kind: ValueKind.STRING, nullable: true });
    const variant = new Type({ kind: ValueKind.STRING, tags: new Set(["é", "z", 'a,"b']) });
    const listedVariant = new Type({ kind: ValueKind.LIST_STRING, element: variant });
    const makeOriginal = () =>
      record(
        {
          é: Type.scalar(ValueKind.BOOLEAN),
          optional: new Type({ kind: ValueKind.INTEGER, nullable: true }),
          required_nullable: requiredNullable,
          a: listedVariant,
        },
        { required: ["é", "required_nullable", "a"] },
      );
    const original = makeOriginal();

    const canonical = canonicalizeInlineType(original);

    expect(canonical).toEqual(original);
    expect(canonical).not.toBe(original);
    expect([...canonical.fields!.keys()]).toEqual(["a", "optional", "required_nullable", "é"]);
    expect(canonical.required).toEqual(new Set(["a", "required_nullable", "é"]));
    expect(canonical.fields!.get("required_nullable")!.nullable).toBe(true);
    expect(canonical.fields!.get("a")).not.toBe(listedVariant);
    expect(canonical.fields!.get("a")!.element).not.toBe(variant);
    expect(canonical.fields!.get("a")!.element!.tags).toEqual(variant.tags!);
    expect(original).toEqual(makeOriginal());
    expect([...original.fields!.keys()]).toEqual(["é", "optional", "required_nullable", "a"]);
  });

  test("canonicalization turns a refined record's undefined required into an empty set", () => {
    const typ = new Type({
      kind: ValueKind.OBJECT,
      fields: new Map([["optional", new Type({ kind: ValueKind.STRING, nullable: true })]]),
    });

    const canonical = canonicalizeInlineType(typ);

    expect(canonical.required).toEqual(new Set());
    expect(typ.required).toBeUndefined();
  });

  test("an inline type copy keeps bounded nested validation", () => {
    const declared = record({
      scores: typeFor("list(Optional(float(-0.5..1)))", new Map()),
      count: typeFor("int(1..12)", new Map()),
    });
    const inferred = infer("${input.payload}", { parentInputTypes: new Map([["payload", declared]]) });
    const copied = canonicalizeInlineType(inferred);

    expect(copied).toEqual(declared);
    expect(copied).not.toBe(declared);
    expect(copied.fields!.get("scores")!.element).not.toBe(declared.fields!.get("scores")!.element);
    expect(buildValueAs(copied, { scores: [null, 1n], count: 4n }).toObject()).toEqual({
      scores: [null, 1n],
      count: 4n,
    });
    let error: unknown;
    try {
      buildValueAs(copied, { scores: [1.1], count: 4n });
    } catch (caught) {
      error = caught;
    }
    expect(error).toBeInstanceOf(TypeCheckError);
    expect((error as TypeCheckError).path).toEqual(["scores", 0]);
  });
});

describe("canonical type labels", () => {
  test.each([
    ValueKind.NONE,
    ValueKind.STRING,
    ValueKind.INTEGER,
    ValueKind.NUMBER,
    ValueKind.DATE,
    ValueKind.DATETIME,
    ValueKind.BOOLEAN,
    ValueKind.OBJECT,
    ValueKind.FILE,
  ])("scalar labels use author vocabulary: %s", (kind) => {
    expect(canonicalTypeLabel(Type.scalar(kind))).toBe(kind);
  });

  test.each(LIST_VALUE_KINDS)("every list kind label uses its structural element: %s", (kind) => {
    const typ = new Type({ kind, element: new Type({ kind: ValueKind.INTEGER, nullable: true }) });
    expect(canonicalTypeLabel(typ)).toBe("list(Optional(int))");
  });

  test.each(LIST_VALUE_KINDS)("flat list label uses its storage kind: %s", (kind) => {
    expect(canonicalTypeLabel(Type.scalar(kind))).toBe(listLabel(kind));
  });

  test("variant label sorts and JSON-quotes adversarial tags", () => {
    const typ = new Type({ kind: ValueKind.STRING, tags: new Set(["é", "z", 'a,"b']) });
    expect(canonicalTypeLabel(typ)).toBe('Literal("a,\\"b", z, é)');
  });

  test("record label sorts and JSON-quotes adversarial keys", () => {
    const fields = new Map([
      ["é", Type.scalar(ValueKind.BOOLEAN)],
      ['quote"key', Type.scalar(ValueKind.STRING)],
      ["a: int, b", Type.scalar(ValueKind.INTEGER)],
    ]);
    const typ = new Type({ kind: ValueKind.OBJECT, fields, required: new Set(fields.keys()) });
    expect(canonicalTypeLabel(typ)).toBe('{"a: int, b":int,"quote\\"key":str,"é":bool}');
  });

  test("the nullable wrapper is outermost for a structural record label", () => {
    const typ = new Type({
      kind: ValueKind.OBJECT,
      fields: new Map([["value", Type.scalar(ValueKind.STRING)]]),
      required: new Set(["value"]),
      nullable: true,
    });
    expect(canonicalTypeLabel(typ)).toBe('Optional({"value":str})');
  });
});

// ---------- harvested conformance cases ----------

test.each<CaseRow>([
  { id: "load-types/compose_nested_template_types__an_empty_list_in_an_inline_call_template", call: assembled },
  { id: "load-types/compose_nested_template_types__inline_call_template_list_of_mixed_types_fails", call: assembled },
  { id: "load-types/compose_union_narrowing__an_inline_body_target_cannot_infer_a_member_field", call: assembled },
])("assemble: $id", runCase);
