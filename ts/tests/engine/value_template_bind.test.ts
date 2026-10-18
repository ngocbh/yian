/**
 * Value templates, an authored container whose nested strings are templates: the carrier half of
 * `tests/engine/test_value_template_bind.py`. `ValueTemplate`'s checks, `iterSourceStrings`,
 * `mapSourceStrings` and `bindingInputGroups` walk and rebuild template trees (including nested
 * typed bindings), and the origin walkers (`iterOrigins`, `mapOrigins`, `rewriteOriginCarrier`)
 * reach the template's origins and every nested binding, while a plain record or list binding
 * source stays opaque to iteration.
 *
 * Changed: Python's rewrite mutates a `copy.deepcopy` in place; `rewriteOriginCarrier` returns the
 * rebuilt carrier instead, so the "private copy" is its return value. Dataclass `==` is
 * `toEqual`.
 *
 * TS-only: a `Parameter` and a `Binding` sharing one contract, rewritten with one `seen`, share
 * one rebuilt contract; a binding whose record source holds a binding, rewritten first at the top
 * level and then inside a `ValueTemplate` with one `seen`, is rebuilt once, and every origin the
 * rebuilt template yields is mapped.
 *
 * Left for later: the binder tests (`test_template_strings_evaluate_with_types_and_unescape`
 * onward) land with `nodes/binding.ts` (`resolveSource`/`bindParams`, then binding capture).
 *
 * Not ported: the rows that differ from a good one only by a tuple where Python wants a list or
 * the reverse (a tuple template, a list of origins, a list path): TS has only arrays; and
 * `test_value_template_keys_must_be_strings_throughout`, which has no counterpart, since a
 * record key is always a string.
 */

import { expect, test } from "bun:test";

import { OriginRef } from "../../src/diagnostics.ts";
import {
  Binding,
  bindingInputGroups,
  ContractOrigins,
  iterOriginCarrier,
  iterSourceStrings,
  mapSourceStrings,
  Parameter,
  rewriteOriginCarrier,
  ValueContract,
  ValueTemplate,
  type ValuePath,
} from "../../src/nodes/contracts.ts";
import { Type, ValueKind } from "../../src/typesys/index.ts";

const INTEGER = Type.scalar(ValueKind.INTEGER);

/** Prefix an origin's key with `m:`. */
function rename(origin: OriginRef): OriginRef {
  return new OriginRef("m:" + origin.key);
}

// --- carrier --- //

test("a value template requires a record or a list", () => {
  expect(new ValueTemplate({ a: "${n.output}" }).value).toEqual({ a: "${n.output}" });
  expect(new ValueTemplate(["${n.output}"]).origins).toEqual([]);
  expect(() => new ValueTemplate("${n.output}" as never)).toThrow(TypeError);
});

test("a value template validates its origins", () => {
  const origin = new OriginRef("site/path/a");
  expect(new ValueTemplate({ a: "x" }, [[["a"], origin]]).origins).toEqual([[["a"], origin]]);
  expect(() => new ValueTemplate({ a: "x" }, [[["a"], "site" as never]])).toThrow(TypeError);
});

// --- iterSourceStrings --- //

test("iterSourceStrings yields a plain string at the root", () => {
  expect([...iterSourceStrings("${a.output}")]).toEqual([[[], "${a.output}"]]);
  expect([...iterSourceStrings(new Binding("${a.output}"))]).toEqual([[[], "${a.output}"]]);
});

test("iterSourceStrings skips literals and plain containers", () => {
  expect([...iterSourceStrings(5n)]).toEqual([]);
  expect([...iterSourceStrings(null)]).toEqual([]);
  expect([...iterSourceStrings({ a: "${x.output}" })]).toEqual([]);
  expect([...iterSourceStrings(new Binding(["${x.output}"]))]).toEqual([]);
  // Exactly one binding layer is unwrapped; a binding of a binding is a literal.
  expect([...iterSourceStrings(new Binding(new Binding("${x.output}")))]).toEqual([]);
});

test("iterSourceStrings walks template trees and nested bindings", () => {
  const template = new ValueTemplate({
    a: "${n.output}",
    b: [1n, "${m.output}", { c: "text" }],
    d: new Binding("${d.output}", { contract: new ValueContract(INTEGER) }),
    e: new Binding({ f: "${e.output}" }),
  });

  const expected: [ValuePath, string][] = [
    [["a"], "${n.output}"],
    [["b", 1], "${m.output}"],
    [["b", 2, "c"], "text"],
    [["d"], "${d.output}"],
    [["e", "f"], "${e.output}"],
  ];
  expect([...iterSourceStrings(template)]).toEqual(expected);
  expect([...iterSourceStrings(new Binding(template))]).toEqual(expected);
});

// --- mapSourceStrings --- //

const upper = (text: string): string => text.toUpperCase();

test("mapSourceStrings maps a plain string and keeps its binding", () => {
  const origin = new OriginRef("b");
  const contract = new ValueContract(INTEGER);
  const mapped = mapSourceStrings(new Binding("${a}", { origin, contract }), upper);

  expect(mapped).toEqual(new Binding("${A}", { origin, contract }));
  expect(mapSourceStrings("x", upper)).toBe("X");
});

test("mapSourceStrings leaves literals and plain containers alone", () => {
  const plain = { a: "${x.output}" };
  expect(mapSourceStrings(plain, upper)).toBe(plain);
  expect(mapSourceStrings(7n, upper)).toBe(7n);
  const binding = new Binding(["${x.output}"]);
  expect(mapSourceStrings(binding, upper)).toBe(binding);
});

test("mapSourceStrings rebuilds template carriers", () => {
  const leafOrigin = new OriginRef("site/path/a");
  const nestedOrigin = new OriginRef("site/type/path/d");
  const contract = new ValueContract(INTEGER, new ContractOrigins({ default: nestedOrigin }));
  const template = new ValueTemplate(
    {
      a: "${n.output}",
      b: [1n, "${m.output}"],
      d: new Binding("${d.output}", { origin: nestedOrigin, contract }),
    },
    [[["a"], leafOrigin]],
  );
  const outer = new Binding(template, { origin: new OriginRef("site"), contract: null });

  const mapped = mapSourceStrings(outer, (text) => text.replace("output", "out"));

  expect(mapped).toBeInstanceOf(Binding);
  const binding = mapped as Binding;
  expect(binding.origin).toEqual(new OriginRef("site"));
  expect(binding.source).toBeInstanceOf(ValueTemplate);
  const source = binding.source as ValueTemplate;
  expect(source.origins).toEqual(template.origins);
  expect(source.value).toEqual({
    a: "${n.out}",
    b: [1n, "${m.out}"],
    d: new Binding("${d.out}", { origin: nestedOrigin, contract }),
  });
  // The input is not mutated.
  expect((template.value as Record<string, unknown>)["a"]).toBe("${n.output}");
});

// --- bindingInputGroups --- //

test("bindingInputGroups for a string is the parameter", () => {
  expect(bindingInputGroups("qty", "${a.output}")).toEqual([["qty", "${a.output}"]]);
  expect(bindingInputGroups("qty", new Binding("${a.output}"))).toEqual([["qty", "${a.output}"]]);
  expect(bindingInputGroups("qty", 5n)).toEqual([]);
  expect(bindingInputGroups("qty", { a: "${x.output}" })).toEqual([]);
});

test("bindingInputGroups name each nested string by its escaped path", () => {
  const template = new ValueTemplate({
    a: "${n.output}",
    b: [1n, "${m.output}"],
    "x/y": "${p.output}",
    "t~u": new Binding("${q.output}", { contract: new ValueContract(INTEGER) }),
  });

  expect(bindingInputGroups("d", new Binding(template))).toEqual([
    ["d/a", "${n.output}"],
    ["d/b/1", "${m.output}"],
    ["d/x~1y", "${p.output}"],
    ["d/t~0u", "${q.output}"],
  ]);
});

// --- origins --- //

test("binding origins include the template and nested bindings", () => {
  const outer = new OriginRef("outer");
  const leaf = new OriginRef("leaf");
  const nested = new OriginRef("nested");
  const nestedContract = new OriginRef("nested/contract");
  const binding = new Binding(
    new ValueTemplate(
      {
        a: "${n.output}",
        d: new Binding("${d.output}", {
          origin: nested,
          contract: new ValueContract(INTEGER, new ContractOrigins({ default: nestedContract })),
        }),
      },
      [[["a"], leaf]],
    ),
    { origin: outer },
  );

  expect([...binding.iterOrigins()]).toEqual([outer, leaf, nested, nestedContract]);

  const mapped = binding.mapOrigins(rename);
  expect([...mapped.iterOrigins()]).toEqual([
    new OriginRef("m:outer"),
    new OriginRef("m:leaf"),
    new OriginRef("m:nested"),
    new OriginRef("m:nested/contract"),
  ]);
  // The original carrier graph is untouched.
  expect([...binding.iterOrigins()]).toEqual([outer, leaf, nested, nestedContract]);
});

test("rewriting a carrier reaches template and nested origins", () => {
  const binding = new Binding(
    new ValueTemplate(
      { a: "${n.output}", b: [new Binding("${d.output}", { origin: new OriginRef("nested") })] },
      [[["a"], new OriginRef("leaf")]],
    ),
    { origin: new OriginRef("outer") },
  );

  const rewritten = rewriteOriginCarrier(binding, rename, new Map());

  expect([...rewritten.iterOrigins()]).toEqual([
    new OriginRef("m:outer"),
    new OriginRef("m:leaf"),
    new OriginRef("m:nested"),
  ]);
  expect([...binding.iterOrigins()]).toEqual([
    new OriginRef("outer"),
    new OriginRef("leaf"),
    new OriginRef("nested"),
  ]);
});

/** Return a template binding whose nested binding holds another one. */
function twoLevelBinding(): Binding {
  const inner = new Binding("${c.output}", { origin: new OriginRef("inner") });
  const middle = new Binding({ item: inner, n: 1n }, { origin: new OriginRef("middle") });
  return new Binding(new ValueTemplate({ order: middle }, [[["order", "item"], new OriginRef("leaf")]]), {
    origin: new OriginRef("outer"),
  });
}

test("every origin walker reaches a binding nested two levels deep", () => {
  const binding = twoLevelBinding();
  const keys = ["outer", "leaf", "middle", "inner"];

  expect([...binding.iterOrigins()].map((origin) => origin.key)).toEqual(keys);
  const mapped = binding.mapOrigins(rename);
  expect([...mapped.iterOrigins()].map((origin) => origin.key)).toEqual(keys.map((key) => "m:" + key));
  const rewritten = rewriteOriginCarrier(binding, rename, new Map());
  expect([...rewritten.iterOrigins()].map((origin) => origin.key)).toEqual(keys.map((key) => "m:" + key));
  expect(rewritten).toEqual(mapped);
});

test("a plain binding with a record source keeps its contents opaque", () => {
  const binding = new Binding(
    { item: new Binding("${c.output}", { origin: new OriginRef("inner") }) },
    { origin: new OriginRef("o") },
  );

  expect([...binding.iterOrigins()]).toEqual([new OriginRef("o")]);
});

// --- TS-only rows --- //

test("a contract shared by a parameter and a binding is rebuilt once", () => {
  const contract = new ValueContract(INTEGER, new ContractOrigins({ default: new OriginRef("type") }));
  const parameter = new Parameter("x", { contract, origin: new OriginRef("param") });
  const binding = new Binding("${a.output}", { origin: new OriginRef("bind"), contract });
  const seen = new Map<object, object>();

  const newParameter = rewriteOriginCarrier(parameter, rename, seen);
  const newBinding = rewriteOriginCarrier(binding, rename, seen);

  expect(newParameter.contract).toBe(newBinding.contract);
  expect(newParameter.contract!.origins.default).toEqual(new OriginRef("m:type"));
  expect(newParameter.origin).toEqual(new OriginRef("m:param"));
  expect(newBinding.origin).toEqual(new OriginRef("m:bind"));
  expect(rewriteOriginCarrier(parameter, rename, seen)).toBe(newParameter);
  expect(parameter.contract).toBe(contract);
});

test("a binding reached at the top level and inside a template is rebuilt once", () => {
  const inner = new Binding("${c.output}", { origin: new OriginRef("inner") });
  const middle = new Binding({ item: inner, n: 1n }, { origin: new OriginRef("middle") });
  const template = new ValueTemplate({ order: middle }, [[["order", "item"], new OriginRef("leaf")]]);
  const seen = new Map<object, object>();

  const top = rewriteOriginCarrier(middle, rename, seen);
  const newTemplate = rewriteOriginCarrier(template, rename, seen);

  expect((newTemplate.value as Record<string, unknown>)["order"]).toBe(top);
  expect(top.origin).toEqual(new OriginRef("m:middle"));
  expect(((top.source as Record<string, unknown>)["item"] as Binding).origin).toEqual(new OriginRef("m:inner"));
  expect((top.source as Record<string, unknown>)["n"]).toBe(1n);
  const keys = [...iterOriginCarrier(newTemplate, new Set())].map((origin) => origin.key);
  expect(keys).toEqual(["m:leaf", "m:middle", "m:inner"]);
  // A literal-only container keeps its identity.
  const literal = { a: [1n] };
  expect(rewriteOriginCarrier(new Binding(literal), rename, new Map()).source).toBe(literal);
});
