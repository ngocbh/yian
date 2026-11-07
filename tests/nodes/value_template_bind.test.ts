/**
 * Value templates, an authored container whose nested strings are templates. `ValueTemplate`'s
 * checks, `iterSourceStrings`, `mapSourceStrings` and `bindingInputGroups` walk and rebuild
 * template trees (including nested typed bindings), and the origin walkers (`iterOrigins`,
 * `mapOrigins`, `rewriteOriginCarrier`) reach the template's origins and every nested binding,
 * while a plain record or list binding source stays opaque to iteration. The binder
 * (`bindParams`) evaluates each nested string of a template and checks each nested typed binding,
 * while a plain record or list source stays a literal.
 *
 * `rewriteOriginCarrier` returns the rebuilt carrier. Shared parts stay
 * shared: a contract shared by a `Parameter` and a `Binding`, a binding rewritten twice with one
 * `seen`, and a record shared by two bindings' sources each come back as one rebuilt object.
 */

import { expect, test } from "bun:test";

import { OriginRef } from "../../src/diagnostics.ts";
import { type RootRead, SourceRef, ValueRoute } from "../../src/expr/lineage.ts";
import { BindingCollector, BindingError, bindParams } from "../../src/nodes/binding.ts";
import {
  Binding,
  bindingInputGroups,
  ContractOrigins,
  iterOriginCarrier,
  iterSourceStrings,
  mapSourceStrings,
  type OriginCarrier,
  Parameter,
  rewriteOriginCarrier,
  ValueContract,
  ValueTemplate,
  type ValuePath,
} from "../../src/nodes/contracts.ts";
import { Type, typeFor, type ValueRecord, ValueKind, VariablePool } from "../../src/typesys/index.ts";
import { readTypedefs } from "../../src/typesys/types.ts";

const INTEGER = Type.scalar(ValueKind.INTEGER);

/** Prefix an origin's key with `m:`. */
/** Every origin `iterOriginCarrier` yields for `carrier`, in order. */
function originsOf(carrier: OriginCarrier): OriginRef[] {
  return [...iterOriginCarrier(carrier, new Set())];
}

function rename(origin: OriginRef): OriginRef {
  return new OriginRef("m:" + origin.key);
}

// --- carrier --- //

test("a value template requires a record or a list", () => {
  expect(new ValueTemplate({ a: "${n.output}" }).value).toEqual({ a: "${n.output}" });
  expect(new ValueTemplate(["${n.output}"]).origins).toEqual([]);
  expect(() => new ValueTemplate("${n.output}" as never)).toThrow(TypeError);
  expect(() => new ValueTemplate("${n.output}" as never)).toThrow(
    /^value template must hold a record or array, got string$/,
  );
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

  expect(originsOf(binding)).toEqual([outer, leaf, nested, nestedContract]);

  const mapped = rewriteOriginCarrier(binding, rename, new Map());
  expect(originsOf(mapped)).toEqual([
    new OriginRef("m:outer"),
    new OriginRef("m:leaf"),
    new OriginRef("m:nested"),
    new OriginRef("m:nested/contract"),
  ]);
  // The original carrier graph is untouched.
  expect(originsOf(binding)).toEqual([outer, leaf, nested, nestedContract]);
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

  expect(originsOf(rewritten)).toEqual([
    new OriginRef("m:outer"),
    new OriginRef("m:leaf"),
    new OriginRef("m:nested"),
  ]);
  expect(originsOf(binding)).toEqual([
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

test("both origin walkers reach a binding nested two levels deep", () => {
  const binding = twoLevelBinding();
  const keys = ["outer", "leaf", "middle", "inner"];

  expect(originsOf(binding).map((origin) => origin.key)).toEqual(keys);
  const rewritten = rewriteOriginCarrier(binding, rename, new Map());
  expect(originsOf(rewritten).map((origin) => origin.key)).toEqual(keys.map((key) => "m:" + key));
});

test("a plain binding with a record source keeps its contents opaque", () => {
  const binding = new Binding(
    { item: new Binding("${c.output}", { origin: new OriginRef("inner") }) },
    { origin: new OriginRef("o") },
  );

  expect(originsOf(binding)).toEqual([new OriginRef("o")]);
});

// --- binder --- //

const BUY = typeFor("Buy", readTypedefs({ Buy: { action: "Literal(buy)", conviction: "float" } }));

/** Return a pool holding each entry of `outputs` as that node's output. */
function poolOf(outputs: Record<string, unknown> = {}): VariablePool {
  const pool = new VariablePool();
  for (const [nodeId, value] of Object.entries(outputs)) pool.set(nodeId, value);
  return pool;
}

/** Return the `BindingError` `call` throws. */
function bindingError(call: () => unknown): BindingError {
  try {
    call();
  } catch (error) {
    expect(error).toBeInstanceOf(BindingError);
    return error as BindingError;
  }
  throw new Error("call did not throw");
}

test("template strings evaluate with types and unescape", () => {
  const template = new ValueTemplate({ a: "${n.output}", b: [1n, "${m.output}"], c: "$$x" });

  const record = bindParams(
    [new Parameter("d")],
    new Map([["d", new Binding(template)]]),
    poolOf({ n: 3.5, m: ["x", 2n] }),
  );

  expect(record).toEqual({ d: { a: 3.5, b: [1n, ["x", 2n]], c: "$x" } });
});

test("embedded template strings are stringified", () => {
  const record = bindParams(
    [new Parameter("d")],
    new Map([["d", new ValueTemplate(["n=${n.output}", "plain"])]]),
    poolOf({ n: 4n }),
  );

  expect(record).toEqual({ d: ["n=4", "plain"] });
});

test("a plain record source stays a literal", () => {
  const source = { a: "${n.output}", b: ["${m.output}"], c: "$$x" };

  const record = bindParams([new Parameter("d")], new Map([["d", new Binding(source)]]), poolOf({ n: 1n, m: 2n }));

  expect(record).toEqual({ d: { a: "${n.output}", b: ["${m.output}"], c: "$$x" } });
  expect(bindParams([new Parameter("d")], new Map([["d", ["${n.output}"]]]), poolOf({ n: 1n }))).toEqual({
    d: ["${n.output}"],
  });
});

test("a nested typed binding passes a valid value unchanged", () => {
  const template = new ValueTemplate({
    order: new Binding("${d.output}", { contract: new ValueContract(BUY) }),
    n: 1n,
  });
  const value = { action: "buy", conviction: 1n };

  const record = bindParams([new Parameter("p")], new Map([["p", template]]), poolOf({ d: value }));

  expect(record).toEqual({ p: { order: { action: "buy", conviction: 1n }, n: 1n } });
  expect(typeof ((record["p"] as Record<string, Record<string, unknown>>)["order"]!["conviction"])).toBe("bigint");
});

test("a nested typed binding rejects a missing field at its origin", () => {
  const nestedOrigin = new OriginRef("site/type/path/order");
  const template = new ValueTemplate({
    order: new Binding("${d.output}", { origin: nestedOrigin, contract: new ValueContract(BUY) }),
  });

  const error = bindingError(() =>
    bindParams(
      [new Parameter("p")],
      new Map([["p", new Binding(template, { origin: new OriginRef("site") })]]),
      poolOf({ d: { conviction: 0.5 } }),
    ),
  );

  expect(error.message).toContain('["action"]');
  expect(error.failureCode).toBe("binding.type");
  expect(error.origin).toEqual(nestedOrigin);
  expect(error.message.startsWith("binding 'p.order' at action:")).toBe(true);
});

test("a nested typed binding prefers the nearest contract origin", () => {
  const fieldOrigin = new OriginRef("site/type/path/b/0/conviction");
  const contract = new ValueContract(BUY, new ContractOrigins({ entries: [[["conviction"], fieldOrigin]] }));
  const template = new ValueTemplate({ b: [new Binding("${d.output}", { origin: new OriginRef("n"), contract })] });

  const error = bindingError(() =>
    bindParams([new Parameter("p")], new Map([["p", template]]), poolOf({ d: { action: "buy", conviction: "high" } })),
  );

  expect(error.failureCode).toBe("binding.type");
  expect(error.origin).toEqual(fieldOrigin);
  expect(error.message.startsWith("binding 'p.b[0]' at conviction:")).toBe(true);
});

test("a nested typed binding over a subtree checks the evaluated record", () => {
  const template = new ValueTemplate({
    order: new Binding({ action: "buy", conviction: "${c.output}" }, { contract: new ValueContract(BUY) }),
  });

  expect(bindParams([new Parameter("p")], new Map([["p", template]]), poolOf({ c: 0.7 }))).toEqual({
    p: { order: { action: "buy", conviction: 0.7 } },
  });
  const error = bindingError(() => bindParams([new Parameter("p")], new Map([["p", template]]), poolOf({ c: "high" })));
  expect(error.failureCode).toBe("binding.type");
});

test("template evaluation errors point at the leaf origin", () => {
  const leafOrigin = new OriginRef("site/path/a");
  const template = new ValueTemplate({ a: "${n.output:?}" }, [[["a"], leafOrigin]]);

  const error = bindingError(() =>
    bindParams(
      [new Parameter("p")],
      new Map([["p", new Binding(template, { origin: new OriginRef("site") })]]),
      poolOf(),
    ),
  );

  expect(error.failureCode).toBe("binding.evaluate");
  expect(error.origin).toEqual(leafOrigin);
  expect(error.message.startsWith("binding 'p.a':")).toBe(true);
});

test("a nested binding origin locates errors inside it", () => {
  const nestedOrigin = new OriginRef("site/type/path/b/1");
  const template = new ValueTemplate({ b: [1n, new Binding({ c: "${n.output:?}" }, { origin: nestedOrigin })] });

  const error = bindingError(() =>
    bindParams(
      [new Parameter("p")],
      new Map([["p", new Binding(template, { origin: new OriginRef("site") })]]),
      poolOf(),
    ),
  );

  expect(error.failureCode).toBe("binding.evaluate");
  expect(error.origin).toEqual(nestedOrigin);
  expect(error.message.startsWith("binding 'p.b[1].c':")).toBe(true);
});

test("the MAP item scope reaches template strings", () => {
  const template = new ValueTemplate({ i: "${item}", n: 1n });

  expect(bindParams([new Parameter("p")], new Map([["p", template]]), poolOf(), { item: 5n })).toEqual({
    p: { i: 5n, n: 1n },
  });
});

test("template routes record each part and its source", () => {
  const template = new ValueTemplate({ a: "${n.output}", b: [1n, "${m.output}"], c: "lit" });
  const roots: Record<string, ValueRecord> = { n: { output: 3n }, m: { output: "x" } };

  const resolve = (names: readonly string[]): RootRead => {
    const [node, field] = names as [string, string];
    const seq = node === "n" ? 1 : 2;
    const source = new SourceRef("run", seq, [field]);
    return { used: 2, present: true, value: roots[node]![field]!, why: null, source };
  };

  const collector = new BindingCollector();
  const delivered = bindParams(
    [new Parameter("p")],
    new Map([["p", new Binding(template)]]),
    poolOf({ n: 3n, m: "x" }),
    { collector, sourceResolver: resolve },
  );
  const captured = collector.freeze();

  expect(captured.boundInputs).toEqual(delivered);
  expect(delivered).toEqual({ p: { a: 3n, b: [1n, "x"], c: "lit" } });
  expect(captured.evaluatedKeys).toEqual(new Set(["p"]));
  const route = captured.route.children[0]!;
  expect(route.kind).toBe("record");
  expect(route.names).toEqual(["a", "b", "c"]);
  expect(route.children[1]!.kind).toBe("list");
  expect(route.children[1]!.children[0]).toEqual(ValueRoute.literal(1n));
  expect(route.children[0]!.source).toEqual(new SourceRef("run", 1, ["output"]));
  expect(route.children[1]!.children[1]!.source).toEqual(new SourceRef("run", 2, ["output"]));
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
});

test("rewriting keeps a literal-only binding source by identity", () => {
  const literal = { a: [1n] };
  expect(rewriteOriginCarrier(new Binding(literal), rename, new Map()).source).toBe(literal);
});

test("a container shared by two binding sources is rebuilt once", () => {
  const shared = { item: new Binding("${c.output}", { origin: new OriginRef("inner") }) };
  const first = new Binding(shared, { origin: new OriginRef("first") });
  const second = new Binding(shared, { origin: new OriginRef("second") });
  const seen = new Map<object, object>();

  const newFirst = rewriteOriginCarrier(first, rename, seen);
  const newSecond = rewriteOriginCarrier(second, rename, seen);

  expect(newFirst.source).not.toBe(shared);
  expect(newSecond.source).toBe(newFirst.source);
  expect(((newFirst.source as Record<string, unknown>)["item"] as Binding).origin).toEqual(new OriginRef("m:inner"));
});
