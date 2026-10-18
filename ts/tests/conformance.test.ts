/**
 * The conformance runner's own unit tests: reading a case file, the format check, case values,
 * comparison, the case directory, selection and the report. Not a port of
 * `tests/test_conformance.py`, which `conformance.ts` is.
 */

import { afterEach, describe, expect, test } from "bun:test";
import {
  existsSync,
  mkdirSync,
  mkdtempSync,
  readdirSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { parseDocument } from "yaml";

import {
  CaseFailure,
  checkCase,
  compare,
  decode,
  expectRaises,
  main,
  prepare,
  raised,
  readCase,
  results,
  same,
  setUp,
} from "./conformance.ts";
import type { Case, CaseContext, Kind, KindRunner } from "./conformance.ts";

const CASES = fileURLToPath(new URL("../../conformance/cases/", import.meta.url));

describe("readCase", () => {
  test("reads YAML 1.2 core scalars, ints as bigint", () => {
    expect(readCase("a: 1")).toEqual({ a: 1n });
    expect(readCase("a: 0x1F")).toEqual({ a: 31n });
    expect(readCase("a: 0o17")).toEqual({ a: 15n });
    expect(readCase("a: 0123")).toEqual({ a: 123n });
    expect(readCase("a: 123456789012345678901234567890")).toEqual({
      a: 123456789012345678901234567890n,
    });
    expect(readCase("a: 1.0")).toEqual({ a: 1 });
    expect(readCase("a: 1e3")).toEqual({ a: 1000 });
    expect(Object.is(readCase("a: -0.0")["a"], -0)).toBe(true);
    expect(readCase("a: yes")).toEqual({ a: "yes" });
    expect(readCase("a: 2026-01-01")).toEqual({ a: "2026-01-01" });
    expect(readCase("a: ~")).toEqual({ a: null });
  });

  test("keeps `<<` and `__proto__` as plain keys", () => {
    const merged = readCase("<<: {a: 1}");
    expect(Object.keys(merged)).toEqual(["<<"]);
    expect(merged["<<"]).toEqual({ a: 1n });
    const proto = readCase("__proto__: 1");
    expect(Object.hasOwn(proto, "__proto__")).toBe(true);
    expect(Object.getPrototypeOf(proto)).toBe(Object.prototype);
  });

  test("accepts an alias key naming text", () => {
    expect(readCase("k: &x a\n*x : 1")).toEqual({ k: "a", a: 1n });
  });

  test("has no alias limit", () => {
    const list = Array.from({ length: 10 }, (_, index) => index + 1);
    const read = readCase(`a: &x [${list.join(", ")}]\nb: [${Array(200).fill("*x").join(", ")}]`);
    expect(read["b"]).toEqual(Array(200).fill(list.map(BigInt)));
  });

  test.each([
    ["a: !!str 12", '"tag:yaml.org,2002:str"'],
    ["a: ! 12", '"!"'],
    ["a: !x 1", '"!x"'],
    ["a: !!timestamp 2020-01-01", '"tag:yaml.org,2002:timestamp"'],
  ])("refuses the tag in %p", (text, tag) => {
    expect(() => readCase(text)).toThrow(`a case file has no tags, found ${tag}`);
  });

  test.each([
    ["a: 1\na: 2", 'duplicate key "a"'],
    ["1: x", "mapping key 1 is not text"],
    ["~: 1", "mapping key null is not text"],
    ["[a]: x", 'mapping key ["a"] is not text'],
    [": x", "mapping key null is not text"],
    ["{1: a, b: 1, b: 2}", "mapping key 1 is not text"],
    ["{b: 1, b: 2, 1: a}", 'duplicate key "b"'],
    ["{a: 1, a: 2}: x", 'duplicate key "a"'],
    ["{1: x}: y", "mapping key 1 is not text"],
    ["[{1: a}]: y", "mapping key 1 is not text"],
    ["- a", "a case file holds one mapping"],
    ["", "a case file holds one mapping"],
  ])("refuses %p", (text, message) => {
    expect(() => readCase(text)).toThrow(message);
  });

  test("throws the parser's message on a syntax error", () => {
    const text = "a: [1";
    const message = parseDocument(text).errors[0]?.message;
    expect(message).toBeString();
    expect(() => readCase(text)).toThrow(message);
  });
});

const LOAD = "kind: load\nflow: x\nexpect: {load: ok}\n";
const RUN = "kind: run\nflow: x\nexpect: {output: 1}\n";
const BARE_RESUME = "kind: resume\nflow: x\nexpect: {output: 1}\n";
const RESUME = `${BARE_RESUME}run: {num_workers: 0}\nhost: {crash_after_record: 2}\n`;
const BARE_BACKWARD = "kind: backward\nflow: x\nexpect: {output: 1}\n";
const BACKWARD = `${BARE_BACKWARD}run: {learnable: true}\nbackward: [{feedback: f}]\n`;
const EXPR = "kind: expr\nexpect: {value: 1}\n";
const CLI = "kind: cli\nexpect: {exit: 0}\n";
const TOOL = (results: string) =>
  `${RUN}host: {tools: {t: {parameters: {}, results: ${results}}}}\n`;
const STRICT = "a strict `template` takes no `item` or `typedefs`";
const KINDS = "load, run, resume, backward, expr, value, defaults, fingerprint, cli";

describe("checkCase", () => {
  test.each([
    ["kind: nope\nexpect: {}", `\`kind\` must be one of ${KINDS}, not "nope"`],
    ["expect: {}", `\`kind\` must be one of ${KINDS}, not null`],
    ["kind: 1\nexpect: {}", `\`kind\` must be one of ${KINDS}, not 1`],
    ["kind: toString\nexpect: {}", `\`kind\` must be one of ${KINDS}, not "toString"`],
    [`${LOAD}x: 1`, "unknown key `x`"],
    [`${LOAD}inputs: {}`, "`inputs` is not a key of a `load` case"],
    ["kind: load\nexpect: {load: ok}", "`flow` is required"],
    ["kind: load\nflow: x", "`expect` is required"],
    ["kind: run\nflow: x\nexpect: 1", "`expect` must be a mapping"],
    ["kind: run\nflow: x\nexpect: {}", "`expect` needs `raises`, or `output`"],
    ["kind: run\nflow: x\nexpect: {raises: 1, output: 1}", "`expect` needs `raises`, or `output`"],
    [
      "kind: value\ntype: int\nvalue: 1\nexpect: {value: 1}",
      "`expect` needs `value` and `kind`, or `raises`",
    ],
    ["kind: run\nflow: x\nexpect: {output: 1, x: 1}", "unknown key `expect.x`"],
    ["kind: run\nflow: x\nexpect: {load: ok}", "`expect.load` is not a key of a `run` case"],
    [`${LOAD}python_expect: {load: ok}`, "`python_expect` needs `rules`"],
    [`${LOAD}rules: []\npython_expect: {load: ok}`, "`python_expect` needs `rules`"],
    [`${LOAD}rules: [text]\npython_expect: {}`, "`python_expect` needs `load`"],
    [`${LOAD}rules: text`, "`rules` must be a list of rule ids"],
    [`${LOAD}rules: [1]`, "`rules` must be a list of rule ids"],
    [`${LOAD}rules: .nan\npython_expect: {load: ok}`, "`rules` must be a list of rule ids"],
    [`${CLI}args: [1]`, "`args` must be a list of text"],
    [`${LOAD}rules: [nope, text, bad]`, "unknown rule ids: nope, bad"],
    [`${LOAD}env: 1`, "`env` must be a mapping"],
    [`${LOAD}env: {A: 1}`, "`env` values must be text"],
    [`${LOAD}files: {a: 1}`, "`files` values must be text"],
    ["kind: load\nflow: 1\nexpect: {load: ok}", "`flow` must be text"],
    [`${LOAD}python_flow: [a]`, "`python_flow` must be text"],
    ["kind: fingerprint\nflow: x\nother: 1\nexpect: {fingerprint: same}", "`other` must be text"],
    [`${CLI}stdin: 1`, "`stdin` must be text"],
    [
      "kind: fingerprint\nflow: x\nother: y\npython_flow: z\nexpect: {fingerprint: same}",
      "`python_flow` is not a key of a `fingerprint` case",
    ],
    [`${RUN}run: 1`, "`run` must be a mapping"],
    [`${RUN}run: {x: 1}`, "unknown key `run.x`"],
    [`${RUN}run: {context: yes}`, "`run.context` must be `present`"],
    [`${RESUME}resume: {run_id: a}`, "unknown key `resume.run_id`"],
    [`${RESUME}resume: {context: 1}`, "`resume.context` must be `present`"],
    [`${RUN}backward: [{feedback: f}]`, "`backward` is not a key of a `run` case"],
    [`${BARE_BACKWARD}run: {learnable: true}`, "`backward` is required"],
    [`${BARE_BACKWARD}backward: []`, "`backward` must be a non-empty list"],
    [`${BARE_BACKWARD}backward: f`, "`backward` must be a non-empty list"],
    [`${BARE_BACKWARD}backward: [1]`, "`backward[0]` must be a mapping"],
    [`${BARE_BACKWARD}backward: [{}]`, "`backward[0].feedback` is required"],
    [`${BARE_BACKWARD}backward: [{feedback: f, x: 1}]`, "unknown key `backward[0].x`"],
    [`${RUN}host: 1`, "`host` must be a mapping"],
    [`${RUN}host: {x: 1}`, "unknown key `host.x`"],
    [
      `${RUN}host: {crash_after_record: 1}`,
      "`host.crash_after_record` is not a key of a `run` case",
    ],
    [`${RUN}host: {tools: 1}`, "`host.tools` must be a mapping"],
    [`${RUN}host: {tools: {t: 1}}`, "`host.tools.t` must be a mapping"],
    [`${RUN}host: {tools: {t: {results: []}}}`, "`host.tools.t.parameters` is required"],
    [`${RUN}host: {tools: {t: {parameters: {}}}}`, "`host.tools.t.results` is required"],
    [
      `${RUN}host: {tools: {t: {parameters: {}, results: [], x: 1}}}`,
      "unknown key `host.tools.t.x`",
    ],
    [TOOL("1"), "`host.tools.t.results` must be a list or a mapping of lists"],
    [TOOL("{n: 1}"), "`host.tools.t.results` must be a list or a mapping of lists"],
    [TOOL("[1]"), "`host.tools.t.results[0]` must be a mapping"],
    [TOOL("[{}]"), "`host.tools.t.results[0]` takes `return` or `raise`"],
    [TOOL("[{return: 1, raise: 1}]"), "`host.tools.t.results[0]` takes `return` or `raise`"],
    [TOOL("{n: [{return: 1}, {x: 1}]}"), "unknown key `host.tools.t.results.n[1].x`"],
    [TOOL("[{raise: 1}]"), "`host.tools.t.results[0].raise` must be a mapping"],
    [TOOL("[{raise: {class: A}}]"), "`host.tools.t.results[0].raise.message` is required"],
    [
      TOOL("[{raise: {class: A, message: m, x: 1}}]"),
      "unknown key `host.tools.t.results[0].raise.x`",
    ],
    [
      TOOL("[{raise: {class: A, message: m, retry_after: 1}}]"),
      "`host.tools.t.results[0].raise.retry_after` is for TransientToolError",
    ],
    [`${RUN}host: {human: 1}`, "`host.human` must be a mapping"],
    [`${RUN}host: {human: {n: a}}`, "`host.human` maps node ids to lists of answers"],
    [
      `${RUN}host: {human: {}, tools: {ask_user: {parameters: {}, results: []}}}`,
      "a case with `host.human` declares no `ask_user` tool",
    ],
    [`${RUN}host: {jitter: 0.5}`, "`host.jitter` must be a list of numbers in [0, 1)"],
    [`${RUN}host: {jitter: [1]}`, "`host.jitter` must be a list of numbers in [0, 1)"],
    [`${RUN}host: {jitter: [-0.1]}`, "`host.jitter` must be a list of numbers in [0, 1)"],
    [`${RUN}host: {jitter: [a]}`, "`host.jitter` must be a list of numbers in [0, 1)"],
    [`${RUN}host: {jitter: [true]}`, "`host.jitter` must be a list of numbers in [0, 1)"],
    [`${RUN}host: {jitter: [.nan]}`, "`host.jitter` must be a list of numbers in [0, 1)"],
    [
      `${BARE_RESUME}run: {num_workers: 0}`,
      "a `resume` case needs `host.crash_after_record`, a record number",
    ],
    [
      `${BARE_RESUME}run: {num_workers: 0}\nhost: {crash_after_record: -1}`,
      "a `resume` case needs `host.crash_after_record`, a record number",
    ],
    [
      `${BARE_RESUME}run: {num_workers: 0}\nhost: {crash_after_record: 1.0}`,
      "a `resume` case needs `host.crash_after_record`, a record number",
    ],
    [
      `${BARE_RESUME}host: {crash_after_record: 1}`,
      "`host.crash_after_record` needs `run.num_workers: 0`",
    ],
    [
      `${BARE_RESUME}run: {num_workers: 1}\nhost: {crash_after_record: 1}`,
      "`host.crash_after_record` needs `run.num_workers: 0`",
    ],
    [
      `${BARE_BACKWARD}backward: [{feedback: f}]`,
      "a `backward` case needs `run.learnable: true`",
    ],
    [
      `${BARE_BACKWARD}run: {learnable: 1}\nbackward: [{feedback: f}]`,
      "a `backward` case needs `run.learnable: true`",
    ],
    [EXPR, "an `expr` case takes one of `expr` and `template`"],
    [`${EXPR}expr: a\ntemplate: b`, "an `expr` case takes one of `expr` and `template`"],
    [`${EXPR}expr: 1`, "`expr` or `template` must be text"],
    [`${EXPR}expr: a\nmode: x`, '`mode` must be one of binding, condition, strict, not "x"'],
    [`${EXPR}template: a\nmode: condition`, "`mode: condition` takes an `expr` and no `item`"],
    [`${EXPR}expr: a\nitem: 1\nmode: condition`, "`mode: condition` takes an `expr` and no `item`"],
    [`${EXPR}template: a\nitem: 1\nmode: strict`, STRICT],
    [`${EXPR}template: a\ntypedefs: {}\nmode: strict`, STRICT],
  ])("refuses %p", (text, message) => {
    expect(() => checkCase(readCase(text))).toThrow(message);
  });

  test.each([
    LOAD,
    `${LOAD}rules: [text]\npython_expect: {load: ok}`,
    `${RUN}host: {jitter: [0, 0.5], human: {n: [a]}}`,
    TOOL("{n: [{return: 1}, {raise: {class: TransientToolError, message: m, retry_after: 1}}]}"),
    RESUME,
    `${BARE_RESUME}run: {num_workers: 0.0}\nhost: {crash_after_record: 1}`,
    `${BARE_RESUME}run: {num_workers: false}\nhost: {crash_after_record: 1}`,
    BACKWARD,
    `${EXPR}expr: a\nitem: 1`,
    `${EXPR}expr: a\nmode: condition`,
    `${EXPR}expr: a\nitem: 1\ntypedefs: {}\nmode: strict`,
    "kind: value\ntype: int\nvalue: 1\nexpect: {value: 1, kind: int}",
    "kind: defaults\nexpect: {defaults: {}}",
    "kind: cli\nargs: [run]\nstdin: x\nexpect: {exit: 0, stdout: x}",
  ])("accepts %p", (text) => {
    const read = readCase(text);
    expect(checkCase(read)).toBe(read);
  });

  test("accepts every committed case file", () => {
    const paths = readdirSync(CASES, { withFileTypes: true })
      .filter((entry) => entry.isDirectory())
      .flatMap((dir) =>
        readdirSync(join(CASES, dir.name))
          .filter((name) => name.endsWith(".yaml"))
          .map((name) => join(CASES, dir.name, name)),
      );
    expect(paths.length).toBeGreaterThan(0);
    for (const path of paths) {
      expect(() => checkCase(readCase(readFileSync(path, "utf8"))), path).not.toThrow();
    }
  });
});

describe("decode", () => {
  test("reads the codec form", () => {
    expect(decode({ $float: "inf" })).toBe(Infinity);
    expect(decode({ $float: "-inf" })).toBe(-Infinity);
    expect(decode({ a: [{ $float: "inf" }], b: 1n })).toEqual({ a: [Infinity], b: 1n });
    expect(decode({ $record: { $float: "inf" } })).toEqual({ $float: "inf" });
    expect(decode({ $record: { a: { $float: "-inf" } } })).toEqual({ a: -Infinity });
    expect(decode({ $float: "inf", b: 1n })).toEqual({ $float: "inf", b: 1n });
  });

  test("keeps `__proto__` a plain key", () => {
    const decoded = decode(readCase("__proto__: {$float: inf}"));
    expect(Object.hasOwn(decoded as object, "__proto__")).toBe(true);
    expect(Object.getPrototypeOf(decoded)).toBe(Object.prototype);
  });

  test("refuses a `$record` that isn't a mapping", () => {
    expect(() => decode({ $record: 1n })).toThrow("`$record` takes a mapping");
  });

  test.each([
    [Infinity, "inf"],
    [-Infinity, "-inf"],
    [NaN, "nan"],
  ])("refuses a bare %p", (value, spelled) => {
    expect(() => decode([value])).toThrow(
      `a bare ${spelled} is not a case value; write {$float: inf} or {$float: -inf}`,
    );
  });
});

describe("prepare", () => {
  test("decodes values except text keys", () => {
    const inf = "{$float: inf}";
    const prepared = prepare(
      readCase(
        `flow: "${inf}"\npython_flow: x\nother: y\nfiles: {a: b}\nstdin: z\n` +
          `inputs: {a: ${inf}}\nexpect: {output: ${inf}}\npython_expect: {output: ${inf}}`,
      ),
    );
    expect(prepared).toEqual({
      flow: inf,
      python_flow: "x",
      other: "y",
      files: { a: "b" },
      stdin: "z",
      inputs: { a: Infinity },
      expect: { output: Infinity },
      python_expect: { output: Infinity },
    });
  });

  test("fills secrets in their keys only", () => {
    const prepared = prepare(
      readCase(
        "run: {api_key: <key:3>}\nresume: {headers: {a: x<key:1>y}}\n" +
          "backward: [{api_key: <key:2>}]\n" +
          "env: {A_API_KEY: <key:3>}\nhost: {human: {n: [<key:4>]}}\nstdin: <key:5>\n" +
          "inputs: {a: <key:3>}\nexpect: {output: <key:3>}",
      ),
    );
    expect(prepared).toEqual({
      run: { api_key: "sk-conformance-3" },
      resume: { headers: { a: "xsk-conformance-1y" } },
      backward: [{ api_key: "sk-conformance-2" }],
      env: { A_API_KEY: "sk-conformance-3" },
      host: { human: { n: ["sk-conformance-4"] } },
      stdin: "sk-conformance-5",
      inputs: { a: "<key:3>" },
      expect: { output: "<key:3>" },
    });
  });
});

describe("same", () => {
  test("is kind-strict", () => {
    expect(same(1n, 1n)).toBe(true);
    expect(same(1n, 1)).toBe(false);
    expect(same(true, 1n)).toBe(false);
    expect(same("1", 1n)).toBe(false);
    expect(same(null, {})).toBe(false);
    expect(same({}, null)).toBe(false);
    expect(same(-0, 0)).toBe(true);
  });

  test("compares an int and a float by value with `byValue`", () => {
    const byValue = { byValue: true };
    expect(same(2n, 2, byValue)).toBe(true);
    expect(same(2, 2n, byValue)).toBe(true);
    expect(same(2n ** 60n, 2 ** 60, byValue)).toBe(true);
    expect(same(2n ** 60n + 1n, 2 ** 60, byValue)).toBe(false);
    expect(same(1n, "1", byValue)).toBe(false);
    expect(same(true, 1n, byValue)).toBe(false);
    expect(same({ a: [2n] }, { a: [2] }, byValue)).toBe(true);
  });

  test("compares mappings by keys, ignoring order, and lists in order", () => {
    expect(same({ a: 1n, b: 2n }, { b: 2n, a: 1n })).toBe(true);
    expect(same({ a: 1n }, { a: 1n, b: 2n })).toBe(false);
    expect(same({ a: 1n, b: 2n }, { a: 1n })).toBe(false);
    expect(same({ a: 1n, c: 2n }, { a: 1n, b: 2n })).toBe(false);
    expect(same([1n, 2n], [1n, 2n])).toBe(true);
    expect(same([2n, 1n], [1n, 2n])).toBe(false);
    expect(same([1n], [1n, 2n])).toBe(false);
    expect(same([1n], { 0: 1n })).toBe(false);
    expect(same({ 0: 1n }, [1n])).toBe(false);
  });

  test("counts only plain objects as mappings", () => {
    expect(same(new Map(), {})).toBe(false);
    expect(same(new Date(0), {})).toBe(false);
    expect(same(Object.create(null), {})).toBe(true);
  });

  test("leaves out `message` the expected mapping leaves out, with `optionalMessage`", () => {
    const optionalMessage = { optionalMessage: true };
    const actual = { code: "c", message: "m", frames: [{ label: "l", message: "n" }] };
    expect(same(actual, { code: "c", frames: [{ label: "l" }] }, optionalMessage)).toBe(true);
    expect(same(actual, { code: "c", frames: [{ label: "l" }] })).toBe(false);
    expect(same(actual, { code: "c", message: "x", frames: [{ label: "l" }] }, optionalMessage))
      .toBe(false);
    expect(same({ code: "c" }, { code: "c", message: "m" }, optionalMessage)).toBe(false);
  });
});

describe("compare", () => {
  test("passes when the sides are the same", () => {
    expect(() => compare("x", { a: 1n }, { a: 1n })).not.toThrow();
    expect(() => compare("x", 2n, 2, { byValue: true })).not.toThrow();
    expect(() => compare("x", { message: "m" }, {}, { optionalMessage: true })).not.toThrow();
  });

  test("fails with both sides as YAML", () => {
    const long = "word ".repeat(40).trim();
    let failure: unknown;
    try {
      compare("x", { a: 1, b: 1n, c: -0, d: Infinity, s: long }, { a: 1n });
    } catch (error) {
      failure = error;
    }
    expect(failure).toBeInstanceOf(CaseFailure);
    expect((failure as Error).name).toBe("CaseFailure");
    expect((failure as Error).message).toBe(
      `x differs\n--- actual\na: 1.0\nb: 1\nc: -0.0\nd: .inf\ns: ${long}\n--- expected\na: 1\n`,
    );
  });
});

describe("raised", () => {
  class Foo extends Error {
    override name = "Foo";
  }

  test("gives the class name and message", () => {
    expect(raised(new RangeError("m"))).toEqual({ class: "RangeError", message: "m" });
    expect(raised(new Foo("m"))).toEqual({ class: "Foo", message: "m" });
  });

  test("reads the name through the error-class table", () => {
    expect(raised(new Foo("m"), new Map([["Foo", "Bar"]]))).toEqual({ class: "Bar", message: "m" });
  });

  test("adds a TypeCheckError's path, list indexes as ints", () => {
    const error = Object.assign(new Error("bad"), { name: "TypeCheckError", path: ["a", 0, "b"] });
    expect(raised(error)).toEqual({
      class: "TypeCheckError",
      message: "bad",
      path: ["a", 0n, "b"],
    });
    const root = Object.assign(new Error("bad"), { name: "TypeCheckError", path: [] });
    expect(raised(root)).toEqual({ class: "TypeCheckError", message: "bad", path: [] });
    const other = Object.assign(new Error("bad"), { path: ["a"] });
    expect(raised(other)).toEqual({ class: "Error", message: "bad" });
  });

  test("names a thrown non-Error by its type", () => {
    expect(raised("oops")).toEqual({ class: "string", message: "oops" });
    expect(raised(42)).toEqual({ class: "number", message: "42" });
  });
});

describe("expectRaises and results", () => {
  const expected = { class: "TypeError", message: "m" };

  test("returns the raised form when it matches", async () => {
    const thrown = () => {
      throw new TypeError("m");
    };
    expect(await expectRaises(expected, thrown)).toEqual(expected);
    expect(await expectRaises(expected, () => Promise.reject(new TypeError("m")))).toEqual(
      expected,
    );
  });

  test("fails on another error", async () => {
    const call = () => {
      throw new RangeError("m");
    };
    const failure = expectRaises(expected, call);
    await expect(failure).rejects.toBeInstanceOf(CaseFailure);
    await expect(failure).rejects.toThrow("raises differs\n--- actual\nclass: RangeError\n");
  });

  test("lets the case's own failure through", async () => {
    const own = new CaseFailure("own");
    const call = () => {
      throw own;
    };
    await expect(expectRaises(expected, call)).rejects.toBe(own);
  });

  test("fails when nothing is raised", async () => {
    const failure = expectRaises({ class: "LoadError", message: "m" }, () => 1);
    await expect(failure).rejects.toBeInstanceOf(CaseFailure);
    await expect(failure).rejects.toThrow("expected LoadError to be raised, but nothing was");
  });

  test("results compares each named result", async () => {
    const expect_ = { value: 1n, kind: "int" };
    await results(expect_, () => ({ value: 1n, kind: "int" }));
    await results(expect_, async () => ({ value: 1n, kind: "int" }));
    await expect(results(expect_, () => ({ value: 1, kind: "int" }))).rejects.toThrow(
      "value differs",
    );
    const extra = results({ value: 1n }, () => ({ value: 1n, other: null }));
    await expect(extra).rejects.toBeInstanceOf(CaseFailure);
    await expect(extra).rejects.toThrow("unexpected result other");
  });

  test("results compares the raises", async () => {
    const thrown = () => {
      throw new TypeError("m");
    };
    await results({ raises: expected }, thrown);
    await expect(results({ raises: expected }, () => ({ value: 1n }))).rejects.toThrow(
      "expected TypeError to be raised, but nothing was",
    );
  });
});

describe("setUp", () => {
  const made: string[] = [];
  const leaks = ["YIAN_LEAK", "FOO_API_KEY", "FOO_BASE_URL", "FOO_ENDPOINT", "FOO_API_VERSION"];
  const saved = new Map([...leaks, "KEEP_ME"].map((name) => [name, process.env[name]]));
  afterEach(() => {
    for (const dir of made.splice(0)) rmSync(dir, { recursive: true, force: true });
    for (const [name, value] of saved) {
      if (value === undefined) delete process.env[name];
      else process.env[name] = value;
    }
  });

  /** Return a new temporary directory, removed after the test. */
  function temp(): string {
    const dir = mkdtempSync(join(tmpdir(), "conformance-test-"));
    made.push(dir);
    return dir;
  }

  test("writes the case directory", () => {
    const fixtures = temp();
    mkdirSync(join(fixtures, "f"));
    writeFileSync(join(fixtures, "f", "one.txt"), "1");
    const ctx = setUp(
      { kind: "load", flow: "id: x", files: { "lib/child.yaml": "c", "a.txt": "é" } },
      fixtures,
    );
    made.push(ctx.dir);
    expect(dirname(ctx.dir)).toBe(tmpdir().replace(/\/$/, ""));
    expect(readFileSync(join(ctx.dir, "flow.yaml"), "utf8")).toBe("id: x");
    expect(readFileSync(join(ctx.dir, "lib", "child.yaml"), "utf8")).toBe("c");
    expect(readFileSync(join(ctx.dir, "a.txt"), "utf8")).toBe("é");
    expect(readFileSync(join(ctx.dir, "fixtures", "f", "one.txt"), "utf8")).toBe("1");
    expect(existsSync(join(ctx.dir, "auth.json"))).toBe(false);
  });

  test("writes no flow or fixtures when there are none", () => {
    const ctx = setUp({ kind: "defaults" }, join(temp(), "missing"));
    made.push(ctx.dir);
    expect(readdirSync(ctx.dir)).toEqual([]);
  });

  test("gives yian only the case's variables", () => {
    for (const name of leaks) process.env[name] = "leak";
    process.env["KEEP_ME"] = "1";
    const ctx = setUp({ kind: "load", env: { YIAN_X: "1", A_API_KEY: "k" } }, temp());
    made.push(ctx.dir);
    for (const name of leaks) expect(Object.hasOwn(ctx.env, name)).toBe(false);
    expect(ctx.env["KEEP_ME"]).toBe("1");
    expect(ctx.env["YIAN_AUTH_FILE"]).toBe(join(ctx.dir, "auth.json"));
    expect(ctx.env["YIAN_X"]).toBe("1");
    expect(ctx.env["A_API_KEY"]).toBe("k");
    // The process environment is never changed.
    expect(process.env["YIAN_LEAK"]).toBe("leak");
    expect(process.env["YIAN_X"]).toBeUndefined();
    expect(process.env["YIAN_AUTH_FILE"]).not.toBe(ctx.env["YIAN_AUTH_FILE"]);
  });

  test("lets the case's `env` set YIAN_AUTH_FILE", () => {
    const ctx = setUp({ kind: "load", env: { YIAN_AUTH_FILE: "x" } }, temp());
    made.push(ctx.dir);
    expect(ctx.env["YIAN_AUTH_FILE"]).toBe("x");
  });

  test("gives a `cli` case only its `env`", () => {
    const ctx = setUp({ kind: "cli", env: { A: "1" } }, temp());
    made.push(ctx.dir);
    expect(ctx.env).toEqual({ A: "1" });
    const bare = setUp({ kind: "cli" }, temp());
    made.push(bare.dir);
    expect(bare.env).toEqual({});
  });
});

describe("main", () => {
  const FILES: Record<string, string> = {
    "a/ok.yaml": "kind: expr\nexpr: pass\nexpect: {value: 1}\n",
    "a/x.yaml": "kind: expr\nexpr: differ\nexpect: {value: 1}\n",
    "a/boom.yaml": "kind: expr\nexpr: throw\nexpect: {value: 1}\n",
    "a/rules.yaml": "kind: expr\nexpr: pass\nrules: [text]\nexpect: {value: 1}\n",
    "a/run.log.jsonl": "{}\n",
    "b/y.yaml": "kind: load\nflow: x\nexpect: {load: ok}\n",
    "b/diag.yaml": "kind: load\nflow: x\nexpect: {load: {diagnostics: [{code: compose.load}]}}\n",
    "b/bad.yaml": "kind: nope\npending: '#391'\nexpect: {}\n",
    "b/pending_fails.yaml": "kind: expr\nexpr: differ\npending: '#391'\nexpect: {value: 1}\n",
    "b/pending_passes.yaml": "kind: expr\nexpr: pass\npending: '#391'\nexpect: {value: 1}\n",
  };
  const BAD =
    "FAIL b/bad: `kind` must be one of load, run, resume, backward, expr, value, defaults, " +
    'fingerprint, cli, not "nope"';
  const X = ["FAIL a/x: x differs", "  --- actual", "  1", "  --- expected", "  2"];
  const PASSES = "FAIL b/pending_passes: passes, but is marked pending (#391); remove the marker";
  const NO_LOAD = "the TS runner doesn't run `load` cases yet";

  const roots: string[] = [];
  let ran: { id: string; ctx: CaseContext; present: boolean; fixture: boolean }[] = [];
  const expr: KindRunner = (case_: Case, ctx: CaseContext) => {
    ran.push({
      id: String(case_["expr"]),
      ctx,
      present: existsSync(ctx.dir),
      fixture: existsSync(join(ctx.dir, "fixtures", "f.txt")),
    });
    if (case_["expr"] === "differ") compare("x", 1n, 2n);
    if (case_["expr"] === "throw") throw new TypeError("boom");
  };

  afterEach(() => {
    for (const dir of roots.splice(0)) rmSync(dir, { recursive: true, force: true });
    ran = [];
  });

  /**
   * Run `main` over `files` in a `cases/` directory, with `fixtures` beside it and `kinds`
   * injected; return its exit code and output.
   */
  async function run(
    argv: string[],
    files: Record<string, string> = FILES,
    fixtures: Record<string, string> = {},
    kinds: Partial<Record<Kind, KindRunner>> = { expr },
  ) {
    const root = mkdtempSync(join(tmpdir(), "conformance-cases-"));
    roots.push(root);
    const casesDir = join(root, "cases");
    const written = [
      ...Object.entries(files).map(([name, text]) => [join(casesDir, name), text] as const),
      ...Object.entries(fixtures).map(
        ([name, text]) => [join(root, "fixtures", name), text] as const,
      ),
    ];
    for (const [path, text] of written) {
      mkdirSync(dirname(path), { recursive: true });
      writeFileSync(path, text);
    }
    const out: string[] = [];
    const err: string[] = [];
    const code = await main(argv, {
      casesDir,
      kinds,
      write: (line) => out.push(line),
      writeError: (line) => err.push(line),
    });
    return { code, out, err };
  }

  test("reports every case and a summary", async () => {
    const { code, out, err } = await run([]);
    expect(code).toBe(1);
    expect(err).toEqual([]);
    expect(out[0]).toBe("FAIL a/boom: TypeError: boom");
    const stack = out.slice(1, out.indexOf(X[0] ?? ""));
    expect(stack.length).toBeGreaterThan(0);
    // The stack's frames, indented under the FAIL line.
    expect(stack.every((line) => line.startsWith("      at "))).toBe(true);
    expect(out.slice(1 + stack.length)).toEqual([
      ...X,
      BAD,
      `FAIL b/diag: ${NO_LOAD}`,
      PASSES,
      `FAIL b/y: ${NO_LOAD}`,
      "9 cases: 2 passed, 6 failed, 1 pending",
    ]);
  });

  test("runs cases in UTF-16 code-unit order, each in a directory removed after", async () => {
    const files = {
      "g/B.yaml": "kind: expr\nexpr: B\nexpect: {value: 1}\n",
      "g/a.yaml": "kind: expr\nexpr: a\nexpect: {value: 1}\n",
      "g/ｱ.yaml": "kind: expr\nexpr: ｱ\nexpect: {value: 1}\n",
      "g/\u{1F600}.yaml": "kind: expr\nexpr: \u{1F600}\nexpect: {value: 1}\n",
    };
    const { code, out } = await run([], files);
    expect(code).toBe(0);
    expect(out).toEqual(["4 cases: 4 passed, 0 failed, 0 pending"]);
    expect(ran.map((entry) => entry.id)).toEqual(["B", "a", "\u{1F600}", "ｱ"]);
    expect(ran.every((entry) => entry.present && !existsSync(entry.ctx.dir))).toBe(true);
    expect(ran.some((entry) => entry.fixture)).toBe(false);
    expect(new Set(ran.map((entry) => entry.ctx.dir)).size).toBe(4);
  });

  test("copies the fixtures beside the cases directory", async () => {
    expect((await run(["--case", "a/ok"], FILES, { "f.txt": "1" })).code).toBe(0);
    expect(ran.map((entry) => entry.fixture)).toEqual([true]);
  });

  test("exits 0 when no case fails, a failing pending case included", async () => {
    expect(await run(["--case", "a/ok"])).toEqual({
      code: 0,
      out: ["1 case: 1 passed, 0 failed, 0 pending"],
      err: [],
    });
    expect(await run(["--case", "b/pending_fails"])).toEqual({
      code: 0,
      out: ["1 case: 0 passed, 0 failed, 1 pending"],
      err: [],
    });
  });

  test("fails a case using a reserved key before setting it up", async () => {
    const files = {
      "r/host.yaml": "kind: run\nflow: x\nhost: {tools: {}, models: {}}\nexpect: {output: {}}\n",
      "r/memory.yaml": "kind: run\nflow: x\nexpect: {output: {}, memory: {}}\n",
      "r/ok.yaml": "kind: run\nflow: x\nhost: {tools: {}}\nexpect: {output: {}}\n",
    };
    const ids: string[] = [];
    const passing: KindRunner = (case_) => {
      ids.push(String(case_["flow"]));
    };
    expect(await run([], files, {}, { run: passing })).toEqual({
      code: 1,
      out: [
        "FAIL r/host: `host.models` is not supported by the TS runner yet",
        "FAIL r/memory: `expect.memory` is not supported by the TS runner yet",
        "3 cases: 1 passed, 2 failed, 0 pending",
      ],
      err: [],
    });
    expect(ids).toEqual(["x"]);
  });

  const ALL = [
    "a/boom", "a/ok", "a/rules", "a/x",
    "b/bad", "b/diag", "b/pending_fails", "b/pending_passes", "b/y",
  ];
  const FAILING = new Set(["a/boom", "a/x", "b/bad", "b/diag", "b/pending_passes", "b/y"]);

  test.each([
    [["--group", "a"], ["a/boom", "a/ok", "a/rules", "a/x"]],
    [["--group", "a", "--group", "b"], ALL],
    [["--case", "a/ok", "--case", "b/y"], ["a/ok", "b/y"]],
    [["--kind", "load"], ["b/bad", "b/diag", "b/y"]],
    [["--rules", "text"], ["a/rules", "b/bad"]],
    [["--diagnostic-code", "compose.load"], ["b/bad", "b/diag"]],
    [["--diagnostic-code", "other"], ["b/bad"]],
    [["--group", "b", "--kind", "expr"], ["b/bad", "b/pending_fails", "b/pending_passes"]],
    [["--group", "b", "--rules", "text"], ["b/bad"]],
    [["--kind", "nope"], ["b/bad"]],
  ])("selects with %p", async (argv, ids) => {
    const { out } = await run(argv);
    expect(out.at(-1)).toStartWith(`${ids.length} ${ids.length === 1 ? "case" : "cases"}: `);
    const failed = out.filter((line) => line.startsWith("FAIL ")).map((line) => line.split(":")[0]);
    expect(failed).toEqual(ids.filter((id) => FAILING.has(id)).map((id) => `FAIL ${id}`));
  });

  test("exits 1 when no case matches", async () => {
    expect(await run(["--group", "a", "--kind", "load"])).toEqual({
      code: 1,
      out: ["no case matches"],
      err: [],
    });
    expect(await run(["--group", "c"])).toEqual({ code: 1, out: ["no case matches"], err: [] });
  });

  test.each([[["--bogus"]], [["--group"]], [["a/ok"]]])("exits 2 on %p", async (argv) => {
    const { code, out, err } = await run(argv);
    expect(code).toBe(2);
    expect(out).toEqual([]);
    expect(err).toEqual([
      "usage: conformance [--group GROUP] [--kind KIND] [--case GROUP/NAME] [--rules RULE] " +
        "[--diagnostic-code CODE]",
    ]);
  });
});
