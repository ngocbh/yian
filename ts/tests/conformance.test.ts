/**
 * The conformance runner's own unit tests: reading a case file, the format check and case
 * values. Not a port of `tests/test_conformance.py`, which `conformance.ts` is.
 */

import { describe, expect, test } from "bun:test";
import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { parseDocument } from "yaml";

import { checkCase, decode, prepare, readCase } from "./conformance.ts";

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
    [`${CLI}args: [1]`, "`args` must be a list of text"],
    [`${LOAD}rules: [nope, text, bad]`, "unknown rule ids: nope, bad"],
    [`${LOAD}env: 1`, "`env` must be a mapping"],
    [`${LOAD}env: {A: 1}`, "`env` values must be text"],
    [`${LOAD}files: {a: 1}`, "`files` values must be text"],
    ["kind: load\nflow: 1\nexpect: {load: ok}", "`flow` must be text"],
    [`${LOAD}python_flow: [a]`, "`python_flow` must be text"],
    ["kind: fingerprint\nflow: x\nother: 1\nexpect: {fingerprint: same}", "`other` must be text"],
    [`${CLI}stdin: 1`, "`stdin` must be text"],
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
