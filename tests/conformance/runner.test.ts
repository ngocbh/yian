/**
 * The conformance runner's own unit tests: reading a case file, the format check, case values,
 * comparison, the case directory, selection and the report, plus an unreached interrupt item,
 * which no case can cover (a passing case reaches every point).
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
  crashSweep,
  sourcesByRecord,
  decode,
  expectRaises,
  KINDS as RUNNERS,
  main,
  prepare,
  raised,
  readCase,
  results,
  runCli,
  same,
  masked,
  pooled,
  setUp,
  sweepable,
  trace,
} from "./runner.ts";
import type { Case, CaseContext, Kind, KindRunner } from "./runner.ts";
import { lintSchema, loadSchema, validate } from "../helpers/schema.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { current, forget, read, reload } from "../../src/defaults.ts";
import { Failure } from "../../src/diagnostics.ts";
import {
  decodeRecord,
  encodeRecord,
  InboxClosedRecord,
  InboxWithdrawnRecord,
  NodeStepRecord,
  NodeWaitingRecord,
  NodeWokenRecord,
  PoolCommitRecord,
  RecordEnvelope,
  type RunRecord,
  RunCancelledRecord,
  RunFailedRecord,
  RunLogSchemaError,
  RunWaitingRecord,
  SCHEMA_VERSION,
} from "../../src/runtime/log.ts";

const CASES = fileURLToPath(new URL("./cases/", import.meta.url));
// A crash sweep runs its case about once per record: under 1 s alone on a fast host, several
// times that on a slow or busy one, past bun's 5 s default.
const SWEEP = 30_000;

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
const EXCHANGE = "request: {method: POST, url: u}, response: {status: 200}";
/** A `cli` case whose `http` lists `exchanges`, each the keys of one exchange. */
const HTTP = (...exchanges: string[]) =>
  `${CLI}http: [${exchanges.map((exchange) => `{${exchange}}`).join(", ")}]\n`;
const RESPONSE = (response: string) =>
  HTTP(`request: {method: POST, url: u}, response: ${response}`);
const STATUS = "`http[0].response.status` must be an int from 100 to 599";
const DELAY = "`http[0].response.delay` must be a finite non-negative number";
const EXISTS = (path: string) =>
  `\`expect.exists\`: \`${path}\` must be a relative path without \`..\``;
const TOOL = (results: string) =>
  `${RUN}host: {tools: {t: {parameters: {}, results: ${results}}}}\n`;
const STRICT = "a strict `template` takes no `item` or `typedefs`";
const JSON_CASE = "kind: json\nexpect: {value: 1}\n";
const ONE_OF = "a `json` case takes one of `read` and `write`";
const READ = "`read` takes `mode` (codec or plain), and no `form`";
const WRITE = "`write` takes `form` (canonical or log), and no `mode`";
const KINDS = "load, run, resume, backward, expr, value, defaults, fingerprint, cli, json";
const TEXT_REPLY = "{format: yian.message.v1, role: assistant, parts: [{type: text, text: hi}]}";
/** A `run` case whose `host.models.n.agent` script is the one `entry`. */
const MODEL = (entry: string) => `${RUN}host: {models: {n: {agent: [${entry}]}}}\n`;
const AT = "host.models.n.agent[0]";
const SENDS =
  "`host.sends` items are {inbox, event, after?|down?}, {close}, {withdraw, after?}, {interrupt, after?} or {complete, output | error}; " +
  "`hold: true` marks an item sent while the run waits that isn't followed by an advance";
const COUNTS = "`host.sends[0].after` counts must be ints from 1";
const USAGE = "{input_tokens: 1, output_tokens: 2, total_tokens: 3}";
const CONSUMER_LIST = "agent, structured_retry, compose, backward, summary";
const MEMORY_CONSUMER = "consumers go under `__memory__`, and `__memory__` holds only them";
/** A `run` case whose `expect` adds `results`. */
const EXPECT = (results: string) => `kind: run\nflow: x\nexpect: {output: 1, ${results}}\n`;

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
    [`${LOAD}rules: [text]`, "unknown key `rules`"],
    [`${LOAD}python_expect: {load: ok}`, "unknown key `python_expect`"],
    [`${CLI}args: [1]`, "`args` must be a list of text"],
    [`${LOAD}env: 1`, "`env` must be a mapping"],
    [`${LOAD}env: {A: 1}`, "`env` values must be text"],
    [`${LOAD}files: {a: 1}`, "`files` values must be text"],
    ["kind: load\nflow: 1\nexpect: {load: ok}", "`flow` must be text"],
    ["kind: fingerprint\nflow: x\nother: 1\nexpect: {fingerprint: same}", "`other` must be text"],
    [`${CLI}stdin: 1`, "`stdin` must be text"],
    [`${CLI}signal: {name: SIGHUP, after: x}`, "`signal.name` must be SIGINT or SIGTERM"],
    [`${CLI}http: {}`, "`http` must be a list"],
    [
      HTTP(`${EXCHANGE}, repeat: true`, EXCHANGE),
      "`http[0].repeat`: only the last exchange repeats",
    ],
    [HTTP(`${EXCHANGE}, repeat: 1`), "`http[0].repeat` must be true or false"],
    [HTTP("request: {method: GET}, response: {status: 200}"), "`http[0].request.url` is required"],
    [
      HTTP("request: {method: POST, url: u, json: {}, form: {}}, response: {status: 200}"),
      "`http[0].request` takes at most one of `json` and `form`",
    ],
    [
      HTTP("request: {method: POST, url: u, form: {a: 1}}, response: {status: 200}"),
      "`http[0].request.form` values must be text",
    ],
    [RESPONSE("{status: 700}"), STATUS],
    [RESPONSE("{status: 200.0}"), STATUS],
    [RESPONSE("{status: 200, delay: -1}"), DELAY],
    [RESPONSE("{status: 200, delay: .inf}"), DELAY],
    [
      "kind: cli\nexpect: {exit: 0, exists: {a: yes}}",
      "`expect.exists` values must be true or false",
    ],
    ["kind: cli\nexpect: {exit: 0, exists: {a/../b: true}}", EXISTS("a/../b")],
    ["kind: cli\nexpect: {exit: 0, exists: {/a: true}}", EXISTS("/a")],
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
    ...["{parameters: {}}", "{parameters: {}, results: [], pending: true}", "{parameters: {}, pending: false}"].map(
      (stub) => [`${RUN}host: {tools: {t: ${stub}}}`, "`host.tools.t` takes `results` or `pending: true`"],
    ),
    [
      `${RUN}host: {tools: {t: {parameters: {}, results: [], x: 1}}}`,
      "unknown key `host.tools.t.x`",
    ],
    [
      `${RUN}host: {tools: {t: {parameters: {}, results: [], concurrency_safe: 1}}}`,
      "`host.tools.t.concurrency_safe` must be a boolean",
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
    ...["null", "'1'", "-1", "-0.5", "{$float: inf}"].map((seconds) => [
      TOOL(`[{raise: {class: TransientToolError, message: m, retry_after: ${seconds}}}]`),
      "`host.tools.t.results[0].raise.retry_after` must be a finite non-negative number",
    ]),
    [`${RUN}host: {human: 1}`, "`host.human` must be a mapping"],
    [`${RUN}host: {human: {n: a}}`, "`host.human` maps node ids to lists of answers"],
    [
      `${RUN}host: {human: {}, tools: {ask_user: {parameters: {}, results: []}}}`,
      "a case with `host.human` declares no `ask` or `ask_user` tool",
    ],
    [
      `${RUN}host: {human: {}, tools: {ask: {parameters: {}, results: []}}}`,
      "a case with `host.human` declares no `ask` or `ask_user` tool",
    ],
    [`${RUN}host: {sends: 1}`, "`host.sends` must be a list"],
    ...[
      "[1]",
      "[{inbox: m}]",
      "[{close: m, inbox: m}]",
      "[{inbox: 1, event: a}]",
      "[{close: [m]}]",
      "[{complete: r/n}]",
      "[{complete: 1, output: a}]",
      "[{complete: r/n, error: 1}]",
      "[{complete: r/n, output: a, error: e}]",
      "[{inbox: m, event: a, down: true}]",
      "[{inbox: m, event: a, after: 1}]",
      "[{inbox: m, event: a, after: {node: chat, tool_call: 1}, down: true}]",
      "[{interrupt: 1}]",
      "[{interrupt: m, after: 1}]",
      "[{interrupt: m, inbox: m}]",
      "[{interrupt: m, down: true}]",
      "[{withdraw: 1}]",
      "[{withdraw: m, down: true}]",
      "[{withdraw: m, after: 1}]",
      "[{withdraw: m, inbox: m}]",
      "[{inbox: m, event: a, hold: false}]",
      "[{inbox: m, event: a, after: {node: chat, model_call: 1}, hold: true}]",
      "[{interrupt: m, after: {node: chat, model_call: 1}, hold: true}]",
      "[{inbox: m, event: a, down: true, hold: true}]",
    ].map((items) => [`${RUN}host: {sends: ${items}}`, SENDS]),
    [`${RUN}host: {sends: [{withdraw: m, after: {tool_call: 1}}]}`, "`host.sends[0].after.node` must be a node id"],
    [`${RUN}host: {sends: [{inbox: m, event: a, after: {tool_call: 1}}]}`, "`host.sends[0].after.node` must be a node id"],
    [`${RUN}host: {sends: [{interrupt: m, after: {tool_call: 1}}]}`, "`host.sends[0].after.node` must be a node id"],
    [`${RUN}host: {sends: [{interrupt: m, after: {node: chat, tool_call: 0}}]}`, COUNTS],
    [
      `${RUN}host: {sends: [{inbox: m, event: a, after: {node: chat, chunk: 1}}]}`,
      "`host.sends[0].after` is `{model_call, chunk}`, `{model_call}` or `{tool_call}`",
    ],
    [`${LOAD}host: {sends: []}`, "`host` is not a key of a `load` case"],
    [`${RUN}host: {models: 1}`, "`host.models` must be a mapping"],
    [`${RUN}host: {models: {n: 1}}`, "`host.models.n` must be a mapping"],
    [
      `${RUN}host: {models: {n: {x: []}}}`,
      `\`host.models.n.x\`: the consumer must be one of ${CONSUMER_LIST}`,
    ],
    [`${RUN}host: {models: {n: {"memory:a": []}}}`, MEMORY_CONSUMER],
    [`${RUN}host: {memory: 1}`, "`host.memory` must be a mapping"],
    [`${RUN}host: {memory: {after: []}}`, "unknown key `host.memory.after`"],
    [`${RUN}host: {memory: {before: {}}}`, "`host.memory.before` must be a list"],
    [`${RUN}host: {memory: {before: [{run: {}}]}}`, "`host.memory.before[0].run.run_id` is required"],
    [`${RUN}host: {memory: {before: [{}]}}`, "`host.memory.before[0].run` is required"],
    [
      `${RUN}host: {memory: {before: [{run: {run_id: r}, x: 1}]}}`,
      "unknown key `host.memory.before[0].x`",
    ],
    [
      `${RUN}host: {memory: {before: [{run: {run_id: r}, inputs: 1}]}}`,
      "`host.memory.before[0].inputs` must be a mapping",
    ],
    [
      `${RUN}host: {memory: {before: [{run: {run_id: r, context: x}}]}}`,
      "`host.memory.before[0].run.context` must be `present`",
    ],
    [
      `${RUN}host: {memory: {before: [{run: {run_id: r, api_key: k}}]}}`,
      "unknown key `host.memory.before[0].run.api_key`",
    ],
    [
      `${RUN}host: {memory: {before: [{run: {run_id: r, learnable: true}, backward: []}]}}`,
      "`host.memory.before[0].backward` must be a non-empty list",
    ],
    [
      `${RUN}host: {memory: {before: [{run: {run_id: r}, backward: [{feedback: f}]}]}}`,
      "`host.memory.before[0].backward` needs `host.memory.before[0].run.learnable: true`",
    ],
    [EXPECT("memory: {a: x}"), "`expect.memory` needs `host.memory`"],
    [`${EXPECT("memory: {a: 1}")}host: {memory: {}}`, "`expect.memory` values must be text"],
    [`${RUN}host: {models: {__memory__: {agent: []}}}`, MEMORY_CONSUMER],
    [`${RUN}host: {models: {n: {agent: {}}}}`, "`host.models.n.agent` must be a list"],
    [MODEL("1"), `\`${AT}\` must be a mapping`],
    [MODEL("{}"), `\`${AT}\` takes \`reply\` or \`chunks\``],
    [MODEL(`{reply: ${TEXT_REPLY}, chunks: []}`), `\`${AT}\` takes \`reply\` or \`chunks\``],
    [MODEL(`{reply: ${TEXT_REPLY}, x: 1}`), `unknown key \`${AT}.x\``],
    [
      MODEL(`{reply: ${TEXT_REPLY}, raise: {class: A, message: m}}`),
      `\`${AT}.raise\` must come with \`chunks\`, not a \`reply\``,
    ],
    [
      MODEL("{reply: {role: assistant, parts: []}}"),
      `\`${AT}.reply\` is not a yian.message.v1 message`,
    ],
    [
      MODEL("{reply: {format: yian.message.v1, role: user, parts: []}}"),
      `\`${AT}.reply.role\` must be \`assistant\``,
    ],
    [
      MODEL("{reply: {format: yian.message.v1, role: assistant, " +
        "parts: [{type: reasoning, text: r}]}}"),
      `\`${AT}.reply.parts[0]\`: this runner doesn't script \`reasoning\` parts yet`,
    ],
    [
      MODEL(`{reply: ${TEXT_REPLY}, usage: {input_tokens: 1}}`),
      `\`${AT}.usage.output_tokens\` is required`,
    ],
    [
      MODEL(`{reply: ${TEXT_REPLY}, usage: {input_tokens: -1, output_tokens: 0, total_tokens: 0}}`),
      `\`${AT}.usage\` counts must be non-negative ints`,
    ],
    [
      MODEL(`{reply: ${TEXT_REPLY}, usage: {input_tokens: 1.0, output_tokens: 0, ` +
        "total_tokens: 0}}"),
      `\`${AT}.usage\` counts must be non-negative ints`,
    ],
    [
      MODEL(`{chunks: [{finish: stop}], usage: ${USAGE}}`),
      `\`${AT}.usage\` must come with a \`reply\`; \`chunks\` carry a usage chunk`,
    ],
    [MODEL("{chunks: 1}"), `\`${AT}.chunks\` must be a list`],
    [
      MODEL("{chunks: []}"),
      `\`${AT}.chunks\` must end in a \`finish\` chunk, unless a \`raise\` follows them`,
    ],
    [MODEL("{chunks: [{text: a}]}"), `\`${AT}.chunks\` must end in a \`finish\` chunk`],
    [
      MODEL("{chunks: [{finish: stop}, {text: a}]}"),
      `\`${AT}.chunks[0]\`: \`finish\` must be the last chunk`,
    ],
    [
      MODEL("{chunks: [{text: a, finish: stop}]}"),
      `\`${AT}.chunks[0]\` takes one of \`text\`, \`tool_call\`, \`usage\` and \`finish\``,
    ],
    [MODEL("{chunks: [{x: a}]}"), `unknown key \`${AT}.chunks[0].x\``],
    [MODEL("{chunks: [{text: 1}, {finish: stop}]}"), `\`${AT}.chunks[0].text\` must be text`],
    [
      MODEL("{chunks: [{tool_call: {index: 0}}, {finish: stop}]}"),
      `\`${AT}.chunks[0].tool_call.arguments\` is required`,
    ],
    [
      MODEL("{chunks: [{tool_call: {index: a, arguments: x}}, {finish: stop}]}"),
      `\`${AT}.chunks[0].tool_call.index\` must be an int`,
    ],
    [
      MODEL("{chunks: [{tool_call: {index: 0, id: 1, arguments: x}}, {finish: stop}]}"),
      `\`${AT}.chunks[0].tool_call\` \`id\`, \`name\` and \`arguments\` must be text`,
    ],
    [
      MODEL("{chunks: [{usage: {input_tokens: 1}}, {finish: stop}]}"),
      `\`${AT}.chunks[0].usage.output_tokens\` is required`,
    ],
    [MODEL("{chunks: [{finish: 1}]}"), `\`${AT}.chunks[0].finish\` must be text or null`],
    [MODEL("{chunks: [], raise: {class: A}}"), `\`${AT}.raise.message\` is required`],
    [MODEL("{chunks: [], raise: {class: A, message: m, x: 1}}"), `unknown key \`${AT}.raise.x\``],
    [
      MODEL("{chunks: [], raise: {class: 1, message: m}}"),
      `\`${AT}.raise.class\` and \`.message\` must be text`,
    ],
    [
      MODEL("{chunks: [], raise: {class: A, message: m, status: 5.0}}"),
      `\`${AT}.raise.status\` must be an int`,
    ],
    [MODEL(`{reply: ${TEXT_REPLY}, request: {}}`), `\`${AT}.request.messages\` is required`],
    [
      MODEL(`{reply: ${TEXT_REPLY}, request: {messages: 1}}`),
      `\`${AT}.request.messages\` must be a list`,
    ],
    [
      MODEL(`{reply: ${TEXT_REPLY}, request: {messages: [], tools: 1}}`),
      `\`${AT}.request.tools\` must be a list`,
    ],
    [
      MODEL(`{reply: ${TEXT_REPLY}, request: {messages: [], tools: []}}`),
      `\`${AT}.request.tools\` is empty; omit \`tools\` when none are bound`,
    ],
    [
      MODEL(`{reply: ${TEXT_REPLY}, request: {messages: [1]}}`),
      `\`${AT}.request.messages[0]\` is not a yian.message.v1 message`,
    ],
    [
      MODEL(`{reply: ${TEXT_REPLY}, request: {messages: [], tools: [{name: t, parameters: {}}]}}`),
      `\`${AT}.request.tools[0].description\` is required`,
    ],
    [
      MODEL(`{reply: ${TEXT_REPLY}, request: {messages: [], schema: 1}}`),
      `\`${AT}.request.schema\` must be a mapping`,
    ],
    [`${RUN}host: {steering: {}}`, "unknown key `host.steering`"],
    [`${RUN}host: {sends: [{inbox: m, event: a, after: {node: chat, model_call: true}}]}`, COUNTS],
    [`${RUN}host: {sends: [{inbox: m, event: a, after: {node: chat, model_call: 1, chunk: 1.0}}]}`, COUNTS],
    [`${RUN}host: {jitter: 0.5}`, "`host.jitter` must be a list of numbers in [0, 1)"],
    [`${RUN}host: {jitter: [1]}`, "`host.jitter` must be a list of numbers in [0, 1)"],
    [`${RUN}host: {jitter: [-0.1]}`, "`host.jitter` must be a list of numbers in [0, 1)"],
    [`${RUN}host: {jitter: [a]}`, "`host.jitter` must be a list of numbers in [0, 1)"],
    [`${RUN}host: {jitter: [true]}`, "`host.jitter` must be a list of numbers in [0, 1)"],
    [`${RUN}host: {jitter: [.nan]}`, "`host.jitter` must be a list of numbers in [0, 1)"],
    [
      `${BARE_RESUME}run: {num_workers: 0}`,
      "a `resume` case needs `host.crash_after_record`, a record number or `end`",
    ],
    [
      `${BARE_RESUME}run: {num_workers: 0}\nhost: {crash_after_record: -1}`,
      "a `resume` case needs `host.crash_after_record`, a record number or `end`",
    ],
    [
      `${BARE_RESUME}run: {num_workers: 0}\nhost: {crash_after_record: 1.0}`,
      "a `resume` case needs `host.crash_after_record`, a record number or `end`",
    ],
    [
      `${BARE_RESUME}run: {num_workers: 0}\nhost: {crash_after_record: later}`,
      "a `resume` case needs `host.crash_after_record`, a record number or `end`",
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
      `${BARE_RESUME}run: {num_workers: 1}\nhost: {stop: {a: {after: {tool_call: 1}}}}`,
      "`host.stop` needs `run.num_workers: 0`",
    ],
    [
      `${BARE_RESUME}run: {num_workers: 0}\nhost: {stop: {a: {after: {tool_call: 1}}}, crash_after_record: 1}`,
      "a `resume` case takes one of `host.crash_after_record` and `host.stop`",
    ],
    [`${RUN}host: {stop: []}`, "`host.stop` must be a mapping"],
    [`${RUN}host: {stop: {a: {}}}`, "`host.stop.a.after` is required"],
    [`${RUN}host: {stop: {a: {after: {tool_call: 1}, steer: [x]}}}`, "unknown key `host.stop.a.steer`"],
    [
      `${RUN}host: {stop: {a: {after: {chunk: 1}}}}`,
      "`host.stop.a.after` is `{model_call, chunk}`, `{model_call}` or `{tool_call}`",
    ],
    [`${RUN}host: {stop: {a: {after: {model_call: 0}}}}`, "`host.stop.a.after` counts must be ints from 1"],
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
    [JSON_CASE, ONE_OF],
    [`${JSON_CASE}read: "1"\nmode: codec\nwrite: 1\nform: log`, ONE_OF],
    [`${JSON_CASE}read: "1"`, READ],
    [`${JSON_CASE}read: "1"\nmode: log`, READ],
    [`${JSON_CASE}read: "1"\nmode: codec\nform: log`, READ],
    [`${JSON_CASE}write: 1`, WRITE],
    [`${JSON_CASE}write: 1\nform: log\nmode: codec`, WRITE],
    [`${JSON_CASE}read: 1\nmode: codec`, "`read` must be text"],
    [
      'kind: json\nread: "1"\nmode: codec\nexpect: {value: 1, text: "1"}',
      "`expect` needs `value`, or `text`, or `raises`",
    ],
  ])("refuses %p", (text, message) => {
    expect(() => checkCase(readCase(text))).toThrow(message);
  });

  test.each([
    LOAD,
    `${RUN}host: {jitter: [0, 0.5], human: {n: [a]}}`,
    MODEL(`{reply: ${TEXT_REPLY}, usage: {input_tokens: 1, output_tokens: 2, total_tokens: 3, ` +
      "reasoning_tokens: 0}}"),
    MODEL(`{reply: ${TEXT_REPLY}, request: {messages: [${TEXT_REPLY}], ` +
      "tools: [{name: t, description: d, parameters: {}}], schema: {}}}"),
    MODEL('{chunks: [{text: a}, {tool_call: {index: 0, id: c, name: t, arguments: "{}"}}, ' +
      `{usage: ${USAGE}}, {finish: null}]}`),
    MODEL("{chunks: [], raise: {class: ConnectionError, message: m}}"),
    MODEL("{chunks: [{text: a}], raise: {class: X, message: m, status: 503}}"),
    `${RUN}host: {memory: {}}`,
    `${EXPECT("memory: {a: x}")}host: {memory: {before: [{inputs: {q: a}, ` +
      "run: {run_id: r, num_workers: 0, learnable: true, context: present}, " +
      "backward: [{feedback: f, api_key: k, headers: {}}]}]}}",
    `${RUN}host: {models: {__memory__: {"memory:a/b": []}, n: {compose: [], ` +
      "structured_retry: []}}}",
    `${RUN}host: {sends: [{interrupt: m, after: {node: chat, model_call: 1, chunk: 2}}, {interrupt: m}]}`,
    `${RUN}host: {sends: [{inbox: m, event: a, after: {node: chat, model_call: 1, chunk: 2}}, {close: m}]}`,
    `${BARE_RESUME}run: {num_workers: 0}\nhost: {crash_after_record: 1, sends: [{inbox: m, event: a, down: true}]}`,
    EXPECT("events: [[ModelCallStarted, {node_id: a, n: 1}]], stream: [{node_id: a, n: 1, index: 0}]"),
    TOOL("{n: [{return: 1}, {raise: {class: TransientToolError, message: m, retry_after: 1}}]}"),
    TOOL("[{raise: {class: TransientToolError, message: m, retry_after: 0.5}}]"),
    RESUME,
    `${BARE_RESUME}run: {num_workers: 0.0}\nhost: {crash_after_record: 1}`,
    `${BARE_RESUME}run: {num_workers: false}\nhost: {crash_after_record: 1}`,
    `${BARE_RESUME}run: {num_workers: 0}\nhost: {crash_after_record: end}`,
    `${BARE_RESUME}run: {num_workers: 0}\nhost: {stop: {a: {after: {model_call: 1, chunk: 2}}}}`,
    `${RUN}host: {stop: {a: {after: {tool_call: 1}}}}`,
    BACKWARD,
    `${EXPR}expr: a\nitem: 1`,
    `${EXPR}expr: a\nmode: condition`,
    `${EXPR}expr: a\nitem: 1\ntypedefs: {}\nmode: strict`,
    "kind: value\ntype: int\nvalue: 1\nexpect: {value: 1, kind: int}",
    "kind: defaults\nexpect: {defaults: {}}",
    "kind: cli\nargs: [run]\nstdin: x\nexpect: {exit: 0, stdout: x}",
    [
      "kind: cli",
      "signal: {name: SIGTERM, after: x}",
      "expect: {exit: 130, exists: {home/a.json: true, b: false}}",
      `http: [{${EXCHANGE}}, {request: {method: GET, url: v, json: {a: 1},`,
      '  headers: {authorization: "Bearer <key:1>"}},',
      "  response: {status: 200, json: {data: [{id: a}]}, delay: 0.5}, repeat: true}]",
    ].join("\n"),
    `${JSON_CASE}read: "1"\nmode: codec`,
    `${JSON_CASE}read: "1"\nmode: plain`,
    'kind: json\nwrite: 1\nform: canonical\nexpect: {text: "1"}',
    'kind: json\nwrite: 1\nform: log\nexpect: {text: "1"}',
  ])("accepts %p", (text) => {
    const read = readCase(text);
    expect(checkCase(read)).toBe(read);
  });

  // Reads every case file (about 1,900), which can pass the 5 s default on a loaded machine.
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
  }, 30_000);
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
        `flow: "${inf}"\nother: y\nfiles: {a: b}\n` +
          `stdin: z\n` +
          `inputs: {a: ${inf}}\nexpect: {output: ${inf}}`,
      ),
    );
    expect(prepared).toEqual({
      flow: inf,
      other: "y",
      files: { a: "b" },
      stdin: "z",
      inputs: { a: Infinity },
      expect: { output: Infinity },
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
    expect(raised(new TypeError("m"))).toEqual({ class: "TypeError", message: "m" });
    expect(raised(new Foo("m"))).toEqual({ class: "Foo", message: "m" });
  });

  test("names a RangeError and a RunLogSchemaError by their own class", () => {
    expect(raised(new RangeError("m"))).toEqual({ class: "RangeError", message: "m" });
    expect(raised(new RunLogSchemaError("m", 9))).toEqual({ class: "RunLogSchemaError", message: "m" });
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
  afterEach(() => {
    for (const dir of made.splice(0)) rmSync(dir, { recursive: true, force: true });
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
    const outer: Record<string, string | undefined> = { KEEP_ME: "1", UNSET: undefined };
    for (const name of leaks) outer[name] = "leak";
    const before = { ...outer };
    const ctx = setUp({ kind: "load", env: { YIAN_X: "1", A_API_KEY: "k" } }, temp(), outer);
    made.push(ctx.dir);
    for (const name of [...leaks, "UNSET"]) expect(Object.hasOwn(ctx.env, name)).toBe(false);
    expect(ctx.env["KEEP_ME"]).toBe("1");
    expect(ctx.env["YIAN_AUTH_FILE"]).toBe(join(ctx.dir, "auth.json"));
    expect(ctx.env["YIAN_X"]).toBe("1");
    expect(ctx.env["A_API_KEY"]).toBe("k");
    // The outer environment is never changed.
    expect(outer).toEqual(before);
  });

  test("inherits the process environment by default", () => {
    const ctx = setUp({ kind: "load" }, temp());
    made.push(ctx.dir);
    expect(ctx.env["PATH"]).toBe(process.env["PATH"]);
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

describe("expr and value", () => {
  const made: string[] = [];
  afterEach(() => {
    for (const dir of made.splice(0)) rmSync(dir, { recursive: true, force: true });
    forget();
  });

  test("a case's env applies to its own reads and leaves the defaults a caller installed", async () => {
    const outer = (name: string) => (name === "YIAN_MAX_VALUE_NESTING" ? "5" : undefined);
    reload(outer);
    const message =
      "expression nests deeper than 2 levels, the limit for a yian expression (each call, list, unary operator, " +
      "comparison, and chain of operators is a level); split it into smaller bindings or steps";
    const cases = [
      // Raises only under the case's limit of 2, not the caller's 5.
      'kind: expr\nexpr: "[[[x]] for x in xs]"\nvalues: {xs: [1, 2]}\nenv: {YIAN_MAX_VALUE_NESTING: "2"}\n' +
        `expect: {raises: {class: ExpressionError, message: "${message}"}}\n`,
      "kind: value\nenv: {YIAN_MAX_VALUE_NESTING: '3'}\ntype: Any\nvalue: [[1]]\n" +
        "expect: {value: [[1]], kind: \"list[Any]\"}\n",
    ];
    for (const text of cases) {
      const case_ = prepare(checkCase(readCase(text)));
      const ctx = setUp(case_, join(tmpdir(), "none"));
      made.push(ctx.dir);
      await RUNNERS[case_["kind"] as Kind]!(case_, ctx);
      expect(current()).toEqual(read(outer));
    }
  });
});

describe("run", () => {
  const made: string[] = [];
  afterEach(() => {
    for (const dir of made.splice(0)) rmSync(dir, { recursive: true, force: true });
  });

  /** Run the `run` case file `text` with the runner's own `run` kind. */
  async function runCase(text: string): Promise<void> {
    const case_ = prepare(checkCase(readCase(text)));
    const ctx = setUp(case_, join(tmpdir(), "none"));
    made.push(ctx.dir);
    await RUNNERS.run!(case_, ctx);
  }

  /**
   * A `run` case of one AGENT `chat` reading steer inbox `notes` and replying `done`, expecting the
   * output `output`; a test adds `host.sends`.
   */
  const STEER_CASE = (output: string) =>
    "kind: run\nflow: |\n  id: steer\n  name: steer\n" +
    "  llm_config: {provider: openai, model: gpt-5.5}\n  input:\n    notes: !Inbox(str)\n" +
    "  nodes:\n    chat:\n      kind: agent\n      steer_inbox: ${input.notes}\n      tools: []\n" +
    "      system_prompt: Help.\n      prompt: Go.\n      output: Optional(str)\n" +
    "  output: ${chat.output}\n" +
    "run: {num_workers: 0}\n" +
    "host:\n  tools: {}\n" +
    "  models: {chat: {agent: [{reply: {format: yian.message.v1, role: assistant, " +
    "parts: [{type: text, text: done}]}}]}}\n" +
    `expect: {output: {status: succeeded, output: ${output}}}\n`;

  /**
   * A `run` case of one TOOL `ask` node `n` passing `prompt: Ship?` and the extra `inputs`,
   * answered `yes` by `host.human`, expecting `tool_calls`.
   */
  const ASK_CASE = (inputs: string, toolCalls: string) =>
    "kind: run\nflow: |\n  id: f\n  name: f\n  input:\n    plan: !str\n  nodes:\n    n:\n" +
    "      kind: tool\n      tool_id: ask\n      input:\n        prompt: Ship?\n" +
    inputs +
    "      output: str\n  output: ${n.output}\n" +
    "inputs: {plan: p1}\nrun: {run_id: r1, num_workers: 0}\nhost: {human: {n: [yes]}}\n" +
    `expect:\n  output: {status: succeeded, output: "yes"}\n  tool_calls: ${toolCalls}\n`;

  /** The answer schema a TOOL `ask` typed `str` sends. */
  const SCHEMA = "[{name: answer, type: str, required: true}]";
  const CALL = "call_id: r1/n, attempt: 1, node: n";

  test("`host.human` answers a TOOL `ask` through the ask slot, recorded under `ask`", async () => {
    const calls =
      `{ask: [{args: {question: Ship?, answer_schema: ${SCHEMA}, node_id: n}, ${CALL}}]}`;
    await runCase(ASK_CASE("", calls));
  });

  test("`host.human` declares the extra inputs a TOOL `ask` passes", async () => {
    const calls =
      `{ask: [{args: {question: Ship?, answer_schema: ${SCHEMA}, node_id: n, ctx: p1}, ${CALL}}]}`;
    await runCase(ASK_CASE("        ctx: |-\n          ${input.plan}\n", calls));
  });

  /** A `run` case whose flow takes every message sent to inbox `m` until it closes. */
  const SENDS_CASE = (sends: string, output: string) =>
    "kind: run\nflow: |\n  id: sends\n  name: sends\n  input:\n    m: !Inbox(str)\n" +
    "  defs:\n    turn:\n      input:\n        got: !list(str)\n        closed: !bool\n" +
    "        m: !Inbox(str)\n      nodes:\n        heard:\n          kind: tool\n" +
    "          tool_id: take\n          input:\n            from: ${input.m}\n" +
    "            take: all\n      output:\n        got: ${input.got + heard.output.events}\n" +
    "        closed: ${heard.output.closed}\n  nodes:\n    chat:\n      kind: loop\n" +
    "      call: turn\n      input:\n        got: []\n        closed: false\n" +
    "        m: ${input.m}\n      while: not closed\n  output: ${chat.output.got}\n" +
    `run: {num_workers: 0}\nhost: {sends: ${sends}}\n` +
    `expect: {output: {status: succeeded, output: ${output}}}\n`;

  test("`host.sends` delivers each item while the run waits", async () => {
    await runCase(SENDS_CASE("[{inbox: m, event: a}, {inbox: m, event: b}, {close: m}]", "[a, b]"));
  });

  test("a `{withdraw}` item takes unread sends back; `hold: true` delivers the next item to the same wait", async () => {
    await runCase(SENDS_CASE("[{inbox: m, event: a, hold: true}, {withdraw: m, hold: true}, {inbox: m, event: b}, {close: m}]", "[b]"));
  });

  /**
   * A `run` case whose agent calls the pending host tool `hold`, then answers with the result it
   * reads, given `host.sends`.
   */
  const PENDING_CASE = (sends: string, output: string) =>
    "kind: run\nflow: |\n  id: f\n  name: f\n  llm_config: {provider: anthropic, model: claude-test}\n" +
    "  nodes:\n    agent:\n      kind: agent\n      tools: [hold]\n      prompt: hi\n      output: str\n" +
    "  output: ${agent.output}\nrun: {run_id: r1, num_workers: 0}\n" +
    "host:\n  tools:\n    hold: {parameters: {type: object, properties: {}}, pending: true}\n" +
    "  models:\n    agent:\n      agent:\n" +
    "      - reply: {format: yian.message.v1, role: assistant, parts: [{type: tool_call, id: c1, name: hold, arguments: {}}]}\n" +
    "      - reply: {format: yian.message.v1, role: assistant, parts: [{type: text, text: done}]}\n" +
    `  sends: ${sends}\nexpect: {output: {status: succeeded, output: ${output}}}\n`;

  test("a `pending: true` host tool waits; a `{complete}` item answers it, with an output or an error", async () => {
    await runCase(PENDING_CASE("[{complete: r1/agent/1/0, output: ok}]", "done"));
    await runCase(PENDING_CASE("[{complete: r1/agent/1/0, error: no}]", "done"));
    const waiting = runCase(PENDING_CASE("[]", "done"));
    await expect(waiting).rejects.toThrow("the run is still waiting after host.sends ran out");
  });

  test("a run still waiting after `host.sends` ran out fails the case", async () => {
    const waiting = runCase(SENDS_CASE("[{inbox: m, event: a}]", "[a]"));
    await expect(waiting).rejects.toBeInstanceOf(CaseFailure);
    await expect(waiting).rejects.toThrow("the run is still waiting after host.sends ran out");
  });

  // test_steering_unreached_action_fails
  test("an interrupt item whose point the run never reaches fails the case after the comparisons", async () => {
    const text = (output: string) =>
      STEER_CASE(output).replace("host:\n", "host:\n  sends: [{interrupt: notes, after: {node: chat, tool_call: 1}}]\n");
    // A divergence shows first.
    await expect(runCase(text("other"))).rejects.toThrow("output differs");
    const unreached = runCase(text("done"));
    await expect(unreached).rejects.toBeInstanceOf(CaseFailure);
    await expect(unreached).rejects.toThrow("`host.sends[0]` was never reached");
  });

  test("an interrupt item with `after` is delivered live when the run reaches it", async () => {
    // At the end of the only reply, which arrived whole: the answer stands, the record is logged.
    const text = STEER_CASE("done").replace("host:\n", "host:\n  sends: [{interrupt: notes, after: {node: chat, model_call: 1}}]\n");
    await runCase(text);
    const kinds = readFileSync(join(made.at(-1)!, "run.jsonl"), "utf8")
      .split("\n")
      .filter((line) => line !== "")
      .map((line) => decodeRecord(line).kind);
    expect(kinds).toContain("interrupted");
  });

  test("an interrupt item without `after` is delivered while the run waits", async () => {
    const text = PENDING_CASE("[{interrupt: notes}]", "null")
      .replace("  nodes:\n", "  input:\n    notes: !Inbox(str)\n  nodes:\n")
      .replace("      output: str\n", "      steer_inbox: ${input.notes}\n      output: Optional(str)\n")
      // The woken agent honors the interrupt instead of calling the model again.
      .replace(/ {6}- reply: \{[^\n]*text: done\}\]\}\n/, "");
    await runCase(text);
  });

  test("a `host.sends` item whose point the run never reaches fails the case", async () => {
    const text = STEER_CASE("done").replace(
      "host:\n",
      "host:\n  sends: [{inbox: notes, event: late, after: {node: chat, tool_call: 1}}]\n",
    );
    const unreached = runCase(text);
    await expect(unreached).rejects.toBeInstanceOf(CaseFailure);
    await expect(unreached).rejects.toThrow("`host.sends[0]` was never reached");
  });

  test("a `host.sends` item with `after` is sent live when the run reaches it", async () => {
    // Sent at the first chunk of the first reply, so the agent reads it late, after that reply,
    // and calls the model again.
    const text = STEER_CASE("done")
      .replace("host:\n", "host:\n  sends: [{inbox: notes, event: late, after: {node: chat, model_call: 1, chunk: 1}}]\n")
      .replace(
        "{agent: [{reply:",
        "{agent: [{reply: {format: yian.message.v1, role: assistant, parts: [{type: text, text: draft}]}}, {reply:",
      );
    await runCase(text);
    const kinds = readFileSync(join(made.at(-1)!, "run.jsonl"), "utf8")
      .split("\n")
      .filter((line) => line !== "")
      .map((line) => decodeRecord(line))
      .map(({ kind, record }) =>
        record instanceof PoolCommitRecord ? record.key : record instanceof NodeStepRecord ? String(record.step["kind"]) : kind,
      );
    const sent = kinds.indexOf("__start__.notes");
    expect(sent).toBeGreaterThan(-1);
    expect(kinds.slice(sent + 1, sent + 4)).toEqual(["model_call", "steer", "model_call"]);
  });

  /**
   * A `run` case whose AGENT `answer` replies `a1` to `q`, declaring `agentMemory` when given, with
   * `host.memory` set to `memory` and `expect.memory` to `expected`.
   */
  const MEMORY_CASE = (agentMemory: string, memory: string, expected: string) =>
    "kind: run\nflow: |\n  id: tutor\n  name: tutor\n" +
    "  llm_config: {provider: openai, model: gpt-5.5}\n" +
    "  input:\n    q: !str\n  nodes:\n    answer:\n      kind: agent\n" +
    (agentMemory ? `      long_term_memory: ${agentMemory}\n` : "") +
    '      input: {q: "${input.q}"}\n      system_prompt: Tutor.\n      prompt: "${q}"\n' +
    "      output: str\n  output: ${answer.output}\n" +
    "inputs: {q: first}\nrun: {num_workers: 0}\n" +
    `host:\n  memory: ${memory}\n` +
    "  models: {answer: {agent: [{reply: {format: yian.message.v1, role: assistant, " +
    "parts: [{type: text, text: a1}]}}]}}\n" +
    `expect: {output: {status: succeeded, output: a1}, memory: ${expected}}\n`;

  test("a key the flow doesn't declare in `expect.memory` fails the case", async () => {
    const read = runCase(MEMORY_CASE("", "{}", "{nope: ''}"));
    await expect(read).rejects.toBeInstanceOf(CaseFailure);
    await expect(read).rejects.toThrow("`expect.memory.nope` names no memory the flow declares");
  });

  test("a `before` entry whose backward call raises fails the case", async () => {
    const before = "{before: [{run: {run_id: r1, learnable: true}, backward: [{feedback: [1]}]}]}";
    const failed = runCase(MEMORY_CASE("", before, "{}"));
    await expect(failed).rejects.toBeInstanceOf(CaseFailure);
    await expect(failed).rejects.toThrow("`host.memory.before[0]` backward raised {\"class\":");
  });
});

describe("main", () => {
  const FILES: Record<string, string> = {
    "a/ok.yaml": "kind: expr\nexpr: pass\nexpect: {value: 1}\n",
    "a/x.yaml": "kind: expr\nexpr: differ\nexpect: {value: 1}\n",
    "a/boom.yaml": "kind: expr\nexpr: throw\nexpect: {value: 1}\n",
    "a/run.log.jsonl": "{}\n",
    "b/y.yaml": "kind: load\nflow: x\nexpect: {load: ok}\n",
    "b/diag.yaml": "kind: load\nflow: x\nexpect: {load: {diagnostics: [{code: compose.load}]}}\n",
    "b/bad.yaml": "kind: nope\npending: '#391'\nexpect: {}\n",
    "b/pending_fails.yaml": "kind: expr\nexpr: differ\npending: '#391'\nexpect: {value: 1}\n",
    "b/pending_passes.yaml": "kind: expr\nexpr: pass\npending: '#391'\nexpect: {value: 1}\n",
  };
  const BAD =
    "FAIL b/bad: `kind` must be one of load, run, resume, backward, expr, value, defaults, " +
    'fingerprint, cli, json, not "nope"';
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

  /** Return a new temporary directory, removed after the test. */
  function caseRoot(): string {
    const root = mkdtempSync(join(tmpdir(), "conformance-cases-"));
    roots.push(root);
    return root;
  }

  /**
   * Run `main` over `files` in `root`'s `cases/` directory, with `fixtures` beside it and `kinds`
   * injected; return its exit code and output. A test reading `root` after passes its own, not
   * the last in `roots`, which a timed-out test still running may have added to.
   */
  async function run(
    argv: string[],
    files: Record<string, string> = FILES,
    fixtures: Record<string, string> = {},
    kinds: Partial<Record<Kind, KindRunner>> = { expr },
    root = caseRoot(),
  ) {
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
      "8 cases: 1 passed, 6 failed, 1 pending",
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
      "r/host.yaml": "kind: run\nflow: x\nhost: {tools: {}, skills: {}}\nexpect: {output: {}}\n",
      "r/skills.yaml": "kind: run\nflow: x\nhost: {skills: {}}\nexpect: {output: {}}\n",
      "r/ok.yaml": "kind: run\nflow: x\nhost: {tools: {}}\nexpect: {output: {}}\n",
    };
    const ids: string[] = [];
    const passing: KindRunner = (case_) => {
      ids.push(String(case_["flow"]));
    };
    expect(await run([], files, {}, { run: passing })).toEqual({
      code: 1,
      out: [
        "FAIL r/host: `host.skills` is not supported by the TS runner yet",
        "FAIL r/skills: `host.skills` is not supported by the TS runner yet",
        "3 cases: 1 passed, 2 failed, 0 pending",
      ],
      err: [],
    });
    expect(ids).toEqual(["x"]);
  });

  const ALL = [
    "a/boom", "a/ok", "a/x",
    "b/bad", "b/diag", "b/pending_fails", "b/pending_passes", "b/y",
  ];
  const FAILING = new Set(["a/boom", "a/x", "b/bad", "b/diag", "b/pending_passes", "b/y"]);

  test.each([
    [["--group", "a"], ["a/boom", "a/ok", "a/x"]],
    [["--group", "a", "--group", "b"], ALL],
    [["--case", "a/ok", "--case", "b/y"], ["a/ok", "b/y"]],
    [["--kind", "load"], ["b/bad", "b/diag", "b/y"]],
    [["--diagnostic-code", "compose.load"], ["b/bad", "b/diag"]],
    [["--diagnostic-code", "other"], ["b/bad"]],
    [["--group", "b", "--kind", "expr"], ["b/bad", "b/pending_fails", "b/pending_passes"]],
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

  test("runs `json` cases with the runner's own kind table", async () => {
    const files = {
      "j/read.yaml": 'kind: json\nread: "[1, 1.0]"\nmode: codec\nexpect: {value: [1, 1.0]}\n',
      "j/log.yaml":
        "kind: json\nwrite: {b: 1.0, a: 1}\nform: log\nexpect: {text: '{\"a\":1,\"b\":1.0}'}\n",
      "j/wrong.yaml": "kind: json\nwrite: 1.0\nform: log\nexpect: {text: '1'}\n",
    };
    expect(await run([], files, {}, RUNNERS)).toEqual({
      code: 1,
      out: [
        "FAIL j/wrong: text differs",
        "  --- actual",
        "  \"1.0\"",
        "  --- expected",
        "  \"1\"",
        "3 cases: 2 passed, 1 failed, 0 pending",
      ],
      err: [],
    });
  });

  test("runs `defaults` cases with the runner's own kind table", async () => {
    const files = {
      "d/ok.yaml":
        'kind: defaults\nenv: {YIAN_MAX_TOTAL_NODES: "20000"}\n' +
        "expect: {defaults: {MAX_TOTAL_NODES: 20000}}\n",
      "d/unknown.yaml":
        "kind: defaults\nenv: {}\nexpect: {defaults: {ZZ: 1, AA: 1, MAX_TOTAL_NODES: 1}}\n",
      "d/raises.yaml":
        "kind: defaults\nenv: {YIAN_MAX_TOTAL_NODES: lots}\n" +
        "expect: {raises: {class: DefaultsError, message: 'YIAN_MAX_TOTAL_NODES=\"lots\" is not " +
        "an integer; use an integer from 1 to 9007199254740991, or unset it'}}\n",
    };
    expect(await run([], files, {}, RUNNERS)).toEqual({
      code: 1,
      out: [
        "FAIL d/unknown: `expect.defaults` names unknown entries: AA, ZZ",
        "3 cases: 2 passed, 1 failed, 0 pending",
      ],
      err: [],
    });
  });

  test("runs `value` cases with the runner's own kind table", async () => {
    const files = {
      "v/kept.yaml":
        "kind: value\ntype: Any\nvalue: {$float: inf}\n" +
        "expect: {value: {$float: inf}, kind: float}\n",
      "v/raises.yaml":
        "kind: value\ntype: list(int)\nvalue: [1, 2.5]\nexpect:\n  raises: " +
        "{class: TypeCheckError, message: 2.5 does not match declared type int, path: [1]}\n",
      "v/wrong.yaml": "kind: value\ntype: float\nvalue: 2.0\nexpect: {value: 2, kind: float}\n",
      "v/cap.yaml":
        'kind: value\nenv: {YIAN_MAX_SHOWN_VALUE_CHARS: "10"}\ntype: int\n' +
        "value: abcdefghijklmnop\nexpect:\n  raises: {class: TypeCheckError, " +
        "message: '\"abcdefgh… does not match declared type int', path: []}\n",
      // Runs after `v/cap.yaml`; with no `env` it is cut at the default 200 again.
      "v/later.yaml":
        "kind: value\ntype: int\nvalue: abcdefghijklmnop\nexpect:\n  raises: " +
        "{class: TypeCheckError, message: '\"abcdefghijklmnop\" does not match declared " +
        "type int', path: []}\n",
    };
    expect(await run([], files, {}, RUNNERS)).toEqual({
      code: 1,
      out: [
        "FAIL v/wrong: value differs",
        "  --- actual",
        "  2.0",
        "  --- expected",
        "  2",
        "5 cases: 4 passed, 1 failed, 0 pending",
      ],
      err: [],
    });
  });

  test("runs `expr` cases with the runner's own kind table", async () => {
    const files = {
      "e/expr.yaml": "kind: expr\nexpr: x.n * 2\nvalues: {x: {n: 3}}\nexpect: {value: 6}\n",
      "e/template.yaml": "kind: expr\ntemplate: n=${n}\nvalues: {n: 1}\nexpect: {value: n=1}\n",
      "e/strict.yaml": 'kind: expr\ntemplate: "${n}!"\nmode: strict\nvalues: {n: 1}\nexpect: {value: "1!"}\n',
    };
    expect(await run([], files, {}, RUNNERS)).toEqual({
      code: 0,
      out: ["3 cases: 3 passed, 0 failed, 0 pending"],
      err: [],
    });
  });

  /** Return `lines` as a YAML literal block, the value of a key at the top of a case file. */
  const block = (...lines: string[]) => `|\n${lines.map((line) => `  ${line}`).join("\n")}\n`;
  const lookup = (tool: string, ...comment: string[]) =>
    block(
      "id: t",
      "name: t",
      "input:",
      "  q: !str",
      "nodes:",
      ...comment,
      `  find: {kind: tool, tool_id: ${tool}, input: {q: "\${input.q}"}}`,
      "output: ${find.output}",
    );
  const CYCLE = block(
    "id: p",
    "name: p",
    "defs:",
    "  child:",
    "    input:",
    "      x: !str",
    "    nodes:",
    '      a: {kind: agent, input: {x: "${b.output}"}, output: str, prompt: "Echo ${x}."}',
    '      b: {kind: agent, input: {x: "${a.output}"}, output: str, prompt: "Echo ${x}."}',
    "    output: ${a.output}",
    "input:",
    "  s: !str",
    "nodes:",
    '  c: {kind: call, call: child, input: {x: "${input.s}"}}',
    "output: ${c.output}",
  );
  const CYCLE_DIAGNOSTIC =
    "{code: compose.load, frames: [" +
    "{label: flow.yaml, range: [5, 5, 11, 1], note: definition 'child'}, " +
    "{label: flow.yaml, range: [8, 7, 8, 8]}, {label: flow.yaml, range: [9, 7, 9, 8]}], " +
    "notes: [a depends on b (a.input.x), b depends on a (b.input.x)]";

  test("runs `load` cases with the runner's own kind table", async () => {
    const files = {
      "l/ok.yaml": `kind: load\nflow: ${lookup("lookup")}expect: {load: ok}\n`,
      "l/cycle.yaml":
        `kind: load\nflow: ${CYCLE}expect: {load: {diagnostics: [${CYCLE_DIAGNOSTIC}, ` +
        `message: 'defs entry ''child'': flow has a cycle involving ["a","b"]; flows must be ` +
        "acyclic'}]}}\n",
      // An expected diagnostic without `message` doesn't compare it.
      "l/no_message.yaml":
        `kind: load\nflow: ${CYCLE}expect: {load: {diagnostics: [${CYCLE_DIAGNOSTIC}}]}}\n`,
      "l/wrong.yaml":
        `kind: load\nflow: ${lookup("lookup")}` +
        "expect: {load: {diagnostics: [{code: compose.load}]}}\n",
    };
    expect(await run([], files, {}, RUNNERS)).toEqual({
      code: 1,
      out: [
        "FAIL l/wrong: load differs",
        "  --- actual",
        "  ok",
        "  --- expected",
        "  diagnostics:",
        "    - code: compose.load",
        "4 cases: 3 passed, 1 failed, 0 pending",
      ],
      err: [],
    });
  });

  test("runs `fingerprint` cases with the runner's own kind table", async () => {
    const pair = (other: string, fingerprint: string) =>
      `kind: fingerprint\nflow: ${lookup("lookup")}other: ${other}` +
      `expect: {fingerprint: ${fingerprint}}\n`;
    const files = {
      "f/same.yaml": pair(lookup("lookup", "  # A comment changes the text, not the flow"), "same"),
      "f/different.yaml": pair(lookup("search"), "different"),
      "f/wrong.yaml": pair(lookup("search"), "same"),
    };
    expect(await run([], files, {}, RUNNERS)).toEqual({
      code: 1,
      out: [
        "FAIL f/wrong: fingerprint differs",
        "  --- actual",
        "  different",
        "  --- expected",
        "  same",
        "3 cases: 2 passed, 1 failed, 0 pending",
      ],
      err: [],
    });
  });

  test.each([[["--bogus"]], [["--group"]], [["a/ok"]]])("exits 2 on %p", async (argv) => {
    const { code, out, err } = await run(argv);
    expect(code).toBe(2);
    expect(out).toEqual([]);
    expect(err).toEqual([
      "usage: conformance [--group GROUP] [--kind KIND] [--case GROUP/NAME] " +
        "[--diagnostic-code CODE] [--crash-sweep] [--write-logs] [--write-expect]",
    ]);
  });

  test("writes a golden log with --write-logs, then compares it byte for byte", async () => {
    const kinds = { run: RUNNERS.run! };
    const root = caseRoot();
    const written = await run(["--write-logs", "--case", "a/two"], { "a/two.yaml": TWO_NODES }, {}, kinds, root);
    expect(written.code).toBe(0);
    const golden = readFileSync(join(root, "cases", "a", "two.log.jsonl"), "utf8");
    const lines = golden.trimEnd().split("\n");
    expect(lines).toHaveLength(17);
    expect(lines[1]).toContain('"payload":{"key":"system.now","kind":"datetime","node_id":"__start__","value":"<masked>"}');
    expect(lines[3]).toContain('"payload":{"key":"system.today","kind":"date","node_id":"__start__","value":"<masked>"}');
    expect(lines.every((line) => line.startsWith('{"at":"<masked>",'))).toBe(true);
    const pinned = { "a/two.yaml": TWO_NODES, "a/two.log.jsonl": golden };
    expect(await run([], pinned, {}, kinds)).toEqual({
      code: 0,
      out: ["1 case: 1 passed, 0 failed, 0 pending"],
      err: [],
    });
    const changed = { ...pinned, "a/two.log.jsonl": golden.replace('"seq":16', '"seq":17') };
    const { code, out } = await run([], changed, {}, kinds);
    expect(code).toBe(1);
    expect(out[0]).toBe("FAIL a/two: the run log differs from two.log.jsonl at record 16");
  });

  test("refuses --write-logs without --case", async () => {
    const { code, out, err } = await run(["--write-logs"], { "a/two.yaml": TWO_NODES });
    expect(code).toBe(2);
    expect(out).toEqual([]);
    expect(err[0]).toBe("--write-logs needs --case: name each case whose golden log to write");
  });

  test("sweeps an eligible run case with --crash-sweep", async () => {
    const files = { "a/two.yaml": TWO_NODES };
    // The sweep's logs, counted before `main` removes the case directory.
    const cuts: string[][] = [];
    const counting: KindRunner = async (case_, ctx) => {
      await RUNNERS.run!(case_, ctx);
      cuts.push(readdirSync(ctx.dir).filter((name) => /^sweep-[0-9]+\.jsonl$/.test(name)));
    };
    const { code, out } = await run(["--crash-sweep"], files, {}, { run: counting });
    expect(out).toEqual(["1 case: 1 passed, 0 failed, 0 pending"]);
    expect(code).toBe(0);
    expect(cuts.map((names) => names.length)).toEqual([16]);
    expect((await run([], files, {}, { run: counting })).code).toBe(0);
    expect(cuts.map((names) => names.length)).toEqual([16, 0]);
  }, SWEEP);

  test("sweeps a learnable run case, whose captures name their sources' records by seq", async () => {
    const learnable = TWO_NODES.replace("num_workers: 0}", "num_workers: 0, learnable: true}");
    expect(learnable).not.toBe(TWO_NODES);
    const files = { "a/two.yaml": learnable };
    const { code, out } = await run(["--crash-sweep"], files, {}, { run: RUNNERS.run! });
    expect(out).toEqual(["1 case: 1 passed, 0 failed, 0 pending"]);
    expect(code).toBe(0);
  }, SWEEP);

  test("restores a changed expectation with --write-expect, in the case's own form", async () => {
    const kinds = { run: RUNNERS.run! };
    const edited = FORMS.replace(
      "{value: {big: 9007199254740993, f: 2.0, id: <id:1>}}",
      "{value: {big: 1, f: 3.0, id: wrong}}",
    );
    expect(edited).not.toBe(FORMS);
    const argv = ["--write-expect", "--case", "a/forms"];
    const root = caseRoot();
    const written = await run(argv, { "a/forms.yaml": edited }, {}, kinds, root);
    expect(written).toEqual({
      code: 0,
      out: ["wrote a/forms: trace", "1 case: 1 passed, 0 failed, 0 pending"],
      err: [],
    });
    const path = join(root, "cases", "a", "forms.yaml");
    const text = readFileSync(path, "utf8");
    // Comments, styles and key order stay.
    expect(text).toBe(FORMS);
    // The float stays a float, the int past 2^53 exact, and the placeholder a placeholder.
    const trace = (readCase(text)["expect"] as Case)["trace"] as unknown[][];
    expect(trace[9]).toEqual([
      "pool_commit",
      "__end__",
      { value: { big: 9007199254740993n, f: 2, id: "<id:1>" } },
    ]);
    expect(typeof (trace[9]![2] as { value: Case }).value["f"]).toBe("number");
    // Run again, it writes nothing.
    const again = await run(argv, { "a/forms.yaml": text }, {}, kinds, root);
    expect(again.out).toEqual(["1 case: 1 passed, 0 failed, 0 pending"]);
    expect(readFileSync(path, "utf8")).toBe(text);
  });

  test("--write-expect writes a whole number in an event or chunk without a fraction", async () => {
    const cut = readFileSync(join(CASES, "run-agent", "model_stream_cut.yaml"), "utf8");
    const wrong = cut.replace("chunk: lo, final: false, n: 1, index: 1}", "chunk: lo, final: false, n: 7, index: 1}");
    expect(wrong).not.toBe(cut);
    const root = caseRoot();
    const written = await run(["--write-expect"], { "a/cut.yaml": wrong }, {}, { run: RUNNERS.run! }, root);
    expect(written.out).toEqual(["wrote a/cut: stream", "1 case: 1 passed, 0 failed, 0 pending"]);
    // Numbers compare by value in a stream, so the int stays an int, not `1.0`.
    expect(readFileSync(join(root, "cases", "a", "cut.yaml"), "utf8")).toBe(cut);
  });

  test("--write-expect leaves a `?` in a flow collection bare, as YAML 1.2 allows", async () => {
    const asked = FORMS.replace('big: "${input.big}"}', 'big: "${input.big}", q: "why?"}')
      .replace("output: {id: <id:1>, f: 2.0, big: 9007199254740993}", "output: {id: <id:1>, f: 2.0, big: 9007199254740993, q: \"why?\"}")
      .replace("{value: {big: 9007199254740993, f: 2.0, id: <id:1>}}", "{value: {big: 1}}");
    expect(asked).not.toBe(FORMS);
    const root = caseRoot();
    const written = await run(["--write-expect"], { "a/forms.yaml": asked }, {}, { run: RUNNERS.run! }, root);
    expect(written.out).toEqual(["wrote a/forms: trace", "1 case: 1 passed, 0 failed, 0 pending"]);
    const text = readFileSync(join(root, "cases", "a", "forms.yaml"), "utf8");
    expect(text).toContain("q: why?}}]");
    expect(text).not.toContain('q: "why?"}}]');
  });

  test("--write-expect changes a block entry in place, keeping its comment and style", async () => {
    const kinds = { run: RUNNERS.run! };
    const block = FORMS.replace(
      "  - [pool_commit, __end__, {value: {big: 9007199254740993, f: 2.0, id: <id:1>}}]\n",
      "  - - pool_commit\n    - __end__\n    # The output, as the flow binds it.\n" +
        "    - value: {big: 9007199254740993, f: 2.0, id: <id:1>}\n",
    );
    expect(block).not.toBe(FORMS);
    const edited = block.replace("    - value: {big: 9007199254740993,", "    - value: {big: 1,");
    expect(edited).not.toBe(block);
    const root = caseRoot();
    const written = await run(["--write-expect"], { "a/forms.yaml": edited }, {}, kinds, root);
    expect(written.out).toEqual(["wrote a/forms: trace", "1 case: 1 passed, 0 failed, 0 pending"]);
    const text = readFileSync(join(root, "cases", "a", "forms.yaml"), "utf8");
    expect(text).toBe(block);
  });

  test("--write-expect fails a case whose expect it can't write back in its own text", async () => {
    const kinds = { run: RUNNERS.run! };
    // Padding inside braces is a spelling the writer doesn't keep.
    const padded = FORMS.replace("{value: 1}]", "{ value: 2 }]");
    const root = caseRoot();
    const { code, out } = await run(["--write-expect"], { "a/forms.yaml": padded }, {}, kinds, root);
    expect(code).toBe(1);
    expect(out[0]).toBe(
      "FAIL a/forms: --write-expect can't write expect back: the writer doesn't reproduce its text",
    );
    expect(out.at(-1)).toBe("1 case: 0 passed, 1 failed, 0 pending");
    expect(readFileSync(join(root, "cases", "a", "forms.yaml"), "utf8")).toBe(padded);
  });

  test("--write-expect leaves a case whose output differs as it is, and fails it", async () => {
    const kinds = { run: RUNNERS.run! };
    const wrong = FORMS.replace("output: {id: <id:1>, f: 2.0,", "output: {id: <id:1>, f: 3.0,")
      .replace("{value: 1}", "{value: 2}");
    const root = caseRoot();
    const { code, out } = await run(["--write-expect"], { "a/forms.yaml": wrong }, {}, kinds, root);
    expect(code).toBe(1);
    expect(out[0]).toBe("FAIL a/forms: output differs");
    expect(out.at(-1)).toBe("1 case: 0 passed, 1 failed, 0 pending");
    expect(readFileSync(join(root, "cases", "a", "forms.yaml"), "utf8")).toBe(wrong);
  });
});

// A run whose trace holds a float with no fraction, an int past 2^53 and a placeholder.
const FORMS = `# The run's own id, a float and an int past 2^53 flow through to the output.
kind: run
flow: |
  id: forms
  name: forms
  input: {n: !int, big: !int}
  nodes:
    one: {kind: code, output: int, code: return 1}
  output: {id: "\${system.run_id}", f: "\${input.n / 2}", big: "\${input.big}"}
inputs: {n: 4, big: 9007199254740993}
run: {num_workers: 0}
expect:
  output:
    status: succeeded
    output: {id: <id:1>, f: 2.0, big: 9007199254740993}
  # Without a run id, the run makes one up.
  trace:
  - [run_started, null, {learnable: false, num_workers: 0}]
  - [pool_commit, __start__, {key: system.now}]
  - [pool_commit, __start__, {key: system.run_id, value: <id:1>}]
  - [pool_commit, __start__, {key: system.today}]
  - [pool_commit, __start__, {value: {big: 9007199254740993, n: 4}}]
  - [node_started, __start__]
  - [pool_commit, __start__, {value: {big: 9007199254740993, n: 4}}]
  - [node_succeeded, __start__, {commit: 6}]
  - [node_started, __end__]
  - [pool_commit, __end__, {value: {big: 9007199254740993, f: 2.0, id: <id:1>}}]
  - [node_succeeded, __end__, {commit: 9}]
  - [node_started, one]
  - [pool_commit, one, {value: 1}]
  - [node_succeeded, one, {commit: 12}]
`;

// A serial run of two CODE nodes: 17 records, so a crash sweep makes 16 cuts, the first 4 inside the
// run start.
const TWO_NODES = `kind: run
flow: |
  id: two
  name: two
  nodes:
    a: {kind: code, code: return "x", output: str}
    b: {kind: code, code: return inputs.v + "y", input: {v: "\${a.output}"}, output: str}
  output: \${b.output}
run: {run_id: r1, num_workers: 0}
expect:
  output: {status: succeeded, output: xy}
`;

describe("sourcesByRecord", () => {
  test("names each capture source by its record's kind and node, and each commit by its fields", () => {
    const source = (seq: bigint) => ({ route: { kind: "source", source: { execution_seq: seq } } });
    const entries = [
      ["run_started", null],
      ["node_started", "a"],
      ["pool_commit", "a", { value: 2n }],
      ["node_succeeded", "a", { commit: 2n, forward_capture: source(1n) }],
    ];
    expect(sourcesByRecord(entries)).toEqual([
      [
        "node_succeeded",
        "a",
        { commit: { value: 2n }, forward_capture: { route: { kind: "source", source: { execution_seq: "node_started a" } } } },
      ],
    ]);
  });

  test("fails on a success naming a seq that isn't a pool_commit", () => {
    const entries = [["node_started", "a"], ["node_succeeded", "a", { commit: 0n }]];
    expect(() => sourcesByRecord(entries)).toThrow(new CaseFailure("a success names seq 0, which isn't a pool_commit"));
  });

  test("fails on a source naming a seq the log doesn't hold", () => {
    const entries = [["node_succeeded", "a", { route: { source: { execution_seq: 7n } } }]];
    expect(() => sourcesByRecord(entries)).toThrow(
      new CaseFailure("a capture source names seq 7, which the log doesn't hold"),
    );
  });
});

describe("pooled", () => {
  const results = (entries: unknown[][]) => ({ trace: entries, events: [], stream: [], tool_calls: {} });

  test("names each from seq by its record's kind and node, so the interleaving doesn't show", () => {
    const aFirst = [
      ["run_started", null, { num_workers: 2n }],
      ["pool_commit", "a", { value: 1n }],
      ["pool_commit", "b", { value: 2n }],
      ["pool_commit", "c", { from: { a: 1n, b: 2n } }],
    ];
    const bFirst = [aFirst[0]!, aFirst[2]!, aFirst[1]!, ["pool_commit", "c", { from: { a: 2n, b: 1n } }]];
    expect(pooled(results(aFirst))["trace"]).toEqual([
      ["run_started", null, { num_workers: 2n }],
      ["pool_commit", "a", { value: 1n }],
      ["pool_commit", "b", { value: 2n }],
      ["pool_commit", "c", { from: { a: "pool_commit a", b: "pool_commit b" } }],
    ]);
    expect(pooled(results(bFirst))).toEqual(pooled(results(aFirst)));
  });

  test("fails on a from seq the log doesn't hold", () => {
    const entries = [["pool_commit", "c", { from: { a: 4n } }]];
    expect(() => pooled(results(entries))).toThrow(new CaseFailure("a derived commit names seq 4, which the log doesn't hold"));
  });

  test("fails on a from seq naming a record that isn't the name's pool_commit", () => {
    const wrong = (source: unknown[]) => [source, ["pool_commit", "c", { from: { a: 0n } }]];
    const message = "a derived commit names seq 0 for `a`, which isn't its pool_commit";
    expect(() => pooled(results(wrong(["node_started", "a"])))).toThrow(new CaseFailure(message));
    expect(() => pooled(results(wrong(["pool_commit", "a", { key: "a#0/__start__" }])))).toThrow(new CaseFailure(message));
    expect(() => pooled(results(wrong(["pool_commit", "b", { value: 1n }])))).toThrow(new CaseFailure(message));
    expect(pooled(results([["pool_commit", "s", { key: "a" }], ["pool_commit", "c", { from: { a: 0n } }]]))["trace"]).toEqual([
      ["pool_commit", "c", { from: { a: "pool_commit s" } }],
      ["pool_commit", "s", { key: "a" }],
    ]);
  });
});

describe("crashSweep", () => {
  const made: string[] = [];
  afterEach(() => {
    for (const dir of made.splice(0)) rmSync(dir, { recursive: true, force: true });
  });

  /** Return the prepared `TWO_NODES` case and its set-up directory. */
  function twoNodes(): [Case, CaseContext] {
    const case_ = prepare(checkCase(readCase(TWO_NODES)));
    const ctx = setUp(case_, join(CASES, "..", "fixtures"));
    made.push(ctx.dir);
    return [case_, ctx];
  }

  test("resumes every cut and passes", async () => {
    const [case_, ctx] = twoNodes();
    const cuts: string[] = [];
    await crashSweep(case_, ctx, { beforeResume: (log) => cuts.push(log) });
    // 16 cuts; the 4 inside the run start are refused instead of resumed.
    expect(cuts).toHaveLength(12);
  }, SWEEP);

  test("fails a resume that drops a record", async () => {
    const [case_, ctx] = twoNodes();
    const dropSecond = (log: string) => {
      const lines = readFileSync(log, "utf8").split("\n");
      if (lines.length > 3) writeFileSync(log, [lines[0], ...lines.slice(2)].join("\n"));
    };
    const failure = crashSweep(case_, ctx, { beforeResume: dropSecond });
    await expect(failure).rejects.toBeInstanceOf(CaseFailure);
    await expect(failure).rejects.toThrow("the resume raised (crash sweep, cut after record 4)");
  }, SWEEP);

  test("a cut after a retried tool's commit resumes from the recorded steps and succeeds", async () => {
    // The flaky tool fails twice, so its node records two steps before the commit. A cut right
    // after the commit runs the node again, replaying those steps: only the entry the last call
    // took goes back, not the two the replayed steps already hold.
    const text = `kind: run
flow: |
  id: t
  name: t
  nodes:
    call:
      kind: tool
      tool_id: flaky
      retries: 2
  output: \${call.output}
env:
  YIAN_TOOL_RETRY_DELAY_SECONDS: '1'
run: {run_id: r1, num_workers: 0}
host:
  tools:
    flaky:
      parameters: {type: object, properties: {}}
      results:
      - raise: {class: TimeoutError, message: a}
      - raise: {class: TimeoutError, message: b}
      - return: ok
  jitter: [0.0, 0.0]
expect:
  output: {status: succeeded, output: ok}
`;
    const case_ = prepare(checkCase(readCase(text)));
    const ctx = setUp(case_, join(CASES, "..", "fixtures"));
    made.push(ctx.dir);
    await crashSweep(case_, ctx);
  }, SWEEP);

  test("a host.models case is swept: each cut resumes with the replies after it put back, and passes", async () => {
    // The agent's first reply calls `lookup`, its second answers. A cut after the first model_call
    // replays that reply and takes the second from the script again; a cut before it takes both.
    const text = `kind: run
flow: |
  id: m
  name: m
  nodes:
    research:
      kind: agent
      tools: [lookup]
      output: str
      llm_config: {provider: openai, model: gpt-5.5}
      prompt: "Look up x."
  output: \${research.output}
env: {OPENAI_API_KEY: "<key:1>"}
run: {run_id: r1, num_workers: 0}
host:
  tools:
    lookup:
      parameters: {type: object, properties: {q: {type: string}}, required: [q]}
      results:
      - return: X1
  models:
    research:
      agent:
      - reply:
          format: yian.message.v1
          role: assistant
          parts: [{type: tool_call, id: c1, name: lookup, arguments: {q: x}}]
      - reply: {format: yian.message.v1, role: assistant, parts: [{type: text, text: found X1}]}
expect:
  output: {status: succeeded, output: found X1}
`;
    const case_ = prepare(checkCase(readCase(text)));
    expect(sweepable(case_)).toBe(true);
    const ctx = setUp(case_, join(CASES, "..", "fixtures"));
    made.push(ctx.dir);
    const cuts: string[] = [];
    await crashSweep(case_, ctx, { beforeResume: (log) => cuts.push(log) });
    expect(cuts.length).toBeGreaterThan(0);
  }, SWEEP);

  test("fails a resume whose model request differs from its scripted one, though its reply is the same", async () => {
    // The agent's second request carries lookup's result. A cut right after that result is
    // recorded makes the resume send the request again, now with the rewritten result; the
    // scripted reply hides it from the output and the node outcomes.
    const text = `kind: run
flow: |
  id: m
  name: m
  nodes:
    research:
      kind: agent
      tools: [lookup]
      output: str
      llm_config: {provider: openai, model: gpt-5.5}
      system_prompt: Help.
      prompt: "Look up x."
  output: \${research.output}
env: {OPENAI_API_KEY: "<key:1>"}
run: {run_id: r1, num_workers: 0}
host:
  tools:
    lookup:
      parameters: {type: object, properties: {q: {type: string}}, required: [q]}
      results:
      - return: X1
  models:
    research:
      agent:
      - reply:
          format: yian.message.v1
          role: assistant
          parts: [{type: tool_call, id: c1, name: lookup, arguments: {q: x}}]
      - reply: {format: yian.message.v1, role: assistant, parts: [{type: text, text: found X1}]}
        request:
          messages:
          - {format: yian.message.v1, role: system, parts: [{type: text, text: Help.}]}
          - {format: yian.message.v1, role: user, parts: [{type: text, text: Look up x.}]}
          - format: yian.message.v1
            role: assistant
            parts: [{type: tool_call, id: c1, name: lookup, arguments: {q: x}}]
          - format: yian.message.v1
            role: tool
            parts: [{type: tool_result, call_id: c1, name: lookup, content: X1}]
          tools:
          - name: lookup
            description: ""
            parameters: {type: object, properties: {q: {type: string}}, required: [q]}
expect:
  output: {status: succeeded, output: found X1}
`;
    const case_ = prepare(checkCase(readCase(text)));
    const ctx = setUp(case_, join(CASES, "..", "fixtures"));
    made.push(ctx.dir);
    const rewrite = (log: string) => {
      writeFileSync(log, readFileSync(log, "utf8").replace('"output":"X1"', '"output":"X9"'));
    };
    const failure = crashSweep(case_, ctx, { beforeResume: rewrite });
    await expect(failure).rejects.toBeInstanceOf(CaseFailure);
    await expect(failure).rejects.toThrow(
      "the resume's host calls differ from the script (crash sweep, cut after record 10)\n" +
        "`host.models.research.agent[1].request` differs",
    );
  }, SWEEP);

  test("fails a resume that leaves a script entry unused", async () => {
    // An entry nothing calls: the uncut run leaves it too, so each resume must report it.
    const spare = "host:\n  tools:\n    spare:\n      parameters: {type: object, properties: {}}\n      results: [{return: s}]\n";
    const case_ = prepare(checkCase(readCase(TWO_NODES.replace("expect:", `${spare}expect:`))));
    const ctx = setUp(case_, join(CASES, "..", "fixtures"));
    made.push(ctx.dir);
    const failure = crashSweep(case_, ctx);
    await expect(failure).rejects.toThrow("`host.tools.spare.results` has 1 entry left (crash sweep, cut after record 4)");
  }, SWEEP);

  test("fails a filled secret in a sweep's run log", async () => {
    const case_ = prepare(checkCase(readCase(TWO_NODES.replace("expect:", 'env: {OPENAI_API_KEY: "<key:1>"}\nexpect:'))));
    const ctx = setUp(case_, join(CASES, "..", "fixtures"));
    made.push(ctx.dir);
    const leak = (log: string) => {
      writeFileSync(log, readFileSync(log, "utf8").replace('"value":"x"', '"value":"sk-conformance-1"'));
    };
    const failure = crashSweep(case_, ctx, { beforeResume: leak });
    await expect(failure).rejects.toThrow("secret sk-conformance-1 appears in the run log (crash sweep, cut after record 9)");
  }, SWEEP);

  test("fails the strong oracle when a recorded value changes before the resume", async () => {
    const [case_, ctx] = twoNodes();
    const rewrite = (log: string) => {
      const text = readFileSync(log, "utf8");
      const recorded = '"kind":"str","node_id":"a","value":"x"';
      writeFileSync(log, text.replace(recorded, recorded.replace('"x"', '"z"')));
    };
    const failure = crashSweep(case_, ctx, { beforeResume: rewrite });
    await expect(failure).rejects.toBeInstanceOf(CaseFailure);
    await expect(failure).rejects.toThrow("output (crash sweep, cut after record 10) differs");
  }, SWEEP);

  test("a sweep still going refuses no other run's appends", async () => {
    const [case_, ctx] = twoNodes();
    // Not awaited yet, as when its test timed out: other runs go on while it cuts.
    let done = false;
    const sweep = crashSweep(case_, ctx).finally(() => {
      done = true;
    });
    let during = 0;
    while (!done) {
      // Each cut writes its own `sweep-<k>.jsonl`; once one exists, the sweep is cutting.
      if (readdirSync(ctx.dir).some((name) => name.startsWith("sweep-"))) during += 1;
      const [other, otherCtx] = twoNodes();
      // Every append outside the sweep's scope goes through, so the case passes.
      await RUNNERS.run!(other, otherCtx);
    }
    await sweep;
    expect(during).toBeGreaterThan(0);
  }, SWEEP);
});

describe("runCli", () => {
  const made: string[] = [];
  afterEach(() => {
    for (const dir of made.splice(0)) rmSync(dir, { recursive: true, force: true });
  });

  /** Return a new temporary directory, removed after the test. */
  function temp(): string {
    const dir = mkdtempSync(join(tmpdir(), "conformance-cli-"));
    made.push(dir);
    return dir;
  }

  /** Run the `cli` case file `text` with the ES module `script` as the command. */
  async function cli(text: string, script: string): Promise<void> {
    const case_ = prepare(checkCase(readCase(text)));
    const ctx = setUp(case_, join(temp(), "none"));
    made.push(ctx.dir);
    const entry = join(temp(), "yian.mjs");
    writeFileSync(entry, script);
    await runCli(case_, ctx, { entry });
  }

  const q = JSON.stringify;
  const SPAWN = 30_000;
  const USERCODE = "https://auth.openai.com/api/accounts/deviceauth/usercode";
  /** A script writing `out` to stdout and `err` to stderr, then exiting `code`. */
  const PRINT = (out: string, err = "", code = 0) =>
    `process.stdout.write(${q(out)});\nprocess.stderr.write(${q(err)});\n` +
    `process.exitCode = ${code};\n`;
  /** A script posting the JSON text `body` to the usercode URL and printing the answer. */
  const POST = (body: string, headers: Record<string, string> = {}) =>
    `{\n  const answer = await fetch(${q(USERCODE)}, {method: "POST", headers: ${q(headers)}, ` +
    `body: ${q(body)}});\n  console.log(answer.status, await answer.text());\n}\n`;
  /** A case posting to the usercode URL with `request` keys added, answered 200. */
  const EXCHANGE = (request = "") =>
    `{request: {method: POST, url: ${q(USERCODE)}${request}}, response: {status: 200}}`;
  /** A script that writes `ready`, then exits 130 after writing `bye` on SIGINT or SIGTERM. */
  const WAIT =
    'const bye = () => { process.stdout.write("bye\\n"); process.exit(130); };\n' +
    'process.on("SIGINT", bye);\nprocess.on("SIGTERM", bye);\n' +
    'process.stdout.write("ready\\n");\nsetInterval(() => {}, 1000);\n';

  test(
    "compares exit, stdout and stderr, the case directory written <tmp>",
    async () => {
      const script =
        'process.stdout.write(`${process.cwd()}  \\n${process.env.HOME} x \\n`);\n' +
        'process.stderr.write("e\\n");\n';
      await cli(
        `kind: cli\nexpect: {exit: 0, stdout: ${q("<tmp>\n<tmp>/home x\n")}, stderr: "e\\n"}\n`,
        script,
      );
      await expect(cli("kind: cli\nexpect: {exit: 0}\n", PRINT("o\n", "e\n", 3))).rejects.toThrow(
        "exit 3, expected 0\n--- stdout\no\n--- stderr\ne\n",
      );
      const stdout = cli('kind: cli\nexpect: {exit: 0, stdout: "p\\n"}\n', PRINT("o\n"));
      await expect(stdout).rejects.toThrow("stdout differs\n--- actual\no\n--- expected\np\n");
      const stderr = cli('kind: cli\nexpect: {exit: 0, stderr: ""}\n', PRINT("", "e\n"));
      await expect(stderr).rejects.toThrow("stderr differs");
    },
    SPAWN,
  );

  test(
    "gives the command only PATH, HOME, COLUMNS and `env`, and `stdin`",
    async () => {
      // Node on macOS adds `__CF_USER_TEXT_ENCODING` to its own environment as it starts, whatever
      // environment it was given.
      const keys =
        "const names = Object.keys(process.env).filter((name) => name !== '__CF_USER_TEXT_ENCODING');\n" +
        "console.log(names.sort().join(' '), process.env.COLUMNS);\n";
      await cli(
        `kind: cli\nenv: {A: "1"}\nexpect: {exit: 0, stdout: "A COLUMNS HOME PATH 1000\\n"}\n`,
        keys,
      );
      const echo = 'let text = "";\nfor await (const chunk of process.stdin) text += chunk;\n' +
        "process.stdout.write(`[${text}]`);\n";
      await cli('kind: cli\nstdin: "hi\\n"\nexpect: {exit: 0, stdout: "[hi\\n]"}\n', echo);
      await cli('kind: cli\nexpect: {exit: 0, stdout: "[]"}\n', echo);
    },
    SPAWN,
  );

  test(
    "checks `json` against cli.schema.json, then compares it in codec form",
    async () => {
      const error = { format: "yian.cli.v1", error: { code: "auth.failed", message: "m" } };
      await cli(
        "kind: cli\nexpect: {exit: 0, json: {format: yian.cli.v1, error: {code: auth.failed}}}\n",
        PRINT(`${q(error)}\n`),
      );
      const models = { format: "yian.cli.v1", models: ["a"] };
      const differs = cli(
        "kind: cli\nexpect: {exit: 0, json: {format: yian.cli.v1, models: [b]}}\n",
        PRINT(`${q(models)}\n`),
      );
      await expect(differs).rejects.toThrow("json differs");
      const bare = "kind: cli\nexpect: {exit: 0, json: {models: [a]}}\n";
      const unformatted = cli(bare, PRINT('{"models": ["a"]}\n'));
      await expect(unformatted).rejects.toThrow(
        "stdout doesn't match tests/conformance/schema/cli.schema.json\n",
      );
      await expect(unformatted).rejects.toThrow("$: missing format");
      const broken = cli("kind: cli\nexpect: {exit: 0, json: {}}\n", PRINT("{nope\n"));
      await expect(broken).rejects.toThrow("stdout is not a JSON document (");
    },
    SPAWN,
  );

  test(
    "reads `json_lines` one document per non-blank line",
    async () => {
      const lines =
        '{"format": "yian.cli.v1", "event": "waiting"}\n\n' +
        '{"format": "yian.cli.v1", "event": "code", "user_code": "A", "verification_uri": "u"}\n';
      const expected =
        "[{format: yian.cli.v1, event: waiting}, " +
        "{format: yian.cli.v1, event: code, user_code: A, verification_uri: u}]";
      await cli(`kind: cli\nexpect: {exit: 0, json_lines: ${expected}}\n`, PRINT(lines));
      const bad = PRINT('{"format": "yian.cli.v1", "event": "nope"}\n');
      await expect(cli("kind: cli\nexpect: {exit: 0, json_lines: []}\n", bad)).rejects.toThrow(
        "stdout doesn't match tests/conformance/schema/cli.schema.json",
      );
    },
    SPAWN,
  );

  test(
    "compares `exists`: a dangling symlink exists, a path under a file doesn't",
    async () => {
      const script =
        'import { symlinkSync, writeFileSync } from "node:fs";\n' +
        'writeFileSync("b", "");\nsymlinkSync("nowhere", "c");\n';
      const exists = "{a: true, b: true, c: true, d: false, a/e: false}";
      await cli(`kind: cli\nfiles: {a: x}\nexpect: {exit: 0, exists: ${exists}}\n`, script);
      await expect(
        cli("kind: cli\nexpect: {exit: 0, exists: {b: false, d: true}}\n", script),
      ).rejects.toThrow("exists differs");
    },
    SPAWN,
  );

  test.each(["SIGINT", "SIGTERM"])(
    "sends %s once `after` appears and compares the command's own exit",
    async (name) => {
      const expected = 'expect: {exit: 130, stdout: "ready\\nbye\\n"}';
      await cli(`kind: cli\nsignal: {name: ${name}, after: ready}\n${expected}\n`, WAIT);
    },
    SPAWN,
  );

  test(
    "fails a signal case whose command exits before `after`, or dies from the signal",
    async () => {
      const early = cli(
        "kind: cli\nsignal: {name: SIGINT, after: ready}\nexpect: {exit: 130}\n",
        PRINT("nope\n"),
      );
      await expect(early).rejects.toThrow(
        'the command exited before "ready" appeared\n--- stdout\nnope\n--- stderr\n',
      );
      const dies = 'process.stdout.write("ready\\n");\nsetInterval(() => {}, 1000);\n';
      await expect(
        cli("kind: cli\nsignal: {name: SIGTERM, after: ready}\nexpect: {exit: 130}\n", dies),
      ).rejects.toThrow("died from signal 15");
    },
    SPAWN,
  );

  test(
    "answers a matching request from the stub",
    async () => {
      const listing =
        '{request: {method: GET, url: "https://gateway.example/v1/models?a=1", ' +
        'headers: {Authorization: "Bearer <key:1>"}}, ' +
        "response: {status: 201, json: {data: [1, 2.0]}}}";
      const form =
        '{request: {method: POST, url: "https://auth.example/token", ' +
        'form: {code: c, empty: ""}}, response: {status: 400}}';
      const script =
        `${POST('{"client_id": "x"}')}` +
        'const list = await fetch("https://gateway.example/v1/models?a=1", ' +
        '{headers: {authorization: "Bearer sk-conformance-1"}});\n' +
        "console.log(list.status, list.headers.get('content-type'), await list.text());\n" +
        'const token = await fetch("https://auth.example/token", {method: "POST", ' +
        'body: new URLSearchParams({empty: "", code: "c"})});\n' +
        "console.log(token.status, JSON.stringify(await token.text()));\n";
      // JSON with `, ` and `: ` separators; a float keeps its fraction.
      const stdout = '200 {"user_code": "A"}\n201 application/json {"data": [1, 2.0]}\n400 ""\n';
      await cli(
        `kind: cli\nhttp: [{request: {method: POST, url: ${q(USERCODE)}, json: {client_id: x}}, ` +
          `response: {status: 200, json: {user_code: A}}}, ${listing}, ${form}]\n` +
          `expect: {exit: 0, stdout: ${q(stdout)}}\n`,
        script,
      );
    },
    SPAWN,
  );

  test(
    "fails an exchange that is never requested",
    async () => {
      const unrequested = cli(`kind: cli\nhttp: [${EXCHANGE()}]\nexpect: {exit: 0}\n`, PRINT(""));
      await expect(unrequested).rejects.toThrow("`http[0]` was never requested");
    },
    SPAWN,
  );

  test.each([
    ["method", EXCHANGE().replace("POST", "GET"), `expected GET ${USERCODE}`],
    [
      "header",
      EXCHANGE(', headers: {Authorization: "Bearer <key:1>"}'),
      'header Authorization: expected "Bearer <key:1>", got null',
    ],
    [
      "json",
      EXCHANGE(", json: {client_id: x}"),
      'body: expected json {"client_id": "x"}, got "{\\"client_id\\": \\"y\\"}"',
    ],
    ["form", EXCHANGE(", form: {client_id: x}"), 'body: expected form {"client_id": "x"}'],
  ])(
    "fails a request whose %s doesn't match, naming the difference",
    async (_, exchange, difference) => {
      const case_ = `kind: cli\nhttp: [${exchange}]\nexpect: {exit: 0}\n`;
      const failure = cli(case_, POST('{"client_id": "y"}'));
      const request = `POST ${USERCODE} doesn't match \`http[0]\``;
      await expect(failure).rejects.toThrow(`${request}: ${difference}`);
      await expect(failure).rejects.not.toThrow("sk-conformance-1");
    },
    SPAWN,
  );

  test(
    "fails a request with no exchange left, `http` absent or used up",
    async () => {
      const unlisted = `POST ${USERCODE}: \`http\` has no exchange left for it`;
      await expect(cli("kind: cli\nexpect: {exit: 0}\n", POST("{}"))).rejects.toThrow(unlisted);
      await expect(
        cli(`kind: cli\nhttp: [${EXCHANGE()}]\nexpect: {exit: 0}\n`, POST("{}") + POST("{}")),
      ).rejects.toThrow(unlisted);
    },
    SPAWN,
  );

  test(
    "lets a `repeat` exchange answer any number of requests",
    async () => {
      const exchange = `{${EXCHANGE().slice(1, -1)}, repeat: true}`;
      const repeat = `kind: cli\nhttp: [${exchange}]\nexpect: {exit: 0}\n`;
      await cli(repeat, PRINT(""));
      await cli(repeat, POST("{}").repeat(3));
    },
    SPAWN,
  );

  test(
    "fails a filled secret in stdout, stderr or a run log",
    async () => {
      const env = 'kind: cli\nenv: {K: "<key:1>"}\nexpect: {exit: 0}\n';
      await expect(cli(env, "console.log(process.env.K);\n")).rejects.toThrow(
        "secret sk-conformance-1 appears in stdout",
      );
      const log =
        'import { writeFileSync } from "node:fs";\nwriteFileSync("run.jsonl", process.env.K);\n';
      await expect(cli(env, log)).rejects.toThrow("secret sk-conformance-1 appears in run.jsonl");
      // Only a secret the case filled counts.
      await cli('kind: cli\nexpect: {exit: 0}\n', 'console.log("sk-conformance-1");\n');
    },
    SPAWN,
  );
});

describe("schema.ts", () => {
  const ROOT = {};

  test("validates allOf, if and then", () => {
    const schema = {
      allOf: [{ required: ["a"] }, { required: ["b"] }],
      if: { properties: { a: { const: 1 } }, required: ["a"] },
      then: { required: ["c"] },
    };
    expect(validate(ROOT, schema, { a: 2, b: 1 })).toEqual([]);
    expect(validate(ROOT, schema, { b: 1 })).toEqual(["$: missing a"]);
    expect(validate(ROOT, schema, { a: 1, b: 1 })).toEqual(["$: missing c"]);
    expect(validate(ROOT, schema, { a: 1, b: 1, c: 1 })).toEqual([]);
    // `if` without `then` constrains nothing.
    expect(validate(ROOT, { if: { required: ["a"] } }, {})).toEqual([]);
  });

  test("validates minItems, maxItems and minLength", () => {
    const schema = { minItems: 1, maxItems: 2, items: { minLength: 2 } };
    expect(validate(ROOT, schema, ["ab", "🙂🙂"])).toEqual([]);
    expect(validate(ROOT, schema, [])).toEqual(["$: fewer than 1 items"]);
    expect(validate(ROOT, schema, ["ab", "ab", "ab"])).toEqual(["$: more than 2 items"]);
    expect(validate(ROOT, schema, ["🙂"])).toEqual(["$[0]: shorter than 2 characters"]);
  });

  test("lints allOf, if, then and the length bounds", () => {
    expect(lintSchema(ROOT, { allOf: [{ x: 1 }], if: { y: 1 }, then: { z: 1 } })).toEqual([
      "$.allOf[0].x: unknown keyword",
      "$.if.y: unknown keyword",
      "$.then.z: unknown keyword",
    ]);
    expect(lintSchema(ROOT, { minItems: -1, maxItems: 1.5, minLength: "1", if: {} })).toEqual([
      "$.minItems: not a non-negative integer",
      "$.maxItems: not a non-negative integer",
      "$.minLength: not a non-negative integer",
    ]);
  });

  test("validates the keys properties doesn't list against a schema additionalProperties", () => {
    const schema = {
      properties: { a: { type: "string" } },
      additionalProperties: { type: "integer", minimum: 0 },
    };
    expect(validate(ROOT, schema, { a: "x", b: 1n, c: 0 })).toEqual([]);
    expect(validate(ROOT, schema, { b: -1, c: "y" })).toEqual([
      "$.b: -1 is below 0",
      "$.c: string is not integer",
    ]);
    expect(validate(ROOT, schema, { b: "y" }, { closed: true })).toEqual(["$.b: string is not integer"]);
    expect(lintSchema(ROOT, { additionalProperties: { x: 1 } })).toEqual([
      "$.additionalProperties.x: unknown keyword",
    ]);
    expect(lintSchema(ROOT, { additionalProperties: true })).toEqual([
      "$.additionalProperties: not a schema object",
    ]);
  });

  test("lintSchema of cli.schema.json is clean", () => {
    const schema = loadSchema("cli.schema.json");
    expect(lintSchema(schema, schema)).toEqual([]);
  });
});

describe("the trace and golden form of the wait records", () => {
  const DEADLINE = "2026-10-24T09:00:00.000Z";
  const AT = "2026-10-24T08:00:00.000Z";
  const records: RunRecord[] = [
    new NodeStepRecord("ask", 1, { kind: "pending", call_id: "r/ask/1", tool: "ask", arguments: { q: "ok?" }, deadline: DEADLINE }),
    new NodeWaitingRecord("ask", ["r/ask/1"]),
    new NodeWaitingRecord("nap", ["r/nap/1"], {
      waits: [{ callId: "r/nap/1", tool: "sleep", inbox: null, mode: null, deadline: DEADLINE }],
    }),
    new NodeWaitingRecord("heard", ["r/heard/1"], {
      waits: [{ callId: "r/heard/1", tool: "take", inbox: "msgs", mode: "one", deadline: null }],
    }),
    new RunWaitingRecord(["ask", "nap", "heard"]),
    new NodeWokenRecord("nap", ["r/nap/1"]),
    new NodeWokenRecord("heard"),
    new PoolCommitRecord("heard", "heard", null, { take: { inbox: "msgs", events: [3], closed: false } }),
    new InboxClosedRecord("msgs"),
    new InboxWithdrawnRecord("msgs", [3]),
    new RunCancelledRecord(),
    new RunFailedRecord(new Failure({ code: "graph.terminal_unreachable", message: "END is unreachable", nodeId: "ask" })),
  ];
  const lines = records.map((record, seq) => encodeRecord(new RecordEnvelope(SCHEMA_VERSION, "r", seq + 1, record, AT)));

  test("each kind has a trace row; deadlines are left out", async () => {
    const flow = await loadFlow(`\
id: f
name: f
input:
  name: !str
nodes:
  greet:
    kind: code
    input:
      name: \${input.name}
    output: str
    code: |
      return "hi " + inputs.name
output: \${greet.output}
`);
    const entries = trace(lines, flow);
    expect(entries.slice(0, -1)).toEqual([
      ["node_step", "ask", { n: 1n, step: { arguments: { q: "ok?" }, call_id: "r/ask/1", kind: "pending", tool: "ask" } }],
      ["node_waiting", "ask", { call_ids: ["r/ask/1"] }],
      ["node_waiting", "nap", { call_ids: ["r/nap/1"], waits: [{ call_id: "r/nap/1", inbox: null, mode: null, tool: "sleep" }] }],
      ["node_waiting", "heard", { call_ids: ["r/heard/1"], waits: [{ call_id: "r/heard/1", inbox: "msgs", mode: "one", tool: "take" }] }],
      ["run_waiting", null, { node_ids: ["ask", "nap", "heard"] }],
      ["node_woken", "nap", { timed_out: ["r/nap/1"] }],
      ["node_woken", "heard"],
      ["pool_commit", "heard", { inbox: "msgs", events: [3n], closed: false }],
      ["inbox_closed", null, { inbox: "msgs" }],
      ["inbox_withdrawn", null, { inbox: "msgs", events: [3n] }],
      ["run_cancelled", null],
    ]);
    const [kind, node, payload] = entries.at(-1)!;
    expect([kind, node]).toEqual(["run_failed", null]);
    expect((payload as Case)["failure"]).toMatchObject({ node: "ask" });
  });

  test("a golden log masks each set deadline", () => {
    const golden = masked(lines.join("\n")).split("\n");
    expect(golden[0]).toContain('"deadline":"<masked>"');
    expect(golden[1]).not.toContain("deadline");
    expect(golden[2]).toContain('"deadline":"<masked>"');
    expect(golden[3]).toContain('"deadline":null');
    expect(golden.join("\n")).not.toContain(DEADLINE);
  });
});
