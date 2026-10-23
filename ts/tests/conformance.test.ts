/**
 * The conformance runner's own unit tests: reading a case file, the format check, case values,
 * comparison, the case directory, selection and the report. Not a port of
 * `tests/test_conformance.py`, which `conformance.ts` is, except for
 * `test_steering_unreached_action_fails`, which no case can cover (a passing case reaches every
 * action), and `test_memory_expect_refuses_a_session_key`. TS-only: an undeclared `expect.memory`
 * key fails the case, and a `host.memory.before` entry whose backward call raises fails it.
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
  setUp,
} from "./conformance.ts";
import type { Case, CaseContext, Kind, KindRunner } from "./conformance.ts";
import { lintSchema, loadSchema, validate } from "./schema.ts";

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
/** A `run` case whose `host.steering` gives node `chat` the `actions`. */
const STEER = (actions: string) => `${RUN}host: {steering: {chat: ${actions}}}\n`;
const STEER_TEXT = "`host.steering.chat[0].steer` must be a non-empty list of text";
const COUNTS = "`host.steering.chat[0].after` counts must be ints from 1";
const USAGE = "{input_tokens: 1, output_tokens: 2, total_tokens: 3}";
const CONSUMER_LIST = "agent, structured_retry, compose, backward, summary";
const MEMORY_CONSUMER = "consumers go under `__memory__`, and `__memory__` holds only them";
/** A `run` case whose `expect` adds `results`. */
const EXPECT = (results: string) => `kind: run\nflow: x\nexpect: {output: 1, ${results}}\n`;
const ATTEMPT_ONE = "writes `attempt: 1`, which the data form leaves out";

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
    [`${LOAD}python_files: {a: 1}`, "`python_files` values must be text"],
    ["kind: load\nflow: 1\nexpect: {load: ok}", "`flow` must be text"],
    [`${LOAD}python_flow: [a]`, "`python_flow` must be text"],
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
    [
      "kind: fingerprint\nflow: x\nother: y\npython_flow: z\nexpect: {fingerprint: same}",
      "`python_flow` is not a key of a `fingerprint` case",
    ],
    [
      "kind: fingerprint\nflow: x\nother: y\npython_files: {}\nexpect: {fingerprint: same}",
      "`python_files` is not a key of a `fingerprint` case",
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
      "a case with `host.human` declares no `ask_user` tool",
    ],
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
    [`${RUN}host: {steering: 1}`, "`host.steering` must be a mapping"],
    [STEER("{steer: [x]}"), "`host.steering` maps node ids to lists of actions"],
    [STEER("[1]"), "`host.steering.chat[0]` must be a mapping"],
    [
      STEER("[{steer: [x], interrupt: true}]"),
      "`host.steering.chat[0]` takes `steer` or `interrupt`",
    ],
    [
      STEER("[{steer: [x]}, {after: {model_call: 1}}]"),
      "`host.steering.chat[1]` takes `steer` or `interrupt`",
    ],
    [STEER("[{steer: x}]"), STEER_TEXT],
    [STEER("[{steer: []}]"), STEER_TEXT],
    [STEER("[{steer: [x, 1]}]"), STEER_TEXT],
    [STEER("[{interrupt: false}]"), "`host.steering.chat[0].interrupt` must be `true`"],
    [STEER("[{stop: true}]"), "unknown key `host.steering.chat[0].stop`"],
    [STEER("[{after: 1, interrupt: true}]"), "`host.steering.chat[0].after` must be a mapping"],
    [
      STEER("[{after: {chunk: 1}, interrupt: true}]"),
      "`host.steering.chat[0].after` is `{model_call, chunk}`, `{model_call}` or `{tool_call}`",
    ],
    [
      STEER("[{interrupt: true}, {after: {model_call: 1, tool_call: 1}, interrupt: true}]"),
      "`host.steering.chat[1].after` is",
    ],
    [STEER("[{after: {tool_call: 0}, interrupt: true}]"), COUNTS],
    [STEER("[{after: {model_call: true}, interrupt: true}]"), COUNTS],
    [STEER("[{after: {model_call: 1, chunk: 1.0}, interrupt: true}]"), COUNTS],
    [
      EXPECT("events: [[ModelCallStarted, {node_id: a, attempt: 1}]]"),
      `\`expect.events[0]\` ${ATTEMPT_ONE}`,
    ],
    [
      EXPECT("events: [[RunStarted], [ModelCallFinished, {attempt: 1.0}]]"),
      `\`expect.events[1]\` ${ATTEMPT_ONE}`,
    ],
    [EXPECT("stream: [{node_id: a, attempt: 1}]"), `\`expect.stream[0]\` ${ATTEMPT_ONE}`],
    [
      `${EXPECT("stream: []")}rules: [llm]\npython_expect: {output: 1, stream: [{attempt: 1}]}`,
      `\`python_expect.stream[0]\` ${ATTEMPT_ONE}`,
    ],
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
    `${LOAD}rules: [text]\npython_expect: {load: ok}`,
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
    STEER("[{steer: [x]}, {after: {model_call: 1, chunk: 2}, interrupt: true}, " +
      "{after: {tool_call: 1}, steer: [y]}]"),
    EXPECT("events: [[NodeStarted, {attempt: 1}], [ModelCallStarted, {attempt: 2}]], " +
      "stream: [{attempt: 2}]"),
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
        `flow: "${inf}"\npython_flow: x\nother: y\nfiles: {a: b}\npython_files: {a: "${inf}"}\n` +
          `stdin: z\n` +
          `inputs: {a: ${inf}}\nexpect: {output: ${inf}}\npython_expect: {output: ${inf}}`,
      ),
    );
    expect(prepared).toEqual({
      flow: inf,
      python_flow: "x",
      other: "y",
      files: { a: "b" },
      python_files: { a: inf },
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
    expect(raised(new TypeError("m"))).toEqual({ class: "TypeError", message: "m" });
    expect(raised(new Foo("m"))).toEqual({ class: "Foo", message: "m" });
  });

  test("names a RangeError ValueError", () => {
    expect(raised(new RangeError("m"))).toEqual({ class: "ValueError", message: "m" });
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
    await expect(failure).rejects.toThrow("raises differs\n--- actual\nclass: ValueError\n");
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
   * A `run` case of one steerable AGENT `chat` replying `done`, whose `host.steering` gives it the
   * `actions`, expecting the output `output`.
   */
  const STEERING_CASE = (actions: string, output: string) =>
    "kind: run\nflow: |\n  id: steer\n  name: steer\n" +
    "  llm_config: {provider: openai, model: gpt-5.5}\n" +
    "  nodes:\n    chat:\n      kind: agent\n      steerable: true\n      tools: []\n" +
    "      system_prompt: Help.\n      prompt: Go.\n      output: str\n" +
    "  output: ${chat.output}\n" +
    "run: {num_workers: 0}\n" +
    `host:\n  steering: {chat: ${actions}}\n  tools: {}\n` +
    "  models: {chat: {agent: [{reply: {format: yian.message.v1, role: assistant, " +
    "parts: [{type: text, text: done}]}}]}}\n" +
    `expect: {output: {status: succeeded, output: ${output}}}\n`;

  // test_steering_unreached_action_fails
  test("an action whose point the run never reaches fails the case after the comparisons", async () => {
    const actions = "[{steer: [early]}, {after: {tool_call: 1}, interrupt: true}]";
    // A divergence shows first.
    await expect(runCase(STEERING_CASE(actions, "other"))).rejects.toThrow("output differs");
    const unreached = runCase(STEERING_CASE(actions, "done"));
    await expect(unreached).rejects.toBeInstanceOf(CaseFailure);
    await expect(unreached).rejects.toThrow("`host.steering.chat[1]` was never reached");
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

  // test_memory_expect_refuses_a_session_key
  test("a session key in `expect.memory` fails the case when the runner reads it", async () => {
    const read = runCase(MEMORY_CASE("{chat: session}", "{}", "{chat: ''}"));
    await expect(read).rejects.toBeInstanceOf(CaseFailure);
    await expect(read).rejects.toThrow("`expect.memory.chat` is a session memory");
  });

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
      "usage: conformance [--group GROUP] [--kind KIND] [--case GROUP/NAME] [--rules RULE] " +
        "[--diagnostic-code CODE] [--crash-sweep] [--write-logs] [--write-expect]",
    ]);
  });

  test("writes a golden log with --write-logs, then compares it byte for byte", async () => {
    const kinds = { run: RUNNERS.run! };
    const written = await run(["--write-logs", "--case", "a/two"], { "a/two.yaml": TWO_NODES }, {}, kinds);
    expect(written.code).toBe(0);
    const golden = readFileSync(join(roots.at(-1)!, "cases", "a", "two.log.jsonl"), "utf8");
    const lines = golden.trimEnd().split("\n");
    expect(lines).toHaveLength(9);
    expect(lines[0]).toContain('"now":{"kind":"datetime","value":"<masked>"}');
    expect(lines[0]).toContain('"today":{"kind":"date","value":"<masked>"}');
    expect(lines.every((line) => line.startsWith('{"at":"<masked>",'))).toBe(true);
    const pinned = { "a/two.yaml": TWO_NODES, "a/two.log.jsonl": golden };
    expect(await run([], pinned, {}, kinds)).toEqual({
      code: 0,
      out: ["1 case: 1 passed, 0 failed, 0 pending"],
      err: [],
    });
    const changed = { ...pinned, "a/two.log.jsonl": golden.replace('"seq":8', '"seq":9') };
    const { code, out } = await run([], changed, {}, kinds);
    expect(code).toBe(1);
    expect(out[0]).toBe("FAIL a/two: the run log differs from two.log.jsonl at record 8");
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
    expect(cuts.map((names) => names.length)).toEqual([8]);
    expect((await run([], files, {}, { run: counting })).code).toBe(0);
    expect(cuts.map((names) => names.length)).toEqual([8, 0]);
  });

  test("sweeps a learnable run case, whose captures name their sources' records by seq", async () => {
    const learnable = TWO_NODES.replace("num_workers: 0}", "num_workers: 0, learnable: true}");
    expect(learnable).not.toBe(TWO_NODES);
    const files = { "a/two.yaml": learnable };
    const { code, out } = await run(["--crash-sweep"], files, {}, { run: RUNNERS.run! });
    expect(out).toEqual(["1 case: 1 passed, 0 failed, 0 pending"]);
    expect(code).toBe(0);
  });

  test("restores a changed expectation with --write-expect, in the case's own form", async () => {
    const kinds = { run: RUNNERS.run! };
    const edited = FORMS.replace(
      "{value: {big: 9007199254740993, f: 2.0, id: <id:1>}}",
      "{value: {big: 1, f: 3.0, id: wrong}}",
    );
    expect(edited).not.toBe(FORMS);
    const argv = ["--write-expect", "--case", "a/forms"];
    const written = await run(argv, { "a/forms.yaml": edited }, {}, kinds);
    expect(written).toEqual({
      code: 0,
      out: ["wrote a/forms: trace", "1 case: 1 passed, 0 failed, 0 pending"],
      err: [],
    });
    const path = join(roots.at(-1)!, "cases", "a", "forms.yaml");
    const text = readFileSync(path, "utf8");
    // Comments, styles and key order stay; `run-log` goes right after `kind`.
    expect(text).toBe(FORMS.replace("kind: run\n", "kind: run\nrules: [run-log]\n"));
    // The float stays a float, the int past 2^53 exact, and the placeholder a placeholder.
    const trace = (readCase(text)["expect"] as Case)["trace"] as unknown[][];
    expect(trace[4]).toEqual([
      "node_succeeded",
      "__end__",
      { value: { big: 9007199254740993n, f: 2, id: "<id:1>" } },
    ]);
    expect(typeof (trace[4]![2] as { value: Case }).value["f"]).toBe("number");
    // Run again, it writes nothing.
    const again = await run(argv, { "a/forms.yaml": text }, {}, kinds);
    expect(again.out).toEqual(["1 case: 1 passed, 0 failed, 0 pending"]);
    expect(readFileSync(join(roots.at(-1)!, "cases", "a", "forms.yaml"), "utf8")).toBe(text);
  });

  test("--write-expect leaves a case whose output differs as it is, and fails it", async () => {
    const kinds = { run: RUNNERS.run! };
    const wrong = FORMS.replace("output: {id: <id:1>, f: 2.0,", "output: {id: <id:1>, f: 3.0,")
      .replace("{value: 1}", "{value: 2}");
    const { code, out } = await run(["--write-expect"], { "a/forms.yaml": wrong }, {}, kinds);
    expect(code).toBe(1);
    expect(out[0]).toBe("FAIL a/forms: output differs");
    expect(out.at(-1)).toBe("1 case: 0 passed, 1 failed, 0 pending");
    expect(readFileSync(join(roots.at(-1)!, "cases", "a", "forms.yaml"), "utf8")).toBe(wrong);
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
  - [run_started, null, {inputs: {big: 9007199254740993, n: 4}, learnable: false, num_workers: 0}]
  - [node_started, __start__]
  - [node_succeeded, __start__, {value: {big: 9007199254740993, n: 4}}]
  - [node_started, __end__]
  - [node_succeeded, __end__, {value: {big: 9007199254740993, f: 2.0, id: <id:1>}}]
  - [node_started, one]
  - [node_succeeded, one, {value: 1}]
`;

// A serial run of two CODE nodes: nine records, so a crash sweep resumes eight cuts.
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
    expect(cuts).toHaveLength(8);
  });

  test("fails a resume that drops a record", async () => {
    const [case_, ctx] = twoNodes();
    const dropSecond = (log: string) => {
      const lines = readFileSync(log, "utf8").split("\n");
      if (lines.length > 3) writeFileSync(log, [lines[0], ...lines.slice(2)].join("\n"));
    };
    const failure = crashSweep(case_, ctx, { beforeResume: dropSecond });
    await expect(failure).rejects.toBeInstanceOf(CaseFailure);
    await expect(failure).rejects.toThrow("the resume raised (crash sweep, cut after record 2)");
  });

  test("fails the strong oracle when a recorded value changes before the resume", async () => {
    const [case_, ctx] = twoNodes();
    const rewrite = (log: string) => {
      const text = readFileSync(log, "utf8");
      const recorded = '"value":{"kind":"str","value":"x"}';
      writeFileSync(log, text.replace(recorded, recorded.replace('"x"', '"z"')));
    };
    const failure = crashSweep(case_, ctx, { beforeResume: rewrite });
    await expect(failure).rejects.toBeInstanceOf(CaseFailure);
    await expect(failure).rejects.toThrow("output (crash sweep, cut after record 4) differs");
  });
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
        "stdout doesn't match conformance/schema/cli.schema.json\n",
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
        "stdout doesn't match conformance/schema/cli.schema.json",
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
      // Written as Python's `json.dumps` writes it: a float keeps its fraction.
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
