// Ported from tests/cli/test_run.py: `test_bad_kv_is_rejected`, `test_inputs_json_must_be_an_object`,
// `test_inputs_file_must_be_utf8` and `test_flag_overrides_json`, on the helpers those runs go
// through (`parseKv`, `readInputsJson`, `mergeInputs`); the non-TTY run outcomes through `main`
// (`test_run_succeeds_with_input_flag`, `test_run_keeps_registered_tools`,
// `test_run_inline_code_in_process`, `test_run_reads_inputs_json`, `test_flag_overrides_json`,
// `test_failed_run_exits_nonzero`, `test_non_tty_run_omits_optional_and_defaulted_inputs`,
// `test_non_tty_inputs_json_preserves_explicit_structured_null`); the input gate
// (`test_undeclared_input_flag_is_a_bad_parameter`,
// `test_undeclared_inputs_json_key_is_a_bad_parameter`,
// `test_non_tty_run_reports_all_missing_required_inputs_before_setup`, and from
// tests/engine/test_cli_prompt.py `test_gather_non_tty_*` and the run half of
// `test_prompt_accepting_omission_preserves_start_owned_defaults`, through `main`, on a fake TTY
// for the prompts). Changed: an unknown name is quoted as `"topik"`; the CODE bodies are
// TS's, and a non-string output prints in yian's text form, not Python's repr. Added here: the
// other `cli/run.ts` helpers (`diagnosticJson`, `jsonValue`, `flowBanner`, `stopOnSignals`),
// `skillsCheck`, and `main`'s command table: the defaults check, usage errors and the routing to
// `cli/auth.ts`; `check`, on its own and confined to `--root`; `run`'s `--json` documents, its
// `run failed:` setup errors and its stops (Ctrl-C at a run's prompt, `io.stop`). `status` is
// covered up to the run inspection, which isn't ported yet; from tests/cli/test_run_log_options.py,
// the run-log checks running before the flow is read.

import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { EventEmitter, getEventListeners } from "node:events";
import { spawnSync } from "node:child_process";
import { existsSync, mkdirSync, mkdtempSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { PassThrough } from "node:stream";
import { fileURLToPath } from "node:url";

import {
  diagnosticJson,
  flowBanner,
  HostStop,
  jsonValue,
  mergeInputs,
  parseKv,
  readInputsJson,
  skillsCheck,
  stopOnSignals,
} from "../../src/cli/run.ts";
import { main } from "../../src/cli/index.ts";
import { type CliIo, printJson, UsageError, type Writer } from "../../src/cli/utils.ts";
import { Diagnostic, DiagnosticFrame, SourceRange } from "../../src/diagnostics.ts";
import { FunctionTool } from "../../src/tools/function_tools.ts";
import { registerTool, TOOL_REGISTRY } from "../../src/tools/registry.ts";
import { readRunLog, RunStartedRecord } from "../../src/runtime/log.ts";
import { FileRef } from "../../src/typesys/values.ts";
import { FakeInput, FakeOutput } from "./terminal_both_runtimes.ts";

let tmp: string;

beforeEach(() => {
  tmp = mkdtempSync(join(tmpdir(), "yian-cli-run-"));
});

afterEach(() => {
  rmSync(tmp, { recursive: true, force: true });
});

/** Write `content` to a fresh `--inputs` file and return its path. */
function inputsFile(content: string | Uint8Array): string {
  const path = join(tmp, "in.json");
  writeFileSync(path, content);
  return path;
}

/** A writer collecting what it is given. */
function sink() {
  const chunks: string[] = [];
  const out: Writer = { write: async (text) => void chunks.push(text) };
  return { out, text: () => chunks.join("") };
}

describe("parseKv", () => {
  test("bad kv is rejected", () => {
    expect(() => parseKv(["nokv"])).toThrow(new UsageError("Invalid value: --input must be k=v (got 'nokv')"));
  });

  test("splits at the first '=' and strips the key", () => {
    const parsed = parseKv([" topic =a=b", "__proto__=x", "empty="]);
    expect(Object.entries(parsed)).toEqual([
      ["topic", "a=b"],
      ["__proto__", "x"],
      ["empty", ""],
    ]);
  });
});

describe("readInputsJson", () => {
  test("inputs json must be an object", () => {
    for (const [text, kind] of [
      ["null", "null"],
      ["[1, 2]", "array"],
      ['[["topic", "rivers"]]', "array"],
      ['"rivers"', "string"],
      ["3", "number"],
      ["2.5", "number"],
      ["true", "boolean"],
    ]) {
      expect(() => readInputsJson(inputsFile(text!))).toThrow(
        new UsageError(`Invalid value for --inputs: must contain a JSON object, got ${kind}`),
      );
    }
  });

  test("invalid JSON names its line and column", () => {
    expect(() => readInputsJson(inputsFile("{bad"))).toThrow(
      new UsageError(
        "Invalid value for --inputs: invalid JSON at line 1, column 2: " +
          "Expecting property name enclosed in double quotes",
      ),
    );
    // Lines count as Python's text-mode read counts them, `\r\n` and `\r` as one break each.
    expect(() => readInputsJson(inputsFile('{\r\n"a": 1,\r"b" 2}'))).toThrow(
      new UsageError("Invalid value for --inputs: invalid JSON at line 3, column 5: Expecting ':' delimiter"),
    );
    // A BOM is kept, so it is refused as Python refuses it.
    expect(() => readInputsJson(inputsFile("\ufeff{\"a\": 1}"))).toThrow(
      new UsageError(
        "Invalid value for --inputs: invalid JSON at line 1, column 1: " +
          "Unexpected UTF-8 BOM (decode using utf-8-sig)",
      ),
    );
  });

  test("inputs file must be utf8", () => {
    // A UTF-16 export is a usage error, not a decoding crash.
    const utf16 = Buffer.concat([Buffer.from([0xff, 0xfe]), Buffer.from('{"topic": "rivers"}', "utf16le")]);
    expect(() => readInputsJson(inputsFile(utf16))).toThrow(
      new UsageError("Invalid value for --inputs: is not UTF-8 text"),
    );
  });

  test("reads an object, ints as bigints", () => {
    expect(readInputsJson(inputsFile('{"topic": "rivers", "n": 3, "x": 1.5}'))).toEqual({
      topic: "rivers",
      n: 3n,
      x: 1.5,
    });
  });
});

describe("mergeInputs", () => {
  test("flag overrides json", () => {
    const path = inputsFile('{"topic": "rivers", "n": 1}');
    expect(mergeInputs(path, ["topic=mountains"])).toEqual({ topic: "mountains", n: 1n });
    expect(mergeInputs(undefined, ["topic=mountains"])).toEqual({ topic: "mountains" });
    expect(mergeInputs(undefined, [])).toEqual({});
  });
});

describe("diagnosticJson", () => {
  test("ranges as ints; empty frames, notes and note left out", async () => {
    const frame = new DiagnosticFrame({ label: "flow.yaml", text: "x\n", range: new SourceRange(6, 9, 6, 23) });
    const noted = new DiagnosticFrame({
      label: "child.yaml",
      text: "y\n",
      range: new SourceRange(1, 1, 2, 1),
      note: "inside leaf",
    });
    const diag = new Diagnostic({ code: "compose.load", message: "bad", frames: [frame, noted], notes: ["see docs"] });
    expect(diagnosticJson(diag)).toEqual({
      code: "compose.load",
      message: "bad",
      frames: [
        { label: "flow.yaml", range: [6n, 9n, 6n, 23n] },
        { label: "child.yaml", range: [1n, 1n, 2n, 1n], note: "inside leaf" },
      ],
      notes: ["see docs"],
    });
    expect(diagnosticJson(new Diagnostic({ code: "run.setup", message: "m" }))).toEqual({
      code: "run.setup",
      message: "m",
    });
    const { out, text } = sink();
    await printJson({ error: diagnosticJson(new Diagnostic({ code: "c", message: "m", frames: [frame] })) }, out);
    expect(text()).toContain('"range":[6,9,6,23]');
  });
});

describe("jsonValue", () => {
  test("a file becomes its mapping, in lists and records too", () => {
    const file = new FileRef({ uri: "file:///a.txt", mime: "text/plain", name: "a.txt" });
    const plain = { uri: "file:///a.txt", mime: "text/plain", name: "a.txt" };
    expect(jsonValue(file)).toEqual(plain);
    expect(jsonValue([file, { inner: [file], n: 1n }])).toEqual([plain, { inner: [plain], n: 1n }]);
    expect(jsonValue("text")).toBe("text");
    expect(jsonValue(null)).toBe(null);
  });

  test("a record shaped like the escape is escaped once", async () => {
    const { out, text } = sink();
    await printJson({ output: jsonValue({ $record: 1n }) }, out);
    expect(text()).toBe('{"format":"yian.cli.v1","output":{"$record":{"$record":1}}}\n');
  });
});

describe("flowBanner", () => {
  test("boxes the name, version and description", () => {
    const { out } = sink();
    const env = () => undefined;
    expect(flowBanner({ name: "demo", description: "does things", version: "1.0" }, out, env)).toBe(
      [
        "╭────────────────────────────────────╮",
        "│ Running flow: demo  (version: 1.0) │",
        "│ Description: does things           │",
        "╰────────────────────────────────────╯",
        "",
      ].join("\n"),
    );
    expect(flowBanner({ name: null, description: "d", version: null }, out, env)).toBe(
      [
        "╭─────────────────────────╮",
        "│ Running flow: (unnamed) │",
        "│ Description: d          │",
        "╰─────────────────────────╯",
        "",
      ].join("\n"),
    );
    expect(flowBanner({ name: null, description: null, version: null }, out, env)).toBeNull();
    expect(flowBanner({ name: "", description: "", version: "" }, out, env)).toBeNull();
  });
});

describe("stopOnSignals", () => {
  const SIGNALS = ["SIGINT", "SIGTERM", "SIGHUP"] as const;
  const listeners = (target: EventEmitter) => SIGNALS.map((name) => target.listenerCount(name));

  test("each signal aborts with a HostStop naming it and removes all three listeners", () => {
    for (const name of SIGNALS) {
      const target = new EventEmitter();
      const controller = new AbortController();
      stopOnSignals(controller, target);
      expect(listeners(target)).toEqual([1, 1, 1]);
      target.emit(name);
      expect(controller.signal.reason).toBeInstanceOf(HostStop);
      expect(controller.signal.reason.signal).toBe(name);
      expect(listeners(target)).toEqual([0, 0, 0]);
    }
  });

  test("the disposer removes the listeners", () => {
    const target = new EventEmitter();
    const controller = new AbortController();
    const dispose = stopOnSignals(controller, target);
    dispose();
    expect(listeners(target)).toEqual([0, 0, 0]);
    expect(controller.signal.aborted).toBe(false);
  });

  test("an already aborted controller keeps its reason and still removes them", () => {
    const target = new EventEmitter();
    const controller = new AbortController();
    const first = new HostStop("SIGINT");
    controller.abort(first);
    stopOnSignals(controller, target);
    target.emit("SIGTERM");
    expect(controller.signal.reason).toBe(first);
    expect(listeners(target)).toEqual([0, 0, 0]);
  });
});

/** Return a `CliIo` over `env` whose streams are collected. */
function testIo(env: Record<string, string> = {}) {
  const out = sink();
  const err = sink();
  const io: CliIo = {
    stdout: out.out,
    stderr: err.out,
    stdin: new PassThrough(),
    stderrStream: new PassThrough(),
    env: (name) => (Object.hasOwn(env, name) ? env[name] : undefined),
  };
  return { io, stdout: out.text, stderr: err.text };
}

/** Write a skill directory at `tmp/skills/<name>` holding `skillMd`, and return its path. */
function skillDir(name: string, skillMd: string): string {
  const dir = join(tmp, "skills", name);
  mkdirSync(dir, { recursive: true });
  writeFileSync(join(dir, "SKILL.md"), skillMd);
  return dir;
}

const GOOD_SKILL = "---\nname: good\ndescription: Does good things.\n---\nDo good things.\n";

describe("skillsCheck", () => {
  test("text: ok lines on stdout, errors on stderr, exit 1 when one fails", async () => {
    const good = skillDir("good", GOOD_SKILL);
    const { io, stdout, stderr } = testIo();
    expect(await skillsCheck([good], { json: false }, io)).toBe(0);
    expect(stdout()).toBe(`${good}: ok (good)\n`);
    expect(stderr()).toBe("");

    const missing = join(tmp, "skills", "missing");
    const failed = testIo();
    expect(await skillsCheck([good, missing], { json: false }, failed.io)).toBe(1);
    expect(failed.stdout()).toBe(`${good}: ok (good)\n`);
    expect(failed.stderr()).toBe(`${missing}: missing SKILL.md\n`);
  });

  test("json: a missing and a bad SKILL.md are skill.invalid, each path as given", async () => {
    const good = skillDir("good", GOOD_SKILL);
    skillDir("bad", "---\nname: bad\n---\nNo description.\n");
    const bad = join(tmp, "skills", "..", "skills", "bad") + "/";
    const empty = join(tmp, "skills", "empty");
    mkdirSync(empty);
    const { io, stdout, stderr } = testIo();
    expect(await skillsCheck([good, bad, empty], { json: true }, io)).toBe(1);
    expect(stderr()).toBe("");
    const document = JSON.parse(stdout());
    expect(document.format).toBe("yian.cli.v1");
    expect(document.skills[0]).toEqual({ path: good, ok: true, name: "good", description: "Does good things." });
    expect(document.skills.map((s: { path: string }) => s.path)).toEqual([good, bad, empty]);
    for (const result of document.skills.slice(1)) {
      expect(result.ok).toBe(false);
      expect(result.error.code).toBe("skill.invalid");
      expect(typeof result.error.message).toBe("string");
    }
  });
});

describe("main", () => {
  const BAD_DEFAULTS = { YIAN_MAX_TOTAL_NODES: "abc" };

  test("a bad YIAN_ default: the error on stderr, a document only under --json, exit 1", async () => {
    const flow = join(tmp, "flow.yaml");
    writeFileSync(flow, "id: f\n");
    const text = testIo(BAD_DEFAULTS);
    expect(await main(["check", flow], text.io)).toBe(1);
    expect(text.stdout()).toBe("");
    expect(text.stderr()).toContain("YIAN_MAX_TOTAL_NODES");

    const json = testIo(BAD_DEFAULTS);
    expect(await main(["check", "--json", flow], json.io)).toBe(1);
    expect(json.stderr()).toBe(text.stderr());
    expect(JSON.parse(json.stdout())).toEqual({
      format: "yian.cli.v1",
      error: { code: "defaults.invalid", message: text.stderr().trimEnd() },
    });
  });

  test("an unknown command or option exits 2 with nothing on stdout", async () => {
    for (const [argv, message] of [
      [["nope"], "No such command 'nope'."],
      [["--nope"], "No such option: --nope"],
      [["skills", "nope"], "No such command 'nope'."],
      [["skills", "check", "--nope", "x"], "No such option: --nope"],
    ] as const) {
      const { io, stdout, stderr } = testIo();
      expect(await main(argv, io)).toBe(2);
      expect(stdout()).toBe("");
      expect(stderr()).toContain(`Error: ${message}\n`);
    }
  });

  test("an unknown command is a usage error before the defaults are checked", async () => {
    const { io, stderr } = testIo(BAD_DEFAULTS);
    expect(await main(["nope"], io)).toBe(2);
    expect(stderr()).toContain("No such command 'nope'.");
  });

  test("cli rejects negative workers", async () => {
    const flow = join(tmp, "flow.yaml");
    writeFileSync(flow, "id: f\n");
    const { io, stdout, stderr } = testIo();
    expect(await main(["run", "-w", "-1", flow], io)).toBe(2);
    expect(stdout()).toBe("");
    expect(stderr()).toStartWith("Usage: yian run [OPTIONS] FLOW\nTry 'yian run --help' for help.\n");
  });

  test("a missing flow, resume log or --root directory exits 2", async () => {
    const flow = join(tmp, "flow.yaml");
    writeFileSync(flow, "id: f\n");
    const missing = join(tmp, "missing.yaml");
    const log = join(tmp, "missing.jsonl");
    const nodir = join(tmp, "nodir");
    for (const [argv, message] of [
      [["run", missing], `Invalid value for 'FLOW': File '${missing}' does not exist.`],
      [["run", flow, "--resume", log], `Invalid value for '--resume': File '${log}' does not exist.`],
      [["check", "--root", nodir, "x.yaml"], `Invalid value for '--root': Directory '${nodir}' does not exist.`],
    ] as const) {
      const { io, stdout, stderr } = testIo();
      expect(await main(argv, io)).toBe(2);
      expect(stdout()).toBe("");
      expect(stderr()).toEndWith(`Error: ${message}\n`);
    }
  });

  test("the provider commands reach cli/auth.ts", async () => {
    let { io, stdout, stderr } = testIo();
    expect(await main(["providers"], io)).toBe(0);
    expect(stdout().split("\n")[0]).toBe("openai: api-key, subscription (OPENAI_API_KEY)");
    expect(stderr()).toBe("");
    ({ io, stdout, stderr } = testIo());
    expect(await main(["auth", "--json"], io)).toBe(2);
    expect(stdout()).toBe("");
    expect(stderr()).toContain(
      "Error: Invalid value: --json applies to --remove and to openai --method subscription --login device\n",
    );
    // `--timeout` is a float option: Click's message, before the handler runs.
    ({ io, stdout, stderr } = testIo());
    expect(await main(["models", "--provider", "openai", "--timeout", "soon"], io)).toBe(2);
    expect(stderr()).toContain("Error: Invalid value for '--timeout': 'soon' is not a valid float.\n");
  });

  test("a command's --help prints its help and exits 0", async () => {
    const { io, stdout } = testIo();
    expect(await main(["skills", "check", "--help"], io)).toBe(0);
    expect(stdout()).toStartWith("Usage: yian skills check [OPTIONS] DIRS...\n");
  });
});

// Bytes that aren't UTF-8: a flow holding them can't be read.
const NOT_UTF8 = new Uint8Array([0x69, 0x64, 0x3a, 0x20, 0xff, 0x0a]);

const FLOW = "id: f\nname: f\ninput:\n  n: !int\nnodes: {}\noutput: ${input.n}\n";

/** Write `content` at `tmp/<name>`, creating its directory, and return its path. */
function file(name: string, content: string | Uint8Array): string {
  const path = join(tmp, name);
  mkdirSync(join(path, ".."), { recursive: true });
  writeFileSync(path, content);
  return path;
}

describe("run", () => {
  test("--resume with --input exits 2 naming the conflict, before the flow is read", async () => {
    const flow = file("flow.yaml", NOT_UTF8);
    const old = file("old.jsonl", "");
    const { io, stdout, stderr } = testIo();
    expect(await main(["run", flow, "--resume", old, "--input", "n=1"], io)).toBe(2);
    expect(stdout()).toBe("");
    expect(stderr()).toEndWith("Error: Invalid value: --resume cannot be combined with --input\n");
  });

  test("an existing --log exits 2, before the flow is read", async () => {
    const flow = file("flow.yaml", NOT_UTF8);
    const log = file("taken.jsonl", "keep");
    const { io, stdout, stderr } = testIo();
    expect(await main(["run", flow, "--log", log], io)).toBe(2);
    expect(stdout()).toBe("");
    expect(stderr()).toEndWith(`Error: Invalid value for --log: --log path already exists: ${log}\n`);
  });

  test("a flow that isn't UTF-8 exits 1, with a compose.load document under --json", async () => {
    const flow = file("flow.yaml", NOT_UTF8);
    const message = `can't read ${flow}: not UTF-8 text`;
    const text = testIo();
    expect(await main(["run", flow], text.io)).toBe(1);
    expect(text.stdout()).toBe("");
    expect(text.stderr()).toBe(`${message}\n`);

    const json = testIo();
    expect(await main(["run", flow, "--json"], json.io)).toBe(1);
    expect(json.stderr()).toBe(`${message}\n`);
    expect(JSON.parse(json.stdout())).toEqual({
      format: "yian.cli.v1",
      error: { code: "compose.load", message },
    });
  });

});

/** A flow with one `topic` input whose CODE node `body` makes the output's `topic`. */
function topicFlow(body: string): string {
  return (
    "id: cli_echo\nname: cli_echo\ninput:\n  topic: !str\nnodes:\n  echo:\n    kind: code\n" +
    `    input:\n      topic: \${input.topic}\n    output: str\n    code: ${body}\n` +
    "output:\n  topic: ${echo.output}\n"
  );
}

const ECHO_FLOW = topicFlow("return inputs.topic");
const INLINE_FLOW = topicFlow("|\n      return inputs.topic.toUpperCase();");
const FAIL_FLOW = topicFlow('throw new Error("always fails")');

const OMITTABLE_INPUTS_FLOW = `id: cli_omittable_inputs
name: cli_omittable_inputs
input:
  topic: !str
  limit: !int 7
  labels: !list(str) [alpha, beta]
  note: !Optional(str)
nodes:
  show:
    kind: code
    input:
      topic: \${input.topic}
      limit: \${input.limit}
      labels: \${input.labels}
      note: \${input.note}
    output: object
    code: return inputs
output: \${show.output}
`;

const NULLABLE_STRUCTURED_INPUT_FLOW = `id: cli_nullable_structured_input
name: cli_nullable_structured_input
input:
  items: !Optional(list(int))
nodes:
  inspect:
    kind: code
    input:
      items: \${input.items}
    output: object
    code: 'return { items: inputs.items, is_none: inputs.items === null }'
output: \${inspect.output}
`;

const REGISTERED_TOOL_FLOW =
  "id: cli_registered_tool\nname: cli_registered_tool\nnodes:\n  call:\n    kind: tool\n" +
  "    tool_id: cli_run_probe\noutput: ${call.output}\n";

/** Run `yian run flow.yaml argv` on a flow holding `body`, its default log under `tmp/home`. */
async function runFlowCli(body: string, argv: readonly string[]) {
  const flow = file("flow.yaml", body);
  mkdirSync(join(tmp, "home"), { recursive: true });
  const { io, stdout, stderr } = testIo({ HOME: join(tmp, "home") });
  const code = await main(["run", flow, ...argv], io);
  return { code, stdout: stdout(), stderr: stderr() };
}

describe("run outcomes", () => {
  test("run succeeds with input flag", async () => {
    const { code, stdout, stderr } = await runFlowCli(ECHO_FLOW, ["--input", "topic=clouds"]);
    expect(code).toBe(0);
    expect(stdout).toBe('{"topic":"clouds"}\n');
    expect(stderr).toContain("✓ echo");
  });

  test("a string output prints as is", async () => {
    const flow = ECHO_FLOW.replace("output:\n  topic: ${echo.output}", "output: ${echo.output}");
    const { code, stdout } = await runFlowCli(flow, ["-q", "--input", "topic=a *b*"]);
    expect(code).toBe(0);
    expect(stdout).toBe("a *b*\n");
  });

  test("run keeps registered tools", async () => {
    const probe = new FunctionTool({
      name: "cli_run_probe",
      description: "Probe.",
      parametersJsonSchema: { type: "object", properties: {} },
      execute: () => "from the registry",
    });
    registerTool("cli_run_probe", probe);
    try {
      const { code, stdout } = await runFlowCli(REGISTERED_TOOL_FLOW, ["-q"]);
      expect(code).toBe(0);
      expect(stdout).toBe("from the registry\n");
    } finally {
      TOOL_REGISTRY.delete("cli_run_probe");
    }
  });

  test("run inline code in process", async () => {
    const { code, stdout } = await runFlowCli(INLINE_FLOW, ["--input", "topic=clouds"]);
    expect(code).toBe(0);
    expect(stdout).toContain("CLOUDS");
  });

  test("run reads inputs json, a flag overriding it", async () => {
    const inputs = inputsFile('{"topic": "rivers"}');
    let { code, stdout } = await runFlowCli(ECHO_FLOW, ["--inputs", inputs]);
    expect(code).toBe(0);
    expect(stdout).toContain("rivers");
    ({ code, stdout } = await runFlowCli(ECHO_FLOW, ["--inputs", inputs, "--input", "topic=mountains"]));
    expect(code).toBe(0);
    expect(stdout).toContain("mountains");
    expect(stdout).not.toContain("rivers");
  });

  test("failed run exits nonzero, the located failure on stderr", async () => {
    const { code, stdout, stderr } = await runFlowCli(FAIL_FLOW, ["-q", "--input", "topic=clouds"]);
    expect(code).toBe(1);
    expect(stdout).toBe("");
    expect(stderr).toContain("flow.yaml:");
    expect(stderr).toContain("always fails");
    expect(stderr).not.toContain("─ traceback ");
  });

  test("failed run under --engine-trace adds the captured traceback", async () => {
    const { code, stderr } = await runFlowCli(FAIL_FLOW, ["-q", "--engine-trace", "--input", "topic=clouds"]);
    expect(code).toBe(1);
    expect(stderr).toContain("always fails");
    expect(stderr).toContain("─ traceback ");
  });

  test("non-TTY run omits optional and defaulted inputs", async () => {
    const log = join(tmp, "run.jsonl");
    const { code, stdout, stderr } = await runFlowCli(OMITTABLE_INPUTS_FLOW, [
      "--input",
      "topic=clouds",
      "--log",
      log,
      "--quiet",
    ]);
    expect(code).toBe(0);
    expect(existsSync(log)).toBe(true);
    expect(JSON.parse(stdout)).toEqual({ topic: "clouds", limit: 7, labels: ["alpha", "beta"], note: null });
    // Quiet: only the run log's path is on stderr.
    expect(stderr).toBe(`Run log: ${log}\n`);
  });

  test("non-TTY inputs json preserves explicit structured null", async () => {
    const inputs = inputsFile('{"items": null}');
    const log = join(tmp, "run.jsonl");
    const { code, stdout } = await runFlowCli(NULLABLE_STRUCTURED_INPUT_FLOW, [
      "--inputs",
      inputs,
      "--log",
      log,
      "--quiet",
    ]);
    expect(code).toBe(0);
    expect(JSON.parse(stdout)).toEqual({ items: null, is_none: true });
  });

  test("the banner precedes progress, and --quiet drops both", async () => {
    const flow = ECHO_FLOW.replace("name: cli_echo\n", "name: cli_echo\ndescription: Echoes.\n");
    const loud = await runFlowCli(flow, ["--input", "topic=x"]);
    expect(loud.stderr).toStartWith("╭");
    expect(loud.stderr).toContain("Running flow: cli_echo");
    expect(loud.stderr.indexOf("Running flow")).toBeLessThan(loud.stderr.indexOf("✓ echo"));
    const quiet = await runFlowCli(flow, ["-q", "--input", "topic=x"]);
    expect(quiet.stderr).toStartWith("Run log: ");
    expect(quiet.stderr).not.toContain("Running flow");
    expect(quiet.stderr).not.toContain("✓");
  });
});

const REQUIRED_INPUTS_FLOW = `id: cli_required_inputs
name: cli_required_inputs
input:
  first: !str
  optional: !Optional(str)
  second: !int
nodes:
  show:
    kind: code
    input:
      first: \${input.first}
      optional: \${input.optional}
      second: \${input.second}
    output: object
    code: return inputs
output: \${show.output}
`;

/** A flow declaring `inputs` (YAML lines under `input:`) whose output is `null`. */
function inputsFlow(inputs: string): string {
  return `id: gate\nname: gate\ninput:\n${inputs}nodes: {}\noutput: null\n`;
}

/** The input names the run at `log` was given, from its `RunStarted` record. */
function suppliedNames(log: string): string[] {
  const record = readRunLog(log)[0]!.record;
  if (!(record instanceof RunStartedRecord)) throw new Error("the log doesn't start with RunStarted");
  return Object.keys(record.startWiringOverride);
}

const INPUT_HINT = "Invalid value for '--input' / '--inputs'";

describe("run's input gate", () => {
  test("undeclared input flag is a bad parameter", async () => {
    const argv = ["--input", "topic=clouds", "--input", "topik=x"];
    const { code, stdout, stderr } = await runFlowCli(ECHO_FLOW, argv);
    expect(code).toBe(2);
    expect(stdout).toBe("");
    expect(stderr).toEndWith(`Error: ${INPUT_HINT}: unknown input "topik"; declared inputs: topic\n`);
  });

  test("undeclared inputs json key is a bad parameter", async () => {
    const inputs = inputsFile('{"topic": "clouds", "topik": "x"}');
    const { code, stderr } = await runFlowCli(ECHO_FLOW, ["--inputs", inputs]);
    expect(code).toBe(2);
    expect(stderr).toContain('unknown input "topik"');
  });

  test("an unknown name fails before a missing required input is reported", async () => {
    const { code, stderr } = await runFlowCli(REQUIRED_INPUTS_FLOW, ["-q", "--input", "nope=1"]);
    expect(code).toBe(2);
    expect(stderr).toContain('unknown input "nope"');
    expect(stderr).not.toContain("missing required");
  });

  test("gather non-TTY leaves optional and defaulted inputs absent", async () => {
    const log = join(tmp, "run.jsonl");
    const flow = inputsFlow("  optional: !Optional(str)\n  defaulted: !int 7\n");
    const { code } = await runFlowCli(flow, ["-q", "--log", log]);
    expect(code).toBe(0);
    expect(suppliedNames(log)).toEqual([]);
  });

  test("gather non-TTY reports all missing required inputs in order", async () => {
    const { code, stdout, stderr } = await runFlowCli(REQUIRED_INPUTS_FLOW, ["-q"]);
    expect(code).toBe(2);
    expect(stdout).toBe("");
    expect(stderr).toEndWith(`Error: ${INPUT_HINT}: missing required flow inputs: first, second\n`);
  });

  test("gather non-TTY filters required inputs by exact key membership", async () => {
    const declared = "  none: !Optional(str)\n  flag: !bool\n  zero: !int\n  blank: !str\n  missing: !int\n";
    const flow = inputsFlow(declared);
    const inputs = inputsFile('{"none": null, "flag": false, "zero": 0, "blank": ""}');
    const { code, stderr } = await runFlowCli(flow, ["-q", "--inputs", inputs]);
    expect(code).toBe(2);
    expect(stderr).toEndWith(`Error: ${INPUT_HINT}: missing required flow input: missing\n`);
  });

  test("non-TTY run reports all missing required inputs before setup", async () => {
    const { code, stderr } = await runFlowCli(REQUIRED_INPUTS_FLOW, ["-q"]);
    expect(code).toBe(2);
    const error = stderr.slice(stderr.indexOf("Error: "));
    expect(error.indexOf("first")).toBeLessThan(error.indexOf("second"));
    expect(error).not.toContain("optional");
    expect(stderr).not.toContain("Run log:");
    // The default log path was never made.
    expect(existsSync(join(tmp, "home", ".yian"))).toBe(false);
  });
});

/** Run `yian run flow.yaml -q argv` on a log at `tmp/run.jsonl`, stdin a fake TTY given `keys`. */
async function runOnTty(body: string, argv: readonly string[], keys: (input: FakeInput) => void) {
  const flow = file("flow.yaml", body);
  const { io, stdout, stderr } = testIo();
  const input = new FakeInput();
  const output = new FakeOutput();
  // Typed ahead: the terminal reads them once it prompts.
  keys(input);
  const code = await main(["run", flow, "-q", "--log", join(tmp, "run.jsonl"), ...argv], {
    ...io,
    stdin: input,
    stderrStream: output,
  });
  return { code, stdout: stdout(), stderr: stderr(), drawn: output.text, input };
}

const PROMPT_DEFAULTS_FLOW = `id: prompt-defaults
name: prompt-defaults
typedefs:
  Mode: Literal(z, a)
input:
  items: !list(str) [a, b]
  mode: !Mode z
  flag: !Optional(bool)
nodes:
  show:
    kind: code
    code: return inputs
    input:
      items: \${input.items}
      mode: \${input.mode}
      flag: \${input.flag}
    output: object
output: \${show.output}
`;

describe("run's input prompts on a TTY", () => {
  test("accepting each omission leaves START's defaults", async () => {
    const enter = (keys: FakeInput) => keys.write("\r\r\r");
    const { code, stdout, drawn, input } = await runOnTty(PROMPT_DEFAULTS_FLOW, [], enter);
    expect(code).toBe(0);
    for (const label of ["items (list(str)) [default: ", "mode (Literal(a, z))", "flag (Optional(bool))"]) {
      expect(drawn).toContain(label);
    }
    expect(input.rawCalls).toEqual([true, false]);
    expect(JSON.parse(stdout)).toEqual({ flag: null, items: ["a", "b"], mode: "z" });
    expect(suppliedNames(join(tmp, "run.jsonl"))).toEqual([]);
  });

  test("a cancelled prompt is run cancelled, run.cancelled under --json", async () => {
    const stops: [string, (keys: FakeInput) => void][] = [
      ["Ctrl-C", (keys) => keys.write("\x03")],
      ["the end of input", (keys) => keys.end()],
    ];
    for (const [name, stop] of stops) {
      const { code, stdout, stderr } = await runOnTty(REQUIRED_INPUTS_FLOW, ["--json"], stop);
      expect(code, name).toBe(1);
      expect(stderr, name).toBe("run cancelled\n");
      expect(JSON.parse(stdout), name).toEqual({
        format: "yian.cli.v1",
        error: { code: "run.cancelled", message: "run cancelled" },
      });
    }
  });
});

const ASK_FLOW =
  "id: ask\nname: ask\nnodes:\n  ok:\n    kind: human_input\n" +
  '    prompt: "Proceed?"\n    output: str\noutput: ${ok.output}\n';

// A WAIT that never ends on its own, after a node whose progress line says the run is going.
const PAUSE_FLOW =
  "id: pause\nname: pause\nnodes:\n  first: {kind: code, output: str, code: return 'x'}\n" +
  '  settle: {kind: wait, until: "2999-01-01", depends_on: [first]}\noutput: ${first.output}\n';

describe("run's stops", () => {
  test("Ctrl-C at a prompt during the run: cancelled by user, exit 130, no document", async () => {
    const ctrlC = (keys: FakeInput) => keys.write("\x03");
    const { code, stdout, stderr, input } = await runOnTty(ASK_FLOW, ["--json"], ctrlC);
    expect(code).toBe(130);
    expect(stdout).toBe("");
    expect(stderr.endsWith("cancelled by user\n")).toBe(true);
    expect(input.rawCalls).toEqual([true, false]);
  });

  test("an already aborted io.stop stops the run before its log exists", async () => {
    const flow = file("flow.yaml", ECHO_FLOW);
    const log = join(tmp, "run.jsonl");
    const { io, stdout, stderr } = testIo();
    const stop = new AbortController();
    stop.abort(new HostStop("SIGTERM"));
    const argv = ["run", flow, "--json", "-q", "--log", log, "--input", "topic=clouds"];
    expect(await main(argv, { ...io, stop: stop.signal })).toBe(130);
    expect(stdout()).toBe("");
    expect(stderr()).toBe("cancelled by user\n");
    expect(existsSync(log)).toBe(false);
  });

  test("io.stop aborting mid-run stops it; its and the signals' listeners are gone", async () => {
    const flow = file("flow.yaml", PAUSE_FLOW);
    const log = join(tmp, "run.jsonl");
    const stop = new AbortController();
    const chunks: string[] = [];
    const stderr: Writer = {
      write: async (text) => {
        chunks.push(text);
        if (text.includes("✓ first")) stop.abort(new HostStop("SIGTERM"));
      },
    };
    const signalListeners = () =>
      ["SIGINT", "SIGTERM", "SIGHUP"].map((name) => process.listenerCount(name));
    const before = signalListeners();
    const io: CliIo = { ...testIo().io, stderr, stop: stop.signal };
    expect(await main(["run", flow, "--log", log], io)).toBe(130);
    expect(chunks.at(-1)).toBe("cancelled by user\n");
    expect(existsSync(log)).toBe(true);
    expect(getEventListeners(stop.signal, "abort")).toHaveLength(0);
    expect(signalListeners()).toEqual(before);
  });
});

// Two nodes that start together, so a TTY shows two spinner lines.
const PARALLEL_FLOW =
  "id: par\nname: par\nnodes:\n  a: {kind: code, output: str, code: return 'a'}\n" +
  "  b: {kind: code, output: str, code: return 'b'}\noutput:\n  a: ${a.output}\n  b: ${b.output}\n";

describe("run setup errors", () => {
  test("on a TTY the run failed: line comes after the spinners are erased", async () => {
    const flow = file("flow.yaml", PARALLEL_FLOW);
    const chunks: string[] = [];
    let broke = false;
    // A stderr whose spinner draw for `b` throws: the reporter's error rejects the run while
    // spinner lines are on screen.
    const stderr: Writer = {
      isTTY: true,
      write: (text) => {
        if (!broke && text.includes("⠋ b")) {
          broke = true;
          throw new TypeError("stderr broke");
        }
        chunks.push(text);
        return Promise.resolve();
      },
    };
    const io: CliIo = {
      stdout: sink().out,
      stderr,
      stdin: new PassThrough(),
      stderrStream: new PassThrough(),
      env: (name) => (name === "NO_COLOR" ? "1" : undefined),
    };
    expect(await main(["run", flow, "-q", "--log", join(tmp, "run.jsonl")], io)).toBe(0);
    chunks.length = 0;
    broke = false;
    expect(await main(["run", flow, "--log", join(tmp, "run2.jsonl")], io)).toBe(1);
    const last = chunks.at(-1)!;
    expect(last).toBe("run failed: stderr broke\n");
    expect(chunks.slice(0, -1).some((chunk) => chunk.startsWith("\r\x1b[K"))).toBe(true);
  });

  test("--engine-trace prints the stack box after run failed:", async () => {
    // A log whose directory is a file: the run can't create it.
    const log = join(file("taken", ""), "run.jsonl");
    const argv = ["-q", "--engine-trace", "--log", log, "--input", "topic=clouds"];
    const { code, stderr } = await runFlowCli(ECHO_FLOW, argv);
    expect(code).toBe(1);
    const failed = stderr.indexOf("run failed: ");
    expect(failed).toBeGreaterThan(-1);
    expect(stderr.indexOf("─ traceback ")).toBeGreaterThan(failed);
    expect(stderr).toContain("ENOTDIR");
    expect(stderr).toContain("    at ");
  });
});

describe("run --json", () => {
  test("succeeded: the output in the codec form, only the document on stdout", async () => {
    const { code, stdout } = await runFlowCli(ECHO_FLOW, ["--json", "--input", "topic=clouds"]);
    expect(code).toBe(0);
    expect(JSON.parse(stdout)).toEqual({ format: "yian.cli.v1", status: "succeeded", output: { topic: "clouds" } });
  });

  test("failed: the diagnostic and the failed node", async () => {
    const { code, stdout } = await runFlowCli(FAIL_FLOW, ["--json", "--input", "topic=clouds"]);
    expect(code).toBe(1);
    const document = JSON.parse(stdout);
    expect(document.status).toBe("failed");
    expect(document.failure.node).toBe("echo");
    expect(document.failure.message).toContain("always fails");
    expect(document.failure.frames[0].label).toBe("flow.yaml");
  });

  test("a load error: the located diagnostic on stderr, compose.load on stdout", async () => {
    const { code, stdout, stderr } = await runFlowCli(ECHO_FLOW.replace("${echo.output}", "${ech.output}"), [
      "--json",
    ]);
    expect(code).toBe(1);
    expect(stderr).toContain("flow.yaml:");
    expect(JSON.parse(stdout).error.code).toBe("compose.load");
  });

  test("an unknown input is a usage error: exit 2, nothing on stdout", async () => {
    const argv = ["--json", "--input", "topic=clouds", "--input", "topik=x"];
    const { code, stdout, stderr } = await runFlowCli(ECHO_FLOW, argv);
    expect(code).toBe(2);
    expect(stdout).toBe("");
    expect(stderr).toContain("unknown input");
  });
});

/** A flow named `main` whose one node calls the `uses:` alias `child`, bound to `ref`. */
function parentFlow(ref: string): string {
  return (
    "id: main\nname: main\ninput:\n  n: !int\nnodes:\n  work:\n    kind: call\n    call: child\n" +
    `    input:\n      n: \${input.n}\noutput: \${work.output}\nuses:\n  child: ${ref}\n`
  );
}

/** Run `yian check --json --root root flow`, which fails, and return its one diagnostic. */
async function rootDiagnostic(root: string, flow: string) {
  const { code, flows } = await checkJson(["--root", root, flow]);
  expect(code).toBe(1);
  expect(flows[0].diagnostics).toHaveLength(1);
  return flows[0].diagnostics[0];
}

/** Run `yian check --json` on `argv` and return the exit code and the document's `flows`. */
async function checkJson(argv: string[]) {
  const { io, stdout, stderr } = testIo();
  const code = await main(["check", "--json", ...argv], io);
  expect(stderr()).toBe("");
  return { code, flows: JSON.parse(stdout()).flows };
}

describe("check", () => {
  test("a missing or non-UTF-8 file is compose.load, without loading", async () => {
    const missing = join(tmp, "missing.yaml");
    const bad = file("bad.yaml", NOT_UTF8);
    expect(await checkJson([missing, bad])).toEqual({
      code: 1,
      flows: [
        {
          path: missing,
          ok: false,
          diagnostics: [
            { code: "compose.load", message: `can't read ${missing}: No such file or directory` },
          ],
        },
        {
          path: bad,
          ok: false,
          diagnostics: [{ code: "compose.load", message: `can't read ${bad}: not UTF-8 text` }],
        },
      ],
    });
  });

  test("text: the path then the message on stderr", async () => {
    const missing = join(tmp, "missing.yaml");
    const { io, stdout, stderr } = testIo();
    expect(await main(["check", missing], io)).toBe(1);
    expect(stdout()).toBe("");
    expect(stderr()).toBe(`${missing}:\ncan't read ${missing}: No such file or directory\n`);
  });

  test("text: a flow that loads is an ok line on stdout", async () => {
    const flow = file("flow.yaml", FLOW);
    const { io, stdout, stderr } = testIo();
    expect(await main(["check", flow], io)).toBe(0);
    expect(stdout()).toBe(`${flow}: ok\n`);
    expect(stderr()).toBe("");
  });

  test("the ok document: canonical type labels, a typedef expanded", async () => {
    const flow = file(
      "profile.yaml",
      "id: profile\nname: profile\ndescription: Greets a person.\n" +
        "typedefs:\n  Person: {name: str, age: int}\n" +
        "input:\n  who: !Person\n  tag: !Optional(str)\n  w: !int 3\n" +
        "nodes: {}\noutput: ${input.who}\n",
    );
    const bare = file("bare.yaml", "id: bare\nname: bare\nnodes: {}\noutput: 1\n");
    expect(await checkJson([flow, bare])).toEqual({
      code: 0,
      flows: [
        {
          path: flow,
          ok: true,
          id: "profile",
          name: "profile",
          description: "Greets a person.",
          inputs: [
            { name: "who", type: '{"age":int,"name":str}', required: true, has_default: false },
            { name: "tag", type: "Optional(str)", required: false, has_default: false },
            { name: "w", type: "int", required: false, has_default: true },
          ],
        },
        { path: bare, ok: true, id: "bare", name: "bare", description: null, inputs: [] },
      ],
    });
  });

  test("--root: a flow outside the root or setting system.paths fails, labelled from the root", async () => {
    const root = join(tmp, "company");
    const outside = file("other/flow.yaml", FLOW);
    const paths = file("company/main.yaml", `system:\n  paths: [lib]\n${FLOW}`);
    expect(await checkJson(["--root", root, outside, paths])).toEqual({
      code: 1,
      flows: [
        {
          path: outside,
          ok: false,
          diagnostics: [{ code: "compose.outside_root", message: "../other/flow.yaml is outside the root" }],
        },
        {
          path: paths,
          ok: false,
          diagnostics: [
            {
              code: "compose.load",
              message: "system.paths is not supported with --root; use a relative uses: path",
            },
          ],
        },
      ],
    });
  });

  test("--root: invalid YAML is located under its path relative to the root", async () => {
    const root = join(tmp, "company");
    const flow = file("company/sub/main.yaml", "id: [f\n");
    const { code, flows } = await checkJson(["--root", root, flow]);
    expect(code).toBe(1);
    const [diagnostic] = flows[0].diagnostics;
    expect(diagnostic.code).toBe("compose.load");
    expect(diagnostic.message).toStartWith("invalid YAML: ");
    expect(diagnostic.frames[0].label).toBe(join("sub", "main.yaml"));
  });

  test("--root: a missing flow can't be read, a symlink loop stays an OS error", async () => {
    const root = join(tmp, "company");
    mkdirSync(root);
    const missing = join(root, "missing.yaml");
    const loop = join(tmp, "loop.yaml");
    symlinkSync(loop, loop);
    const { code, flows } = await checkJson(["--root", root, missing, loop]);
    expect(code).toBe(1);
    expect(flows[0].diagnostics).toEqual([
      { code: "compose.load", message: `can't read ${missing}: No such file or directory` },
    ]);
    expect(flows[1].diagnostics[0].code).toBe("compose.load");
    expect(flows[1].diagnostics[0].message).toStartWith(`can't read ${loop}: Too many symbolic links`);
  });

  test("--root: a flow and its uses: child load from the importing file's directory", async () => {
    const root = join(tmp, "company");
    const flow = file("company/main.yaml", parentFlow("lib/child"));
    file("company/lib/child.yaml", FLOW);
    expect(await checkJson(["--root", root, flow])).toEqual({
      code: 0,
      flows: [
        {
          path: flow,
          ok: true,
          id: "main",
          name: "main",
          description: null,
          inputs: [{ name: "n", type: "int", required: true, has_default: false }],
        },
      ],
    });
  });

  test("--root: a missing child is not found, at its path relative to the root", async () => {
    const root = join(tmp, "company");
    const flow = file("company/main.yaml", parentFlow("lib/missing"));
    const diagnostic = await rootDiagnostic(root, flow);
    expect(diagnostic.code).toBe("compose.load");
    expect(diagnostic.message).toEndWith(`flow 'lib/missing' not found: ${join("lib", "missing.yaml")}`);
    expect(diagnostic.frames.map((f: { note: string }) => f.note)).toEqual(["uses alias 'child'"]);
  });

  test("--root: a child under a regular file is not found", async () => {
    const root = join(tmp, "company");
    const flow = file("company/main.yaml", parentFlow("lib/child"));
    file("company/lib", "not a directory\n");
    const diagnostic = await rootDiagnostic(root, flow);
    expect(diagnostic.code).toBe("compose.load");
    expect(diagnostic.message).toEndWith(`flow 'lib/child' not found: ${join("lib", "child.yaml")}`);
  });

  test("--root: a child outside the root keeps its outside_root code", async () => {
    const root = join(tmp, "company");
    const flow = file("company/main.yaml", parentFlow("../lib/child"));
    file("lib/child.yaml", FLOW);
    const diagnostic = await rootDiagnostic(root, flow);
    expect(diagnostic.code).toBe("compose.outside_root");
    expect(diagnostic.message).toBe(`${join("..", "lib", "child.yaml")} is outside the root`);
  });

  test("--root: a reference cycle names the flow it reaches again", async () => {
    const root = join(tmp, "company");
    const a = file("company/a.yaml", parentFlow("b").replace("id: main", "id: a"));
    file("company/b.yaml", parentFlow("a").replace("id: main", "id: b"));
    const diagnostic = await rootDiagnostic(root, a);
    expect(diagnostic.code).toBe("compose.load");
    expect(diagnostic.message).toEndWith("flow-reference cycle at a.yaml");
    expect(diagnostic.frames.map((f: { label: string }) => f.label)).toEqual(["a.yaml", "b.yaml"]);
  });

  test("--root: a pinned version the child doesn't have", async () => {
    const root = join(tmp, "company");
    const pinned = file("company/pinned.yaml", parentFlow("lib/child@2"));
    file("company/lib/child.yaml", FLOW.replace("name: f\n", "name: f\nversion: '1'\n"));
    expect((await rootDiagnostic(root, pinned)).message).toEndWith("lib/child is version '1', not '2'");
    const unversioned = file("company/unversioned.yaml", parentFlow("lib/bare@2"));
    file("company/lib/bare.yaml", FLOW);
    expect((await rootDiagnostic(root, unversioned)).message).toEndWith("lib/bare is version null, not '2'");
  });

  test("a CODE reference resolves against its own file's directory, not the cwd", async () => {
    const node = "  a:\n    kind: code\n    code: ./m.ts:f\n    output: str\n";
    const fn = "export function f() { return 'x'; }\n";
    const alone = file("alone/flow.yaml", `id: alone\nname: alone\nnodes:\n${node}output: \${a.output}\n`);
    file("alone/m.ts", fn);
    const root = join(tmp, "company");
    const flow = file(
      "company/sub/main.yaml",
      `id: main\nname: main\nnodes:\n${node}  work:\n    kind: call\n    call: child\n` +
        "output: ${work.output}\nuses:\n  child: lib/child\n",
    );
    file("company/sub/m.ts", fn);
    // The child's own file is only beside it, so resolving it against its importer's directory fails.
    const childNode = node.replace("./m.ts:f", "./n.ts:f");
    file("company/sub/lib/child.yaml", `id: child\nname: child\nnodes:\n${childNode}output: \${a.output}\n`);
    file("company/sub/lib/n.ts", fn);
    for (const argv of [[alone], ["--root", root, flow]]) {
      const { code, flows } = await checkJson(argv);
      expect([code, flows[0].ok]).toEqual([0, true]);
    }
  });
});

describe("status", () => {
  test("a missing log exits 2", async () => {
    const log = join(tmp, "missing.jsonl");
    const { io, stdout, stderr } = testIo();
    expect(await main(["status", log], io)).toBe(2);
    expect(stdout()).toBe("");
    expect(stderr()).toEndWith(`Error: Invalid value for 'LOG': File '${log}' does not exist.\n`);
  });

  test("an existing log reaches the run inspection", async () => {
    const log = file("run.jsonl", "");
    const { io, stdout, stderr } = testIo();
    expect(await main(["status", log], io)).toBe(1);
    expect(stdout()).toBe("");
    expect(stderr()).toContain("inspectRun is not ported to TypeScript yet");
  });
});

// The entry as the conformance runner runs it: `.ts` from source, `.js` when built.
const CLI_ENTRY = fileURLToPath(
  new URL(`../../src/cli/index.${import.meta.url.endsWith(".ts") ? "ts" : "js"}`, import.meta.url),
);

// Guards truncation at exit through the real entry, not `flush` itself, which progress writes need.
test("the entry writes a large --json document whole to piped stdout", () => {
  const dirs = Array.from({ length: 3000 }, (_, i) => `missing-skill-directory-${String(i).padStart(4, "0")}`);
  const runtime = process.versions.bun !== undefined ? ["--no-install"] : [];
  const ran = spawnSync(process.execPath, [...runtime, CLI_ENTRY, "skills", "check", "--json", ...dirs], {
    cwd: tmp,
    env: { HOME: tmp },
    encoding: "utf8",
    maxBuffer: 64 * 1024 * 1024,
    timeout: 30_000,
  });
  expect(ran.status).toBe(1);
  expect(ran.stdout.length).toBeGreaterThan(128 * 1024);
  const document = JSON.parse(ran.stdout);
  expect(document.skills.map((s: { path: string }) => s.path)).toEqual(dirs);
}, 30_000);
