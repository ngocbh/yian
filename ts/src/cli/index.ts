#!/usr/bin/env node
/**
 * The `yian` command line: the command table and `main`, which routes the arguments to a command.
 *
 * `COMMANDS` holds one entry per command: its `spec`, plain data the parser and the help text read,
 * and a `load` that imports the handler's module on demand, so `yian --help` or a usage error loads
 * no command module. A two-word key (`skills check`) is a command in a group (`skills`).
 *
 * Before a command parses its arguments, `main` checks the `YIAN_<NAME>` defaults in the command's
 * environment, as Python's root callback does.
 *
 * Run as the entry script, `yian run ... --json` runs in a child of this runtime whose fd 1 goes to
 * stderr and whose `stdout` writer is fd 3, so stdout holds only the document even when a CODE body
 * or a tool writes to fd 1 (Python points fd 1 at stderr with `os.dup2`, which Node and Bun lack).
 */

import { spawn } from "node:child_process";
import { createWriteStream, writeSync } from "node:fs";
import { constants } from "node:os";

import { DefaultsError, read } from "../defaults.ts";
import { compiled, VERSION, withWarnings } from "../platform.ts";
import {
  type CliIo,
  type CommandSpec,
  abortOnSignal,
  flush,
  helpText,
  HostStop,
  type OptionSpec,
  type ParsedArgs,
  parseCommand,
  printJson,
  processIo,
  relay,
  streamWriter,
  UsageError,
  usageError,
} from "./utils.ts";

/** A command's handler: it takes the parsed command line and returns the exit code. */
type Handler = (command: ParsedArgs, io: CliIo) => Promise<number>;

/** A command: its spec, and a loader importing its handler. */
interface Command {
  spec: CommandSpec;
  load: () => Promise<Handler>;
}

const JSON_HELP = "Print one yian.cli.v1 JSON document on stdout.";
const JSON_OPTION: OptionSpec = { name: "json", type: "boolean", help: JSON_HELP };

/** The commands by name, in the order the root help lists them. */
const COMMANDS: Readonly<Record<string, Command>> = {
  run: {
    spec: {
      name: "run",
      summary: "Start or recover a durable flow run and print its terminal output.",
      positionals: [
        { name: "flow", help: "Path to a flow .yaml", path: { exists: true, dir: false, readable: true } },
      ],
      options: [
        { name: "input", short: "i", type: "string", multiple: true, help: "An input as k=v (repeatable)." },
        {
          name: "inputs",
          type: "string",
          path: { exists: true, dir: false, readable: true },
          help: "A JSON file of inputs.",
        },
        {
          name: "log",
          type: "string",
          path: { dir: false },
          help: "Create a fresh durable run log at PATH instead of the default location.",
        },
        {
          name: "resume",
          type: "string",
          path: { exists: true, dir: false, readable: true, writable: true },
          help: "Recover the run recorded at PATH; fresh input, provider/model, and log flags are forbidden.",
        },
        { name: "quiet", short: "q", type: "boolean", help: "Suppress per-node progress." },
        {
          name: "memory-root",
          type: "string",
          path: { dir: true },
          metavar: "DIR",
          help:
            "Keep every declared built-in memory under DIR/<flow id>/ (created on first use). Also " +
            "accepted with --resume. Without either flag, memory lives in process RAM.",
        },
        {
          name: "memory-storage",
          type: "string",
          multiple: true,
          metavar: "KEY=PATH",
          help:
            "Store the built-in memory KEY (for example notes or lib.notes) at PATH, whose directory " +
            "must exist (repeatable; overrides --memory-root for KEY). Also accepted with --resume.",
        },
        { name: "verbose", short: "v", type: "boolean", help: "Also print each node's output as it finishes." },
        {
          name: "num-workers",
          short: "w",
          type: "int",
          min: 0,
          help:
            "Worker pool size. Omitted means 8 for a fresh run and preserves the recorded value on " +
            "resume. Use 0 for serial execution.",
        },
        {
          name: "provider",
          type: "string",
          help: "Override the LLM provider for agents that set none (cascade).",
        },
        { name: "model", type: "string", help: "Override the LLM model for agents that set none (cascade)." },
        {
          name: "engine-trace",
          type: "boolean",
          help:
            "On a compile, run-log, or runtime node failure, also print the engine traceback; by " +
            "default only the concise or located error is shown.",
        },
        {
          name: "mcp-config",
          type: "string",
          multiple: true,
          path: { exists: true, dir: false, readable: true },
          metavar: "FILE",
          help:
            "Read MCP servers from FILE (mcpServers JSON; repeatable). ~/.yian/mcp.json is read " +
            "first when it exists.",
        },
        {
          name: "json",
          type: "boolean",
          help: "Print one yian.cli.v1 JSON document on stdout: the run's result, or the error that stopped it.",
        },
      ],
    },
    load: async () => {
      const { run } = await import("./run.ts");
      return ({ args, options: o }, io) =>
        run(
          args["flow"] as string,
          {
            inputValues: o["input"] as string[],
            inputs: o["inputs"] as string | undefined,
            log: o["log"] as string | undefined,
            resume: o["resume"] as string | undefined,
            quiet: o["quiet"] === true,
            memoryRoot: o["memory-root"] as string | undefined,
            memoryStorage: o["memory-storage"] as string[],
            verbose: o["verbose"] === true,
            numWorkers: o["num-workers"] as number | undefined,
            provider: o["provider"] as string | undefined,
            model: o["model"] as string | undefined,
            engineTrace: o["engine-trace"] === true,
            mcpConfig: o["mcp-config"] as string[],
            json: o["json"] === true,
          },
          io,
        );
    },
  },
  auth: {
    spec: {
      name: "auth",
      summary: "List, set up, choose the default among, or remove saved provider logins.",
      positionals: [
        {
          name: "provider",
          help: "Provider to set up, make the default, or remove; omit to list logins.",
          optional: true,
        },
      ],
      options: [
        { name: "method", type: "string", help: "OpenAI only: subscription or api-key; prompts when omitted." },
        { name: "login", type: "string", help: "OpenAI subscription sign-in: browser or device (for SSH)." },
        {
          name: "model",
          type: "string",
          help: "Model to save (azure: the deployment name); prompts when omitted.",
        },
        {
          name: "api-key-env",
          type: "string",
          help:
            "Environment variable holding the API key; defaults to the provider's own, such as " +
            "ANTHROPIC_API_KEY.",
        },
        {
          name: "endpoint",
          type: "string",
          help: "Azure resource endpoint, such as https://NAME.openai.azure.com/.",
        },
        {
          name: "organization",
          type: "string",
          help: "OpenAI API key only: organization id sent with each request.",
        },
        { name: "project", type: "string", help: "OpenAI API key only: project id sent with each request." },
        {
          name: "header",
          type: "string",
          multiple: true,
          help:
            "Extra request header, such as a gateway token, saved with the key; repeatable. " +
            "NAME=VARIABLE reads the value from that environment variable, NAME alone from a hidden prompt.",
        },
        { name: "default", type: "boolean", help: "Make an already-saved provider the default." },
        {
          name: "remove",
          type: "boolean",
          help: "Delete the provider's saved login; without a provider, delete the whole --auth-file.",
        },
        {
          name: "auth-file",
          type: "string",
          help: "Auth file to use instead of $YIAN_AUTH_FILE or ~/.config/yian/auth.json.",
        },
        {
          name: "json",
          type: "boolean",
          help:
            "Print yian.cli.v1 JSON on stdout; applies to --remove and to openai --method " +
            "subscription --login device.",
        },
      ],
    },
    load: async () => {
      const { auth } = await import("./auth.ts");
      return auth;
    },
  },
  check: {
    spec: {
      name: "check",
      summary: "Load each flow without running it and report whether it loads.",
      // No path check: a missing flow is a `compose.load` result, not a usage error.
      positionals: [{ name: "flows", help: "Paths to flow .yaml files.", variadic: true }],
      options: [
        JSON_OPTION,
        {
          name: "root",
          type: "string",
          path: { exists: true, dir: true },
          help: "Load each flow and its uses: children only from inside this directory.",
        },
      ],
    },
    load: async () => {
      const { check } = await import("./run.ts");
      return ({ args, options }, io) =>
        check(
          args["flows"] as string[],
          { root: options["root"] as string | undefined, json: options["json"] === true },
          io,
        );
    },
  },
  status: {
    spec: {
      name: "status",
      summary: "Report the run recorded in a run log.",
      positionals: [
        {
          name: "log",
          help: "Path to a run log, such as ~/.yian/runs/<run_id>/log.jsonl",
          path: { exists: true, dir: false },
        },
      ],
      options: [JSON_OPTION],
    },
    load: async () => {
      const { status } = await import("./run.ts");
      return ({ args, options }, io) =>
        status(args["log"] as string, { json: options["json"] === true }, io);
    },
  },
  providers: {
    spec: {
      name: "providers",
      summary: "List the supported providers, the login kinds each takes, and its API-key variable.",
      positionals: [],
      options: [JSON_OPTION],
    },
    load: async () => {
      const { providers } = await import("./auth.ts");
      return providers;
    },
  },
  models: {
    spec: {
      name: "models",
      summary: "List the models an openai subscription login or a provider's key offers.",
      positionals: [],
      options: [
        { name: "auth-file", type: "string", help: "Auth file whose openai subscription login is listed." },
        { name: "provider", type: "string", help: "Provider whose host is listed, with the key on stdin." },
        { name: "base-url", type: "string", help: "Host to list instead of the provider's own; --provider only." },
        { name: "timeout", type: "float", metavar: "FLOAT", help: "Seconds for the whole listing." },
        JSON_OPTION,
      ],
    },
    load: async () => {
      const { models } = await import("./auth.ts");
      return models;
    },
  },
  "config check": {
    spec: {
      name: "config check",
      summary: "Check the host llm_config layer given as a JSON object on stdin, key included.",
      positionals: [],
      options: [JSON_OPTION],
    },
    load: async () => {
      const { configCheck } = await import("./auth.ts");
      return configCheck;
    },
  },
  "skills check": {
    spec: {
      name: "skills check",
      summary: "Load each skill directory and report whether it holds a valid SKILL.md.",
      positionals: [{ name: "dirs", help: "Paths to skill directories.", variadic: true }],
      options: [JSON_OPTION],
    },
    load: async () => {
      const { skillsCheck } = await import("./run.ts");
      return ({ args, options }, io) => skillsCheck(args["dirs"] as string[], { json: options["json"] === true }, io);
    },
  },
};

/** The groups by name, each with its one-line summary. */
const GROUPS: Readonly<Record<string, string>> = {
  config: "Check an llm_config layer.",
  skills: "Check skill directories.",
};

const APP_HELP = "Yian - The Agent Composer. Run agent flows from the command line.";
const HELP_HELP = "Show this message and exit.";
const LICENSES_HELP = "Show the licenses of yian and of what its binaries bundle, and exit.";
const VERSION_HELP = "Show the version and exit.";

/** Return the `Usage:` line of the root (`group` undefined) or of `group`. */
function groupUsage(group?: string): string {
  return `Usage: yian ${group === undefined ? "" : `${group} `}[OPTIONS] COMMAND [ARGS]...`;
}

/**
 * Return the help text of the root (`group` undefined) or of `group`: usage, summary, options and
 * commands.
 */
function groupHelp(group?: string): string {
  const prefix = group === undefined ? "" : `${group} `;
  const rows = new Map<string, string>();
  for (const [key, { spec }] of Object.entries(COMMANDS)) {
    if (!key.startsWith(prefix)) continue;
    const [word, sub] = key.slice(prefix.length).split(" ") as [string, string?];
    if (!rows.has(word)) rows.set(word, sub === undefined ? spec.summary : GROUPS[word]!);
  }
  const width = Math.max(...[...rows.keys()].map((word) => word.length));
  const commands = [...rows].map(([word, summary]) => `  ${word.padEnd(width)}  ${summary}`).join("\n");
  const summary = group === undefined ? APP_HELP : GROUPS[group];
  // Only the root takes `--licenses` and `--version`.
  const options =
    group === undefined
      ? `  --licenses  ${LICENSES_HELP}\n  --version   ${VERSION_HELP}\n  --help      ${HELP_HELP}`
      : `  --help  ${HELP_HELP}`;
  return `${groupUsage(group)}\n\n  ${summary}\n\nOptions:\n${options}\n\nCommands:\n${commands}\n`;
}

/**
 * Run the `yian` command line `argv` (the arguments after `yian`) and return its exit code.
 *
 * No arguments print the root help and return 2, `--help` prints it and returns 0, and
 * `--licenses` prints `LICENSES` and `--version` the version, each returning 0. The command name is
 * resolved first (an unknown one is a usage error, 2), then the `YIAN_<NAME>` defaults are checked
 * (a bad value is reported on stderr, and as a `defaults.invalid` document when `argv` holds
 * `--json`, 1). A group without its command prints the group's help and returns 2. A command line
 * its spec refuses, or a `UsageError` from the handler, prints the usage and the error and returns
 * 2; any other error prints its stack and returns 1. A warning yian raises while the command runs
 * prints as a `warning: <message>` line on stderr.
 */
export async function main(argv: readonly string[], io: CliIo = processIo()): Promise<number> {
  const [first, ...rest] = argv;
  if (first === undefined || first === "--help") {
    await io.stdout.write(groupHelp());
    return first === undefined ? 2 : 0;
  }
  if (first === "--version") {
    await io.stdout.write(`${VERSION}\n`);
    return 0;
  }
  if (first === "--licenses") {
    // Imported here only: the text is long, and no other command needs it.
    const { LICENSES } = await import("./licenses.ts");
    await io.stdout.write(LICENSES);
    return 0;
  }
  const fail = async (group: string | undefined, message: string) => {
    const name = group === undefined ? "yian" : `yian ${group}`;
    await io.stderr.write(`${groupUsage(group)}\nTry '${name} --help' for help.\n\nError: ${message}\n`);
    return 2;
  };
  if (first.startsWith("-")) return fail(undefined, `No such option: ${first}`);
  const inGroup = Object.hasOwn(GROUPS, first);
  if (!inGroup && !Object.hasOwn(COMMANDS, first)) return fail(undefined, `No such command '${first}'.`);

  try {
    read(io.env);
  } catch (error) {
    if (!(error instanceof DefaultsError)) throw error;
    await io.stderr.write(`${error.message}\n`);
    // The defaults come before any command parses its options, so `--json` is looked for in the
    // raw arguments; a `--json` after `--` matches too.
    if (argv.includes("--json")) {
      await printJson({ error: { code: "defaults.invalid", message: error.message } }, io.stdout);
    }
    return 1;
  }

  let key = first;
  let args = rest;
  if (inGroup) {
    const [sub, ...subRest] = rest;
    if (sub === undefined || sub === "--help") {
      await io.stdout.write(groupHelp(first));
      return sub === undefined ? 2 : 0;
    }
    if (sub.startsWith("-")) return fail(first, `No such option: ${sub}`);
    key = `${first} ${sub}`;
    if (!Object.hasOwn(COMMANDS, key)) return fail(first, `No such command '${sub}'.`);
    args = subRest;
  }

  const { spec, load } = COMMANDS[key]!;
  try {
    const parsed = parseCommand(spec, args);
    if (parsed.help) {
      await io.stdout.write(helpText(spec));
      return 0;
    }
    const handler = await load();
    // The CLI is the host, so yian's warnings print as its own lines; a library host keeps
    // `process.emitWarning`.
    return await withWarnings(
      (message) => void io.stderr.write(`warning: ${message}\n`).catch(() => {}),
      () => handler(parsed, io),
    );
  } catch (error) {
    if (error instanceof UsageError) {
      await io.stderr.write(usageError(spec, error));
      return 2;
    }
    await io.stderr.write(`${(error instanceof Error && error.stack) || String(error)}\n`);
    return 1;
  }
}

/**
 * The internal first argument the entry re-runs itself with for `run --json`: the child it marks
 * writes the document on fd 3. Users never pass it, and `main` never sees it.
 */
const JSON_CHILD = "--yian-json-child";

/**
 * Return the arguments the entry re-runs itself with to run `args` as the `run --json` child. A
 * compiled binary (`binary`) is its own runtime and script: its `argv[1]` is a file inside it and
 * its `execArgv` only `BUN_OPTIONS`, which the inherited environment passes on, so it gets the
 * marker and `args` only.
 */
export function childCommand(args: readonly string[], binary: boolean): string[] {
  if (binary) return [JSON_CHILD, ...args];
  return [...process.execArgv, process.argv[1]!, JSON_CHILD, ...args];
}

/** Whether `args` is a `run` command with `--json` before any `--`. */
function isRunJson(args: readonly string[]): boolean {
  const end = args.indexOf("--");
  return args[0] === "run" && args.slice(0, end === -1 ? undefined : end).includes("--json");
}

/**
 * Run `args` in a child of this runtime marked `JSON_CHILD`, whose fd 1 is this process's fd 2 (as
 * Python's `os.dup2(2, 1)`), copy its fd 3 to stdout, and exit as it did.
 *
 * The first SIGINT, SIGTERM or SIGHUP is forwarded as SIGUSR2 (except on Windows), a signal no
 * terminal sends, so a group Ctrl-C (which the child gets too) and the forwarded stop are one
 * stop; the second kills the child and exits 130. A child that died by SIGUSR2, before its
 * listener was on, exits 130. When the child can't start, or stdout fails other than by a closed
 * pipe (once the child has ended), it prints the error and exits 1.
 */
async function runJsonParent(args: readonly string[]): Promise<never> {
  const io = processIo();
  let stops = 0;
  // Listeners run from the event loop, after the synchronous `spawn` below has set `child`.
  const onStop = () => {
    stops += 1;
    if (stops === 1) {
      // Windows has no SIGUSR2 (`kill` throws `ERR_UNKNOWN_SIGNAL`); there the console's Ctrl-C
      // reaches the child itself, which takes SIGINT, and a second stop still kills it.
      if (process.platform !== "win32") child.kill("SIGUSR2");
      return;
    }
    // A run whose synchronous tool blocks its loop never takes the forwarded stop.
    child.kill("SIGKILL");
    process.exit(130);
  };
  const signals = ["SIGINT", "SIGTERM", "SIGHUP"] as const;
  // On before the spawn, so a stop during it isn't the default action; it reaches the child after.
  for (const name of signals) process.on(name, onStop);
  const command = childCommand(args, compiled);
  const child = spawn(process.execPath, command, { stdio: ["inherit", 2, "inherit", "pipe"] });
  child.on("error", (error) => {
    // Once started, only a failed `kill` errors, and the run goes on.
    if (child.pid !== undefined) return;
    // Synchronous, so it exits before the `close` that can follow a failed start.
    writeSync(2, `Error: cannot start the run: ${error.message}\n`);
    process.exit(1);
  });
  const closed = new Promise<[number | null, NodeJS.Signals | null]>((resolve) => {
    child.on("close", (code, signal) => resolve([code, signal]));
  });
  // Ending the relay early destroys the pipe, so the child's later writes end quietly too.
  const relayed = relay(child.stdio[3] as NodeJS.ReadableStream, io.stdout).then(
    () => undefined,
    (error: unknown) => error,
  );
  const [[code, signal], failed] = await Promise.all([closed, relayed]);
  if (failed !== undefined) {
    const reason = failed instanceof Error ? failed.message : String(failed);
    await io.stderr.write(`Error: cannot write to stdout: ${reason}\n`);
    await flush(io);
    process.exit(1);
  }
  if (code !== null) process.exit(code);
  if (signal === "SIGUSR2") process.exit(130);
  for (const name of signals) process.off(name, onStop);
  process.kill(process.pid, signal!);
  // Reached only when the signal doesn't end this process.
  process.exit(128 + constants.signals[signal!]);
}

/**
 * Run `args` as the `run --json` child: `main` with `stdout` writing to fd 3, and `CliIo.stop`
 * aborted with a `HostStop("SIGINT")` by the SIGUSR2 the parent forwards. Return the exit code.
 */
async function runJsonChild(args: readonly string[]): Promise<number> {
  // First, so a stop during start-up is kept until `run` links it.
  const stop = abortOnSignal("SIGUSR2", new HostStop("SIGINT"));
  const stdout = streamWriter(createWriteStream("", { fd: 3 }));
  const io: CliIo = { ...processIo(), stdout, stop };
  const code = await main(args, io);
  await flush(io);
  return code;
}

// True for the script Bun or Node (22.18 and later) runs and for a compiled binary's entry; false
// under a test runner, whose entry is the test file.
if (import.meta.main) {
  const args = process.argv.slice(2);
  // The explicit exits end a run whose tool left a timer, as Python's process exit does.
  if (args[0] === JSON_CHILD) process.exit(await runJsonChild(args.slice(1)));
  if (isRunJson(args)) await runJsonParent(args);
  const io = processIo();
  const code = await main(args, io);
  await flush(io);
  process.exit(code);
}
