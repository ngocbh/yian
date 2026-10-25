/**
 * `yian chat`: a terminal host that sends each typed line to a flow's `Inbox(str)` or
 * `Inbox(Message)` input and advances the run.
 *
 * The chat starts the flow, then loops while the run waits: read a line, `send` it, `advance`.
 * The engine holds no prompt and the run holds no worker between turns, and an `advance` finishes
 * before the next prompt opens, so the prompt never fights the run's output (lines typed during
 * an `advance` wait for the next prompt). While a line prompt is up, the chat also advances the
 * run at its `wakeAt`, so `sleep` and `ask`/`take` timeouts fire on time.
 *
 * Knows about: the run boundary (`runFlow`, `advance`, `send`, `close`, `cancel`), the waits'
 * `wakeAt` and the record-time reader, compose, the terminal and the run command's helpers.
 */

import { basename, dirname } from "node:path";

import type { ComposedFlow } from "../compose/build/pipeline.ts";
import { LoadError } from "../compose/helpers/errors.ts";
import { loadFlow } from "../compose/loader.ts";
import { NodeSucceeded, type RunEvent } from "../events.ts";
import { AuthError } from "../llm_clients/auth/index.ts";
import { checkInputNames } from "../nodes/binding.ts";
import type { Parameter } from "../nodes/contracts.ts";
import { realScheduler, type Scheduler, SLEEP_CHUNK_MS } from "../platform.ts";
import {
  advance,
  cancel,
  close,
  DeliveryRefused,
  interrupt as interruptReply,
  RunBusyError,
  type RunResult,
  runFlow,
  send,
} from "../run.ts";
import { recordTimeMs } from "../runtime/log.ts";
import { wakeAt } from "../runtime/waits.ts";
import { registeredTools, type ToolEntry } from "../tools/registry.ts";
import { defaultRunId } from "../typesys/seeding.ts";
import { builtinType } from "../typesys/types.ts";
import { canonicalTypeLabel, isInboxType, typeEquals, type Value, writeAsText } from "../typesys/values.ts";
import {
  defaultRunLogPath,
  ensureProviderKeys,
  gatherMissingInputs,
  INPUTS_HINT,
  isSetupError,
  mergeInputs,
  NotUtf8Error,
  readFlowText,
  renderDiagnostic,
  renderRunError,
  reportRunLog,
  validateRunLogOptions,
  waitingText,
} from "./run.ts";
import { Terminal, TerminalStopped } from "./terminal.ts";
import { type CliIo, CommandEnv, HostStop, MissingKeyError, UsageError } from "./utils.ts";

/**
 * The options `chat` reads; an absent option is `undefined`, an absent repeatable one an empty
 * list, an absent flag `false`.
 */
export interface ChatOptions {
  inbox?: string | undefined;
  inputValues: readonly string[];
  inputs?: string | undefined;
  log?: string | undefined;
  numWorkers?: number | undefined;
  provider?: string | undefined;
  model?: string | undefined;
  engineTrace: boolean;
}

/**
 * @internal Test seams: the clock the run and the chat's timers share, the source of the process
 * signals, and host tools added to the registered ones.
 */
export interface ChatDeps {
  scheduler?: Scheduler;
  signals?: Pick<NodeJS.EventEmitter, "on" | "off">;
  tools?: Readonly<Record<string, ToolEntry>>;
}

const INTERRUPTING = "(interrupting the reply; Ctrl-C again within 2 s to cancel the run)";
// A second Ctrl-C this soon after the one that interrupted a turn cancels the run.
const SECOND_PRESS_MS = 2_000;
const EXIT_LINES = new Set(["/exit", "/quit"]);

/** The inbox lines go to: its name, and whether it takes messages (`Inbox(Message)`) or text. */
interface LineInbox {
  name: string;
  message: boolean;
}

/**
 * Return the `Inbox(str)` or `Inbox(Message)` input lines go to: `chosen`, or the flow's only one.
 *
 * @throws {UsageError} When the flow has none, several and `chosen` is absent, or `chosen` isn't
 *   one of them.
 */
function pickInbox(flow: ComposedFlow, chosen: string | undefined): LineInbox {
  const inboxes = flow.input.flatMap((param): LineInbox[] => {
    const type = param.contract?.type ?? null;
    if (!isInboxType(type)) return [];
    if (canonicalTypeLabel(type!.element!) === "str") return [{ name: param.name, message: false }];
    return typeEquals(type!.element!, builtinType("Message")!) ? [{ name: param.name, message: true }] : [];
  });
  const names = inboxes.map((inbox) => inbox.name);
  const id = flow.compiled.flowId ?? flow.name;
  if (names.length === 0) {
    throw new UsageError(`flow '${id}' has no Inbox(str) or Inbox(Message) input; yian chat sends each line to one`);
  }
  if (chosen !== undefined) {
    const picked = inboxes.find((inbox) => inbox.name === chosen);
    if (picked !== undefined) return picked;
    throw new UsageError(
      `--inbox '${chosen}' is not one of the flow's Inbox(str) or Inbox(Message) inputs (${names.join(", ")})`,
    );
  }
  if (names.length > 1) {
    throw new UsageError(
      `flow '${id}' has several Inbox(str) or Inbox(Message) inputs (${names.join(", ")}); pick one with --inbox`,
    );
  }
  return inboxes[0]!;
}

/** Return the inbox event a typed line becomes: the line, or a user message holding it as text. */
function lineEvent(line: string, message: boolean): string | Record<string, unknown> {
  return message ? { role: "user", parts: [{ type: "text", text: line }] } : line;
}

/**
 * Resolve once `scheduler`'s clock reaches `at` (epoch milliseconds), re-reading it after each
 * chunk, so a laptop that slept still wakes on time.
 *
 * @throws `signal.reason` once it aborts.
 */
async function sleepUntil(scheduler: Scheduler, at: number, signal: AbortSignal): Promise<void> {
  for (let left = at - scheduler.now(); left > 0; left = at - scheduler.now()) {
    await scheduler.sleep(Math.min(left, SLEEP_CHUNK_MS), signal);
  }
}

/**
 * Chat with the flow at `flowPath`: start it, send each line typed at the `You:` prompt to its
 * one `Inbox(str)` or `Inbox(Message)` input (or `options.inbox`; a `Message` inbox gets the line
 * as a user message holding it as text), and advance the run after each send, printing the
 * output of each node named `reply` on stdout as it finishes; returns the exit code.
 *
 * `/exit`, `/quit` or the end of input closes the inbox; the run finishes, `session ended` goes to
 * stderr and the chat exits 0. Ctrl-C (SIGINT) while a turn runs interrupts the agents reading the
 * inbox as their steer inbox (`interrupt`): the reply is cut, `(interrupted)` goes to stderr and
 * the chat waits for the next line. A second Ctrl-C within 2 s, or one at the prompt, cancels the
 * run (`chat cancelled`, exit 130). SIGTERM, SIGHUP or
 * `io.stop` stops the run and leaves its log (`chat stopped; the run is in <log>`, exit 130), as
 * does Ctrl-C at an `ask` prompt. While the prompt is up, the run is advanced at its `wakeAt`;
 * what was typed so far comes back at the next prompt. Off a terminal (stdin or stderr isn't a
 * TTY), each prompt is written once, with no escape codes. An `ask` node prompts at the terminal.
 * A line the inbox refuses prints `not sent: <reason>` and the chat goes on. A failed run prints its diagnostic and exits 1; a
 * run that ends waiting after the inbox closed, with no deadline to wake it, prints what it waits
 * on and exits 0.
 *
 * The flow loads, inputs, provider keys and log path are handled as `yian run`'s, except that the
 * inbox input is never asked for or supplied. The chat opens no MCP servers, loads no skills and
 * builds no memories. A host failure prints `chat failed: <message>` and exits 1.
 *
 * @throws {UsageError} As `validateRunLogOptions` and `mergeInputs` do; for a flow without exactly
 *   one `Inbox(str)` or `Inbox(Message)` input to pick; for an input the flow doesn't declare
 *   (the inbox included); off a TTY, for missing required inputs.
 */
export async function chat(flowPath: string, options: ChatOptions, io: CliIo, deps: ChatDeps = {}): Promise<number> {
  validateRunLogOptions({ ...options, flow: flowPath });
  let text: string;
  try {
    text = readFlowText(flowPath);
  } catch (error) {
    if (!(error instanceof NotUtf8Error)) throw error;
    await io.stderr.write(`can't read ${flowPath}: not UTF-8 text\n`);
    return 1;
  }
  let flow: ComposedFlow;
  try {
    flow = await loadFlow(text, { searchPaths: [dirname(flowPath)], sourceLabel: basename(flowPath), env: io.env });
  } catch (error) {
    if (!(error instanceof LoadError)) throw error;
    await renderDiagnostic(error.diagnostic, io.stderr, io.env);
    return 1;
  }
  const inbox = pickInbox(flow, options.inbox);
  // An inbox takes events through `send`, never a value at the start.
  const values: Parameter[] = flow.input.filter((param) => !isInboxType(param.contract?.type ?? null));
  const supplied = mergeInputs(options.inputs, options.inputValues);
  try {
    checkInputNames(values, supplied);
  } catch (error) {
    if (!(error instanceof RangeError)) throw error;
    throw new UsageError(`${INPUTS_HINT}: ${error.message}`);
  }

  const scheduler = deps.scheduler ?? realScheduler;
  const signals = deps.signals ?? process;
  // SIGTERM, SIGHUP, `io.stop`, or Ctrl-C at an `ask` prompt: the run stops; its log stays.
  const stop = new AbortController();
  // Set once the run starts: before it, a stopped prompt answers null and the chat reports that.
  let log: string | null = null;
  const env = new CommandEnv(io.env);
  // The line prompt (or, once the inbox is closed, the wait for a deadline) that a deadline, a
  // stop or a cancel takes down.
  let line: AbortController | null = null;
  // Whether a turn runs (the run is being started or advanced), and the scheduler time of the
  // Ctrl-C that last interrupted one.
  let turning = false;
  let pressedAt: number | null = null;
  // Settles true once the run is cancelled, false when it had already ended.
  let cancelling: Promise<boolean> | null = null;
  // The interrupt a Ctrl-C started; settled before the chat sends its next line or returns.
  let interrupting: Promise<void> | null = null;
  const cancelRun = async (path: string): Promise<boolean> => {
    try {
      await cancel(path, { env: env.read });
      return true;
    } catch (error) {
      if (error instanceof DeliveryRefused) return false;
      throw error;
    }
  };
  const interrupt = (): void => {
    if (log === null) {
      stop.abort(new HostStop("SIGINT"));
      return;
    }
    const now = scheduler.now();
    // A press while a turn runs interrupts its reply; a second within 2 s, or one at the prompt,
    // cancels.
    if (turning && (pressedAt === null || now - pressedAt > SECOND_PRESS_MS)) {
      pressedAt = now;
      void io.stderr.write(`${INTERRUPTING}\n`).catch(() => {});
      // The live engine takes it at once; with no reply running it names nobody and appends nothing.
      const sent = interruptReply(log, inbox.name, { env: env.read }).catch((error: unknown) => {
        if (!(error instanceof DeliveryRefused)) stop.abort(error);
      });
      interrupting = Promise.all([interrupting, sent]).then(() => {});
      return;
    }
    // A live engine takes the cancel, and its `advance` resolves `cancelled`.
    if (cancelling === null) {
      cancelling = cancelRun(log);
      // Awaited by the loop; until then its failure mustn't count as unhandled.
      cancelling.catch(() => {});
    }
    line?.abort();
  };
  const tty = io.stdin.isTTY === true && (io.stderrStream as { isTTY?: boolean }).isTTY === true;
  const terminal = new Terminal({
    input: io.stdin,
    output: io.stderrStream,
    // Off a terminal, stderr gets the prompts once each, with no escape codes.
    plain: !tty,
    onStop: (reason) => {
      if (log !== null) stop.abort(reason === "interrupt" ? new HostStop("SIGINT") : new TerminalStopped("eof"));
    },
  });
  const onStopSignal = (signal: NodeJS.Signals) => () => stop.abort(new HostStop(signal));
  const listeners = [
    ["SIGINT", interrupt],
    ["SIGTERM", onStopSignal("SIGTERM")],
    ["SIGHUP", onStopSignal("SIGHUP")],
  ] as const;
  for (const [name, listener] of listeners) signals.on(name, listener);
  const outside = io.stop;
  const forwardStop = () => stop.abort(outside?.reason);
  if (outside?.aborted) forwardStop();
  else outside?.addEventListener("abort", forwardStop, { once: true });
  const takeDown = () => {
    line?.abort();
    // Before the run, a stop ends an input or key prompt rather than waiting for its answer.
    if (log === null) terminal.close();
  };
  stop.signal.addEventListener("abort", takeDown, { once: true });

  try {
    const prompter = io.stdin.isTTY === true ? terminal : null;
    const gathered = await gatherMissingInputs(values, supplied, prompter);
    stop.signal.throwIfAborted();
    if (gathered === null) {
      await io.stderr.write("chat cancelled\n");
      return 1;
    }
    Object.assign(supplied, gathered);
    // The flags supply the outermost cascade layer, as `yian run`'s.
    const cliCfg: Record<string, string> = {};
    if (options.provider) cliCfg["provider"] = options.provider;
    if (options.model) cliCfg["model"] = options.model;
    try {
      await ensureProviderKeys(flow, cliCfg, supplied, { env, terminal: prompter });
    } catch (error) {
      stop.signal.throwIfAborted();
      if (!(error instanceof MissingKeyError || error instanceof AuthError)) throw error;
      await io.stderr.write(`${error.message}\n`);
      return 1;
    }
    stop.signal.throwIfAborted();

    const runId = defaultRunId();
    const path = options.log ?? defaultRunLogPath(runId, io.env);
    await reportRunLog(path, io.stderr, io.env);
    log = path;
    // Python's rule: a node named `reply`, at any depth, is what the chat says.
    const onEvent = (event: RunEvent) => {
      if (!(event instanceof NodeSucceeded)) return;
      if (event.nodeId.slice(event.nodeId.lastIndexOf("/") + 1) !== "reply") return;
      // An interrupted reply succeeds with null when its output is `Optional(...)`.
      if (event.interrupted) return io.stderr.write("(interrupted)\n");
      if (event.output === null) return;
      return io.stdout.write(`${writeAsText(event.output as Value)}\n`);
    };
    const shared = {
      onEvent,
      tools: { ...registeredTools(), ...deps.tools, ask: terminal.askUser },
      env: env.read,
      signal: stop.signal,
      scheduler,
    };
    /** Run `move` as a turn: a Ctrl-C meanwhile interrupts the reply instead of cancelling. */
    const turn = async (move: () => Promise<RunResult>): Promise<RunResult> => {
      turning = true;
      try {
        return await move();
      } finally {
        turning = false;
        // An interrupt sent during the turn lands before the next line is sent.
        await interrupting;
        interrupting = null;
      }
    };
    const moveOn = () => turn(() => advance({ log: path, ...shared }));
    let result: RunResult = await turn(() =>
      runFlow(flow, supplied, {
        log: path,
        runId,
        llmConfig: Object.keys(cliCfg).length > 0 ? cliCfg : null,
        numWorkers: options.numWorkers,
        ...shared,
      }),
    );
    let closed = false;
    let cancelled = false;
    // The deadline the run was last moved at. A node is due once any of its calls passes its
    // deadline, so that move times it out and `wakeAt` comes later; were it to stay, a timer for
    // it would only spin.
    let spent: string | null = null;
    // What was typed at a prompt a deadline took down, shown again at the next one.
    let draft = "";
    while (result.status === "waiting") {
      if (cancelling !== null) {
        cancelled = await cancelling;
        cancelling = null;
        if (cancelled) break;
        // The run had ended under the chat: read how.
        result = await moveOn();
        continue;
      }
      const next = wakeAt(result.waits);
      const wake: string | null = next === spent ? null : next;
      if (closed) {
        // Nothing more will be typed: only a deadline can move the run now.
        if (wake === null) {
          await io.stderr.write(waitingText(result));
          return 0;
        }
        const nap = new AbortController();
        line = nap;
        if (stop.signal.aborted) nap.abort();
        await sleepUntil(scheduler, recordTimeMs(wake, "wakeAt"), nap.signal).catch(() => {});
        line = null;
        stop.signal.throwIfAborted();
        if (cancelling === null) {
          spent = wake;
          result = await moveOn();
        }
        continue;
      }
      const prompt = new AbortController();
      line = prompt;
      const timer = new AbortController();
      if (wake !== null) {
        sleepUntil(scheduler, recordTimeMs(wake, "wakeAt"), timer.signal).then(
          () => prompt.abort(),
          () => {},
        );
      }
      if (stop.signal.aborted) prompt.abort();
      const read = await terminal.readLine("You:", { signal: prompt.signal, initial: draft });
      draft = "aborted" in read ? read.draft : "";
      timer.abort();
      line = null;
      stop.signal.throwIfAborted();
      if (cancelling !== null) continue;
      if ("aborted" in read) {
        // A deadline passed.
        spent = wake;
        result = await moveOn();
        continue;
      }
      if ("interrupt" in read) {
        interrupt();
        continue;
      }
      if ("eof" in read || EXIT_LINES.has(read.value.trim())) {
        await close(path, inbox.name, { env: env.read });
        closed = true;
      } else {
        pressedAt = null;
        try {
          await send(path, inbox.name, lineEvent(read.value, inbox.message), { env: env.read });
        } catch (error) {
          if (!(error instanceof DeliveryRefused)) throw error;
          await io.stderr.write(`not sent: ${error.message}\n`);
          continue;
        }
      }
      result = await moveOn();
    }
    // A cancel that lost to the run's end is refused; its other failures are the chat's.
    if (cancelling !== null) await cancelling;
    if (cancelled || result.status === "cancelled") {
      await io.stderr.write("chat cancelled\n");
      return 130;
    }
    if (result.status === "succeeded") {
      await io.stderr.write("session ended\n");
      return 0;
    }
    await renderRunError(result, flow, options.engineTrace, io.stderr, io.env);
    return 1;
  } catch (error) {
    if (error instanceof HostStop || (error instanceof TerminalStopped && error.reason === "eof")) {
      await io.stderr.write(log === null ? "chat cancelled\n" : `chat stopped; the run is in ${log}\n`);
      return error instanceof HostStop ? 130 : 1;
    }
    // Another writer holding the run's log is a busy run, not a bug.
    if (!(error instanceof RunBusyError || isSetupError(error))) throw error;
    await io.stderr.write(`chat failed: ${error.message}\n`);
    return 1;
  } finally {
    await interrupting;
    stop.signal.removeEventListener("abort", takeDown);
    terminal.close();
    for (const [name, listener] of listeners) signals.off(name, listener);
    outside?.removeEventListener("abort", forwardStop);
  }
}
