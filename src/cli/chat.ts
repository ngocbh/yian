/**
 * `yian chat`: a terminal host that sends each typed line to a flow's `Inbox(str)` or
 * `Inbox(Message)` input and advances the run.
 *
 * The chat starts the flow, then loops while the run waits: read a line, `send` it, `advance`.
 * The engine holds no prompt and the run holds no worker between turns, and an `advance` finishes
 * before the next prompt opens, so the prompt never fights the run's output (lines typed during
 * an `advance` wait for the next prompt). While a line prompt is up, the chat also advances the
 * run at its `wakeAt`, so `sleep` and `ask`/`take` timeouts fire on time. With `--resume`, the chat
 * picks up a run a chat left (stopped or waiting) by advancing it, then goes on the same way.
 *
 * Knows about: the run boundary (`runFlow`, `advance`, `send`, `close`, `cancel`,
 * `loadRecordedFlow`, `inspectRun`), the waits' `wakeAt`, the record-time reader and the log's
 * records, compose, the terminal and the run command's helpers.
 */

import type { ComposedFlow } from "../compose/build/pipeline.ts";
import { LoadError } from "../compose/helpers/errors.ts";
import { Diagnostic } from "../diagnostics.ts";
import { NodeSucceeded, type RunEvent } from "../events.ts";
import { realScheduler, type Scheduler, SLEEP_CHUNK_MS } from "../platform.ts";
import {
  advance,
  cancel,
  close,
  DeliveryRefused,
  inspectRun,
  interrupt as interruptReply,
  loadRecordedFlow,
  RunBusyError,
  type RunResult,
  runFlow,
  send,
  withdraw,
} from "../run.ts";
import { type InboxClosedRecord, iterRunRecords, recordTimeMs } from "../runtime/log.ts";
import { wakeAt } from "../runtime/waits.ts";
import { registeredTools, type ToolEntry } from "../tools/registry.ts";
import { defaultRunId } from "../typesys/seeding.ts";
import { builtinType } from "../typesys/types.ts";
import { canonicalTypeLabel, isInboxType, typeEquals, type Value, writeAsText } from "../typesys/values.ts";
import { printStyled, renderDiagnostic, writeEngineTrace } from "./render.ts";
import {
  checkSuppliedInputs,
  gatherMissingInputs,
  indexRun,
  isSetupError,
  mergeInputs,
  readGrantedSkills,
  removeEmptyDir,
  renderRunError,
  reportRunLog,
  ResumeKeyError,
  resumeKeyCheck,
  validateRunLogOptions,
  waitingText,
} from "./run.ts";
import { cliLayer, freshLog, loadFlowFile, preflightKeys } from "./startup.ts";
import { Terminal, TerminalStopped } from "./terminal.ts";
import { atTerminal, type CliIo, CommandEnv, forTerminal, HostStop, shellQuote, UsageError } from "./utils.ts";

/**
 * The options `chat` reads; an absent option is `undefined`, an absent repeatable one an empty
 * list, an absent flag `false`.
 */
export interface ChatOptions {
  inbox?: string | undefined;
  inputValues: readonly string[];
  inputs?: string | undefined;
  log?: string | undefined;
  resume?: string | undefined;
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

/**
 * The inbox lines go to: its name, whether it takes messages (`Inbox(Message)`) or text, and
 * whether the flow has others lines could go to, so it must be named with `--inbox`.
 */
interface LineInbox {
  name: string;
  message: boolean;
  several: boolean;
}

/**
 * Return the `Inbox(str)` or `Inbox(Message)` input lines go to: `chosen`, or the flow's only one.
 *
 * @throws {UsageError} When the flow has none, several and `chosen` is absent, or `chosen` isn't
 *   one of them.
 */
function pickInbox(flow: ComposedFlow, chosen: string | undefined): LineInbox {
  const inboxes = flow.input.flatMap((param): Omit<LineInbox, "several">[] => {
    const type = param.contract?.type ?? null;
    if (!isInboxType(type)) return [];
    if (canonicalTypeLabel(type.element) === "str") return [{ name: param.name, message: false }];
    return typeEquals(type.element, builtinType("Message")!) ? [{ name: param.name, message: true }] : [];
  });
  const several = inboxes.length > 1;
  const names = inboxes.map((inbox) => inbox.name);
  const id = flow.compiled.flowId ?? flow.name;
  if (names.length === 0) {
    throw new UsageError(`flow '${id}' has no Inbox(str) or Inbox(Message) input; yian chat sends each line to one`);
  }
  if (chosen !== undefined) {
    const picked = inboxes.find((inbox) => inbox.name === chosen);
    if (picked !== undefined) return { ...picked, several };
    throw new UsageError(
      `--inbox '${chosen}' is not one of the flow's Inbox(str) or Inbox(Message) inputs (${names.join(", ")})`,
    );
  }
  if (names.length > 1) {
    throw new UsageError(
      `flow '${id}' has several Inbox(str) or Inbox(Message) inputs (${names.join(", ")}); pick one with --inbox`,
    );
  }
  return { ...inboxes[0]!, several: false };
}

/** Return the inbox event a typed line becomes: the line, or a user message holding it as text. */
function lineEvent(line: string, message: boolean): string | Record<string, unknown> {
  return message ? { role: "user", parts: [{ type: "text", text: line }] } : line;
}

/** Return the text a withdrawn event holds: a `str` as it is, a message's text parts joined. */
function valueText(value: unknown, message: boolean): string {
  if (!message) return String(value);
  const parts = (value as { parts?: readonly { type?: unknown; text?: unknown }[] }).parts ?? [];
  return parts.map((part) => (part.type === "text" && typeof part.text === "string" ? part.text : "")).join("");
}

/**
 * Return the flow the run at `log` recorded, for `--resume`; or print why it can't go on and
 * return `null`: the run has ended, its log can't be read, or its flow was built in code or no
 * longer loads (located, with its stack under `engineTrace`).
 *
 * @throws What `inspectRun` and `loadRecordedFlow` throw that isn't a load or setup error.
 */
async function recordedFlow(log: string, engineTrace: boolean, io: CliIo): Promise<ComposedFlow | null> {
  const failed = async (error: Error) => {
    await io.stderr.write(`chat failed: ${forTerminal(error.message, io.stderr)}\n`);
    await writeEngineTrace(error, engineTrace, io);
    return null;
  };
  try {
    const { status } = inspectRun(log);
    if (status === "succeeded" || status === "failed" || status === "cancelled") {
      return await failed(new RangeError(`the run in ${log} has already ended (${status}); start a new chat`));
    }
    const flow = await loadRecordedFlow(log, { env: io.env });
    if (flow !== null) return flow;
    return await failed(new RangeError("this run's flow was built in code; resume it with advance({log, flow})"));
  } catch (error) {
    if (error instanceof LoadError) {
      await renderDiagnostic(error.diagnostic, io.stderr, io.env);
      await writeEngineTrace(error, engineTrace, io);
      return null;
    }
    if (!isSetupError(error)) throw error;
    return failed(error);
  }
}

/**
 * Return what a stopped chat prints: where its run is, and the `yian chat --resume` that continues
 * it, naming the inbox when the flow has several (the log doesn't record which one the chat used).
 */
function stoppedText(log: string, inbox: LineInbox): string {
  const named = inbox.several ? ` --inbox ${shellQuote(inbox.name)}` : "";
  return `chat stopped; the run is in ${log}\ncontinue with: yian chat --resume ${shellQuote(log)}${named}\n`;
}

/** Whether the run log at `log` records that the top-level inbox `inbox` was closed. */
function inboxClosed(log: string, inbox: string): boolean {
  for (const entry of iterRunRecords(log, { kinds: ["inbox_closed"] })) {
    if ((entry.envelope.record as InboxClosedRecord).inbox === inbox) return true;
  }
  return false;
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
 * run (`chat cancelled`, exit 130). On a terminal, Esc, during a turn or at the prompt, takes back
 * the lines the run hasn't read yet (`withdraw`) and puts their text back in the prompt
 * (`(withdrew N messages; back in the prompt)`; after `/exit` it prints them), then stops the
 * turn (`interrupt`; `(stopping the reply)` when a Stop is recorded): a running reply is cut, and
 * during a sleep or an approval the reply after it stops before its first model call. At the take
 * Esc does nothing; it never cancels. While a turn runs on a terminal, the input is raw, so Ctrl-C
 * arrives as a key with the same rule, Ctrl-Z and Ctrl-\ raise no signal, and other keys are kept
 * for the next prompt. SIGTERM, SIGHUP or
 * `io.stop` stops the run and leaves its log (`chat stopped; the run is in <log>` and the
 * `yian chat --resume <log>` that continues it, exit 130), as does Ctrl-C at an `ask` prompt,
 * which ignores Esc. While the prompt is up, the run is advanced
 * at its `wakeAt`; what was typed so far comes back at the next prompt. Off a terminal (stdin or
 * stderr isn't a TTY), each prompt is written once, with no escape codes, and Esc isn't seen. An
 * `ask` node prompts at the terminal. A line the inbox refuses prints `not sent: <reason>` and the
 * chat goes on. A failed run prints its diagnostic and exits 1; a
 * run that ends waiting after the inbox closed, with no deadline to wake it, prints what it waits
 * on and exits 0.
 *
 * The flow loads, inputs, provider keys, log path and run index are handled as `yian run`'s
 * (`--engine-trace` included), except that the inbox input is never asked for or supplied. Skills
 * are read as `yian run` reads them, once per chat, before any input prompt. The chat opens no MCP
 * servers and builds no memories. A host failure prints `chat failed: <message>`, its stack below
 * under `engineTrace`, and exits 1.
 *
 * With `options.resume` there is no `flowPath`: the chat picks up the run recorded in that log,
 * with the flow it recorded (`loadRecordedFlow`), as `yian run --resume` does. Its first move is an
 * `advance` (`numWorkers` absent keeps the recorded count), so a reply the stop cut runs again and
 * prints; replies printed before aren't shown again. Then the chat goes on as above; when the log
 * records that the inbox was closed, no prompt comes back. Each move that can call a model first
 * checks the keys the recorded configs need, as `yian run --resume` does; a refused key prints its
 * message and exits 1. A run that has ended prints `chat failed: the run in <log> has already
 * ended (<status>); ...`, appends nothing and exits 1, as do a log that can't be read and a flow
 * built in code; a recorded flow that no longer loads is reported located.
 *
 * @throws {UsageError} As `validateRunLogOptions` and `mergeInputs` do; for a flow without exactly
 *   one `Inbox(str)` or `Inbox(Message)` input to pick; for an input the flow doesn't declare
 *   (the inbox included); off a TTY, for missing required inputs.
 */
export async function chat(
  flowPath: string | undefined,
  options: ChatOptions,
  io: CliIo,
  deps: ChatDeps = {},
): Promise<number> {
  validateRunLogOptions({ ...options, flow: flowPath });
  const resumed = options.resume ?? null;
  const flow =
    resumed === null
      ? await loadFlowFile(flowPath!, options.engineTrace, io)
      : await recordedFlow(resumed, options.engineTrace, io);
  if (flow === null || flow instanceof Diagnostic) return 1;
  const inbox = pickInbox(flow, options.inbox);
  const supplied = mergeInputs(options.inputs, options.inputValues);
  checkSuppliedInputs(flow.input, supplied);

  const scheduler = deps.scheduler ?? realScheduler;
  const signals = deps.signals ?? process;
  // SIGTERM, SIGHUP, `io.stop`, or Ctrl-C at an `ask` prompt: the run stops; its log stays.
  const stop = new AbortController();
  // Set once the run starts: before it, a stopped prompt answers null and the chat reports that.
  let log: string | null = null;
  // The run folder the chat made for its default log path, removed at the end if still empty.
  let runDir: string | null = null;
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
  // The interrupt a Ctrl-C or Esc started; settled before the chat sends its next line or returns.
  let interrupting: Promise<void> | null = null;
  // What was typed at a prompt a deadline or Esc took down, plus what Esc withdrew, shown again at
  // the next prompt.
  let draft = "";
  // Set once the inbox is closed (`/exit`, `/quit`, the end of input): no prompt comes back.
  let closed = false;
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
      // The live engine takes it at once. With no reply running it records the Stop for the turn,
      // or nothing when the chat waits at the take.
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
  // Esc: take back what the run hasn't read, then stop the turn; the text goes back to the prompt.
  // It never cancels and isn't a Ctrl-C press.
  const escape = (): void => {
    if (log === null) return;
    const path = log;
    const done = (async () => {
      const back = await withdraw(path, inbox.name, { env: env.read });
      // Kept before the interrupt, which may be refused.
      if (back.length > 0) {
        const texts = back.map((event) => valueText(event.value, inbox.message)).filter(Boolean);
        const count = `${back.length} message${back.length === 1 ? "" : "s"}`;
        // After /exit no prompt comes back, so the text is printed rather than lost.
        if (closed) await io.stderr.write(`(withdrew ${count}: ${texts.join(" ")})\n`);
        else {
          // Sent before the draft was typed, so it goes first.
          draft = [...texts, draft].filter(Boolean).join(" ");
          await io.stderr.write(`(withdrew ${count}; back in the prompt)\n`);
        }
      }
      const stopped = await interruptReply(path, inbox.name, { env: env.read });
      if (stopped.seq !== null) await io.stderr.write("(stopping the reply)\n");
    })().catch((error: unknown) => {
      if (!(error instanceof DeliveryRefused)) stop.abort(error);
    });
    interrupting = Promise.all([interrupting, done]).then(() => {});
  };
  const terminal = new Terminal({
    input: io.stdin,
    output: io.stderrStream,
    // Off a terminal, stderr gets the prompts once each, with no escape codes.
    plain: !atTerminal(io),
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
    // Read once: every `advance` must pass the skills the run recorded at its start.
    const skills = await readGrantedSkills(flow, env.read, io);
    const prompter = io.stdin.isTTY === true ? terminal : null;
    const runId = defaultRunId();
    const cliCfg = cliLayer(options);
    let path: string;
    if (resumed === null) {
      const gathered = await gatherMissingInputs(flow.input, supplied, prompter);
      stop.signal.throwIfAborted();
      if (gathered === null) {
        await io.stderr.write("chat cancelled\n");
        return 1;
      }
      Object.assign(supplied, gathered);
      const missing = await preflightKeys(flow, cliCfg, supplied, { env, terminal: prompter }, stop.signal);
      if (missing !== null) {
        await printStyled(io, missing, "red");
        return 1;
      }
      stop.signal.throwIfAborted();
      const fresh = freshLog(options.log, runId, io.env);
      path = fresh.log;
      runDir = fresh.runDir;
    } else {
      path = resumed;
      // A chat that ended with /exit before it stopped shows no prompt: no line can be sent.
      closed = inboxClosed(path, inbox.name);
    }
    await reportRunLog(path, io.stderr, io.env);
    log = path;
    // A resume runs with the configs the log recorded: each move that can call a model first
    // checks their keys, prompting for a missing one on a terminal.
    const checkLlmConfigs =
      resumed === null ? null : resumeKeyCheck({ env, terminal: prompter, signal: stop.signal });
    // A node named `reply`, at any depth, is what the chat says.
    const onEvent = (event: RunEvent) => {
      if (!(event instanceof NodeSucceeded)) return;
      if (event.nodeId.slice(event.nodeId.lastIndexOf("/") + 1) !== "reply") return;
      // An interrupted reply succeeds with null when its output is `Optional(...)`.
      if (event.interrupted) return io.stderr.write("(interrupted)\n");
      if (event.output === null) return;
      return io.stdout.write(`${forTerminal(writeAsText(event.output as Value), io.stdout)}\n`);
    };
    const shared = {
      onEvent,
      tools: { ...registeredTools(), ...deps.tools, ask: terminal.askUser },
      skills,
      env: env.read,
      signal: stop.signal,
      scheduler,
    };
    /**
     * Run `move` as a turn: a Ctrl-C meanwhile interrupts the reply instead of cancelling, and on a
     * terminal, Esc withdraws and stops it (`escape`).
     */
    const turn = async (move: () => Promise<RunResult>): Promise<RunResult> => {
      turning = true;
      const keys = new AbortController();
      terminal.watchKeys(keys.signal, (key) => (key === "escape" ? escape() : interrupt()));
      try {
        return await move();
      } finally {
        keys.abort();
        turning = false;
        // An interrupt sent during the turn lands before the next line is sent.
        await interrupting;
        interrupting = null;
      }
    };
    const moveOn = () => turn(() => advance({ log: path, checkLlmConfigs, ...shared }));
    let result: RunResult = await turn(() =>
      resumed === null
        ? runFlow(flow, supplied, {
            log: path,
            runId,
            llmConfig: Object.keys(cliCfg).length > 0 ? cliCfg : null,
            numWorkers: options.numWorkers,
            ...shared,
          })
        : advance({ log: path, numWorkers: options.numWorkers ?? null, checkLlmConfigs, ...shared }),
    );
    let cancelled = false;
    // The deadline the run was last moved at. A node is due once any of its calls passes its
    // deadline, so that move times it out and `wakeAt` comes later; were it to stay, a timer for
    // it would only spin.
    let spent: string | null = null;
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
      const read = await terminal.readLine("You:", { signal: prompt.signal, initial: draft, escape: true });
      draft = "aborted" in read || "escape" in read ? read.draft : "";
      timer.abort();
      line = null;
      stop.signal.throwIfAborted();
      if (cancelling !== null) continue;
      if ("escape" in read) {
        // Offline: a turn that waits mid-way (a sleep, an approval) has the Stop recorded for its
        // reply; at the take it is a no-op.
        escape();
        await interrupting;
        interrupting = null;
        continue;
      }
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
      // A run that ended under the chat (a cancel from outside) refuses the line or the close as
      // `ended`; the move below then reads how it ended, and the loop stops.
      if ("eof" in read || EXIT_LINES.has(read.value.trim())) {
        try {
          await close(path, inbox.name, { env: env.read });
        } catch (error) {
          if (!(error instanceof DeliveryRefused && error.reason === "ended")) throw error;
        }
        closed = true;
      } else {
        pressedAt = null;
        try {
          await send(path, inbox.name, lineEvent(read.value, inbox.message), { env: env.read });
        } catch (error) {
          if (!(error instanceof DeliveryRefused)) throw error;
          if (error.reason !== "ended") {
            await io.stderr.write(`not sent: ${forTerminal(error.message, io.stderr)}\n`);
            continue;
          }
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
      await io.stderr.write(log === null ? "chat cancelled\n" : stoppedText(log, inbox));
      return error instanceof HostStop ? 130 : 1;
    }
    if (error instanceof ResumeKeyError) {
      await printStyled(io, error.message, "red");
      return 1;
    }
    // Another writer holding the run's log is a busy run, not a bug.
    if (!(error instanceof RunBusyError || isSetupError(error))) throw error;
    await io.stderr.write(`chat failed: ${forTerminal(error.message, io.stderr)}\n`);
    await writeEngineTrace(error, options.engineTrace, io);
    return 1;
  } finally {
    await interrupting;
    stop.signal.removeEventListener("abort", takeDown);
    terminal.close();
    for (const [name, listener] of listeners) signals.off(name, listener);
    outside?.removeEventListener("abort", forwardStop);
    if (log !== null) await indexRun(log, io);
    if (runDir !== null) removeEmptyDir(runDir);
  }
}
