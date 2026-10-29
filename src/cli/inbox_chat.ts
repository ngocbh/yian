/**
 * The inbox chat: a terminal host that sends each typed line to a flow's `Inbox(str)` or
 * `Inbox(Message)` input and advances the run. `yian run` comes here first (`runOrChat`): on a
 * terminal, a flow the chat can take is chatted with, and any other goes on to `run`. `yian chat`
 * (`chat.ts`) runs its bundled assistant through it.
 *
 * The chat starts the flow, then loops while the run waits: read a line, `send` it, `advance`.
 * The engine holds no prompt and the run holds no worker between turns, and an `advance` finishes
 * before the next prompt opens, so the prompt never fights the run's output (lines typed during
 * an `advance` wait for the next prompt). While a line prompt is up, the chat also advances the
 * run at its `wakeAt`, so `sleep` and `ask`/`take` timeouts fire on time. With `--resume`, the chat
 * picks up a run a chat left (stopped or waiting) by advancing it, then goes on the same way.
 *
 * Given child events (`InboxChatHost.events`), the chat also sends each into the inbox as a message
 * authored `yian`, through its flusher (`chat_flush.ts`), which it tells when turns begin and end,
 * when the prompt is up and when the user interrupts or sends a line.
 *
 * Knows about: the run boundary (`runFlow`, `advance`, `send`, `close`, `cancel`,
 * `loadRecordedFlow`, `inspectRun`), the waits' `wakeAt`, the record-time reader and the log's
 * records, compose, the terminal, the start-up it shares with `yian run` (`startup.ts`), the run
 * command's helpers and the chat's flusher.
 */

import { resolveMemoryLayout } from "../compile/memory_layout.ts";
import type { ComposedFlow } from "../compose/build/pipeline.ts";
import { LoadError } from "../compose/helpers/errors.ts";
import { Diagnostic } from "../diagnostics.ts";
import { NodeSucceeded, type RunEvent, RunStarted, type StreamChunk } from "../events.ts";
import type { ModelAccess } from "../llm_clients/factory.ts";
import { realScheduler, type Scheduler } from "../platform.ts";
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
import type { Skill } from "../skills/catalog.ts";
import { registeredTools, type ToolEntry } from "../tools/registry.ts";
import { builtinType } from "../typesys/types.ts";
import { canonicalTypeLabel, isInboxType, typeEquals, type Value, writeAsText } from "../typesys/values.ts";
import type { ChildEvent } from "./chat_events.ts";
import { type ChatEvents, ChatFlusher } from "./chat_flush.ts";
import { printStyled, renderDiagnostic, writeEngineTrace } from "./render.ts";
import {
  checkSuppliedInputs,
  gatherMissingInputs,
  indexRun,
  isSetupError,
  mergeInputs,
  needsMcp,
  readGrantedSkills,
  removeEmptyDir,
  renderRunError,
  reportRunLog,
  ResumeKeyError,
  resumeKeyCheck,
  run,
  type RunOptions,
  validateRunLogOptions,
  waitingText,
} from "./run.ts";
import { sanitize } from "./sanitize.ts";
import { cliLayer, freshLog, loadFlowFile, preflightKeys } from "./startup.ts";
import { type Key, type LineEditor, type PromptList, Terminal, TerminalStopped } from "./terminal.ts";
import { atTerminal, type CliIo, CommandEnv, forTerminal, HostStop, shellQuote, sleepUntil, UsageError } from "./utils.ts";

/**
 * The options `inboxChat` reads, for `yian chat`'s assistant and the chats `runOrChat` hands over; an
 * absent option is `undefined`, an absent repeatable one an empty list, an absent flag `false`.
 */
export interface InboxChatOptions {
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
 * signals, the model builder AGENT nodes use, and the `advance` each turn after the first calls.
 */
export interface InboxChatDeps {
  scheduler?: Scheduler;
  signals?: Pick<NodeJS.EventEmitter, "on" | "off">;
  models?: ModelAccess | null;
  advance?: typeof advance;
}

/**
 * What a host running a flow through the chat gives it beside the command line, as `yian chat`
 * runs its bundled assistant: the flow itself, inputs, skills, tools, child events and a run id.
 */
export interface InboxChatHost {
  /** The flow at `flowPath`, already loaded; read from the file when absent. */
  flow?: ComposedFlow;
  /** Inputs the host supplies itself, beside those given on the command line; never prompted for. */
  inputs?: Readonly<Record<string, unknown>>;
  /** The skills the run gets, instead of those read from the skill folders. */
  skills?: ReadonlyMap<string, Skill> | null;
  /** The run's id; a fresh one when absent. */
  runId?: string;
  /** Host tools added to the registered ones; a function gets the chat's terminal, to ask through. */
  tools?: Readonly<Record<string, ToolEntry>> | ((terminal: Terminal) => Readonly<Record<string, ToolEntry>>);
  /** The child events the chat sends into the inbox. */
  events?: ChatEvents;
  /** How the chat draws itself on a terminal (`yian chat`'s look); its plain lines when null or absent. */
  view?: InboxChatView | null;
  /**
   * With `view`, the rows listed under the chat's input box (the runs a host started, say), and
   * what Enter on one does: `open` shows it, through the chat's terminal, until the user leaves.
   */
  rows?: { list: PromptList; open(index: number, terminal: Terminal): Promise<void> } | null;
}

/**
 * What a host drawing the chat's look gives it (`ChatView`): the chat hands it the run's events
 * and its stream, its turns, the times a prompt has the terminal, the reply and the child events
 * sent, and the input box it keeps up while a turn runs; and frames its prompt.
 */
export interface InboxChatView {
  /** The hint under the framed prompt. */
  readonly hint: string;
  onEvent(event: RunEvent): void;
  onStream(chunk: StreamChunk): void;
  /** A turn starts (`true`) or ends. */
  turn(active: boolean): void;
  /** A prompt takes the terminal (`true`) or gives it back, having left rows on it when `printed`. */
  active(active: boolean, printed: boolean): void;
  /** The output of a node named `reply`, as text. */
  reply(text: string): void;
  /** A child event, once sent. */
  childEvent(event: ChildEvent): void;
  /** `text`, whole lines of the chat's own notes, styled. */
  note(text: string): string;
  /** Keep `input` under the spinner while a turn runs, or none (`null`). */
  input(input: LineEditor | null): void;
  /** Draw the input box again, once a key changed it. */
  refresh(): void;
  /** A line sent from the input box while a turn runs, printed as the prompt leaves one. */
  echo(line: string): void;
}

const INTERRUPTING = "(interrupting the reply; Ctrl-C again within 2 s to cancel the run)";
// A second Ctrl-C this soon after the one that interrupted a turn cancels the run.
const SECOND_PRESS_MS = 2_000;
const EXIT_LINES = new Set(["/exit", "/quit"]);

/** `yian run`'s options, and the inbox a chat it hands over sends lines to. */
export type RunOrChatOptions = RunOptions & { inbox?: string | undefined };

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
 * Whether the chat provides all `flow` needs: it declares no memories and needs no MCP tools.
 * Never throws.
 */
function chattable(flow: ComposedFlow): boolean {
  try {
    return !resolveMemoryLayout(flow.compiled).enabled && !needsMcp(flow);
  } catch {
    // A memory layout that doesn't resolve is left to `yian run` to report.
    return false;
  }
}

/** Return `flow`'s `Inbox(str)` and `Inbox(Message)` inputs, in declaration order. */
function lineInboxes(flow: ComposedFlow): Omit<LineInbox, "several">[] {
  return flow.input.flatMap((param): Omit<LineInbox, "several">[] => {
    const type = param.contract?.type ?? null;
    if (!isInboxType(type)) return [];
    if (canonicalTypeLabel(type.element) === "str") return [{ name: param.name, message: false }];
    return typeEquals(type.element, builtinType("Message")!) ? [{ name: param.name, message: true }] : [];
  });
}

/**
 * Return the `Inbox(str)` or `Inbox(Message)` input lines go to: `chosen`, or the flow's only one.
 *
 * @throws {UsageError} When the flow has none, several and `chosen` is absent, or `chosen` isn't
 *   one of them.
 */
function pickInbox(flow: ComposedFlow, chosen: string | undefined): LineInbox {
  const inboxes = lineInboxes(flow);
  const several = inboxes.length > 1;
  const names = inboxes.map((inbox) => inbox.name);
  const id = flow.compiled.flowId ?? flow.name;
  if (names.length === 0) {
    throw new UsageError(`flow '${id}' has no Inbox(str) or Inbox(Message) input; the chat sends each line to one`);
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

/** The seq of the last `run_waiting` record of the log at `path`, or -1 when there is none. */
function lastWaitingSeq(path: string): number {
  let last = -1;
  for (const entry of iterRunRecords(path, { kinds: ["run_waiting"], payloads: false })) last = entry.seq;
  return last;
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
 * Return what a stopped chat prints: where its run is, and the `yian run --resume` that continues
 * it, naming the inbox when the flow has several (the log doesn't record which one the chat used).
 */
function stoppedText(log: string, inbox: LineInbox): string {
  const named = inbox.several ? ` --inbox ${shellQuote(inbox.name)}` : "";
  return `chat stopped; the run is in ${log}\ncontinue with: yian run --resume ${shellQuote(log)}${named}\n`;
}

/** Whether the run log at `log` records that the top-level inbox `inbox` was closed. */
function inboxClosed(log: string, inbox: string): boolean {
  for (const entry of iterRunRecords(log, { kinds: ["inbox_closed"] })) {
    if ((entry.envelope.record as InboxClosedRecord).inbox === inbox) return true;
  }
  return false;
}

/**
 * Chat with the flow at `flowPath`: start it, send each line typed at the `You:` prompt to its
 * one `Inbox(str)` or `Inbox(Message)` input (or `options.inbox`; a `Message` inbox gets the line
 * as a user message holding it as text), and advance the run after each send, printing the
 * output of each node named `reply` on stdout as it finishes, through `sanitize`; returns the exit
 * code.
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
 * for the next prompt (unless the box stays up, below). SIGTERM, SIGHUP or
 * `io.stop` stops the run and leaves its log (`chat stopped; the run is in <log>` and the
 * `yian run --resume <log>` that continues it, exit 130), as does Ctrl-C at an `ask` prompt,
 * which ignores Esc. While the prompt is up, the run is advanced
 * at its `wakeAt`; what was typed so far comes back at the next prompt. Off a terminal (stdin or
 * stderr isn't a TTY), each prompt is written once, with no escape codes, and Esc isn't seen. An
 * `ask` node prompts at the terminal. A line the inbox refuses prints `not sent: <reason>` and the
 * chat goes on. A failed run prints its diagnostic and exits 1; a
 * run that ends waiting after the inbox closed, with no deadline to wake it, prints what it waits
 * on and exits 0.
 *
 * With `host.view`, the chat draws its look through it (`InboxChatView`): it gets the run's events,
 * its stream and the reply instead of stdout, the turns, the child events sent and the times a
 * prompt has the terminal; the chat's own notes are styled by it, and the prompt is framed with its
 * hint. On a terminal the framed box then stays up while a turn runs, under the spinner, holding
 * what was typed: Enter there sends the line to the inbox at once (an agent reading it as its
 * steer inbox reads it before its next model call; `/exit` or `/quit` closes the inbox, a blank
 * line is not sent), and the keys are kept for no later prompt. Ctrl-C and Esc keep their
 * meanings there; a prompt the run shows (an ask, an approval) takes the terminal from the box,
 * which comes back after it. A line sent once the engine stopped moves the run again, as a child
 * event does. With `host.rows`, their list is drawn under the box's hint: Down on the box's last
 * row moves into it, Up and Down move along it, Up from its top row or Esc goes back to the box
 * (an Esc there withdraws nothing), and Enter on a row calls `host.rows.open`, which the chat
 * doesn't wait for: its turns, deadlines and child events go on while the row is open, and the
 * next prompt waits behind it.
 *
 * With `host.events`, each child event queued is sent to the inbox as a message authored `yian`
 * (as text to an `Inbox(str)`): while a turn runs, at once, unless a tool call is in flight, then
 * when the last one ends; while the prompt is up, the prompt is taken down, the events are sent
 * in a turn of their own and what was typed comes back. A turn that leaves the run waiting with an
 * event sent after its engine stopped is advanced again. Ctrl-C or Esc during a turn calls
 * `events.onInterrupt`; it, or Esc at the prompt, holds the queue until the next line, which the
 * queued events go before, and each event held is announced above the prompt (`(child event
 * waiting; sent with your next message: <event>)`). Esc puts the events it takes back (messages
 * authored `yian`; to an `Inbox(str)`, the chat's own sends) at the front of the queue, not in the
 * prompt. The events still queued when the inbox closes or the chat ends are printed (`(not sent:
 * <event>)`).
 *
 * The flow loads, inputs, provider keys, log path and run index are handled as `yian run`'s
 * (`--engine-trace` included), except that the inbox input is never asked for or supplied and
 * `host.inputs` are supplied beside the given ones. Skills are `host.skills`, or else read as `yian
 * run` reads them, once per chat, before any input prompt. The chat opens no MCP servers and builds
 * no memories. A host failure prints `chat failed: <message>`, its stack below under
 * `engineTrace`, and exits 1.
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
 * @param host What a host running a flow through the chat supplies (`InboxChatHost`).
 * @throws {UsageError} As `validateRunLogOptions` and `mergeInputs` do; for a flow without exactly
 *   one `Inbox(str)` or `Inbox(Message)` input to pick; for an input the flow doesn't declare
 *   (the inbox included); off a TTY, for missing required inputs.
 */
export async function inboxChat(
  flowPath: string | undefined,
  options: InboxChatOptions,
  io: CliIo,
  deps: InboxChatDeps = {},
  host: InboxChatHost = {},
): Promise<number> {
  validateRunLogOptions({ ...options, flow: flowPath });
  const resumed = options.resume ?? null;
  const flow =
    resumed === null
      ? (host.flow ?? (await loadFlowFile(flowPath!, io, { engineTrace: options.engineTrace })))
      : await recordedFlow(resumed, options.engineTrace, io);
  if (flow === null || flow instanceof Diagnostic) return 1;
  const inbox = pickInbox(flow, options.inbox);
  const supplied = { ...mergeInputs(options.inputs, options.inputValues), ...host.inputs };
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
  // The sends of lines typed in the input box while a turn ran, in order, and their seqs.
  let liveSends: Promise<void> = Promise.resolve();
  const liveSeqs: number[] = [];
  // Set once `/exit` or `/quit` was entered in the box: it takes no more keys.
  let exiting = false;
  const move = deps.advance ?? advance;
  const view = host.view ?? null;
  // The chat's own notes, styled when it draws its look.
  const note = (text: string): Promise<void> => io.stderr.write(view === null ? text : view.note(text));
  // Sends the child events, once the run's log exists (`started`).
  const flusher =
    host.events === undefined
      ? null
      : new ChatFlusher(host.events, {
          message: inbox.message,
          send: (value) => send(log!, inbox.name, value, { env: env.read }),
          write: note,
          fault: (error) => stop.abort(error),
          sent: view === null ? undefined : (event) => view.childEvent(event),
        });
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
    if (turning) flusher?.interrupted(true);
    // A press while a turn runs interrupts its reply; a second within 2 s, or one at the prompt,
    // cancels.
    if (turning && (pressedAt === null || now - pressedAt > SECOND_PRESS_MS)) {
      pressedAt = now;
      void note(`${INTERRUPTING}\n`).catch(() => {});
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
    flusher?.interrupted(turning);
    const done = (async () => {
      // A line entered in the box just before is in the log first, so Esc takes it back too.
      await liveSends;
      let back = await withdraw(path, inbox.name, { env: env.read });
      // What was taken back is no longer for the run to read: it moves the run no more.
      const gone = new Set(back.map((event) => event.seq));
      liveSeqs.splice(0, liveSeqs.length, ...liveSeqs.filter((seq) => !gone.has(seq)));
      // The child events among them go back to the queue (or, after /exit, are printed).
      if (flusher !== null) back = await flusher.reclaim(back, closed);
      // Kept before the interrupt, which may be refused.
      if (back.length > 0) {
        const texts = back.map((event) => valueText(event.value, inbox.message)).filter(Boolean);
        const count = `${back.length} message${back.length === 1 ? "" : "s"}`;
        // After /exit no prompt comes back, so the text is printed rather than lost.
        if (closed) await note(`(withdrew ${count}: ${texts.join(" ")})\n`);
        else {
          // Sent before the draft was typed, so it goes first. While a turn runs, the draft is
          // what the input box holds.
          const box = turning ? editor : null;
          draft = [...texts, box?.text() ?? draft].filter(Boolean).join(" ");
          if (box !== null) {
            box.set(draft);
            view?.refresh();
          }
          await note(`(withdrew ${count}; back in the prompt)\n`);
        }
      }
      const stopped = await interruptReply(path, inbox.name, { env: env.read });
      if (stopped.seq !== null) await note("(stopping the reply)\n");
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
    onActive: view === null ? undefined : (active, printed) => view.active(active, printed),
  });
  // With the chat's look on a terminal, the prompt's box stays up while a turn runs.
  const editor =
    view !== null && atTerminal(io) ? terminal.lineEditor({ hint: view.hint, list: host.rows?.list }) : null;
  // Opens a row of `host.rows` without waiting for the user to leave it: the chat goes on meanwhile
  // (its prompt queues behind the row's screen), and only a failure stops it.
  const openRow = (index: number): void => {
    host.rows?.open(index, terminal).catch((error: unknown) => stop.abort(error));
  };
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
    const skills = host.skills !== undefined ? host.skills : await readGrantedSkills(flow, env.read, io);
    const prompter = io.stdin.isTTY === true ? terminal : null;
    const cliCfg = cliLayer(options);
    let path: string;
    // A fresh chat's run id; a resumed one keeps the id its log recorded.
    let runId: string | null = null;
    if (resumed === null) {
      const gathered = await gatherMissingInputs(flow.input, supplied, prompter);
      stop.signal.throwIfAborted();
      if (gathered === null) {
        await note("chat cancelled\n");
        return 1;
      }
      Object.assign(supplied, gathered);
      const keyRefusal = await preflightKeys(flow, cliCfg, supplied, io, {
        env,
        terminal: prompter,
        signal: stop.signal,
      });
      if (keyRefusal !== null) return 1;
      stop.signal.throwIfAborted();
      const fresh = freshLog(options.log, io.env);
      runId = host.runId ?? fresh.runId;
      path = fresh.path;
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
      if (event instanceof RunStarted) flusher?.started();
      view?.onEvent(event);
      if (!(event instanceof NodeSucceeded)) return;
      if (event.nodeId.slice(event.nodeId.lastIndexOf("/") + 1) !== "reply") return;
      // An interrupted reply succeeds with null when its output is `Optional(...)`.
      if (event.interrupted) return note("(interrupted)\n");
      if (event.output === null) return;
      const text = writeAsText(event.output as Value);
      if (view !== null) return view.reply(text);
      // A reply is the model's or the flow's text: on a terminal its escape sequences are dropped and
      // its control characters shown; piped, it keeps every byte.
      return io.stdout.write(`${io.stdout.isTTY === true ? sanitize(text) : text}\n`);
    };
    const shared = {
      onEvent,
      onStream: view === null ? null : (chunk: StreamChunk) => view.onStream(chunk),
      tools: {
        ...registeredTools(),
        ...(typeof host.tools === "function" ? host.tools(terminal) : host.tools),
        ask: terminal.askUser,
      },
      models: deps.models,
      skills,
      env: env.read,
      signal: stop.signal,
      scheduler,
    };
    /**
     * Send `line`, entered in the input box while a turn runs, after the lines entered before it:
     * to the inbox at once (an agent reading it as its steer inbox reads it), or, for `/exit` or
     * `/quit`, close the inbox as the prompt does. A blank line isn't sent.
     */
    const sendLive = (line: string): void => {
      if (line.trim() === "") return;
      view?.echo(line);
      const exit = EXIT_LINES.has(line.trim());
      // No line comes after it: the box goes.
      if (exit) {
        exiting = true;
        view?.input(null);
      }
      else pressedAt = null;
      liveSends = liveSends
        .then(async () => {
          if (exit) {
            if (closed) return;
            closed = true;
            await flusher?.printLeft();
            flusher?.close();
            await close(path, inbox.name, { env: env.read });
            return;
          }
          // The hold ends with this line: held events, and those withdrawn, go before it.
          await flusher?.lineSent();
          liveSeqs.push(await send(path, inbox.name, lineEvent(line, inbox.message), { env: env.read }));
        })
        .catch(async (error: unknown) => {
          if (!(error instanceof DeliveryRefused)) return stop.abort(error);
          if (error.reason !== "ended") await note(`not sent: ${forTerminal(error.message, io.stderr)}\n`);
        });
    };
    /** A key at the input box while a turn runs; false for one it leaves to `watchKeys` (Esc). */
    const liveKey = (key: Key): boolean => {
      // Once the inbox is closing, no prompt comes back: the keys go to no box.
      if (closed || exiting) return false;
      const action = editor!.key(key);
      if (action !== null && "escape" in action) return false;
      if (action !== null && "value" in action) {
        editor!.set("");
        sendLive(action.value);
      } else if (action !== null && "open" in action) {
        openRow(action.open);
      }
      view?.refresh();
      return true;
    };
    /**
     * Run `move` as a turn: a Ctrl-C meanwhile interrupts the reply instead of cancelling, and on a
     * terminal, Esc withdraws and stops it (`escape`). With the chat's look, the input box stays
     * up under the spinner, its lines sent at once (`sendLive`).
     */
    const turn = async (step: () => Promise<RunResult>): Promise<RunResult> => {
      turning = true;
      const keys = new AbortController();
      const live = editor !== null && !closed && !exiting;
      if (live) editor.set(draft);
      const onKey = (key: "escape" | "interrupt") => (key === "escape" ? escape() : interrupt());
      terminal.watchKeys(keys.signal, onKey, live ? liveKey : null);
      flusher?.beginTurn();
      view?.turn(true);
      if (live) view?.input(editor);
      try {
        return await step();
      } finally {
        view?.input(null);
        view?.turn(false);
        keys.abort();
        if (live) draft = editor.text();
        turning = false;
        // An interrupt sent during the turn lands before the next line is sent, and so do the
        // lines sent during it.
        await interrupting;
        interrupting = null;
        await liveSends;
      }
    };
    /**
     * Run `first` as a turn, then, while it leaves the run waiting with an event it sent unread,
     * advance it again: a send made once the engine's loop had ended is appended to the log after
     * its last `run_waiting`, and only a move reads it.
     */
    const settle = async (first: () => Promise<RunResult>): Promise<RunResult> => {
      let settled = await turn(first);
      for (;;) {
        const seqs = flusher === null ? [] : await flusher.endTurn();
        // The lines sent from the input box during the turn, which may be stranded the same way.
        const lines = liveSeqs.splice(0);
        if (settled.status !== "waiting") break;
        const tracked = flusher?.tracking(seqs) ?? false;
        if (!tracked && lines.length === 0) break;
        // Read once per turn, for both the stranded sends and the ones the run has read.
        const last = lastWaitingSeq(path);
        const events = tracked && flusher!.afterTurn(seqs, last);
        if (!events && !lines.some((seq) => seq > last)) break;
        settled = await turn(() => move({ log: path, checkLlmConfigs, ...shared }));
      }
      return settled;
    };
    const moveOn = () => settle(() => move({ log: path, checkLlmConfigs, ...shared }));
    let result: RunResult = await settle(() =>
      resumed === null
        ? runFlow(flow, supplied, {
            log: path,
            runId,
            llmConfig: Object.keys(cliCfg).length > 0 ? cliCfg : null,
            numWorkers: options.numWorkers,
            ...shared,
          })
        : move({ log: path, numWorkers: options.numWorkers ?? null, checkLlmConfigs, ...shared }),
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
      if (flusher?.ready()) {
        // Events queued while the prompt was down, or that took it down: a turn of their own.
        await flusher.sendQueued();
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
      await flusher?.announce();
      const prompt = new AbortController();
      line = prompt;
      flusher?.promptUp(() => prompt.abort());
      const timer = new AbortController();
      if (wake !== null) {
        sleepUntil(scheduler, recordTimeMs(wake, "wakeAt"), timer.signal).then(
          () => prompt.abort(),
          () => {},
        );
      }
      if (stop.signal.aborted) prompt.abort();
      const framing = view === null ? {} : { frame: true, hint: view.hint };
      const read = await terminal.readLine("You:", {
        signal: prompt.signal,
        initial: draft,
        escape: true,
        ...framing,
        ...(editor === null ? {} : { editor }),
      });
      draft = "aborted" in read || "escape" in read || "open" in read ? read.draft : "";
      timer.abort();
      line = null;
      const flushed = flusher?.promptDown() ?? false;
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
        // The flusher took the prompt down, to send queued events or announce a held one: the loop
        // does it.
        if (flushed) continue;
        // A deadline passed.
        spent = wake;
        result = await moveOn();
        continue;
      }
      if ("interrupt" in read) {
        interrupt();
        continue;
      }
      if ("open" in read) {
        openRow(read.open);
        continue;
      }
      // A run that ended under the chat (a cancel from outside) refuses the line or the close as
      // `ended`; the move below then reads how it ended, and the loop stops.
      if ("eof" in read || EXIT_LINES.has(read.value.trim())) {
        // No turn comes to send them in: they are printed instead.
        await flusher?.printLeft();
        try {
          await close(path, inbox.name, { env: env.read });
        } catch (error) {
          if (!(error instanceof DeliveryRefused && error.reason === "ended")) throw error;
        }
        closed = true;
        flusher?.close();
      } else {
        pressedAt = null;
        // The hold ends with this line: held events, and those withdrawn, go before it.
        await flusher?.lineSent();
        try {
          await send(path, inbox.name, lineEvent(read.value, inbox.message), { env: env.read });
        } catch (error) {
          if (!(error instanceof DeliveryRefused)) throw error;
          if (error.reason !== "ended") {
            await note(`not sent: ${forTerminal(error.message, io.stderr)}\n`);
            continue;
          }
        }
      }
      result = await moveOn();
    }
    // A cancel that lost to the run's end is refused; its other failures are the chat's.
    if (cancelling !== null) await cancelling;
    if (cancelled || result.status === "cancelled") {
      await note("chat cancelled\n");
      return 130;
    }
    if (result.status === "succeeded") {
      await note("session ended\n");
      return 0;
    }
    await renderRunError(result, flow, options.engineTrace, io.stderr, io.env);
    return 1;
  } catch (error) {
    if (error instanceof HostStop || (error instanceof TerminalStopped && error.reason === "eof")) {
      await note(log === null ? "chat cancelled\n" : stoppedText(log, inbox));
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
    flusher?.dispose();
    // Pushed after /exit, or held when the run was cancelled: printed, as /exit prints them.
    await flusher?.printLeft();
    stop.signal.removeEventListener("abort", takeDown);
    terminal.close();
    for (const [name, listener] of listeners) signals.off(name, listener);
    outside?.removeEventListener("abort", forwardStop);
    if (log !== null) await indexRun(log, io);
    if (runDir !== null) removeEmptyDir(runDir);
  }
}

/**
 * `yian run`: chat with the flow at `flowPath` (`inboxChat`, given `deps`) when stdin and stderr
 * are a TTY, the run is fresh (no `resume`, `json`, `memoryRoot` or `memoryStorage`), and the flow
 * loads, declares no memories, needs no MCP tools, and has an `Inbox(str)` or `Inbox(Message)`
 * input to type into: its only one, or `inbox`. Otherwise hand the command to `run`, which reads
 * the flow again and reports what is wrong with it; checking writes nothing. Returns the exit code.
 *
 * With `resume`, the chat a stop left in that log goes on (`inboxChat` with `resume`) under the
 * same conditions, judged on the flow the log recorded, and when no fresh-run flag is given;
 * a run that has ended, or a log that can't be read, goes to `run` to report.
 *
 * The chat's start-up comes before `run` sets anything up, so the two never share the terminal,
 * the signals or stderr.
 *
 * @throws {UsageError} As `run` or `inboxChat` does; for an `inbox` that isn't one of the
 *   `Inbox(str)` or `Inbox(Message)` inputs of a flow the chat could otherwise take.
 */
export async function runOrChat(
  flowPath: string | undefined,
  options: RunOrChatOptions,
  io: CliIo,
  deps: InboxChatDeps = {},
): Promise<number> {
  if (options.resume !== undefined) {
    if (!(await chatsOnResume(flowPath, options, io))) return run(flowPath, options, io);
    const { inbox, numWorkers, engineTrace, resume } = options;
    return inboxChat(undefined, { inbox, inputValues: [], numWorkers, engineTrace, resume }, io, deps);
  }
  const flow = await chatFlow(flowPath, options, io);
  if (flow === null) return run(flowPath, options, io);
  const { inbox, inputValues, inputs, log, numWorkers, provider, model, engineTrace } = options;
  const chatOptions = { inbox, inputValues, inputs, log, numWorkers, provider, model, engineTrace };
  return inboxChat(flowPath!, chatOptions, io, deps, { flow });
}

/**
 * Whether `runOrChat` goes on with the chat recorded at `options.resume`: on a TTY, not under
 * `json`, with no FLOW, fresh-run or memory flag, for a run that hasn't ended whose recorded flow
 * the chat could take. Writes nothing.
 */
async function chatsOnResume(flowPath: string | undefined, options: RunOrChatOptions, io: CliIo): Promise<boolean> {
  if (flowPath !== undefined || options.json || !atTerminal(io)) return false;
  const fresh = [options.inputs, options.log, options.provider, options.model, options.memoryRoot];
  if (fresh.some((value) => value !== undefined) || options.inputValues.length > 0) return false;
  if (options.memoryStorage.length > 0) return false;
  let flow: ComposedFlow | null;
  try {
    const { status } = inspectRun(options.resume!);
    if (status === "succeeded" || status === "failed" || status === "cancelled") return false;
    flow = await loadRecordedFlow(options.resume!, { env: io.env });
  } catch {
    return false;
  }
  if (flow === null || !chattable(flow)) return false;
  const count = lineInboxes(flow).length;
  return options.inbox !== undefined ? count > 0 : count === 1;
}

/**
 * Return the flow at `flowPath` when `runOrChat` chats with it, else `null`, writing nothing. A
 * flow the chat could take whose `inbox` names none of its inboxes is returned too, so that
 * `inboxChat` refuses the name.
 */
async function chatFlow(
  flowPath: string | undefined,
  options: RunOrChatOptions,
  io: CliIo,
): Promise<ComposedFlow | null> {
  if (flowPath === undefined || options.resume !== undefined || options.json || !atTerminal(io)) return null;
  // `run` checks the memory flags against the flow and reports what they can't take.
  if (options.memoryRoot !== undefined || options.memoryStorage.length > 0) return null;
  const silent: CliIo = { ...io, stderr: { write: async () => {} } };
  let flow: ComposedFlow | Diagnostic;
  try {
    flow = await loadFlowFile(flowPath, silent, { engineTrace: false });
  } catch {
    return null;
  }
  if (flow instanceof Diagnostic || !chattable(flow)) return null;
  const count = lineInboxes(flow).length;
  return (options.inbox !== undefined ? count > 0 : count === 1) ? flow : null;
}
