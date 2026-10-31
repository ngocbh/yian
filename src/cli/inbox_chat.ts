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
import { Diagnostic, shellQuote } from "../diagnostics.ts";
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
  type WaitInfo,
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
import {
  type Draft,
  draftLabel,
  draftText,
  joinDrafts,
  type Key,
  type LineEditor,
  type PromptList,
  Terminal,
  TerminalStopped,
  textDraft,
} from "./terminal.ts";
import { atTerminal, type CliIo, CommandEnv, forTerminal, HostStop, sleepUntil, UsageError } from "./utils.ts";

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
 * signals, the model builder AGENT nodes use, the `advance` each turn after the first calls, and
 * the `send` the lines and child events go through.
 */
export interface InboxChatDeps {
  scheduler?: Scheduler;
  signals?: Pick<NodeJS.EventEmitter, "on" | "off">;
  models?: ModelAccess | null;
  advance?: typeof advance;
  send?: typeof send;
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
   * what Enter on one does: `open` shows the row's item, by its key, through the chat's terminal,
   * until the user leaves.
   */
  rows?: { list: PromptList; open(key: string, terminal: Terminal): Promise<void> } | null;
  /**
   * Runs first, on the chat's terminal, before the inputs are gathered and the run starts (say, to
   * pick what the run goes on from, into `inputs`): an exit code ends the chat with it at once,
   * `null` goes on.
   */
  start?: ((terminal: Terminal) => Promise<number | null>) | null;
  /** The host's own commands, typed as a line at the prompt (`ChatCommands`). */
  commands?: ChatCommands | null;
  /**
   * What a stopped chat prints, given its log: where the run is and how to go on; the
   * `yian run --resume` that continues it when absent.
   */
  stopped?: ((log: string) => string) | null;
  /** Keys typed ahead before the chat (an earlier chat's, through `onTyped`), read first. */
  typed?: readonly Key[];
  /** Given the keys typed ahead that the chat didn't read, as it ends. */
  onTyped?: ((keys: readonly Key[]) => void) | null;
}

/**
 * What a host's command does to the chat: send `send` to the inbox and advance the run, as a
 * line is, calling `sent` once the inbox took it; or end the chat, as `/exit` does; or nothing
 * (`null`).
 */
export type ChatCommandOutcome = { send: unknown; sent?: () => void } | { exit: true } | null;

/**
 * Commands a host adds to the chat's prompt, each a whole line such as `/model`. One entered in
 * the input box while a turn runs waits for the turn to end.
 */
export interface ChatCommands {
  /** The lines that are commands. */
  readonly names: ReadonlySet<string>;
  /** Run the command `name` at the prompt, through the chat's terminal. */
  run(name: string, terminal: Terminal): Promise<ChatCommandOutcome>;
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
  /** A prompt takes the terminal (`true`) or gives it back, having left rows when `printed`. */
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

/**
 * Whether the run has read every line sent to `inbox`, so none can be withdrawn: a take of it
 * waits, unsettled, among `waits`, which it does only once nothing sent is left to read.
 */
export function inboxDrained(waits: readonly WaitInfo[], inbox: string): boolean {
  return waits.some((wait) => wait.tool === "take" && wait.inbox === inbox && !wait.settled);
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
 * run (`chat cancelled`, exit 130). A line's turn starts as the prompt takes it, before it is sent,
 * and ends once the run has settled, so a Ctrl-C typed with the line or right after it counts as
 * one during the turn and stops the reply that line starts. On a terminal, Esc, during a turn or
 * at the prompt, takes back the lines the run hasn't read yet (`withdraw`) and puts their text
 * back in the prompt (`(withdrew N messages; back in the prompt)`; after `/exit` it prints them),
 * then stops the turn (`interrupt`; `(stopping the reply)` when a Stop is recorded): a running
 * reply is cut, and during a sleep or an approval the reply after it stops before its first model
 * call. At the take Esc does nothing; it never cancels. On a terminal the input stays raw from
 * the first turn until the inbox closes, between turns too, so no key typed ahead is lost: Ctrl-C
 * arrives as a key with the same rule, Ctrl-Z and Ctrl-\ raise no signal, and other keys are
 * kept, in order, for the next prompt (unless the box stays up, below). SIGTERM, SIGHUP or
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
 * line is not sent), and the keys are kept for no later prompt: those typed before the turn
 * started go to the box first, in order. Ctrl-C and Esc keep their
 * meanings there; a prompt the run shows (an ask, an approval) takes the terminal from the box,
 * which comes back after it. A line sent once the engine stopped moves the run again, as a child
 * event does. A long paste is a token in the box (`LineEditor`): the inbox gets the text it
 * stands for, the line echoed shows the token, and Esc puts a line it withdraws back with its
 * tokens. With `host.rows`, their list is drawn under the box's hint: Down on the box's last
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
 * With `host.commands`, a line that names one runs it at the prompt and does what it returns:
 * sends its value and moves the run on, as a line; closes the inbox, as `/exit`; or nothing. One
 * entered in the box during a turn waits for the turn's end. `host.start` runs on the terminal
 * before anything else and may end the chat with an exit code; `host.typed` are keys typed ahead
 * for the first prompts, and `host.onTyped` gets those left unread as the chat ends. A command's
 * send that Esc withdraws is sent again, as it isn't a typed line; `host.stopped` gives the text
 * a stopped chat prints.
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
  let draft: Draft = [];
  // The lines sent from the framed box, by seq, as the box held them: Esc puts one it withdraws
  // back as it was, its tokens and all.
  const sent = new Map<number, Draft>();
  // Set once the inbox is closed (`/exit`, `/quit`, the end of input): no prompt comes back.
  let closed = false;
  // The sends of lines typed in the input box while a turn ran, in order, and their seqs.
  let liveSends: Promise<void> = Promise.resolve();
  const liveSeqs: number[] = [];
  // Set once `/exit` or `/quit` was entered in the box: it takes no more keys.
  let exiting = false;
  // A host command entered in the box while a turn ran, run once the turn has ended.
  let heldCommand: string | null = null;
  // What host commands sent, by seq: Esc never takes one back, as it isn't a line typed.
  const commandSends = new Map<number, unknown>();
  // The send of the line the prompt took, while it runs (its turn has started): a Ctrl-C or Esc
  // meanwhile acts once the line is in the log, so it stops the reply that line starts.
  let lineSending: Promise<void> | null = null;
  // Set while a turn ends: the box sends no more lines, and keys typed into it wait for the prompt.
  let ending = false;
  // Ends the terminal's raw hold between turns, taken while lines can still be typed.
  let releaseRaw = (): void => {};
  const move = deps.advance ?? advance;
  const deliver = deps.send ?? send;
  const view = host.view ?? null;
  // The chat's own notes, styled when it draws its look.
  const note = (text: string): Promise<void> => io.stderr.write(view === null ? text : view.note(text));
  // Sends the child events, once the run's log exists (`started`).
  const flusher =
    host.events === undefined
      ? null
      : new ChatFlusher(host.events, {
          message: inbox.message,
          send: (value) => deliver(log!, inbox.name, value, { env: env.read }),
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
  /**
   * Hold the child events for the next line (`ChatFlusher.interrupted`), once the line being sent
   * is in the log: its send ends the hold.
   */
  const hold = (during: boolean): void => {
    if (lineSending === null) flusher?.interrupted(during);
    else void lineSending.then(() => flusher?.interrupted(during));
  };
  const interrupt = (): void => {
    if (log === null) {
      stop.abort(new HostStop("SIGINT"));
      return;
    }
    const path = log;
    const now = scheduler.now();
    if (turning) hold(true);
    // A press while a turn runs interrupts its reply; a second within 2 s, or one at the prompt,
    // cancels.
    if (turning && (pressedAt === null || now - pressedAt > SECOND_PRESS_MS)) {
      pressedAt = now;
      void note(`${INTERRUPTING}\n`).catch(() => {});
      const landing = lineSending;
      // The live engine takes it at once. With no reply running it records the Stop for the turn,
      // or nothing when the chat waits at the take.
      const sent = (async () => {
        if (landing !== null) await landing;
        await interruptReply(path, inbox.name, { env: env.read });
      })().catch((error: unknown) => {
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
    hold(turning);
    const done = (async () => {
      // The line the prompt took, or one entered in the box just before, is in the log first, so
      // Esc takes it back too.
      await liveSends;
      // The lines a withdraw may take back; one sent meanwhile is kept for the next.
      const known = [...sent.keys()];
      let back = await withdraw(path, inbox.name, { env: env.read });
      // A host command's send isn't a line to take back: it goes to the run again, as it was.
      const commands = back.filter((event) => commandSends.has(event.seq));
      back = back.filter((event) => !commandSends.has(event.seq));
      for (const event of commands) {
        const value = commandSends.get(event.seq);
        commandSends.delete(event.seq);
        try {
          commandSends.set(await deliver(path, inbox.name, value, { env: env.read }), value);
        } catch (error) {
          // Once the inbox is closed, nothing more reaches the run.
          if (!(error instanceof DeliveryRefused)) throw error;
        }
      }
      // What was taken back is no longer for the run to read: it moves the run no more.
      const gone = new Set(back.map((event) => event.seq));
      liveSeqs.splice(0, liveSeqs.length, ...liveSeqs.filter((seq) => !gone.has(seq)));
      // The child events among them go back to the queue (or, after /exit, are printed).
      if (flusher !== null) back = await flusher.reclaim(back, closed);
      // Kept before the interrupt, which may be refused.
      if (back.length > 0) {
        // A line sent from the box comes back as it held it; one sent before a resume, as a
        // paste would (a long one as a token).
        const drafts = back.map((event) => {
          const text = valueText(event.value, inbox.message);
          return sent.get(event.seq) ?? (editor === null || closed ? [text] : textDraft(text));
        });
        const count = `${back.length} message${back.length === 1 ? "" : "s"}`;
        // After /exit no prompt comes back, so the text is printed rather than lost.
        if (closed) await note(`(withdrew ${count}: ${draftLabel(joinDrafts(drafts))})\n`);
        else {
          // Sent before the draft was typed, so it goes first. While a turn runs, the draft is
          // what the input box holds.
          const box = turning ? editor : null;
          draft = joinDrafts([...drafts, box?.draft() ?? draft]);
          if (box !== null) {
            box.set(draft);
            view?.refresh();
          }
          await note(`(withdrew ${count}; back in the prompt)\n`);
        }
      }
      // The run has read every line it didn't give back, so none of those can come back.
      for (const seq of known) sent.delete(seq);
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
    noColor: Boolean(io.env("NO_COLOR")),
    typed: host.typed,
  });
  // With the chat's look on a terminal, the prompt's box stays up while a turn runs.
  const editor =
    view !== null && atTerminal(io)
      ? terminal.lineEditor({ hint: view.hint, list: host.rows?.list })
      : null;
  // Opens a row of `host.rows` without waiting for the user to leave it: the chat goes on meanwhile
  // (its prompt queues behind the row's screen), and only a failure stops it.
  const openRow = (key: string): void => {
    host.rows?.open(key, terminal).catch((error: unknown) => stop.abort(error));
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
    if (host.start) {
      const code = await host.start(terminal);
      if (code !== null) return code;
      // What the host set up for the run.
      Object.assign(supplied, host.inputs);
    }
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
    // The box is up from the start, so the first prompt and the first turn draw over the same box.
    if (editor !== null && !closed) view?.input(editor);
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
     * Send `entered`, entered in the input box while a turn runs, after the lines entered before
     * it: to the inbox at once (an agent reading it as its steer inbox reads it), its tokens
     * expanded, or, for `/exit` or `/quit`, close the inbox as the prompt does. A blank line isn't
     * sent. The echo shows the tokens.
     */
    const sendLive = (entered: Draft): void => {
      const line = draftText(entered);
      if (line.trim() === "") return;
      view?.echo(draftLabel(entered));
      if (host.commands?.names.has(line.trim())) {
        heldCommand = line.trim();
        void note(`(${heldCommand} once the reply ends)\n`).catch(() => {});
        return;
      }
      const exit = EXIT_LINES.has(line.trim());
      // No line comes after it: the box goes.
      if (exit) {
        exiting = true;
        view?.input(null);
        releaseRaw();
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
          const event = lineEvent(line, inbox.message);
          const seq = await deliver(path, inbox.name, event, { env: env.read });
          liveSeqs.push(seq);
          sent.set(seq, entered);
        })
        .catch(async (error: unknown) => {
          if (!(error instanceof DeliveryRefused)) return stop.abort(error);
          if (error.reason === "ended") return;
          await note(`not sent: ${forTerminal(error.message, io.stderr)}\n`);
        });
    };
    /** A key at the input box while a turn runs; false for one it leaves to `watchKeys` (Esc). */
    const liveKey = (key: Key): boolean => {
      // Once the inbox is closing, no prompt comes back: the keys go to no box. Once the turn is
      // ending, they wait for the prompt, which sends a line entered then.
      if (closed || exiting || ending) return false;
      const action = editor!.key(key);
      if (action !== null && "escape" in action) return false;
      if (action !== null && "value" in action) {
        const entered = editor!.draft();
        editor!.set("");
        sendLive(entered);
      } else if (action !== null && "open" in action) {
        openRow(action.open);
      }
      view?.refresh();
      return true;
    };
    /**
     * Wait until the interrupts and the box lines sent so far have landed, and those made
     * meanwhile, then clear `interrupting`.
     */
    const land = async (): Promise<void> => {
      for (;;) {
        const landing = [interrupting, liveSends] as const;
        await Promise.all(landing);
        if (landing[0] === interrupting && landing[1] === liveSends) break;
      }
      interrupting = null;
    };
    /**
     * Run `body` as a turn: a Ctrl-C meanwhile interrupts the reply instead of cancelling, and on a
     * terminal, Esc withdraws and stops it (`escape`). With the chat's look, the input box stays
     * up under the spinner, its lines sent at once (`sendLive`). The keys typed since the last
     * prompt ended count as typed in the turn. It ends once `body` has, and the interrupts and the
     * lines sent during it have landed, so the next line goes after them.
     */
    const inTurn = async <T>(body: () => Promise<T>): Promise<T> => {
      turning = true;
      const keys = new AbortController();
      const live = editor !== null && !closed && !exiting;
      if (live) editor.set(draft);
      view?.input(live ? editor : null);
      view?.turn(true);
      const onKey = (key: "escape" | "interrupt") => (key === "escape" ? escape() : interrupt());
      try {
        // Last: it hands the keys typed since the prompt ended to the turn at once.
        terminal.watchKeys(keys.signal, onKey, live ? liveKey : null);
        return await body();
      } finally {
        // The box stays up as the turn ends, where the prompt coming next draws the same box.
        view?.turn(false);
        ending = true;
        // Still watched, so a Ctrl-C while they land interrupts rather than cancels.
        await land();
        keys.abort();
        if (live) draft = editor.draft();
        ending = false;
        turning = false;
      }
    };
    /** Run `step` as a move of the turn: the flusher sends child events at once meanwhile. */
    const moveStep = (step: () => Promise<RunResult>): Promise<RunResult> => {
      flusher?.beginTurn();
      return step();
    };
    /**
     * Run `first` as a move, then, while it leaves the run waiting with an event it sent unread,
     * advance it again: a send made once the engine's loop had ended is appended to the log after
     * its last `run_waiting`, and only a move reads it. Runs inside a turn (`inTurn`), so a press
     * between the moves counts as one during them.
     */
    const settle = async (first: () => Promise<RunResult>): Promise<RunResult> => {
      let settled = await moveStep(first);
      for (;;) {
        const seqs = flusher === null ? [] : await flusher.endTurn();
        // A box line still being sent lands after the move's last `run_waiting`; reading the seqs
        // before it lands would leave it unread behind the prompt. An interrupt lands before the
        // next move, as its Stop is for the reply that move would give.
        await land();
        // The lines sent from the input box during the turn, which may be stranded the same way.
        const lines = liveSeqs.splice(0);
        if (settled.status !== "waiting") break;
        const tracked = flusher?.tracking(seqs) ?? false;
        if (!tracked && lines.length === 0) break;
        // Read once per turn, for both the stranded sends and the ones the run has read.
        const last = lastWaitingSeq(path);
        const events = tracked && flusher!.afterTurn(seqs, last);
        if (!events && !lines.some((seq) => seq > last)) break;
        settled = await moveStep(() => move({ log: path, checkLlmConfigs, ...shared }));
      }
      return settled;
    };
    /**
     * Make `sending` the send the next turn starts with: a press or Esc meanwhile acts once it has
     * landed, and the box lines sent meanwhile go after it. Called before the turn starts, as the
     * keys typed since the prompt ended reach the turn as it starts.
     */
    const startsWith = (sending: Promise<unknown>): void => {
      // Settled however the send ends, for the presses and box lines that wait for it.
      const landed = sending.then(
        () => {},
        () => {},
      );
      lineSending = landed;
      liveSends = liveSends.then(() => landed);
    };
    const advanceRun = () => settle(() => move({ log: path, checkLlmConfigs, ...shared }));
    const moveOn = () => inTurn(advanceRun);
    /**
     * Close the inbox, as `/exit` does, and move the run on. A run that ended under the chat (a
     * cancel from outside) refuses the close as `ended`; the move then reads how it ended.
     */
    const leave = async (): Promise<RunResult> => {
      releaseRaw();
      // No turn comes to send them in: they are printed instead.
      await flusher?.printLeft();
      try {
        await close(path, inbox.name, { env: env.read });
      } catch (error) {
        if (!(error instanceof DeliveryRefused && error.reason === "ended")) throw error;
      }
      closed = true;
      flusher?.close();
      // No line comes after it: the box goes.
      view?.input(null);
      return moveOn();
    };
    /**
     * Send `event`, a line's or a command's, and move the run on in a turn; `typed` is what the
     * box held, which Esc puts back. Returns null when the inbox refused it and nothing moved.
     */
    const sendEvent = async (
      event: unknown,
      typed: Draft | null,
      command?: () => void,
    ): Promise<RunResult | null> => {
      pressedAt = null;
      const sending = (async () => {
        // The hold ends with this line: held events, and those withdrawn, go before it.
        await flusher?.lineSent();
        const seq = await deliver(path, inbox.name, event, { env: env.read });
        if (typed !== null) sent.set(seq, typed);
        if (command !== undefined) {
          commandSends.set(seq, event);
          command();
        }
      })();
      startsWith(sending);
      // The turn starts with the send, so a Ctrl-C from here on interrupts rather than cancels.
      return inTurn(async () => {
        try {
          await sending;
        } catch (error) {
          if (!(error instanceof DeliveryRefused)) throw error;
          // A run that ended under the chat refuses it as `ended`; the move reads how it ended.
          if (error.reason !== "ended") {
            await note(`not sent: ${forTerminal(error.message, io.stderr)}\n`);
            // A line entered in the box meanwhile was sent; only a move reads it.
            await land();
            if (liveSeqs.length === 0) return null;
          }
        } finally {
          lineSending = null;
        }
        // A press while the line was sent goes first: its Stop is for the reply the line starts.
        await interrupting;
        return advanceRun();
      });
    };
    /** Run the host's command `name`, then do what it says; null when nothing moved. */
    const runCommand = async (name: string): Promise<RunResult | null> => {
      const outcome = await host.commands!.run(name, terminal);
      stop.signal.throwIfAborted();
      if (outcome === null) return null;
      if ("exit" in outcome) return leave();
      return sendEvent(outcome.send, null, outcome.sent ?? (() => {}));
    };
    // Raw between turns too: a cooked tty drops what is typed past its queue (about 2 KB on macOS)
    // while the flow replies. Released once the inbox closes, so Ctrl-C raises SIGINT again while
    // the chat waits for a deadline.
    if (!closed) releaseRaw = terminal.holdRaw();
    let result: RunResult = await inTurn(() =>
      settle(() =>
        resumed === null
          ? runFlow(flow, supplied, {
              log: path,
              runId,
              llmConfig: Object.keys(cliCfg).length > 0 ? cliCfg : null,
              numWorkers: options.numWorkers,
              ...shared,
            })
          : move({ log: path, numWorkers: options.numWorkers ?? null, checkLlmConfigs, ...shared }),
      ),
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
        const sending = flusher.sendQueued();
        startsWith(sending);
        result = await inTurn(async () => {
          try {
            await sending;
          } finally {
            lineSending = null;
          }
          // A press while they were sent goes first: its Stop is for the reply they start.
          await interrupting;
          return advanceRun();
        });
        continue;
      }
      // None of the lines sent so far can be withdrawn any more.
      if (inboxDrained(result.waits, inbox.name)) sent.clear();
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
      if (heldCommand !== null) {
        const name = heldCommand;
        heldCommand = null;
        result = (await runCommand(name)) ?? result;
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
        // An approval or a question a child run's tool asks waits behind the prompt otherwise.
        yields: true,
        ...framing,
        ...(editor === null ? {} : { editor }),
      });
      // The box keeps the tokens a draft holds; a plain prompt has none. A line entered has left
      // the box, empty for the next.
      const typed =
        editor === null
          ? "draft" in read
            ? [read.draft]
            : []
          : "value" in read
            ? editor.entered()
            : editor.draft();
      const kept = "aborted" in read || "yielded" in read || "escape" in read || "open" in read;
      draft = kept ? typed : [];
      timer.abort();
      line = null;
      const flushed = flusher?.promptDown() ?? false;
      stop.signal.throwIfAborted();
      if (cancelling !== null) continue;
      // The prompt comes back once the one queued behind it is answered.
      if ("yielded" in read) continue;
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
      if ("eof" in read || EXIT_LINES.has(read.value.trim())) {
        result = await leave();
        continue;
      }
      if (!host.commands?.names.has(read.value.trim())) {
        const event = lineEvent(read.value, inbox.message);
        result = (await sendEvent(event, editor === null ? null : typed)) ?? result;
        continue;
      }
      result = (await runCommand(read.value.trim())) ?? result;
    }
    // The notes below go out on the terminal as it was.
    releaseRaw();
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
    releaseRaw();
    if (error instanceof HostStop || (error instanceof TerminalStopped && error.reason === "eof")) {
      const stopped = host.stopped ?? ((path: string) => stoppedText(path, inbox));
      await note(log === null ? "chat cancelled\n" : stopped(log));
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
    releaseRaw();
    await interrupting;
    flusher?.dispose();
    // Pushed after /exit, or held when the run was cancelled: printed, as /exit prints them.
    await flusher?.printLeft();
    stop.signal.removeEventListener("abort", takeDown);
    host.onTyped?.(terminal.typedAhead());
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
