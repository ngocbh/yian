/**
 * The child flows runs start and own, as `yian chat`'s assistant starts flows: each child flow
 * runs in the background under one driver (`ChildFlows`), which is the only caller of `runFlow` and
 * `advance` on its log, and publishes the child's status each time a pass of it settles differently.
 * A run holds its children through an owner handle (`ChildFlows.owner`), which starts them and acts
 * on the runs it reaches; `childFlows` gives the owner's model those acts as one tool group.
 *
 * A child moves only when its driver runs a pass (`runFlow`, then `advance`): `advance` refuses a
 * run whose engine still loops, so a second caller would race the first. A delivery reaches a
 * live engine directly or is appended to the log offline; either way the deliverer then nudges
 * the driver, which runs one more pass once the current one ends. Nothing else wakes a waiting
 * run at its `wakeAt`, so between passes the driver sleeps until then.
 *
 * Every status a child publishes carries the replies and memory notes no reader has marked read
 * (`markRead`), so a reader that falls behind loses none; asks are always the run's open ones.
 * `shownStatus` is a status as a model reads it, and `childEvent` the one line that says what a
 * status means.
 *
 * What the host decides comes in through its hooks (`ChildFlowsOptions`): the key preflight, how a
 * child's memory inputs are bound and kept out of what the driver reports, the console sink, and
 * who hears each status. Every pass of a child gets the same memory instances. A group's `start`
 * asks its host which flow a call names and whether it may start (`ChildFlowStart`).
 *
 * The host names who speaks: an owner by its label, which signs its model's messages to its children
 * and names it on each delivery its model makes (`by`), and a child by its own, `<flow> r<n>` unless
 * the host gives one. No label may pass for another.
 *
 * A child's `ask` returns `pending()`, so the child waits durably for the user's answer, which the
 * host relays with `answer`. The children's stop is their own, tied to the host's exit by
 * `dispose`, never to the call that started them.
 *
 * Knows about: the run boundary (`runFlow`, `advance`, the deliveries, `pendingCalls`,
 * `runLogPath`, the gate's hook, its input intake and library refusal), input coercion, the waits'
 * `wakeAt` and the record-time reader, the platform's scheduler and folder tidying, compose, memory's
 * base type, the run events, the host tool API and its errors, the tool registry and the `pending`
 * marker, skills, types and values.
 */

import { dirname } from "node:path";
import { isDeepStrictEqual } from "node:util";

import type { ComposedFlow } from "./compose/build/pipeline.ts";
import type { Env } from "./defaults.ts";
import { reprText } from "./diagnostics.ts";
import { NodeSucceeded, type RunEvent, RunStarted } from "./events.ts";
import type { ModelAccess } from "./llm_clients/factory.ts";
import type { Author, MemoryBase } from "./memory/base.ts";
import { coerceInputs } from "./nodes/binding.ts";
import { isSystemError, removeEmptyDir, type Scheduler, sleepUntil } from "./platform.ts";
import {
  advance,
  type BeforeToolCall,
  cancel,
  close,
  complete,
  type ConsoleHook,
  type DeliveryOptions,
  type DeliveryRefusal,
  DeliveryRefused,
  hostInputs,
  type JsonObject,
  libraryRunMessage,
  type MemoryDrift,
  pendingCalls,
  type RunResult,
  runFlow,
  runLogPath,
  send,
} from "./run.ts";
import { recordTimeMs, type RunParent } from "./runtime/log.ts";
import { wakeAt } from "./runtime/waits.ts";
import type { Skill } from "./skills/catalog.ts";
import { pending, TAKE } from "./tools/context.ts";
import { ReportedToolError } from "./tools/errors.ts";
import { type FunctionTool, stringArg, stringArguments, tool } from "./tools/function_tools.ts";
import { registeredTools, type ToolEntry } from "./tools/registry.ts";
import { YianRangeError, YianTypeError } from "./typesys/errors.ts";
import { builtinType } from "./typesys/types.ts";
import {
  buildValueAs,
  isInboxType,
  isMemoryType,
  isRecord,
  renderValuePath,
  TypeCheckError,
  typeEquals,
  type Value,
  writeAsJson,
  writeAsText,
} from "./typesys/values.ts";

/** The most characters of a child event's line. */
const EVENT_BODY_CHARS = 2000;

/** The most asks, or waits, a child event or a status shown to the model lists. */
const LIST_ITEMS = 20;

/** The most characters of each text field a run tool's status shows the model. */
const STATUS_TEXT_CHARS = 4000;

/** The most characters of the flow part of a child's label. */
const LABEL_FLOW_CHARS = 40;

/**
 * What a label may not hold, as it doesn't show or doesn't read as what it is: control, format and
 * default-ignorable characters (bidi controls among them), lone surrogates, the braille blank and
 * the Hangul fillers.
 */
const UNSEEN = /[\p{Cc}\p{Cf}\p{Cs}\p{Default_Ignorable_Code_Point}\p{Bidi_Control}\u2800\u115f\u1160\u3164\uffa0]/u;

/** Whitespace a label may not hold once NFKC-normalized: at an end, doubled, or other than a space. */
const LOOSE_SPACE = /^\s|\s$|\s\s|[^\S ]/u;

/** What no label may read as (by `labelKey`): nothing is signed `yian` or `office`. */
const RESERVED: ReadonlySet<string> = new Set(["yian", "office"]);

/** A child's label (`<flow> r<n>`), as `labelKey` reads a label: one whose last word is `r` and digits. */
const CHILD_LABEL = /(^|\s)r\p{Nd}+$/u;

/** A relay's label (`<person>, via <employee>`), as `labelKey` and `VIA_LOOKALIKES` read one: the word `via`. */
const RELAY_LABEL = /(^|[^\p{L}\p{N}])via($|[^\p{L}\p{N}])/u;

/** Letters of other scripts that look like `v`, `i` or `a`, read as those when looking for `via`. */
const VIA_LOOKALIKES: Readonly<Record<string, string>> = {
  "\u03bd": "v", "\u0475": "v", "\u1d20": "v",
  "\u0131": "i", "\u0269": "i", "\u03b9": "i", "\u0456": "i", "\u04cf": "i",
  "\u0251": "a", "\u03b1": "a", "\u0430": "a",
};

/** @internal The longest a group's `wait` waits, in seconds. */
export const WAIT_MAX_S = 120;

/** The built-in `Message` type, which a model-made value may not sign. */
const MESSAGE = builtinType("Message")!;

/** Where a child run stands, as the model of the run that owns it is told. */
export interface ChildStatus {
  /** The child's id among its driver's children (`r1`, `r2`, ...). */
  readonly run: string;
  /**
   * `running` from the start and from each delivery the run took until the pass it causes
   * settles; `waiting`, `succeeded`, `failed` or `cancelled` as a pass leaves it. A pass at a
   * `wakeAt` publishes only where it settles.
   */
  readonly status: "running" | "waiting" | "succeeded" | "failed" | "cancelled";
  /** The flow's output, once it succeeded. */
  readonly output?: unknown;
  /**
   * Why it failed, once the run failed. A child whose driver a host fault stopped (the log
   * unreadable, another writer holding it) is `failed` with only a `note`: such a fault has no
   * failure code, the codes being the run's closed list.
   */
  readonly failure?: { readonly code: string; readonly message: string; readonly node: string | null };
  /** What a waiting child waits on (`WaitInfo`s). */
  readonly waits?: readonly unknown[];
  /** When a waiting child next wakes by itself, as a record time, or `null` when only a delivery wakes it. */
  readonly wake_at?: string | null;
  /** The questions a waiting child asks, each answered by its call id. */
  readonly asks?: readonly { readonly call: string; readonly question: string }[];
  /**
   * The tool calls of the child's agents that the children's `beforeToolCall` held, each by its
   * call id, with its tool and the model's arguments. `answer` doesn't settle them.
   */
  readonly held?: readonly { readonly call: string; readonly tool: string; readonly arguments: JsonObject | null }[];
  /**
   * What the child's nodes named `reply` said and no reader has marked read (`markRead`), oldest
   * first.
   */
  readonly replies?: readonly string[];
  /**
   * What the host noted about the child's memories and no reader has marked read, oldest first: a
   * memory its start created, or one a crash resume found changed since the run started.
   */
  readonly memory_notes?: readonly string[];
  /**
   * Something the driver couldn't do with the run, in words: a delivery the run refused or one made
   * after `dispose`, or a signed message that went as plain text (only on the status that call
   * returns); or the error that stopped the driver of a failed child.
   */
  readonly note?: string;
  /** The child's run log. */
  readonly log: string;
  /**
   * The status's number: every status any child publishes gets the next one, so a reader marks
   * the statuses it has seen by the last number it read.
   */
  readonly seq: number;
}

/** A status before it is numbered. */
type Unnumbered = Omit<ChildStatus, "seq">;

/** A child's memory instances, by input name. */
export type BoundMemories = Readonly<Record<string, MemoryBase>>;

/** How a host binds the children's memory inputs, and keeps them out of what the driver reports. */
export interface MemoryHooks {
  /**
   * Bind `flow`'s memory inputs for a child whose flow file is in the folder `project` (`null` for
   * a flow with no file); every pass of the child gets these instances. `note` takes a line for the
   * child's status each time one of them creates its memory (`ChildStatus.memory_notes`).
   *
   * @throws To refuse the start, before anything is written.
   */
  bind(flow: ComposedFlow, project: string | null, note: (line: string) => void): BoundMemories;
  /** The line for a memory a crash resume found changed since its run started. */
  drift(drift: MemoryDrift): string;
  /**
   * `error` as the driver may report it: a start's rejection, a failed child's note. `memory` is the
   * child's instances, or none for a bind's own error.
   */
  scrub(error: unknown, memory: BoundMemories): unknown;
}

/** What a `ChildFlows` needs from its host. */
export interface ChildFlowsOptions {
  /**
   * The folder a child's CODE files must resolve inside, when its flow was loaded confined to
   * one: each delivery to it passes this as `codeRoot`.
   */
  codeRoot: string;
  /** The folder each child's run folder goes in. */
  logDir: string;
  env: Env;
  /** The `--provider`/`--model` layer, possibly empty; `setLlmConfig` replaces it. */
  llmConfig: Readonly<Record<string, string>>;
  /** For tests: the model builder AGENT nodes use. */
  models?: ModelAccess | null;
  /** The clock the children's records and waits, and the group's `wait`, run on. */
  scheduler: Scheduler;
  /** How the children's memory inputs are bound. */
  memories: MemoryHooks;
  /**
   * Check that `flow` can start with `llmConfig` and `inputs` before anything is written: that
   * the provider keys its agents need are there, say. What it throws, `start` throws.
   */
  preflight: (
    flow: ComposedFlow,
    llmConfig: Readonly<Record<string, string>>,
    inputs: Readonly<Record<string, unknown>>,
  ) => Promise<void> | void;
  /** The skills a child's flow grants, as `yian run` reads them, or `null` for none. */
  skillsFor: (flow: ComposedFlow) => ReadonlyMap<string, Skill> | null;
  /** Gets each line the children's CODE nodes print; `null` (the default) leaves it on the console. */
  onConsole?: ConsoleHook | null;
  /**
   * The host's check before each tool call the children's agents make, as `runFlow`'s; a call it
   * holds is listed in the child's status (`ChildStatus.held`). `null` (the default) checks none.
   */
  beforeToolCall?: BeforeToolCall | null;
  /**
   * Sees each status any child publishes, once it is the child's (not a child's first `running`,
   * which `start` returns). What it throws is dropped.
   */
  onStatus?: ((status: ChildStatus) => void) | null;
  /** @internal For tests: the `advance` the drivers call. */
  advance?: typeof advance;
  /** @internal For tests: the `cancel` `stop` and `dispose` call. */
  cancel?: typeof cancel;
}

/** How `ChildFlowOwner.start` starts a child. */
export interface ChildStartOptions {
  /** The caller's: aborting it before the start begins rejects, and after it does nothing to the child. */
  signal?: AbortSignal | null;
  /** What `live` lists the child by (the flow's file, say); the flow's name when absent. */
  name?: string;
  /**
   * Host tools this child alone gets, beside the registered ones; or a function that builds them
   * from the child's run id, for tools that act as the child (its own `ChildFlows.owner`, say).
   */
  tools?: Readonly<Record<string, ToolEntry>> | ((runId: string) => Readonly<Record<string, ToolEntry>>);
  /** Its skills in place of `skillsFor`'s, when given. */
  skills?: ReadonlyMap<string, Skill>;
  /** The flow file's project folder, which `memories.bind` keys the child's memories by; `null` for none. */
  project?: string | null;
  /**
   * The owner's node that starts the child (`ToolContext.node`), which it records in its parent;
   * `null` or `""` (the default) when no one node does.
   */
  node?: string | null;
  /**
   * The child's label, which signs what it says to its owner, in place of `<flow> r<n>`: an
   * instrument's name, say (`shell 1`). It may not read as its owner's, a person's, a relay's
   * (`<person>, via <employee>`) or another child's, whatever its case, spacing or zero-width
   * characters. `null` (the default) for `<flow> r<n>`.
   */
  label?: string | null;
}

/** A child once its log holds its start: its id, its log, and the seq of its first status. */
export interface StartedChild {
  readonly id: string;
  readonly log: string;
  readonly seq: number;
}

/** The flow a group's `start` call names, as its host's `resolve` finds it. */
export interface ChildFlowTarget {
  readonly flow: ComposedFlow;
  /** What `live` lists the child by (`ChildStartOptions.name`); the flow's name when absent. */
  readonly name?: string;
  /** The folder `memories.bind` keys the child's memories by; `null` (the default) for none. */
  readonly project?: string | null;
}

/** How a group's `start` finds the flow a call names, and asks its host before it starts it. */
export interface ChildFlowStart<T extends ChildFlowTarget = ChildFlowTarget> {
  /**
   * The start of the tool's description: what `flow` names, and who approves a start. The group adds
   * what the tool returns.
   */
  readonly description: string;
  /**
   * The flow a call names: `flow`, read as the description says, and `content`, a draft's whole text
   * to run in its place, or `null`.
   *
   * @throws {ReportedToolError} What the model reads when they name no flow it may start.
   */
  resolve(call: { readonly flow: string; readonly content: string | null }): Promise<T> | T;
  /** Whether `target` may start with `inputs`, which passed the input check: `null` once it may, else why not. */
  approve(
    target: T,
    inputs: Readonly<Record<string, unknown>>,
    signal: AbortSignal,
  ): Promise<string | null> | string | null;
}

/** What `childFlows` builds an owner's group with. */
export interface ChildFlowGroupOptions<T extends ChildFlowTarget = ChildFlowTarget> {
  /** With it, the group has `start`; without it (the default), only the host starts children. */
  readonly start?: ChildFlowStart<T> | null;
}

/** What `ChildFlows.owner` makes an owner handle from. */
export interface ChildFlowOwnerOptions {
  /** The owner's run: each child it starts has the run id `<runId>-r<n>` and records it as its parent. */
  readonly runId: string;
  /**
   * The owner's author label, the employee's name (`yian-chat`): a message its model sends a child
   * is signed with it, and each delivery its model makes is recorded as its run's agent by it. It
   * may not read as a child's label (`<flow> r<n>`) or a relay's (`<person>, via <employee>`). For a
   * run this driver started, it is that run's label, the default; any other run needs one.
   */
  readonly label?: string | null;
  /**
   * The labels of the people who act through the owner (`the user` in `yian chat`), which no child's
   * label may read as. Each is checked as `label` is. An owner of a run this driver started has its
   * starter's by default; any other none.
   */
  readonly people?: readonly string[] | null;
  /**
   * The owner's inbox its children report to, which each records in its parent; `null` (the
   * default) for none.
   */
  readonly inbox?: string | null;
  /**
   * The key the host passes the owner's tools under (`childFlows`), which their texts name them
   * by: `<group>_status`, and so on.
   */
  readonly group: string;
  /**
   * The host's record of what the owner's model has seen, by run and status seq. With it, the
   * owner's tools mark each status they show read (`ChildFlows.markRead`) and tell `markSeen`, and
   * `wait` returns at once for a status past `seen(run)`. Without it they mark nothing, so the
   * host's other readers still get every reply. `null` (the default) for none.
   */
  readonly seen?: { markSeen(run: string, seq: number): void; seen(run: string): number } | null;
}

/** A child left waiting when `dispose` ends its parent's hold on it, and what it waits on. */
export interface WaitingChild {
  readonly run: string;
  /**
   * `question` when it asks something, else `held` when it waits for a held call's approval, else
   * `inbox` when it waits for a message, else `timer`; `halted` for a running one `haltAll` stopped
   * before its cancel was done.
   */
  readonly kind: "question" | "held" | "inbox" | "timer" | "halted";
  readonly log: string;
}

/** A child still running or waiting, as `live` lists it. */
export interface LiveChild {
  readonly id: string;
  /** What it runs, as its start named it: the flow's file, or else the flow's name. */
  readonly name: string;
  /** When it started, on the children's clock. */
  readonly since: number;
  readonly status: ChildStatus;
}

/** A `next` call waiting for a status numbered after `after`. */
interface Waiter {
  readonly after: number;
  readonly resolve: (status: ChildStatus) => void;
}

/** One child and its driver's state. */
interface Child {
  readonly id: string;
  /** Its run id, `<owner's run id>-<id>`. */
  readonly runId: string;
  /** What it records as its parent: its owner's run, the node that started it, the owner's inbox, and its label. */
  readonly parent: RunParent & { readonly label: string };
  readonly flow: ComposedFlow;
  /** What its start named it (`LiveChild.name`). */
  readonly name: string;
  /** When it started, on the children's clock. */
  readonly since: number;
  readonly log: string;
  /** The tools, skills and memories every pass of the child is given. */
  readonly tools: Record<string, ToolEntry>;
  readonly skills: ReadonlyMap<string, Skill> | null;
  readonly memory: BoundMemories;
  /** The last published status. */
  status: ChildStatus;
  readonly waiters: Set<Waiter>;
  /** Set by a nudge: the driver runs one more pass before it rests. */
  dirty: boolean;
  /** The driver's rest between passes, which a nudge ends; `null` while it doesn't rest. */
  rest: AbortController | null;
  /**
   * What the child's `reply` nodes said, and the host noted about its memories, during the
   * current pass.
   */
  readonly fresh: Said[];
  /**
   * The replies and memory notes published and not marked read, each with the seq of the status
   * that first held it.
   */
  unread: (Said & { readonly seq: number })[];
  /** Whether the run's log holds its start. */
  started: boolean;
  /** The child's own stop: it stops the driver and leaves the log as it is. */
  readonly halt: AbortController;
  /** Settles once the driver has stopped. */
  done: Promise<void>;
}

/** A reply of the child's, or a note of the host's about its memories. */
interface Said {
  readonly kind: "reply" | "memory";
  readonly text: string;
}

/** How a delivery may be made again (`ChildFlows.answer`'s `options.resent`). */
interface Resend {
  readonly resent?: boolean;
}

/** @internal Who a delivery to a child is from. */
export interface ChildSender {
  /** The author a message to an `Inbox(Message)` carries. */
  readonly label: string;
  /** Who the child's log records made the call. */
  readonly by: Author;
}

/** How a delivery is made: who from, and again or not. */
interface Sent extends Resend {
  /**
   * @internal Who it is from: the owner's model, as `ChildFlowOwner` passes it. `null`, the
   * default, for the host itself, which signs nothing and records no one.
   */
  readonly from?: ChildSender | null;
}

/**
 * An owner as its children see it: its run, its label, its people's labels, its inbox, and the
 * labels above it (`heard`): the owners up the tree from it and their people's.
 */
interface Owned {
  readonly runId: string;
  readonly label: string;
  readonly people: readonly string[];
  readonly inbox: string | null;
  readonly heard: readonly string[];
}

/** The refusals of a delivery made again that say it landed before, or that the run moved past it. */
const LANDED: ReadonlySet<DeliveryRefusal> = new Set(["ended", "unknown_call", "call_settled"]);

/** The terminal statuses: no pass follows them. */
const ENDED: ReadonlySet<ChildStatus["status"]> = new Set(["succeeded", "failed", "cancelled"]);

/** Whether `event` ends a node named `reply`, at any depth: what an inbox child says each turn. */
function isReply(event: RunEvent): event is NodeSucceeded {
  return event instanceof NodeSucceeded && event.nodeId.slice(event.nodeId.lastIndexOf("/") + 1) === "reply";
}

/** The question text of an ask's recorded arguments: a TOOL `ask`'s prompt, an agent's question. */
function questionText(args: Readonly<Record<string, unknown>> | null): string {
  const text = args?.["prompt"] ?? args?.["question"];
  if (typeof text === "string" && text !== "") return text;
  // A form asks its questions as a list.
  return args?.["questions"] == null ? "" : JSON.stringify(args["questions"]);
}

/**
 * One child event: the line the owner's model reads, and the status it came from. `run` is `null`
 * (and `seq` 0) for one that came from no status of the chat's: a message authored `yian` the chat
 * didn't send, taken back from the inbox.
 */
export interface ChildEvent {
  readonly run: string | null;
  readonly seq: number;
  readonly text: string;
  /** The event as the chat's view draws it; absent for one that came from no status. */
  readonly card?: EventCard;
}

/**
 * A child event as a card on the terminal: the status it came from, what the run now does (its
 * first clause's verb: `asks`, `succeeded`, ...) and the rest of the line.
 */
export interface EventCard {
  readonly status: ChildStatus["status"];
  readonly verb: string;
  readonly body: string;
}

/** What a waiting status waits on, as far as `childEvent` reads it. */
interface Wait {
  readonly tool?: unknown;
  readonly inbox?: unknown;
  readonly settled?: unknown;
}

/**
 * The event `status` is for the owner's model, or `null` when it says nothing to act on (a run
 * `running`, or waiting only for a timer with nothing said). The line names the run, then what
 * it now does (asks, with each question's call id; waits for a message on an inbox; succeeded,
 * failed or was cancelled), then what it said and what the host noted about its memories; child
 * text is JSON-quoted, so the line stays one line, and the line is cut at `EVENT_BODY_CHARS` with
 * a note.
 */
export function childEvent(status: ChildStatus): ChildEvent | null {
  if (status.status === "running") return null;
  // Each clause is its verb and what follows it, which starts with a space when it isn't empty.
  const clauses: [string, string][] = [];
  const quote = (value: unknown) => JSON.stringify(value) ?? "null";
  if (status.status === "succeeded") clauses.push(["succeeded", ` with output ${quote(status.output ?? null)}`]);
  else if (status.status === "cancelled") clauses.push(["was cancelled", ""]);
  else if (status.status === "failed") {
    const failure = status.failure;
    clauses.push([
      "failed",
      failure === undefined
        ? `: ${quote(status.note ?? "")}`
        : ` (${failure.code}${failure.node === null ? "" : ` at ${failure.node}`}): ${quote(failure.message)}`,
    ]);
  } else if ((status.asks ?? []).length > 0) {
    const asks = status.asks!.slice(0, LIST_ITEMS).map((ask) => `(call ${ask.call}): ${quote(ask.question)}`);
    const more = status.asks!.length - LIST_ITEMS;
    if (more > 0) asks.push(`(${more} more; its status shows all)`);
    clauses.push(["asks", ` ${asks.join(", ")}`]);
  } else {
    const waits = (status.waits ?? []) as readonly Wait[];
    const inbox = waits.find((wait) => wait.tool === TAKE && wait.settled !== true && typeof wait.inbox === "string");
    if (inbox !== undefined) clauses.push(["waits for a message", ` on inbox ${quote(inbox.inbox)}`]);
  }
  if ((status.replies ?? []).length > 0) clauses.push(["said", ` ${status.replies!.map(quote).join(", ")}`]);
  const notes = status.memory_notes ?? [];
  if (notes.length > 0) clauses.push(["noted", ` ${notes.map(quote).join(", ")}`]);
  if (clauses.length === 0) return null;
  const cut = (line: string) =>
    line.length > EVENT_BODY_CHARS ? `${line.slice(0, EVENT_BODY_CHARS)}... (cut; its status shows all)` : line;
  const text = cut(`run ${status.run} ${clauses.map(([verb, rest]) => verb + rest).join("; ")}`);
  const [[verb, first], ...others] = clauses as [[string, string], ...[string, string][]];
  const body = cut([first.trim(), ...others.map(([word, rest]) => word + rest)].filter(Boolean).join("; "));
  return { run: status.run, seq: status.seq, text, card: { status: status.status, verb, body } };
}

/** @internal `text` cut to `STATUS_TEXT_CHARS` characters, with a note when it was cut. */
export function cutStatusText(text: string): string {
  if (text.length <= STATUS_TEXT_CHARS) return text;
  return `${text.slice(0, STATUS_TEXT_CHARS)}... (cut at ${STATUS_TEXT_CHARS} characters)`;
}

/** `value` as a status shows it: a string cut, and a value too long as JSON as its cut JSON text. */
function shownValue(value: unknown): unknown {
  if (typeof value === "string") return cutStatusText(value);
  const json = writeAsJson(value as Value);
  return json.length > STATUS_TEXT_CHARS ? cutStatusText(json) : value;
}

/**
 * @internal `status` as the model reads it: its text fields cut, its asks, held calls and waits
 * listed up to `LIST_ITEMS` each (`not_shown` says how many more), without the seq it doesn't need.
 */
export function shownStatus(status: ChildStatus): Record<string, unknown> {
  const { seq: _, output, failure, asks, held, replies, note, waits, ...rest } = status;
  const shown: Record<string, unknown> = { ...rest };
  const more: string[] = [];
  if (waits !== undefined) {
    shown["waits"] = waits.slice(0, LIST_ITEMS);
    if (waits.length > LIST_ITEMS) more.push(`${waits.length - LIST_ITEMS} more waits`);
  }
  if (output !== undefined) shown["output"] = shownValue(output);
  if (failure !== undefined) shown["failure"] = { ...failure, message: cutStatusText(failure.message) };
  if (held !== undefined) {
    shown["held"] = held.slice(0, LIST_ITEMS).map((call) => ({ ...call, arguments: shownValue(call.arguments) }));
    if (held.length > LIST_ITEMS) more.unshift(`${held.length - LIST_ITEMS} more held calls`);
  }
  if (asks !== undefined) {
    shown["asks"] = asks.slice(0, LIST_ITEMS).map((ask) => ({ ...ask, question: cutStatusText(ask.question) }));
    if (asks.length > LIST_ITEMS) more.unshift(`${asks.length - LIST_ITEMS} more asks`);
  }
  if (replies !== undefined) shown["replies"] = replies.map(cutStatusText);
  if (note !== undefined) shown["note"] = cutStatusText(note);
  if (more.length > 0) shown["not_shown"] = `${more.join(" and ")}; the run's log has all`;
  return shown;
}

/** `label` as labels are compared: NFKC-normalized and fully case-folded (`ß` as `ss`). */
function labelKey(label: string): string {
  // Upper then lower case folds as full case folding does.
  return label.normalize("NFKC").toUpperCase().toLowerCase().normalize("NFKC");
}

/**
 * `label` once it may sign what `what` names, reading as it looks: it holds no `UNSEEN` character
 * and no loose whitespace (`LOOSE_SPACE`), and doesn't read as a reserved label, a relay
 * (`<person>, via <employee>`) or a child's (`<flow> r<n>`), which only the driver gives.
 *
 * @throws {TypeError} When `label` isn't a string.
 * @throws {RangeError} When it is refused.
 */
function checkedLabel(label: unknown, what: string): string {
  if (typeof label !== "string") throw new YianTypeError(`${what} must be a non-empty string`);
  if (label === "") throw new YianRangeError(`${what} must be a non-empty string`);
  const refused = (why: string) => new YianRangeError(`${what} ${reprText(label)} ${why}`);
  const unseen = UNSEEN.exec(label)?.[0];
  if (unseen !== undefined) {
    const code = unseen.codePointAt(0)!.toString(16).toUpperCase().padStart(4, "0");
    throw refused(`holds an invisible or control character (U+${code})`);
  }
  if (LOOSE_SPACE.test(label.normalize("NFKC"))) {
    throw refused("has whitespace at an end, doubled, or other than a space");
  }
  const key = labelKey(label);
  if (RESERVED.has(key)) throw refused("is reserved: nothing is signed yian or office");
  const plain = Array.from(key, (char) => VIA_LOOKALIKES[char] ?? char).join("");
  if (RELAY_LABEL.test(plain)) throw refused("reads as a relay (<person>, via <employee>)");
  if (CHILD_LABEL.test(key)) throw refused("reads as a child run's (<flow> r<n>)");
  return label;
}

/**
 * `people`, each checked as `checkedLabel` checks it, once none reads as the owner's `label` or as
 * another person's.
 *
 * @throws {TypeError} When a label isn't a string.
 * @throws {RangeError} When one is refused.
 */
function checkedPeople(people: readonly unknown[], label: string): string[] {
  const seen = new Map<string, string>();
  return people.map((each) => {
    const person = checkedLabel(each, "a person's label");
    const key = labelKey(person);
    const same = seen.get(key);
    if (key === labelKey(label)) throw new YianRangeError(`a person's label ${reprText(person)} reads as the owner's`);
    if (same !== undefined) {
      throw new YianRangeError(`a person's label ${reprText(person)} reads as the person ${reprText(same)}`);
    }
    seen.set(key, person);
    return person;
  });
}

/**
 * The host's `label` for a child of `owner`, once it reads as no one else's: as `checkedLabel`
 * checks it, and not as its owner's, one of its people's, a label above the owner (`Owned.heard`),
 * or a label of `siblings`, its owner's children that still run.
 *
 * @throws {TypeError} When `label` isn't a string.
 * @throws {RangeError} When it is refused.
 */
function checkedChildLabel(
  label: unknown,
  owner: Owned,
  siblings: readonly { readonly id: string; readonly label: string }[],
): string {
  const checked = checkedLabel(label, "a child's label");
  const key = labelKey(checked);
  const as = (whose: string) => new YianRangeError(`a child's label ${reprText(checked)} reads as ${whose}`);
  if (key === labelKey(owner.label)) throw as("its owner's");
  const person = owner.people.find((each) => labelKey(each) === key);
  if (person !== undefined) throw as(`the person ${reprText(person)}`);
  const above = owner.heard.find((each) => labelKey(each) === key);
  if (above !== undefined) throw as(`${reprText(above)}, a label above it`);
  const sibling = siblings.find((each) => labelKey(each.label) === key);
  if (sibling !== undefined) throw as(`run ${sibling.id}'s, which still runs`);
  return checked;
}

/**
 * A child's own label, `<flow> <id>`: `flow`, its flow's id (or file name), with its accents dropped
 * and each run of other characters than `A-Za-z0-9._-` made one `_`, cut at `LABEL_FLOW_CHARS`.
 * With no comma, space or quote in its flow part, it can't read as a relay.
 */
function childLabel(flow: string, id: string): string {
  const plain = flow.normalize("NFKD").replace(/\p{M}/gu, "").replace(/[^A-Za-z0-9._-]+/g, "_");
  return `${plain.slice(0, LABEL_FLOW_CHARS).replace(/^_+|_+$/g, "") || "flow"} ${id}`;
}

/**
 * Who an owner's model's deliveries are from: signed with the owner's label, and recorded as the
 * agent of its run by that label. It names no node, whose id changes each turn of a loop, so every
 * delivery, and every take-back of one, by the owner's model is the same sender's.
 */
function modelSender(owner: Owned): ChildSender {
  const by: Author = { kind: "agent", name: owner.label, run: owner.runId };
  return Object.freeze({ label: owner.label, by: Object.freeze(by) });
}

/**
 * The child flows of a host's runs: each started in the background by its owner (`owner`), its
 * status read as it changes, and answers and messages relayed to it.
 */
export class ChildFlows {
  readonly #options: ChildFlowsOptions;
  /** The provider and model layer the children started from now on get. */
  #llmConfig: Readonly<Record<string, string>>;
  readonly #children = new Map<string, Child>();
  /**
   * By the run id of each child a start began and didn't fail, the child as an owner: its label,
   * its starter's people, and the labels above it.
   */
  readonly #owners = new Map<string, Omit<Owned, "runId" | "inbox">>();
  #seq = 0;
  /** The number of the last child id a start took. */
  #started = 0;
  /** The start in progress: starts run one at a time, in call order. */
  #starting: Promise<unknown> = Promise.resolve();
  #disposed = false;
  /** Aborted by `haltAll`: a dispose waits no longer. */
  readonly #halting = new AbortController();

  constructor(options: ChildFlowsOptions) {
    this.#options = options;
    this.#llmConfig = options.llmConfig;
  }

  /**
   * A handle on run `options.runId`'s hold on the children it starts (`ChildFlowOwner`).
   *
   * @throws {TypeError} When a label isn't a string.
   * @throws {RangeError} When it has no label, a label is refused (`ChildFlowOwnerOptions.label`),
   *   the label differs from the one this driver gave the run, or the inbox isn't a non-empty string
   *   or `null`.
   */
  owner(options: ChildFlowOwnerOptions): ChildFlowOwner {
    const { runId, inbox = null } = options;
    const started = this.#owners.get(runId);
    const given = options.label ?? null;
    if (started !== undefined && given !== null && given !== started.label) {
      throw new YianRangeError(`run ${reprText(runId)} is labelled ${reprText(started.label)}`);
    }
    if (started === undefined && given === null) {
      throw new YianRangeError(`run ${reprText(runId)} needs an owner's label`);
    }
    const label = started?.label ?? checkedLabel(given, "an owner's label");
    const people = Object.freeze(checkedPeople(options.people ?? started?.people ?? [], label));
    if (inbox !== null && (typeof inbox !== "string" || inbox === "")) {
      throw new YianRangeError("an owner's inbox must be a non-empty string or null");
    }
    const owned: Owned = Object.freeze({ runId, label, people, inbox, heard: started?.heard ?? [] });
    return new ChildFlowOwner(this, options, owned, {
      start: (flow, inputs, given) => this.#start(owned, flow, inputs, given),
      reaches: (id) => this.#reaches(runId, id),
      scheduler: this.#options.scheduler,
    });
  }

  /** Give the children started from now on the provider and model layer `layer`. */
  setLlmConfig(layer: Readonly<Record<string, string>>): void {
    this.#llmConfig = layer;
  }

  /** Start `flow` as the next child of `owner`, as `ChildFlowOwner.start` says. */
  #start(
    owner: Owned,
    flow: ComposedFlow,
    inputs: Readonly<Record<string, unknown>>,
    options: ChildStartOptions,
  ): Promise<StartedChild> {
    const name = options.name ?? flow.name ?? flow.compiled.flowId ?? "flow";
    const tools = options.tools ?? {};
    const skills = options.skills ?? null;
    const signal = options.signal ?? null;
    const project = options.project ?? null;
    const node = options.node || null;
    const label = options.label ?? null;
    const given = { name, tools, skills, project, node, label };
    const turn = this.#starting.then(() => this.#begin(owner, flow, inputs, given, signal));
    this.#starting = turn.catch(() => {});
    return turn;
  }

  /** Start `flow` as the next child of `owner`, once the starts called before it are done. */
  async #begin(
    owner: Owned,
    flow: ComposedFlow,
    inputs: Readonly<Record<string, unknown>>,
    given: {
      name: string;
      tools: NonNullable<ChildStartOptions["tools"]>;
      skills: ReadonlyMap<string, Skill> | null;
      project: string | null;
      node: string | null;
      label: string | null;
    },
    signal: AbortSignal | null,
  ): Promise<StartedChild> {
    const { name, skills, project, node } = given;
    const check = () => {
      if (this.#disposed) throw new Error("the runs were disposed");
      signal?.throwIfAborted();
    };
    check();
    if (given.label !== null) {
      const siblings = [...this.#children.values()]
        .filter((child) => child.parent.runId === owner.runId && this.#driving(child))
        .map((child) => ({ id: child.id, label: child.parent.label }));
      checkedChildLabel(given.label, owner, siblings);
    }
    const { logDir, skillsFor, memories } = this.#options;
    try {
      await this.#options.preflight(flow, this.#llmConfig, inputs);
    } catch (error) {
      throw memories.scrub(error, {});
    }
    check();
    // A memory the start creates is said with the pass.
    const fresh: Said[] = [];
    let memory: BoundMemories;
    try {
      memory = memories.bind(flow, project, (text) => void fresh.push({ kind: "memory", text }));
    } catch (error) {
      throw memories.scrub(error, {});
    }
    // Making the run's folder takes its id. A run an earlier process started keeps its folder,
    // though this driver knows nothing of it, so its id is passed over.
    let number = this.#started;
    let log: string | null = null;
    while (log === null) {
      number += 1;
      try {
        log = runLogPath(logDir, `${owner.runId}-r${number}`);
      } catch (error) {
        if (!isSystemError(error) || error.code !== "EEXIST") throw error;
      }
    }
    const id = `r${number}`;
    const runId = `${owner.runId}-${id}`;
    const label = given.label ?? childLabel(flow.compiled.flowId ?? name, id);
    // Before its tools are built, as they may make the child an owner.
    const heard = Object.freeze([owner.label, ...owner.people, ...owner.heard]);
    this.#owners.set(runId, Object.freeze({ label, people: owner.people, heard }));
    const tools = typeof given.tools === "function" ? given.tools(runId) : given.tools;
    const seq = ++this.#seq;
    const child: Child = {
      id,
      runId,
      parent: Object.freeze({ runId: owner.runId, node, inbox: owner.inbox, label }),
      flow,
      name,
      since: this.#options.scheduler.now(),
      log,
      // One handler serves `ask` and `ask_user` (passing both is refused).
      tools: { ...registeredTools(), ...tools, ask: () => pending() },
      skills: skills ?? skillsFor(flow),
      memory,
      status: { run: id, status: "running", log, seq },
      waiters: new Set(),
      dirty: false,
      rest: null,
      fresh,
      unread: [],
      started: false,
      halt: new AbortController(),
      done: Promise.resolve(),
    };
    this.#children.set(id, child);
    const started = Promise.withResolvers<void>();
    child.done = this.#drive(child, inputs, started);
    try {
      await started.promise;
    } catch (error) {
      this.#children.delete(id);
      this.#owners.delete(runId);
      throw error;
    }
    this.#started = number;
    return { id, log, seq };
  }

  /**
   * Return child `id`'s last published status.
   *
   * @throws {RangeError} When no child has the id.
   */
  status(id: string): ChildStatus {
    return this.#child(id).status;
  }

  /**
   * Resolve with child `id`'s latest status once its seq is past `after` (without it, past the
   * current status's, so a status published before the call isn't seen; pass the `seq` read
   * before). Resolve with its current status instead once its driver has stopped and none will
   * follow, `cancelWaits` is called, or `options.signal` aborts (at once when it already has).
   *
   * @throws {RangeError} When no child has the id.
   */
  next(id: string, after?: number, options: { signal?: AbortSignal | null } = {}): Promise<ChildStatus> {
    const child = this.#child(id);
    const since = after ?? child.status.seq;
    const signal = options.signal ?? null;
    if (child.status.seq > since || !this.#driving(child) || signal?.aborted) return Promise.resolve(child.status);
    const { promise, resolve } = Promise.withResolvers<ChildStatus>();
    const onAbort = () => {
      child.waiters.delete(waiter);
      resolve(child.status);
    };
    const waiter: Waiter = {
      after: since,
      resolve: (status) => {
        signal?.removeEventListener("abort", onAbort);
        resolve(status);
      },
    };
    child.waiters.add(waiter);
    signal?.addEventListener("abort", onAbort, { once: true });
    return promise;
  }

  /** The children still running or waiting (and driven), by id. */
  live(): LiveChild[] {
    return [...this.#children.values()]
      .filter((child) => this.#driving(child))
      .map(({ id, name, since, status }) => ({ id, name, since, status }));
  }

  /** @internal For tests: how many `next` calls wait, over all children. */
  waitingNexts(): number {
    let count = 0;
    for (const child of this.#children.values()) count += child.waiters.size;
    return count;
  }

  /**
   * Stop every child's driver at once, leaving each log as it is, and end a `dispose` that waits
   * on a cancel: it returns with each running child it hadn't cancelled yet as `halted`. For a
   * host that must stop at once, such as `yian chat` on a second Ctrl-C.
   */
  haltAll(): void {
    this.#halting.abort();
    for (const child of this.#children.values()) child.halt.abort();
  }

  /** Resolve every waiting `next` call with its child's current status. */
  cancelWaits(): void {
    for (const child of this.#children.values()) this.#release(child);
  }

  /**
   * Mark child `id`'s replies and memory notes published at or before status `seq` read: later
   * statuses leave them out. Call it when a status is handed to the model.
   *
   * @throws {RangeError} When no child has the id.
   */
  markRead(id: string, seq: number): void {
    const child = this.#child(id);
    child.unread = child.unread.filter((said) => said.seq > seq);
  }

  /**
   * Have child `id`'s driver move it again: at once when it rests, else once its current pass
   * ends. Each delivery here nudges; call it after a delivery made elsewhere.
   *
   * @throws {RangeError} When no child has the id.
   */
  nudge(id: string): void {
    const child = this.#child(id);
    child.dirty = true;
    child.rest?.abort();
  }

  /**
   * Answer child `id`'s open question `call` with `value` (any JSON but `null`). Returns its
   * status: `running` once the run took the answer, else the current one with a `note`. A held
   * call isn't a question: answering it would make up its tool's result, so the note refuses it.
   * Nor may `value` sign a message: nothing in it that reads as a `Message` carries an `author` or
   * an action part, whatever type the question declares, since only the driver signs one.
   * With `options.resent`, the call may have been made before, as a resumed agent's call is: a
   * refusal saying it landed or the run moved past it (`ended`, `unknown_call`, `call_settled`)
   * returns the status with no note. The child's log records `options.from`'s `by` as who made
   * the call, and each user message in `value` with no author is signed with its label
   * (`signedBy`); without it (the default), the host itself, recorded as no one, signs nothing.
   *
   * @throws {RangeError} When no child has the id.
   * @throws {TypeError} When `value` is `undefined`.
   * @throws What a delivery fault throws (the log unreadable, another writer holding it), as
   *   `memories.scrub` leaves it.
   */
  answer(id: string, call: string, value: unknown, options: Sent = {}): Promise<ChildStatus> {
    const child = this.#child(id);
    const refusal = () => {
      const signed = signedMessageAt(value);
      if (signed !== null) return signedNote(signed);
      const held = child.started && pendingCalls(child.log).some((each) => each.callId === call && each.held);
      return held ? `call '${call}' is a held tool call, not a question` : null;
    };
    const from = options.from ?? null;
    const output = from === null ? value : signedBy(value, from.label);
    const answer = (log: string, delivery: DeliveryOptions) => complete(log, call, { output }, delivery);
    return this.#deliver(child, answer, { ...options, refusal });
  }

  /**
   * Send `text` to child `id`'s inbox `inbox`: to an `Inbox(Message)` as a user message holding it,
   * authored by `options.from`'s label (unauthored without it); to another inbox, such as an
   * `Inbox(str)`, as is, and then the status's `note` says the run can't tell who wrote it. Returns
   * its status, and throws, as `answer` does. `options.key` is the send's idempotency key: a send
   * repeated with it, by the same sender, appends nothing.
   *
   * @throws {RangeError} When no child has the id.
   */
  message(id: string, inbox: string, text: string, options: Sent & { key?: string | null } = {}): Promise<ChildStatus> {
    const child = this.#child(id);
    const param = child.flow.input.find((input) => input.name === inbox);
    const type = param?.contract?.type ?? null;
    const message = isInboxType(type) && typeEquals(type!.element!, MESSAGE);
    const from = options.from ?? null;
    const author = from === null ? {} : { author: from.label };
    const event = message ? { role: "user", ...author, parts: [{ type: "text", text }] } : text;
    const key = options.key ?? null;
    const sent = (log: string, delivery: DeliveryOptions) => send(log, inbox, event, { ...delivery, key });
    return this.#deliver(child, sent, options).then((status) =>
      from === null || message || status.note !== undefined
        ? status
        : { ...status, note: "sent as plain text: the run can't tell who wrote it" },
    );
  }

  /**
   * Close child `id`'s inbox `inbox`, so its turns end. Returns its status, and throws, as `answer`
   * does.
   *
   * @throws {RangeError} When no child has the id.
   */
  closeInbox(id: string, inbox: string, options: Sent = {}): Promise<ChildStatus> {
    return this.#deliver(this.#child(id), (log, delivery) => close(log, inbox, delivery), options);
  }

  /**
   * Cancel child `id`. Returns its status, and throws, as `answer` does; the note says when it had
   * ended.
   *
   * @throws {RangeError} When no child has the id.
   */
  stop(id: string, options: Sent = {}): Promise<ChildStatus> {
    const move = this.#options.cancel ?? cancel;
    return this.#deliver(this.#child(id), (log, delivery) => move(log, delivery), options);
  }

  /**
   * End the parent's hold on its children, by each one's published status: cancel each `running`
   * one; stop the driver of each `waiting` one, a pass at its `wakeAt` included, leaving its log to
   * resume; and resolve every waiting `next`. Returns the waiting children, by id. Later starts
   * throw, and later deliveries change nothing.
   */
  async dispose(): Promise<WaitingChild[]> {
    this.#disposed = true;
    const left: WaitingChild[] = [];
    const halted = new Promise<"halted">((resolve) => {
      if (this.#halting.signal.aborted) resolve("halted");
      this.#halting.signal.addEventListener("abort", () => resolve("halted"), { once: true });
    });
    const move = this.#options.cancel ?? cancel;
    const stopping = [...this.#children.values()].map(async (child) => {
      if (!this.#driving(child)) return;
      if (child.status.status === "running") {
        try {
          const cancelled = await Promise.race([move(child.log, this.#deliveryOptions(child)), halted]);
          if (cancelled === "halted") left.push({ run: child.id, kind: "halted", log: child.log });
          else this.nudge(child.id);
        } catch (error) {
          // One that ended meanwhile needs nothing; any other failure leaves the stop to the halt.
          if (!(error instanceof DeliveryRefused)) child.halt.abort();
        }
      } else {
        left.push({ run: child.id, kind: this.#waitKind(child), log: child.log });
        child.halt.abort();
      }
      // A halt ends the driver's pass; a pass slow to stop isn't waited for once halted.
      await Promise.race([child.done, halted]);
    });
    await Promise.all(stopping);
    this.cancelWaits();
    return left.sort((a, b) => Number(a.run.slice(1)) - Number(b.run.slice(1)));
  }

  /** @throws {RangeError} When no child has the id. */
  #child(id: string): Child {
    const child = this.#children.get(id);
    if (child === undefined) throw new YianRangeError(`no run '${id}'`);
    return child;
  }

  /** Whether child `id` was started by run `ownerRunId`, or by a child that run reaches. */
  #reaches(ownerRunId: string, id: string): boolean {
    let child = this.#children.get(id);
    while (child !== undefined) {
      if (child.parent.runId === ownerRunId) return true;
      const by = child.parent.runId;
      child = [...this.#children.values()].find((each) => each.runId === by);
    }
    return false;
  }

  /** Whether `child`'s driver may still publish a status. */
  #driving(child: Child): boolean {
    return !ENDED.has(child.status.status) && !child.halt.signal.aborted;
  }

  /** What waiting `child` waits on, by its last status. */
  #waitKind(child: Child): WaitingChild["kind"] {
    if ((child.status.asks ?? []).length > 0) return "question";
    if ((child.status.held ?? []).length > 0) return "held";
    const waits = (child.status.waits ?? []) as readonly { tool: string; settled: boolean }[];
    return waits.some((wait) => wait.tool === TAKE && !wait.settled) ? "inbox" : "timer";
  }

  /** The options a delivery to `child`'s log passes. */
  #deliveryOptions(child: Child): DeliveryOptions {
    return {
      env: this.#options.env,
      // A flow loaded confined to a folder is held to it; a bundled one was loaded with no root.
      codeRoot: child.flow.source?.codeRoot == null ? null : this.#options.codeRoot,
      // A flow built in code is the run's; one loaded from files is compiled again from the log.
      flow: child.flow.source === null ? child.flow : null,
    };
  }

  /**
   * Make `deliver`'s delivery to `child`'s log, by `options.from`, publish the child `running` and
   * nudge its driver; return its status. A refusal (the run's, or `options.refusal`'s text, asked
   * first), or a delivery after `dispose`, is a note on the status returned, and changes nothing;
   * `options.resent` is as `answer` says. Every error goes through `memories.scrub`.
   */
  async #deliver(
    child: Child,
    deliver: (log: string, options: DeliveryOptions) => Promise<unknown>,
    options: Sent & { refusal?: () => string | null } = {},
  ): Promise<ChildStatus> {
    // Its runs are left as they were, to resume later.
    if (this.#disposed) return { ...child.status, note: "the runs were disposed" };
    try {
      const refused = options.refusal?.() ?? null;
      if (refused !== null) return { ...child.status, note: refused };
      await deliver(child.log, { ...this.#deliveryOptions(child), by: options.from?.by ?? null });
    } catch (error) {
      const scrubbed = this.#options.memories.scrub(error, child.memory);
      if (!(error instanceof DeliveryRefused)) throw scrubbed;
      if (options.resent === true && LANDED.has(error.reason)) return child.status;
      return { ...child.status, note: (scrubbed as Error | null)?.message ?? String(scrubbed) };
    }
    // Until its pass settles, the child moves on what it was given.
    if (this.#driving(child)) this.#publish(child, { run: child.id, status: "running", log: child.log });
    this.nudge(child.id);
    return child.status;
  }

  /**
   * Drive `child` from its start to an end: a pass, then, while it waits, one more pass after
   * each nudge and at each `wakeAt`. `started` settles once the log holds the run's start, or
   * rejects with what stopped it before.
   */
  async #drive(
    child: Child,
    inputs: Readonly<Record<string, unknown>>,
    started: PromiseWithResolvers<void>,
  ): Promise<void> {
    const { env, models = null, scheduler, memories, beforeToolCall = null, advance: move = advance } = this.#options;
    const llmConfig = this.#llmConfig;
    const onEvent = (event: RunEvent) => {
      if (event instanceof RunStarted) {
        child.started = true;
        started.resolve();
      }
      // An interrupted reply succeeds with null when its output is `Optional(...)`.
      if (isReply(event) && !event.interrupted && event.output !== null) {
        child.fresh.push({ kind: "reply", text: writeAsText(event.output as Value) });
      }
    };
    const common = {
      tools: child.tools,
      skills: child.skills,
      onEvent,
      onConsole: this.#options.onConsole ?? null,
      signal: child.halt.signal,
      env,
      scheduler,
      models,
      beforeToolCall,
    };
    try {
      let result = await runFlow(child.flow, { ...inputs, ...child.memory }, {
        log: child.log,
        runId: child.runId,
        parent: child.parent,
        llmConfig: Object.keys(llmConfig).length > 0 ? llmConfig : null,
        ...common,
      });
      // Already settled by `RunStarted`; kept so that `start` can never hang.
      started.resolve();
      // The deadline the run was last moved at. A node is due once any of its calls passes its
      // deadline, so that pass times it out and `wakeAt` comes later; were it to stay, resting
      // until it would only spin.
      let spent: string | null = null;
      for (;;) {
        for (const drift of result.memoryDrift) child.fresh.push({ kind: "memory", text: memories.drift(drift) });
        this.#publish(child, this.#settledStatus(child, result));
        if (result.status !== "waiting") return;
        const next = wakeAt(result.waits);
        const wake: string | null = next === spent ? null : next;
        if (await this.#rest(child, wake)) spent = wake;
        if (child.halt.signal.aborted) return;
        result = await move({ log: child.log, ...this.#deliveryOptions(child), ...common, memory: child.memory });
      }
    } catch (thrown) {
      // A memory's open may name its folder or store.
      const error = memories.scrub(thrown, child.memory);
      // Before the start, `start` throws it; after, it is the child's failure.
      started.reject(error);
      // A halted driver leaves the run as it is, and a child that never started has no status.
      if (child.halt.signal.aborted || !child.started) return;
      this.#publish(child, {
        run: child.id,
        status: "failed",
        note: `couldn't move the run: ${(error as Error | null)?.message ?? String(error)}`,
        log: child.log,
      });
    } finally {
      // A run that never started leaves no folder behind.
      removeEmptyDir(dirname(child.log));
      this.#release(child);
    }
  }

  /**
   * Rest until a nudge, the child's halt or `wake` (a record time; `null` waits for a nudge only),
   * at once when a nudge came during the pass. Returns whether `wake` ended it.
   */
  async #rest(child: Child, wake: string | null): Promise<boolean> {
    if (child.dirty) {
      child.dirty = false;
      return false;
    }
    const rest = new AbortController();
    child.rest = rest;
    const halt = () => rest.abort();
    child.halt.signal.addEventListener("abort", halt, { once: true });
    if (child.halt.signal.aborted) rest.abort();
    try {
      if (wake === null) {
        if (!rest.signal.aborted) {
          await new Promise<void>((resolve) => rest.signal.addEventListener("abort", () => resolve(), { once: true }));
        }
        return false;
      }
      return await sleepUntil(this.#options.scheduler, recordTimeMs(wake, "wakeAt"), rest.signal).then(
        () => true,
        // A nudge or the halt.
        () => false,
      );
    } finally {
      child.halt.signal.removeEventListener("abort", halt);
      child.rest = null;
      child.dirty = false;
    }
  }

  /** The status `result` leaves `child` in. */
  #settledStatus(child: Child, result: RunResult): Unnumbered {
    const base = { run: child.id, log: child.log };
    if (result.status === "succeeded") return { ...base, status: "succeeded", output: result.output };
    if (result.status === "failed") {
      const failure = result.failure!;
      return { ...base, status: "failed", failure: { code: failure.code, message: failure.message, node: failure.nodeId } };
    }
    if (result.status === "cancelled") return { ...base, status: "cancelled" };
    // A pass over the whole log, once per settled pass: the asks are what the log says is open.
    const calls = pendingCalls(child.log);
    const asks = calls.filter((call) => !call.held).map((call) => ({
      call: call.callId,
      question: questionText(call.arguments),
    }));
    const held = calls.filter((call) => call.held).map(({ callId, tool, arguments: args }) => ({
      call: callId,
      tool,
      arguments: args,
    }));
    return {
      ...base,
      status: "waiting",
      // A held call is listed apart.
      waits: result.waits.map(({ held: _, ...wait }) => wait),
      wake_at: wakeAt(result.waits),
      ...(asks.length > 0 ? { asks } : {}),
      ...(held.length > 0 ? { held } : {}),
    };
  }

  /**
   * Make `status` `child`'s, numbered and with its unread replies and memory notes, unless it says
   * what the last one did and the pass said nothing new; tell `onStatus`, and wake its readers.
   */
  #publish(child: Child, status: Unnumbered): void {
    const fresh = child.fresh.splice(0);
    const seq = this.#seq + 1;
    const unread = [...child.unread, ...fresh.map((said) => ({ ...said, seq }))];
    const replies = unread.filter((said) => said.kind === "reply").map((said) => said.text);
    const notes = unread.filter((said) => said.kind === "memory").map((said) => said.text);
    const next: Unnumbered = {
      ...status,
      ...(replies.length > 0 ? { replies } : {}),
      ...(notes.length > 0 ? { memory_notes: notes } : {}),
    };
    const { seq: _, ...last } = child.status;
    if (fresh.length === 0 && isDeepStrictEqual(last, next)) return;
    this.#seq = seq;
    child.unread = unread;
    child.status = { ...next, seq };
    try {
      this.#options.onStatus?.(child.status);
    } catch {
      // A reader's fault mustn't stop the child's driver.
    }
    for (const waiter of [...child.waiters]) {
      if (seq > waiter.after) {
        child.waiters.delete(waiter);
        waiter.resolve(child.status);
      }
    }
  }

  /** Resolve every waiting `next` call on `child` with its current status. */
  #release(child: Child): void {
    const waiting = [...child.waiters];
    child.waiters.clear();
    for (const waiter of waiting) waiter.resolve(child.status);
  }
}

/** What an owner handle needs of its driver beyond the driver's public methods. */
interface OwnerCore {
  start(
    flow: ComposedFlow,
    inputs: Readonly<Record<string, unknown>>,
    options: ChildStartOptions,
  ): Promise<StartedChild>;
  reaches(id: string): boolean;
  readonly scheduler: Scheduler;
}

/**
 * One run's hold on the child flows it starts (`ChildFlows.owner`): it starts children, and reads
 * and delivers to the runs it reaches, the ones it started and the ones those started, by id. A run
 * it doesn't reach is as unknown to it as an id no child has. `childFlows` gives the owner's model
 * these acts as tools.
 */
export class ChildFlowOwner {
  readonly runId: string;
  readonly group: string;
  /** Its author label (`ChildFlowOwnerOptions.label`). */
  readonly label: string;
  /** The labels of the people who act through it. */
  readonly people: readonly string[];
  /** Its inbox, or `null` for none. */
  readonly inbox: string | null;
  readonly seen: NonNullable<ChildFlowOwnerOptions["seen"]> | null;
  readonly #driver: ChildFlows;
  readonly #core: OwnerCore;
  /** Who its model's deliveries are from. */
  readonly #from: ChildSender;
  /** By run, the last status seq this owner's tools showed its model, whichever group showed it. */
  readonly #shown = new Map<string, number>();

  /** @internal `ChildFlows.owner` makes owners. */
  constructor(driver: ChildFlows, options: ChildFlowOwnerOptions, owned: Owned, core: OwnerCore) {
    this.runId = owned.runId;
    this.group = options.group;
    this.label = owned.label;
    this.people = owned.people;
    this.inbox = owned.inbox;
    this.seen = options.seen ?? null;
    this.#driver = driver;
    this.#core = core;
    this.#from = modelSender(owned);
  }

  /**
   * Start `flow` with `inputs` as the driver's next child (`r1`, `r2`, ..., numbered across its
   * owners): return its id, its log and the `seq` of its first status (`running`) once the log holds
   * the run's start, while the run goes on in the background; `wait` then sees every later status.
   *
   * Starts run one at a time in call order, so ids follow it, and an id is taken only by a start
   * that succeeds. Each first runs the host's `preflight`, then binds the flow's memory inputs
   * (`memories.bind`) for `options.project`; every pass of the child gets the same instances.
   *
   * The child records as its parent this owner's run, `options.node`, this owner's inbox, and its
   * label: `options.label`, or `<flow> r<n>`, its flow's id (or `options.name`) reduced to
   * `A-Za-z0-9._-` and cut at 40 characters, then its id (`review_flow r1`).
   *
   * @throws {RangeError} When `options.label` is refused, before anything is written.
   * @throws What `preflight` throws, and what `memories.bind` throws as `memories.scrub` leaves it,
   *   before anything is written.
   * @throws {Error} When the runs were disposed; the signal's reason once it aborted; what `runFlow`
   *   throws before the run starts (a tool the flow names that the run lacks, say), as
   *   `memories.scrub` leaves it, with no log left behind.
   */
  start(
    flow: ComposedFlow,
    inputs: Readonly<Record<string, unknown>>,
    options: ChildStartOptions = {},
  ): Promise<StartedChild> {
    return this.#core.start(flow, inputs, options);
  }

  /** Whether this owner reaches child `id`: it started it, or a run it reaches did. */
  reaches(id: string): boolean {
    return this.#core.reaches(id);
  }

  /** As `ChildFlows.status`. @throws {RangeError} When this owner doesn't reach run `id`. */
  status(id: string): ChildStatus {
    return this.#driver.status(this.#mine(id));
  }

  /**
   * Resolve with run `id`'s latest status once it is numbered past `options.after` (by default its
   * current status's), its driver stops, `options.seconds` pass on the children's clock, or
   * `options.signal` aborts.
   *
   * @throws {RangeError} When this owner doesn't reach run `id`.
   */
  async wait(
    id: string,
    options: { after?: number; seconds?: number | null; signal?: AbortSignal | null } = {},
  ): Promise<ChildStatus> {
    this.#mine(id);
    const { after, seconds = null, signal = null } = options;
    const done = new AbortController();
    const end = () => done.abort();
    signal?.addEventListener("abort", end, { once: true });
    if (signal?.aborted) end();
    if (seconds !== null) this.#core.scheduler.sleep(seconds * 1000, done.signal).then(end, () => {});
    try {
      return await this.#driver.next(id, after, { signal: done.signal });
    } finally {
      signal?.removeEventListener("abort", end);
      end();
    }
  }

  /** As `ChildFlows.answer`, from this owner's model. @throws {RangeError} When it doesn't reach run `id`. */
  answer(id: string, call: string, value: unknown, options: Resend = {}): Promise<ChildStatus> {
    return this.#driver.answer(this.#mine(id), call, value, { ...options, from: this.#from });
  }

  /** As `ChildFlows.message`, from this owner's model. @throws {RangeError} When it doesn't reach run `id`. */
  message(id: string, inbox: string, text: string, options: Resend & { key?: string | null } = {}): Promise<ChildStatus> {
    return this.#driver.message(this.#mine(id), inbox, text, { ...options, from: this.#from });
  }

  /** As `ChildFlows.closeInbox`, from this owner's model. @throws {RangeError} When it doesn't reach run `id`. */
  closeInbox(id: string, inbox: string, options: Resend = {}): Promise<ChildStatus> {
    return this.#driver.closeInbox(this.#mine(id), inbox, { ...options, from: this.#from });
  }

  /** As `ChildFlows.stop`, from this owner's model. @throws {RangeError} When it doesn't reach run `id`. */
  stop(id: string, options: Resend = {}): Promise<ChildStatus> {
    return this.#driver.stop(this.#mine(id), { ...options, from: this.#from });
  }

  /** @internal The last status seq of run `id` this owner's model has seen: shown, or told `seen`. */
  lastSeen(id: string): number {
    return Math.max(this.#shown.get(id) ?? 0, this.seen?.seen(id) ?? 0);
  }

  /**
   * @internal Note that the owner's model is shown `status`, marking it read with `seen`; return it
   * without the replies and memory notes the model was shown before.
   */
  show(status: ChildStatus): ChildStatus {
    const last = this.lastSeen(status.run);
    if (this.seen !== null) {
      this.#driver.markRead(status.run, status.seq);
      this.seen.markSeen(status.run, status.seq);
    }
    this.#shown.set(status.run, Math.max(status.seq, last));
    // The stored status keeps the replies and notes it was published with; once shown, they are read.
    const { replies: _, memory_notes: __, ...unrepeated } = status;
    return status.seq <= last ? unrepeated : status;
  }

  /** @throws {RangeError} When this owner doesn't reach run `id`. */
  #mine(id: string): string {
    if (!this.reaches(id)) throw new YianRangeError(`no run '${id}'`);
    return id;
  }
}

/**
 * Where in `value` an object that reads as a `Message` (`checkMessage`) carries an author or an
 * action part, as a value path; `null` when none does. The value is read, not the type the question
 * declares: an untyped answer reaches an agent as a message too. Only the host signs a message.
 */
function signedMessageAt(value: unknown, path: readonly (string | number)[] = []): (string | number)[] | null {
  if (typeof value !== "object" || value === null) return null;
  const { author, parts } = value as { author?: unknown; parts?: unknown };
  const action = Array.isArray(parts) ? parts.findIndex((part) => (part as { type?: unknown } | null)?.type === "action") : -1;
  // The shape first, as it is cheap; then the whole value must be a `Message`.
  if (((author !== undefined && author !== null) || action !== -1) && isMessage(value)) {
    return author !== undefined && author !== null ? [...path, "author"] : [...path, "parts", action];
  }
  for (const [key, item] of Array.isArray(value) ? value.entries() : Object.entries(value)) {
    const at = signedMessageAt(item, [...path, key]);
    if (at !== null) return at;
  }
  return null;
}

/**
 * `value` with each user message in it that has no author signed by `label`: every object, at any
 * depth, that reads as a user `Message` (`checkMessage`) with no `author`. What the owner's model
 * passes speaks as the owner, never as the child's own user. A message with an author is
 * `signedMessageAt`'s to refuse.
 */
function signedBy(value: unknown, label: string): unknown {
  // Only JSON's arrays and objects: a host's value, such as a `FileRef`, stays as it is.
  if (!Array.isArray(value) && !isRecord(value)) return value;
  const { role, author } = value as { role?: unknown; author?: unknown };
  const unsigned = role === "user" && (author === undefined || author === null);
  if (unsigned && isMessage(value)) return { ...value, author: label };
  if (Array.isArray(value)) return value.map((item) => signedBy(item, label));
  return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, signedBy(item, label)]));
}

/** Whether `value` reads as a `Message`. */
function isMessage(value: unknown): boolean {
  try {
    buildValueAs(MESSAGE, value);
    return true;
  } catch (error) {
    if (error instanceof TypeCheckError) return false;
    throw error;
  }
}

/** The note refusing a value that signs a message at `path`. */
function signedNote(path: readonly (string | number)[]): string {
  return `a message you pass can't carry an author or an action part (at ${renderValuePath(path)})`;
}

/**
 * Return `inputs` once they could start `flow`, as its start would check them: no unknown names, every
 * required one given, each value of its declared type; and no message in them signed, as an answer's
 * may not be. A memory input is never given: the host binds it to its default (`memories.bind`).
 *
 * @throws {ReportedToolError} `bad inputs: <why>`.
 */
function checkStartInputs(flow: ComposedFlow, inputs: unknown): Readonly<Record<string, unknown>> {
  try {
    const given = typeof inputs === "object" && inputs !== null ? inputs : {};
    const memory = flow.input.find((param) => isMemoryType(param.contract?.type) && Object.hasOwn(given, param.name));
    if (memory !== undefined) {
      throw new YianRangeError(`\`${memory.name}\` is a memory; leave it out, and the run gets its default`);
    }
    const coerced = coerceInputs(flow.input, hostInputs(flow.input, inputs));
    for (const param of flow.input) {
      if (isInboxType(param.contract?.type ?? null) || isMemoryType(param.contract?.type)) continue;
      if (!Object.hasOwn(coerced, param.name)) {
        if (!param.required || param.hasDefault) continue;
        throw new YianRangeError(`required input ${reprText(param.name)} is missing`);
      }
      if (param.contract === null) continue;
      try {
        buildValueAs(param.contract.type, coerced[param.name]);
      } catch (error) {
        if (!(error instanceof Error)) throw error;
        // Name where in the value it failed, as the start's own check does.
        const at = error instanceof TypeCheckError && error.path.length > 0 ? ` at ${renderValuePath(error.path)}` : "";
        throw new YianTypeError(`input ${reprText(param.name)}${at}: ${error.message}`);
      }
    }
    const signed = signedMessageAt(given);
    if (signed !== null) throw new YianRangeError(signedNote(signed));
    return inputs as Readonly<Record<string, unknown>>;
  } catch (error) {
    if (error instanceof Error) throw new ReportedToolError(`bad inputs: ${error.message}`);
    throw error;
  }
}

/**
 * The tools a run's model drives the child flows `owner` reaches with, for the host to pass under
 * `owner.group` in that run's tools, so the model sees `<group>_status`, and so on: `status` and
 * `wait` read a run's status, and `message`, `answer`, `close_inbox` and `stop` deliver to it.
 *
 * With `options.start`, the group has `start` too, first. A call names a flow and its inputs; the
 * host's `resolve` finds the flow, a library is refused, the inputs are checked as the run's start
 * would check them (and may sign no message), the host's `approve` is asked, and only then does the
 * owner start it. A library, a refused approval or a start that fails returns `not started: <why>`;
 * bad inputs are a `ReportedToolError`, `bad inputs: <why>`. A crash-cut start isn't sent again,
 * since it may have started. No text names another tool of the group, since a host may pass only
 * some of them.
 *
 * Each returns the run's status as `shownStatus` does; once the owner showed them, its replies and
 * memory notes aren't shown again, and with `owner.seen` they are marked read. A run the owner
 * doesn't reach, or an argument of the wrong type, is a `ReportedToolError`. The deliveries are
 * `resumable`: a resumed agent sends a crash-cut call again under its call id, which keys
 * `message`'s send, and a repeated answer, close or stop changes nothing; a call sent again
 * (`ctx.attempt` above 1) is `resent` (`ChildFlows.answer`).
 */
export function childFlows<T extends ChildFlowTarget>(
  owner: ChildFlowOwner,
  options: ChildFlowGroupOptions<T> = {},
): FunctionTool[] {
  const { start = null } = options;
  const run = "The run's id (r1, r2, ...).";
  const status =
    "Returns the run's status, a record: status (running, waiting, succeeded, failed or cancelled), " +
    "output, failure, waits, asks (questions, each with its call id), held (tool calls it holds for " +
    "approval, which no answer settles), replies (what it said since you last looked), " +
    "memory_notes (a memory it created or that changed, since you last looked), note (something " +
    "that couldn't be done) and log.";
  const show = (given: ChildStatus) => shownStatus(owner.show(given));
  const known = (args: Record<string, unknown>) => {
    const id = stringArg(args, "run");
    if (!owner.reaches(id)) throw new ReportedToolError(`no run '${id}'`);
    return id;
  };
  const runArguments = (...more: [name: string, description: string][]) =>
    stringArguments(["run", run, true], ...more.map(([name, text]): [string, string, boolean] => [name, text, true]));
  return [
    ...(start === null ? [] : [startTool(start)]),
    tool({
      name: "status",
      description: `Read where a run you started stands now, without waiting. ${status}`,
      parameters: runArguments(),
      isConcurrencySafe: () => true,
      execute: (args) => show(owner.status(known(args))),
    }),
    tool({
      name: "wait",
      description:
        "Wait for a run you started to change (a reply, a question, its end), up to `seconds` (at " +
        `most ${WAIT_MAX_S}), and return its status then, or at the timeout. Returns at once when it ` +
        `changed since you last saw it. ${status}`,
      parameters: {
        type: "object",
        properties: {
          run: { type: "string", description: run },
          seconds: { type: "number", description: `How long to wait at most, in seconds (at most ${WAIT_MAX_S}).` },
        },
        required: ["run", "seconds"],
        additionalProperties: false,
      },
      isConcurrencySafe: () => true,
      execute: async (args, ctx) => {
        const id = known(args);
        const seconds = args["seconds"];
        if (typeof seconds !== "number" || !Number.isFinite(seconds) || seconds < 0) {
          throw new ReportedToolError("seconds must be a number of seconds, 0 or more");
        }
        const after = owner.lastSeen(id);
        return show(await owner.wait(id, { after, seconds: Math.min(seconds, WAIT_MAX_S), signal: ctx.signal }));
      },
    }),
    tool({
      name: "message",
      description:
        "Send text to an Inbox input of a run you started, as a message to a flow that takes them; " +
        `its status's waits name the inbox it waits on. ${status}`,
      parameters: runArguments(["inbox", "The Inbox input's name."], ["text", "The message."]),
      resumable: true,
      execute: async (args, ctx) => {
        const id = known(args);
        const sent = { key: ctx.callId, resent: ctx.attempt > 1 };
        return show(await owner.message(id, stringArg(args, "inbox"), stringArg(args, "text"), sent));
      },
    }),
    tool({
      name: "answer",
      description:
        "Answer a question a run you started asks (its status's asks), by its call id, with the answer " +
        "of the person it is for: don't make one up, and don't answer a question twice. The answer is " +
        `any JSON value but null; text for a plain question. ${status}`,
      parameters: {
        type: "object",
        properties: {
          run: { type: "string", description: run },
          call: { type: "string", description: "The question's call id, from the status's asks." },
          answer: { description: "Any JSON value; text for a plain question." },
        },
        required: ["run", "call", "answer"],
        additionalProperties: false,
      },
      resumable: true,
      execute: async (args, ctx) => {
        const id = known(args);
        const call = stringArg(args, "call");
        const answer = args["answer"];
        if (answer === null || answer === undefined) throw new ReportedToolError("answer can't be null");
        return show(await owner.answer(id, call, answer, { resent: ctx.attempt > 1 }));
      },
    }),
    tool({
      name: "close_inbox",
      description:
        "Close an Inbox input of a run you started, so the flow stops waiting for messages there and " +
        `can finish. ${status}`,
      parameters: runArguments(["inbox", "The Inbox input's name."]),
      resumable: true,
      execute: async (args, ctx) => {
        const id = known(args);
        return show(await owner.closeInbox(id, stringArg(args, "inbox"), { resent: ctx.attempt > 1 }));
      },
    }),
    tool({
      name: "stop",
      description: `Cancel a run you started; use it only when asked. ${status}`,
      parameters: runArguments(),
      resumable: true,
      execute: async (args, ctx) => show(await owner.stop(known(args), { resent: ctx.attempt > 1 })),
    }),
  ];

  /** The group's `start`, over the host's `hooks`. */
  function startTool(hooks: ChildFlowStart<T>): FunctionTool {
    return tool({
      name: "start",
      description:
        `${hooks.description} An Inbox input is never given here: message the run once it waits on it. ` +
        `${status} The status's run is the id the group's other tools take. Returns \`not started: <why>\` ` +
        "when the start is refused or can't happen; don't retry a refused start unchanged.",
      parameters: {
        type: "object",
        properties: {
          flow: { type: "string", description: "The flow to start, as the tool's description says." },
          inputs: { type: "object", description: "The flow's inputs by name; {} for none." },
          content: { type: "string", description: "A draft: the flow's whole text, run in place of what flow names." },
        },
        required: ["flow", "inputs"],
        additionalProperties: false,
      },
      execute: async (args, ctx) => {
        const content = args["content"] === undefined ? null : stringArg(args, "content");
        const target = await hooks.resolve({ flow: stringArg(args, "flow"), content });
        const { flow, name, project = null } = target;
        if (flow.library !== null) return `not started: ${libraryRunMessage(flow)}`;
        const checked = checkStartInputs(flow, args["inputs"]);
        const inputs = signedBy(checked, owner.label) as Readonly<Record<string, unknown>>;
        const refused = await hooks.approve(target, inputs, ctx.signal);
        if (refused !== null) return `not started: ${refused}`;
        let id: string;
        try {
          ({ id } = await owner.start(flow, inputs, { signal: ctx.signal, name, project, node: ctx.node }));
        } catch (error) {
          if (error instanceof Error) return `not started: ${error.message}`;
          throw error;
        }
        return show(owner.status(id));
      },
    });
  }
}
