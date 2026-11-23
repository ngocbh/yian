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
 * asks its host which flow a call names and whether it may start (`ChildFlowStart`). A group's
 * `approve` shows the owner's own `beforeToolCall` the child's held call it would decide, read from
 * the child's log, and never runs without that check.
 *
 * The host names who speaks: an owner by its label, which signs its model's messages to its children
 * and names it on each delivery its model makes (`by`), and a child by its own, `<flow> r<n>` unless
 * the host gives one. No label may pass for another. A person acts on a child through its owner
 * (`ChildFlowOwner.press`): their words reach it relayed by the owner, and the owner's run gets an
 * action saying what they did.
 *
 * A child whose owner has an inbox reports to it after each pass with news, signed by its label:
 * what it said, asks or holds, its end, and what others than its owner told it, read from its log.
 * A run a child starts through an owner with no inbox reports where that child does. Reports wait
 * while the owner's group tools run, and what a report or a tool's status told the owner's model
 * is read for both. The driver moves an owner's run on a report only when it drives that run; the
 * host hears of each one (`onReport`) to move its own, and may hold them while a person has the
 * floor (`ChildFlowOwner.hold`).
 *
 * A child's `ask` returns `pending()`, so the child waits durably for the user's answer, which the
 * host relays with `answer`. The children's stop is their own, tied to the host's exit by
 * `dispose`, never to the call that started them.
 *
 * Knows about: the run boundary (`runFlow`, `advance`, the deliveries, `pendingCalls`, `inspectRun`,
 * `runLogPath`, the gate's hook, its input intake and library refusal), the run log's records and
 * their reader, input coercion, the waits' `wakeAt` and seqs, the platform's scheduler, folder
 * tidying and warnings, compose, memory's base type, the run events, the host tool API and its
 * errors, the tool registry and the `pending` marker, skills, types and values.
 */

import { randomUUID } from "node:crypto";
import { statSync } from "node:fs";
import { dirname } from "node:path";
import { isDeepStrictEqual } from "node:util";

import type { ComposedFlow } from "./compose/build/pipeline.ts";
import type { Env } from "./defaults.ts";
import { reprText } from "./diagnostics.ts";
import { NodeSucceeded, type RunEvent, RunStarted } from "./events.ts";
import type { ModelAccess } from "./llm_clients/factory.ts";
import type { Author, MemoryBase } from "./memory/base.ts";
import { coerceInputs } from "./nodes/binding.ts";
import { isSystemError, removeEmptyDir, type Scheduler, sleepUntil, warn } from "./platform.ts";
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
  inspectRun,
  interrupt,
  type InterruptResult,
  type JsonObject,
  type JsonValue,
  libraryRunMessage,
  loadRecordedFlow,
  type MemoryDrift,
  type PendingCall,
  pendingCalls,
  release,
  RunBusyError,
  type RunInspection,
  type RunResult,
  runFlow,
  runLogPath,
  send,
  type SendOptions,
  withdraw,
  type WithdrawnEvent,
} from "./run.ts";
import {
  iterRunRecords,
  NodeStepRecord,
  PoolCommitRecord,
  recordTimeMs,
  type RunParent,
} from "./runtime/log.ts";
import { seqList, wakeAt } from "./runtime/waits.ts";
import type { Skill } from "./skills/catalog.ts";
import { pending, TAKE, type ToolContext } from "./tools/context.ts";
import { ReportedToolError } from "./tools/errors.ts";
import { checkGateText } from "./tools/gate.ts";
import {
  type FunctionTool,
  type StandardSchemaV1,
  stringArg,
  stringArguments,
  tool,
  withExecute,
} from "./tools/function_tools.ts";
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
  TypedValue,
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

/** How many more times a report or a press's action is sent to an owner's log while another writer holds it. */
const OWNER_SEND_RETRIES = 3;

/** How long a send to an owner's busy log waits before it tries again, in milliseconds. */
const OWNER_SEND_RETRY_MS = 100;

/**
 * How many of an owner run's latest reports the driver remembers what each told, so that a hold
 * that takes one back unsent can give its news back.
 */
const SENT_REPORTS_KEPT = 100;

/** @internal The longest a group's `wait` waits, in seconds. */
export const WAIT_MAX_S = 120;

/** The most characters of an `approve`'s reason, so its release note fits `GATE_TEXT_MAX_BYTES`. */
const REASON_MAX_CHARS = 1000;

/** A control character other than a line break or a tab, which an `approve`'s reason may not hold. */
const REASON_CONTROL = /[^\P{Cc}\n\t]/u;

/**
 * What an owner's model reads for an `approve` in a run with no `beforeToolCall`: nothing would check
 * the approval, so only a person gives one.
 */
const APPROVE_UNGATED =
  "Only the user can approve or refuse a held call: ask them in the chat, or they can open the run and approve it.";

/** The group's tools a person may press, each with what a failed press's summary calls its call. */
const PRESSES: Readonly<Record<keyof PressArguments, string>> = {
  message: "message to",
  answer: "answer to",
  interrupt: "interrupt of",
  close_inbox: "inbox close on",
  stop: "stop of",
  approve: "decision on",
};

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
   * The seq of the last record of the child's log once the pass that left it so settled: its
   * report's key (`report:<run id>:<at>`). Absent while `running`, and on a failure of the driver
   * itself. A later pass that changes nothing else moves it on.
   */
  readonly at?: number;
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

/** What `ChildFlows.interrupt` did: the child's status, the sends it took back, and what the interrupt did. */
export interface ChildInterrupt extends InterruptResult {
  /** The child's status, as `ChildFlows.answer` returns it. */
  readonly status: ChildStatus;
  /** The caller's unread sends it took back, in send order, which the run never reads. */
  readonly taken: readonly WithdrawnEvent[];
  /**
   * Whether the interrupt reached the run. When `false` (the run refused it or the take-back, or a
   * fault stopped it after the take-back), the status's `note` says why, and the rest is empty.
   */
  readonly landed: boolean;
  /**
   * When it named no agent but left a Stop for the next one to read the inbox: the unread sends to
   * the inbox from others than the caller, which that agent's turn may use up unanswered; else 0.
   */
  readonly queued: number;
}

/** The tools of an owner's group a person may press (`ChildFlowOwner.press`), each with its arguments. */
export interface PressArguments {
  readonly message: { readonly run: string; readonly inbox: string; readonly text: string };
  readonly answer: { readonly run: string; readonly call: string; readonly answer: unknown };
  readonly interrupt: { readonly run: string; readonly inbox: string };
  readonly close_inbox: { readonly run: string; readonly inbox: string };
  readonly stop: { readonly run: string };
  readonly approve: {
    readonly run: string;
    readonly call: string;
    readonly decision: "approve" | "refuse";
    readonly reason?: string | null;
  };
}

/**
 * What a press records in place of its own, each field when given: to keep a secret the person
 * typed out of the owner's run, say. A press that failed keeps its own summary and no result, and
 * takes only `arguments`.
 */
export interface PressRecord {
  /** A line saying what the person did, such as `Ada typed into shell 1`. */
  readonly summary?: string | null;
  /** The arguments to record; `null` records none, and no result either, as it may echo them. */
  readonly arguments?: JsonObject | null;
  /** What to record the person did; `null` records nothing. */
  readonly result?: string | null;
}

/** How a person presses a tool (`ChildFlowOwner.press`). */
export interface PressOptions {
  /** The person's label: one of the owner's `people`. Check who they are before you press. */
  readonly person: string;
  /** What to record in place of what the press would; `null` (the default) for its own. */
  readonly record?: PressRecord | null;
}

/**
 * Why a press's action isn't in its owner's run: a refusal of its send (`unknown_inbox` and
 * `invalid_value` before anything runs, for an inbox that doesn't take messages), or `no_inbox` (the
 * owner has none), `wrong_log` (its log is another run's), `busy` (its log stayed busy) or `failed`
 * (a fault, once the run had the press).
 */
export type PressUnrecorded = DeliveryRefusal | "no_inbox" | "wrong_log" | "busy" | "failed";

/** Why the driver refused a delivery: the run's refusal, or its own. */
type ChildRefusal = DeliveryRefusal | "disposed" | "signed_message" | "held_call";

/** What a press did. */
export interface Pressed<R> {
  /** What the driver's method returned, the run's status; `null` when the press ran nothing. */
  readonly result: R | null;
  /** Whether the owner's run holds the press's action. */
  readonly recorded: boolean;
  /** Why it doesn't, or `null` when it does. */
  readonly reason: PressUnrecorded | null;
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
   * The owner's inbox, an `Inbox(Message)` of its run, which each child records in its parent and
   * reports to after each pass with news (`childFlows`). `null` (the default) for none: the children
   * of a run this driver started then report where that run does.
   */
  readonly inbox?: string | null;
  /**
   * The owner's run log, where its `inbox` is, which its children's reports and a person's presses
   * are sent to. An owner with an inbox needs one; for a run this driver started, it is that run's
   * log by default.
   */
  readonly log?: string | null;
  /**
   * How reports and a press's action are sent to the owner's log, as the host calls take them:
   * `env` (the driver's by default), `codeRoot` (the recorded one by default) and `flow`, the
   * owner's flow, needed when its run was built in code.
   */
  readonly delivery?: {
    readonly env?: Env | null;
    readonly codeRoot?: string | null;
    readonly flow?: ComposedFlow | null;
  } | null;
  /**
   * The key the host passes the owner's tools under (`childFlows`), which their texts name them
   * by: `<group>_status`, and so on.
   */
  readonly group: string;
  /**
   * Told what becomes of each report (`ChildReport.outcome`): for a host that moves the owner's run
   * itself (the driver moves only the runs it started), or shows the reports. What it throws is
   * dropped.
   */
  readonly onReport?: ((report: ChildReport) => void) | null;
}

/** A report, and what became of it (`ChildFlowOwnerOptions.onReport`). */
export interface ChildReport {
  /**
   * `sent`: it is in the owner's log. `held`: it came due while the host holds the reports
   * (`ChildFlowOwner.hold`), and is sent once the hold is let go. `refused`: the owner's run has
   * ended or closed its inbox, so it wasn't sent, and its news stays unread for `status` and `wait`.
   */
  readonly outcome: "sent" | "held" | "refused";
  /** The send's seq in the owner's log once `sent`, else `null`. */
  readonly seq: number | null;
  /** The child's run id (`<owner's run id>-r<n>`). */
  readonly runId: string;
  /** The child's label, which signs it. */
  readonly label: string;
  /** Its line, which the owner's model reads, and its card (`childEvent`). */
  readonly event: ChildEvent;
}

/** A host's hold on an owner run's reports (`ChildFlowOwner.hold`). */
export interface ReportHold {
  /** The reports the owner's inbox held unread, taken back, oldest first: `release` sends them again. */
  readonly taken: readonly TakenReport[];
  /**
   * Send the reports taken back again, then let the held ones go; resolves once they are sent, or
   * failed (`ChildFlowOwner.hold`). A second call does nothing.
   */
  release(): Promise<void>;
}

/** A report taken back from its owner's inbox (`ReportHold.taken`). */
export interface TakenReport {
  /** The seq of the send taken back. */
  readonly seq: number;
  /** The child's run id. */
  readonly runId: string;
  /** The child's label, which signs it. */
  readonly label: string;
  /** Its line. */
  readonly text: string;
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
  /** Its owner, whose own sends are no news to it. */
  readonly owner: Owned;
  /** Where its reports go, from the owner handle that started it; `null` when that owner has no inbox. */
  readonly reportTo: ReportTo | null;
  /** Where the driver's read of its log stopped: the end of the last record read, and its seq (-1 for none). */
  readonly read: { end: number; seq: number };
  /** The sends to it that no take or steer has read yet and that would be news to its owner, by seq. */
  readonly sends: Map<number, HeardSend>;
  /** The sends of `sends` a take or a steer has read since its last report, in the order read. */
  readonly heard: HeardSend[];
  /** What its owner's model was last told of it, by a report or a status its owner's tools showed. */
  told: Told;
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
  /** Its reports a hold took back whose sending again failed, in order: they go before its next report. */
  readonly again: Taken[];
  /** Whether the run's log holds its start. */
  started: boolean;
  /** The child's own stop: it stops the driver and leaves the log as it is. */
  readonly halt: AbortController;
  /** Settles once the driver has stopped. */
  done: Promise<void>;
}

/**
 * A send to a child that its owner didn't make, which a child's report says it heard: the inbox, the
 * message's author (`null` for a value that has none), who made the send (`by`; `null` for no one)
 * and its text.
 */
export interface HeardSend {
  readonly inbox: string;
  readonly author: string | null;
  readonly by: Author | null;
  readonly text: string;
}

/** What an owner's model was told of a child: the status's seq, its open calls, and whether it ended. */
interface Told {
  readonly seq: number;
  readonly calls: ReadonlySet<string>;
  readonly ended: boolean;
}

/**
 * A report in its owner's inbox: its event, the child's replies and memory notes sending it marked
 * read, and what the owner's model was told of the child before it and with it.
 */
interface SentReport {
  readonly event: ChildEvent;
  readonly read: readonly (Said & { readonly seq: number })[];
  readonly before: Told;
  readonly after: Told;
}

/** A report a hold took back, to send again: the send's seq, value and `by`, its child, and what it told. */
interface Taken {
  readonly seq: number;
  readonly value: unknown;
  readonly by: Author | null;
  readonly child: Child;
  readonly sent: SentReport | null;
}

/** Where a send to an owner goes: its run, the run's log and inbox, and how the send is made. */
interface OwnerLog {
  readonly runId: string;
  readonly log: string;
  readonly inbox: string;
  readonly delivery: DeliveryOptions;
}

/** Where the reports of the children an owner handle starts go, and the run's reports they join. */
interface ReportTo {
  readonly owner: OwnerLog;
  readonly reports: Reports;
  readonly onReport: ChildFlowOwnerOptions["onReport"];
  /** Set once the owner's run has ended or closed its inbox: the news then stays unread. */
  gone: boolean;
}

/** The reports of an owner run's children, which the group tools of all its handles hold. */
interface Reports {
  readonly runId: string;
  /** How many of the owner's group tools and host holds there are now: no report is sent meanwhile. */
  busy: number;
  /** How many of those are the host's holds (`ChildFlowOwner.hold`): a report due meanwhile is told as held. */
  held: number;
  /** The children a pass of which settled since their last look, in the order they settled. */
  readonly due: Set<Child>;
  /** The children whose report failed to send: they are due again at the owner's next group tool's end. */
  readonly stalled: Set<Child>;
  /** The reports being sent now, or `null`. */
  sending: Promise<void> | null;
  /** The latest reports sent, by their seq in the owner's log, up to `SENT_REPORTS_KEPT`. */
  readonly sent: Map<number, SentReport>;
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

/** @internal A child's tool call held for approval, as its log lists it, with the child's run id and label. */
export interface HeldChildCall {
  readonly runId: string;
  readonly label: string;
  readonly call: PendingCall;
}

/** How a delivery is made: who from, and again or not. */
interface Sent extends Resend {
  /**
   * @internal Who it is from: the owner's model, as `ChildFlowOwner` passes it. `null`, the
   * default, for the host itself, which signs nothing and records no one.
   */
  readonly from?: ChildSender | null;
  /** @internal Told the code of the refusal the returned status's note says. */
  readonly onRefused?: ((reason: ChildRefusal) => void) | null;
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

/** One child event: the line the owner's model reads, the status it came from, and its card. */
export interface ChildEvent {
  /** The child's id among its driver's children (`r1`). */
  readonly run: string;
  /** The seq of the status it came from (`ChildStatus.seq`). */
  readonly seq: number;
  /** The line. */
  readonly text: string;
  /** The event as a host's view draws it. */
  readonly card: EventCard;
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
 * it now does (asks, with each question's call id, or holds calls for approval; waits for a
 * message on an inbox; succeeded, failed or was cancelled), then what it heard from others
 * (`heard`), what it said and what the host noted about its memories; child text is JSON-quoted,
 * so the line stays one line, and the line is cut at `EVENT_BODY_CHARS` with a note.
 */
export function childEvent(status: ChildStatus, heard: readonly HeardSend[] = []): ChildEvent | null {
  if (status.status === "running") return null;
  // Each clause is its verb and what follows it, which starts with a space when it isn't empty.
  const clauses: [string, string][] = [];
  const quote = (value: unknown) => writeAsJson((value ?? null) as Value);
  const listed = (items: readonly string[], all = "its status shows all") => {
    const more = items.length - LIST_ITEMS;
    return [...items.slice(0, LIST_ITEMS), ...(more > 0 ? [`(${more} more; ${all})`] : [])].join(", ");
  };
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
  } else {
    const asks = status.asks ?? [];
    const held = status.held ?? [];
    if (asks.length > 0) clauses.push(["asks", ` ${listed(asks.map((ask) => `(call ${ask.call}): ${quote(ask.question)}`))}`]);
    if (held.length > 0) {
      const calls = held.map((call) => `(call ${call.call}): ${call.tool} ${quote(call.arguments)}`);
      clauses.push(["holds for approval", ` ${listed(calls)}`]);
    }
    const waits = (status.waits ?? []) as readonly Wait[];
    const inbox = waits.find((wait) => wait.tool === TAKE && wait.settled !== true && typeof wait.inbox === "string");
    if (asks.length + held.length === 0 && inbox !== undefined) {
      clauses.push(["waits for a message", ` on inbox ${quote(inbox.inbox)}`]);
    }
  }
  if (heard.length > 0) {
    // A message's author is who it speaks as; the send's `by`, who made it, is said too when it differs.
    const from = ({ author, by }: HeardSend) => {
      if (author === null) return by === null ? "" : ` from ${quote(by.name)}`;
      return ` from ${quote(author)}${by === null || by.name === author ? "" : ` (sent by ${by.kind} ${quote(by.name)})`}`;
    };
    const sends = heard.map((send) => `${from(send)} on inbox ${quote(send.inbox)}: ${quote(send.text)}`.trim());
    clauses.push(["heard", ` ${listed(sends, "its log has all")}`]);
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
 * Who a person's press through `owner` is from: signed `<person>, via <owner's label>`, and recorded
 * as the person relayed by the owner's run, so their interrupt takes back only their own sends
 * through this owner.
 */
function relaySender(person: string, owner: Owned): ChildSender {
  const by: Author = { kind: "user", name: person, run: owner.runId };
  return Object.freeze({ label: `${person}, via ${owner.label}`, by: Object.freeze(by) });
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
   * its starter's people, the labels above it, its log, and where it reports.
   */
  readonly #owners = new Map<
    string,
    Omit<Owned, "runId" | "inbox"> & { readonly log: string; readonly reportTo: ReportTo | null }
  >();
  /** The reports of each owner run with an inbox, by its run id. */
  readonly #reports = new Map<string, Reports>();
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
   *   the label differs from the one this driver gave the run, the inbox or the log isn't a
   *   non-empty string or `null`, or it has an inbox and no log.
   */
  owner(options: ChildFlowOwnerOptions): ChildFlowOwner {
    const { runId, inbox = null } = options;
    const started = this.#owners.get(runId);
    const log = options.log ?? started?.log ?? null;
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
    if (log !== null && (typeof log !== "string" || log === "")) {
      throw new YianRangeError("an owner's log must be a non-empty string or null");
    }
    if (inbox !== null && log === null) throw new YianRangeError("an owner with an inbox needs its run's log");
    const owned: Owned = Object.freeze({ runId, label, people, inbox, heard: started?.heard ?? [] });
    const { env = null, codeRoot = null, flow = null } = options.delivery ?? {};
    const delivery: DeliveryOptions = Object.freeze({ env: env ?? this.#options.env, codeRoot, flow });
    const onReport = options.onReport ?? null;
    // With no inbox, its children report where its run does.
    const reportTo: ReportTo | null =
      inbox === null
        ? (started?.reportTo ?? null)
        : { owner: { runId, log: log!, inbox, delivery }, reports: this.#reportsOf(runId), onReport, gone: false };
    // The runs that report to it, whichever of its handles, or of its children's, started them.
    const mine = (id: string) => {
      const child = this.#children.get(id);
      return child !== undefined && child.reportTo?.owner.runId === runId ? child : null;
    };
    return new ChildFlowOwner(this, { ...options, log }, owned, {
      start: (flow, inputs, given) => this.#start(owned, reportTo, flow, inputs, given),
      reaches: (id) => this.#reaches(runId, id),
      held: (id, call) => this.#heldCall(id, call),
      scheduler: this.#options.scheduler,
      delivery,
      hold: () => this.#hold(this.#reports.get(runId) ?? null),
      holdReports: () => this.#holdReports(inbox === null ? null : reportTo),
      unread: (id, seq) => this.#unreadAt(id, seq),
      shown: (status) => {
        const child = mine(status.run);
        if (child !== null) this.#told(child, status);
      },
      told: (id) => mine(id)?.told.seq ?? 0,
    });
  }

  /** The reports of owner run `runId`'s children: one set for the run, whichever of its handles asks. */
  #reportsOf(runId: string): Reports {
    const fresh: Reports = { runId, busy: 0, held: 0, due: new Set(), stalled: new Set(), sending: null, sent: new Map() };
    const made = this.#reports.get(runId) ?? fresh;
    this.#reports.set(runId, made);
    return made;
  }

  /**
   * Hold `reports` while a group tool of their owner runs: return the reports being sent, which the
   * tool waits for, and what to call once it ends. That lets them go, with those whose send failed.
   */
  #hold(reports: Reports | null): { readonly sending: Promise<void> | null; release(): void } {
    if (reports === null) return { sending: null, release: () => {} };
    reports.busy += 1;
    const release = () => {
      reports.busy -= 1;
      for (const child of reports.stalled) reports.due.add(child);
      reports.stalled.clear();
      this.#flush(reports);
    };
    return { sending: reports.sending, release };
  }

  /** A host's hold on the reports `to` sends, as `ChildFlowOwner.hold` says; one doing nothing for none. */
  async #holdReports(to: ReportTo | null): Promise<ReportHold> {
    if (to === null) return { taken: [], release: async () => {} };
    const { owner, reports } = to;
    const hold = this.#hold(reports);
    reports.held += 1;
    const letGo = (): void => {
      reports.held -= 1;
      hold.release();
    };
    const taken: Taken[] = [];
    try {
      if (hold.sending !== null) await hold.sending;
      // What came due before the hold is held now.
      for (const child of new Set([...reports.due, ...reports.stalled])) this.#tellHeld(child, child.reportTo!);
      for (const child of this.#children.values()) {
        if (child.reportTo?.owner.runId !== owner.runId) continue;
        const by: Author = { kind: "run", name: child.parent.label, run: child.runId };
        try {
          for (const { seq, value } of await withdraw(owner.log, owner.inbox, { ...owner.delivery, by })) {
            taken.push({ seq, value, by, child, sent: reports.sent.get(seq) ?? null });
          }
        } catch (error) {
          if (!(error instanceof DeliveryRefused)) throw error;
        }
      }
    } catch (error) {
      letGo();
      throw error;
    }
    // The others were read.
    reports.sent.clear();
    taken.sort((a, b) => a.seq - b.seq);
    let releasing: Promise<void> | null = null;
    const release = async (): Promise<void> => {
      const failed = new Map<Child, unknown>();
      for (const item of taken) {
        const { child } = item;
        const to = child.reportTo!;
        // Behind one of its child's that failed, as the reports keep their order.
        if (failed.has(child)) child.again.push(item);
        else if (to.gone) this.#giveBack(item, to);
        else {
          const error = await this.#sendAgain(item, to);
          if (error !== null) {
            child.again.push(item);
            failed.set(child, error);
          }
        }
      }
      letGo();
      // Due again at the owner's next group tool's end, not at this one.
      for (const [child, error] of failed) this.#stall(child, child.reportTo!, error);
      if (reports.sending !== null) await reports.sending;
    };
    return {
      taken: taken.map(({ seq, value, child }) => ({
        seq,
        runId: child.runId,
        label: child.parent.label,
        text: messageText(value) ?? "",
      })),
      release: () => (releasing ??= release()),
    };
  }

  /**
   * Send `item`, a report a hold took back, again to `to`: told as sent, or as refused once the
   * owner's run has ended or closed its inbox, its news then given back. Returns why it failed
   * otherwise, or `null`.
   */
  async #sendAgain(item: Taken, to: ReportTo): Promise<unknown> {
    const { child } = item;
    // A withdrawn send keeps its key, so it goes again under a key of its own.
    const key = `report:${child.runId}:again:${item.seq}`;
    const sent = await sendToOwner(to.owner, item.value, { by: item.by, key }, this.#options.scheduler);
    if (sent.reason === "ended" || sent.reason === "inbox_closed") {
      to.gone = true;
      this.#giveBack(item, to);
      return null;
    }
    if (sent.reason !== null) return sent.error;
    if (item.sent !== null) remember(to.reports, sent.seq, item.sent);
    this.#movesOn(to);
    tell(to, { outcome: "sent", seq: sent.seq, runId: child.runId, label: child.parent.label, event: eventOf(item) });
    return null;
  }

  /**
   * Give back the news of `item`, a report taken back that won't be sent again, as the owner's run
   * has ended or closed its inbox: what it marked read is unread again, and what it told is untold
   * unless the child was told more since. Tell `to`'s host it was refused.
   */
  #giveBack(item: Taken, to: ReportTo): void {
    const { child, sent } = item;
    if (sent !== null) {
      const kept = new Set(child.unread);
      child.unread = [...sent.read.filter((said) => !kept.has(said)), ...child.unread].sort((a, b) => a.seq - b.seq);
      if (child.told === sent.after) child.told = sent.before;
    }
    tell(to, { outcome: "refused", seq: null, runId: child.runId, label: child.parent.label, event: eventOf(item) });
  }

  /** Move on the owner's run `to` sends to when this driver drives it; its host moves any other. */
  #movesOn(to: ReportTo): void {
    const owner = [...this.#children.values()].find((each) => each.runId === to.owner.runId);
    if (owner !== undefined) this.#moving(owner);
  }

  /** Child `id`'s replies and memory notes published at or before status `seq` that no reader marked read. */
  #unreadAt(id: string, seq: number): Pick<ChildStatus, "replies" | "memory_notes"> {
    const unread = this.#child(id).unread.filter((said) => said.seq <= seq);
    const replies = unread.filter((said) => said.kind === "reply").map((said) => said.text);
    const notes = unread.filter((said) => said.kind === "memory").map((said) => said.text);
    return { ...(replies.length > 0 ? { replies } : {}), ...(notes.length > 0 ? { memory_notes: notes } : {}) };
  }

  /** Give the children started from now on the provider and model layer `layer`. */
  setLlmConfig(layer: Readonly<Record<string, string>>): void {
    this.#llmConfig = layer;
  }

  /** Start `flow` as the next child of `owner`, reporting as `reportTo` says, as `ChildFlowOwner.start` says. */
  #start(
    owner: Owned,
    reportTo: ReportTo | null,
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
    const turn = this.#starting.then(() => this.#begin(owner, reportTo, flow, inputs, given, signal));
    this.#starting = turn.catch(() => {});
    return turn;
  }

  /** Start `flow` as the next child of `owner`, once the starts called before it are done. */
  async #begin(
    owner: Owned,
    reportTo: ReportTo | null,
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
    this.#owners.set(runId, Object.freeze({ label, people: owner.people, heard, log, reportTo }));
    const tools = typeof given.tools === "function" ? given.tools(runId) : given.tools;
    const seq = ++this.#seq;
    const child: Child = {
      id,
      runId,
      parent: Object.freeze({ runId: owner.runId, node, inbox: owner.inbox, label }),
      owner,
      reportTo,
      read: { end: 0, seq: -1 },
      sends: new Map(),
      heard: [],
      told: { seq: 0, calls: new Set(), ended: false },
      again: [],
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
      if (signed !== null) return { reason: "signed_message" as const, note: signedNote(signed) };
      if (this.#held(child, call) === null) return null;
      return { reason: "held_call" as const, note: `call '${call}' is a held tool call, not a question` };
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
   * Interrupt child `id`'s inbox `inbox`, as `interrupt` does: the agents reading it as their steer
   * inbox stop their current step, and their open calls and released calls not yet run are closed;
   * the run goes on (`stop` ends it). With no agent reading it now, the next one to read it stops at
   * once, though it may already have taken another sender's message, which then goes unanswered
   * (`queued`). First, so the cut agents never read them, take back the unread sends to `inbox`
   * that `options.from` made: never anyone else's, so none without it, as the host's own sends
   * are recorded as no one's. Returns its status, as `answer` does, with the sends taken back and
   * the interrupt's result (`landed` says whether it happened). When the take-back landed but the
   * interrupt then fails, the fault is the status's note, so what came back is still returned.
   *
   * @throws {RangeError} When no child has the id.
   * @throws What a delivery fault throws, as `answer` says, when nothing was taken back.
   */
  async interrupt(id: string, inbox: string, options: Sent = {}): Promise<ChildInterrupt> {
    const child = this.#child(id);
    const by = options.from?.by ?? null;
    let taken: readonly WithdrawnEvent[] = [];
    let cut = null as InterruptResult | null;
    const none = { seq: null, nodes: [], calls: [], released: [], queued: 0, landed: false };
    let status: ChildStatus;
    try {
      status = await this.#deliver(
        child,
        async (log, delivery) => {
          if (by !== null) taken = await withdraw(log, inbox, delivery);
          cut = await interrupt(log, inbox, delivery);
          return taken.length > 0 || cut.seq !== null;
        },
        options,
      );
    } catch (error) {
      if (taken.length === 0) throw error;
      // The take-back may have reopened a take.
      this.nudge(id);
      const note = (error as Error | null)?.message ?? String(error);
      return { ...none, taken, status: { ...child.status, note } };
    }
    if (cut === null) {
      if (taken.length > 0) this.nudge(id);
      return { ...none, taken, status };
    }
    // A Stop left standing: the next reader may take these before it stops.
    let queued = 0;
    if (cut.seq !== null && cut.nodes.length === 0) {
      try {
        const unread = inspectRun(child.log).unread;
        queued = unread.filter((send) => send.inbox === inbox && !isDeepStrictEqual(send.by, by)).length;
      } catch (error) {
        // The interrupt landed, so a fault counting them is only a note.
        const scrubbed = this.#options.memories.scrub(error, child.memory);
        status = { ...status, note: (scrubbed as Error | null)?.message ?? String(scrubbed) };
      }
    }
    return { ...cut, landed: true, queued, status, taken };
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
   * Decide child `id`'s held call `call` (`ChildStatus.held`): `approve` releases it, so the child's
   * agent runs it once as its model asked it, with `options.reason` as the release's note; `refuse`
   * answers it with the error `Refused by <label>: <reason>`, which that model reads, `<label>`
   * being `options.from`'s (just `Refused` without it). Returns its status, and throws, as `answer`
   * does. A call its log doesn't list as held, such as a question, is refused with a note, since
   * refusing it would answer it. The child's log records `options.from`'s `by` as who decided.
   *
   * @throws {RangeError} When no child has the id, or `options.reason` is empty, has a lone
   *   surrogate or is over `GATE_TEXT_MAX_BYTES` UTF-8 bytes.
   * @throws {TypeError} When `decision` is neither `approve` nor `refuse`.
   */
  approve(
    id: string,
    call: string,
    decision: "approve" | "refuse",
    options: Sent & { reason?: string | null } = {},
  ): Promise<ChildStatus> {
    const child = this.#child(id);
    if (decision !== "approve" && decision !== "refuse") throw new YianTypeError("decision must be approve or refuse");
    const reason = options.reason ?? null;
    if (reason !== null) checkGateText(reason, "reason");
    const from = options.from ?? null;
    const refused = from === null ? "Refused" : `Refused by ${from.label}`;
    const error = reason === null ? `${refused}.` : `${refused}: ${reason}`;
    const note = `call '${call}' isn't a held tool call`;
    const refusal = () => (this.#held(child, call) === null ? { reason: "not_held" as const, note } : null);
    const decide = (log: string, delivery: DeliveryOptions) =>
      decision === "approve" ? release(log, call, { ...delivery, note: reason }) : complete(log, call, { error }, delivery);
    return this.#deliver(child, decide, { ...options, refusal });
  }

  /**
   * End the parent's hold on its children, by each one's published status: cancel each `running`
   * one; stop the driver of each `waiting` one, a pass at its `wakeAt` included, leaving its log to
   * resume; and resolve every waiting `next`. Returns the waiting children, by id, once the
   * reports being sent are, or `haltAll` is called. Later starts throw, later deliveries change
   * nothing, and no report is sent.
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
    await Promise.race([Promise.all([...this.#reports.values()].map((reports) => reports.sending)), halted]);
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

  /** `child`'s call `call` as its log lists it now, when a host's check holds it; else `null`. */
  #held(child: Child, call: string): PendingCall | null {
    if (!child.started) return null;
    return pendingCalls(child.log).find((each) => each.callId === call && each.held) ?? null;
  }

  /** `OwnerCore.held`. @throws {RangeError} When no child has the id. */
  #heldCall(id: string, call: string): HeldChildCall | null {
    const child = this.#child(id);
    let held: PendingCall | null;
    try {
      held = this.#held(child, call);
    } catch (error) {
      throw this.#options.memories.scrub(error, child.memory);
    }
    return held === null ? null : { runId: child.runId, label: child.parent.label, call: held };
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
   * nudge its driver; return its status. A delivery that resolves `false` changed nothing, so the
   * status is returned as it is. A refusal (the run's, or `options.refusal`'s, asked first), or a
   * delivery after `dispose`, is a note on the status returned, and changes nothing; its code goes
   * to `options.onRefused`. `options.resent` is as `answer` says. Every error goes through
   * `memories.scrub`.
   */
  async #deliver(
    child: Child,
    deliver: (log: string, options: DeliveryOptions) => Promise<unknown>,
    options: Sent & { refusal?: () => { reason: ChildRefusal; note: string } | null } = {},
  ): Promise<ChildStatus> {
    const refuse = (reason: ChildRefusal, note: string) => {
      options.onRefused?.(reason);
      return { ...child.status, note };
    };
    // Its runs are left as they were, to resume later.
    if (this.#disposed) return refuse("disposed", "the runs were disposed");
    try {
      const refused = options.refusal?.() ?? null;
      if (refused !== null) return refuse(refused.reason, refused.note);
      const moved = await deliver(child.log, { ...this.#deliveryOptions(child), by: options.from?.by ?? null });
      if (moved === false) return child.status;
    } catch (error) {
      const scrubbed = this.#options.memories.scrub(error, child.memory);
      if (!(error instanceof DeliveryRefused)) throw scrubbed;
      if (options.resent === true && LANDED.has(error.reason)) return child.status;
      return refuse(error.reason, (scrubbed as Error | null)?.message ?? String(scrubbed));
    }
    this.#moving(child);
    return child.status;
  }

  /** Publish `child` `running` and nudge its driver, as something it was given moves it. */
  #moving(child: Child): void {
    // Until its pass settles, the child moves on what it was given.
    if (this.#driving(child)) this.#publish(child, { run: child.id, status: "running", log: child.log });
    this.nudge(child.id);
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
        this.#settled(child);
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
      this.#settled(child);
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
    const base = { run: child.id, log: child.log, at: this.#scan(child) };
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
    const { seq: _, at: __, ...last } = child.status;
    const { at, ...unchanged } = next;
    if (fresh.length === 0 && isDeepStrictEqual(last, unchanged)) {
      // A pass that changed nothing else moves the status's `at` on, unpublished.
      if (at !== undefined) child.status = { ...child.status, at };
      return;
    }
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

  /**
   * Read `child`'s log on from where the last read stopped; return the seq of its last record. For
   * a child that reports to an owner still there, note each send to it that would be news to the
   * owner (`sends`), and each of those a take or a steer then read (`heard`). A withdrawn send stays
   * in `sends`, as nothing reads it.
   */
  #scan(child: Child): number {
    const { read } = child;
    const after = read.seq < 0 ? {} : { afterOffset: read.end, runId: child.runId, afterSeq: read.seq };
    const start = child.flow.compiled.startId;
    const payloads = child.reportTo !== null && !child.reportTo.gone;
    for (const { seq, end, envelope } of iterRunRecords(child.log, { ...after, payloads })) {
      read.end = end;
      read.seq = seq;
      const record = envelope?.record;
      if (record instanceof PoolCommitRecord && record.take !== null) this.#read(child, record.take.events);
      else if (record instanceof NodeStepRecord && record.step["kind"] === "steer") {
        this.#read(child, seqList(record.step["from"]) ?? []);
      } else if (record instanceof PoolCommitRecord && record.value !== null && record.nodeId === start) {
        // A send is the START's commit under `<start>.<inbox>`; the run start's own are under other keys.
        const heard = record.key.startsWith(`${start}.`) ? this.#news(child, record) : null;
        if (heard !== null) child.sends.set(seq, heard);
      }
    }
    return read.seq;
  }

  /** Note that a take or a steer of `child` read its sends `seqs`: those news to its owner are heard. */
  #read(child: Child, seqs: readonly number[]): void {
    for (const seq of seqs) {
      const send = child.sends.get(seq);
      if (send === undefined) continue;
      child.sends.delete(seq);
      child.heard.push(send);
    }
  }

  /**
   * The send `record` to `child` as its report says it heard it, or `null` when it is no news to
   * the child's owner: the owner's model made it, or the model of the run it reports to, a person
   * acting through the owner did, or it is a report of the child's own child.
   */
  #news(child: Child, record: PoolCommitRecord): HeardSend | null {
    const { by } = record;
    const { owner } = child;
    if (by?.kind === "agent" && by.name === owner.label && by.run === owner.runId) return null;
    if (by?.kind === "agent" && by.run === child.reportTo?.owner.runId) return null;
    if (by?.kind === "user" && owner.people.includes(by.name)) return null;
    const own = (each: Child) => each.runId === by?.run && each.parent.runId === child.runId;
    if (by?.kind === "run" && [...this.#children.values()].some(own)) return null;
    const value = TypedValue.fromWire(record.value).toObject();
    const author = isRecord(value) && typeof value["author"] === "string" ? value["author"] : null;
    const text = cutStatusText(messageText(value) ?? writeAsText(value));
    return { inbox: record.key.slice(record.nodeId.length + 1), author, by, text };
  }

  /** Have `child`'s owner look for news in its pass that settled, once none of its group tools runs. */
  #settled(child: Child): void {
    const to = child.reportTo;
    if (to === null || to.gone || this.#disposed) return;
    to.reports.due.add(child);
    if (to.reports.held > 0) this.#tellHeld(child, to);
    this.#flush(to.reports);
  }

  /** Tell `to`'s host of `child`'s report, if it has news, as held. */
  #tellHeld(child: Child, to: ReportTo): void {
    let pending: Pending | null;
    try {
      pending = this.#pending(child);
    } catch {
      return;
    }
    if (pending === null) return;
    const { event } = pending;
    tell(to, { outcome: "held", seq: null, runId: child.runId, label: child.parent.label, event });
  }

  /**
   * What `child`'s report would say now, or `null` when it has no news: a reply, a question or held
   * call its owner's model wasn't told of, the run's end, or a send of another's it read.
   *
   * @throws What building its line throws.
   */
  #pending(child: Child): Pending | null {
    const status = child.status;
    // A delivery moved it on: the pass that follows reports it all.
    if (status.status === "running") return null;
    const ended = ENDED.has(status.status);
    const heard = child.heard.slice();
    const said = (kind: Said["kind"]) => child.unread.filter((each) => each.kind === kind).map((each) => each.text);
    const [replies, notes] = [said("reply"), said("memory")];
    const asked = [...openCalls(status)].some((call) => !child.told.calls.has(call));
    if (heard.length === 0 && replies.length === 0 && !asked && (!ended || child.told.ended)) return null;
    const { replies: _, memory_notes: __, ...rest } = status;
    const unread = { ...(replies.length > 0 ? { replies } : {}), ...(notes.length > 0 ? { memory_notes: notes } : {}) };
    return { status, heard, event: childEvent({ ...rest, ...unread }, heard)! };
  }

  /** Start sending `reports`' due reports, unless they're being sent or a group tool of the owner runs. */
  #flush(reports: Reports): void {
    if (reports.busy > 0 || reports.sending !== null || reports.due.size === 0) return;
    reports.sending = this.#drain(reports).finally(() => {
      reports.sending = null;
      // A child that settled as the last send ended.
      this.#flush(reports);
    });
  }

  /** Report each due child in turn, while no group tool of the owner runs. */
  async #drain(reports: Reports): Promise<void> {
    for (const child of reports.due) {
      if (reports.busy > 0 || this.#disposed) return;
      reports.due.delete(child);
      await this.#report(child, child.reportTo!);
    }
  }

  /**
   * Send `child`'s owner what its model wasn't told of the child, when that holds news: a reply, a
   * question or held call it wasn't told of, the run's end, or a send of another's the child read.
   * The report is the child's event line (`childEvent`), signed by the child's label, recorded as
   * sent by the child's run and keyed on its status's `at`. When the owner's run has ended or closed
   * its inbox, the news stays unread. When the send fails otherwise, even after a busy log is tried
   * again, it stays unread too, with a warning, and goes again when one of the owner's group tools
   * next ends, or the child next settles.
   */
  async #report(child: Child, to: ReportTo): Promise<void> {
    // Its reports a hold took back go first.
    while (child.again.length > 0) {
      const item = child.again[0]!;
      if (to.gone) this.#giveBack(item, to);
      else {
        const error = await this.#sendAgain(item, to);
        if (error !== null) return this.#stall(child, to, error);
      }
      child.again.shift();
    }
    if (to.gone) return;
    let pending: Pending | null;
    try {
      pending = this.#pending(child);
    } catch (error) {
      return this.#stall(child, to, error);
    }
    if (pending === null) return;
    const { status, heard, event } = pending;
    const label = child.parent.label;
    const by: Author = { kind: "run", name: label, run: child.runId };
    // A failure of the driver itself isn't in the log, so nothing could derive its report again.
    const key = status.at === undefined ? null : `report:${child.runId}:${status.at}`;
    const message = { role: "user", author: label, parts: [{ type: "text", text: event.text }] };
    const sent = await sendToOwner(to.owner, message, { by, key }, this.#options.scheduler);
    const told = { runId: child.runId, label, event };
    if (sent.reason === "ended" || sent.reason === "inbox_closed") {
      to.gone = true;
      return tell(to, { ...told, outcome: "refused", seq: null });
    }
    if (sent.reason !== null) return this.#stall(child, to, sent.error);
    child.heard.splice(0, heard.length);
    const before = child.told;
    const read = child.unread.filter((said) => said.seq <= status.seq);
    this.#told(child, status);
    remember(to.reports, sent.seq, { event, read, before, after: child.told });
    this.#movesOn(to);
    tell(to, { ...told, outcome: "sent", seq: sent.seq });
  }

  /** Keep `child`'s news, whose report `error` stopped, for the next end of an owner's group tool, with a warning. */
  #stall(child: Child, to: ReportTo, error: unknown): void {
    to.reports.stalled.add(child);
    const why = (error as Error | null)?.message ?? String(error);
    warn(`run ${child.id}'s report to run ${to.owner.runId} wasn't sent, so its news stays unread for now: ${why}`);
  }

  /** Note that `child`'s owner's model was told `status`, by a report or a tool's status: what it holds is read. */
  #told(child: Child, status: ChildStatus): void {
    this.markRead(child.id, status.seq);
    if (status.status === "running" || status.seq < child.told.seq) return;
    child.told = { seq: status.seq, calls: openCalls(status), ended: ENDED.has(status.status) };
  }
}

/** A report a child would send now: the status it reports, the sends it heard, and its event. */
interface Pending {
  readonly status: ChildStatus;
  readonly heard: readonly HeardSend[];
  readonly event: ChildEvent;
}

/** Remember `report`, sent to `reports`' owner at `seq`, forgetting the oldest beyond `SENT_REPORTS_KEPT`. */
function remember(reports: Reports, seq: number, report: SentReport): void {
  reports.sent.set(seq, report);
  if (reports.sent.size > SENT_REPORTS_KEPT) reports.sent.delete(reports.sent.keys().next().value!);
}

/**
 * The event of `item`, a report taken back: the one it was sent with, or, for one the driver no
 * longer remembers, its line on a card of the child's status.
 */
function eventOf(item: Taken): ChildEvent {
  if (item.sent !== null) return item.sent.event;
  const text = messageText(item.value) ?? "";
  const { status } = item.child;
  return { run: item.child.id, seq: status.seq, text, card: { status: status.status, verb: "reported", body: text } };
}

/** Tell `to`'s host what became of a report (`ChildFlowOwnerOptions.onReport`), dropping what it throws. */
function tell(to: ReportTo, report: ChildReport): void {
  try {
    to.onReport?.(report);
  } catch {
    // A host's fault mustn't stop the reports.
  }
}

/** What `sendToOwner` did: the send's seq, or why it didn't send, with the error. */
type OwnerSend =
  | { readonly seq: number; readonly reason: null }
  | { readonly reason: DeliveryRefusal | "wrong_log" | "busy" | "failed"; readonly error: unknown };

/**
 * Send `message`, a report or a press's action, to `owner`'s inbox by `sent.by` under `sent.key`,
 * trying again on `scheduler` while another writer holds the log: the send's seq once it is sent,
 * else why not, with the error: the refusal's reason, `wrong_log` when the log is another run's,
 * `busy` once the retries are spent, or `failed`. It never throws.
 */
async function sendToOwner(
  owner: OwnerLog,
  message: unknown,
  sent: Pick<SendOptions, "by" | "key">,
  scheduler: Scheduler,
): Promise<OwnerSend> {
  const { runId, log, inbox, delivery } = owner;
  for (let tried = 0; ; tried += 1) {
    try {
      // A send to another run's log would land: its first record names its run.
      for (const { runId: found } of iterRunRecords(log, { payloads: false })) {
        if (found !== runId) {
          return { reason: "wrong_log", error: new YianRangeError(`${log} is the log of run ${reprText(found)}`) };
        }
        break;
      }
      return { seq: await send(log, inbox, message, { ...delivery, ...sent }), reason: null };
    } catch (error) {
      if (error instanceof DeliveryRefused) return { reason: error.reason, error };
      if (!(error instanceof RunBusyError)) return { reason: "failed", error };
      if (tried === OWNER_SEND_RETRIES) return { reason: "busy", error };
      await scheduler.sleep(OWNER_SEND_RETRY_MS);
    }
  }
}

/** Resolve once `sending`, which never rejects, is done, or reject with `signal`'s reason once it aborts. */
function untilSent(sending: Promise<void>, signal: AbortSignal): Promise<void> {
  if (signal.aborted) return Promise.reject(signal.reason);
  return new Promise((resolve, reject) => {
    const onAbort = () => reject(signal.reason);
    signal.addEventListener("abort", onAbort, { once: true });
    void sending.then(() => {
      signal.removeEventListener("abort", onAbort);
      resolve();
    });
  });
}

/** The call ids of `status`'s questions and held calls. */
function openCalls(status: ChildStatus): ReadonlySet<string> {
  return new Set([...(status.asks ?? []), ...(status.held ?? [])].map((each) => each.call));
}

/** What an owner handle needs of its driver beyond the driver's public methods. */
interface OwnerCore {
  start(
    flow: ComposedFlow,
    inputs: Readonly<Record<string, unknown>>,
    options: ChildStartOptions,
  ): Promise<StartedChild>;
  reaches(id: string): boolean;
  /** Child `id`'s held call `call`, or `null`; what reading its log throws goes through `memories.scrub`. */
  held(id: string, call: string): HeldChildCall | null;
  readonly scheduler: Scheduler;
  /** How reports and a press's action are sent to the owner's log (`ChildFlowOwnerOptions.delivery`). */
  readonly delivery: DeliveryOptions;
  /** Hold the owner's reports while a group tool runs: the reports being sent, and what lets them go. */
  hold(): { readonly sending: Promise<void> | null; release(): void };
  /** The host's hold on the owner's reports (`ChildFlowOwner.hold`). */
  holdReports(): Promise<ReportHold>;
  /** Child `id`'s replies and memory notes up to status `seq` that no reader marked read. */
  unread(id: string, seq: number): Pick<ChildStatus, "replies" | "memory_notes">;
  /** Note that a tool of the owner showed its model `status`, and returned it. */
  shown(status: ChildStatus): void;
  /** The seq of the last status of the owner's child `id` that its model was told, by a report or a tool; else 0. */
  told(id: string): number;
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
  /** Its run's log, or `null` for none. */
  readonly log: string | null;
  readonly #driver: ChildFlows;
  readonly #core: OwnerCore;
  /**
   * What its presses are signed and recorded by, kept apart from the public fields, which a caller
   * could change: its run, checked labels and inbox, its log and its group.
   */
  readonly #owned: Owned;
  readonly #log: string | null;
  readonly #group: string;
  /** How a press's action is sent to its log. */
  readonly #delivery: DeliveryOptions;
  /** Who its model's deliveries are from. */
  readonly #from: ChildSender;
  /** By run, the last status seq this owner's tools showed its model, whichever group showed it. */
  readonly #shown = new Map<string, number>();
  /**
   * Why its inbox can't take an action, read once from its run's flow: `null` when it can, and
   * `undefined` until read.
   */
  #inboxRefusal: "unknown_inbox" | "invalid_value" | null | undefined = undefined;

  /** @internal `ChildFlows.owner` makes owners. */
  constructor(driver: ChildFlows, options: ChildFlowOwnerOptions, owned: Owned, core: OwnerCore) {
    this.runId = owned.runId;
    this.group = options.group;
    this.label = owned.label;
    this.people = owned.people;
    this.inbox = owned.inbox;
    this.log = options.log ?? null;
    this.#driver = driver;
    this.#core = core;
    this.#owned = owned;
    this.#log = this.log;
    this.#group = this.group;
    this.#delivery = core.delivery;
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

  /** As `ChildFlows.interrupt`, from this owner's model. @throws {RangeError} When it doesn't reach run `id`. */
  interrupt(id: string, inbox: string): Promise<ChildInterrupt> {
    return this.#driver.interrupt(this.#mine(id), inbox, { from: this.#from });
  }

  /** As `ChildFlows.closeInbox`, from this owner's model. @throws {RangeError} When it doesn't reach run `id`. */
  closeInbox(id: string, inbox: string, options: Resend = {}): Promise<ChildStatus> {
    return this.#driver.closeInbox(this.#mine(id), inbox, { ...options, from: this.#from });
  }

  /** As `ChildFlows.stop`, from this owner's model. @throws {RangeError} When it doesn't reach run `id`. */
  stop(id: string, options: Resend = {}): Promise<ChildStatus> {
    return this.#driver.stop(this.#mine(id), { ...options, from: this.#from });
  }

  /**
   * @internal As `ChildFlows.approve`, from this owner's model, unchecked: only the group's gated
   * `approve` calls it.
   *
   * @throws {RangeError} When it doesn't reach run `id`.
   */
  approve(
    id: string,
    call: string,
    decision: "approve" | "refuse",
    options: { reason?: string | null } = {},
  ): Promise<ChildStatus> {
    return this.#driver.approve(this.#mine(id), call, decision, { ...options, from: this.#from });
  }

  /**
   * @internal Run `id`'s call `call` as its log lists it now, when the children's check holds it;
   * else `null`.
   *
   * @throws {RangeError} When this owner doesn't reach run `id`.
   * @throws What reading the run's log throws, as `memories.scrub` leaves it.
   */
  heldCall(id: string, call: string): HeldChildCall | null {
    return this.#core.held(this.#mine(id), call);
  }

  /**
   * Hold the reports to this owner's run while a person has the floor, as after they interrupt its
   * agent: none is sent until the hold is let go, each that comes due is told to `onReport` as
   * `held`, and its news stays unread for `status` and `wait`. Resolves once a report being sent
   * has landed and the reports its inbox holds unread are taken back (`withdraw` by each child's
   * `by`), so its agent reads none of them now. `release` sends those again, each under a key of
   * its own (`report:<child's run id>:again:<seq taken back>`) and told to `onReport` as sent, then
   * lets the held ones go. One the owner's run refuses, as it has ended or closed its inbox, is told
   * as refused, and its news is unread again; one that fails otherwise stays with the driver, with a
   * warning, and goes before its child's next report. A report taken back and never sent again
   * stays read. An owner with no inbox holds nothing.
   *
   * @throws What taking a report back throws, other than a refusal (the run has ended, say).
   */
  hold(): Promise<ReportHold> {
    return this.#core.holdReports();
  }

  /**
   * Press the group's tool `tool` as `options.person`, one of this owner's people, with `args` as the
   * tool takes them: never gated (check who the person is first), and marking no status read. The
   * run gets the person's words as they are, signed `<person>, via <label>` and recorded as theirs
   * relayed by this owner's run (`by: {kind: "user", name, run}`), so their `interrupt` takes back
   * only their own sends through this owner.
   *
   * Then this owner's `inbox` gets the press's action, from the person, which its agent reads
   * without a turn: `tool` (`<group>_<tool>`), `arguments`, and `result`, only what the person did
   * (`sent`, `answered`, `interrupted (took back 1 message)`, `nothing to interrupt (...)`,
   * `closed`, `stopped`, `approved`, `refused`), never what the run said. A press the run refuses,
   * or whose tool throws, records the `summary` `<person>'s message to r1 was refused (<reason>)` (or
   * `failed`) instead, never the refusal's message, and no result but an interrupt's take-back.
   * `options.record` is as `PressRecord` says.
   *
   * Before it runs anything, it returns `{result: null, recorded: false, reason}` when this owner
   * can't take the action: it has no inbox (`no_inbox`), its run hasn't started (`not_started`), the
   * log is another run's (`wrong_log`), the run has ended (`ended`), or the inbox isn't one of its
   * flow's (`unknown_inbox`), takes no `Message` (`invalid_value`) or is closed (`inbox_closed`).
   * Press a run a child started through the top owner, which reaches it.
   *
   * Once the run has the press, it doesn't throw. Each press has a key of its own, which keys its
   * action and, for `message`, its message to the run: a busy log is tried again (3 more times), then
   * it returns `recorded: false` (`busy`); a refused send gives the refusal's reason, and a fault
   * `failed`. A crash between the run's delivery and the action leaves the run's record by the person
   * with no action in this owner's run, which picking the run back up must reconcile; only a
   * `message`'s record holds the press's key to match it by.
   *
   * @throws {TypeError} When `tool` isn't one of these, or an argument, the person or a field of
   *   `options.record` is of the wrong type.
   * @throws {RangeError} When the person isn't one of this owner's people, `reason` is refused as
   *   `ChildFlows.approve` refuses it, this owner doesn't reach the run, or its run's flow was built
   *   in code and `delivery.flow` isn't given (or was recorded and is).
   * @throws What reading this owner's log throws, and what the tool throws before the run has the
   *   press, once its failure is recorded.
   */
  async press<K extends keyof PressArguments>(
    tool: K,
    args: PressArguments[K],
    options: PressOptions,
  ): Promise<Pressed<K extends "interrupt" ? ChildInterrupt : ChildStatus>> {
    const given = pressArguments(tool, args);
    const { person } = options;
    const owned = this.#owned;
    if (typeof person !== "string") throw new YianTypeError("a press's person must be a string");
    if (!owned.people.includes(person)) {
      throw new YianRangeError(`${reprText(person)} isn't one of the people of run ${reprText(owned.runId)}`);
    }
    const record = checkedRecord(options.record ?? null);
    const id = this.#mine(given["run"] as string);
    const unrecordable = await this.#unrecordable();
    if (unrecordable !== null) return { result: null, recorded: false, reason: unrecordable };
    const key = `press-${randomUUID()}`;
    const from = relaySender(person, owned);
    const said = (acted: Acted | null) => pressMessage(this.#group, tool, id, person, given, record, acted);
    let acted: Acted;
    try {
      acted = await this.#act(tool, id, given, from, key);
    } catch (error) {
      await this.#record(said(null), from.by, key);
      throw error;
    }
    const reason = await this.#record(said(acted), from.by, key);
    return { result: acted.result as never, recorded: reason === null, reason };
  }

  /** Run `tool` on run `id` with `args` as `from`, keying a message by `key`, and say what came of it. */
  async #act(tool: keyof PressArguments, id: string, args: JsonObject, from: ChildSender, key: string): Promise<Acted> {
    let refused: ChildRefusal | null = null;
    const sent: Sent = { from, onRefused: (reason) => void (refused = reason) };
    const text = (name: string) => args[name] as string;
    const done = async (result: ChildStatus | Promise<ChildStatus>, did: string): Promise<Acted> => {
      const status = await result;
      return { result: status, did, failed: refused !== null, refused };
    };
    const driver = this.#driver;
    switch (tool) {
      case "message":
        return done(driver.message(id, text("inbox"), text("text"), { ...sent, key }), "sent");
      case "answer":
        return done(driver.answer(id, text("call"), args["answer"], sent), "answered");
      case "interrupt": {
        const cut = await driver.interrupt(id, text("inbox"), sent);
        return { result: cut, did: pressedInterrupt(cut), failed: !cut.landed, refused };
      }
      case "close_inbox":
        return done(driver.closeInbox(id, text("inbox"), sent), "closed");
      case "stop":
        return done(driver.stop(id, sent), "stopped");
      case "approve": {
        const decision = text("decision") as "approve" | "refuse";
        const reason = (args["reason"] as string | undefined) ?? null;
        const decided = driver.approve(id, text("call"), decision, { ...sent, reason });
        return done(decided, decision === "approve" ? "approved" : "refused");
      }
    }
  }

  /** Why this owner's run can't take a press's action now, as `press` says; `null` when it can. */
  async #unrecordable(): Promise<PressUnrecorded | null> {
    const { runId, inbox } = this.#owned;
    const log = this.#log;
    if (inbox === null || log === null) return "no_inbox";
    let run: RunInspection;
    try {
      if (statSync(log).size === 0) return "not_started";
      run = inspectRun(log);
    } catch (error) {
      if (isSystemError(error) && error.code === "ENOENT") return "not_started";
      throw error;
    }
    if (run.startedAt === null || run.status === "unstarted") return "not_started";
    if (run.runId !== runId) return "wrong_log";
    if (run.endedAt !== null) return "ended";
    this.#inboxRefusal ??= await this.#inboxTakes(log, inbox);
    if (this.#inboxRefusal !== null) return this.#inboxRefusal;
    return run.closedInboxes.includes(inbox) ? "inbox_closed" : null;
  }

  /**
   * Whether `inbox` of the run at `log` takes a `Message`, as its flow declares it: `null` when it
   * does, `unknown_inbox` when the flow has no such input, else `invalid_value`.
   *
   * @throws {RangeError} When the run's flow was built in code and `delivery.flow` isn't given, or
   *   was recorded and is.
   * @throws What `loadRecordedFlow` throws.
   */
  async #inboxTakes(log: string, inbox: string): Promise<"unknown_inbox" | "invalid_value" | null> {
    const { env, codeRoot, flow: given = null } = this.#delivery;
    const recorded = await loadRecordedFlow(log, { env, codeRoot });
    const run = reprText(this.#owned.runId);
    if (recorded === null && given === null) {
      throw new YianRangeError(`run ${run}'s flow was built in code: pass it as the owner's delivery.flow`);
    }
    if (recorded !== null && given !== null) {
      throw new YianRangeError(`run ${run} recorded its flow: pass no delivery.flow`);
    }
    const param = (recorded ?? given)!.input.find((input) => input.name === inbox);
    if (param === undefined) return "unknown_inbox";
    const type = param.contract?.type ?? null;
    return isInboxType(type) && typeEquals(type!.element!, MESSAGE) ? null : "invalid_value";
  }

  /**
   * Send `message`, a press's action, to this owner's inbox by `by` under `key`, trying again on a
   * busy log: why it isn't recorded, or `null` once it is. It never throws.
   */
  async #record(message: unknown, by: Author, key: string): Promise<PressUnrecorded | null> {
    const owner = { runId: this.#owned.runId, log: this.#log!, inbox: this.#owned.inbox!, delivery: this.#delivery };
    return (await sendToOwner(owner, message, { by, key }, this.#core.scheduler)).reason;
  }

  /** @internal The last status seq of run `id` this owner's model has seen: shown or reported. */
  lastSeen(id: string): number {
    return Math.max(this.#shown.get(id) ?? 0, this.#core.told(id));
  }

  /**
   * @internal Run `work`, a call of one of its group's tools aborted by `signal`, holding the owner's
   * reports until it ends, once the ones being sent are. When it returns unaborted, its model was
   * told `shown`, the statuses it showed.
   */
  async holding<T>(work: () => T | Promise<T>, signal: AbortSignal, shown: readonly ChildStatus[] = []): Promise<T> {
    const { sending, release } = this.#core.hold();
    try {
      if (sending !== null) await untilSent(sending, signal);
      const result = await work();
      if (!signal.aborted) for (const status of shown) this.#core.shown(status);
      return result;
    } finally {
      release();
    }
  }

  /**
   * @internal Note that the owner's model is shown `status`; return it with only the replies and
   * memory notes still unread, and none the model was shown before. What a report of the owner's
   * would repeat is marked once the tool returns (`holding`).
   */
  show(status: ChildStatus): ChildStatus {
    const last = this.lastSeen(status.run);
    // The stored status keeps the replies and notes it was published with, which may have been
    // read since: shown, or told by a report.
    const { replies: _, memory_notes: __, ...rest } = status;
    const unread = status.seq <= last ? {} : this.#core.unread(status.run, status.seq);
    this.#shown.set(status.run, Math.max(status.seq, last));
    return { ...rest, ...unread };
  }

  /** @throws {RangeError} When this owner doesn't reach run `id`. */
  #mine(id: string): string {
    if (!this.#core.reaches(id)) throw new YianRangeError(`no run '${id}'`);
    return id;
  }
}

/** What came of a press's tool: the driver's result, what the person did, and whether it failed, by what refusal. */
interface Acted {
  readonly result: ChildStatus | ChildInterrupt;
  readonly did: string;
  readonly failed: boolean;
  /** The code of the refusal, or `null` when nothing refused it. */
  readonly refused: ChildRefusal | null;
}

/**
 * The message recording `person`'s press of `tool` on run `id` with `args`, as `record` and what came
 * of it (`acted`, `null` for a throw) say: one action from the person, as `ChildFlowOwner.press` says.
 */
function pressMessage(
  group: string,
  tool: keyof PressArguments,
  id: string,
  person: string,
  args: JsonObject,
  record: PressRecord,
  acted: Acted | null,
) {
  const recorded = record.arguments === undefined ? args : record.arguments;
  let summary = record.summary ?? null;
  // A result may echo the arguments, so replacing them drops it unless the host gives its own.
  let result = record.arguments === undefined ? (acted?.did ?? null) : null;
  if (record.result !== undefined) result = record.result;
  if (acted === null || acted.failed) {
    const refused = acted?.refused ?? null;
    summary = `${person}'s ${PRESSES[tool]} ${id} ${refused === null ? "failed" : `was refused (${refused})`}`;
    // Only an interrupt's take-back did something.
    result = tool === "interrupt" && record.arguments === undefined ? (acted?.did ?? null) : null;
  }
  if (recorded === null) result = null;
  const action = {
    type: "action",
    tool: `${group}_${tool}`,
    ...(summary === null ? {} : { summary }),
    ...(recorded === null ? {} : { arguments: recorded }),
    ...(result === null ? {} : { result }),
  };
  return { role: "user", author: person, parts: [action] };
}

/**
 * `args` of a press of `tool` as its action records them: the fields the tool takes, each checked.
 *
 * @throws {TypeError} When `tool` isn't one a person may press, or a field is missing or of the
 *   wrong type.
 * @throws {RangeError} When an `approve`'s reason is refused (`checkGateText`).
 */
function pressArguments(tool: unknown, args: unknown): JsonObject {
  if (typeof tool !== "string" || !Object.hasOwn(PRESSES, tool)) {
    throw new YianTypeError(`a person can't press ${reprText(String(tool))}`);
  }
  if (!isRecord(args)) throw new YianTypeError("a press's arguments must be an object");
  const text = (name: string) => {
    const value = args[name];
    if (typeof value !== "string") throw new YianTypeError(`a press's ${name} must be a string`);
    return value;
  };
  const run = text("run");
  switch (tool as keyof PressArguments) {
    case "message":
      return { run, inbox: text("inbox"), text: text("text") };
    case "answer": {
      const answer = args["answer"];
      if (answer === undefined || answer === null) throw new YianTypeError("a press's answer can't be null");
      return { run, call: text("call"), answer: answer as JsonValue };
    }
    case "interrupt":
    case "close_inbox":
      return { run, inbox: text("inbox") };
    case "stop":
      return { run };
    case "approve": {
      const call = text("call");
      const decision = text("decision");
      if (decision !== "approve" && decision !== "refuse") {
        throw new YianTypeError("a press's decision must be approve or refuse");
      }
      if (args["reason"] === undefined || args["reason"] === null) return { run, call, decision };
      return { run, call, decision, reason: checkGateText(text("reason"), "reason") };
    }
  }
}

/**
 * `record` once each field it gives has its type: `summary` and `result` text or `null`, and
 * `arguments` an object or `null`. A field left `undefined` isn't given.
 *
 * @throws {TypeError} When one doesn't.
 */
function checkedRecord(record: PressRecord | null): PressRecord {
  if (record === null) return {};
  if (!isRecord(record)) throw new YianTypeError("a press's record must be an object or null");
  const given = Object.fromEntries(Object.entries(record).filter(([, value]) => value !== undefined));
  for (const name of ["summary", "result"]) {
    const value = given[name];
    if (value !== undefined && value !== null && typeof value !== "string") {
      throw new YianTypeError(`a press's record.${name} must be a string or null`);
    }
  }
  const args = given["arguments"];
  if (args !== undefined && args !== null && !isRecord(args)) {
    throw new YianTypeError("a press's record.arguments must be an object or null");
  }
  return given as PressRecord;
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

/** A message's text, its text parts joined; `null` for a value that isn't one or holds another part. */
function messageText(value: unknown): string | null {
  const parts = isRecord(value) ? value["parts"] : null;
  const texts = Array.isArray(parts) ? parts.map((part) => (isRecord(part) ? part["text"] : null)) : [];
  return texts.length > 0 && texts.every((text) => typeof text === "string") ? texts.join("") : null;
}

/** What a send taken back said, as the model reads it again: a message's text, else the value as a status shows it. */
function sentText(value: unknown): unknown {
  const text = messageText(value);
  return text === null ? shownValue(value) : cutStatusText(text);
}

/** What `done`, a person's interrupt, did, as their press records it: nothing of the run's state or inbox. */
function pressedInterrupt(done: ChildInterrupt): string {
  const took = `took back ${done.taken.length} message${done.taken.length === 1 ? "" : "s"}`;
  if (!done.landed) return `not interrupted (${took})`;
  return done.seq === null ? `nothing to interrupt (${took})` : `interrupted (${took})`;
}

/** What `done`, an interrupt of `inbox`, did, in a line the model reads. */
function interruptText(inbox: string, done: ChildInterrupt): string {
  const count = (n: number, what: string) => `${n} ${what}${n === 1 ? "" : "s"}`;
  const taken = done.taken.length;
  const took = `took back ${count(taken, "message")}${taken > LIST_ITEMS ? `, the first ${LIST_ITEMS} shown` : ""}`;
  if (!done.landed) return `not interrupted (${took})`;
  if (done.seq === null) {
    return `nothing to interrupt: the run waits for a message on inbox '${inbox}', or is already stopping (${took})`;
  }
  if (done.nodes.length === 0) {
    const queued = `, and may use up ${count(done.queued, "queued message")} from others unanswered`;
    return `nothing to stop now; the next turn on '${inbox}' stops at once${done.queued === 0 ? "" : queued} (${took})`;
  }
  const held = done.calls.filter((call) => call.held).length;
  const closed = [
    ...(done.calls.length > held ? [count(done.calls.length - held, "question")] : []),
    ...(held > 0 ? [count(held, "held call")] : []),
    ...(done.released.length > 0 ? [`${count(done.released.length, "approved call")} not yet run`] : []),
  ];
  const list = closed.length < 2 ? closed.join("") : `${closed.slice(0, -1).join(", ")} and ${closed.at(-1)}`;
  return `interrupted (${took}${list === "" ? "" : `; closed ${list}`})`;
}

/**
 * The tools a run's model drives the child flows `owner` reaches with, for the host to pass under
 * `owner.group` in that run's tools, so the model sees `<group>_status`, and so on: `status` and
 * `wait` read a run's status, and `message`, `answer`, `interrupt`, `close_inbox`, `stop` and
 * `approve` deliver to it. `interrupt` first takes back the model's own unread sends to the inbox,
 * and adds to the status what it did (`interrupted`) and the first `LIST_ITEMS` texts it took back
 * (`taken_back`).
 *
 * `approve` decides a run's held call (`ChildFlowOwner.approve`) only under the `beforeToolCall` of
 * the owner's run. That hook sees the held call itself as the call's `target`, read from the run's
 * log, beside the run's id and label and, apart from the call, the model's decision and reason:
 * `{run, label, call: {callId, node, tool, source, arguments, target}, decision, reason}`, where a
 * held approve's own `target` shows the call it would release. A call naming no held call is refused
 * before the hook is asked, and its target throws, which refuses it too, should the call stop being
 * held meanwhile. Without the hook, and from a TOOL node, it never runs (`refuseUngated`). When it
 * runs, it decides the call its target named, if the log still holds it. A crash-cut approve isn't
 * sent again: the run's status says whether it landed.
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
 * memory notes aren't shown again. For an owner with an inbox, its children's reports, and those
 * of the runs they start through owners with none, wait while any of these tools runs, and a report
 * and a status they show are read for each other: no report repeats what a status showed its model,
 * and no status repeats a reply a report told, nor `wait` returns for a status one did. A run the
 * owner doesn't reach, or an argument of the wrong type, is a `ReportedToolError`. The deliveries but
 * `interrupt` are `resumable`: a resumed agent sends a crash-cut call again under its call id, which
 * keys `message`'s send, and a repeated answer, close or stop changes nothing; a call sent again
 * (`ctx.attempt` above 1) is `resent` (`ChildFlows.answer`). A take-back or an interrupt has no key,
 * so one sent again couldn't tell it had landed: it would say nothing was taken back, and could cut
 * a turn the run began since. A crash-cut `interrupt` reads that its outcome is unknown.
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
    "that couldn't be done), log, and at (its log's position when it last paused).";
  // By call, the statuses its tool showed, which its model is told of once it returns.
  const shownBy = new WeakMap<ToolContext, ChildStatus[]>();
  const show = (given: ChildStatus, ctx: ToolContext) => {
    shownBy.get(ctx)?.push(given);
    return shownStatus(owner.show(given));
  };
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
      execute: (args, ctx) => show(owner.status(known(args)), ctx),
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
        return show(await owner.wait(id, { after, seconds: Math.min(seconds, WAIT_MAX_S), signal: ctx.signal }), ctx);
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
        return show(await owner.message(id, stringArg(args, "inbox"), stringArg(args, "text"), sent), ctx);
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
        return show(await owner.answer(id, call, answer, { resent: ctx.attempt > 1 }), ctx);
      },
    }),
    tool({
      name: "interrupt",
      description:
        "Stop what a run you started is doing on one of its Inbox inputs: first take back your messages " +
        "there that the run hasn't read, then cut the current step of its agents reading that inbox, " +
        "closing their questions and held calls. The run goes on, and waits for its next message; to " +
        "end it, cancel it. Returns its status with interrupted (what it did) and taken_back (the " +
        `messages taken back, which the run never reads; send again what still applies). ${status}`,
      parameters: runArguments(["inbox", "The Inbox input's name."]),
      execute: async (args, ctx) => {
        const id = known(args);
        const inbox = stringArg(args, "inbox");
        const done = await owner.interrupt(id, inbox);
        const taken = done.taken.slice(0, LIST_ITEMS).map((send) => sentText(send.value));
        return { ...show(done.status, ctx), interrupted: interruptText(inbox, done), taken_back: taken };
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
        return show(await owner.closeInbox(id, stringArg(args, "inbox"), { resent: ctx.attempt > 1 }), ctx);
      },
    }),
    tool({
      name: "stop",
      description: `Cancel a run you started; use it only when asked. ${status}`,
      parameters: runArguments(),
      resumable: true,
      execute: async (args, ctx) => show(await owner.stop(known(args), { resent: ctx.attempt > 1 }), ctx),
    }),
    approveTool(),
  ].map((each) =>
    withExecute(each, (args, ctx) => {
      const shown: ChildStatus[] = [];
      shownBy.set(ctx, shown);
      return owner.holding(() => each.execute(args, ctx), ctx.signal, shown);
    }),
  );

  /**
   * The held call an `approve` call names, as its run's log lists it now, and the model's decision
   * and reason: what the tool's check, its target and its run each read.
   *
   * @throws {ReportedToolError} When the run isn't the owner's, an argument is wrong, or the call
   *   isn't held; its text holds no held call's arguments.
   */
  function approval(args: Record<string, unknown>) {
    const id = known(args);
    const call = stringArg(args, "call");
    // The schema allows no other.
    const decision = args["decision"] as "approve" | "refuse";
    const reason = args["reason"] === undefined ? "" : stringArg(args, "reason");
    if (!reason.isWellFormed() || [...reason].length > REASON_MAX_CHARS || REASON_CONTROL.test(reason)) {
      throw new ReportedToolError(
        `reason must be text of at most ${REASON_MAX_CHARS} characters, with no control characters but line breaks and tabs`,
      );
    }
    const held = owner.heldCall(id, call);
    if (held === null) throw new ReportedToolError(`call '${call}' isn't a call run ${id} holds for approval`);
    return { id, call, decision, reason: reason === "" ? null : reason, held } as const;
  }

  /** The group's `approve`, which only a host's check lets run. */
  function approveTool(): FunctionTool {
    // Refuses a call that names no held call before the host's check is asked, so the model reads why.
    const checked: StandardSchemaV1<unknown, Record<string, unknown>> = {
      "~standard": {
        version: 1,
        vendor: "yian",
        validate: (value) => {
          approval(value as Record<string, unknown>);
          return { value: value as Record<string, unknown> };
        },
      },
    };
    return tool({
      name: "approve",
      description:
        "Approve or refuse a tool call a run you started holds for approval (its status's held), by its " +
        "call id. approve runs the call once, as the run's agent asked it; refuse answers it with an " +
        "error, with your reason, which that agent reads. The host checks each decision first: it may " +
        `ask the user, or refuse it. ${status}`,
      parameters: checked,
      parametersJsonSchema: {
        type: "object",
        properties: {
          run: { type: "string", description: run },
          call: { type: "string", description: "The held call's id, from the status's held." },
          decision: { type: "string", enum: ["approve", "refuse"], description: "approve runs the call; refuse doesn't." },
          reason: { type: "string", maxLength: REASON_MAX_CHARS, description: "Why, in a sentence or two." },
        },
        required: ["run", "call", "decision"],
        additionalProperties: false,
      },
      gateTarget: (args) => {
        const { decision, reason, held } = approval(args);
        // A held approve's own target, so a nested approval shows the call it would release.
        const { callId, node, tool: name, source, arguments: given, target } = held.call;
        const call = { callId, node, tool: name, source: source as JsonObject | null, arguments: given, target };
        return { run: held.runId, label: held.label, call, decision, reason };
      },
      refuseUngated: APPROVE_UNGATED,
      execute: async (args, ctx) => {
        const { id, call, decision, reason } = approval(args);
        return show(await owner.approve(id, call, decision, { reason }), ctx);
      },
    });
  }

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
        return show(owner.status(id), ctx);
      },
    });
  }
}
