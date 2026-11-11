/**
 * The live channel of one run: what a viewer needs beside the run log to follow it as it runs.
 *
 * Frames are plain JSON (snake_case keys, as in the log), so a host can send them anywhere with
 * `logJson`. A `record` frame carries a line exactly as the log appended it, after the append
 * returned; `start`, `chunk` and `end` frames follow one AGENT model call, named by the step its
 * record becomes (`node` + `n`), its chunks numbered by `index` from 0. Records never go through
 * `onEvent`, and frames never enter the log.
 *
 * The channel keeps the chunks of each call in flight, one buffer per `(node, n)`, so the calls of
 * parallel nodes stay apart. A call's buffer goes when its step record is appended (a failed
 * attempt's too, before its end frame), when its node settles, or when a record ends the run. A
 * late viewer calls `follow`, which sends the snapshot, then every later frame; it reads the log up
 * to the snapshot's `last_seq` (this may be async), then applies the frames that came meanwhile and
 * follows. Every frame after the snapshot is new to it: records continue after `last_seq` and each
 * call's chunks after its `next_index`, so it skips nothing. Yian ships no transport: a host sends
 * the frames over its own channel, and the reading side folds them with a `LiveView`.
 *
 * The stream is ephemeral: a subscriber's failure never reaches the run. A subscriber that throws is
 * dropped and reported to the host.
 *
 * Knows about: platform (warnings), runtime/log.
 */

import { warn } from "../platform.ts";
import { YianRangeError, YianTypeError } from "../typesys/errors.ts";
import { decodeRecordHeader, encodeRecord, NodeStepRecord, type RecordEnvelope, type RunLog } from "./log.ts";

/** One live frame: a record the log appended, or the start, a chunk or the end of a model call. */
export type LiveFrame =
  | { type: "record"; seq: number; line: string }
  | { type: "start"; node: string; n: number; model: string | null; purpose: string; started_at: string }
  | { type: "chunk"; node: string; n: number; index: number; text: string }
  | { type: "end"; node: string; n: number; error: string | null };

/** A model call in flight: its step, its chunks so far and the `index` of its next chunk. */
export interface LiveCall {
  node: string;
  n: number;
  chunks: string[];
  next_index: number;
}

/** The calls in flight after the record `last_seq` (`-1` before any record). Plain JSON. */
export interface LiveSnapshot {
  last_seq: number;
  calls: LiveCall[];
}

/** What `LiveChannel.follow` sends: a snapshot first, then live frames. Plain JSON. */
export type LiveFollowFrame = LiveFrame | { type: "snapshot"; snapshot: LiveSnapshot };

/** How a `LiveChannel` reports a subscriber that threw, which it then dropped. */
export interface LiveChannelOptions {
  /**
   * Gets the error and the frame being sent; a yian warning (`warn`) when omitted. Anything it
   * throws is ignored.
   */
  onSubscriberError?: ((error: unknown, frame: LiveFollowFrame) => void) | null;
}

/**
 * Forwards a run's records and model call frames to subscribers, and keeps the calls in flight for
 * `snapshot()`.
 *
 * A host creates one, subscribes, and passes it as `runFlow`'s or `advance`'s `live`; the
 * producer methods (`wrap`, `start`, `chunk`, `end`, `settle`) are for the engine and the run
 * boundary. Each updates the calls, then calls each subscriber synchronously, in subscription order.
 * A subscriber that throws is dropped and reported through `onSubscriberError`; the frame still
 * reaches the others, and nothing is thrown to the run.
 */
export class LiveChannel {
  readonly #subscribers = new Set<(frame: LiveFrame) => void>();
  /** The chunks of each call in flight by node, then by `n`, in start order. */
  readonly #calls = new Map<string, Map<number, string[]>>();
  readonly #onSubscriberError: (error: unknown, frame: LiveFollowFrame) => void;
  /** The seq after the last record seen. */
  #nextSeq = 0;

  /** @throws TypeError when `onSubscriberError` isn't a function. */
  constructor(options: LiveChannelOptions = {}) {
    const { onSubscriberError = null } = options;
    if (onSubscriberError !== null && typeof onSubscriberError !== "function") {
      throw new YianTypeError("onSubscriberError must be a function or null");
    }
    this.#onSubscriberError = onSubscriberError ?? reportAsWarning;
  }

  /** Call `fn` with every later frame; returns the function that stops it. */
  subscribe(fn: (frame: LiveFrame) => void): () => void {
    // A wrapper, so subscribing one function twice gives two independent subscriptions.
    const subscriber = (frame: LiveFrame) => fn(frame);
    this.#subscribers.add(subscriber);
    return () => void this.#subscribers.delete(subscriber);
  }

  /**
   * Call `fn` with a snapshot frame, then with every later frame; returns the function that stops
   * it. The snapshot and the subscription are taken in one synchronous turn, so every frame after
   * the snapshot is new to `fn`: records continue after its `last_seq` and each call's chunks after
   * its `next_index`. A viewer reads the log up to `last_seq`, then applies the frames that came
   * meanwhile (`LiveView`). A `fn` that throws, on the snapshot too, is dropped and reported.
   */
  follow(fn: (frame: LiveFollowFrame) => void): () => void {
    const frame: LiveFollowFrame = { type: "snapshot", snapshot: this.snapshot() };
    const stop = this.subscribe(fn);
    try {
      fn(frame);
    } catch (error) {
      stop();
      this.#report(error, frame);
    }
    return stop;
  }

  /**
   * Return the last record's seq and a copy of each call in flight. `follow` takes it in the same
   * synchronous turn as its subscription, so the frames after it are exactly the ones it sends.
   */
  snapshot(): LiveSnapshot {
    const calls: LiveCall[] = [];
    for (const [node, byN] of this.#calls) {
      // Chunk indexes are dense from 0, so the next one is the count so far.
      for (const [n, chunks] of byN) calls.push({ node, n, chunks: [...chunks], next_index: chunks.length });
    }
    return { last_seq: this.#nextSeq - 1, calls };
  }

  /** @internal Engine only: send the start of the model call of `node`'s step `n`. */
  start(node: string, n: number, model: string | null, purpose: string, startedAt: string): void {
    setCall(this.#calls, node, n, []);
    this.#send({ type: "start", node, n, model, purpose, started_at: startedAt });
  }

  /** @internal Engine only: send chunk `index` (the call's next) of the call of `node`'s step `n`. */
  chunk(node: string, n: number, index: number, text: string): void {
    this.#calls.get(node)?.get(n)?.push(text);
    this.#send({ type: "chunk", node, n, index, text });
  }

  /**
   * @internal Engine only: send the end of the call of `node`'s step `n`. Its step record came
   * first (a reply's, a cut reply's or a failed attempt's), and already dropped the call.
   */
  end(node: string, n: number, error: string | null): void {
    this.#send({ type: "end", node, n, error });
  }

  /** @internal Engine only: drop the calls of `node`, which settled; no frame is sent. */
  settle(node: string): void {
    this.#calls.delete(node);
  }

  /**
   * @internal Run boundary only: return a log that appends to `log` and sends each appended record
   * to this channel once the append returns; a `node_step` drops its call, and a `node_failed`,
   * `run_stopped` or `run_cancelled`, which end the run, drop every call. The other members read
   * `log`; closing it closes `log`. The records `log` already holds (a resumed run's) count as seen:
   * the snapshot's `last_seq` starts at the last.
   */
  wrap(log: RunLog): RunLog {
    this.#nextSeq = log.nextSequence;
    const recorded = (envelope: RecordEnvelope) => {
      const { seq, record } = envelope;
      this.#nextSeq = seq + 1;
      if (record instanceof NodeStepRecord) deleteCall(this.#calls, record.nodeId, record.n);
      // A node failure, a stop or a cancel ends the run: its calls get no record, and their nodes
      // settle only later, so a snapshot taken in between would list calls that nothing ever drops.
      else if (RUN_ENDING_KINDS.has(envelope.kind)) this.#calls.clear();
      this.#send({ type: "record", seq, line: encodeRecord(envelope) });
    };
    return {
      get runId() {
        return log.runId;
      },
      get path() {
        return log.path;
      },
      get nextSequence() {
        return log.nextSequence;
      },
      [Symbol.iterator]: () => log[Symbol.iterator](),
      records: () => log.records(),
      append(record) {
        const envelope = log.append(record);
        recorded(envelope);
        return envelope;
      },
      close: () => log.close(),
      [Symbol.dispose]: () => log.close(),
    };
  }

  /** Call each subscriber with `frame`; drop and report one that throws. */
  #send(frame: LiveFrame): void {
    for (const subscriber of [...this.#subscribers]) {
      try {
        subscriber(frame);
      } catch (error) {
        this.#subscribers.delete(subscriber);
        this.#report(error, frame);
      }
    }
  }

  /** Report a dropped subscriber's error; nothing the report throws gets out. */
  #report(error: unknown, frame: LiveFollowFrame): void {
    try {
      this.#onSubscriberError(error, frame);
    } catch {
      // The stream is ephemeral: nothing a viewer's side does may reach the run.
    }
  }
}

/** Report a dropped subscriber's error as a yian warning. */
function reportAsWarning(error: unknown): void {
  const text = error instanceof Error ? `${error.name}: ${error.message}` : String(error);
  warn(`live channel subscriber failed and was dropped: ${text}`);
}

/** A call in flight as a `LiveView` holds it: its step, its text so far and its next chunk's `index`. */
export interface LiveViewCall {
  readonly node: string;
  readonly n: number;
  readonly text: string;
  readonly nextIndex: number;
}

/**
 * A reader's fold of a live run: the seq of the last record it has, and each model call in flight
 * with its text so far, one buffer per `(node, n)`.
 *
 * A host that serves viewers in another process keeps one per run, built from the log up to a
 * `follow` snapshot's `last_seq` plus that snapshot, then fed every later frame. It never reads
 * files: the host gives it records, and frames in the order they were sent (a duplicate start
 * delivered after its call's step record would bring the call back). A record at or below
 * `lastSeq`, the start of a call in flight, or a chunk below its call's next index is a duplicate
 * and changes nothing; a record past the next seq, a chunk past its call's next index, or a chunk of
 * a call not in flight is a gap: `apply` returns false and changes nothing, and the host asks
 * `follow` again. A call goes when its step record arrives or when it ends with an error; a node
 * failure, a run stop or a cancel ends the run and drops every call. A run that ends without one (a
 * host fault, or its process gone) sends nothing more, so the host drops the view when the run ends.
 */
export class LiveView {
  /** The text and next chunk index of each call in flight by node, then by `n`, in start order. */
  readonly #calls = new Map<string, Map<number, { text: string; nextIndex: number }>>();
  #lastSeq: number;

  private constructor(lastSeq: number) {
    this.#lastSeq = lastSeq;
  }

  /**
   * Start from the records read from the log, at least up to `snapshot.last_seq`, and the snapshot
   * a `follow` sent; records past `last_seq` are ignored, as the frames after the snapshot bring
   * them again. Only each record's `seq` is read, so `iterRunRecords` entries, with or without
   * payloads, and log envelopes both work.
   *
   * @throws RangeError when the records up to `last_seq` aren't seqs 0, 1, ... `last_seq`.
   */
  static from(records: Iterable<{ readonly seq: number }>, snapshot: LiveSnapshot): LiveView {
    const { last_seq: lastSeq, calls } = snapshot;
    let next = 0;
    for (const { seq } of records) {
      if (seq > lastSeq) break;
      if (seq !== next) throw new YianRangeError(`records must run from seq 0 to ${lastSeq}; got seq ${seq} for ${next}`);
      next++;
    }
    if (next !== lastSeq + 1) {
      throw new YianRangeError(`records must run from seq 0 to ${lastSeq}; they stop before seq ${next}`);
    }
    const view = new LiveView(lastSeq);
    for (const { node, n, chunks, next_index } of calls) {
      setCall(view.#calls, node, n, { text: chunks.join(""), nextIndex: next_index });
    }
    return view;
  }

  /** The seq of the last record applied (`-1` before any record). */
  get lastSeq(): number {
    return this.#lastSeq;
  }

  /** The calls in flight, in start order: node, n, text so far and next chunk index. */
  calls(): readonly LiveViewCall[] {
    const calls: LiveViewCall[] = [];
    for (const [node, byN] of this.#calls) {
      for (const [n, { text, nextIndex }] of byN) calls.push({ node, n, text, nextIndex });
    }
    return calls;
  }

  /**
   * Apply one frame; returns false on a gap, meaning "follow again", and true otherwise, for a
   * duplicate and a frame of a type it doesn't know too.
   *
   * @throws what `decodeRecordHeader` throws when a new record frame's `line` isn't a valid
   *   envelope; only the envelope and a step's node and number are read, not the payload.
   */
  apply(frame: LiveFrame): boolean {
    switch (frame.type) {
      case "record": {
        if (frame.seq <= this.#lastSeq) return true;
        if (frame.seq !== this.#lastSeq + 1) return false;
        const { kind, step } = decodeRecordHeader(frame.line);
        this.#lastSeq = frame.seq;
        if (step !== null) deleteCall(this.#calls, step.node, step.n);
        else if (RUN_ENDING_KINDS.has(kind)) this.#calls.clear();
        return true;
      }
      case "start":
        if (this.#calls.get(frame.node)?.has(frame.n) !== true) {
          setCall(this.#calls, frame.node, frame.n, { text: "", nextIndex: 0 });
        }
        return true;
      case "chunk": {
        const call = this.#calls.get(frame.node)?.get(frame.n);
        if (call === undefined || frame.index > call.nextIndex) return false;
        if (frame.index === call.nextIndex) {
          call.text += frame.text;
          call.nextIndex++;
        }
        return true;
      }
      case "end":
        // A call's step record comes before its end; an error end drops the call even without one.
        if (frame.error !== null) deleteCall(this.#calls, frame.node, frame.n);
        return true;
      default:
        return true;
    }
  }
}

/** Set the call of `node`'s step `n` in `calls`, a map by node, then by `n`. */
function setCall<V>(calls: Map<string, Map<number, V>>, node: string, n: number, value: V): void {
  let byN = calls.get(node);
  if (byN === undefined) calls.set(node, (byN = new Map()));
  byN.set(n, value);
}

/**
 * The kinds of the records that end the run, so no call in flight gets a step record: a node
 * failure, a stop or a cancel.
 */
const RUN_ENDING_KINDS: ReadonlySet<string> = new Set(["node_failed", "run_stopped", "run_cancelled"]);

/** Delete the call of `node`'s step `n` from `calls`, and its node's map once empty. */
function deleteCall<V>(calls: Map<string, Map<number, V>>, node: string, n: number): void {
  const byN = calls.get(node);
  if (byN === undefined) return;
  byN.delete(n);
  if (byN.size === 0) calls.delete(node);
}
