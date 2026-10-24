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
 * parallel nodes stay apart. A call's buffer goes when its step record is appended, when it ends
 * with an error, or when its node settles. A late viewer subscribes, reads the log to its end,
 * takes `snapshot()`, then follows the frames, skipping records at or below the snapshot's
 * `last_seq` and chunks below a call's `next_index`.
 *
 * The stream is ephemeral: a subscriber's failure never reaches the run. A subscriber that throws is
 * dropped and reported to the host.
 *
 * Knows about:   platform (warnings), runtime/log.
 * Never imports: nodes, the run boundary, or the CLI.
 */

import { warn } from "../platform.ts";
import { encodeRecord, NodeStepRecord, type RecordEnvelope, type RunLog } from "./log.ts";

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

/** How a `LiveChannel` reports a subscriber that threw, which it then dropped. */
export interface LiveChannelOptions {
  /**
   * Gets the error and the frame being sent; a yian warning (`warn`) when omitted. Anything it
   * throws is ignored.
   */
  onSubscriberError?: ((error: unknown, frame: LiveFrame) => void) | null;
}

/**
 * Forwards a run's records and model call frames to subscribers, and keeps the calls in flight for
 * `snapshot()`.
 *
 * A host creates one, subscribes, and passes it as `runFlow`'s or `resumeFlow`'s `live`; the
 * producer methods (`wrap`, `start`, `chunk`, `end`, `settle`) are for the engine and the run
 * boundary. Each updates the calls, then calls each subscriber synchronously, in subscription order.
 * A subscriber that throws is dropped and reported through `onSubscriberError`; the frame still
 * reaches the others, and nothing is thrown to the run.
 */
export class LiveChannel {
  readonly #subscribers = new Set<(frame: LiveFrame) => void>();
  /** The calls in flight by node, then by `n`, in start order. */
  readonly #calls = new Map<string, Map<number, LiveCall>>();
  readonly #onSubscriberError: (error: unknown, frame: LiveFrame) => void;
  /** The seq after the last record seen. */
  #nextSeq = 0;

  /** @throws TypeError when `onSubscriberError` isn't a function. */
  constructor(options: LiveChannelOptions = {}) {
    const { onSubscriberError = null } = options;
    if (onSubscriberError !== null && typeof onSubscriberError !== "function") {
      throw new TypeError("onSubscriberError must be a function or null");
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

  /** Return the last record's seq and a copy of each call in flight. */
  snapshot(): LiveSnapshot {
    const calls: LiveCall[] = [];
    for (const byN of this.#calls.values()) {
      for (const call of byN.values()) calls.push({ ...call, chunks: [...call.chunks] });
    }
    return { last_seq: this.#nextSeq - 1, calls };
  }

  /** Engine only: send the start of the model call of `node`'s step `n`. */
  start(node: string, n: number, model: string | null, purpose: string, startedAt: string): void {
    let byN = this.#calls.get(node);
    if (byN === undefined) this.#calls.set(node, (byN = new Map()));
    byN.set(n, { node, n, chunks: [], next_index: 0 });
    this.#send({ type: "start", node, n, model, purpose, started_at: startedAt });
  }

  /** Engine only: send chunk `index` of the call of `node`'s step `n`. */
  chunk(node: string, n: number, index: number, text: string): void {
    const call = this.#calls.get(node)?.get(n);
    if (call !== undefined) {
      call.chunks.push(text);
      call.next_index = index + 1;
    }
    this.#send({ type: "chunk", node, n, index, text });
  }

  /** Engine only: send the end of the call of `node`'s step `n`; one that ended with `error` is dropped. */
  end(node: string, n: number, error: string | null): void {
    if (error !== null) this.#drop(node, n);
    this.#send({ type: "end", node, n, error });
  }

  /** Engine only: drop the calls of `node`, which settled; no frame is sent. */
  settle(node: string): void {
    this.#calls.delete(node);
  }

  /**
   * Run boundary only: return a log that appends to `log` and sends each appended record to this
   * channel once the append returns; a `node_step` drops its call. The other members read `log`;
   * closing it closes `log`. The records `log` already holds (a resumed run's) count as seen: the
   * snapshot's `last_seq` starts at the last.
   */
  wrap(log: RunLog): RunLog {
    this.#nextSeq = Math.max(this.#nextSeq, log.nextSequence);
    const recorded = (envelope: RecordEnvelope) => {
      const { seq, record } = envelope;
      this.#nextSeq = seq + 1;
      if (record instanceof NodeStepRecord) this.#drop(record.nodeId, record.n);
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

  #drop(node: string, n: number): void {
    const byN = this.#calls.get(node);
    if (byN === undefined) return;
    byN.delete(n);
    if (byN.size === 0) this.#calls.delete(node);
  }

  /** Call each subscriber with `frame`; drop and report one that throws. */
  #send(frame: LiveFrame): void {
    for (const subscriber of [...this.#subscribers]) {
      try {
        subscriber(frame);
      } catch (error) {
        this.#subscribers.delete(subscriber);
        try {
          this.#onSubscriberError(error, frame);
        } catch {
          // The stream is ephemeral: nothing a viewer's side does may reach the run.
        }
      }
    }
  }
}

/** Report a dropped subscriber's error as a yian warning. */
function reportAsWarning(error: unknown): void {
  const text = error instanceof Error ? `${error.name}: ${error.message}` : String(error);
  warn(`live channel subscriber failed and was dropped: ${text}`);
}
