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
 * Knows about:   runtime/log.
 * Never imports: nodes, the run boundary, or the CLI.
 */

import { decodeRecord, encodeRecord, NodeStepRecord, type RecordEnvelope, type RunLog, type RunRecord } from "./log.ts";

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

/**
 * Forwards a run's records and model call frames to subscribers, and keeps the calls in flight for
 * `snapshot()`.
 *
 * The producer methods (`record`, `start`, `chunk`, `end`, `settle`) update the calls, then call
 * each subscriber synchronously, in subscription order. A subscriber that throws ends the run as a
 * host fault, as a failing `onStream` does.
 */
export class LiveChannel {
  readonly #subscribers = new Set<(frame: LiveFrame) => void>();
  /** The calls in flight by node, then by `n`, in start order. */
  readonly #calls = new Map<string, Map<number, LiveCall>>();
  #lastSeq = -1;

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
    return { last_seq: this.#lastSeq, calls };
  }

  /**
   * Forward the record `line` the log appended under `seq`; a `node_step` drops its call.
   *
   * @throws RangeError when `line` isn't a record line, read only while a call is in flight.
   */
  record(seq: number, line: string): void {
    // A line is decoded only to find the call it closes.
    this.#recorded(seq, line, this.#calls.size === 0 ? null : decodeRecord(line).record);
  }

  /** Forward the start of the model call of `node`'s step `n`. */
  start(node: string, n: number, model: string | null, purpose: string, startedAt: string): void {
    let byN = this.#calls.get(node);
    if (byN === undefined) this.#calls.set(node, (byN = new Map()));
    byN.set(n, { node, n, chunks: [], next_index: 0 });
    this.#send({ type: "start", node, n, model, purpose, started_at: startedAt });
  }

  /** Forward chunk `index` of the call of `node`'s step `n`. */
  chunk(node: string, n: number, index: number, text: string): void {
    const call = this.#calls.get(node)?.get(n);
    if (call !== undefined) {
      call.chunks.push(text);
      call.next_index = index + 1;
    }
    this.#send({ type: "chunk", node, n, index, text });
  }

  /** Forward the end of the call of `node`'s step `n`; one that ended with `error` is dropped. */
  end(node: string, n: number, error: string | null): void {
    if (error !== null) this.#drop(node, n);
    this.#send({ type: "end", node, n, error });
  }

  /** Drop the calls of `node`, which settled; no frame is sent. */
  settle(node: string): void {
    this.#calls.delete(node);
  }

  /**
   * Return a log that appends to `log` and forwards each appended record to this channel once the
   * append returns. The other members read `log`; closing it closes `log`.
   */
  wrap(log: RunLog): RunLog {
    const forward = (envelope: RecordEnvelope) => this.#recorded(envelope.seq, encodeRecord(envelope), envelope.record);
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
        forward(envelope);
        return envelope;
      },
      close: () => log.close(),
      [Symbol.dispose]: () => log.close(),
    };
  }

  /** Note record `seq`, drop the call a step record closes, then send the record. */
  #recorded(seq: number, line: string, record: RunRecord | null): void {
    this.#lastSeq = seq;
    if (record instanceof NodeStepRecord) this.#drop(record.nodeId, record.n);
    this.#send({ type: "record", seq, line });
  }

  #drop(node: string, n: number): void {
    const byN = this.#calls.get(node);
    if (byN === undefined) return;
    byN.delete(n);
    if (byN.size === 0) this.#calls.delete(node);
  }

  #send(frame: LiveFrame): void {
    for (const subscriber of [...this.#subscribers]) subscriber(frame);
  }
}
