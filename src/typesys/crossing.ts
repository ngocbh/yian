/**
 * How values and throws cross a worker thread's port, for the CODE worker and a tool module's
 * thread. Values cross by structured clone, which drops a `FileRef`'s class, so a `FileRef`
 * crosses as a `Map` of its fields (no yian value holds a `Map`). A value too deep for the copy
 * is refused before it is posted.
 *
 * Knows about: typesys/values and defaults.
 */

import { current } from "../defaults.ts";
import {
  FileRef,
  fromHost,
  isList,
  isRecord,
  jsTypeName,
  rejectDeepNesting,
  setField,
  toHost,
  TypeCheckError,
  type Value,
} from "./values.ts";

/** A thrown `Error`'s fields. */
export interface ThrownError {
  kind: "error";
  name: string;
  message: string;
  stack?: string;
}

/** A thrown value as it crosses: an `Error`'s fields, a yian value (packed), or its text. */
export type Thrown = ThrownError | { kind: "value"; value: unknown } | { kind: "text"; text: string };

/** Return a copy of a host value with each `FileRef` as a `Map` of its fields. */
export function packFiles(value: unknown): unknown {
  return swapLeaves(value, (part) =>
    FileRef.isFileRef(part)
      ? new Map([
          ["uri", part.uri],
          ["mime", part.mime],
          ["name", part.name],
        ])
      : part,
  );
}

/** Return a copy of a packed value with each `Map` back as a `FileRef`. */
export function unpackFiles(value: unknown): unknown {
  return swapLeaves(value, (part) =>
    part instanceof Map ? new FileRef({ uri: part.get("uri"), mime: part.get("mime"), name: part.get("name") }) : part,
  );
}

/** An open array or record in `swapLeaves`, with the fresh copy being filled. */
interface SwapFrame {
  readonly source: Record<string | number, unknown>;
  /** The record's keys, or `null` for an array. */
  readonly keys: readonly string[] | null;
  readonly copy: unknown[] | Record<string, unknown>;
  /** The number of items or keys to copy. */
  readonly size: number;
  index: number;
}

/**
 * Return a copy of the arrays and records of `value` with each other object part replaced by
 * `swap(part)`. A part shared by two parents is copied once, and the walk uses an explicit
 * stack, as `toHost` does, so any depth works.
 */
function swapLeaves(value: unknown, swap: (part: object) => unknown): unknown {
  const stack: SwapFrame[] = [];
  const copies = new Map<object, unknown[] | Record<string, unknown>>();
  const enter = (item: unknown): unknown => {
    if (typeof item !== "object" || item === null) return item;
    const list = isList(item);
    if (!list && !isRecord(item)) return swap(item);
    const shared = copies.get(item);
    if (shared !== undefined) return shared;
    const copy = list ? [] : {};
    copies.set(item, copy);
    const keys = list ? null : Object.keys(item);
    const size = keys === null ? (item as unknown[]).length : keys.length;
    stack.push({ source: item as Record<string, unknown>, keys, copy, size, index: 0 });
    return copy;
  };
  const root = enter(value);
  while (stack.length > 0) {
    const frame = stack[stack.length - 1]!;
    if (frame.index === frame.size) {
      stack.pop();
      continue;
    }
    const index = frame.index++;
    if (frame.keys === null) {
      (frame.copy as unknown[]).push(enter(frame.source[index]));
    } else {
      const key = frame.keys[index]!;
      setField(frame.copy as Record<string, Value>, key, enter(frame.source[key]) as Value);
    }
  }
  return root;
}

/**
 * Return how a thrown value crosses to the other thread: an `Error`'s fields, a yian value, else
 * its text. A yian value nested too deep to copy back crosses as its type name.
 */
export function thrownOf(value: unknown): Thrown {
  if (value instanceof Error) {
    const fields: ThrownError = { kind: "error", name: String(value.name), message: String(value.message) };
    if (typeof value.stack === "string") fields.stack = value.stack;
    return fields;
  }
  try {
    const thrown = fromHost(value, "error");
    // Too deep for the copy back, which would overflow in `postMessage`: name its type instead.
    if (tooDeepToCopy(thrown)) return { kind: "text", text: jsTypeName(value) };
    return { kind: "value", value: packFiles(toHost(thrown)) };
  } catch {
    // Not a yian value: its text, else its type name when it has none, as a stage marker reads.
    try {
      return { kind: "text", text: String(value) };
    } catch {
      return { kind: "text", text: jsTypeName(value) };
    }
  }
}

// Copying a value back recurses on both threads: measured, Node's main thread reads about
// 2,100 levels and Bun about 40,000. A deeper result is refused before it is posted; a shallower
// one over the limit is copied and still meets the commit's own nesting check.
// Not a YIAN_* setting: YIAN_MAX_VALUE_NESTING is at most 100, so this bound never refuses an
// accepted value; it only picks which check reports a too-deep result. Raised past what the
// copy carries (about 2,100 levels on Node), it would let the copy crash instead.
const COPY_DEPTH = 1000;

/** Whether `value` nests deeper than the copy back to the main thread can carry. */
export function tooDeepToCopy(value: Value): boolean {
  try {
    rejectDeepNesting(value, "output", { enclosingLevels: current().maxValueNesting - COPY_DEPTH });
    return false;
  } catch (error) {
    if (error instanceof TypeCheckError) return true;
    throw error;
  }
}
