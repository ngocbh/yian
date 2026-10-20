/**
 * Opaque context summaries never replace backend-owned raw conversation truth: a port of
 * `tests/engine/test_session_summaries.py`.
 *
 * Ported: `test_settings_require_exact_integer_bounds`,
 * `test_defaults_and_below_budget_do_not_summarize`, `test_whole_prior_turns_and_current_are_separate`,
 * `test_oversized_summary_reuses_same_prefix_and_extends_only_new_turns`,
 * `test_no_eligible_older_turn_does_not_call_summary`,
 * `test_invalid_summary_retains_raw_history_and_previous_cache`,
 * `test_summary_exception_identity_and_no_candidate_publication`,
 * `test_defaults_keep_four_prior_turns_even_over_budget`,
 * `test_cache_is_private_versioned_and_does_not_rewind_to_old_pin`,
 * `test_cache_is_shared_by_consumers_and_isolated_by_namespace`,
 * `test_cached_summary_under_budget_does_not_extend_eligible_prefix`,
 * `test_failed_extension_keeps_previous_summary`,
 * `test_empty_crashed_and_failed_turns_count_in_recent_selection`,
 * `test_callback_history_change_rejects_candidate_without_rewinding`,
 * `test_close_during_projection_prevents_starting_summary_callback`,
 * `test_snapshot_pins_and_learned_notes_survive_summary_and_note_commit`,
 * `test_invalid_projection_does_not_call_summary_or_poison_cache` and
 * `test_callback_drains_before_fifo_lease_release`.
 *
 * TS-only: `prepareHistory`'s up-front checks (an empty `projectionVersion` is a `RangeError`, a
 * non-function callback a `TypeError`); async `project` and `summarize` are awaited; an async
 * `summarize` that rejects leaves the cache and raw history unchanged; the budget counts code
 * points, so two astral characters fit a budget of 2.
 *
 * Changed:
 * - `test_settings_require_exact_integer_bounds`: `1.0` is the int 1 in JS, so the `recent_turns`
 *   `1.0` row is accepted; every other row pins its error class and message.
 * - `test_callback_drains_before_fifo_lease_release`: threads are concurrent promises; the callback
 *   and the events are `Promise.withResolvers()`; "is waiting" is a macrotask later with the private
 *   `queue` length; the owner's `_HostStop` and the queued waiter's checker are `AbortController`s
 *   aborted while the summary callback is pending (no stop check blocks).
 * - The projection is `JSON.stringify` of the run ids, so expected projections have no space after
 *   commas (Python's `json.dumps` writes `", "`); `with` blocks are `withHarness`/`withTurn`;
 *   private members are read by element access (`memory["summaries"]`); the notes row builds its
 *   notes as session_notes.test.ts does, with `bigint` JSON ints.
 *
 * Not ported: none.
 */

import crypto from "node:crypto";

import { beforeEach, describe, expect, test } from "bun:test";

import { BUILTIN, install } from "../../src/defaults.ts";
import {
  type ConversationInvocation,
  type ConversationView,
  MemoryMutation,
  MemoryOperationError,
} from "../../src/memory/contracts.ts";
import { SessionMemory, type Turn } from "../../src/memory/session.ts";
import { logJson } from "../../src/typesys/json.ts";
import {
  appendReply,
  caught,
  failure,
  macrotask,
  type Mutable,
  NEVER,
  rejected,
  request,
  settle,
  snapshot,
  withHarness,
  withTurn,
} from "./session_fakes.ts";

const STALE_SESSION = "memory session is closed or belongs to an old generation";
const RESET_BUSY = "cannot reset active or queued conversations";

// Python's tests assume the built-in defaults (32 000 characters, 4 recent turns).
beforeEach(() => install(BUILTIN));

/** Record `count` turns of `consumer`, each with one 20-character reply. */
async function seed(memory: SessionMemory, count: number, consumer: string[] = ["root", "agent"]): Promise<void> {
  await withHarness(async (opened) => {
    for (let index = 0; index < count; index++) {
      await withTurn(opened.frame, request(String(index), 1, consumer), (turn) =>
        appendReply(turn, `item-${consumer.at(-1)}-${index}`, "x".repeat(20)),
      );
    }
  }, memory);
}

/** Project the selected invocations as the JSON array of their run ids. */
function project(_view: ConversationView, invocations: readonly ConversationInvocation[]): string {
  return JSON.stringify(invocations.map((invocation) => invocation.runId));
}

type Calls = [text: string, target: number][];

/** Prepare history with a summary callback that records its calls and returns `result`. */
function prepare(
  turn: Turn,
  calls: Calls,
  options: { version?: string; result?: unknown; project?: typeof project } = {},
): Promise<readonly [string | null, string]> {
  const { version = "test-v1", result = "summary", project: projectFn = project } = options;
  return turn.prepareHistory({
    projectionVersion: version,
    project: projectFn,
    summarize: (text, target) => {
      calls.push([text, target]);
      return result as string;
    },
  });
}

/** Run `fn` inside one open turn of a fresh run of `memory`. */
function inTurn<T>(memory: SessionMemory, runId: string, fn: (turn: Turn) => T | Promise<T>): Promise<T> {
  return withHarness((opened) => withTurn(opened.frame, request(runId), fn), memory);
}

/** One closed note with its source identity, as session_notes.test.ts builds it. */
function note(options: { runId?: string; index?: bigint; text?: string } = {}): Mutable {
  const { runId = "a", index = 0n, text = "Check units." } = options;
  const value = {
    consumer_static_path: ["root", "agent"],
    run_id: runId,
    execution_seq: 1n,
    message_index: index,
    text,
  };
  return { note_id: crypto.createHash("sha256").update(logJson(value)).digest("hex"), ...value };
}

/** The append-note mutation for `value`. */
function mutation(value: Mutable): MemoryMutation {
  return new MemoryMutation(value.note_id, { operation: "append_note", note: value });
}

describe("settings require exact integer bounds", () => {
  const rows: [string, unknown, ErrorConstructor | null, string][] = [
    ["historyBudgetChars", 0, RangeError, "history_budget_chars must be at least 1"],
    ["historyBudgetChars", -1, RangeError, "history_budget_chars must be at least 1"],
    ["historyBudgetChars", true, TypeError, "history_budget_chars must be an exact integer"],
    ["historyBudgetChars", 1.5, TypeError, "history_budget_chars must be an exact integer"],
    ["recentTurns", -1, RangeError, "recent_turns must be at least 0"],
    ["recentTurns", false, TypeError, "recent_turns must be an exact integer"],
    // Changed: `1.0` is the int 1 in JS.
    ["recentTurns", 1.0, null, ""],
  ];
  for (const [field, value, cls, message] of rows) {
    test(`${field} ${String(value)}`, () => {
      if (cls === null) {
        expect(new SessionMemory({ [field]: value })["recentTurns"]).toBe(1);
        return;
      }
      const error = caught(() => new SessionMemory({ [field]: value }));
      expect(Object.getPrototypeOf(error)).toBe(cls.prototype);
      expect((error as Error).message).toBe(message);
    });
  }
});

test("defaults and below budget do not summarize", async () => {
  const memory = new SessionMemory();
  const calls: Calls = [];
  await seed(memory, 8);
  await inTurn(memory, "current", async (turn) => {
    expect(await prepare(turn, calls)).toEqual([null, JSON.stringify(["0", "1", "2", "3", "4", "5", "6", "7"])]);
  });
  expect(calls).toEqual([]);
});

describe("whole prior turns and current are separate", () => {
  for (const recent of [0, 4]) {
    test(`recent ${recent}`, async () => {
      const memory = new SessionMemory({ historyBudgetChars: 1, recentTurns: recent });
      const calls: Calls = [];
      await seed(memory, 6);
      const before: Mutable = snapshot(memory);
      await inTurn(memory, "current", async (turn) => {
        const [summary, remaining] = await prepare(turn, calls);
        expect(summary).toBe("summary");
        expect(JSON.parse(remaining)).toEqual(["0", "1", "2", "3", "4", "5"].slice(6 - recent));
        expect(calls.length).toBe(1);
        expect(calls[0]![1]).toBe(1);
        expect(calls[0]![0]).not.toContain("current");
        for (let index = 0; index < 6 - recent; index++) expect(calls[0]![0]).toContain(`"${index}"`);
      });
      const after: Mutable = snapshot(memory);
      expect(after.conversation.items).toEqual(before.conversation.items);
      expect(after.notes).toEqual(before.notes);
    });
  }
});

test("oversized summary reuses same prefix and extends only new turns", async () => {
  const memory = new SessionMemory({ historyBudgetChars: 1, recentTurns: 1 });
  const calls: Calls = [];
  await seed(memory, 3);
  await withHarness(async (opened) => {
    await withTurn(opened.frame, request("current"), async (turn) => {
      const expected = await prepare(turn, calls, { result: "oversized summary" });
      expect(await prepare(turn, calls)).toEqual(expected);
      expect(calls.length).toBe(1);
    });
    await withTurn(opened.frame, request("next"), async (turn) => {
      expect((await prepare(turn, calls, { result: "extended" }))[0]).toBe("extended");
    });
  }, memory);
  expect(calls.length).toBe(2);
  expect(calls[1]![0]).toContain("oversized summary");
  expect(calls[1]![0]).toContain('"2"');
  expect(calls[1]![0]).not.toContain('"0"');
  expect(calls[1]![0]).not.toContain('"1"');
});

test("no eligible older turn does not call summary", async () => {
  const memory = new SessionMemory({ historyBudgetChars: 1 });
  const calls: Calls = [];
  await seed(memory, 4);
  await inTurn(memory, "current", async (turn) => {
    expect((await prepare(turn, calls))[0]).toBeNull();
  });
  expect(calls).toEqual([]);
});

describe("invalid summary retains raw history and previous cache", () => {
  const rows: [unknown, ErrorConstructor, string][] = [
    [null, TypeError, "summary must be a string"],
    ["", RangeError, "summary must contain nonempty text"],
    ["   ", RangeError, "summary must contain nonempty text"],
    [42, TypeError, "summary must be a string"],
    [["text"], TypeError, "summary must be a string"],
  ];
  for (const [bad, cls, message] of rows) {
    test(JSON.stringify(bad), async () => {
      const memory = new SessionMemory({ historyBudgetChars: 1, recentTurns: 0 });
      const calls: Calls = [];
      await seed(memory, 1);
      await inTurn(memory, "current", async (turn) => {
        const before = turn.view;
        const error = await rejected(prepare(turn, calls, { result: bad }));
        expect(Object.getPrototypeOf(error)).toBe(cls.prototype);
        expect((error as Error).message).toBe(message);
        expect(turn.view).toBe(before);
        expect((await prepare(turn, calls, { result: "valid" }))[0]).toBe("valid");
        expect(calls.length).toBe(2);
      });
    });
  }
});

test("summary exception identity and no candidate publication", async () => {
  const memory = new SessionMemory({ historyBudgetChars: 1, recentTurns: 0 });
  await seed(memory, 1);
  const error = new Error("summary callback failed");
  await inTurn(memory, "current", async (turn) => {
    const fail = (): string => {
      throw error;
    };
    expect(await rejected(turn.prepareHistory({ projectionVersion: "test-v1", project, summarize: fail }))).toBe(error);
    const calls: Calls = [];
    expect((await prepare(turn, calls))[0]).toBe("summary");
    expect(calls.length).toBe(1);
  });
});

test("defaults keep four prior turns even over budget", async () => {
  const memory = new SessionMemory();
  const calls: Calls = [];
  await seed(memory, 5);
  const largeProject = (_view: ConversationView, selected: readonly ConversationInvocation[]): string =>
    "x".repeat(10_000 * selected.length);
  await inTurn(memory, "current", async (turn) => {
    const [summary, recent] = await prepare(turn, calls, { project: largeProject });
    expect(summary).toBe("summary");
    expect(recent.length).toBe(40_000);
  });
  expect(calls.length).toBe(1);
  expect(calls[0]![1]).toBe(32_000);
  const text = calls[0]![0];
  expect(text.slice(text.indexOf("\n") + 1)).toBe("x".repeat(10_000));
});

test("cache is private versioned and does not rewind to old pin", async () => {
  const memory = new SessionMemory({ historyBudgetChars: 1, recentTurns: 0 });
  const calls: Calls = [];
  await seed(memory, 1);
  const pinned = snapshot(memory);
  let session = memory.openRun({ snapshot: pinned });
  try {
    await withTurn(session, request("current"), async (turn) => {
      expect((await prepare(turn, calls, { result: "first" }))[0]).toBe("first");
      expect((await prepare(turn, calls, { version: "test-v2", result: "second" }))[0]).toBe("second");
      expect((await prepare(turn, calls))[0]).toBe("first");
      expect(calls.length).toBe(2);
    });
    expect(session.snapshot).toEqual(pinned);
  } finally {
    session.close();
  }
  session = memory.openRun({ snapshot: pinned });
  try {
    expect(session.snapshot).toEqual(pinned);
    await withTurn(session, request("later"), async (turn) => {
      expect((await prepare(turn, calls, { result: "extended" }))[0]).toBe("extended");
    });
  } finally {
    session.close();
  }
  expect(calls.at(-1)![0]).toContain("first");
  expect(calls.at(-1)![0]).toContain('"current"');
  expect(calls.at(-1)![0]).not.toContain('"0"');
  const fields = ["conversation", "generation", "namespace", "notes", "version"];
  expect(Object.keys(snapshot(memory)).toSorted()).toEqual(fields);
  memory.reset();
  await seed(memory, 1);
  await inTurn(memory, "fresh", async (turn) => {
    expect((await prepare(turn, calls))[0]).toBe("summary");
  });
  expect(calls.at(-1)![0]).not.toContain("Previous historical summary");
});

test("cache is shared by consumers and isolated by namespace", async () => {
  const memory = new SessionMemory({ historyBudgetChars: 1, recentTurns: 0 });
  const calls: Calls = [];
  await seed(memory, 1);
  await withHarness(async (opened) => {
    // A second consumer extends the same conversation; its invocation must differ from the first
    // seed's.
    await withTurn(opened.frame, request("other", 1, ["root", "other"]), (turn) =>
      appendReply(turn, "item-other-0", "x".repeat(20)),
    );
    await withTurn(opened.frame, request("current"), async (turn) => {
      await prepare(turn, calls, { result: "agent summary" });
    });
    await withTurn(opened.frame, request("current", 2, ["root", "other"]), async (turn) => {
      await prepare(turn, calls, { result: "other summary" });
    });
  }, memory);
  expect(calls.length).toBe(2);
  expect(calls.at(-1)![0]).toContain("agent summary");
  const restored = new SessionMemory({ historyBudgetChars: 1, recentTurns: 0 });
  const session = restored.openRun({ snapshot: snapshot(memory) });
  try {
    await withTurn(session, request("restored"), async (turn) => {
      await prepare(turn, calls);
    });
  } finally {
    session.close();
  }
  expect(calls.at(-1)![0]).not.toContain("Previous historical summary");
});

test("cached summary under budget does not extend eligible prefix", async () => {
  const memory = new SessionMemory({ historyBudgetChars: 20, recentTurns: 0 });
  const calls: Calls = [];
  await seed(memory, 5);
  await withHarness(async (opened) => {
    await withTurn(opened.frame, request("a"), async (turn) => {
      expect(await prepare(turn, calls, { result: "s" })).toEqual(["s", "[]"]);
    });
    await withTurn(opened.frame, request("b"), async (turn) => {
      expect(await prepare(turn, calls)).toEqual(["s", '["a"]']);
    });
  }, memory);
  expect(calls.length).toBe(1);
});

test("failed extension keeps previous summary", async () => {
  const memory = new SessionMemory({ historyBudgetChars: 1, recentTurns: 0 });
  const calls: Calls = [];
  await seed(memory, 1);
  await withHarness(async (opened) => {
    await withTurn(opened.frame, request("a"), async (turn) => {
      await prepare(turn, calls, { result: "original summary" });
    });
    await withTurn(opened.frame, request("b"), async (turn) => {
      expect(await rejected(prepare(turn, calls, { result: "" }))).toBeInstanceOf(RangeError);
      await prepare(turn, calls, { result: "valid extension" });
    });
  }, memory);
  expect(calls.at(-1)![0]).toContain("original summary");
  expect(calls.at(-1)![0]).toBe(calls.at(-2)![0]);
});

test("empty crashed and failed turns count in recent selection", async () => {
  const memory = new SessionMemory({ historyBudgetChars: 1, recentTurns: 2 });
  const calls: Calls = [];
  await seed(memory, 1);
  let crashedPin: Mutable;
  await withHarness(async (opened) => {
    const failedTurn = withTurn(opened.frame, request("failed"), () => {
      throw new Error("empty failed turn");
    });
    expect(await rejected(failedTurn)).toBeInstanceOf(Error);
    const turn = await opened.frame.openTurn(request("crashed"), { signal: NEVER });
    crashedPin = snapshot(memory);
    turn.close();
  }, memory);
  const restored = new SessionMemory({ historyBudgetChars: 1, recentTurns: 2 });
  const session = restored.openRun({ snapshot: crashedPin });
  try {
    await withTurn(session, request("current"), async (turn) => {
      expect((await prepare(turn, calls))[1]).toBe('["failed","crashed"]');
    });
  } finally {
    session.close();
  }
  expect(calls[0]![0]).toContain('"0"');
});

test("callback history change rejects candidate without rewinding", async () => {
  const memory = new SessionMemory({ historyBudgetChars: 1, recentTurns: 0 });
  const calls: Calls = [];
  await seed(memory, 1);
  await inTurn(memory, "current", async (turn) => {
    const changed = (): string => {
      appendReply(turn, "new", "real evidence");
      return "not publishable";
    };
    const error = (await rejected(
      turn.prepareHistory({ projectionVersion: "test-v1", project, summarize: changed }),
    )) as MemoryOperationError;
    expect(error).toBeInstanceOf(MemoryOperationError);
    expect([error.code, error.message]).toEqual(["session.conflict", "history changed during context preparation"]);
    expect(turn.view.items.at(-1)!.payload.content).toBe("real evidence");
    expect((await prepare(turn, calls))[0]).toBe("summary");
    expect(calls.length).toBe(1);
  });
});

test("close during projection prevents starting summary callback", async () => {
  const memory = new SessionMemory({ historyBudgetChars: 1, recentTurns: 0 });
  const calls: Calls = [];
  await seed(memory, 1);
  await withHarness(async (opened) => {
    await withTurn(opened.frame, request("current"), async (turn) => {
      const close = (view: ConversationView, selected: readonly ConversationInvocation[]): string => {
        opened.session.close();
        return project(view, selected);
      };
      const error = (await rejected(prepare(turn, calls, { project: close }))) as MemoryOperationError;
      expect(error).toBeInstanceOf(MemoryOperationError);
      expect([error.code, error.message]).toEqual(["session.stale", STALE_SESSION]);
    });
  }, memory);
  expect(calls).toEqual([]);
});

test("snapshot pins and learned notes survive summary and note commit", async () => {
  let memory = new SessionMemory({ historyBudgetChars: 1, recentTurns: 0 });
  await seed(memory, 1);
  const value: Mutable = structuredClone(snapshot(memory));
  value.notes = [note({ runId: "0", text: "Already learned" })];
  memory = new SessionMemory({ historyBudgetChars: 1, recentTurns: 0 });
  const learned = note({ runId: "0", index: 1n, text: "Learned during callback" });
  const session = memory.openRun({ snapshot: value });
  try {
    await withTurn(session, request("current"), async (turn) => {
      const original = turn.view;
      const summarize = (): string => {
        session.commit("notes", [mutation(learned)]);
        return "summary";
      };
      expect((await turn.prepareHistory({ projectionVersion: "test-v1", project, summarize }))[0]).toBe("summary");
      expect(turn.view).toBe(original);
    });
    expect(session.snapshot).toEqual(value);
  } finally {
    session.close();
  }
  const final: Mutable = snapshot(memory);
  expect(final.notes).toEqual([...value.notes, learned]);
  expect(final.conversation.items).toEqual(value.conversation.items);
});

describe("invalid projection does not call summary or poison cache", () => {
  for (const bad of [null, 1, [], {}]) {
    test(JSON.stringify(bad), async () => {
      const memory = new SessionMemory({ historyBudgetChars: 1, recentTurns: 0 });
      const calls: Calls = [];
      await seed(memory, 1);
      await inTurn(memory, "current", async (turn) => {
        const before = turn.view;
        const error = await rejected(prepare(turn, calls, { project: () => bad as string }));
        expect(Object.getPrototypeOf(error)).toBe(TypeError.prototype);
        expect((error as Error).message).toBe("history projection must be a string");
        expect(calls).toEqual([]);
        expect(turn.view).toBe(before);
        expect((await prepare(turn, calls))[0]).toBe("summary");
      });
    });
  }
});

describe("callback drains before fifo lease release", () => {
  for (const action of ["normal", "stop", "close"] as const) {
    test(action, async () => {
      const memory = new SessionMemory({ historyBudgetChars: 1, recentTurns: 1 });
      await seed(memory, 2);
      const entered = Promise.withResolvers<void>();
      const release = Promise.withResolvers<void>();
      const events: string[] = [];
      const stop = new AbortController();
      const primary = new Error("host stop");
      const queuedStop = new AbortController();
      const queuedError = new Error("queued cancelled");
      let ownerSession: Mutable;
      const owner = settle(
        withHarness(async (opened) => {
          ownerSession = opened.session;
          await withTurn(
            opened.frame,
            request("owner"),
            async (turn) => {
              await turn.prepareHistory({
                projectionVersion: "test-v1",
                project,
                summarize: async () => {
                  entered.resolve();
                  await release.promise;
                  events.push("summary returned");
                  return "retained actual summary";
                },
              });
              events.push("owner complete");
            },
            stop.signal,
          );
        }, memory),
      );
      const waiter = (name: string, signal: AbortSignal) =>
        settle(
          withHarness(
            (opened) =>
              withTurn(
                opened.frame,
                request(name),
                () => {
                  events.push(name);
                  expect(events).toContain("summary returned");
                  const texts = [...memory["summaries"].values()].map((summary) => summary.text);
                  expect(texts).toContain("retained actual summary");
                },
                signal,
              ),
            memory,
          ),
        );
      await entered.promise;
      let next: ReturnType<typeof waiter> | null = null;
      try {
        // Opening a run while the callback holds the lease proves the callback runs outside any
        // backend critical section.
        await withHarness((opened) => {
          expect(opened.session.snapshot.namespace).toBeTruthy();
        }, memory);
        events.push("other");
        const cancelled = waiter("cancelled", queuedStop.signal);
        await macrotask();
        expect(memory["queue"].length).toBe(1);
        next = waiter("next", NEVER);
        await macrotask();
        expect(memory["queue"].length).toBe(2);
        const before = snapshot(memory);
        failure(() => memory.reset(), "session.busy", RESET_BUSY);
        expect(snapshot(memory)).toEqual(before);
        queuedStop.abort(queuedError);
        expect(await rejected(cancelled.promise)).toBe(queuedError);
        expect(events).not.toContain("cancelled");
        if (action === "stop") stop.abort(primary);
        else if (action === "close") ownerSession.close();
        expect(events).not.toContain("next");
        expect(events).not.toContain("summary returned");
        failure(() => memory.reset(), "session.busy", RESET_BUSY);
      } finally {
        release.resolve();
        await owner.promise.catch(() => {});
        await next?.promise.catch(() => {});
      }
      expect(events.indexOf("summary returned")).toBeLessThan(events.indexOf("next"));
      expect(next!.error).toBeUndefined();
      if (action === "stop") {
        expect(owner.error).toBe(primary);
        expect(events).not.toContain("owner complete");
      } else if (action === "close") {
        const error = owner.error as MemoryOperationError;
        expect(error).toBeInstanceOf(MemoryOperationError);
        expect([error.code, error.message]).toEqual(["session.stale", STALE_SESSION]);
        expect(events).not.toContain("owner complete");
      } else {
        expect(owner.error).toBeUndefined();
      }
      memory.reset();
    });
  }
});

describe("TS-only", () => {
  test("prepareHistory checks its version and callbacks first", async () => {
    const memory = new SessionMemory({ historyBudgetChars: 1, recentTurns: 0 });
    await seed(memory, 1);
    await inTurn(memory, "current", async (turn) => {
      const summarize = (): string => "summary";
      const empty = await rejected(turn.prepareHistory({ projectionVersion: "", project, summarize }));
      expect(Object.getPrototypeOf(empty)).toBe(RangeError.prototype);
      expect((empty as Error).message).toBe("projection_version must be a nonempty string");
      const callback = await rejected(
        turn.prepareHistory({ projectionVersion: "test-v1", project: "no" as never, summarize }),
      );
      expect(Object.getPrototypeOf(callback)).toBe(TypeError.prototype);
      expect((callback as Error).message).toBe("history callbacks must be callable");
    });
  });

  test("async callbacks are awaited and a rejecting summary leaves the cache unchanged", async () => {
    const memory = new SessionMemory({ historyBudgetChars: 1, recentTurns: 0 });
    await seed(memory, 1);
    await inTurn(memory, "current", async (turn) => {
      const before = turn.view;
      const reason = new Error("async summary failed");
      const failing = turn.prepareHistory({
        projectionVersion: "test-v1",
        project: async (view, selected) => project(view, selected),
        summarize: async () => {
          throw reason;
        },
      });
      expect(await rejected(failing)).toBe(reason);
      expect(memory["summaries"].size).toBe(0);
      expect(turn.view).toBe(before);
      const result = await turn.prepareHistory({
        projectionVersion: "test-v1",
        project: async (view, selected) => project(view, selected),
        summarize: async () => "async summary",
      });
      expect(result).toEqual(["async summary", "[]"]);
      expect([...memory["summaries"].values()].map((summary) => summary.text)).toEqual(["async summary"]);
    });
  });

  test("the budget counts code points", async () => {
    for (const [text, summarized] of [
      ["😀😀", false],
      ["😀😀😀", true],
    ] as const) {
      const memory = new SessionMemory({ historyBudgetChars: 2, recentTurns: 0 });
      const calls: Calls = [];
      await seed(memory, 1);
      await inTurn(memory, "current", async (turn) => {
        await prepare(turn, calls, { project: () => text });
      });
      expect(calls.length).toBe(summarized ? 1 : 0);
    }
  });
});
