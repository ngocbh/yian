/**
 * Call nesting and child resolution: a port of `tests/compose/test_call_nesting_limit.py`
 * (`test_call_nesting_counts_levels_not_paths`, over fake flows with `nodes` and
 * `originChildren()`).
 *
 * TS-only: direct rows for `resolveChild`, which Python tests only through `load_flow`: a
 * `LoadError` from the resolver passes through; another error becomes `could not resolve`, and so
 * does a `null` resolver (Python's tail is CPython's `'NoneType' object is not callable`); the
 * check before resolving with the count at the limit; the check after resolving with a deep fake
 * child; `callNestingScope` restores the count after a throw; the limit follows
 * `YIAN_MAX_CALL_NESTING`; and a MAP or LOOP build whose child already nests at the limit fails,
 * since a MAP and a LOOP resolve their child through `resolveChild` as a CALL does.
 *
 * Later (they need the loader): `test_chain_at_the_limit_loads_runs_and_resumes`,
 * `test_chain_past_the_limit_is_located_at_the_root_call`,
 * `test_long_chain_fails_at_the_first_call_past_the_limit`,
 * `test_chain_written_callee_first_is_caught_from_built_callables`,
 * `test_imports_load_before_defs`, `test_broken_import_is_reported_at_its_alias_before_defs`,
 * `test_alias_collision_wins_over_a_broken_target`,
 * `test_resolver_exceptions_become_load_errors_at_the_alias`, `test_uses_files_count`,
 * `test_long_uses_chain_fails_at_the_import_past_the_limit`, `test_import_only_chain_counts`,
 * `test_import_depth_does_not_depend_on_load_order`,
 * `test_import_depth_holds_through_a_memoizing_resolver`,
 * `test_resolver_that_calls_load_flow_is_bounded`,
 * `test_import_chain_follows_the_configured_limit`,
 * `test_resolved_flow_with_a_bad_import_depth_is_refused`,
 * `test_loaded_flow_records_its_import_depth`,
 * `test_imports_and_a_calling_defs_chain_count_apart`,
 * `test_imports_at_the_limit_over_defs_at_the_limit_load` and
 * `test_long_cross_file_cycle_reports_the_depth_limit` (T3-22); `test_inline_bodies_count`
 * (T3-17); `test_call_directive_counts` and `test_map_and_loop_bodies_count` (T3-22 and T4, their
 * chains of `defs:` loaded and run).
 */

import { afterEach, beforeEach, describe, expect, test } from "bun:test";

import { LoadDiagnostics, LoadError } from "../../src/compose/helpers/errors.ts";
import { BuildContext } from "../../src/compose/nodes/base.ts";
import { LoopDescriptor } from "../../src/compose/nodes/loop.ts";
import { MapDescriptor } from "../../src/compose/nodes/map.ts";
import {
  callNesting,
  callNestingScope,
  type ChildResolver,
  type ResolvedChild,
  resolveChild,
} from "../../src/compose/nodes/resolver.ts";
import type { CompiledFlow } from "../../src/compile/model.ts";
import { forget, reload } from "../../src/defaults.ts";

/** A node whose only static child is `child`, counting how often it is asked. */
class FakeNode {
  calls = 0;
  readonly child: object | null;

  constructor(child: object | null) {
    this.child = child;
  }

  originChildren(): readonly unknown[] {
    this.calls += 1;
    return this.child !== null ? [this.child] : [];
  }
}

/** A flow with the given nodes keyed by position. */
class FakeFlow {
  readonly nodes: Map<string, FakeNode>;

  constructor(nodes: FakeNode[]) {
    this.nodes = new Map(nodes.map((node, i) => [`n${i}`, node]));
  }
}

/** Return a flow nesting `levels` bodies, one per level. */
function chain(levels: number): FakeFlow {
  let flow = new FakeFlow([]);
  for (let i = 0; i < levels; i++) flow = new FakeFlow([new FakeNode(flow)]);
  return flow;
}

/** Return a resolved child whose compiled flow nests `levels` bodies. */
function resolved(levels: number): ResolvedChild {
  return { compiled: chain(levels) as unknown as CompiledFlow, input: [] };
}

/** Return the error `fn` throws. */
function thrown(fn: () => unknown): unknown {
  try {
    fn();
  } catch (error) {
    return error;
  }
  throw new Error("expected a throw");
}

/** Run `fn` inside `depth` nested calls of `resolveChild`, the innermost calling `fn`. */
function nestedCalls<T>(depth: number, fn: () => T): T {
  if (depth === 0) return fn();
  let result!: T;
  const resolver: ChildResolver = () => {
    result = nestedCalls(depth - 1, fn);
    return resolved(0);
  };
  resolveChild(`n${depth}`, `c${depth}`, resolver);
  return result;
}

/** Return the error resolving `n`'s callable `c` to `child` throws. */
function resolveError(child: ResolvedChild): unknown {
  return thrown(() => resolveChild("n", "c", () => child));
}

beforeEach(() => forget());
afterEach(() => forget());

test("call nesting counts levels, not paths", () => {
  // Each flow's two nodes share the next flow, so there are 2**40 paths but 40 levels.
  let flow = new FakeFlow([]);
  const nodes: FakeNode[] = [];
  for (let i = 0; i < 40; i++) {
    const pair = [new FakeNode(flow), new FakeNode(flow)];
    nodes.push(...pair);
    flow = new FakeFlow(pair);
  }

  expect(callNesting(flow)).toBe(40);
  expect(nodes.reduce((sum, node) => sum + node.calls, 0)).toBe(80);
  expect(nodes.every((node) => node.calls === 1)).toBe(true);
});

describe("resolveChild", () => {
  test("returns the resolved child, asking for no version", () => {
    const child = resolved(3);
    const asked: [string, string | null][] = [];
    const got = resolveChild("n", "c", (flowId, version) => {
      asked.push([flowId, version]);
      return child;
    });
    expect(got).toBe(child);
    expect(asked).toEqual([["c", null]]);
  });

  test("a LoadError from the resolver passes through", () => {
    const inner = new LoadDiagnostics().error("inner failure");
    const error = thrown(() =>
      resolveChild("n", "c", () => {
        throw inner;
      }),
    );
    expect(error).toBe(inner);
  });

  test("another error becomes could not resolve, with the error as its cause", () => {
    const inner = new Error("no such flow");
    const error = thrown(() =>
      resolveChild("n", "c", () => {
        throw inner;
      }),
    );
    expect(error).toBeInstanceOf(LoadError);
    expect((error as LoadError).message).toBe("call node 'n': could not resolve callable 'c': no such flow");
    expect((error as LoadError).cause).toBe(inner);
  });

  test("a thrown non-Error is named by its text", () => {
    const error = thrown(() =>
      resolveChild("n", "c", () => {
        throw "gone";
      }),
    );
    expect((error as LoadError).message).toBe("call node 'n': could not resolve callable 'c': gone");
  });

  test("a null resolver can't resolve the callable", () => {
    const error = thrown(() => resolveChild("c", "child", null));
    expect(error).toBeInstanceOf(LoadError);
    expect((error as LoadError).message).toBe(
      "call node 'c': could not resolve callable 'child': no child resolver was given",
    );
  });

  test("a child label names the callable", () => {
    const error = thrown(() =>
      resolveChild(
        "n",
        "c",
        () => {
          throw new Error("x");
        },
        "the def 'c'",
      ),
    );
    expect((error as LoadError).message).toBe("call node 'n': could not resolve the def 'c': x");
  });

  test("the count at the limit fails before resolving", () => {
    let asked = 0;
    const error = callNestingScope(() =>
      nestedCalls(50, () =>
        thrown(() =>
          resolveChild("deep", "c", () => {
            asked += 1;
            return resolved(0);
          }),
        ),
      ),
    );
    expect(asked).toBe(0);
    expect((error as LoadError).message).toBe(
      "call node 'deep': calling callable 'c' nests calls more than 50 levels deep, past the call " +
        "nesting limit of 50 (each call, map, or loop body is a level); flatten the chain",
    );
  });

  test("one below the limit still resolves", () => {
    const child = resolved(0);
    expect(callNestingScope(() => nestedCalls(49, () => resolveChild("n", "c", () => child)))).toBe(child);
  });

  test("a child built deeper than the limit fails after resolving", () => {
    expect(resolveChild("n", "c", () => resolved(49)).compiled).toBeDefined();
    const error = resolveError(resolved(50));
    expect((error as LoadError).message).toBe(
      "call node 'n': calling callable 'c' nests calls 51 levels deep, past the call nesting " +
        "limit of 50 (each call, map, or loop body is a level); flatten the chain",
    );
  });

  test("the post-check counts the active calls", () => {
    const error = callNestingScope(() => nestedCalls(10, () => resolveError(resolved(40))));
    expect((error as LoadError).message).toContain("nests calls 51 levels deep");
  });

  test("the limit follows YIAN_MAX_CALL_NESTING", () => {
    reload((name) => (name === "YIAN_MAX_CALL_NESTING" ? "3" : undefined));
    expect(resolveChild("n", "c", () => resolved(2)).compiled).toBeDefined();
    const after = resolveError(resolved(3));
    expect((after as LoadError).message).toContain(
      "nests calls 4 levels deep, past the call nesting limit of 3",
    );
    const before = callNestingScope(() => nestedCalls(3, () => resolveError(resolved(0))));
    expect((before as LoadError).message).toContain(
      "more than 3 levels deep, past the call nesting limit of 3",
    );
  });
});

test.each([
  ["map", new MapDescriptor({ id: "m", call: "c", over: "${input.xs}" })],
  ["loop", new LoopDescriptor({ id: "m", call: "c", times: 1n })],
])("a %s whose child already nests at the limit fails", (_kind, desc) => {
  const ctx = new BuildContext({ registry: new Map(), resolver: () => resolved(50) });

  const error = thrown(() => desc.build(ctx));

  expect(error).toBeInstanceOf(LoadError);
  expect((error as LoadError).message).toBe(
    "call node 'm': calling callable 'c' nests calls 51 levels deep, past the call nesting " +
      "limit of 50 (each call, map, or loop body is a level); flatten the chain",
  );
});

describe("callNestingScope", () => {
  test("counts from zero and restores the count after a throw", () => {
    reload((name) => (name === "YIAN_MAX_CALL_NESTING" ? "2" : undefined));
    const outcome = nestedCalls(2, () => {
      // A fresh scope counts from zero, so two more levels fit inside the outer two.
      expect(callNestingScope(() => nestedCalls(2, () => "inner"))).toBe("inner");
      expect(() =>
        callNestingScope(() =>
          nestedCalls(1, () => {
            throw new Error("boom");
          }),
        ),
      ).toThrow("boom");
      // Back in the outer scope at the limit: a third level fails before resolving.
      return resolveError(resolved(0));
    });
    expect((outcome as LoadError).message).toContain("more than 2 levels deep");
    // Outside every call the count is zero again.
    expect(nestedCalls(2, () => "done")).toBe("done");
  });
});
