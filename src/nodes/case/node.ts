/**
 * CASE: deterministic conditional routing.
 *
 * Strict (a pure function of its declared inputs): the node declares `inputs` (bound via `from:`
 * like any leaf), and each case's `when:` interpolates only those inputs as bare `${name}`,
 * evaluated against the bound input record, not the pool. Routing is purely deterministic: the
 * engine makes no LLM decision here. To branch on a judgment, an upstream node produces a variable
 * (e.g. a classifier AGENT writing a label, or a CODE node a number) which the CASE binds as an
 * input and the case compares. Cases are tried in order; the first truthy one wins. If none match,
 * the reserved else handle (`ELSE_HANDLE`) is taken.
 *
 * The node writes no state: it returns the selected handle on its result, and the engine's
 * branching does the skip-flood.
 *
 * Knows about:   expr, nodes/base, nodes/contracts and diagnostics.
 * Never imports: compile, compose, runtime, the run boundary, the CLI, or provider SDKs.
 */

import { type FailureCode, type OriginRef, reprText } from "../../diagnostics.ts";
import { evaluateWhenRecord, localDeclared } from "../../expr/expressions.ts";
import type { ValueRecord } from "../../typesys/values.ts";
import { ELSE_HANDLE, isWeakKey, Node, type NodeCaps, NodeKind, parseNamedCondition, Route } from "../base.ts";
import { mapOrigin, type OriginMapper, validateOptionalOrigin } from "../contracts.ts";

/**
 * One ordered branch predicate and its labeled route. Frozen.
 *
 * `origin` is the optional source-neutral identity of this exact `when` occurrence. Duplicate
 * expression text remains distinct by list position.
 */
export class Case {
  /** The branch label; matches the outgoing edge's source handle. */
  readonly handle: string;
  /** The branch predicate, evaluated against the bound input record. */
  readonly when: string | null;
  /** The authored predicate occurrence, when known. */
  readonly origin: OriginRef | null;

  /**
   * Build one branch; `when` and `origin` default to `null`.
   *
   * @throws TypeError when `origin` isn't an `OriginRef` or `null`.
   * @throws RangeError when `handle` is the reserved else handle.
   */
  constructor(handle: string, options: { when?: string | null; origin?: OriginRef | null } = {}) {
    const { when = null, origin = null } = options;
    validateOptionalOrigin(origin, "CASE predicate origin");
    if (handle === ELSE_HANDLE) {
      throw new RangeError(
        `CASE handle ${reprText(handle)} is reserved for the else route; name the branch by its target node id`,
      );
    }
    this.handle = handle;
    this.when = when;
    this.origin = origin;
    Object.freeze(this);
  }
}

/**
 * The `Case` whose `when` threw each predicate evaluation error, so the error itself stays
 * unchanged; an error carries no frames to read the active `Case` from.
 */
const failedCases = new WeakMap<object, Case>();

/**
 * Deterministic conditional routing, with no LLM decision.
 *
 * Cases are tried in order; the first whose `when:` is truthy (evaluated against the bound input
 * record) wins and its handle is returned on the result. If none match, the reserved else handle
 * (`ELSE_HANDLE`) is taken. The node writes no value: the engine's branching does the skip-flood
 * off the chosen handle.
 */
export class CaseNode extends Node {
  readonly kind = NodeKind.CASE;
  /** The labeled branches, tried in order; each must carry a `when` expression. */
  cases: readonly Case[];

  /**
   * Build a CASE node.
   *
   * @param nodeId The node's unique id.
   * @param cases The labeled branches, tried in order.
   * @param options.title A display title.
   */
  constructor(nodeId: string, cases: readonly Case[], options: { title?: string | null } = {}) {
    const { title = null } = options;
    super(nodeId, { title });
    this.cases = cases;
  }

  /**
   * Require every compiled branch to carry its predicate expression.
   *
   * @throws TypeError when a case's origin isn't an `OriginRef` or `null`, or its `when` isn't a
   *   string.
   * @throws RangeError when a case holds the reserved else handle or has no `when`.
   */
  override validateExecutionSetup(): void {
    super.validateExecutionSetup();
    for (const branch of this.cases) {
      validateOptionalOrigin(branch.origin, "CASE predicate origin");
      if (branch.handle === ELSE_HANDLE) {
        throw new RangeError(
          `node ${reprText(this.id)} case handle ${reprText(branch.handle)} is reserved for the ` +
            "else route; name the branch by its target node id",
        );
      }
      if (branch.when === null) {
        throw new RangeError(`node ${reprText(this.id)} case ${reprText(branch.handle)} has no \`when\` expression`);
      }
      if (typeof branch.when !== "string") {
        throw new TypeError(`node ${reprText(this.id)} case ${reprText(branch.handle)} \`when\` must be a string`);
      }
    }
  }

  /**
   * Parse the node's assertions, then every case's `when` expression.
   *
   * @param options.path The static node path to name in the error instead of the node's id.
   * @throws ExpressionError when an assertion or a `when` doesn't parse; the message names the
   *   node, the case handle and the expression.
   */
  override validateConditions(options: { path?: string | null } = {}): void {
    super.validateConditions(options);
    const { path = null } = options;
    for (const branch of this.cases) {
      // A missing or non-string `when` is rejected by `validateExecutionSetup`.
      if (typeof branch.when === "string") {
        parseNamedCondition(
          branch.when,
          `node ${reprText(path || this.id)} case ${reprText(branch.handle)} \`when:\` ${reprText(branch.when)}`,
          this.types,
        );
      }
    }
  }

  /** Yield the common origins, then the ordered predicate origins, each `Case` once. */
  override *iterLocalOrigins(seen: Set<object> = new Set()): Generator<OriginRef> {
    yield* super.iterLocalOrigins(seen);
    for (const branch of this.cases) {
      if (seen.has(branch)) continue;
      seen.add(branch);
      validateOptionalOrigin(branch.origin, "CASE predicate origin");
      if (branch.origin !== null) yield branch.origin;
    }
  }

  /**
   * Rewrite the common and the ordered predicate origins on a copied CASE.
   *
   * Assigns a fresh `cases` array of rebuilt `Case`s; a `Case` shared by several arms is rebuilt
   * once and stays shared.
   */
  override rewriteLocalOrigins(mapper: OriginMapper, seen: Map<object, object> = new Map()): void {
    super.rewriteLocalOrigins(mapper, seen);
    this.cases = this.cases.map((branch) => {
      const done = seen.get(branch);
      if (done !== undefined) return done as Case;
      validateOptionalOrigin(branch.origin, "CASE predicate origin");
      const rebuilt = new Case(branch.handle, { when: branch.when, origin: mapOrigin(branch.origin, mapper) });
      seen.set(branch, rebuilt);
      return rebuilt;
    });
  }

  /**
   * Exclude lowered CASE controls from ordinary declared input targets.
   *
   * @returns Declared names except the generated `__on` and `__rN` controls, preserving directly
   *   declared ordinary inputs in order.
   */
  override backwardInputNames(): readonly string[] {
    return super.backwardInputNames().filter((name) => name !== "__on" && !/^__r[0-9]+$/.test(name));
  }

  /**
   * Return the failure code for one body error: `condition.evaluate` for a predicate that failed
   * to evaluate, else the base's.
   */
  override exceptionCode(error: unknown): FailureCode {
    const failed = isWeakKey(error) ? failedCases.get(error) : undefined;
    return failed !== undefined && this.cases.includes(failed) ? "condition.evaluate" : super.exceptionCode(error);
  }

  /**
   * Return the exact ordered predicate whose evaluation threw.
   *
   * The `Case` is matched by identity, so duplicate predicate text stays independently
   * locatable. Failures outside predicate evaluation, and source-free predicates, fall back to
   * the authored node origin.
   */
  override exceptionOrigin(error: unknown): OriginRef | null {
    const failed = isWeakKey(error) ? failedCases.get(error) : undefined;
    if (failed === undefined || !this.cases.includes(failed)) return super.exceptionOrigin(error);
    validateOptionalOrigin(failed.origin, "CASE predicate origin");
    return failed.origin ?? super.exceptionOrigin(error);
  }

  /**
   * Route on the bound input record: the first case whose `when` is truthy, else the else handle.
   * Each `when` reads this node's declared inputs by bare name, not the pool, through their
   * declared types (`caps.localTypes`).
   *
   * @throws ExpressionError (unchanged) when a `when` fails to evaluate.
   */
  override forward(inputs: Record<string, unknown>, caps: NodeCaps = {}): Route {
    const declared = localDeclared(caps.localTypes ?? null);
    for (const branch of this.cases) {
      try {
        if (evaluateWhenRecord(branch.when as string, inputs as ValueRecord, this.types, declared)) {
          return new Route(branch.handle);
        }
      } catch (error) {
        if (isWeakKey(error)) failedCases.set(error, branch);
        throw error;
      }
    }
    return new Route(ELSE_HANDLE);
  }
}
