/**
 * LOOP: the self-respawning `while:`/`until:`/`times:` driver.
 *
 * Each loop iteration is a fresh `LoopNode` driver clone whose `run` owns the whole loop policy:
 * the continue-vs-stop decision, the iteration index, the runaway guard and the self-respawn
 * grow. There is no engine-side loop step: the engine core stays kind-blind (splice `Grow`, apply
 * `prune`, commit `Output`, fold the durable ledger).
 *
 * Names:
 *   L      the compiled loop node id (iteration-0 driver, `originId === L === this.id`).
 *   L~k    a fresh driver clone for iteration `k >= 1` (`iteration = k`, `originId = L`, id `L~k`).
 *   L#k/…  the cloned body for iteration `k` at callsite `mapCallsite(L, k)`, `L#k`.
 *
 * Per iteration (driver@k is `L` when k is 0, else `L~k`, run on the carried record):
 *   STOP: `Output(carried, commitAs = origin)`, a generic commit-and-advance under `L`.
 *   CONT: `Grow(seed = [carried, k], prune = {body_{k-1} ids} ∪ ({this.id} if not the origin))`.
 *   The reducer retires each body_{k-1} node with its whole grown subtree (nested CALL/MAP/LOOP)
 *   once every node in it is settled; an unfinished side step stays live until a later growth.
 *
 * The compile-owned injected builder constructs body and driver at the origin callsite
 * (`this.originId`), never `this.id`, so live and durable-replay windows are byte-identical. The
 * node also owns origin traversal for its cached body signature and exposes the baked body
 * through the source-neutral `originChildren` hook.
 *
 * Knows about:   nodes/base, nodes/contracts, nodes/ids, expr, diagnostics and typesys.
 * Never imports: compile, compose, runtime, the run boundary, the CLI, or provider SDKs.
 */

import { type FailureCode, type OriginRef, reprText } from "../../diagnostics.ts";
import { evaluateWhenRecord } from "../../expr/expressions.ts";
import type { FeedbackValue } from "../../expr/index.ts";
import {
  isInboxType,
  isRecord,
  normalizeFiles,
  setField,
  type Type,
  type Value,
  type ValueRecord,
} from "../../typesys/values.ts";
import {
  type BackwardContext,
  BackwardResult,
  Grow,
  Node,
  NodeKind,
  Output,
  parseNamedCondition,
  projectStructuralFeedback,
} from "../base.ts";
import {
  ExecutionPosition,
  ExecutionScope,
  isExactly,
  iterOriginCarrier,
  type OriginMapper,
  Parameter,
  PartialRecordContract,
  rewriteOriginCarrier,
  sameExecutionPosition,
  ValueContract,
} from "../contracts.ts";
import { mapCallsite, ns } from "../ids.ts";

/**
 * The while/until runaway guard tripped (the loop would start more than `max` body runs).
 *
 * Thrown only when the loop sets `max:`; a while/until loop without one has no limit.
 *
 * Thrown inside `LoopNode.run` so the evaluator records a located `Failure` whose code is
 * `loop.max_exceeded` and whose message names the exceeded limit.
 */
export class LoopMaxExceeded extends Error {
  override name = "LoopMaxExceeded";
}

/**
 * Each error a `while:`/`until:` predicate threw while evaluating. TS classifies such a failure
 * as `condition.evaluate` (Python: `node.execute`); marking the error rather than wrapping it
 * leaves it unchanged for `exceptionCode` to classify.
 */
const failedPredicates = new WeakSet<object>();

/**
 * A higher-order `('a -> 'a) -> 'a -> 'a` driver over a carried record.
 *
 * Its fields are baked at load by Compose and copied onto every clone by `respawn`.
 */
export class LoopNode extends Node {
  readonly kind = NodeKind.LOOP;
  /** Grows the graph: each continuing iteration resolves to a seed-first `Grow`. */
  override readonly isSpawner: boolean = true;
  /** The fixpoint-iteration driver: the ledger stays single-record. */
  override readonly isLoop: boolean = true;
  /** Its seed, and a stopping driver's value, only read earlier pool values through its binding. */
  override readonly derived: boolean = true;
  /** Each body inherits the enclosing call depth without consuming a level. */
  override readonly growDepthDelta: number | null = 0;
  /** The id of the body flow. */
  flowId: string;
  /** A pinned body flow version, if any. */
  flowVersion: string | null;
  /** The compiled body flow (`'a -> 'a`) used by the injected growth builder. */
  child: unknown;
  /**
   * The body's canonical parameters (a subset of the carried record); an omitted signature is
   * stored as an empty list, as CALL and MAP do.
   */
  childInputs: readonly Parameter[];
  /** One of `while`, `until` or `times`. */
  predicateKind: "while" | "until" | "times";
  /**
   * The boolean source for `while`/`until`, e.g. `not exited` (bare names read the carried
   * record); `null` for `times`.
   */
  predicate: string | null;
  /** The fixed run count when `predicateKind` is `times`; `null` otherwise. */
  times: number | null;
  /** The runaway guard for `while`/`until` (`null` is no limit); for `times`, the count N. */
  maxIters: number | null;
  /**
   * The carried fields the body returns, in carried order; `null` when the body returns every
   * carried field (a total loop). Each omitted field keeps the previous iteration's value through
   * `respawn`.
   */
  returnedFields: readonly string[] | null;
  /** This driver's iteration index `k`: 0 on the compiled `L`, `k` on a clone `L~k`. */
  iteration: number;

  /**
   * Build a LOOP driver.
   *
   * @param nodeId The node's unique id.
   * @param options.flowId The id of the body flow.
   * @param options.flowVersion A pinned body flow version, if any.
   * @param options.child The compiled body flow; `null` until Compose bakes it.
   * @param options.childInputs The body's canonical parameters; omitted, none.
   * @param options.predicateKind `while` (the default), `until` or `times`.
   * @param options.predicate The `while`/`until` boolean source; `null` for `times`.
   * @param options.times The fixed run count of a `times` loop.
   * @param options.maxIters The runaway guard, or the `times` count; `null` is no limit.
   * @param options.title A display title.
   * @param options.iteration This driver's iteration index; defaults to 0.
   * @param options.originId The origin loop id; omitted, the node's own id.
   * @param options.returnedFields The carried fields the body returns; `null` for all of them.
   */
  constructor(
    nodeId: string,
    options: {
      flowId: string;
      flowVersion?: string | null;
      child?: unknown;
      childInputs?: readonly Parameter[] | null;
      predicateKind?: "while" | "until" | "times";
      predicate?: string | null;
      times?: number | null;
      maxIters?: number | null;
      title?: string | null;
      iteration?: number;
      originId?: string | null;
      returnedFields?: readonly string[] | null;
    },
  ) {
    const {
      flowId,
      flowVersion = null,
      child = null,
      childInputs = null,
      predicateKind = "while",
      predicate = null,
      times = null,
      maxIters = null,
      title = null,
      iteration = 0,
      originId = null,
      returnedFields = null,
    } = options;
    super(nodeId, { title });
    this.flowId = flowId;
    this.flowVersion = flowVersion;
    this.child = child;
    this.childInputs = childInputs ?? [];
    this.predicateKind = predicateKind;
    this.predicate = predicate;
    this.times = times;
    this.maxIters = maxIters;
    this.returnedFields = returnedFields;
    this.iteration = iteration;
    // The compiled `L` sets originId to its own id (uniform); a clone `L~k` carries the origin.
    this.originId = originId || nodeId;
  }

  /** Return the compiled loop origin that receives a stopped driver output. */
  override skippedOutputTarget(): string {
    return this.originId ?? this.id;
  }

  /** Yield the common origins, then the cached body signature's. */
  override *iterLocalOrigins(seen: Set<object> = new Set()): Generator<OriginRef> {
    yield* super.iterLocalOrigins(seen);
    for (const parameter of this.childInputs) yield* iterOriginCarrier(parameter, seen);
  }

  /** Rewrite the common and the cached signature's origins on a copied LOOP. */
  override rewriteLocalOrigins(mapper: OriginMapper, seen: Map<object, object> = new Map()): void {
    super.rewriteLocalOrigins(mapper, seen);
    this.childInputs = this.childInputs.map((parameter) => rewriteOriginCarrier(parameter, mapper, seen));
  }

  /** Return the baked body flow, or no child for an unbaked LOOP. */
  override originChildren(): readonly unknown[] {
    return this.child === null ? [] : [this.child];
  }

  /**
   * Classify the runaway guard as `loop.max_exceeded` and a predicate that failed to evaluate as
   * `condition.evaluate`; defer everything else.
   */
  override exceptionCode(error: unknown): FailureCode {
    if (error instanceof LoopMaxExceeded) return "loop.max_exceeded";
    if (isObject(error) && failedPredicates.has(error)) return "condition.evaluate";
    return super.exceptionCode(error);
  }

  /**
   * Require the body flow that Compose must bake before execution.
   *
   * @throws Error when the body flow isn't baked.
   */
  override validateExecutionSetup(): void {
    super.validateExecutionSetup();
    if (this.child === null) throw new Error(`loop node ${reprText(this.id)} was not baked with a body child`);
  }

  /**
   * Parse the node's assertions, then its `while`/`until` predicate.
   *
   * @param options.path The static node path to name in the error instead of the node's id.
   * @throws ExpressionError when an assertion or the predicate doesn't parse; the message names
   *   the node, the keyword (`while:`/`until:`) and the expression.
   */
  override validateConditions(options: { path?: string | null } = {}): void {
    super.validateConditions(options);
    const { path = null } = options;
    if ((this.predicateKind === "while" || this.predicateKind === "until") && typeof this.predicate === "string") {
      parseNamedCondition(
        this.predicate,
        `node ${reprText(path || this.id)} \`${this.predicateKind}:\` ${reprText(this.predicate)}`,
        this.types,
      );
    }
  }

  /**
   * Capture every carried input, including fields the body doesn't consume.
   *
   * @returns The `input_output_paths` record from each carried input name to its same-named field
   *   in a stopped driver's record output.
   */
  override captureBackwardContext(): Record<string, unknown> {
    return { input_output_paths: Object.fromEntries(this.backwardInputNames().map((name) => [name, [name]])) };
  }

  /**
   * Project a stopped driver's recorded carried output without testing its predicate.
   *
   * @param context Recorded carried inputs, accepted output or failure, the full input inventory
   *   and the captured named paths.
   * @param feedback Ordered text or structural selections over the accepted carried record.
   * @returns Input messages for a successful stopping driver. Failed or continuing drivers stop;
   *   growth connections handle the separate continuation boundary.
   * @throws TypeError when a message batch or captured path has an invalid type.
   * @throws RangeError when feedback or a captured path doesn't match the recorded output, or the
   *   required structural layout is missing.
   */
  override backward(context: BackwardContext, feedback: readonly FeedbackValue[]): BackwardResult {
    if (context.failure !== null || !isRecord(context.output)) return new BackwardResult();
    return projectStructuralFeedback(context, feedback);
  }

  /**
   * The loop's hard iteration budget: `true` once `iteration` has reached `maxIters`.
   *
   * Pure, node-owned budget policy; always `false` when there is no max (`maxIters` is `null`).
   * `run` consults this both as the `times` stop count and as the `while`/`until` runaway guard.
   */
  shouldStop(iteration: number): boolean {
    if (this.maxIters === null) return false;
    return iteration >= this.maxIters;
  }

  /**
   * Return a fresh driver clone for iteration `k` at id `` `${originId}~${k}` ``.
   *
   * The clone carries the same baked fields and the origin id, so it attributes its grow and
   * commit to the compiled loop and self-prunes (`id !== originId`). Pure: no engine state.
   *
   * A total loop shares the origin's `params`. For a partial loop (`returnedFields` set), each
   * carried field the body omits gets a `Parameter` default holding its value in `carried`, read
   * through the body's typed input of that name (`childInputs`) so an omitted file stays a
   * `FileRef`; the next driver then binds the merged record. `carried` is the JSON-decoded
   * growth-seed record that seeds the body's START, so live growth and durable replay merge
   * identically. The origin, origin context, memory scope, output contract, asserts and type
   * table are copied; the execution scope moves to iteration `k`.
   *
   * @param k The clone's iteration index.
   * @param carried The record this iteration's body was seeded with; required when
   *   `returnedFields` is set.
   * @returns The driver for iteration `k`.
   * @throws RangeError when `returnedFields` is set and `carried` is `null`, or the execution
   *   scope doesn't end in this driver's position.
   * @throws TypeError when the execution scope isn't an `ExecutionScope`.
   */
  respawn(k: number, carried: Readonly<Record<string, Value>> | null = null): LoopNode {
    const originId = this.originId ?? this.id;
    if (this.returnedFields !== null && carried === null) {
      throw new RangeError(
        `loop ${reprText(originId)} returns a subset of its record; respawn needs the carried record`,
      );
    }
    const clone = new LoopNode(`${originId}~${k}`, {
      flowId: this.flowId,
      flowVersion: this.flowVersion,
      child: this.child,
      childInputs: this.childInputs,
      predicateKind: this.predicateKind,
      predicate: this.predicate,
      times: this.times,
      maxIters: this.maxIters,
      title: this.title,
      iteration: k,
      originId,
      returnedFields: this.returnedFields,
    });
    clone.origin = this.origin;
    // Keep only the enclosing ancestry: `failureTrace()` appends this driver's authored origin, so
    // copying that origin into the context would duplicate it on every later iteration.
    clone.originContext = this.originContext;
    clone.memoryScope = this.memoryScope;
    if (this.executionScope !== null) {
      const scope = this.executionScope;
      if (!isExactly(scope, ExecutionScope)) throw new TypeError("loop execution scope must be an ExecutionScope");
      const { staticPath, flowPath, occurrencePath, boundary } = scope;
      // Re-validate a scope that may have bypassed the constructor, as Python's rebuild does.
      new ExecutionScope(staticPath, { flowPath, occurrencePath, boundary });
      const current = new ExecutionPosition("loop_driver", staticPath, this.iteration);
      const last = occurrencePath.at(-1);
      if (last === undefined || !sameExecutionPosition(last, current)) {
        throw new RangeError("loop scope must end in its current driver position");
      }
      clone.executionScope = new ExecutionScope(staticPath, {
        flowPath,
        occurrencePath: [...occurrencePath.slice(0, -1), new ExecutionPosition("loop_driver", staticPath, k)],
        boundary,
      });
    }
    if (this.returnedFields === null) {
      clone.params = this.params;
    } else {
      const returned = this.returnedFields;
      const record = carried!;
      clone.params = this.params?.map((param) =>
        Object.hasOwn(record, param.name) && !returned.includes(param.name)
          ? new Parameter(param.name, {
              contract: param.contract,
              coerceStrings: param.coerceStrings,
              origin: param.origin,
              defaultOrigin: param.defaultOrigin,
              required: false,
              hasDefault: true,
              default: this.carriedValue(param.name, record[param.name]!),
            })
          : param,
      ) ?? null;
    }
    clone.outputContract = this.outputContract;
    clone.preAsserts = [...this.preAsserts];
    clone.postAsserts = [...this.postAsserts];
    clone.types = this.types;
    return clone;
  }

  /**
   * Read a JSON-decoded carried field as the body's typed input `name` reads it.
   *
   * The driver's own params have no contract, so the body's START parameter gives the field's
   * type; a `file` position turns the file's JSON form back into a `FileRef`. An untyped or
   * unknown field is returned as given.
   */
  private carriedValue(name: string, value: Value): Value {
    for (const parameter of this.childInputs) {
      if (parameter.name === name && parameter.contract !== null) {
        return normalizeFiles(parameter.contract.type, value);
      }
    }
    return value;
  }

  /**
   * Own the whole loop policy: decide continue-vs-stop on the carried record, and on continue
   * return a self-respawning `Grow` (body_k and a fresh `L~(k+1)` driver) that prunes this driver
   * and the previous body.
   *
   * Stop rule (`k = this.iteration`, the index of the body about to run):
   *   while: stop when the predicate is false on the carried record.
   *   until: do-while; stop when `k >= 1` and the predicate is true (`k` 0 always continues).
   *   times: stop when `k >= this.maxIters` (`times N` is baked as `maxIters`).
   * On stop: `Output(carried, commitAs = origin)`, without the fields the body takes as inboxes.
   * On continue: the runaway guard, only when `max:` is set, then `Grow([carried, k], prune)`,
   * pruning body_{k-1} and this driver unless it is the origin. The reducer retires each
   * body_{k-1} node with its whole grown subtree once every node in it is settled.
   *
   * @param inputs The carried record, keyed by the loop's input field names.
   * @returns The stopping `Output` or the continuing `Grow`; the seed's iteration is a `bigint`.
   * @throws LoopMaxExceeded when a `while`/`until` loop would start more than `max` body runs.
   * @throws ExpressionError (unchanged) when the predicate fails to evaluate.
   */
  override forward(inputs: Record<string, unknown>): Output | Grow {
    const origin = this.originId ?? this.id;
    const carried = this.readCarried({ ...inputs } as ValueRecord);
    const k = this.iteration;
    if (this.shouldStopNow(carried, k)) return new Output(this.withoutInboxes(carried), { commitAs: origin });
    // Guard on `shouldStop(k)`, not `k + 1`: driver@k grows body_k, the (k+1)-th body, which is
    // within budget while `k < maxIters`, so `max: M` permits exactly M body runs. A `times` loop
    // stops above instead.
    if ((this.predicateKind === "while" || this.predicateKind === "until") && this.shouldStop(k)) {
      throw new LoopMaxExceeded(`loop ${reprText(origin)} exceeded max (${this.maxIters})`);
    }
    const prune = new Set<string>();
    if (k >= 1) {
      for (const childId of (this.child as { nodes: ReadonlyMap<string, unknown> }).nodes.keys()) {
        prune.add(ns(mapCallsite(origin, k - 1), childId));
      }
    }
    // Self-prune, never the origin `L`.
    if (this.id !== origin) prune.add(this.id);
    return new Grow([carried, BigInt(k)], prune);
  }

  /**
   * Read the carried record through the loop's output contract.
   *
   * A field can reach the driver holding a file as its JSON form: a returned field bound from a
   * whole-reference body output, since the body's END has no contract for it, and an omitted
   * field the body doesn't take as input, since its carried default has no body parameter to read
   * it through. Reading every typed field through the contract the record is committed under
   * makes each `file` position a `FileRef` and each `object` position's nested files mappings, so
   * the predicate reads every iteration alike. Untyped fields are read as given.
   */
  private readCarried(carried: ValueRecord): ValueRecord {
    const contract = this.outputContract;
    if (isExactly(contract, ValueContract)) return normalizeFiles(contract.type, carried) as ValueRecord;
    if (isExactly(contract, PartialRecordContract)) {
      const typed = new Map<string, Type>();
      for (const [name, field] of contract.fields) if (field !== null) typed.set(name, field.type);
      const read: ValueRecord = {};
      for (const [name, value] of Object.entries(carried)) {
        const type = typed.get(name);
        setField(read, name, type === undefined ? value : normalizeFiles(type, value));
      }
      return read;
    }
    return carried;
  }

  /** Return `carried` without the fields the body takes as inboxes: a loop carries them, never returns them. */
  private withoutInboxes(carried: ValueRecord): ValueRecord {
    const inboxes = this.childInputs.filter((param) => isInboxType(param.contract?.type)).map((param) => param.name);
    if (inboxes.length === 0) return carried;
    const kept: ValueRecord = {};
    for (const [name, value] of Object.entries(carried)) if (!inboxes.includes(name)) setField(kept, name, value);
    return kept;
  }

  /** Continue-vs-stop on the carried record at index `k`; see `forward` for the stop rule. */
  private shouldStopNow(carried: ValueRecord, k: number): boolean {
    if (this.predicateKind === "times") return this.shouldStop(k);
    // Do-while: the seed is never tested.
    if (this.predicateKind === "until" && k === 0) return false;
    let truth: boolean;
    try {
      // The setup check has parsed the `while`/`until` predicate, a string.
      truth = evaluateWhenRecord(this.predicate as string, carried, this.types);
    } catch (error) {
      if (isObject(error)) failedPredicates.add(error);
      throw error;
    }
    return this.predicateKind === "while" ? !truth : truth;
  }
}

/** Return whether `value` can be held in a `WeakSet`. */
function isObject(value: unknown): value is object {
  return (typeof value === "object" && value !== null) || typeof value === "function";
}
