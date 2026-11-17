/**
 * kind=call|map: function application of a callable, the `call` node's descriptor.
 *
 * Owns the `call` node's `CallDescriptor`, which inherits the template `parse` from
 * `NodeDescriptor` (which records each field's `SourceRange`) and rejects a `call:` YAML read as
 * non-text. `map.ts`'s `MapDescriptor` subclasses it, adding only the map field surface (`over:`,
 * `parallel:`) and `kind = "map"`; `map` imports `call`, never the reverse, since an ES module
 * cycle between a class and its subclass's module throws when `call.ts` loads first. `call` and
 * `map` share the resolve-and-bake `build`, which builds a `MapNode` for a `map` and a
 * `CallNode` otherwise. Registered in `KINDS` and `BINDING_FIELDS` by `compose/nodes/index.ts`.
 *
 * Knows about: diagnostics, the CALL and MAP runtime nodes, node contracts, typesys, compose
 * helpers, `base.ts` and `resolver.ts`.
 */

import { reprText, type SourceRange } from "../../diagnostics.ts";
import { CallNode } from "../../nodes/call/node.ts";
import { ContractOrigins, isExactly, type Parameter, ValueContract } from "../../nodes/contracts.ts";
import { MapNode } from "../../nodes/map/node.ts";
import { isInboxType, isMemoryType, listKindForElement, Type } from "../../typesys/values.ts";
import type { LoadDiagnostics } from "../helpers/errors.ts";
import type { SourceCursor } from "../source/index.ts";
import {
  type BuildContext,
  type BuildResult,
  NodeDescriptor,
  type NodeDescriptorFields,
  readCallTarget,
  siteKey,
} from "./base.ts";
import { callableDisplay, type ChildSignature, childSignature, resolveChild } from "./resolver.ts";

/**
 * Says whether a `call` node's `call:` value names a tool, which the node then runs as a TOOL
 * node, or a flow; the flow file's rule, given the node's id and where its `call:` is. A lifted
 * `call(...)` gives its `site` (`node 'n' input 'x' call(lessons)`), which a message names in
 * place of the node, writing its fix as a directive. A `map` or `loop` gives its `kind`: a loop
 * runs any tool, and a map any but `take`.
 *
 * @throws LoadError at `range` for a name no `call:` may take, or `take` on a `map`.
 */
export type CallsTool = (
  nodeId: string,
  call: unknown,
  range: SourceRange | null,
  where?: { site?: string | null; kind?: CallingKind },
) => boolean;

/** The kinds whose `call:` names what they run. */
export type CallingKind = "call" | "map" | "loop";

/** The fields `CallDescriptor`'s constructor takes: the common ones and the CALL's own. */
export interface CallDescriptorFields extends NodeDescriptorFields {
  call: string;
  def?: string | null;
  childLabel?: string | null;
  over?: unknown;
  parallel?: unknown;
  asserts?: readonly string[] | null;
}

/**
 * kind=call|map: apply a callable (`call:`) to the call-argument bindings in `input:`.
 *
 * `call` names the callable: a `defs:` entry or a `uses:` alias (the loader composes that
 * resolution). A `call` node whose `call:` names a tool parses as a TOOL node instead
 * (`parseNodes`), and a map or loop naming one calls a body generated for it
 * (`extractInlineBodies`). `def` names a def of the
 * library `call` names, from `call: {billing: quote}` or `call(billing.quote, ...)`, and is
 * `null` for a flow.
 * `kind: call` is a single application; `kind: map` with `over: ${list}` (`MapDescriptor`)
 * applies it to each element, with `${item}` in element scope. `parallel:` is kept but inert:
 * run-level workers control concurrency.
 *
 * `allowed` adds `call`, `inputs` and `asserts` to the commons, and `call` is required.
 * `bindingFields` is `inputs` and `over`: the call-argument bindings and the map's iteration
 * source (a plain `call` leaves `over` null, so the kind-blind walk skips it). `over` and
 * `parallel` live here, on the shared class; only `map` lists them in its `allowed`.
 *
 * `childLabel` is compiler-owned diagnostic text for an anonymous inline child; it is not in
 * `allowed`, so never authored.
 */
export class CallDescriptor extends NodeDescriptor {
  // Typed `: string` so `MapDescriptor` can override it with `"map"`.
  static override readonly kind: string = "call";
  static override readonly allowed = this.allow("call", "inputs", "asserts");
  static override readonly required = this.require("call");
  static override readonly bindingFields: readonly string[] = ["inputs", "over"];

  /** The callable's name: a `defs:` entry or a `uses:` alias. */
  readonly call: string;
  /** The def of the library `call` names; `null` when `call` names a flow. */
  readonly def: string | null;
  /** The diagnostic label of an anonymous inline child; `null` on an authored node. */
  readonly childLabel: string | null;
  /**
   * The map's iteration source as YAML read it: a reference, a `call(...)` directive or a list
   * literal; `null` on a plain `call`.
   */
  readonly over: unknown;
  /** The authored `parallel:` as YAML read it; inert. */
  readonly parallel: unknown;
  /** The node asserts. */
  readonly asserts: readonly string[];

  /**
   * Create a descriptor from `fields`.
   *
   * @throws TypeError when `origin` isn't an `OriginRef` or `null`.
   */
  constructor(fields: CallDescriptorFields) {
    super(fields);
    this.call = fields.call;
    this.def = fields.def ?? null;
    this.childLabel = fields.childLabel ?? null;
    this.over = fields.over ?? null;
    // Not `?? false`: an authored `parallel: null` stays null.
    this.parallel = fields.parallel === undefined ? false : fields.parallel;
    this.asserts = fields.asserts ?? [];
  }

  /**
   * Read `call:` (`readCallTarget`), then run the base checks.
   *
   * @throws LoadError at `call:` (else at the node).
   */
  static override _validateFields(
    nodeId: string,
    fields: Record<string, unknown>,
    ranges: Map<string, SourceRange>,
    ownRange: SourceRange | null,
    diagnostics: LoadDiagnostics,
    cursor: SourceCursor,
  ): void {
    readCallTarget(this.kind, nodeId, fields, ranges.get(siteKey("field", "call")) ?? ownRange, diagnostics);
    super._validateFields(nodeId, fields, ranges, ownRange, diagnostics, cursor);
  }

  /**
   * Build the runtime `CallNode` (a single application) or `MapNode` (`kind: map` with `over:`)
   * and its flow-owned wiring, resolving the callable and baking its compiled flow onto the node.
   *
   * The node carries the authored `input:` names as `params`; the wiring holds their sources. A
   * MAP's `over:` iteration source rides the wiring first, under the reserved `over` key. The
   * output contract comes from the child's `ChildSignature`: a CALL re-exports the child's
   * output; a MAP wraps an exact output in `list(U)` keeping only its default origin, and leaves
   * a partial or unknown one unset. The bindings are then checked by name and arity against the
   * child's parameters; every problem found is reported together, at the node.
   *
   * @throws LoadError when a MAP's `input:` names `over`, the callable can't be resolved or
   *   nests too deep, a MAP's child declares no output or an inbox input, or a binding names no
   *   child input or a required child input is unbound.
   */
  override build(ctx: BuildContext): BuildResult {
    // `MapDescriptor` alone sets `kind` to "map"; checking it spares `call.ts` importing `map.ts`,
    // whose class extends this one (the import cycle would throw when `call.ts` loads first).
    const isMap = this.kind === "map";
    if (isMap && Object.hasOwn(this.inputs, "over")) {
      // A MAP's `over` is the iteration source, under the reserved wiring key; a parameter
      // named `over` would collide with it.
      throw ctx.diagnostics.error(
        `map node ${reprText(this.id)}: input name 'over' is reserved (the iteration source); ` +
          "rename the `input:` entry",
        { range: this.fieldRanges.get(siteKey("input", "over")) ?? this.range },
      );
    }
    const child = resolveChild(this.id, this.call, ctx.resolver, this.childLabel, {
      diagnostics: ctx.diagnostics,
      range: this.fieldRanges.get(siteKey("field", "call")) ?? this.range,
      kind: this.kind,
      def: this.def,
      // A lifted `call(...)` names its site, and its fix is written as a directive.
      label: this.fieldLabels.get("call") ?? null,
      directive: this.callSite !== null,
    });
    const display = callableDisplay(this.call, this.childLabel, this.def);
    const sig = childSignature(child);
    const errors: string[] = [];
    let node: CallNode | MapNode;
    let wiring: Map<string, unknown>;
    if (isMap) {
      node = new MapNode(this.id, {
        flowId: this.call,
        def: this.def,
        child: child.compiled,
        childInputs: sig.inputs,
        // Inert: only an authored `true` reads as parallel.
        parallel: this.parallel === true,
        title: this.nodeName,
      });
      // `over` first: the wiring order is the edge order.
      wiring = new Map([["over", this.over], ...this.sinkWiring()]);
      // A def the body calls can get an inbox only from the body, so the body's own inputs
      // cover its whole subtree.
      for (const input of sig.inputs) {
        if (!isInboxType(input.contract?.type)) continue;
        errors.push(
          `map node ${reprText(this.id)}: ${display} declares inbox input ${reprText(input.name)}; a map body ` +
            "can't read an inbox (per-instance inboxes aren't supported yet)",
        );
      }
      if (sig.outputNames.length === 0) {
        errors.push(
          `map node ${reprText(this.id)}: ${display} must declare a ` +
            "non-empty `output:`; a mapped call (over:) returns a list of the child's results",
        );
      } else if (isExactly(sig.output, ValueContract)) {
        node.outputContract = new ValueContract(
          listOf(sig.output.type),
          new ContractOrigins({ default: sig.output.origins.default }),
        );
      }
    } else {
      node = new CallNode(this.id, {
        flowId: this.call,
        def: this.def,
        child: child.compiled,
        childInputs: sig.inputs,
        title: this.nodeName,
      });
      wiring = this.sinkWiring();
      node.outputContract = sig.output;
    }
    node.params = this.sinkParams();
    checkChildBindings(this.kind, this.id, display, node.params, sig, errors, this.fieldLabels);
    if (errors.length > 0) throw ctx.diagnostics.error(errors.join("\n  "), { range: this.range });
    return this._finish(node, wiring);
  }
}

/**
 * Why the map or loop `nodeId` can't pass inbox `argument` to its tool `tool`, other than `take`:
 * only `take` reads an inbox.
 */
export function toolInboxError(kind: "map" | "loop", nodeId: string, tool: string, argument: string): string {
  const why = kind === "map" ? ", so take before the map" : " (a loop passes every carried field to its tool)";
  return (
    `${kind} node ${reprText(nodeId)}: passes inbox ${reprText(argument)} to tool \`${tool}\`; only \`take\` ` +
    `reads an inbox${why}`
  );
}

/** Return the type `list(element)`, a MAP's output, stored as `resolveType` stores a list. */
function listOf(element: Type): Type {
  return new Type({ kind: listKindForElement(element), element });
}

/**
 * Check a CALL's or MAP's parameters by name and arity against the child's signature, appending
 * one message per problem to `errors`: a parameter the child doesn't declare, and a required
 * child input or a memory input left unbound. The binding types are checked later, over the flow's wiring.
 *
 * @param kind The node's kind (`call` or `map`), named in messages.
 * @param display How messages name the callable (`callableDisplay`).
 * @param labels The descriptor's `fieldLabels`: an inline call's `input '<p>'` entry names its
 *   argument and its `call` entry names the call, in place of the generated node.
 */
function checkChildBindings(
  kind: string,
  nodeId: string,
  display: string,
  params: readonly Parameter[],
  sig: ChildSignature,
  errors: string[],
  labels: ReadonlyMap<string, string>,
): void {
  const declared = new Set(sig.inputs.map((input) => input.name));
  const bound = new Set<string>();
  for (const param of params) {
    bound.add(param.name);
    if (!declared.has(param.name)) {
      const subject =
        labels.get(`input ${reprText(param.name)}`) ||
        `${kind} node ${reprText(nodeId)}: binding ${reprText(param.name)}`;
      errors.push(`${subject} is not a declared input of ${display}`);
    }
  }
  const callLabel = labels.get("call") || `${kind} node ${reprText(nodeId)}`;
  for (const input of sig.inputs) {
    if (bound.has(input.name)) continue;
    if (isMemoryType(input.contract?.type)) {
      // Even an Optional one: the child would see null where its caller has a memory.
      errors.push(
        `${callLabel}: the runner binds only the top-level flow's memories; pass \`${input.name}\` from the caller`,
      );
    } else if (input.required) {
      errors.push(`${callLabel}: required callable input ${reprText(input.name)} is unbound`);
    }
  }
}
