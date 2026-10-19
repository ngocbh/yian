/**
 * kind=call|map: function application of a callable, the `call` node's descriptor.
 *
 * Owns the `call` node's `CallDescriptor`, which inherits the template `parse` from
 * `NodeDescriptor` (which records each field's `SourceRange`) and rejects a `call:` YAML read as
 * non-text. `map.ts`'s `MapDescriptor` subclasses it, adding only the map field surface (`over:`,
 * `parallel:`) and `kind = "map"`; `map` imports `call`, never the reverse, since an ES module
 * cycle between a class and its subclass's module throws when `call.ts` loads first. Registered
 * in `KINDS` and `BINDING_FIELDS` by `compose/nodes/index.ts`.
 *
 * Knows about: diagnostics and compose helpers.
 * Never imports: `map.ts`, compose/parse, the runtime or the CLI.
 */

import type { SourceRange } from "../../diagnostics.ts";
import type { LoadDiagnostics } from "../helpers/errors.ts";
import { NodeDescriptor, type NodeDescriptorFields, rejectNonText, siteKey } from "./base.ts";

/** The fields `CallDescriptor`'s constructor takes: the common ones and the CALL's own. */
export interface CallDescriptorFields extends NodeDescriptorFields {
  call: string;
  childLabel?: string | null;
  callSite?: readonly (string | number | null)[] | null;
  over?: unknown;
  parallel?: unknown;
  asserts?: readonly string[] | null;
}

/**
 * kind=call|map: apply a callable (`call:`) to the call-argument bindings in `input:`.
 *
 * `call` names the callable, resolved first among the flow's `defs:`, else as an external flow
 * id (the loader composes that resolution). `kind: call` is a single application; `kind: map`
 * with `over: ${list}` (`MapDescriptor`) applies it to each element, with `${item}` in element
 * scope. `parallel:` is kept but inert: run-level workers control concurrency.
 *
 * `allowed` adds `call`, `inputs` and `asserts` to the commons, and `call` is required.
 * `bindingFields` is `inputs` and `over`: the call-argument bindings and the map's iteration
 * source (a plain `call` leaves `over` null, so the kind-blind walk skips it). `over` and
 * `parallel` live here, on the shared class; only `map` lists them in its `allowed`.
 *
 * `childLabel` is compiler-owned diagnostic text for an anonymous inline child; it is not in
 * `allowed`, so never authored. `callSite` is compiler-owned too: the address of the authored
 * binding a lifted `call(...)` was written in: `["input", hostId, key]` for a node input (a value
 * entry's key is its id), `["field", hostId, field]` for a MAP `over:`, `["output", key]` for a
 * flow output (`key` is `null` for a single output), `["assert", index]` for a flow `asserts:`
 * item, and `["then", caseId, index]` / `["else", caseId]` for a CASE branch. It is `null` on an
 * authored node.
 */
export class CallDescriptor extends NodeDescriptor {
  // Typed `: string` so `MapDescriptor` can override it with `"map"`.
  static override readonly kind: string = "call";
  static override readonly allowed = this.allow("call", "inputs", "asserts");
  static override readonly required = this.require("call");
  static override readonly bindingFields: readonly string[] = ["inputs", "over"];

  /** The callable's name: a `defs:` entry or an external flow id. */
  readonly call: string;
  /** The diagnostic label of an anonymous inline child; `null` on an authored node. */
  readonly childLabel: string | null;
  /** The address of the binding a lifted `call(...)` was written in; `null` on an authored node. */
  readonly callSite: readonly (string | number | null)[] | null;
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
    this.childLabel = fields.childLabel ?? null;
    this.callSite = fields.callSite ?? null;
    this.over = fields.over ?? null;
    // Not `?? false`: an authored `parallel: null` stays null, as Python passes it through.
    this.parallel = fields.parallel === undefined ? false : fields.parallel;
    this.asserts = fields.asserts ?? [];
  }

  /**
   * Reject a `call:` YAML read as non-text (including `null`), then run the base checks.
   *
   * @throws LoadError at `call:` (else at the node).
   */
  static override _validateFields(
    nodeId: string,
    fields: Record<string, unknown>,
    ranges: Map<string, SourceRange>,
    ownRange: SourceRange | null,
    diagnostics: LoadDiagnostics,
  ): void {
    rejectNonText(
      this.kind,
      nodeId,
      "call",
      fields["call"] ?? null,
      ranges.get(siteKey("field", "call")) ?? ownRange,
      diagnostics,
      { required: true },
    );
    super._validateFields(nodeId, fields, ranges, ownRange, diagnostics);
  }
}
