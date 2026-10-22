/**
 * kind=code: the deterministic-code node's descriptor.
 *
 * Owns the `code` node's `CodeDescriptor`, which inherits the template `parse` from
 * `NodeDescriptor` (which records each field's `SourceRange`, the `code:` body's included),
 * checks its own fields, attaches the source origins of the `code:` scalar and of its exact
 * literal lines without keeping YAML objects, and builds the runtime `CodeNode`. Nothing is
 * compiled at build: the node takes the body as text, or a value entry's built-in identity
 * form. The awaited load step runs after assembly: `checkFlowCode` checks every CODE node of a
 * composed flow and its child flows, and `checkCodeNode` turns a failed check into a located
 * `LoadError`. Registered in `KINDS` and `BINDING_FIELDS` by `compose/nodes/index.ts`.
 *
 * Knows about: diagnostics, platform, compose source and helpers, and nodes (base, code).
 * Never imports: compose/parse, the runtime or the CLI.
 */

import { type OriginRef, reprText, type SourceRange } from "../../diagnostics.ts";
import type { Node } from "../../nodes/base.ts";
import { type BuiltinCode, CodeNode, selectInlineCodeOrigin, validateCodeOrigins } from "../../nodes/code/node.ts";
import { TypeStripError } from "../../platform.ts";
import type { LoadDiagnostics } from "../helpers/errors.ts";
import type { SourceCatalogBuilder } from "../source/catalog.ts";
import type { SourceCursor } from "../source/index.ts";
import {
  type BuildContext,
  type BuildResult,
  NodeDescriptor,
  type NodeDescriptorFields,
  type OriginPresenter,
  rejectNonText,
  siteKey,
} from "./base.ts";

/** The fields `CodeDescriptor`'s constructor takes: the common ones and the CODE's own. */
export interface CodeDescriptorFields extends NodeDescriptorFields {
  /** Text, or the built-in form only a value entry's reader passes. */
  code: string | BuiltinCode;
  asserts?: readonly string[] | null;
  codeOrigin?: OriginRef | null;
  codeLineOrigins?: readonly OriginRef[];
}

/**
 * kind=code: deterministic code, a `./file.ts:fn` reference or an inline body.
 *
 * `allowed` adds `code`, `inputs`/`outputs` and `asserts` to the commons; `code` is required. The
 * `code:` body is source text, not a `${...}` binding surface, so only `inputs` is a binding
 * field. `codeOrigin` and `codeLineOrigins` are diagnostic-only: the authored identities of the
 * whole `code:` scalar and of its exact constructed lines.
 */
export class CodeDescriptor extends NodeDescriptor {
  static override readonly kind = "code";
  static override readonly allowed = this.allow("code", "inputs", "outputs", "asserts");
  static override readonly required = this.require("code");
  static override readonly bindingFields: readonly string[] = ["inputs"];

  /**
   * The `code:` field: a reference or an inline body, or the built-in identity a value entry
   * lowers to (an authored `code:` is always text).
   */
  readonly code: string | BuiltinCode;
  /** The node asserts. */
  readonly asserts: readonly string[];
  /** The authored identity of the whole `code:` scalar, or `null`. */
  readonly codeOrigin: OriginRef | null;
  /** The authored identities of the exact constructed body lines, in order; empty when unknown. */
  readonly codeLineOrigins: readonly OriginRef[];

  /**
   * Create a descriptor from `fields`; `asserts` reads `null` as empty, the code origins default
   * to none.
   *
   * @throws TypeError when `origin` or `codeOrigin` isn't an `OriginRef` or `null`, or
   *   `codeLineOrigins` isn't an array of `OriginRef`s.
   */
  constructor(fields: CodeDescriptorFields) {
    super(fields);
    const { codeOrigin = null, codeLineOrigins = [] } = fields;
    validateCodeOrigins(codeOrigin, codeLineOrigins);
    this.code = fields.code;
    this.asserts = fields.asserts ?? [];
    this.codeOrigin = codeOrigin;
    this.codeLineOrigins = codeLineOrigins;
  }

  /**
   * Reject a non-text `code:` (`null` included), then run the base checks.
   *
   * @throws LoadError at the field for a bad shape.
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
      "code",
      fields["code"] ?? null,
      ranges.get(siteKey("field", "code")) ?? ownRange,
      diagnostics,
      { required: true },
    );
    super._validateFields(nodeId, fields, ranges, ownRange, diagnostics);
  }

  /**
   * Return a copy carrying the origins of the whole `code:` scalar and of its exact literal lines.
   *
   * A folded, quoted, or value-changing projected scalar has only a whole-value site, and a
   * source-free one has none. Literal lines are registered in constructed-value order, not by
   * physical position. A slot with no registrable site keeps the origin the descriptor carries.
   *
   * @param cursor The normalized authored CODE node occurrence.
   * @param options.sitePath The node occurrence's complete semantic path.
   * @param options.catalogBuilder The file's registry that mints source-neutral origins.
   */
  override withSourceOrigins(
    cursor: SourceCursor,
    options: { sitePath: readonly (string | number)[]; catalogBuilder: SourceCatalogBuilder },
  ): this {
    const { sitePath, catalogBuilder } = options;
    const codeCursor = cursor.child("code");
    const fieldPath = [...sitePath, "field", "code"];
    const codeOrigin = catalogBuilder.register(fieldPath, codeCursor) ?? this.codeOrigin;

    let codeLineOrigins = this.codeLineOrigins;
    const lineRanges = codeCursor.scalarSource?.constructedLineRanges ?? null;
    if (lineRanges !== null) {
      const registered: OriginRef[] = [];
      for (let index = 0; index < lineRanges.length; index++) {
        const origin = catalogBuilder.registerConstructedLine([...fieldPath, "line", index], codeCursor, index);
        if (origin === null) break;
        registered.push(origin);
      }
      // A partial set would claim exact lines it can't back, so only a complete set replaces them.
      if (registered.length === lineRanges.length) codeLineOrigins = registered;
    }
    return this.replace({ codeOrigin, codeLineOrigins });
  }

  /**
   * Build the runtime `CodeNode` and its flow-owned wiring.
   *
   * The node takes the body as text, or a value entry's built-in identity form, `ctx.baseDir` as
   * the directory a reference resolves against, `ctx.sourceFile` as the label a failing body's
   * stack frames are named after, and the descriptor's code origins unchanged;
   * nothing is compiled. Sets the node's `params`
   * from the authored `input:` and its output contract from `output:`; `_finish` stamps the
   * descriptor origin.
   *
   * @throws LoadError when `output:` isn't a valid type expression.
   */
  override build(ctx: BuildContext): BuildResult {
    const node = new CodeNode(this.id, {
      code: this.code,
      baseDir: ctx.baseDir,
      sourceFile: ctx.sourceFile,
      codeOrigin: this.codeOrigin,
      codeLineOrigins: this.codeLineOrigins,
      title: this.nodeName,
    });
    node.outputContract = this.outputContract(ctx);
    node.params = this.sinkParams();
    return this._finish(node, this.sinkWiring());
  }
}

/** What the CODE load check needs: where errors are raised and how an origin is presented. */
export interface CodeCheckOptions {
  /** The context the `LoadError` is raised in. */
  diagnostics: LoadDiagnostics;
  /** Presents an origin as a frame; `null` leaves the error unframed. */
  resolveOrigin: OriginPresenter | null;
  /** Flows already checked, shared across checks of one load; a fresh set when absent. */
  seen?: Set<object>;
  /** Checks one child flow of a node; `visit` checks it in place. `visit` when absent. */
  child?: (flow: CheckedFlow, visit: (flow: CheckedFlow) => Promise<void>) => Promise<void>;
}

/**
 * Run a CODE node's load check (`CodeNode.check`) and turn a failure into a located `LoadError`.
 *
 * A body that doesn't strip is located at its authored line when it has an exact line origin;
 * every other failure at the whole `code:` value, then at the node. The message is
 * `node '<id>': ` plus the failure's own.
 *
 * @throws LoadError when the check fails, its cause the original error.
 */
export async function checkCodeNode(node: CodeNode, options: CodeCheckOptions): Promise<void> {
  const { diagnostics, resolveOrigin } = options;
  try {
    await node.check();
  } catch (error) {
    const origin = selectInlineCodeOrigin(error instanceof TypeStripError ? error.line : null, {
      codeOrigin: node.codeOrigin,
      codeLineOrigins: node.codeLineOrigins,
      nodeOrigin: node.origin,
    });
    const frame = resolveOrigin !== null && origin !== null ? resolveOrigin(origin) : null;
    const message = error instanceof Error ? error.message : String(error);
    throw diagnostics.error(`node ${reprText(node.id)}: ${message}`, {
      frames: frame === null ? [] : [frame],
      cause: error,
    });
  }
}

/** A composed flow as the CODE load check reads it: its nodes, in order. */
export interface CheckedFlow {
  readonly nodes: ReadonlyMap<string, Node>;
}

/**
 * Check every CODE node of a composed flow and of its child flows, in node order.
 *
 * A node's child flows (`originChildren()`) are checked right after it; a flow reached twice is
 * checked once. A flow in `options.seen` was checked already and is skipped. Each child flow goes
 * through `options.child` when given.
 *
 * @throws LoadError at the first CODE node whose check fails.
 */
export async function checkFlowCode(flow: CheckedFlow, options: CodeCheckOptions): Promise<void> {
  // Keyed by identity: a shared child flow holds the same nodes.
  const seen = options.seen ?? new Set<object>();
  const visitChild = options.child ?? ((child, visit) => visit(child));
  const visit = async (current: CheckedFlow): Promise<void> => {
    if (seen.has(current)) return;
    seen.add(current);
    for (const node of current.nodes.values()) {
      if (node instanceof CodeNode) await checkCodeNode(node, options);
      for (const child of node.originChildren()) await visitChild(child as CheckedFlow, visit);
    }
  };
  await visit(flow);
}
