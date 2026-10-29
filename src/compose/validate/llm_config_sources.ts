/**
 * Graph-stage: the sources of `llm_config` values must be typed and fit their fields.
 *
 * A value in an AGENT's own `llm_config`, in an adaptive composer's, or in the flow's
 * `llm_config` layer is checked where it is written, against the flow's contracts as written (a
 * config sees no CASE narrowing):
 *
 * - the source needs a proven type or a tag; an unproven or untagged `Any` source is rejected
 *   with "tag it", and a tagged source must be able to fit its tag;
 * - `provider` and `model` take `str`, and a `Literal` provider needs at least one known
 *   provider among its members;
 * - a provider-block field whose annotation is `str`, `int`, `float` or `bool` (with its
 *   inclusive bounds) or a `Literal` takes that type; any other block field is checked when the
 *   value resolves;
 * - a whole record must be a record of `llm_config` fields other than the ones only the host
 *   sets and `inherit`, or `object`; a whole block a record of that block's fields, or `object`;
 * - no `isinstance` or constructor call.
 *
 * Only a disjoint type is a load error; a narrower one is checked when the value resolves. The
 * flow layer is checked once, so a value every agent overrides is still checked, its references
 * included.
 *
 * Knows about: `compile/validation` (the type relation and reference paths), `compose/helpers`
 * (constructor names, tag labels, load diagnostics, output contracts), `compose/nodes/base` (build
 * results), `compose/validate/typed_bindings` (binding type inference), `llm_clients`, `nodes`,
 * `typesys`, `expr` and `diagnostics`.
 * Never imports: the loader or the compose package facade; only the pipeline imports this.
 */

import { classifyPath, typeRelation } from "../../compile/validation.ts";
import { compareCodeUnits, type SourceRange } from "../../diagnostics.ts";
import { ExpressionError } from "../../expr/index.ts";
import { exprRefsOf } from "../../expr/template.ts";
import { LLM_CONFIG_KEYS, SUPPORTED_LLM_PROVIDERS } from "../../llm_clients/config.ts";
import { blockFieldAnnotation, type BlockFieldType, PROVIDER_BLOCKS } from "../../llm_clients/provider_blocks.ts";
import { declaredHostOnlyError, valueLeaves } from "../../nodes/agent/llm_values.ts";
import { AgentNode } from "../../nodes/agent/node.ts";
import { NEW_MESSAGES } from "../../typesys/pool.ts";
import {
  Binding,
  iterSourceStrings,
  type OutputContract,
  type ValueContract,
  type ValuePath,
  ValueTemplate,
} from "../../nodes/contracts.ts";
import type { TypeRegistry } from "../../typesys/types.ts";
import { isRecord, Type, ValueKind } from "../../typesys/values.ts";
import { LoadDiagnostics, type LoadError } from "../helpers/errors.ts";
import { collectConstructorNames, collectIsinstanceNames } from "../helpers/isinstance_types.ts";
import type { TypeFn } from "../../compile/output_contracts.ts";
import { namedTypeLabel } from "../helpers/tags.ts";
import { type BuildResult, siteKey } from "../nodes/base.ts";
import { inferBindingType, type ResolveOrigin, typedValueError } from "./typed_bindings.ts";

const TEXT = Type.scalar(ValueKind.STRING);
const KINDS: Readonly<Record<BlockFieldType["base"], ValueKind>> = {
  str: ValueKind.STRING,
  int: ValueKind.INTEGER,
  float: ValueKind.NUMBER,
  bool: ValueKind.BOOLEAN,
};

/** Builds the located error for one value's message. */
type Fail = (message: string) => LoadError;

/**
 * Reject an `llm_config` value whose source is untyped or can never fit its field.
 *
 * @param buildResults Built nodes with their transient ranges and labels; each AGENT's own layer
 *   is checked.
 * @param flowLlmConfig The flow's `llm_config:` layer: a record, a `ValueTemplate` or a `Binding`.
 * @param flowLabel Names the flow layer in a message, such as `"flow llm_config"`.
 * @param producers Node output contracts by producer id.
 * @param flowInputs Flow input contracts by name.
 * @param options.typeFn Resolves a constructor name to its type for inference.
 * @param options.registry The flow's typedefs, so a message names a declared record by its
 *   typedef.
 * @param options.resolveOrigin Resolves a value's registered origin to its source frame, the
 *   preferred location.
 * @param options.diagnostics Factory for located load errors.
 * @throws LoadError at the first value that reads an unknown name, calls `isinstance` or a
 *   constructor, has no proven type, or has a type disjoint from its field.
 */
export function checkLlmConfigSources(
  buildResults: ReadonlyMap<string, BuildResult>,
  flowLlmConfig: unknown,
  flowLabel: string,
  producers: ReadonlyMap<string, OutputContract>,
  flowInputs: ReadonlyMap<string, ValueContract>,
  options: {
    typeFn?: TypeFn | null;
    registry?: TypeRegistry | null;
    resolveOrigin?: ResolveOrigin | null;
    diagnostics?: LoadDiagnostics | null;
  } = {},
): void {
  const checker = new Checker(
    producers,
    flowInputs,
    new Set(buildResults.keys()),
    options.typeFn ?? null,
    options.registry ?? new Map(),
    options.resolveOrigin ?? null,
    options.diagnostics ?? new LoadDiagnostics(),
  );
  for (const [nodeId, result] of buildResults) {
    const node = result.node;
    if (node instanceof AgentNode) {
      checker.layer(
        node.ownLlmConfig,
        result.siteName(nodeId, "llm_config"),
        result.fieldRanges.get(siteKey("field", "llm_config")) ?? result.nodeRange,
        // The reference pass checked the carried values.
        { checkRefs: false },
      );
    }
  }
  checker.layer(flowLlmConfig, flowLabel, null, { checkRefs: true });
}

/** Checks the values of one layer at a time against one flow's contracts. */
class Checker {
  readonly producers: ReadonlyMap<string, OutputContract>;
  readonly flowInputs: ReadonlyMap<string, ValueContract>;
  readonly nodeIds: ReadonlySet<string>;
  readonly typeFn: TypeFn | null;
  readonly registry: TypeRegistry;
  readonly resolveOrigin: ResolveOrigin | null;
  readonly diagnostics: LoadDiagnostics;

  constructor(
    producers: ReadonlyMap<string, OutputContract>,
    flowInputs: ReadonlyMap<string, ValueContract>,
    nodeIds: ReadonlySet<string>,
    typeFn: TypeFn | null,
    registry: TypeRegistry,
    resolveOrigin: ResolveOrigin | null,
    diagnostics: LoadDiagnostics,
  ) {
    this.producers = producers;
    this.flowInputs = flowInputs;
    this.nodeIds = nodeIds;
    this.typeFn = typeFn;
    this.registry = registry;
    this.resolveOrigin = resolveOrigin;
    this.diagnostics = diagnostics;
  }

  /**
   * Check every value in `layer`, naming it `label` in messages.
   *
   * @throws LoadError as `checkLlmConfigSources` describes.
   */
  layer(layer: unknown, label: string, fallback: SourceRange | null, options: { checkRefs: boolean }): void {
    const tree = layer instanceof ValueTemplate ? layer.value : {};
    for (const [path, leaf] of valueLeaves(layer)) {
      const where = label + path.map((part) => `.${part}`).join("");
      const origin =
        layer instanceof Binding
          ? layer.origin
          : leaf instanceof Binding && leaf.origin !== null
            ? leaf.origin
            : layer instanceof ValueTemplate
              ? layer.originAt(path)
              : null;
      const frame = this.resolveOrigin !== null && origin !== null ? this.resolveOrigin(origin) : null;
      const fail: Fail = (message) =>
        frame !== null
          ? this.diagnostics.error(`${where}: ${message}`, { frames: [frame] })
          : this.diagnostics.error(`${where}: ${message}`, { range: fallback });
      this.names(leaf, options.checkRefs, fail);
      this.tag(path, leaf, fail);
      this.type(path, leaf, tree, fail);
    }
  }

  /** Reject an unknown reference (when asked), `isinstance` or a constructor. */
  private names(leaf: unknown, checkRefs: boolean, fail: Fail): void {
    let isinstanceNames: string[];
    let constructors: string[];
    let refs: string[];
    try {
      isinstanceNames = collectIsinstanceNames(leaf);
      constructors = collectConstructorNames(leaf);
      refs = [...iterSourceStrings(leaf)].flatMap(([, text]) => exprRefsOf(text));
    } catch (error) {
      if (error instanceof ExpressionError) throw fail(error.message);
      throw error;
    }
    if (isinstanceNames.length > 0 || constructors.length > 0) {
      const used = isinstanceNames.length > 0 ? "isinstance" : `${constructors[0]}(...)`;
      throw fail(
        `${used} is not supported in an llm_config value; compute the value in a ` +
          "CODE node and read its output",
      );
    }
    const inputNames = new Set(this.flowInputs.keys());
    for (const ref of refs) {
      // The reference pass accepts an AGENT's new messages for the node's other reads, so a
      // node layer refuses them here too: its node-id targets have no `<id>.new_messages`.
      if (!checkRefs && ref.split(".")[1] !== NEW_MESSAGES) continue;
      const problem = classifyPath(ref, this.nodeIds, inputNames);
      if (problem !== null) throw fail(problem);
    }
  }

  /** Reject a tagged value whose source can never fit its tag. */
  private tag(path: ValuePath, leaf: unknown, fail: Fail): void {
    const message = typedValueError(leaf, this.producers, this.flowInputs, {
      typeFn: this.typeFn,
      registry: this.registry,
      slot: "llm_config" + path.map((part) => `.${part}`).join(""),
    });
    if (message !== null) throw fail(message);
  }

  /** Reject a value with no proven type, or one disjoint from its field. */
  private type(path: ValuePath, leaf: unknown, tree: unknown, fail: Fail): void {
    const inferred = inferBindingType(leaf, this.producers, this.flowInputs, { typeFn: this.typeFn });
    const tagged = leaf instanceof Binding && leaf.contract !== null;
    if (inferred === null || (inferred.kind === ValueKind.ANY && !tagged)) {
      throw fail("the value's type isn't known; tag it with the type it holds, e.g. `model: !str ${...}`");
    }
    if (inferred.kind === ValueKind.NONE) return; // null means unset
    const label = namedTypeLabel(inferred, this.registry);
    const head = path[0];
    const block = typeof head === "string" && Object.hasOwn(PROVIDER_BLOCKS, head) ? head : null;
    if (path.length === 0) {
      this.record(inferred, LLM_CONFIG_KEYS, "llm_config", label, fail);
    } else if (block !== null && path.length === 1) {
      const keys = new Set(Object.keys(PROVIDER_BLOCKS[block]!.fields));
      this.record(inferred, keys, `the ${block} block`, label, fail);
    } else if (block !== null) {
      const blockTree = isRecord(tree) ? tree[block] : undefined;
      const field = blockFieldAnnotation(block, path.slice(1).map(String), blockTree);
      if (field !== null) this.fits(inferred, blockFieldType(field), label, fail);
    } else if (path.length === 1 && head === "provider") {
      this.fits(inferred, TEXT, label, fail);
      const members = [...(inferred.tags ?? [])].sort(compareCodeUnits);
      const known: ReadonlySet<string> = new Set(SUPPORTED_LLM_PROVIDERS);
      if (members.length > 0 && !members.some((member) => known.has(member))) {
        throw fail(
          `none of ${members.join(", ")} is a known provider; known providers ` +
            `are ${SUPPORTED_LLM_PROVIDERS.join(", ")}`,
        );
      }
    } else if (path.length === 1 && head === "model") {
      this.fits(inferred, TEXT, label, fail);
    }
  }

  /** Reject `inferred` when no non-null value of it fits `declared`. */
  private fits(inferred: Type, declared: Type, label: string, fail: Fail): void {
    if (inferred.kind === ValueKind.ANY || typeRelation(inferred, declared) !== "disjoint") return;
    throw fail(`must be ${namedTypeLabel(declared, this.registry)}, but the value is ${label}`);
  }

  /** Reject a whole record or block whose type isn't a record of `keys` or `object`. */
  private record(inferred: Type, keys: ReadonlySet<string>, owner: string, label: string, fail: Fail): void {
    if (inferred.kind === ValueKind.ANY) return;
    if (inferred.kind !== ValueKind.OBJECT) {
      throw fail(`must be a record of ${owner}'s fields, but the value is ${label}`);
    }
    const declared = [...(inferred.fields?.keys() ?? [])];
    for (const member of inferred.variants ?? []) {
      for (const name of member.fields?.keys() ?? []) if (!declared.includes(name)) declared.push(name);
    }
    const hostOnly = owner === "llm_config" ? declaredHostOnlyError(declared) : null;
    if (hostOnly !== null) throw fail(hostOnly);
    for (const name of declared) {
      if (name === "inherit" && owner === "llm_config") {
        throw fail(
          "its type declares `inherit`, which decides which layers apply before " +
            "any value is known; write it as true or false in the agent's own " +
            "llm_config",
        );
      }
      if (!keys.has(name)) throw fail(`its type declares \`${name}\`, which is not a field of ${owner}`);
    }
  }
}

/** Return the type a value for a simple provider-block field must overlap. */
function blockFieldType(field: BlockFieldType): Type {
  const kind = KINDS[field.base];
  if (field.literals.length > 0 && field.base === "str") return new Type({ kind, tags: new Set(field.literals) });
  // The bounds already have their kind's number type: `bigint` for an int field.
  const bounded = (kind === ValueKind.INTEGER || kind === ValueKind.NUMBER) && (field.ge !== null || field.le !== null);
  return new Type({ kind, bounds: bounded ? [field.ge, field.le] : undefined });
}
