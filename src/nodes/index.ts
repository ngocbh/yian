/**
 * The pure node contract, as Python's `yian.nodes`.
 *
 * Nodes consume bound input records and return the closed `Output | Route | Grow` result sum.
 * Spawners return only a canonical `Grow` seed (plus ids to prune); they never build compiled graph
 * fragments themselves. Frozen semantic carriers enumerate and map only their declared origin
 * slots. A node owns hooks for its local origins and static children; higher layers may traverse
 * them without kind dispatch, while the runtime does not interpret this provenance. The base
 * contract also owns detached backward context and feedback results; execution scopes carry
 * authored locations independently of memory declarations.
 *
 * `base.ts` owns the `Node` base class and its results, `contracts.ts` the semantic carriers,
 * `binding.ts` the binder, and `ids.ts` the deterministic child ids. This index re-exports
 * Python's `__all__` and nothing else: no node kind, none of the binder and none of the ids; their
 * callers import the module that defines them.
 *
 * Knows about: diagnostics, expr, memory contracts, typesys, and peer node modules.
 * Never imports: compile, compose, runtime, the package-root run boundary, or the CLI.
 */

export type { NodeResult } from "./base.ts";
export {
  BackwardContext,
  BackwardResult,
  Grow,
  Node,
  NodeKind,
  Output,
  Route,
  validateBackwardResult,
} from "./base.ts";
export type { OutputContract, ValuePath } from "./contracts.ts";
export {
  Assertion,
  asBinding,
  Binding,
  bindingSource,
  ContractOrigins,
  ExecutionPosition,
  ExecutionScope,
  Parameter,
  PartialRecordContract,
  ValueContract,
} from "./contracts.ts";
