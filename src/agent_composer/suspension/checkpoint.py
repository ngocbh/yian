"""RunCheckpoint — the serializable envelope that makes a run resumable.

The graph *topology* is intentionally NOT stored (it is rebuilt from the flow
spec), keeping checkpoints small and decoupled from code changes to node bodies.
Only run state is captured:

- `pool`            — the typed variable pool (lossless via AnyValue tags)
- `node_state` / `edge_state` — per-node/edge scheduling state
- `paused_nodes`    — parked leaves; stay TAKEN, the stored answer is delivered as Output on resume (no re-run)
- `deferred_nodes`  — became ready while suspending; enqueue on resume
- `pause_reasons`   — what each paused node awaits (drives the watcher/CLI)
- `expansions`      — descriptor tree for runtime-grown subgraphs (replayed on restore)
- `num_workers`     — the producer/consumer drive mode the run used (0 = serial, >=1 = pool size)

A round-trip through `dumps()`/`loads()` reconstructs an equivalent run in any
process.
"""

import json

from pydantic import BaseModel, Field

from agent_composer._state import NodeState
from agent_composer.suspension.expansions import GrowRecord
from agent_composer.suspension.pause import PauseReason
from agent_composer.typesys.pool import VariablePool

# Checkpoint blob schema version. 1.0 -> 2.0: the single-value store migration
# (pool.store collapsed node_id->key->TypedValue to node_id->TypedValue). 2.0 -> 3.0: the
# inputs-namespace retirement: `pool.inputs` is gone — `${input.X}`
# resolves via `store[START_ID]` — so the serialized pool shape changed.
# 3.0 -> 4.0: the type-surface rename (the `ValueKind.value` tags moved to the one
# Python-surface vocabulary, `str`/`int`/`float`/`bool`/..., so a pre-4.0 blob's
# `kind` tags are unreadable by the AnyValue discriminated union). A pre-4.0
# blob is NOT loadable. 4.0 -> 5.0: adds the additive `expansions` field (the
# descriptor tree for runtime-grown subgraphs; defaults []). A 4.0 body is otherwise
# wire-compatible — `pause_reasons` is unchanged — but pre-5.0 blobs are rejected as a
# forward-compat hard cutover: this is the first build to write/expect the field, and a
# pre-5.0 checkpoint of a grown run carries no descriptors for the (forthcoming) replay.
# Pre-5.0 blobs are NOT loadable. 5.0 -> 6.0: adds the additive `num_workers` field (the
# producer/consumer drive mode the run used); pre-6.0 blobs are NOT loadable (hard cutover,
# consistent with prior bumps). 6.0 -> 7.0: the expansion ledger unified from the four-variant
# `*Expansion` union (CallExpansion/MapExpansion/AgentExpansion/LoopExpansion) to a single flat
# `GrowRecord(spawner_id, seed, children)` — one uniform record per grow, kind-blind replay.
# Pre-7.0 blobs are NOT loadable (hard cutover). 8.0: MAP grows a synthetic `map#/__start__`
# fan-out node, so a pre-8.0 checkpoint would replay a topology missing that node — hard cutover.
CHECKPOINT_VERSION = "8.0"


class RunCheckpoint(BaseModel):
    """
    The serializable envelope that makes a paused run resumable.

    The graph topology is intentionally NOT stored — it is rebuilt from the flow spec —
    so checkpoints stay small and decoupled from code changes to node bodies. Only run
    state is captured. A round-trip through `dumps()` / `loads()` reconstructs an
    equivalent run in any process.

    Attributes:
        version (`str`, *optional*, defaults to `CHECKPOINT_VERSION`):
            The blob schema version. `loads()` rejects any other version up front.
        pool (`VariablePool`):
            The typed variable pool (lossless via the `AnyValue` tags).
        ready (`list[str]`, *optional*, defaults to `[]`):
            Node ids that were ready to run at suspend time; re-enqueued on resume.
        node_state (`dict[str, NodeState]`, *optional*, defaults to `{}`):
            Per-node scheduling state at suspend time.
        edge_state (`dict[str, NodeState]`, *optional*, defaults to `{}`):
            Per-edge scheduling state at suspend time.
        paused_nodes (`list[str]`, *optional*, defaults to `[]`):
            Parked leaves; they stay `TAKEN` and the delivered answer becomes their
            Output on resume (no re-run).
        deferred_nodes (`list[str]`, *optional*, defaults to `[]`):
            Nodes that became ready while suspending; enqueued on resume.
        pause_reasons (`list[PauseReason]`, *optional*, defaults to `[]`):
            What each paused node awaits — drives the watcher / CLI.
        expansions (`list[GrowRecord]`, *optional*, defaults to `[]`):
            Ledger of runtime-grown subgraphs, replayed top-down on restore. One flat
            `GrowRecord` per grow (nested grows ride under their parent's `children`).
            Empty for any run that never expanded.
        num_workers (`int`, *optional*, defaults to `0`):
            The producer/consumer drive mode the run used: 0 = single-threaded inline
            drain, >=1 = worker-pool size. Lets a checkpointed run resume under the same
            (or a chosen) drive mode.
    """

    version: str = CHECKPOINT_VERSION

    pool: VariablePool
    ready: list[str] = Field(default_factory=list)
    node_state: dict[str, NodeState] = Field(default_factory=dict)
    edge_state: dict[str, NodeState] = Field(default_factory=dict)
    paused_nodes: list[str] = Field(default_factory=list)
    deferred_nodes: list[str] = Field(default_factory=list)
    pause_reasons: list[PauseReason] = Field(default_factory=list)

    # Ledger of runtime-grown subgraphs; `snapshot()` populates it from `engine.expansions`
    # and `restore()` replays it top-down (`_replay_expansions`) to re-grow the cloned subgraphs
    # before resume. One flat `GrowRecord` per grow (nested grows ride under their parent's
    # `children`). Empty for any run that never expanded (pure static flows).
    expansions: list[GrowRecord] = Field(default_factory=list)

    # The producer/consumer drive mode the run used: 0 = single-threaded inline drain,
    # >=1 = worker-pool size. snapshot() captures engine.num_workers; restore() rebuilds
    # the engine at this count (overridable) so a checkpointed run resumes under the same
    # (or a chosen) drive mode. Defaults 0 (serial) for a checkpoint built without it.
    num_workers: int = 0

    def dumps(self) -> str:
        return self.model_dump_json()

    @classmethod
    def loads(cls, blob: str) -> "RunCheckpoint":
        # Gate on the version from the RAW JSON BEFORE model validation: a 1.0 blob's
        # old nested-store shape would otherwise fail `extra="forbid"`/type validation
        # with an opaque pydantic error instead of this clear message.
        try:
            raw_version = json.loads(blob).get("version")
        except (ValueError, AttributeError):
            raw_version = None
        if raw_version != CHECKPOINT_VERSION:
            raise ValueError(
                f"incompatible checkpoint version {raw_version!r}; this build reads "
                f"{CHECKPOINT_VERSION!r} (unified GrowRecord expansion ledger)"
            )
        return cls.model_validate_json(blob)
