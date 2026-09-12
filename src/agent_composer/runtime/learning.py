"""Source-neutral reducer projections for durable memory learning.

This module turns reducer-accepted graph outcomes into immutable learning
episodes and owns the deterministic frame and policy state carried by
``RunState``.  It performs no backend work: runtime orchestration may later use
the projected state to call injected memory sessions, but the durable record
history remains the sole authority.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
import json

from agent_composer.diagnostics import Failure
from agent_composer.memory import (
    EvaluationEvidence,
    LearnContext,
    MemoryEpisode,
    MemoryEpisodeStatus,
    MemoryFrameRef,
    MemoryFrameSpec,
    MemoryMutation,
    MemoryOccurrenceRef,
    MemoryOperationFailure,
    MemoryOwner,
)
from agent_composer.runtime.log import (
    NodeExpandedRecord,
    NodeFailedRecord,
    NodeRoutedRecord,
    NodeSucceededRecord,
    canonical_seed,
)


_PRESERVE = object()


class FrameLearningStatus(str, Enum):
    """Closed reducer states for one frame-local learn operation."""

    PENDING = "pending"
    STARTED = "started"
    LEARNED = "learned"
    FAILED_RETRYABLE = "failed_retryable"
    FAILED_FINAL = "failed_final"


class PolicyCommitStatus(str, Enum):
    """Closed reducer states for one policy-local commit operation."""

    UNREQUESTED = "unrequested"
    REQUESTED = "requested"
    SUCCEEDED = "succeeded"
    FAILED_RETRYABLE = "failed_retryable"
    FAILED_FINAL = "failed_final"


@dataclass(frozen=True, slots=True)
class FrameLearningState:
    """Authoritative state of one frame's latest durable learn attempt."""

    status: FrameLearningStatus = FrameLearningStatus.PENDING
    occurrences: tuple[MemoryOccurrenceRef, ...] = ()
    mutations: tuple[MemoryMutation, ...] = ()
    failure: MemoryOperationFailure | None = None

    def __post_init__(self) -> None:
        """Revalidate exact carriers and the closed state shape."""
        if type(self.status) is not FrameLearningStatus:
            raise TypeError("frame learning status must be a FrameLearningStatus")
        if type(self.occurrences) is not tuple:
            raise TypeError("frame learning occurrences must be a tuple")
        for occurrence in self.occurrences:
            if type(occurrence) is not MemoryOccurrenceRef:
                raise TypeError(
                    "frame learning occurrences must contain MemoryOccurrenceRef values"
                )
            MemoryOccurrenceRef(occurrence.frame_ref, occurrence.occurrence_path)
        if type(self.mutations) is not tuple:
            raise TypeError("frame learning mutations must be a tuple")
        for mutation in self.mutations:
            if type(mutation) is not MemoryMutation:
                raise TypeError(
                    "frame learning mutations must contain MemoryMutation values"
                )
            MemoryMutation(
                mutation.mutation_id,
                mutation.frame_ref,
                mutation.payload,
            )
        if self.failure is not None:
            if type(self.failure) is not MemoryOperationFailure:
                raise TypeError(
                    "frame learning failure must be a MemoryOperationFailure or None"
                )
            MemoryOperationFailure(
                self.failure.stage,
                self.failure.code,
                self.failure.message,
                self.failure.retryable,
            )

        if self.status is FrameLearningStatus.PENDING:
            if self.occurrences or self.mutations or self.failure is not None:
                raise ValueError("pending frame learning state must be empty")
        elif self.status is FrameLearningStatus.STARTED:
            if self.mutations or self.failure is not None:
                raise ValueError("started frame learning state cannot be resolved")
        elif self.status is FrameLearningStatus.LEARNED:
            if self.failure is not None:
                raise ValueError("learned frame state cannot carry a failure")
        else:
            if self.mutations or self.failure is None:
                raise ValueError("failed frame learning state requires one failure")
            expected_retryable = self.status is FrameLearningStatus.FAILED_RETRYABLE
            if self.failure.retryable is not expected_retryable:
                raise ValueError("frame learning status disagrees with retryability")


@dataclass(frozen=True, slots=True)
class PolicyCommitState:
    """Authoritative state of one policy's latest durable commit attempt."""

    status: PolicyCommitStatus = PolicyCommitStatus.UNREQUESTED
    commit_id: str | None = None
    mutations: tuple[MemoryMutation, ...] = ()
    failure: MemoryOperationFailure | None = None

    def __post_init__(self) -> None:
        """Revalidate exact carriers and the closed state shape."""
        if type(self.status) is not PolicyCommitStatus:
            raise TypeError("policy commit status must be a PolicyCommitStatus")
        if self.commit_id is not None and (
            type(self.commit_id) is not str or not self.commit_id
        ):
            raise ValueError("policy commit_id must be a non-empty string or None")
        if type(self.mutations) is not tuple:
            raise TypeError("policy commit mutations must be a tuple")
        for mutation in self.mutations:
            if type(mutation) is not MemoryMutation:
                raise TypeError(
                    "policy commit mutations must contain MemoryMutation values"
                )
            MemoryMutation(
                mutation.mutation_id,
                mutation.frame_ref,
                mutation.payload,
            )
        if self.failure is not None:
            if type(self.failure) is not MemoryOperationFailure:
                raise TypeError(
                    "policy commit failure must be a MemoryOperationFailure or None"
                )
            MemoryOperationFailure(
                self.failure.stage,
                self.failure.code,
                self.failure.message,
                self.failure.retryable,
            )

        if self.status is PolicyCommitStatus.UNREQUESTED:
            if self.commit_id is not None or self.mutations or self.failure is not None:
                raise ValueError("unrequested policy commit state must be empty")
        elif self.status in (
            PolicyCommitStatus.REQUESTED,
            PolicyCommitStatus.SUCCEEDED,
        ):
            if self.commit_id is None or self.failure is not None:
                raise ValueError("requested or succeeded commit state is malformed")
        else:
            if self.commit_id is None or self.failure is None:
                raise ValueError("failed commit state requires identity and failure")
            expected_retryable = self.status is PolicyCommitStatus.FAILED_RETRYABLE
            if self.failure.retryable is not expected_retryable:
                raise ValueError("policy commit status disagrees with retryability")


@dataclass(frozen=True, slots=True)
class _EpisodeLog:
    """Persistent append log with canonical power-of-two chronological chunks.

    Level zero holds one episode, level one holds two, and so on.  Occupied
    levels are read from highest to lowest to recover record order.  Appending
    merges only the binary-counter carry path, so retaining a long MAP/LOOP
    history does not copy or revalidate every earlier episode per outcome.
    """

    levels: tuple[tuple[MemoryEpisode, ...] | None, ...] = ()

    def __post_init__(self) -> None:
        """Validate a directly constructed log once at its public boundary."""
        if type(self.levels) is not tuple:
            raise TypeError("episode log levels must be a tuple")
        if self.levels and self.levels[-1] is None:
            raise ValueError("episode log cannot have trailing empty levels")
        for index, chunk in enumerate(self.levels):
            if chunk is None:
                continue
            if type(chunk) is not tuple or len(chunk) != 1 << index:
                raise ValueError("episode log chunks must match their binary level")
            for episode in chunk:
                _revalidate_episode(episode)

    @classmethod
    def _from_validated(
        cls,
        levels: tuple[tuple[MemoryEpisode, ...] | None, ...],
    ) -> _EpisodeLog:
        """Build from an existing valid log and one already-validated carry."""
        value = object.__new__(cls)
        object.__setattr__(value, "levels", levels)
        return value

    def append(self, episode: MemoryEpisode) -> _EpisodeLog:
        """Return a log with one validated episode appended in record order."""
        _revalidate_episode(episode)
        levels = list(self.levels)
        carry = (episode,)
        index = 0
        while index < len(levels) and levels[index] is not None:
            prior = levels[index]
            if prior is None:  # narrowed by the loop condition
                raise AssertionError("occupied episode level disappeared")
            carry = prior + carry
            levels[index] = None
            index += 1
        if index == len(levels):
            levels.append(carry)
        else:
            levels[index] = carry
        return self._from_validated(tuple(levels))

    def to_tuple(self) -> tuple[MemoryEpisode, ...]:
        """Materialize chronological episodes only at the learning boundary."""
        return tuple(
            episode
            for chunk in reversed(self.levels)
            if chunk is not None
            for episode in chunk
        )


def frame_key(ref: MemoryFrameRef) -> tuple[str, tuple[str, ...], str]:
    """Return the canonical structural ordering key for one frame."""
    if type(ref) is not MemoryFrameRef:
        raise TypeError("frame key requires a MemoryFrameRef")
    return (ref.root_flow_id, ref.static_path, ref.owner.value)


def occurrence_key(
    occurrence: MemoryOccurrenceRef,
) -> tuple[
    tuple[str, tuple[str, ...], str],
    tuple[tuple[str, tuple[str, ...], int], ...],
]:
    """Return the canonical structural ordering key for one occurrence."""
    if type(occurrence) is not MemoryOccurrenceRef:
        raise TypeError("occurrence key requires a MemoryOccurrenceRef")
    return (
        frame_key(occurrence.frame_ref),
        tuple(
            (segment.kind.value, segment.static_path, segment.index)
            for segment in occurrence.occurrence_path
        ),
    )


@dataclass(frozen=True, slots=True)
class RuntimeLearningState:
    """Immutable reducer projection aligned to recorded frame/policy order."""

    frame_order: tuple[MemoryFrameRef, ...] = ()
    policy_order: tuple[str, ...] = ()
    policy_frame_indices: tuple[tuple[int, ...], ...] = ()
    episodes: tuple[_EpisodeLog, ...] = ()
    frames: tuple[FrameLearningState, ...] = ()
    commits: tuple[PolicyCommitState, ...] = ()
    episode_run_id: str | None = None
    active_frame_ref: MemoryFrameRef | None = None

    def __post_init__(self) -> None:
        """Require canonical inventories and revalidate all nested values."""
        if type(self.frame_order) is not tuple:
            raise TypeError("learning frame_order must be a tuple")
        for ref in self.frame_order:
            if type(ref) is not MemoryFrameRef:
                raise TypeError("learning frame_order items must be MemoryFrameRef")
            MemoryFrameRef(ref.root_flow_id, ref.static_path, ref.owner)
        keys = tuple(frame_key(ref) for ref in self.frame_order)
        if keys != tuple(sorted(set(keys))):
            raise ValueError("learning frame_order must be sorted and unique")
        if type(self.policy_order) is not tuple or any(
            type(name) is not str or not name for name in self.policy_order
        ):
            raise TypeError(
                "learning policy_order must be a tuple of non-empty strings"
            )
        if self.policy_order != tuple(sorted(set(self.policy_order))):
            raise ValueError("learning policy_order must be sorted and unique")
        if type(self.policy_frame_indices) is not tuple:
            raise TypeError("policy frame indices must be a tuple")
        if type(self.episodes) is not tuple:
            raise TypeError("learning episodes must be a tuple")
        if type(self.frames) is not tuple or type(self.commits) is not tuple:
            raise TypeError("learning frame and commit states must be tuples")
        if len(self.policy_frame_indices) != len(self.policy_order):
            raise ValueError("policy frame indices must align with policy_order")
        if len(self.episodes) != len(self.frame_order):
            raise ValueError("learning episodes must align with frame_order")
        if len(self.frames) != len(self.frame_order):
            raise ValueError("learning frame states must align with frame_order")
        if len(self.commits) != len(self.policy_order):
            raise ValueError("learning commit states must align with policy_order")
        flattened_indices: list[int] = []
        for indices in self.policy_frame_indices:
            if type(indices) is not tuple or any(
                type(index) is not int for index in indices
            ):
                raise TypeError("policy frame indices must contain integer tuples")
            if indices != tuple(sorted(set(indices))):
                raise ValueError("policy frame indices must be sorted and unique")
            flattened_indices.extend(indices)
        if sorted(flattened_indices) != list(range(len(self.frame_order))):
            raise ValueError("policy frame indices must partition frame_order")
        if self.episode_run_id is not None and (
            type(self.episode_run_id) is not str or not self.episode_run_id
        ):
            raise ValueError("episode_run_id must be a non-empty string or None")
        if self.active_frame_ref is not None and (
            type(self.active_frame_ref) is not MemoryFrameRef
        ):
            raise TypeError("active_frame_ref must be a MemoryFrameRef or None")

        active_ref: MemoryFrameRef | None = None
        episode_run_ids: set[str] = set()
        for index, episode_log in enumerate(self.episodes):
            if type(episode_log) is not _EpisodeLog:
                raise TypeError("frame episodes must be persistent episode logs")
            _EpisodeLog(episode_log.levels)
            ref = self.frame_order[index]
            for episode in episode_log.to_tuple():
                episode_run_ids.add(episode.run_id)
                if episode.occurrence.frame_ref != ref:
                    raise ValueError("episode must align with its frame")
        if len(episode_run_ids) > 1:
            raise ValueError("all learning episodes must share one run_id")
        expected_episode_run_id = next(iter(episode_run_ids), None)
        if self.episode_run_id != expected_episode_run_id:
            raise ValueError("episode_run_id must match retained episodes")
        for index, state in enumerate(self.frames):
            if type(state) is not FrameLearningState:
                raise TypeError("frames must contain FrameLearningState values")
            FrameLearningState(
                state.status,
                state.occurrences,
                state.mutations,
                state.failure,
            )
            ref = self.frame_order[index]
            if any(
                occurrence.frame_ref != ref
                for occurrence in state.occurrences
            ):
                raise ValueError("frame state occurrences must align with frame_order")
            if any(mutation.frame_ref != ref for mutation in state.mutations):
                raise ValueError("frame state mutations must align with frame_order")
            if (
                state.status is not FrameLearningStatus.PENDING
                and state.occurrences != self._occurrence_inventory_at(index)
            ):
                raise ValueError(
                    "frame state occurrences must equal accepted inventory"
                )
            if state.status is FrameLearningStatus.STARTED:
                if active_ref is not None:
                    raise ValueError(
                        "at most one frame learning operation may be active"
                    )
                active_ref = ref
        if self.active_frame_ref != active_ref:
            raise ValueError("active_frame_ref must match the started frame")
        for state in self.commits:
            if type(state) is not PolicyCommitState:
                raise TypeError("commits must contain PolicyCommitState values")
            PolicyCommitState(
                state.status,
                state.commit_id,
                state.mutations,
                state.failure,
            )

    @classmethod
    def for_layout(
        cls,
        layout: tuple[MemoryFrameSpec, ...] | None,
    ) -> RuntimeLearningState:
        """Return pristine learning state for one canonical recorded layout."""
        if layout is None:
            return cls()
        _validate_layout(layout)
        frame_order = tuple(spec.ref for spec in layout)
        policy_order = tuple(sorted({spec.policy_name for spec in layout}))
        policy_buckets = {name: [] for name in policy_order}
        for index, spec in enumerate(layout):
            policy_buckets[spec.policy_name].append(index)
        policy_frame_indices = tuple(
            tuple(policy_buckets[name]) for name in policy_order
        )
        return cls(
            frame_order=frame_order,
            policy_order=policy_order,
            policy_frame_indices=policy_frame_indices,
            episodes=tuple(_EpisodeLog() for _ in frame_order),
            frames=tuple(FrameLearningState() for _ in frame_order),
            commits=tuple(PolicyCommitState() for _ in policy_order),
        )

    def validate_layout(self, layout: tuple[MemoryFrameSpec, ...] | None) -> None:
        """Require inventories to remain aligned with authoritative layout."""
        if layout is None:
            expected_frames: tuple[MemoryFrameRef, ...] = ()
            expected_policies: tuple[str, ...] = ()
        else:
            if type(layout) is not tuple:
                raise TypeError("memory_layout must be a tuple when enabled")
            if any(type(spec) is not MemoryFrameSpec for spec in layout):
                raise TypeError(
                    "memory_layout items must be MemoryFrameSpec values"
                )
            expected_frames = tuple(spec.ref for spec in layout)
            expected_policies = tuple(
                sorted({spec.policy_name for spec in layout})
            )
        if self.frame_order != expected_frames:
            raise ValueError("learning frame inventory disagrees with memory_layout")
        if self.policy_order != expected_policies:
            raise ValueError("learning policy inventory disagrees with memory_layout")
        expected_buckets = {name: [] for name in expected_policies}
        if layout is not None:
            for index, spec in enumerate(layout):
                expected_buckets[spec.policy_name].append(index)
        expected_indices = tuple(
            tuple(expected_buckets[name]) for name in expected_policies
        )
        if self.policy_frame_indices != expected_indices:
            raise ValueError("learning policy frames disagree with memory_layout")

    def episodes_for(self, ref: MemoryFrameRef) -> tuple[MemoryEpisode, ...]:
        """Return accepted episodes for one exact recorded frame."""
        return self.episodes[self._frame_index(ref)].to_tuple()

    def frame_state(self, ref: MemoryFrameRef) -> FrameLearningState:
        """Return durable learn state for one exact recorded frame."""
        return self.frames[self._frame_index(ref)]

    def commit_state(self, policy_name: str) -> PolicyCommitState:
        """Return durable commit state for one exact recorded policy."""
        return self.commits[self._policy_index(policy_name)]

    @property
    def active_frame(self) -> MemoryFrameRef | None:
        """Return the sole started frame, if one exists."""
        return self.active_frame_ref

    def occurrence_inventory(
        self,
        ref: MemoryFrameRef,
    ) -> tuple[MemoryOccurrenceRef, ...]:
        """Return the sorted unique structural occurrences accepted for a frame."""
        return self._occurrence_inventory_at(self._frame_index(ref))

    def applicable_frames(
        self,
        root_flow_id: str,
        static_path: tuple[str, ...],
    ) -> tuple[MemoryFrameRef, ...]:
        """Return exact lexical FLOW ancestors followed by a local AGENT frame."""
        if type(root_flow_id) is not str or not root_flow_id:
            raise ValueError(
                "applicable frame lookup requires a non-empty root_flow_id"
            )
        if (
            type(static_path) is not tuple
            or not static_path
            or any(type(part) is not str or not part for part in static_path)
        ):
            raise ValueError("applicable frame lookup requires a non-empty static_path")
        if static_path[0] != root_flow_id:
            raise ValueError("applicable frame path must begin with root_flow_id")

        applicable: list[MemoryFrameRef] = []
        for length in range(1, len(static_path)):
            ref = self._frame_for_key(
                (root_flow_id, static_path[:length], MemoryOwner.FLOW.value)
            )
            if ref is not None:
                applicable.append(ref)
        local_agent = self._frame_for_key(
            (root_flow_id, static_path, MemoryOwner.AGENT.value)
        )
        if local_agent is not None:
            applicable.append(local_agent)
        return tuple(applicable)

    def learn_context(self, run_id: str, ref: MemoryFrameRef) -> LearnContext:
        """Return a fresh closed context with ordered episodes and no feedback."""
        if type(run_id) is not str or not run_id:
            raise ValueError("learning context requires a non-empty run_id")
        if self.episode_run_id is not None and self.episode_run_id != run_id:
            raise ValueError("learning context run_id disagrees with its episodes")
        episodes = self.episodes_for(ref)
        return LearnContext(
            run_id=run_id,
            frame_ref=ref,
            episodes=episodes,
            feedback=(),
        )

    def append_episodes(
        self,
        additions: tuple[tuple[MemoryFrameRef, MemoryEpisode], ...],
    ) -> RuntimeLearningState:
        """Append one accepted outcome's projections in durable record order."""
        if type(additions) is not tuple:
            raise TypeError("episode additions must be a tuple")
        if not additions:
            return self
        updated = list(self.episodes)
        addition_run_ids: set[str] = set()
        for ref, episode in additions:
            index = self._frame_index(ref)
            appended = updated[index].append(episode)
            if episode.occurrence.frame_ref != ref:
                raise ValueError("episode addition frame does not match occurrence")
            addition_run_ids.add(episode.run_id)
            updated[index] = appended
        if len(addition_run_ids) != 1:
            raise ValueError("one accepted outcome must carry one episode run_id")
        addition_run_id = next(iter(addition_run_ids))
        if (
            self.episode_run_id is not None
            and addition_run_id != self.episode_run_id
        ):
            raise ValueError("episode addition run_id disagrees with retained history")
        return self._from_validated(
            episodes=tuple(updated),
            episode_run_id=addition_run_id,
        )

    def with_frame_state(
        self,
        ref: MemoryFrameRef,
        state: FrameLearningState,
    ) -> RuntimeLearningState:
        """Replace one frame state while retaining canonical layout order."""
        if type(state) is not FrameLearningState:
            raise TypeError("frame replacement must be a FrameLearningState")
        FrameLearningState(
            state.status,
            state.occurrences,
            state.mutations,
            state.failure,
        )
        index = self._frame_index(ref)
        if any(occurrence.frame_ref != ref for occurrence in state.occurrences):
            raise ValueError("frame state occurrences must align with its frame")
        if any(mutation.frame_ref != ref for mutation in state.mutations):
            raise ValueError("frame state mutations must align with its frame")
        if (
            state.status is not FrameLearningStatus.PENDING
            and state.occurrences != self._occurrence_inventory_at(index)
        ):
            raise ValueError("frame state occurrences must equal accepted inventory")
        if state.status is FrameLearningStatus.STARTED:
            active = self.active_frame
            if active is not None and active != ref:
                raise ValueError("at most one frame learning operation may be active")
        next_active = self.active_frame_ref
        if state.status is FrameLearningStatus.STARTED:
            next_active = ref
        elif next_active == ref:
            next_active = None
        updated = list(self.frames)
        updated[index] = state
        return self._from_validated(
            frames=tuple(updated),
            active_frame_ref=next_active,
        )

    def with_commit_state(
        self,
        policy_name: str,
        state: PolicyCommitState,
    ) -> RuntimeLearningState:
        """Replace one policy state while retaining canonical policy order."""
        if type(state) is not PolicyCommitState:
            raise TypeError("commit replacement must be a PolicyCommitState")
        PolicyCommitState(
            state.status,
            state.commit_id,
            state.mutations,
            state.failure,
        )
        updated = list(self.commits)
        updated[self._policy_index(policy_name)] = state
        return self._from_validated(commits=tuple(updated))

    def policy_mutations(
        self,
        policy_name: str,
    ) -> tuple[MemoryMutation, ...]:
        """Aggregate one policy in cached frame order without scanning layout."""
        policy_index = self._policy_index(policy_name)
        mutations: list[MemoryMutation] = []
        for frame_index in self.policy_frame_indices[policy_index]:
            state = self.frames[frame_index]
            if state.status is FrameLearningStatus.LEARNED:
                mutations.extend(state.mutations)
            elif state.status is FrameLearningStatus.FAILED_FINAL:
                continue
            else:
                raise ValueError(
                    "policy commit requires every frame to be learned or failed_final"
                )
        return tuple(mutations)

    def _occurrence_inventory_at(
        self,
        index: int,
    ) -> tuple[MemoryOccurrenceRef, ...]:
        """Return one frame's structural inventory from its persistent log."""
        occurrences = {
            episode.occurrence for episode in self.episodes[index].to_tuple()
        }
        return tuple(sorted(occurrences, key=occurrence_key))

    def _from_validated(
        self,
        *,
        episodes: tuple[_EpisodeLog, ...] | None = None,
        frames: tuple[FrameLearningState, ...] | None = None,
        commits: tuple[PolicyCommitState, ...] | None = None,
        episode_run_id: str | None = None,
        active_frame_ref: MemoryFrameRef | None | object = _PRESERVE,
    ) -> RuntimeLearningState:
        """Replace validated aligned roots without rescanning frozen history."""
        value = object.__new__(type(self))
        object.__setattr__(value, "frame_order", self.frame_order)
        object.__setattr__(value, "policy_order", self.policy_order)
        object.__setattr__(
            value,
            "policy_frame_indices",
            self.policy_frame_indices,
        )
        object.__setattr__(
            value,
            "episodes",
            self.episodes if episodes is None else episodes,
        )
        object.__setattr__(value, "frames", self.frames if frames is None else frames)
        object.__setattr__(
            value,
            "commits",
            self.commits if commits is None else commits,
        )
        object.__setattr__(
            value,
            "episode_run_id",
            self.episode_run_id if episode_run_id is None else episode_run_id,
        )
        object.__setattr__(
            value,
            "active_frame_ref",
            (
                self.active_frame_ref
                if active_frame_ref is _PRESERVE
                else active_frame_ref
            ),
        )
        return value

    def _frame_index(self, ref: MemoryFrameRef) -> int:
        """Resolve one exact frame or reject an unrecorded identity."""
        if type(ref) is not MemoryFrameRef:
            raise TypeError("frame lookup requires a MemoryFrameRef")
        key = frame_key(ref)
        index = bisect_left(self.frame_order, key, key=frame_key)
        if index >= len(self.frame_order) or self.frame_order[index] != ref:
            raise ValueError("memory frame is absent from recorded layout")
        return index

    def _policy_index(self, policy_name: str) -> int:
        """Resolve one exact policy or reject an unrecorded identity."""
        if type(policy_name) is not str:
            raise TypeError("policy name must be a string")
        index = bisect_left(self.policy_order, policy_name)
        if (
            index >= len(self.policy_order)
            or self.policy_order[index] != policy_name
        ):
            raise ValueError("memory policy is absent from recorded layout")
        return index

    def _frame_for_key(
        self,
        key: tuple[str, tuple[str, ...], str],
    ) -> MemoryFrameRef | None:
        """Return the exact sorted frame at ``key`` without scanning inventory."""
        index = bisect_left(self.frame_order, key, key=frame_key)
        if index >= len(self.frame_order):
            return None
        ref = self.frame_order[index]
        return ref if frame_key(ref) == key else None


def _validate_layout(layout: tuple[MemoryFrameSpec, ...]) -> None:
    """Revalidate exact layout carriers and canonical structural order."""
    if type(layout) is not tuple:
        raise TypeError("memory_layout must be a tuple when enabled")
    for spec in layout:
        if type(spec) is not MemoryFrameSpec:
            raise TypeError("memory_layout items must be MemoryFrameSpec values")
        MemoryFrameSpec(spec.ref, spec.policy_name, spec.config)
    keys = tuple(frame_key(spec.ref) for spec in layout)
    if keys != tuple(sorted(set(keys))):
        raise ValueError("memory_layout frame refs must be sorted and unique")
    if len({spec.ref.root_flow_id for spec in layout}) > 1:
        raise ValueError("memory_layout frame refs must share one root_flow_id")


def _revalidate_episode(episode: MemoryEpisode) -> None:
    """Replay every exact public invariant of one immutable episode."""
    if type(episode) is not MemoryEpisode:
        raise TypeError("episodes must contain MemoryEpisode values")
    MemoryEpisode(
        run_id=episode.run_id,
        occurrence=episode.occurrence,
        owner_static_path=episode.owner_static_path,
        status=episode.status,
        inputs=episode.inputs,
        output=episode.output,
        failure=episode.failure,
        interactions=episode.interactions,
        base_system_prompt=episode.base_system_prompt,
        prompt=episode.prompt,
        memory_sections=episode.memory_sections,
    )


def failure_json(failure: Failure) -> dict[str, object]:
    """Project one canonical durable Failure into source-neutral JSON."""
    if type(failure) is not Failure:
        raise TypeError("failure projection requires a Failure")
    Failure(
        code=failure.code,
        message=failure.message,
        node_id=failure.node_id,
        trace=failure.trace,
        cause_type=failure.cause_type,
        traceback=failure.traceback,
    )
    return {
        "code": failure.code,
        "message": failure.message,
        "node_id": failure.node_id,
        "trace": [{"key": origin.key} for origin in failure.trace],
        "cause_type": failure.cause_type,
        "traceback": failure.traceback,
    }


def project_outcome_episodes(
    run_id: str,
    record: (
        NodeSucceededRecord
        | NodeRoutedRecord
        | NodeExpandedRecord
        | NodeFailedRecord
    ),
) -> tuple[tuple[MemoryFrameRef, MemoryEpisode], ...]:
    """Project one canonical accepted graph outcome into every scoped frame."""
    if type(run_id) is not str or not run_id:
        raise ValueError("episode projection requires a non-empty run_id")
    evidence = record.evidence
    if evidence is None:
        return ()
    if type(evidence) is not EvaluationEvidence:
        raise TypeError("episode projection evidence must be EvaluationEvidence")
    if evidence.scope is None:
        return ()

    if type(record) is NodeSucceededRecord:
        if type(record.value) is not dict or "value" not in record.value:
            raise ValueError("successful record has no canonical typed value")
        output = canonical_seed(record.value["value"])[1]
        status = MemoryEpisodeStatus.SUCCEEDED
        failure = None
    elif type(record) is NodeRoutedRecord:
        output = {"kind": "route", "handle": record.handle}
        status = MemoryEpisodeStatus.SUCCEEDED
        failure = None
    elif type(record) is NodeExpandedRecord:
        output = {
            "kind": "grow",
            "seed": canonical_seed(record.seed)[1],
            "prune_ids": list(record.prune_ids),
        }
        status = MemoryEpisodeStatus.SUCCEEDED
        failure = None
    elif type(record) is NodeFailedRecord:
        output = None
        status = MemoryEpisodeStatus.FAILED
        failure = failure_json(record.failure)
    else:
        raise TypeError("unsupported graph outcome for episode projection")

    additions: list[tuple[MemoryFrameRef, MemoryEpisode]] = []
    for frame_ref in evidence.scope.frame_stack:
        occurrence = MemoryOccurrenceRef(
            frame_ref,
            evidence.scope.occurrence_path,
        )
        additions.append(
            (
                frame_ref,
                MemoryEpisode(
                    run_id=run_id,
                    occurrence=occurrence,
                    owner_static_path=evidence.scope.static_path,
                    status=status,
                    inputs=evidence.bound_inputs,
                    output=output,
                    failure=failure,
                    interactions=evidence.interactions,
                    base_system_prompt=evidence.base_system_prompt,
                    prompt=evidence.prompt,
                    memory_sections=evidence.memory_sections,
                ),
            )
        )
    return tuple(additions)


def aggregate_policy_mutations(
    layout: tuple[MemoryFrameSpec, ...],
    learning: RuntimeLearningState,
    policy_name: str,
) -> tuple[MemoryMutation, ...]:
    """Aggregate learned mutations in recorded frame and proposal order."""
    _validate_layout(layout)
    if type(learning) is not RuntimeLearningState:
        raise TypeError("mutation aggregation requires RuntimeLearningState")
    learning.validate_layout(layout)
    if type(policy_name) is not str or not policy_name:
        raise ValueError("policy_name must be a non-empty string")
    if policy_name not in learning.policy_order:
        raise ValueError("memory policy is absent from recorded layout")
    return learning.policy_mutations(policy_name)


def memory_commit_id(
    run_id: str,
    policy_name: str,
    mutations: tuple[MemoryMutation, ...],
) -> str:
    """Return the sole deterministic ID for a policy's canonical mutation batch.

    The SHA-256 input is UTF-8 compact JSON with sorted object keys and no NaN:
    ``[run_id, policy_name, canonical_batch]``.  Batch order is semantic and each
    item contains ``mutation_id``, exact frame-ref fields, and the JSON payload.
    """
    if type(run_id) is not str or not run_id:
        raise ValueError("run_id must be a non-empty string")
    if type(policy_name) is not str or not policy_name:
        raise ValueError("policy_name must be a non-empty string")
    if type(mutations) is not tuple:
        raise TypeError("mutations must be a tuple")
    batch: list[dict[str, object]] = []
    for mutation in mutations:
        if type(mutation) is not MemoryMutation:
            raise TypeError("mutations items must be MemoryMutation values")
        MemoryMutation(
            mutation.mutation_id,
            mutation.frame_ref,
            mutation.payload,
        )
        batch.append(
            {
                "mutation_id": mutation.mutation_id,
                "frame_ref": {
                    "root_flow_id": mutation.frame_ref.root_flow_id,
                    "static_path": list(mutation.frame_ref.static_path),
                    "owner": mutation.frame_ref.owner.value,
                },
                "payload": mutation.payload,
            }
        )
    encoded = json.dumps(
        [run_id, policy_name, batch],
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"ac-memory-v1:{sha256(encoded).hexdigest()}"


__all__ = [
    "FrameLearningState",
    "FrameLearningStatus",
    "PolicyCommitState",
    "PolicyCommitStatus",
    "RuntimeLearningState",
    "aggregate_policy_mutations",
    "failure_json",
    "frame_key",
    "memory_commit_id",
    "occurrence_key",
    "project_outcome_episodes",
]
