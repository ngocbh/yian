"""Unit-only accepted-checkpoint harness without a provider or runtime log."""

from contextlib import contextmanager
from types import SimpleNamespace

import agent_composer.memory.contracts as c
from agent_composer.memory.conversation import advance_conversation
from agent_composer.memory.session import SessionMemory

REF = c.MemoryFrameRef("root", ("root",), c.MemoryOwner.FLOW)


def request(run_id="a", sequence=1, consumer=("root", "agent"), compatibility=None):
    return c.ConversationRequest(
        consumer,
        c.ConversationInvocation(run_id, sequence),
        {"format": "fake-v1"} if compatibility is None else compatibility,
    )


def item(item_id, text, role="assistant"):
    return c.ConversationItem(item_id, "fake-v1", {"role": role, "content": text})


def snapshot(memory):
    with memory.open_run() as opened:
        return opened.snapshot


@contextmanager
def harness(memory=None, *, policy="session", ref=REF):
    memory = SessionMemory() if memory is None else memory
    with memory.open_run() as opened:
        frame = opened.open_frame(c.MemoryFrameSpec(ref, policy, {}))
        accepted = []
        accepted_views = {}

        def accept(checkpoint):
            before = (
                checkpoint.baseline
                if type(checkpoint) is c.ConversationAdmitted
                else accepted_views[checkpoint.stamp.key]
            )
            apply = opened.prepare_checkpoint(checkpoint, before=before)
            after = advance_conversation(before, checkpoint)
            accepted.append(checkpoint)
            accepted_views[checkpoint.stamp.key] = after
            return apply()

        yield SimpleNamespace(
            memory=memory,
            session=opened,
            frame=frame,
            accepted=accepted,
            accepted_views=accepted_views,
            checkpoint=accept,
        )


def append_reply(turn, item_id, text):
    ordinal = turn.request("model", {"prompt": text})
    turn.observe(ordinal, "completed", items=(item(item_id, text),))
