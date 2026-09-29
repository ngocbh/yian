"""AGENT reply streaming: visible chat text reaches `emit_stream` as it arrives.

A chat request streams only when the host supplied a stream sink. Its chunks share
the request's `exchange_id`. A completed request ends with one empty `final=True`
chunk; a failed request ends without one. Without a stream sink the model is
invoked exactly as before.
"""

from agent_composer.events import StreamChunk


def test_stream_chunk_exchange_id_defaults_to_none():
    assert StreamChunk("n", "text", "hi").exchange_id is None
    assert StreamChunk("n", "text", "hi", exchange_id="x").exchange_id == "x"
