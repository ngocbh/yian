"""Synthetic Responses HTTP traffic for provider and real agent-loop tests."""

import json

import httpx

REASONING = {
    "id": "rs_test",
    "type": "reasoning",
    "summary": [],
    "encrypted_content": "synthetic-continuation",
}
TOOL_CALL = {
    "id": "fc_test",
    "type": "function_call",
    "name": "double",
    "arguments": '{"value": 3}',
    "call_id": "call_test",
    "status": "completed",
}
ANSWER = {
    "id": "msg_test",
    "type": "message",
    "role": "assistant",
    "status": "completed",
    "content": [{"type": "output_text", "text": "6", "annotations": []}],
}


class EventStream(httpx.SyncByteStream, httpx.AsyncByteStream):
    """Track closure of synthetic network streams, including interrupted reads."""

    def __init__(self, events):
        self.chunks = [
            f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode()
            for event in events
        ]
        self.closed = False

    def __iter__(self):
        yield from self.chunks

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk

    def close(self):
        self.closed = True

    async def aclose(self):
        self.close()


class ResponsesServer:
    """Offer one tool call, then text replies, through real SDK wire parsing."""

    def __init__(self, *, terminal="completed", http_status=200, tool_first=True):
        self.terminal = terminal
        self.http_status = http_status
        self.tool_first = tool_first
        self.requests = []
        self.bodies = []
        self.streams = []

    def __call__(self, request):
        self.requests.append(request)
        body = json.loads(request.content)
        self.bodies.append(body)
        if self.http_status != 200:
            return httpx.Response(
                self.http_status, json={"error": {"message": "denied"}}
            )
        number = len(self.requests)
        is_tool = self.tool_first and number == 1
        output = [REASONING, TOOL_CALL if is_tool else ANSWER]
        response = {
            "id": f"resp_{number}",
            "object": "response",
            "created_at": 0,
            "status": self.terminal or "in_progress",
            "model": "gpt-5.5",
            "output": output,
            "error": None,
            "incomplete_details": None,
            "usage": None,
        }
        if not body.get("stream"):
            return httpx.Response(200, json=response)
        events = [
            {"type": "response.created", "response": {**response, "output": []}},
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {**REASONING, "encrypted_content": None},
            },
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": REASONING,
            },
        ]
        if is_tool:
            events.extend(
                [
                    {
                        "type": "response.output_item.added",
                        "output_index": 1,
                        "item": {**TOOL_CALL, "arguments": "", "status": "in_progress"},
                    },
                    {
                        "type": "response.function_call_arguments.delta",
                        "output_index": 1,
                        "item_id": "fc_test",
                        "delta": '{"value":',
                    },
                    {
                        "type": "response.function_call_arguments.delta",
                        "output_index": 1,
                        "item_id": "fc_test",
                        "delta": " 3}",
                    },
                ]
            )
        else:
            events.extend(
                [
                    {
                        "type": "response.output_text.delta",
                        "output_index": 1,
                        "content_index": 0,
                        "item_id": "msg_test",
                        "delta": "6",
                    },
                    {
                        "type": "response.output_text.done",
                        "output_index": 1,
                        "content_index": 0,
                        "item_id": "msg_test",
                        "text": "6",
                    },
                ]
            )
        if self.terminal:
            events.append({"type": f"response.{self.terminal}", "response": response})
        stream = EventStream(events)
        self.streams.append(stream)
        return httpx.Response(
            200, stream=stream, headers={"content-type": "text/event-stream"}
        )
