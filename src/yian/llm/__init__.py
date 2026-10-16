"""Provider-neutral model calls: messages, streams, errors, and HTTP transport.

The core that model adapters build on, independent of any provider SDK: yian's
message, stream-event and error vocabulary with a stable JSON shape; an httpx2
transport that refuses redirects and sends no environment-derived headers; SSE and
NDJSON framing; auth objects that list the credentials they send; routes and an
executor that retries only before the first frame; and test helpers (cassettes, a
recorder, a scripted model). It sits beside `llm_clients`, which will build on it.

Knows about:   the standard library, httpx2, yian.defaults.
Never imports: any other yian package, provider SDKs, langchain. `llm_clients`
               imports this package, never the reverse.
"""
