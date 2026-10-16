"""Provider-neutral model calls: messages, streams, errors, and HTTP transport.

The core that model adapters build on, independent of any provider SDK: yian's
message, stream-event and error vocabulary with a stable JSON shape; an httpx2
transport that refuses redirects and sends no environment-derived headers; SSE and
NDJSON framing; auth objects that list the credentials they send; and routes and an
executor that retries only before the first frame. Test helpers (cassettes, a
recorder, a scripted model) live in `yian.llm.testing`, which ``import yian.llm`` does
not load. The package sits beside `llm_clients`, which will build on it.

Knows about:   the standard library, httpx2, yian.defaults.
Never imports: any other yian package, provider SDKs, langchain. `llm_clients`
               imports this package, never the reverse.
"""

from yian.llm.auth import Auth, BearerAuth, HeaderKeyAuth, NoAuth, Secret, SubscriptionAuth
from yian.llm.errors import ErrorKind, ProviderError, error_from_response, parse_retry_after
from yian.llm.events import (
    ChatModel,
    Event,
    EventStream,
    Finish,
    PartDone,
    ReasoningDelta,
    TextDelta,
    ToolCallDelta,
    UsageReport,
    collect,
    parse_arguments,
)
from yian.llm.executor import Executor, FrameStream, Framing
from yian.llm.framing import SSEEvent, iter_ndjson, iter_sse
from yian.llm.messages import (
    MESSAGE_FORMAT,
    MediaPart,
    Message,
    MessageFormatError,
    NamedTool,
    Part,
    ReasoningPart,
    Request,
    Response,
    Role,
    TextPart,
    ToolCallPart,
    ToolChoice,
    ToolResultPart,
    ToolSpec,
    Usage,
)
from yian.llm.route import Route
from yian.llm.transport import DEFAULT_TIMEOUT, USER_AGENT, build_client, check_client

__all__ = [
    "DEFAULT_TIMEOUT",
    "MESSAGE_FORMAT",
    "USER_AGENT",
    "Auth",
    "BearerAuth",
    "ChatModel",
    "ErrorKind",
    "Event",
    "EventStream",
    "Executor",
    "Finish",
    "FrameStream",
    "Framing",
    "HeaderKeyAuth",
    "MediaPart",
    "Message",
    "MessageFormatError",
    "NamedTool",
    "NoAuth",
    "Part",
    "PartDone",
    "ProviderError",
    "ReasoningDelta",
    "ReasoningPart",
    "Request",
    "Response",
    "Role",
    "Route",
    "SSEEvent",
    "Secret",
    "SubscriptionAuth",
    "TextDelta",
    "TextPart",
    "ToolCallDelta",
    "ToolCallPart",
    "ToolChoice",
    "ToolResultPart",
    "ToolSpec",
    "Usage",
    "UsageReport",
    "build_client",
    "check_client",
    "collect",
    "error_from_response",
    "iter_ndjson",
    "iter_sse",
    "parse_arguments",
    "parse_retry_after",
]
