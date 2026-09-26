"""LLM client wrappers and `LLMConfig` for AGENT nodes — agent_composer's model seam.

The model-access leaf: `LLMConfig` (the selection
carrier) and `model_from_config` (resolve it to a ready `BaseChatModel`), plus the
per-provider client wrappers behind `create_llm_client`. The AGENT node and the
Compose loader's `llm_config:` field draw on this; `chat/` and `cli/` use it too.
Codex is a native-only selection: the chat-model factories reject it rather than
falling back to an API provider. The private `codex/` package isolates the pinned
native SDK in a supervised process, without owning agent policy or session memory.
The direct OpenAI subscription transport accepts an injected access source and
returns a normal chat model; it owns no login, credential persistence, or tool loop.
It is an internal construction seam pending authentication/factory integration.
OpenAI Responses messages retain their full continuation blocks; presentation
extracts text without mutating the messages used for the next request.

Knows about:   the package env-based defaults (`_settings`), langchain, provider SDKs.
Never imports: any other agent_composer subpackage — it is a leaf (peer of state /
               events), so nodes/compose/runtime may import it, never the reverse.
"""

from agent_composer.llm_clients.base_client import BaseLLMClient
from agent_composer.llm_clients.config import LLMConfig
from agent_composer.llm_clients.factory import (
    create_llm_client,
    model_from_config,
)

__all__ = [
    "BaseLLMClient",
    "LLMConfig",
    "create_llm_client",
    "model_from_config",
]
