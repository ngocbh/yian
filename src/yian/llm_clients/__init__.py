"""LLM client wrappers and `LLMConfig` for AGENT nodes — yian's model seam.

The model-access leaf: `LLMConfig` (the selection
carrier) and `model_from_config` (resolve it to a ready `BaseChatModel`), plus the
per-provider client wrappers behind `create_llm_client`. The AGENT node and the
Compose loader's `llm_config:` field draw on this; `chat/` and `cli/` use it too.
`provider_blocks` holds the schema of each provider block in `LLMConfig`: the request
body of the API that provider's calls reach, checked at load and again at run start
against the API actually called, then sent as written by the client wrappers.
The direct OpenAI subscription transport accepts an injected access source and
returns a normal chat model; it owns no login, credential persistence, or tool loop.
Both OpenAI authentication methods use this model interface. `auth/` owns staged
login, private atomic storage of one saved login per provider, and refresh.
`connection` snapshots the environment and saved defaults (there is no built-in
provider or model) and nonsecret connection identities for the host, and resolves
pinned credentials for requests. `connection_settings` checks the connection settings
that travel with a key (an OpenAI organization or project id and extra request
headers) for the carrier, the auth store, and `yian auth`. `layers` checks one
`llm_config` layer and resolves an agent's merged layers, the host defaults and saved
models into its complete effective configuration, for the compile-time cascade.
OpenAI Responses messages retain their full continuation blocks; presentation
extracts text without mutating the messages used for the next request.
`list_models` lists the model ids a provider's host offers, using only the key,
base URL, and connection settings passed in.

Knows about:   langchain, provider SDKs, and the stdlib-only yian.defaults.
Never imports: any other yian subpackage except the stdlib-only yian.defaults
               leaf — it is a leaf (peer of state / events), so
               nodes/compose/runtime may import it, never the reverse.
"""

from yian.llm_clients.base_client import BaseLLMClient
from yian.llm_clients.config import LLM_CONFIG_KEYS, LLMConfig
from yian.llm_clients.factory import (
    create_llm_client,
    model_from_config,
)
from yian.llm_clients.model_list import ModelListError, list_models

__all__ = [
    "BaseLLMClient",
    "LLM_CONFIG_KEYS",
    "LLMConfig",
    "ModelListError",
    "create_llm_client",
    "list_models",
    "model_from_config",
]
