# Installation

Agent Composer needs **Python 3.10+**.

The current documentation describes the `main` branch. PyPI release `0.0.1`
contains an older, incompatible package, so install the current version from a
clone until the next release:

```console
git clone https://github.com/ngocbh/agent-composer.git
cd agent-composer
python -m pip install .
```

The current source uses distribution name `agent-composer`, import name
`agent_composer`, and CLI command `ac`.

```console
ac --help
```

## Provider extras

The core (engine + `ac` CLI) installs with **no LLM SDK**. Each provider is an
optional extra — install the one(s) you actually use:

```console
python -m pip install ".[anthropic]"   # Claude  (langchain-anthropic)
python -m pip install ".[openai]"      # GPT     (langchain-openai)
python -m pip install ".[google]"      # Gemini  (langchain-google-genai)
python -m pip install ".[ollama]"      # local   (langchain-ollama)
python -m pip install ".[codex]"       # native Codex SDK and bundled runtime
python -m pip install ".[all]"         # all provider extras
```

Run these commands from the repository root.

The `codex` extra pins the unchanged official `openai-codex==0.155.1` SDK and its
bundled runtime. Composer uses its lower-level client in a supervised worker;
upgrades require compatibility testing because that SDK interface is internal.
This path requires a POSIX host with process-group support; unsupported hosts
are rejected before launch. Offline compatibility tests use a fake app-server,
without credentials or model calls.

Provider SDKs are imported **lazily** — only when a flow actually runs an AGENT node
against that provider. Importing a provider you haven't installed raises a clear
`pip install agent-composer[...]` hint rather than a bare `ImportError`.

## Choosing a provider and model

The default provider and model are read from the environment, plus each
provider's own credential variable:

```console
export AGENT_COMPOSER_DEFAULT_PROVIDER=anthropic        # or openai / google / ollama
export AGENT_COMPOSER_DEFAULT_MODEL=claude-sonnet-4-5
export ANTHROPIC_API_KEY=...                            # the provider's own key var
```

A flow can also pin its own model per AGENT node; the environment defaults apply
only where a node leaves the model unset.

When you `ac run` a flow whose AGENT nodes need a provider key that isn't set, the CLI
prompts for it up front (in an interactive terminal) and continues once entered —
so a missing key surfaces as a prompt rather than a failure on the first model
call. Off a terminal (CI, a pipe) a missing required key is a clean error before
the run starts. Ollama and native Codex are skipped; Codex handles its own
authentication.

### Codex with existing authentication

Composer reuses the SDK's existing Codex authentication. ChatGPT sign-in uses
your eligible subscription; API-key sign-in uses separately billed API access.
Account limits and workspace permissions still apply. If the Codex CLI is already
installed, `codex login` starts browser sign-in. See the
[official authentication guide](https://learn.chatgpt.com/docs/auth).

Alternatively, the pinned SDK provides an explicit host-side login helper:

```python
from openai_codex import Codex

with Codex() as codex:
    login = codex.login_chatgpt()
    print(login.auth_url)  # Open this URL to sign in.
    print(login.wait().success)
```

Run login separately, before executing flows. Composer adds no login or
credential-management path and never falls back to an API provider. The SDK bundles its runtime,
so a global `codex` executable is not required for Composer execution. See the
[official SDK guide](https://learn.chatgpt.com/docs/codex-sdk).

Use `llm_config: {provider: codex, model: your-codex-model}` with
`mode: tool_calling` and `tools: [builtin_tools]`. Filesystem access defaults to
read-only; `llm_config.sandbox: workspace-write` permits workspace writes.
This explicitly enables Codex's configured tool environment, including enabled
MCP/app tools, and is not a complete native-tool allowlist. See
[native execution](syntax.md#codex-native-execution) for the complete example,
tool boundary, and forward-only restrictions.

### Local models with Ollama

No API key is needed — point at a running Ollama endpoint:

```console
export AGENT_COMPOSER_DEFAULT_PROVIDER=ollama
export AGENT_COMPOSER_DEFAULT_MODEL=llama3.2:3b
export OLLAMA_BASE_URL=http://localhost:11434
ac run examples/hello.yaml --input name=Ada
```

The example path exists in the clone; run this command from its repository root.

## Development install

To work on Agent Composer itself, install it editable with the test extras:

```console
git clone https://github.com/ngocbh/agent-composer
cd agent-composer
pip install -e ".[all,dev]"
pytest
```

## Local development environment

The provider/model/key variables above are read from the **shell environment**.
`ac` does **not** auto-load a `.env` file — it only reads variables already
present in the environment. A `.env` is therefore just a convenient place to
keep them; you make it take effect by *sourcing* it (or having direnv do that
for you).

Copy the checked-in template and edit it for your provider:

```console
cp .env.example .env      # then edit; .env is gitignored
```

The template keeps an `export` prefix on each line, so a plain source exports
them into your current shell:

```console
source .env               # once per shell, in the project directory
ac run examples/hello.yaml --input name=Ada
```

### Optional: auto-load with direnv

[direnv](https://direnv.net) loads/unloads the environment automatically as you
`cd` in and out of the project — no manual `source`. After
[installing direnv](https://direnv.net/docs/installation.html) and adding its
shell hook (e.g. `eval "$(direnv hook zsh)"` in `~/.zshrc`), drop an `.envrc`
in the project root:

```bash
# .envrc — activate the project venv and load .env on entry
export VIRTUAL_ENV="$(expand_path .venv)"
PATH_add "$VIRTUAL_ENV/bin"
dotenv
```

Then authorize it once (direnv never runs an unreviewed `.envrc`):

```console
direnv allow
```

Now `cd`-ing into the directory activates `.venv` (so `ac`, `pytest`, `python`
resolve from it) and loads `.env`; leaving unloads them again. `.env` and
`.envrc` are gitignored — keep secrets out of version control.

## Next

- [The `ac` CLI](cli.md) — run flows from the terminal.
- [Flow syntax](syntax.md) — the Compose-YAML reference.
