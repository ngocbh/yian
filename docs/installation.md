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
`agent_composer`, and CLI command `composer`.

```console
composer --help
```

Earlier builds installed this command as ac, and no alias is kept. On macOS, a
bare ac now runs the unrelated system tool /usr/sbin/ac. If PHP Composer is also
installed, activate your virtual environment first so the venv's `composer`
comes first on `PATH`.

## Provider extras

The core (engine + `composer` CLI) installs with **no LLM SDK**. Each provider is an
optional extra — install the one(s) you actually use:

```console
python -m pip install ".[anthropic]"   # Claude  (langchain-anthropic)
python -m pip install ".[openai]"      # GPT     (langchain-openai)
python -m pip install ".[google]"      # Gemini  (langchain-google-genai)
python -m pip install ".[ollama]"      # local   (langchain-ollama)
python -m pip install ".[all]"         # all provider extras
```

Run these commands from the repository root.

Provider SDKs are imported **lazily** — only when a flow actually runs an AGENT node
against that provider. Importing a provider you haven't installed raises a clear
`pip install agent-composer[...]` hint rather than a bare `ImportError`.

## Choosing a provider and model

OpenAI is the default provider. Run `composer auth openai` to choose authentication
and a model. Environment variables can override the saved provider/model
defaults; other providers use their own credential variables:

```console
export AGENT_COMPOSER_DEFAULT_PROVIDER=anthropic        # or openai / google / ollama
export AGENT_COMPOSER_DEFAULT_MODEL=claude-sonnet-4-5
export ANTHROPIC_API_KEY=...                            # the provider's own key var
```

A Python host can instead pass the key in `run_flow(..., llm_config={"provider": ...,
"api_key": key})`, which takes precedence over the variable and is never recorded.
Pass it again to `resume_flow` and `RunResult.backward` with `api_key=key`; agents
marked for the key never fall back to the variable. Add `"base_url": url` beside the
key to send it to another endpoint, such as an Azure OpenAI resource's
`https://<resource>.openai.azure.com/openai/v1/` with `"provider": "openai"` and the
deployment name as `model`; no Azure environment variables are needed.

A flow can also pin its own provider and model per AGENT node. Defaults fill
only fields left unset by the flow or CLI. A default model belongs to its
provider: selecting another provider never borrows the OpenAI model. Without
a matching saved or environment model, OpenAI uses `gpt-5.5` and Anthropic uses
`claude-sonnet-4-5`; other providers require an explicit model.

When you `composer run` a flow whose AGENT nodes need a provider key that isn't set, the CLI
prompts for it up front (in an interactive terminal) and continues once entered —
so a missing key surfaces as a prompt rather than a failure on the first model
call. Off a terminal (CI, a pipe) a missing required key is a clean error before
the run starts. Ollama is keyless; saved OpenAI authentication is checked through
the selected connection method.

### OpenAI authentication

```console
composer auth openai
```

Choose ChatGPT subscription sign-in (browser or device login for SSH) or an
OpenAI API key, then select a model. Successful setup saves the method and model
as the default together. Cancelled or failed setup leaves the previous selection
intact. The `openai` extra supports both methods.
Subscription-backed nodes must omit `llm_config.temperature`, which that
transport does not accept.

For SSH, select device login. Flags can supply choices without interactive
menus; subscription login still requires completing sign-in in a browser:

```console
composer auth openai --method subscription --login device --model gpt-5.5
composer auth openai --method api-key --model gpt-5.5
```

API-key setup reads `OPENAI_API_KEY`, or prompts without echoing the key in a
terminal. `--api-key-env NAME` selects another environment variable. Model
choices come from the selected account's authenticated catalog.

Credentials belong to Composer's private user-local store; tokens never belong
in flow YAML or run logs. Subscription selection takes precedence over an
incidental `OPENAI_API_KEY`. Selecting an API key uses separately billed API
access. Reconnecting changes the connection identity; an unfinished run cannot
silently switch accounts or billing methods.

The store is `$XDG_CONFIG_HOME/agent-composer/auth.json`, falling back to
`~/.config/agent-composer/auth.json`. Set `AGENT_COMPOSER_AUTH_FILE` to choose
another private file. The store uses private filesystem permissions and atomic
writes on POSIX hosts; it is not encrypted. Token refresh preserves the selected connection.
Failed authentication never falls back to another billing method.

Migrate native flows to `provider: openai` and remove `sandbox`, `cwd`, and
`builtin_tools`. Grant registered Composer tools explicitly. Use fresh memory
for old native sessions; ordinary message histories remain supported. Historical
native execution requires the prior Composer version to resume.

### Local models with Ollama

No API key is needed — point at a running Ollama endpoint:

```console
export AGENT_COMPOSER_DEFAULT_PROVIDER=ollama
export AGENT_COMPOSER_DEFAULT_MODEL=llama3.2:3b
export OLLAMA_BASE_URL=http://localhost:11434
composer run examples/hello.yaml --input name=Ada
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
`composer` does **not** auto-load a `.env` file — it only reads variables already
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
composer run examples/hello.yaml --input name=Ada
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

Now `cd`-ing into the directory activates `.venv` (so `composer`, `pytest`, `python`
resolve from it) and loads `.env`; leaving unloads them again. `.env` and
`.envrc` are gitignored — keep secrets out of version control.

## Next

- [The `composer` CLI](cli.md) — run flows from the terminal.
- [Flow syntax](syntax.md) — the Compose-YAML reference.
