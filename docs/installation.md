# Installation

Agent Composer needs **Python 3.10+**.

```console
pip install agent-composer
```

The distribution name on PyPI is `agent-composer`; the import name is
`agent_composer`; the CLI is `ac`.

```console
ac --help
```

## Provider extras

The core (engine + `ac` CLI) installs with **no LLM SDK**. Each provider is an
optional extra — install the one(s) you actually use:

```console
pip install "agent-composer[anthropic]"   # Claude  (langchain-anthropic)
pip install "agent-composer[openai]"      # GPT     (langchain-openai)
pip install "agent-composer[google]"      # Gemini  (langchain-google-genai)
pip install "agent-composer[ollama]"      # local   (langchain-ollama)
pip install "agent-composer[all]"         # all four
```

Provider SDKs are imported **lazily** — only when a flow actually runs an AGENT
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
the run starts; keyless providers like Ollama are skipped.

### Local models with Ollama

No API key is needed — point at a running Ollama endpoint:

```console
export AGENT_COMPOSER_DEFAULT_PROVIDER=ollama
export AGENT_COMPOSER_DEFAULT_MODEL=llama3.2:3b
export OLLAMA_BASE_URL=http://localhost:11434
ac run examples/hello.yaml --input name=Ada
```

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
