# Installation

Yian needs **Python 3.10+**.

Install from PyPI:

```console
python -m pip install yian
```

The distribution, import name, and CLI command are all `yian`.

```console
yian --help
```

## Provider extras

The core (engine + `yian` CLI) installs with **no LLM SDK**. Each provider is an
optional extra — install the one(s) you actually use:

```console
python -m pip install "yian[anthropic]"   # Claude  (langchain-anthropic)
python -m pip install "yian[openai]"      # GPT     (langchain-openai)
python -m pip install "yian[google]"      # Gemini  (langchain-google-genai)
python -m pip install "yian[ollama]"      # local   (langchain-ollama)
python -m pip install "yian[all]"         # all provider extras
```

Provider SDKs are imported **lazily** — only when a flow actually runs an AGENT node
against that provider. Importing a provider you haven't installed raises a clear
`pip install yian[...]` hint rather than a bare `ImportError`.

## Choosing a provider and model

OpenAI is the default provider. Run `yian auth openai` to choose authentication
and a model. Environment variables can override the saved provider/model
defaults; other providers use their own credential variables:

```console
export YIAN_DEFAULT_PROVIDER=anthropic        # or openai / google / ollama
export YIAN_DEFAULT_MODEL=claude-sonnet-4-5
export ANTHROPIC_API_KEY=...                            # the provider's own key var
```

A `YIAN_DEFAULT_PROVIDER` that names no supported provider fails the run before
its first node, naming the variable.

A Python host can instead pass the key in `run_flow(..., llm_config={"provider": ...,
"api_key": key})`, which takes precedence over the variable and is never recorded.
Pass it again to `resume_flow` and `RunResult.backward` with `api_key=key`; agents
marked for the key never fall back to the variable. Add `"base_url": url` beside the
key to send it to another endpoint, such as an Azure OpenAI resource's
`https://<resource>.openai.azure.com/openai/v1/` with `"provider": "openai"` and the
deployment name as `model`; no Azure environment variables are needed. Without
`base_url`, the provider's default endpoint or its environment variables (such as
`OPENAI_BASE_URL`) decide where the key goes, so pass `base_url` with the key when
the endpoint must be fixed.

A flow can also pin its own provider and model per AGENT node. Defaults fill
only fields left unset by the flow or CLI. A default model belongs to its
provider: selecting another provider never borrows the OpenAI model. Without
a matching saved or environment model, OpenAI uses `gpt-5.5` and Anthropic uses
`claude-sonnet-4-5`; other providers require an explicit model.

When you `yian run` a flow whose AGENT nodes need a provider key that isn't set, the CLI
prompts for it up front (in an interactive terminal) and continues once entered —
so a missing key surfaces as a prompt rather than a failure on the first model
call. Off a terminal (CI, a pipe) a missing required key is a clean error before
the run starts. Ollama is keyless; saved OpenAI authentication is checked through
the selected connection method.

### OpenAI authentication

```console
yian auth openai
```

Choose ChatGPT subscription sign-in (browser or device login for SSH) or an
OpenAI API key, then select a model. Successful setup saves the method and model
as the default together. Cancelled or failed setup leaves the previous selection
intact. The `openai` extra supports both methods.
Subscription-backed nodes are checked at run start against the ChatGPT
subscription endpoint's schema: the `openai` block of the Responses API without
`temperature` and `max_output_tokens`. A run that sets either for such an agent
fails before its first node, naming the agent.

For SSH, select device login. Flags can supply choices without interactive
menus; subscription login still requires completing sign-in in a browser:

```console
yian auth openai --method subscription --login device --model gpt-5.5
yian auth openai --method api-key --model gpt-5.5
```

API-key setup reads `OPENAI_API_KEY`, or prompts without echoing the key in a
terminal. `--api-key-env NAME` selects another environment variable. Model
choices come from the selected account's authenticated catalog.

Credentials belong to Yian's private user-local store; tokens never belong
in flow YAML or run logs. Subscription selection takes precedence over an
incidental `OPENAI_API_KEY`. Selecting an API key uses separately billed API
access. Reconnecting changes the connection identity; an unfinished run cannot
silently switch accounts or billing methods.

The store is `$XDG_CONFIG_HOME/yian/auth.json`, falling back to
`~/.config/yian/auth.json`. Set `YIAN_AUTH_FILE` to choose
another private file. The store uses private filesystem permissions and atomic
writes on POSIX hosts; it is not encrypted. Token refresh preserves the selected connection.
Failed authentication never falls back to another billing method.

Migrate native flows to `provider: openai` and remove `sandbox`, `cwd`, and
`builtin_tools`. Grant registered Yian tools explicitly. Use fresh memory
for old native sessions; ordinary message histories remain supported. Historical
native execution requires the prior Yian version to resume.

### Azure OpenAI

`provider: azure` takes everything that picks the deployment from `llm_config`:
`model` is the deployment name and `base_url` is the resource endpoint. Neither has
a default, and the run records both when it starts, so a resume calls the same
deployment at the same endpoint:

```yaml
llm_config:
  provider: azure
  model: gpt-5.4                                  # the deployment name
  base_url: https://<resource>.openai.azure.com/  # the resource endpoint
```

A flow file may set `base_url` only beside `provider: azure`. An `/openai` or
`/openai/v1` suffix is stripped. An Azure agent without a `base_url` fails before the
run's first node, saying to set it. The key and API version stay in the environment
(`AZURE_OPENAI_API_KEY`, `OPENAI_API_VERSION`); a Python host can pass the key as
`api_key` instead, and then must pass `base_url` beside it. `AZURE_OPENAI_ENDPOINT`
and `AZURE_OPENAI_DEPLOYMENT_NAME` are not read: move their values into `base_url`
and `model`.

### Local models with Ollama

No API key is needed — point at a running Ollama endpoint:

```console
export YIAN_DEFAULT_PROVIDER=ollama
export YIAN_DEFAULT_MODEL=llama3.2:3b
export OLLAMA_BASE_URL=http://localhost:11434
yian run hello.yaml --input name=Ada
```

`hello.yaml` is any flow file, for example this one-node flow:

```yaml
id: hello
name: hello
input:
  name: !str
output: str
kind: agent
prompt: Write a warm one-sentence greeting to ${name}.
```

## Development install

The documentation describes the `main` branch, which can be ahead of the latest
release. To run `main` or work on Yian itself, install a clone editable with the
test extras:

```console
git clone https://github.com/ngocbh/yian
cd yian
pip install -e ".[all,dev]"
pytest
```

## Local development environment

The provider/model/key variables above are read from the **shell environment**.
`yian` does **not** auto-load a `.env` file — it only reads variables already
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
yian run examples/hello.yaml --input name=Ada
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

Now `cd`-ing into the directory activates `.venv` (so `yian`, `pytest`, `python`
resolve from it) and loads `.env`; leaving unloads them again. `.env` and
`.envrc` are gitignored — keep secrets out of version control.

## Next

- [The `yian` CLI](cli.md) — run flows from the terminal.
- [Flow syntax](syntax.md) — the Compose-YAML reference.
