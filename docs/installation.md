# Installation

Yian needs **Python 3.11+**.

Install from PyPI:

```console
python -m pip install yian
```

The distribution, import name, and CLI command are all `yian`.

```console
yian --help
```

Changed (TS yian): TS yian is the npm package `@ngocbh/yian`, whose CLI command is also
`yian`. It needs Node 22.18 or later, or Bun 1.4 or later:

```console
npm install @ngocbh/yian
npx @ngocbh/yian --help
```

Under Bun, the package's entry points resolve to its TypeScript sources. The MCP SDK is an
optional peer dependency (see [MCP servers](api.md#mcp-servers)). There are also single-file
binaries that need neither Node nor Bun: `yian-darwin-arm64`, `yian-darwin-x64`,
`yian-linux-x64` and `yian-linux-arm64`, built with `bun run build:bin` in `ts/`. They bundle the
MCP SDK, and import a CODE node's referenced file from disk when it runs. `yian --licenses`
prints yian's notice, Bun's (MIT; it statically links JavaScriptCore, under the LGPL-2) and the
licenses of the packages the binaries bundle.

## Provider extras

The core (engine + `yian` CLI) installs with **no LLM SDK**. Each provider is an
optional extra — install the one(s) you actually use:

```console
python -m pip install "yian[anthropic]"   # Claude  (langchain-anthropic)
python -m pip install "yian[openai]"      # GPT     (langchain-openai)
python -m pip install "yian[google]"      # Gemini  (langchain-google-genai)
python -m pip install "yian[ollama]"      # local   (langchain-ollama)
python -m pip install "yian[all]"         # all provider extras and mcp
```

Provider SDKs are imported **lazily** — only when a flow actually runs an AGENT node
against that provider. Importing a provider you haven't installed raises a clear
`pip install yian[...]` hint rather than a bare `ImportError`.

## MCP extra

The `mcp` extra installs the official MCP SDK (`mcp>=2.3,<3`), which yian needs to
call the tools of MCP servers (see [MCP servers](api.md#mcp-servers) and
[the CLI's MCP servers](cli.md#mcp-servers)):

```console
python -m pip install "yian[mcp]"
```

It is imported only when a server opens; opening one without it raises an
`ImportError` that names the extra. `yian[all]` includes it.

## Choosing a provider and model

There is no built-in provider or model. Each AGENT node takes its provider and
model from the first of these that sets them:

1. the flow's `llm_config` (the node, then its flow, then each parent flow); a field
   there may come from a `${...}` value, such as `model: ${input.model}`, and one
   that resolves to null is unset (see
   [Values in `llm_config`](syntax.md#values-in-llm_config));
2. `yian run --provider/--model`, or the `llm_config` a Python host passes to
   `run_flow`;
3. the environment: `YIAN_DEFAULT_PROVIDER` and `YIAN_DEFAULT_MODEL`;
4. the saved login from `yian auth <provider>`.

Save a login once for each provider you use:

```console
yian auth openai
yian auth anthropic --model claude-sonnet-4-5
yian auth azure --endpoint https://<resource>.openai.azure.com/ --model <deployment>
yian auth ollama --model llama3.2:3b
```

A provider you set up becomes the default when no default is saved yet. See
[Saved logins](#saved-logins-yian-auth) for listing, switching, and removing them.

Environment variables override the saved default provider and model:

```console
export YIAN_DEFAULT_PROVIDER=anthropic        # or openai / google / ollama / ...
export YIAN_DEFAULT_MODEL=claude-sonnet-4-5
export ANTHROPIC_API_KEY=...                  # the provider's own key variable
```

A default model belongs to its provider. `YIAN_DEFAULT_MODEL` applies to the
default provider: `YIAN_DEFAULT_PROVIDER`, else the saved default. With no default
provider it is ignored. Each saved login also supplies the model for its own
provider, so a flow that sets `provider: anthropic` without a model uses the saved
Anthropic model. Selecting a provider never borrows another provider's model.

If no layer gives an agent a provider and model, the run fails before any node runs
and before the log file is created. The error names the agent and the fixes:

```text
Agent 'write' has no provider or model; set llm_config.provider/model, pass --provider/--model, or run 'yian auth <provider>'
```

An agent with a provider but no model gets `Agent 'write' has no model for provider
'anthropic'; ...`, ending in `run 'yian auth anthropic'`. When `YIAN_DEFAULT_MODEL`
was ignored because no default provider is set, the error says so. A
`YIAN_DEFAULT_PROVIDER` that names no supported provider fails the run before its
first node, naming the variable.

An agent's credential comes from, highest first: a Python host's
`llm_config.api_key`, the provider's saved login, then the provider's key variable
(`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, ...). Whatever its source, a credential
that a provider echoes in an error is replaced with `[REDACTED]` before the failure
is recorded or printed. A request carries only one credential: yian does not send
the headers in `OPENAI_CUSTOM_HEADERS` or `ANTHROPIC_CUSTOM_HEADERS`, or the
organization and project from `OPENAI_ORG_ID`, `OPENAI_ORGANIZATION`, and
`OPENAI_PROJECT_ID`, so an OpenAI organization or project can no longer be chosen
through them. This is a breaking change: earlier versions sent them. yian warns
once when it drops a header from these variables; unset the variable to silence it.
Send them as [connection settings](#connection-settings) next to the key instead:
saved with the login by `yian auth`, or beside a Python host's `llm_config.api_key`.
yian never sends `OLLAMA_API_KEY` either (see [Local models with Ollama](#local-models-with-ollama)).
A key yian passes to Azure (a Python host's key or a saved login) is sent even when
`AZURE_OPENAI_AD_TOKEN` is set. Endpoints still come from
environment variables at run time (`OPENAI_BASE_URL`, `ANTHROPIC_BASE_URL`,
`GOOGLE_GEMINI_BASE_URL`, `VLLM_BASE_URL`, `OLLAMA_BASE_URL`, ...); a saved login
stores only the key and its connection settings, plus the endpoint for Azure.
`GOOGLE_GENAI_USE_VERTEXAI` and `GOOGLE_GENAI_USE_ENTERPRISE` do not change the
endpoint: the `google` provider
always calls the Gemini Developer API with the key yian resolved, and yian warns
once when either is set. Vertex AI is not supported, so a setup that relied on
Vertex AI Application Default Credentials without an API key now fails with a
missing-key error. This is a breaking change: earlier versions honored the
variable. Azure's API version always comes from `OPENAI_API_VERSION`.

A Python host can instead pass the key in `run_flow(..., llm_config={"provider": ...,
"api_key": key})`, which takes precedence over the saved login and the variable and
is never recorded.
Pass it again to `resume_flow` and `RunResult.backward` with `api_key=key`; agents
marked for the key never fall back to the variable. Add `"base_url": url` beside the
key to send it to another endpoint, such as an Azure OpenAI resource's
`https://<resource>.openai.azure.com/openai/v1/` with `"provider": "openai"` and the
deployment name as `model`; no Azure environment variables are needed. Without
`base_url`, the provider's default endpoint or its environment variables (such as
`OPENAI_BASE_URL`) decide where the key goes, so pass `base_url` with the key when
the endpoint must be fixed.

Changed (TS yian): for model calls and saved logins, yian reads only the variables it
documents, through one environment accessor: the provider key variables,
`GOOGLE_API_KEY` then `GEMINI_API_KEY`, `AZURE_OPENAI_API_KEY` and
`AZURE_OPENAI_AD_TOKEN`, `OPENAI_API_VERSION`, the base URL variables
(`OPENAI_BASE_URL`, `ANTHROPIC_BASE_URL`, `VLLM_BASE_URL`, `OLLAMA_BASE_URL`),
`YIAN_DEFAULT_PROVIDER`, `YIAN_DEFAULT_MODEL`, `YIAN_AUTH_FILE`, `XDG_CONFIG_HOME`,
`HOME` and the `YIAN_*` defaults. An empty variable counts as unset, `YIAN_AUTH_FILE`
too. `GOOGLE_GEMINI_BASE_URL` is not read. `ANTHROPIC_AUTH_TOKEN`, `OLLAMA_API_KEY`,
`OPENAI_CUSTOM_HEADERS`, `ANTHROPIC_CUSTOM_HEADERS`, the OpenAI organization and
project variables, and `GOOGLE_GENAI_USE_VERTEXAI` and `GOOGLE_GENAI_USE_ENTERPRISE`
are not read and draw no warning. Azure doesn't fall back to `OPENAI_API_KEY`. Errors
are scrubbed of a saved subscription's account id too.
{: #ts-environment }

Changed (TS yian): reasoning that a provider binds to an account goes back to it only from
the same account. Each replay record carries a short hash of the API key sent (OpenAI,
Anthropic, a Responses-routed deployment), of the ChatGPT account id, or of the Azure
resource on an AD token. So after a key is rotated, earlier reasoning from the old key is
left out of the next request, and a `replay_degraded` event says why.

When you `yian run` a flow whose AGENT nodes need a provider key that isn't set, the CLI
prompts for it up front (in an interactive terminal) and continues once entered —
so a missing key surfaces as a prompt rather than a failure on the first model
call. Off a terminal (CI, a pipe) a missing required key is a clean error before
the run starts. Ollama is keyless, and vLLM needs `VLLM_API_KEY` only when its
server was started with a key. A saved login is checked through its stored
credential, so it never prompts.

### Saved logins (`yian auth`) {#saved-logins-yian-auth}

`yian auth` keeps one login per provider and an optional default provider:

```console
yian auth                       # list saved logins and the default
yian auth PROVIDER [options]    # set up PROVIDER: credential and model
yian auth PROVIDER --default    # make a saved provider the default
yian auth PROVIDER --remove     # delete PROVIDER's saved login
yian auth --remove --auth-file F  # delete the whole file F
```

The list shows each provider, its kind (`api-key`, `subscription`, or `none`), its
model, an Azure endpoint, its [connection settings](#connection-settings) (an
OpenAI organization and project, and the names of its headers), and which one is
the default. It notes when `YIAN_DEFAULT_PROVIDER` or `YIAN_DEFAULT_MODEL` overrides
the saved choice. It never prints a key, token, or header value.

Setup saves the provider's credential and its model together:

- An API-key provider reads its key variable (such as `ANTHROPIC_API_KEY`), or the
  variable named by `--api-key-env NAME`, or asks for the key with a hidden prompt.
- OpenAI can also use a ChatGPT subscription; see [OpenAI](#openai-authentication).
- Azure takes the resource endpoint with `--endpoint URL` and a key.
- Ollama saves no credential. vLLM saves its key only if you give one; without a
  key the login's kind is `none`, and runs need no `VLLM_API_KEY`.
- `--organization ID` and `--project ID` (OpenAI API key only) and
  `--header NAME=VARIABLE` (any provider given a key) save
  [connection settings](#connection-settings) with the key.

`--model M` names the model; in a terminal, setup asks when it is omitted. The model
is accepted when the provider's model listing has it. Otherwise, such as an alias the
listing does not show, or Azure, which has no listing, one short request to the
model must succeed. With a base-URL variable such as `ANTHROPIC_BASE_URL` set, setup
skips the listing, so the key never goes to the public host, and checks the model with
one request. OpenAI accepts only a model in the account's catalog. A failed or
cancelled setup saves nothing and leaves the previous login in place.

Setting up a provider makes it the default only when no default is saved yet.
`yian auth PROVIDER --default` switches the default to another saved provider
without signing in again. `yian auth PROVIDER --remove` deletes that provider's login;
removing the default leaves no default provider.

Credentials belong to Yian's private user-local store; keys and tokens never belong
in flow YAML or run logs. A saved login's credential takes precedence over the
provider's key variable, so a saved OpenAI subscription wins over an incidental
`OPENAI_API_KEY`. Each run records which connection every agent of a keyed provider
uses: a saved login, by its credential identity, or the provider's key variable.
It never records the key. Re-running `yian auth PROVIDER` or removing the login makes
an unfinished run that used it fail on resume, naming `yian auth PROVIDER`; it
cannot silently switch accounts or billing methods. Ollama agents record no
connection.

The store is `$XDG_CONFIG_HOME/yian/auth.json`, falling back to
`~/.config/yian/auth.json`. Set `YIAN_AUTH_FILE` to choose another private file, or
give any `yian auth` form `--auth-file F` to use the file `F` (relative to the working
directory). `yian auth --remove --auth-file F` deletes the whole file, keeping its
lock file, and succeeds when `F` doesn't exist. The store uses private filesystem
permissions (mode `0600`) and atomic writes on POSIX hosts; it is not encrypted.
Token refresh preserves the selected connection. Failed authentication never falls
back to another credential.

Changed (TS yian): the file keeps the same format, so TS yian uses logins saved by
Python yian. An `expires_at` that is a whole number of seconds is written without
`.0` (`1767225600`, where Python writes `1767225600.0`); both read either. TS yian
locks `auth.json.lock` (the file's path plus `.lock`) with its own lock, which
Python's doesn't see: don't run Python and TS yian against the same saved logins at
the same time (a refresh rotates the token); point `YIAN_AUTH_FILE` at a copy
instead. The path expands only `~` and a leading `~/`, not `~user`, and an empty
`YIAN_AUTH_FILE` counts as unset.

Changed (TS yian): model calls in one process that find the same subscription token
expiring share one refresh instead of each waiting for the file lock. The browser
sign-in answers each connection to its local callback on its own, so a stray idle
connection can't hold up the real one.

#### OpenAI {#openai-authentication}

```console
yian auth openai
```

Choose ChatGPT subscription sign-in (browser or device login for SSH) or an
OpenAI API key, then select a model from the account's catalog. The `openai`
extra supports both methods. Selecting an API key uses separately billed API
access.
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
terminal. `--api-key-env NAME` selects another environment variable.

Migrate native flows to `provider: openai` and remove `sandbox`, `cwd`, and
`builtin_tools`. Grant registered Yian tools explicitly. Use fresh memory
for old native sessions; ordinary message histories remain supported. Historical
native execution requires the prior Yian version to resume.

### Connection settings {#connection-settings}

Some keys need more than the key: an OpenAI key in several organizations or
projects, or a gateway in front of a provider that wants its own headers. These
travel with the key as connection settings:

- `organization` and `project` are the OpenAI organization and project ids, sent as
  the `OpenAI-Organization` and `OpenAI-Project` headers. Only `provider: openai`
  with an API key takes them, not a ChatGPT subscription. Each is non-empty visible
  ASCII of at most 256 characters (`YIAN_MAX_ORGANIZATION_CHARS`).
- `headers` are extra request headers, such as a gateway token, for every provider
  yian sends a key to: not Ollama, a keyless vLLM server, or an OpenAI subscription.
  Names are HTTP tokens compared without case, so two spellings of one name are
  refused. Names yian sets itself are refused: the credential headers
  `Authorization`, `Proxy-Authorization`, `x-api-key`, `api-key`, and
  `x-goog-api-key`, the framing headers `Host`, `Content-Length`,
  `Transfer-Encoding`, `Connection`, and `Content-Type`, and for `provider: openai`
  also `OpenAI-Organization` and `OpenAI-Project` (use `organization` and `project`).
  Values are visible ASCII, with spaces or tabs only between characters. A
  connection sends at most 32 headers (`YIAN_MAX_CONNECTION_HEADERS`), and a name or
  value is at most 8192 characters (`YIAN_MAX_HEADER_VALUE_CHARS`).

The settings come with the key, from one place, never mixed:

- A Python host passes them beside `api_key` and `provider` in the run's
  `llm_config`, and they go to the agents that key goes to:

  ```python
  llm_config={
      "provider": "openai",
      "api_key": key,
      "organization": "org-...",
      "project": "proj_...",
      "headers": {"X-Gateway-Token": token},
  }
  ```

  The run records `organization`, `project`, and the header names on those agents,
  never the header values. Pass the values again with the key, as
  `resume_flow(..., api_key=key, headers={...})` and
  `result.backward(..., api_key=key, headers={...})` (see [the Python
  API](api.md#run_flow)).
- A saved login sends the settings `yian auth` saved with it. The run records them
  with the login's identity, header names only, and re-saving the login makes an
  unfinished run fail on resume, as it does for a new key.
- A key from the provider's variable, such as `OPENAI_API_KEY`, sends none.

A flow file cannot set them at any level, and no `${...}` value supplies them.
Header values are secrets: no run log, event, result, or manifest holds them, and a
value a provider echoes in an error is replaced with `[REDACTED]`, however short it
is. A model that sends any of these settings never follows a redirect, so neither the
key nor the headers reach another host.

To move off `OPENAI_ORG_ID`, `OPENAI_PROJECT_ID`, or the `*_CUSTOM_HEADERS`
variables, save the settings with the login:

```console
yian auth openai --method api-key --api-key-env OPENAI_API_KEY \
    --organization org-... --project proj_...
yian auth anthropic --header X-Gateway-Token=GATEWAY_TOKEN
```

`--header NAME=VARIABLE` reads the value from the environment variable `VARIABLE`
(`GATEWAY_TOKEN` above), and `--header NAME` asks for it with a hidden prompt; a
value never comes from the command line. `--header` repeats for more headers. The
model listing and the model check during setup send the settings too. Every save
states them again: `yian auth PROVIDER` without them saves a login with none.

Headers go only beside a key, so a keyless vLLM server (no `VLLM_API_KEY`) cannot
take them, and `yian auth vllm --header ...` needs a key. When a gateway in front of
vLLM wants a token, pass the token as the vLLM key instead (`VLLM_API_KEY`,
`--api-key-env`, or the host's `api_key`); yian sends it as
`Authorization: Bearer <token>`.

### Azure OpenAI

`provider: azure` takes everything that picks the deployment from `llm_config`:
`model` is the deployment name and `base_url` is the resource endpoint. The run
records both when it starts, so a resume calls the same deployment at the same
endpoint:

```yaml
llm_config:
  provider: azure
  model: gpt-5.4                                  # the deployment name
  base_url: https://<resource>.openai.azure.com/  # the resource endpoint
```

A saved Azure login fills whichever of `base_url` and `model` an Azure agent leaves
unset:

```console
export OPENAI_API_VERSION=<version>
yian auth azure --endpoint https://<resource>.openai.azure.com/ --model <deployment>
```

Its endpoint becomes the agent's `base_url` and its deployment the agent's `model`.
Its key goes only to that resource: an agent whose `base_url` names another resource
uses `AZURE_OPENAI_API_KEY`. Endpoints are compared after removing a trailing `/`
and an `/openai` or `/openai/v1` suffix. Such an agent still gets the saved
deployment as its `model` when it sets none, so it should set its own `model`.

A flow file may set `base_url` only beside `provider: azure`. An `/openai` or
`/openai/v1` suffix is stripped. An Azure agent that ends up without a `base_url`
fails before the run's first node, saying to set it. The API version always comes
from `OPENAI_API_VERSION`, for runs and for `yian auth azure`'s model check. Without
a saved login, the key stays in `AZURE_OPENAI_API_KEY` (the Azure SDK prefers
`AZURE_OPENAI_AD_TOKEN` when both are set); a Python host can pass the
key as `api_key` instead, and then must pass `base_url` beside it.
`AZURE_OPENAI_ENDPOINT` and `AZURE_OPENAI_DEPLOYMENT_NAME` are not read: move their
values into `base_url` and `model`, or save them with `yian auth azure`.

Changed (TS yian): there is no Azure SDK, so a call given no key never falls back to
`OPENAI_API_KEY`. `AZURE_OPENAI_AD_TOKEN` is scrubbed from errors whenever it is set; see
[the environment note](#ts-environment).

### Local models with Ollama

No API key is needed. Save Ollama as a login, or set the defaults in the
environment, and point at a running Ollama endpoint:

```console
yian auth ollama --model llama3.2:3b
# or: export YIAN_DEFAULT_PROVIDER=ollama YIAN_DEFAULT_MODEL=llama3.2:3b
export OLLAMA_BASE_URL=http://localhost:11434
yian run hello.yaml --input name=Ada
```

The endpoint always comes from `OLLAMA_BASE_URL` (default
`http://localhost:11434`); the saved login stores only the model.

yian sends no key to Ollama. It does not send `OLLAMA_API_KEY`, and warns once when
it drops it; unset the variable to silence the warning. So a host that needs a key,
such as Ollama's hosted API, can no longer be used, and nothing replaces it yet. This
is a breaking change: earlier versions sent the variable as a Bearer token. A user
name and password written in `OLLAMA_BASE_URL` are still sent as Basic auth.

Changed (TS yian): there is no SDK, so yian doesn't read `OLLAMA_API_KEY` and doesn't
warn about it; see [the environment note](#ts-environment). `OLLAMA_BASE_URL` is checked
when a model is built: an `http://` or `https://` URL of a host, an optional port and a
path (a scheme-less `host:port` gets `http://`, and a trailing `/v1` is dropped), without a
query or fragment. A value that doesn't fit fails the model instead of falling back to
another endpoint. A user name and password in it are sent as Basic auth and scrubbed from
errors. A config `api_key` for `ollama` is accepted and never sent.

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

### Proxies and certificates (TS yian) {#proxies-and-certificates}

Changed (TS yian): proxies, CA files and TLS checks are the runtime's settings, and
yian reads none of them (except `NODE_TLS_REJECT_UNAUTHORIZED`, for its warning below), so
they apply the same way to model calls, sign-in, model listing and MCP over HTTP.

- On Node, set `NODE_USE_ENV_PROXY=1` so `HTTPS_PROXY`, `HTTP_PROXY` and `NO_PROXY`
  apply (Node 22.21 or later; Node 22 prints an experimental warning). Bun reads
  those variables by itself, upper or lower case. `localhost` is not exempt unless
  `NO_PROXY` names it.
- Add a CA file with `NODE_EXTRA_CA_CERTS=/path/to/ca.pem`, on Node and Bun.
  `SSL_CERT_FILE` and `SSL_CERT_DIR` are not portable: official Node builds ignore
  them.
- `NODE_TLS_REJECT_UNAUTHORIZED=0` turns certificate checks off on both runtimes. yian
  reads it only to warn once per process that every request is unverified (Node prints
  its own warning as well).
- No request follows a redirect, and a URL holding a user name or password is
  refused without being echoed, so credentials can't travel in a base URL
  (`OLLAMA_BASE_URL` is the exception: its user name and password become Basic auth).

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
