---
title: "Models and providers"
description: "Log in, pick a provider and model, and set connection options."
---

This page explains how yian reaches a model: saving a login, choosing a provider and model for
each agent, and setting request and connection options. yian has no built-in provider or
model.

## Log in once

Save a login for each provider you use. `yian auth` checks the model and stores the credential
and model together:

```console
$ yian auth anthropic --model claude-sonnet-4-5
$ yian auth openai
$ yian auth ollama --model llama3.2:3b
```

When no default is saved, the provider you save becomes the default. In a terminal, `yian auth` asks for anything
you leave out; for an API key it reads the provider's key variable or asks with a hidden prompt.

| Command | What it does |
|---|---|
| `yian auth` | List the saved logins and the default; never prints a secret |
| `yian auth PROVIDER` | Save or replace that provider's login |
| `yian auth PROVIDER --default` | Make a saved provider the default |
| `yian auth PROVIDER --remove` | Delete that provider's login |
| `yian providers` | List every provider, the login kinds it takes and its key variables |
| `yian models --provider P` | List the models P offers to the key read from stdin (not for Azure) |

Logins live in `~/.config/yian/auth.json` (or `$XDG_CONFIG_HOME/yian/auth.json`), a file only
you can read. Set `YIAN_AUTH_FILE`, or pass `--auth-file F`, to use another file. Its folder must
be yours too: mode `0700` and not a symbolic link. yian creates a missing folder that way. The
[CLI](/cli/) page lists every flag.

## Providers

| Provider | Key variable | Login kinds |
|---|---|---|
| `openai` | `OPENAI_API_KEY` | API key, or a ChatGPT subscription |
| `anthropic` | `ANTHROPIC_API_KEY` | API key |
| `google` | `GOOGLE_API_KEY`, else `GEMINI_API_KEY` | API key |
| `azure` | `AZURE_OPENAI_API_KEY` | API key |
| `xai`, `deepseek`, `openrouter` | `XAI_API_KEY`, `DEEPSEEK_API_KEY`, `OPENROUTER_API_KEY` | API key |
| `qwen`, `qwen-cn` | `DASHSCOPE_API_KEY`, `DASHSCOPE_CN_API_KEY` | API key |
| `glm`, `glm-cn` | `ZHIPU_API_KEY`, `ZHIPU_CN_API_KEY` | API key |
| `minimax`, `minimax-cn` | `MINIMAX_API_KEY`, `MINIMAX_CN_API_KEY` | API key |
| `vllm` | `VLLM_API_KEY` (optional) | API key, or none |
| `ollama` | none | none |

The `-cn` providers are the China-region services; their keys don't work on the international
ones. For Google, runs, `yian run` and `yian auth` read `GEMINI_API_KEY` when `GOOGLE_API_KEY` is
unset, and ask for `GOOGLE_API_KEY` when neither is set. An agent's key comes from, highest
first: a key the host passes, the saved login, then the key variable. `yian run` asks for a
missing key before the run starts when it can prompt, and fails early when it can't.

## Choosing a provider and model

Each agent picks its provider, model and request settings field by field. A field takes the
first value set in this order:

1. the agent's own `llm_config:`;
2. the enclosing flow's top-level `llm_config:`, then each parent flow's, outward;
3. `yian run --provider/--model`, or the `llm_config` a host passes to the run;
4. the provider: `YIAN_DEFAULT_PROVIDER`, else the saved default;
5. the model: `YIAN_DEFAULT_MODEL` for the default provider only, else the provider's saved model.

<!-- yian-test {"replies": ["Ship it.", "Approved."]} -->

```yaml title="review.yaml" check=runnable
id: review
name: review
llm_config:
  provider: anthropic
  model: claude-sonnet-4-5
input:
  change: !str
nodes:
  draft:
    kind: agent
    input:
      change: ${input.change}
    prompt: Write a one-line review of ${change}.
  grade:
    kind: agent
    input:
      review: ${draft.output}
    llm_config:
      provider: openai
      model: gpt-5.5
      inherit: false
    prompt: Approve or reject this review. ${review}
output: ${grade.output}
```

```console
$ yian run review.yaml --input change="Fix the login timeout"
Approved.
```

- `draft` takes both fields from the flow. `grade` sets its own, and `inherit: false` makes it
  skip the flow, parent flows and CLI layers; only its own config and the defaults apply.
  `inherit:` goes only in an agent's `llm_config:`, written as `true` or `false`.
- `--provider` and `--model` fill gaps only. They never override a value the flow sets.
- Each field resolves on its own, except that an agent (or flow) that sets `provider:` but no
  `model:` drops a model an outer layer chose for another provider. A model from outer layers
  that name no provider, such as `--model` alone, is kept. With no model left, it gets its
  provider's saved model.
- An agent that ends up with no provider or model fails the run before any node runs:

```text
Agent 'draft' has no provider or model; set llm_config.provider/model, pass --provider/--model, or run 'yian auth <provider>'
```

A field may also come from a value, such as `model: ${input.model}`; a value that resolves to
null leaves the field unset. A value in a flow's `llm_config:` that an agent in a child flow
would inherit must read the root flow's input, or the load fails; pass it to the child as an
input instead. Every `llm_config:` is checked at load: an unknown key, an unknown
provider or a bad setting fails there.

## Request settings

Everything about the request other than the provider and model goes in a block named after a
provider. A block mirrors the request body of the API yian calls, with the same names and
values, and yian sends it as written. An agent uses only the block of the provider it resolves
to, so one flow can carry settings for several providers.

```yaml check=fragment
llm_config:
  provider: openai
  model: gpt-5.5
  openai:
    reasoning: {effort: high}
    max_output_tokens: 4096
  anthropic:
    max_tokens: 16000
    thinking: {type: adaptive}
```

| Block | API it mirrors |
|---|---|
| `openai` | OpenAI Responses |
| `anthropic` | Anthropic Messages |
| `google` | Gemini `generateContent` (camelCase keys, such as `generationConfig`) |
| `ollama` | Ollama `/api/chat` (`think`, `options`, `keep_alive`, ...) |
| `azure`, `xai`, `deepseek`, `qwen`, `glm`, `minimax`, `openrouter`, `vllm` and the `-cn` ones | that provider's Chat Completions |

- A node's block and its flow's block of the same name merge field by field; the node wins.
  A field's value is replaced whole.
- Fields yian builds itself, such as the messages, tools, the output format and streaming, can't
  be set. Setting one fails at load and says why.
- `openai: {store: false}` makes the agent's requests stateless, so OpenAI stores no response.
  yian then sends the system prompt as `instructions`, asks for `reasoning.encrypted_content`,
  and replays the model's reasoning from that encrypted content, which the run log records. A
  resumed run sends the same requests. Without `store`, OpenAI's default stores responses.
- On Azure and the Chat Completions providers, a model whose name starts with `gpt-5-pro`,
  `gpt-5.2-pro`, `gpt-5.4-pro`, `gpt-5.5-pro` or `gpt-5.6-sol`, or contains `codex`, is called
  through the Responses API; its block must then be empty. A block there fails at load, or before
  the agent runs when the provider or model is set elsewhere. `yian config check` refuses a layer
  that sets both the provider and such a model with a block, as if no flow or agent overrides them.
- A setting one model doesn't accept passes the load check; the provider's error then fails the
  node.

## Connections

A connection is where requests go and the credentials sent with them. None of it goes in a flow
file: `api_key`, `organization`, `project`
and `headers` in a flow's `llm_config:` fail at load, and so does `base_url` outside Azure.

| Setting | How to set it |
|---|---|
| OpenAI organization and project | `yian auth openai --method api-key --organization ID --project ID` |
| Extra headers, such as a gateway token | `yian auth PROVIDER --header NAME=VARIABLE` (the value is read from that variable) |
| Another endpoint | `OPENAI_BASE_URL`, `ANTHROPIC_BASE_URL`, `VLLM_BASE_URL` or `OLLAMA_BASE_URL`, or a host's `base_url` beside its key |

Header values are secrets: run logs record only the header names, and a value a provider echoes
in an error is replaced with `[REDACTED]`. Each save replaces these settings, so
`yian auth PROVIDER` without them saves a login with none.

**Azure.** `model` is the deployment name and `base_url` the resource endpoint. Both may go in
a flow. `OPENAI_API_VERSION` must hold the API version, for runs and for `yian auth azure`:

```yaml check=fragment
llm_config:
  provider: azure
  model: my-deployment
  base_url: https://my-resource.openai.azure.com/
```

`yian auth azure --endpoint URL --model DEPLOYMENT` saves both, and fills them for Azure agents
that leave them unset. Without a key from the host or a saved login, a run uses
`AZURE_OPENAI_AD_TOKEN` (a bearer token), then `AZURE_OPENAI_API_KEY`.

**Local models.** Ollama takes no key and is called at `OLLAMA_BASE_URL` (default
`http://localhost:11434`). vLLM is called at `VLLM_BASE_URL` (default `http://localhost:8000/v1`)
and needs `VLLM_API_KEY` only when its server was started with a key.

## ChatGPT subscription

`provider: openai` can run on a ChatGPT subscription instead of a billed API key. Sign in with a
browser, or with a device code when you work over SSH:

```console
$ yian auth openai --method subscription --login device --model gpt-5.5
```

The login refreshes its token by itself and wins over an `OPENAI_API_KEY` in the environment.
Its requests are always stateless, as with `store: false`, so the `openai` block can't set
`store`; the subscription endpoint also refuses `temperature` and `max_output_tokens`. A run that
sets any of them for such an agent fails before its first node. List the models the
subscription offers with `yian models --auth-file F`; `F` is the auth file, even the default one.

A run records which login each agent used, never the key. Saving the same login again, even
with another model, keeps it. If you replace that login (another key, header, endpoint,
organization or project, or a new subscription sign-in) or remove it, an unfinished run's resume
fails and names `yian auth PROVIDER`, rather than switching accounts silently. Until then, a run
keeps using an API key it has read; on a subscription it fails at its next model call.
