# Conformance cases

A conformance case pins one thing yian does, written as data rather than code. Each case is a
YAML file, `cases/<group>/<name>.yaml`, whose id is `<group>/<name>`. It gives an input (a flow,
an expression, a value, a CLI call, ...) and under `expect` the results yian must produce: load
diagnostics, the run's output, its run log projected to a trace, events, and so on. The runner,
`tests/conformance/runner.ts`, turns each case into calls on yian and compares the results.

[`SPEC.md`](SPEC.md) defines the case format: the keys of each kind, the host as data, the
trace projection, placeholders and error classes.

```yaml
kind: run
flow: |
  id: code-greeting
  name: code_greeting
  input:
    name: !str
  nodes:
    greet:
      kind: code
      input:
        name: ${input.name}
      output: !str
      code: |
        return "hi " + inputs.name
  output: ${greet.output}
inputs: {name: Ada}
run: {run_id: r1, num_workers: 0}
expect:
  output: {status: succeeded, output: hi Ada}
  trace: [...]
```

## Running cases

```bash
bun tests/conformance/runner.ts                                  # every case
bun tests/conformance/runner.ts --case run-core/code_greeting    # one case
bun tests/conformance/runner.ts --group defaults --kind defaults # a selection
bun tests/conformance/runner.ts --crash-sweep                    # also cut and resume each run case
```

The runner prints a `FAIL` line for each failing case and exits 1 when any case fails.

## Adding a case

1. Pick the group whose subject the case covers (SPEC.md's Groups table) and write
   `cases/<group>/<name>.yaml`. Give it `kind`, its input, and the `expect` results you know,
   such as `output` or `raises`.
2. For a `run` or `resume` case, add the projections you want pinned as empty lists
   (`trace: []`, `events: []`), then fill them in from a real run:
   `bun tests/conformance/runner.ts --write-expect --case <group>/<name>`.
3. Read the written values and check each one is what yian should do, then run the case again.

Files several cases share (CODE helpers, stub data) go in `fixtures/`; a case's flow names them
as `./fixtures/<file>`.
