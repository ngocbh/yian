# The yian conformance suite

Test cases that pin what a flow does, written as data: each case under `cases/<group>/<name>.yaml`
gives a flow (or an expression, a value, a CLI call, ...) and the results yian must produce. The
runner, `tests/conformance.ts`, turns each case into calls on yian and compares the results.

[`SPEC.md`](SPEC.md) defines the case format, what each kind checks, and the runner contract.

## Layout

```
conformance/
  README.md                       # this file
  SPEC.md                         # the case format and the runner contract
  cases/<group>/<name>.yaml       # one case per file; its id is <group>/<name>
  cases/<group>/<name>.log.jsonl  # optional golden run log
  fixtures/                       # files cases share: CODE helpers, golden profiles, stub data
  schema/                         # JSON Schemas of the documents yian gives hosts
```

## Running the suite

```bash
bun tests/conformance.ts                                  # every case, on Bun
bun tests/conformance.ts --group defaults --kind defaults # a selection
bun tests/conformance.ts --crash-sweep                    # also cut and resume each run case
bun run build && node dist/tests/conformance.js           # every case, on Node
```

See [Running the suite](SPEC.md#running-the-suite) for every flag, golden logs and
`--write-expect`.
