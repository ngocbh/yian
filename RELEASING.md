# Releasing

yian is published to npm as `@ngocbh/yian`. Run network steps with the proxy variables unset.

**By hand** (every release until Actions billing is fixed, and the first release, since npm sets up
a trusted publisher only on an existing package):

1. Merge a PR that sets `version` in `package.json` and `VERSION` in `src/platform.ts`.
2. In a fresh clone at the merge commit, with Node 22.18 or newer on `PATH` (CI tests the 22.18
   floor): `bun install --frozen-lockfile`, then
   `bun run typecheck && bun run build && bun test ./tests && bun run test:node && bun tests/conformance/runner.ts`
   and `YIAN_TEST_PACK=1 bun test tests/repo/packaging.test.ts -t "npm tarball"`.
3. `npm login` (account `ngocbh`, 2FA on), then `npm publish --access public --dry-run`: check the
   file list has `README.md`, `LICENSE`, `NOTICE` and no `tests/`.
4. `npm publish --access public` (it asks for the one-time password).
5. `gh release create v<version> --target <merge sha> --title "yian <version>"`. Once Actions
   run, this starts `npm-publish.yml`, which fails harmlessly: the version already exists.

**Trusted publishing** (once, after the first release): on npmjs.com, `@ngocbh/yian` → Settings →
Trusted Publisher → GitHub Actions: user `ngocbh`, repository `yian`, workflow `npm-publish.yml`,
environment `npm`. Set publishing access to "Require two-factor authentication and disallow
tokens". On GitHub, create the environment `npm` (optionally with a reviewer). Then publishing a
GitHub Release runs `npm-publish.yml` with no stored token; running it by hand is a dry run. If
the `npm` environment is limited to `v*` tags, dry runs dispatched from main won't start.

No provenance while the repo is private; once public, restore `repository` and npm adds it.
