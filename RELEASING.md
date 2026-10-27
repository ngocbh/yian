# Releasing

yian is published to npm as `@ngocbh/yian`. Run network steps with the proxy variables unset.

**A release by hand** (every release until Actions billing is fixed; 0.1.0 must be one, since npm
sets up a trusted publisher only on a package that exists):

1. Merge a PR that sets `version` in `package.json` and `VERSION` in `src/platform.ts`.
2. In a fresh clone at the merge commit: `bun install --frozen-lockfile`, then
   `bun run typecheck && bun run build && bun test ./tests` and
   `YIAN_TEST_PACK=1 bun test tests/packaging.test.ts -t "npm tarball"`.
3. `npm login` (account `ngocbh`, 2FA on), then `npm publish --access public --dry-run`: check the
   file list has `README.md`, `LICENSE`, `NOTICE` and no `tests/`.
4. `npm publish --access public` (it asks for the one-time password).
5. Tag it: `gh release create v<version> --target <merge sha> --title "yian <version>"`.

**Trusted publishing** (once, after 0.1.0): on npmjs.com, `@ngocbh/yian` → Settings → Trusted
Publisher → GitHub Actions: user `ngocbh`, repository `yian`, workflow `npm-publish.yml`,
environment `npm`. Set publishing access to "Require two-factor authentication and disallow
tokens". On GitHub, create the environment `npm` (optionally `v*` tags only, with a reviewer).
Then a pushed `v*` tag publishes through `.github/workflows/npm-publish.yml`, with no stored token.

The workflow can't run while Actions are disabled. It adds no provenance while the repo is private;
when it goes public, restore `repository` in `package.json` and npm attaches provenance itself.
