#!/bin/sh
# Puts this checkout's `yian` on PATH: finds Bun 1.4 or later (installing it when there is none),
# installs the checkout's dependencies, writes a `yian` wrapper into YIAN_BIN_DIR (default
# ~/.local/bin), and adds that folder to PATH in the shell's rc file when it isn't there yet.
# Running it again updates the wrapper. `uninstall.sh` takes it back out.
#
#   ./install.sh
#
# Environment: YIAN_BUN, the Bun to use; YIAN_BIN_DIR, the folder for the wrapper.

set -eu

MARK="yian install.sh"
BUN_FLOOR_MAJOR=1
BUN_FLOOR_MINOR=4

die() {
  printf 'Error: %s\n' "$1" >&2
  exit 1
}

# Single-quotes $1 for a shell script.
quote() {
  printf "'%s'" "$(printf '%s' "$1" | sed "s/'/'\\\\''/g")"
}

ROOT=$(cd "$(dirname "$0")" && pwd -P)
[ -f "$ROOT/src/cli/bin.ts" ] || die "$ROOT is not a yian checkout (no src/cli/bin.ts)."
BIN_DIR=${YIAN_BIN_DIR:-$HOME/.local/bin}
TARGET=$BIN_DIR/yian

# Refuse early, before installing anything, if the target is someone else's file.
if [ -e "$TARGET" ] && ! grep -q "$MARK" "$TARGET" 2>/dev/null; then
  die "$TARGET exists and was not written by install.sh. Move it away, or set YIAN_BIN_DIR."
fi

# Find Bun: YIAN_BUN, then bun on PATH, then Bun's own install folder.
BUN_HOME=${BUN_INSTALL:-$HOME/.bun}
if [ -n "${YIAN_BUN:-}" ]; then
  BUN=$YIAN_BUN
  [ -x "$BUN" ] || die "YIAN_BUN=$BUN is not an executable file."
elif command -v bun >/dev/null 2>&1; then
  BUN=$(command -v bun)
elif [ -x "$BUN_HOME/bin/bun" ]; then
  BUN=$BUN_HOME/bin/bun
else
  command -v curl >/dev/null 2>&1 || die "Bun is not installed and curl is missing. Install Bun from https://bun.sh, then run this again."
  printf 'Bun not found; installing it with https://bun.sh/install\n'
  curl -fsSL https://bun.sh/install | bash
  BUN=$BUN_HOME/bin/bun
  [ -x "$BUN" ] || die "Bun's installer did not leave $BUN. Install Bun from https://bun.sh, then run this again."
fi

BUN_VERSION=$("$BUN" --version)
major=$(printf '%s' "$BUN_VERSION" | cut -d. -f1)
minor=$(printf '%s' "$BUN_VERSION" | cut -d. -f2)
case "$major.$minor" in
  *[!0-9.]* | .* | *.) die "could not read a version from $BUN --version: $BUN_VERSION" ;;
esac
if [ "$major" -lt "$BUN_FLOOR_MAJOR" ] || { [ "$major" -eq "$BUN_FLOOR_MAJOR" ] && [ "$minor" -lt "$BUN_FLOOR_MINOR" ]; }; then
  die "yian needs Bun $BUN_FLOOR_MAJOR.$BUN_FLOOR_MINOR or later; $BUN is $BUN_VERSION. Run 'bun upgrade', then run this again."
fi
BUN=$(cd "$(dirname "$BUN")" && pwd -P)/$(basename "$BUN")

(cd "$ROOT" && "$BUN" install --frozen-lockfile) || die "bun install failed in $ROOT."

mkdir -p "$BIN_DIR"
TMP=$TARGET.tmp.$$
cat >"$TMP" <<EOF
#!/bin/sh
# Written by $MARK: runs the yian checkout at $ROOT with Bun.
# The flags keep Bun from reading the current folder's .env and bunfig.toml and from installing packages.
exec $(quote "$BUN") --no-install --no-env-file --config=/dev/null $(quote "$ROOT/src/cli/bin.ts") "\$@"
EOF
chmod 755 "$TMP"
mv -f "$TMP" "$TARGET"

# Put BIN_DIR on PATH for new shells, unless it already is.
RC=
case ":$PATH:" in
  *":$BIN_DIR:"*) ;;
  *)
    case "$(basename "${SHELL:-sh}")" in
      bash) RC=$HOME/.bashrc ;;
      zsh) RC=${ZDOTDIR:-$HOME}/.zshrc ;;
      fish) RC= ;;
      *) RC=$HOME/.profile ;;
    esac
    if [ -z "$RC" ]; then
      printf 'Add %s to PATH in your shell config: fish_add_path %s\n' "$BIN_DIR" "$(quote "$BIN_DIR")"
    elif ! grep -q "$MARK" "$RC" 2>/dev/null; then
      printf 'export PATH=%s:"$PATH"  # added by %s\n' "$(quote "$BIN_DIR")" "$MARK" >>"$RC"
    else
      RC=
    fi
    ;;
esac

printf 'Installed yian %s at %s (Bun %s).\n' "$("$TARGET" --version)" "$TARGET" "$BUN_VERSION"
if [ -n "$RC" ]; then
  printf 'Added %s to PATH in %s. Open a new shell, or run: export PATH=%s:"$PATH"\n' "$BIN_DIR" "$RC" "$(quote "$BIN_DIR")"
fi
