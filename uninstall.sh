#!/bin/sh
# Takes out what `install.sh` put in: the `yian` wrapper in YIAN_BIN_DIR (default ~/.local/bin)
# and the PATH line in the shell rc files. Leaves the checkout, Bun, and yian's own data (run logs,
# saved logins) in place.
#
#   ./uninstall.sh

set -eu

MARK="yian install.sh"
BIN_DIR=${YIAN_BIN_DIR:-$HOME/.local/bin}
TARGET=$BIN_DIR/yian

if [ -e "$TARGET" ]; then
  if grep -q "$MARK" "$TARGET" 2>/dev/null; then
    rm -f "$TARGET"
    printf 'Removed %s.\n' "$TARGET"
  else
    printf 'Left %s in place: install.sh did not write it.\n' "$TARGET"
  fi
fi

for rc in "$HOME/.bashrc" "${ZDOTDIR:-$HOME}/.zshrc" "$HOME/.profile"; do
  [ -f "$rc" ] && grep -q "$MARK" "$rc" || continue
  tmp=$(mktemp)
  grep -v "$MARK" "$rc" >"$tmp" || true
  # Write back through the original file, so its permissions and any symlink stay.
  cat "$tmp" >"$rc"
  rm -f "$tmp"
  printf 'Removed the PATH line from %s.\n' "$rc"
done
