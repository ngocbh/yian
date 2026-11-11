#!/bin/sh
# Takes out what `install.sh` put in: the `yian` wrapper in YIAN_BIN_DIR (default ~/.local/bin), the
# PATH lines in the shell rc files, and the clone in --dir or YIAN_INSTALL_DIR (default
# ~/.local/share/yian) when install.sh made it and it holds no local work. Leaves Bun and yian's own
# data (saved logins, run logs) in place. It runs piped from curl as well as from a checkout.
#
#   curl -fsSL https://raw.githubusercontent.com/ngocbh/yian/main/uninstall.sh | bash
#   ./uninstall.sh [--dir DIR]
#
# The body is one function, called on the last line, so a download cut short runs nothing, and a
# run that removes the clone holding this file has already read all of it.

set -eu

MARK="yian install.sh"
# The file in a clone's .git that says install.sh made the clone.
CLONE_MARK=yian-install

if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
  C_STEP=$(printf '\033[0;36m') C_OK=$(printf '\033[0;32m') C_WARN=$(printf '\033[0;33m')
  C_ERR=$(printf '\033[0;31m') C_OFF=$(printf '\033[0m')
else
  C_STEP='' C_OK='' C_WARN='' C_ERR='' C_OFF=''
fi

step() { printf '%s→%s %s\n' "$C_STEP" "$C_OFF" "$1"; }
ok() { printf '%s✓%s %s\n' "$C_OK" "$C_OFF" "$1"; }
warn() { printf '%s⚠%s %s\n' "$C_WARN" "$C_OFF" "$1"; }
die() {
  printf '%s✗%s %s\n' "$C_ERR" "$C_OFF" "$1" >&2
  exit 1
}

parse_args() {
  INSTALL_DIR=${YIAN_INSTALL_DIR:-$HOME/.local/share/yian}
  while [ $# -gt 0 ]; do
    case $1 in
      --dir)
        [ $# -ge 2 ] || die "--dir needs a value."
        INSTALL_DIR=$2
        shift 2
        ;;
      -h | --help)
        printf 'Usage: uninstall.sh [--dir DIR]\n\n--dir is the folder install.sh cloned yian into (YIAN_INSTALL_DIR, default ~/.local/share/yian).\n'
        exit 0
        ;;
      *) die "Unknown option $1. Run with --help to list the options." ;;
    esac
  done
  case $INSTALL_DIR in
    /*) ;;
    *) INSTALL_DIR=$(pwd -P)/$INSTALL_DIR ;;
  esac
}

remove_wrapper() {
  target=${YIAN_BIN_DIR:-$HOME/.local/bin}/yian
  [ -e "$target" ] || return 0
  if grep -q "$MARK" "$target" 2>/dev/null; then
    rm -f "$target"
    ok "Removed $target"
  else
    warn "Left $target in place: install.sh did not write it."
  fi
}

remove_path_lines() {
  zdir=${ZDOTDIR:-$HOME}
  for rc in "$HOME/.bashrc" "$HOME/.profile" "$HOME/.bash_profile" "$HOME/.bash_login" \
    "$zdir/.zshrc" "$zdir/.zprofile" "${XDG_CONFIG_HOME:-$HOME/.config}/fish/config.fish"; do
    [ -f "$rc" ] && grep -q "$MARK" "$rc" || continue
    tmp=$(mktemp)
    grep -v "$MARK" "$rc" >"$tmp" || true
    # Write back through the original file, so its permissions and any symlink stay.
    cat "$tmp" >"$rc"
    rm -f "$tmp"
    ok "Removed the PATH line from $rc"
  done
}

# Prints what local work the clone $1 holds, or nothing when it holds none; fails when git can't
# tell. node_modules folders are what `bun install` wrote; any other untracked or ignored file,
# such as a .env, is someone's.
local_work() {
  changes=$(git -C "$1" status --porcelain --ignored) || return 1
  first=$(printf '%s\n' "$changes" | grep -v -E '^!! (.*/)?node_modules/$' | sed -n '1s/^...//p')
  if [ -n "$first" ]; then
    printf 'files that are not committed, such as %s' "$first"
    return 0
  fi
  stashes=$(git -C "$1" stash list) || return 1
  if [ -n "$stashes" ]; then
    printf 'stashed changes'
    return 0
  fi
  unpushed=$(git -C "$1" log --oneline HEAD --branches --not --remotes) || return 1
  if [ -n "$unpushed" ]; then printf 'commits that are not pushed'; fi
}

remove_clone() {
  [ -d "$INSTALL_DIR" ] || return 0
  if [ ! -f "$INSTALL_DIR/.git/$CLONE_MARK" ]; then
    step "Kept $INSTALL_DIR: install.sh didn't clone it."
    return 0
  fi
  if ! work=$(local_work "$INSTALL_DIR"); then
    warn "Kept $INSTALL_DIR: git could not tell whether it holds local work."
    return 0
  fi
  if [ -n "$work" ]; then
    warn "Kept $INSTALL_DIR: it has $work. Remove it yourself once that work is safe."
    return 0
  fi
  case $INSTALL_DIR in
    / | "$HOME" | "$HOME"/) die "Refusing to remove $INSTALL_DIR." ;;
  esac
  rm -rf "$INSTALL_DIR"
  ok "Removed $INSTALL_DIR"
}

main() {
  parse_args "$@"
  remove_wrapper
  remove_path_lines
  remove_clone
  step "Left Bun, your saved logins in ${XDG_CONFIG_HOME:-$HOME/.config}/yian and your run logs in $HOME/.yian in place."
}

main "$@"
