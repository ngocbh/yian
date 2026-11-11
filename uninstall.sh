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
# An exported CDPATH would send a relative cd elsewhere, and make it print the folder.
unset CDPATH

MARK="yian install.sh"
# The wrapper's own line that says install.sh wrote it, matched whole.
WRAPPER_MARK="# Written by $MARK"
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
fail() {
  printf '%s✗%s %s\n' "$C_ERR" "$C_OFF" "$1" >&2
  FAILED=1
}
die() {
  fail "$1"
  exit 1
}

# Single-quotes $1 for a shell script.
quote() {
  printf "'%s'" "$(printf '%s' "$1" | sed "s/'/'\\\\''/g")"
}

# The lines install.sh adds to rc files to put the folder $1 on PATH: one guarded line each, with
# the mark at its end. These are the only lines uninstall.sh removes.
sh_path_line() {
  dir=$(quote "$1")
  printf '%s' "case \":\$PATH:\" in *:$dir:*) ;; *) export PATH=$dir:\"\$PATH\" ;; esac  # added by $MARK"
}
fish_path_line() {
  printf '%s' "fish_add_path --path $(quote "$1")  # added by $MARK"
}

# The size of the file $1 in bytes.
size() {
  wc -c <"$1" | tr -d ' '
}

# Refuses the path $2, which $1 names, unless it's absolute and holds no newline: a relative folder
# on PATH would run whatever the current folder holds, and a newline would break the lines written.
check_path() {
  case $2 in
    /*) ;;
    *) die "$1 must be an absolute path, not '$2'." ;;
  esac
  case $2 in
    *"
"*) die "$1 must not hold a newline." ;;
  esac
}

parse_args() {
  check_path HOME "${HOME:-}"
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
  check_path "--dir or YIAN_INSTALL_DIR" "$INSTALL_DIR"
  BIN_DIR=${YIAN_BIN_DIR:-$HOME/.local/bin}
  check_path YIAN_BIN_DIR "$BIN_DIR"
}

remove_wrapper() {
  target=$BIN_DIR/yian
  [ -e "$target" ] || return 0
  if LC_ALL=C grep -a -q -x -F -e "$WRAPPER_MARK" "$target" 2>/dev/null; then
    rm -f "$target"
    ok "Removed $target"
  else
    warn "Left $target in place: install.sh did not write it."
  fi
}

# Removes the marked PATH lines from the rc file $1. grep reads it as bytes, whatever the locale
# and whatever it holds. The new text is checked to be the old one less exactly those lines before
# it's written back, through the file so its permissions and any symlink stay; a failed write puts
# the old text back.
strip_path_lines() {
  if [ ! -w "$1" ]; then
    warn "Left the PATH line in $1: the file isn't writable."
    return 0
  fi
  if ! NEW=$(mktemp) || ! OLD=$(mktemp); then
    fail "Could not make a temp file, so the PATH line is still in $1."
    return 0
  fi
  code=0
  LC_ALL=C grep -a -v -x -F -e "$SH_LINE" -e "$FISH_LINE" "$1" >"$NEW" || code=$?
  # grep ends a last line that has no newline with one.
  expected=$(($(size "$1") - $(LC_ALL=C grep -a -x -F -e "$SH_LINE" -e "$FISH_LINE" "$1" | wc -c)))
  if [ -n "$(tail -c 1 "$1")" ]; then expected=$((expected + 1)); fi
  if [ "$code" -gt 1 ] || [ "$(size "$NEW")" -ne "$expected" ] ||
    ! cat "$1" >"$OLD" || [ "$(size "$OLD")" -ne "$(size "$1")" ]; then
    fail "Could not write the new text of $1, so it's as it was, PATH line included."
  elif cat "$NEW" >"$1" && [ "$(size "$1")" -eq "$expected" ]; then
    ok "Removed the PATH line from $1"
  elif cat "$OLD" >"$1"; then
    fail "Could not write $1, so it's as it was, PATH line included."
  else
    fail "Could not write $1. Its old text is in $OLD."
    OLD=
  fi
  rm -f "$NEW" ${OLD:+"$OLD"}
  NEW= OLD=
}

remove_path_lines() {
  SH_LINE=$(sh_path_line "$BIN_DIR")
  FISH_LINE=$(fish_path_line "$BIN_DIR")
  zdir=${ZDOTDIR:-$HOME}
  for rc in "$HOME/.bashrc" "$HOME/.profile" "$HOME/.bash_profile" "$HOME/.bash_login" \
    "$zdir/.zshrc" "$zdir/.zprofile" "${XDG_CONFIG_HOME:-$HOME/.config}/fish/config.fish"; do
    if [ -f "$rc" ] && LC_ALL=C grep -a -q -x -F -e "$SH_LINE" -e "$FISH_LINE" "$rc"; then
      strip_path_lines "$rc"
    fi
  done
}

# Prints what local work the clone $1 holds, or nothing when it holds none; fails when git can't
# tell. The checks don't follow the user's git config, and every untracked or ignored file counts,
# whichever ignore file hides it, except the node_modules folders `bun install` writes.
local_work() {
  files=$(git --no-optional-locks -C "$1" status --porcelain --untracked-files=all --ignored=matching) || return 1
  first=$(printf '%s\n' "$files" | LC_ALL=C grep -a -v -E '^!! (.*/)?node_modules/$' | LC_ALL=C sed -n '1s/^...//p')
  if [ -n "$first" ]; then
    printf 'files that are not committed, such as %s' "$first"
    return 0
  fi
  stashes=$(git --no-optional-locks -C "$1" stash list) || return 1
  if [ -n "$stashes" ]; then
    printf 'stashed changes'
    return 0
  fi
  worktrees=$(git --no-optional-locks -C "$1" worktree list --porcelain | LC_ALL=C grep -a -c '^worktree ') || return 1
  if [ "$worktrees" -gt 1 ]; then
    printf 'other worktrees (git worktree list shows them)'
    return 0
  fi
  # Any ref's commits: branches, tags, a detached HEAD.
  unpushed=$(git --no-optional-locks -C "$1" rev-list --all --not --remotes) || return 1
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
  FAILED= NEW= OLD=
  trap 'rm -f ${NEW:+"$NEW"} ${OLD:+"$OLD"}' EXIT
  trap 'exit 1' HUP INT TERM
  parse_args "$@"
  remove_wrapper
  remove_path_lines
  remove_clone
  step "Left Bun, your saved logins in ${XDG_CONFIG_HOME:-$HOME/.config}/yian and your run logs in $HOME/.yian in place."
  if [ -n "$FAILED" ]; then exit 1; fi
}

# In braces, so a download cut anywhere in this line is a syntax error and runs nothing.
{ main "$@"; }
