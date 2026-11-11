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
# The wrapper's own line that says install.sh wrote it, matched whole, and the line an earlier
# install.sh wrote there instead (an extended regular expression, matched whole).
WRAPPER_MARK="# Written by $MARK"
OLD_WRAPPER_MARK='# Written by yian install\.sh: runs the yian checkout at /.* with Bun\.'
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

# The unguarded line an earlier install.sh added to one rc file, which install.sh swaps for
# sh_path_line's and uninstall.sh removes.
old_path_line() {
  printf '%s' "export PATH=$(quote "$1"):\"\$PATH\"  # added by $MARK"
}

# True when the file $1 is a wrapper install.sh wrote, now or in its earlier form.
our_wrapper() {
  LC_ALL=C grep -a -q -x -F -e "$WRAPPER_MARK" "$1" 2>/dev/null ||
    LC_ALL=C grep -a -q -x -E -e "$OLD_WRAPPER_MARK" "$1" 2>/dev/null
}

# The size of the file $1 in bytes.
size() {
  wc -c <"$1" | tr -d ' '
}

# Removes from the rc file $1 the lines given after it, matched whole, and fails when $1 is left as
# it was. grep reads the file as bytes, whatever the locale and whatever it holds. The new text is
# checked to be the old one less exactly those lines before it's written back, through the file so
# its permissions and any symlink stay; a failed write puts the old text back.
strip_lines() {
  file=$1
  shift
  for stripped_line do
    set -- "$@" -e "$stripped_line"
    shift
  done
  if [ ! -w "$file" ]; then
    warn "Left the PATH line in $file: the file isn't writable."
    return 1
  fi
  if ! NEW=$(mktemp) || ! OLD=$(mktemp); then
    fail "Could not make a temp file, so the PATH line is still in $file."
    return 1
  fi
  code=0
  LC_ALL=C grep -a -v -x -F "$@" "$file" >"$NEW" || code=$?
  # grep ends a last line that has no newline with one.
  expected=$(($(size "$file") - $(LC_ALL=C grep -a -x -F "$@" "$file" | wc -c)))
  if [ -n "$(tail -c 1 "$file")" ]; then expected=$((expected + 1)); fi
  stripped=
  if [ "$code" -gt 1 ] || [ "$(size "$NEW")" -ne "$expected" ] ||
    ! cat "$file" >"$OLD" || [ "$(size "$OLD")" -ne "$(size "$file")" ]; then
    fail "Could not write the new text of $file, so it's as it was, PATH line included."
  elif cat "$NEW" >"$file" && [ "$(size "$file")" -eq "$expected" ]; then
    stripped=1
  elif cat "$OLD" >"$file"; then
    fail "Could not write $file, so it's as it was, PATH line included."
  else
    fail "Could not write $file. Its old text is in $OLD."
    OLD=
  fi
  rm -f "$NEW" ${OLD:+"$OLD"}
  NEW= OLD=
  [ -n "$stripped" ]
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
  if our_wrapper "$target"; then
    rm -f "$target"
    ok "Removed $target"
  else
    warn "Left $target in place: install.sh did not write it."
  fi
}

remove_path_lines() {
  SH_LINE=$(sh_path_line "$BIN_DIR")
  FISH_LINE=$(fish_path_line "$BIN_DIR")
  OLD_LINE=$(old_path_line "$BIN_DIR")
  zdir=${ZDOTDIR:-$HOME}
  for rc in "$HOME/.bashrc" "$HOME/.profile" "$HOME/.bash_profile" "$HOME/.bash_login" \
    "$zdir/.zshrc" "$zdir/.zprofile" "${XDG_CONFIG_HOME:-$HOME/.config}/fish/config.fish"; do
    if [ -f "$rc" ] && LC_ALL=C grep -a -q -x -F -e "$SH_LINE" -e "$FISH_LINE" -e "$OLD_LINE" "$rc" &&
      strip_lines "$rc" "$SH_LINE" "$FISH_LINE" "$OLD_LINE"; then
      ok "Removed the PATH line from $rc"
    fi
  done
}

# Prints the first five of the lines $1 holds, joined with commas, and how many more there are.
first_five() {
  printf '%s\n' "$1" | LC_ALL=C awk 'NF { n++; if (n <= 5) printf "%s%s", (n > 1 ? ", " : ""), $0 }
    END { if (n > 5) printf ", and %d more", n - 5 }'
}

# Prints what local work the clone $1 holds, naming up to five of the files or refs that hold it,
# or nothing when it holds none; fails when git can't tell. The checks don't follow the user's git
# config, and every untracked or ignored file counts, whichever ignore file hides it, except the
# node_modules folders `bun install` writes and macOS's .DS_Store files.
local_work() {
  files=$(git --no-optional-locks -C "$1" status --porcelain --untracked-files=all --ignored=matching) || return 1
  files=$(printf '%s\n' "$files" | LC_ALL=C grep -a -v -E -e '^!! (.*/)?node_modules/$' -e '^.. (.*/)?\.DS_Store$' |
    LC_ALL=C sed -n 's/^...//p')
  if [ -n "$files" ]; then
    printf 'files that are not committed: %s' "$(first_five "$files")"
    return 0
  fi
  stashes=$(git --no-optional-locks -C "$1" stash list --format=%gd) || return 1
  if [ -n "$stashes" ]; then
    printf 'stashed changes: %s' "$(first_five "$stashes")"
    return 0
  fi
  # The first worktree listed is the clone itself.
  worktrees=$(git --no-optional-locks -C "$1" worktree list --porcelain) || return 1
  others=$(printf '%s\n' "$worktrees" | LC_ALL=C sed -n 's/^worktree //p' | sed 1d)
  if [ -n "$others" ]; then
    printf 'other worktrees: %s' "$(first_five "$others")"
    return 0
  fi
  # Any ref's commits that no remote branch holds: branches, tags, notes, a detached HEAD.
  unpushed=$(git --no-optional-locks -C "$1" rev-list --all --not --remotes) || return 1
  [ -n "$unpushed" ] || return 0
  refs=$(git --no-optional-locks -C "$1" for-each-ref --format='%(refname) %(refname:short)') || return 1
  if ! git -C "$1" symbolic-ref --quiet HEAD >/dev/null; then refs=$(printf 'HEAD HEAD\n%s' "$refs"); fi
  holders=$(printf '%s\n' "$refs" | while read -r ref name; do
    case $ref in
      '' | refs/remotes/* | refs/stash) continue ;;
    esac
    if [ -n "$(git --no-optional-locks -C "$1" rev-list --max-count=1 "$ref" --not --remotes)" ]; then
      printf '%s\n' "$name"
    fi
  done)
  printf 'commits that are not pushed'
  if [ -n "$holders" ]; then printf ': %s' "$(first_five "$holders")"; fi
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
  if LC_ALL=C grep -a -q -x -F 'folder=created' "$INSTALL_DIR/.git/$CLONE_MARK"; then
    rm -rf "$INSTALL_DIR"
    ok "Removed $INSTALL_DIR"
    return 0
  fi
  # The folder was there before install.sh cloned into it, so only what's in it goes.
  for entry in "$INSTALL_DIR"/* "$INSTALL_DIR"/.[!.]* "$INSTALL_DIR"/..?*; do
    if [ -e "$entry" ] || [ -L "$entry" ]; then rm -rf "$entry"; fi
  done
  ok "Removed the clone in $INSTALL_DIR, and kept the folder, which was there before install.sh"
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
