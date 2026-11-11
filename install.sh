#!/bin/sh
# Installs the `yian` command. Piped from curl, it clones yian into ~/.local/share/yian, or updates
# the clone it made there; run as ./install.sh from a checkout, it installs that checkout. Then it
# finds Bun 1.4 or later (installing it when there is none), installs the checkout's dependencies,
# writes a `yian` wrapper into ~/.local/bin, and adds that folder to PATH in the login shell's rc
# files. Running it again updates the install. `uninstall.sh` takes it back out.
#
#   curl -fsSL https://raw.githubusercontent.com/ngocbh/yian/main/install.sh | bash
#   curl -fsSL https://raw.githubusercontent.com/ngocbh/yian/main/install.sh | bash -s -- --dir DIR
#   ./install.sh
#
# Options: --dir DIR, --branch BRANCH and --repo URL pick a piped install's clone; --no-modify-path
# leaves the rc files alone; --skip-setup skips the next-step hint. --help lists them.
# Environment: YIAN_INSTALL_DIR, the clone's folder; YIAN_REPO_URL, the repository to clone;
# YIAN_BUN, the Bun to use; YIAN_BIN_DIR, the folder for the wrapper.
#
# The body is one function, called on the last line, so a download cut short runs nothing.

set -eu

MARK="yian install.sh"
# The file in a clone's .git that says install.sh made the clone, so uninstall.sh may remove it.
CLONE_MARK=yian-install
DEFAULT_REPO=https://github.com/ngocbh/yian.git
BUN_FLOOR_MAJOR=1
BUN_FLOOR_MINOR=4

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

# Single-quotes $1 for a shell script.
quote() {
  printf "'%s'" "$(printf '%s' "$1" | sed "s/'/'\\\\''/g")"
}

usage() {
  cat <<'EOF'
Usage: install.sh [options]

Piped from curl, clones yian and installs the clone; run from a checkout, installs that checkout.

  --dir DIR         where a piped install clones yian (YIAN_INSTALL_DIR, default ~/.local/share/yian)
  --branch BRANCH   the branch to clone (default main)
  --repo URL        the repository to clone (YIAN_REPO_URL, default https://github.com/ngocbh/yian.git)
  --no-modify-path  leave the shell rc files alone
  --skip-setup      don't print the next step

YIAN_BUN sets the Bun to use, and YIAN_BIN_DIR the folder for the wrapper (default ~/.local/bin).
EOF
}

parse_args() {
  INSTALL_DIR=${YIAN_INSTALL_DIR:-$HOME/.local/share/yian}
  REPO_URL=${YIAN_REPO_URL:-$DEFAULT_REPO}
  BRANCH=main
  BRANCH_GIVEN=
  CLONE_FLAGS=
  MODIFY_PATH=1
  SKIP_SETUP=
  while [ $# -gt 0 ]; do
    case $1 in
      --dir | --branch | --repo)
        [ $# -ge 2 ] || die "$1 needs a value."
        case $1 in
          --dir) INSTALL_DIR=$2 ;;
          --branch) BRANCH=$2 BRANCH_GIVEN=1 ;;
          --repo) REPO_URL=$2 ;;
        esac
        CLONE_FLAGS="$CLONE_FLAGS $1"
        shift 2
        ;;
      --no-modify-path) MODIFY_PATH= && shift ;;
      --skip-setup) SKIP_SETUP=1 && shift ;;
      -h | --help) usage && exit 0 ;;
      *) die "Unknown option $1. Run with --help to list the options." ;;
    esac
  done
  case $INSTALL_DIR in
    /*) ;;
    *) INSTALL_DIR=$(pwd -P)/$INSTALL_DIR ;;
  esac
}

# Sets ROOT to the checkout this script is in, or leaves it empty when the script was piped.
find_checkout() {
  ROOT=
  # Run from a file, $0 is that file; piped through `curl | bash`, it's the shell.
  if [ -f "$0" ]; then
    here=$(cd "$(dirname "$0")" && pwd -P)
    if [ -f "$here/src/cli/bin.ts" ]; then ROOT=$here; fi
  fi
}

check_prerequisites() {
  case $(uname -s) in
    Linux | Darwin) ;;
    *) die "install.sh runs on macOS and Linux. Elsewhere, install the npm package: npm install -g @ngocbh/yian" ;;
  esac
  if [ -z "$ROOT" ] && ! command -v git >/dev/null 2>&1; then
    die "git is missing. Install git, then run this again."
  fi
}

# Refuses, before anything is installed, to replace a `yian` that install.sh didn't write.
check_target() {
  BIN_DIR=${YIAN_BIN_DIR:-$HOME/.local/bin}
  TARGET=$BIN_DIR/yian
  if [ -e "$TARGET" ] && ! grep -q "$MARK" "$TARGET" 2>/dev/null; then
    die "$TARGET exists and was not written by install.sh. Move it away, or set YIAN_BIN_DIR."
  fi
}

# Sets CLONE to `new` when INSTALL_DIR is free to clone into, or to `update` when it holds a clean
# clone of REPO_URL on a branch, whichever branch that is. Refuses anything else, so local work is
# never lost.
check_clone() {
  CLONE=new
  [ -e "$INSTALL_DIR" ] || return 0
  [ -d "$INSTALL_DIR" ] || die "$INSTALL_DIR is a file, not a folder. Move it away, or pass --dir for another folder."
  if [ ! -e "$INSTALL_DIR/.git" ]; then
    [ -n "$(ls -A "$INSTALL_DIR")" ] || return 0
    die "$INSTALL_DIR holds files but isn't a git clone. Move it away, or pass --dir for another folder."
  fi
  origin=$(git -C "$INSTALL_DIR" config --get remote.origin.url) || origin=
  if [ "$origin" != "$REPO_URL" ]; then
    die "$INSTALL_DIR is a clone of ${origin:-no remote}, not $REPO_URL. Pass --dir for another folder, or --repo for the repository it clones."
  fi
  current=$(git -C "$INSTALL_DIR" symbolic-ref --quiet --short HEAD) ||
    die "$INSTALL_DIR is not on a branch. Switch it to one with git switch, then run this again."
  if [ -n "$(git -C "$INSTALL_DIR" status --porcelain)" ]; then
    die "$INSTALL_DIR has uncommitted changes. Commit or stash them, then run this again."
  fi
  if [ -n "$BRANCH_GIVEN" ] && [ "$current" != "$BRANCH" ]; then
    warn "$INSTALL_DIR is on branch $current, not $BRANCH, so this updates $current. To change, run git -C $(quote "$INSTALL_DIR") switch $(quote "$BRANCH"), then this again."
  fi
  CLONE=update
}

# Sets BUN to an absolute path to Bun 1.4 or later, and BUN_VERSION to its version: YIAN_BUN, then
# bun on PATH, then Bun's own install folder, where Bun's installer puts it when none is found.
find_bun() {
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
    step "Installing Bun with https://bun.sh/install"
    curl -fsSL https://bun.sh/install | bash
    BUN=$BUN_HOME/bin/bun
    [ -x "$BUN" ] || die "Bun's installer did not leave $BUN. Install Bun from https://bun.sh, then run this again."
  fi

  BUN_VERSION=$("$BUN" --version)
  major=$(printf '%s' "$BUN_VERSION" | cut -d. -f1)
  minor=$(printf '%s' "$BUN_VERSION" | cut -d. -f2)
  case "$major.$minor" in
    *[!0-9.]* | .* | *.) die "Could not read a version from $BUN --version: $BUN_VERSION" ;;
  esac
  if [ "$major" -lt "$BUN_FLOOR_MAJOR" ] || { [ "$major" -eq "$BUN_FLOOR_MAJOR" ] && [ "$minor" -lt "$BUN_FLOOR_MINOR" ]; }; then
    die "yian needs Bun $BUN_FLOOR_MAJOR.$BUN_FLOOR_MINOR or later; $BUN is $BUN_VERSION. Run 'bun upgrade', then run this again."
  fi
  BUN=$(cd "$(dirname "$BUN")" && pwd -P)/$(basename "$BUN")
  ok "Bun $BUN_VERSION at $BUN"
}

# Clones REPO_URL into INSTALL_DIR, or fast-forwards the clone there, and sets ROOT to it.
get_clone() {
  if [ "$CLONE" = new ]; then
    step "Cloning $REPO_URL ($BRANCH) into $INSTALL_DIR"
    mkdir -p "$(dirname "$INSTALL_DIR")"
    git clone --quiet --recurse-submodules --branch "$BRANCH" "$REPO_URL" "$INSTALL_DIR" ||
      die "git clone failed. Check --repo and --branch, then run this again."
    printf 'Cloned by %s; uninstall.sh removes this clone when it holds no local work.\n' "$MARK" >"$INSTALL_DIR/.git/$CLONE_MARK"
    ok "Cloned yian $(git -C "$INSTALL_DIR" rev-parse --short HEAD) into $INSTALL_DIR"
  else
    before=$(git -C "$INSTALL_DIR" rev-parse --short HEAD)
    step "Updating $INSTALL_DIR"
    git -C "$INSTALL_DIR" pull --quiet --ff-only ||
      die "git pull --ff-only failed in $INSTALL_DIR. Bring it up to date with its remote, then run this again."
    git -C "$INSTALL_DIR" submodule --quiet update --init --recursive ||
      die "git submodule update failed in $INSTALL_DIR."
    after=$(git -C "$INSTALL_DIR" rev-parse --short HEAD)
    if [ "$before" = "$after" ]; then
      ok "$INSTALL_DIR is up to date ($after)"
    else
      ok "Updated $INSTALL_DIR from $before to $after"
    fi
  fi
  [ -f "$INSTALL_DIR/src/cli/bin.ts" ] || die "$INSTALL_DIR is not a yian checkout (no src/cli/bin.ts)."
  ROOT=$INSTALL_DIR
}

install_dependencies() {
  step "Installing dependencies in $ROOT"
  (cd "$ROOT" && "$BUN" install --frozen-lockfile) || die "bun install failed in $ROOT."
  ok "Installed dependencies"
}

# Writes the wrapper to a temp file and moves it into place, so a run never sees half of it.
write_wrapper() {
  mkdir -p "$BIN_DIR"
  tmp=$TARGET.tmp.$$
  cat >"$tmp" <<EOF
#!/bin/sh
# Written by $MARK: runs the yian checkout at $ROOT with Bun, or with YIAN_BUN when it's set.
# The flags keep Bun from reading the current folder's .env and bunfig.toml and from installing packages.
bun=\${YIAN_BUN:-$(quote "$BUN")}
exec "\$bun" --no-install --no-env-file --config=/dev/null $(quote "$ROOT/src/cli/bin.ts") "\$@"
EOF
  chmod 755 "$tmp"
  mv -f "$tmp" "$TARGET"
}

# True when the rc file $1 already puts BIN_DIR on PATH: with the line install.sh writes, or with a
# line of its own that names the folder, by its path or from $HOME.
puts_on_path() {
  [ -f "$1" ] || return 1
  if grep -qF "$MARK" "$1"; then return 0; fi
  from_home=${BIN_DIR#"$HOME"/}
  grep -v '^[[:space:]]*#' "$1" | grep -E 'PATH|fish_add_path' |
    grep -qF -e "$BIN_DIR" -e "\$HOME/$from_home" -e "\${HOME}/$from_home" -e "~/$from_home"
}

# Appends the line $2, marked for uninstall.sh, to the rc file $1, unless $1 already puts BIN_DIR
# on PATH.
add_path_line() {
  if puts_on_path "$1"; then return 0; fi
  mkdir -p "$(dirname "$1")"
  # A last line with no newline stays whole.
  newline=
  if [ -s "$1" ] && [ -n "$(tail -c 1 "$1")" ]; then newline='\n'; fi
  printf "$newline%s  # added by %s\n" "$2" "$MARK" >>"$1" ||
    die "Could not write $1. Pass --no-modify-path, and put $BIN_DIR on PATH yourself."
  ok "Added $BIN_DIR to PATH in $1"
}

# Puts BIN_DIR on PATH for new shells in the login shell's rc files, unless --no-modify-path was
# given or it's on PATH already, and sets RELOAD to the command that reads those files into this
# shell.
wire_path() {
  RELOAD=
  [ -n "$MODIFY_PATH" ] || return 0
  case ":$PATH:" in
    *":$BIN_DIR:"*) return 0 ;;
  esac
  dir=$(quote "$BIN_DIR")
  # Guarded, since a shell can read more than one of these files.
  line="case \":\$PATH:\" in *:$dir:*) ;; *) export PATH=$dir:\"\$PATH\" ;; esac"
  case $(basename "${SHELL:-sh}") in
    zsh)
      add_path_line "${ZDOTDIR:-$HOME}/.zshrc" "$line"
      add_path_line "${ZDOTDIR:-$HOME}/.zprofile" "$line"
      RELOAD="source $(quote "${ZDOTDIR:-$HOME}/.zshrc")"
      ;;
    bash)
      add_path_line "$HOME/.bashrc" "$line"
      add_path_line "$HOME/.profile" "$line"
      # A login bash reads the first of .bash_profile, .bash_login and .profile it finds.
      for rc in "$HOME/.bash_profile" "$HOME/.bash_login"; do
        if [ -f "$rc" ]; then add_path_line "$rc" "$line"; fi
      done
      RELOAD="source $(quote "$HOME/.bashrc")"
      ;;
    fish)
      rc=${XDG_CONFIG_HOME:-$HOME/.config}/fish/config.fish
      # -g sets fish_user_paths for the session, not saved, so removing the line undoes it.
      add_path_line "$rc" "fish_add_path -g $dir"
      RELOAD="source $(quote "$rc")"
      ;;
    *)
      add_path_line "$HOME/.profile" "$line"
      RELOAD=". $(quote "$HOME/.profile")"
      ;;
  esac
}

# Says how to run yian from the shell that ran this, whose PATH the rc files can't change, when
# BIN_DIR isn't on it.
path_hint() {
  case ":$PATH:" in
    *":$BIN_DIR:"*) return 0 ;;
  esac
  if [ -n "$RELOAD" ]; then
    step "To use yian in this shell, open a new terminal or run: $RELOAD"
  else
    warn "$BIN_DIR is not on PATH. Put it on PATH, or run $TARGET."
  fi
}

main() {
  parse_args "$@"
  find_checkout
  check_prerequisites
  check_target
  if [ -n "$ROOT" ]; then
    if [ -n "$CLONE_FLAGS" ]; then
      warn "Ignoring$CLONE_FLAGS: they pick a piped install's clone, and this run installs the checkout at $ROOT."
    fi
    step "Installing yian from the checkout at $ROOT"
  else
    check_clone
  fi
  find_bun
  if [ -z "$ROOT" ]; then get_clone; fi
  install_dependencies
  write_wrapper

  wire_path

  ok "Installed yian $("$TARGET" --version) at $TARGET"
  if [ -z "$SKIP_SETUP" ]; then
    step "Next, save a login for a model provider: yian auth anthropic"
    step "For other providers, see https://yian.readthedocs.io/models/"
  fi
  path_hint
}

main "$@"
