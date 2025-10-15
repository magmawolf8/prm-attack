#!/usr/bin/env bash
set -euo pipefail

# start_attackers_tmux.sh
# Usage:
#   ./start_attackers_tmux.sh <conda_env_name> [--overwrite] [--close-on-exit]
# Example:
#   ./start_attackers_tmux.sh myenv --overwrite

if [ $# -lt 1 ]; then
  echo "Usage: $0 <conda_env_name> [--overwrite] [--close-on-exit]"
  exit 1
fi

CONDA_ENV="$1"
shift || true

OVERWRITE=false
CLOSE_ON_EXIT=false

for arg in "$@"; do
  case "$arg" in
    --overwrite) OVERWRITE=true ;;
    --close-on-exit) CLOSE_ON_EXIT=true ;;
    *) echo "Unknown option: $arg"; exit 1 ;;
  esac
done

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux not found. Install tmux and retry."
  exit 2
fi

# Determine the conda base path (so we can source it)
CONDA_BASE=$(conda info --base)

WORKDIR="$(pwd)"
mapfile -t SCRIPTS < <(printf '%s\n' attacker_*.sh 2>/dev/null | sort -V)

if [ "${#SCRIPTS[@]}" -eq 0 ]; then
  echo "No attacker_*.sh scripts found in $WORKDIR"
  exit 0
fi

echo "Starting ${#SCRIPTS[@]} attacker scripts in tmux with conda env '$CONDA_ENV'..."

for script in "${SCRIPTS[@]}"; do
  if [ ! -f "$script" ]; then
    echo "Skipping non-file: $script"
    continue
  fi

  session_name="$(basename "$script" .sh)"

  if "$OVERWRITE"; then
    if tmux has-session -t "$session_name" 2>/dev/null; then
      echo "Killing existing tmux session: $session_name"
      tmux kill-session -t "$session_name"
    fi
  else
    if tmux has-session -t "$session_name" 2>/dev/null; then
      echo "Session $session_name already exists — skipping (use --overwrite to replace)"
      continue
    fi
  fi

  chmod +x "$script"

  if "$CLOSE_ON_EXIT"; then
    tmux new-session -d -s "$session_name" "
      source \"$CONDA_BASE/etc/profile.d/conda.sh\" &&
      conda activate \"$CONDA_ENV\" &&
      cd \"$WORKDIR\" &&
      bash \"$script\"
    "
    echo "Started $script in detached session: $session_name (auto-close)"
  else
    tmux new-session -d -s "$session_name" "
      source \"$CONDA_BASE/etc/profile.d/conda.sh\" &&
      conda activate \"$CONDA_ENV\" &&
      cd \"$WORKDIR\" &&
      bash \"$script\"; 
      echo \"[${session_name}] finished with exit \$?\"; 
      echo \"Press Enter to keep shell open...\"; 
      read -r; 
      exec bash
    "
    echo "Started $script in detached session: $session_name (shell left open)"
  fi
done

echo "All sessions started."
echo "List sessions with: tmux ls"
echo "Attach to one with: tmux attach -t attacker_0"
