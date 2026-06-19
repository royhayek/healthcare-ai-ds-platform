#!/usr/bin/env bash
#
# Reproducible backend test runner.
#
# Why this script exists:
#   weasyprint (used by the PDF deliverable generators) links against the
#   Homebrew-installed pango/cairo/glib stack. On Apple Silicon those libraries
#   live under $(brew --prefix)/lib, which Python's ctypes loader does NOT search
#   by default, so weasyprint raises "cannot load library 'libpango-1.0-0'".
#   Exporting DYLD_FALLBACK_LIBRARY_PATH points the loader at them.
#
#   It also pins execution to the project venv (backend/.venv) rather than a
#   global/Anaconda interpreter. Anaconda ships its own incompatible copies of
#   glib/cairo/harfbuzz that segfault when mixed with Homebrew's pango, so the
#   venv must be a clean (non-conda) interpreter — e.g. the python.org build:
#       /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m venv .venv
#       .venv/bin/python -m pip install -e ".[dev]"   # or the pinned deps
#
# Usage:
#   ./run_tests.sh                 # full unit suite
#   ./run_tests.sh -k profiler     # any extra args are forwarded to pytest
#   ./run_tests.sh -m integration  # integration suite (needs Postgres + Redis)
set -euo pipefail

cd "$(dirname "$0")"

VENV_PY="./.venv/bin/python"
if [[ ! -x "$VENV_PY" ]]; then
  echo "error: $VENV_PY not found. Create it with a non-Anaconda interpreter:" >&2
  echo "  /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m venv .venv" >&2
  echo "  .venv/bin/python -m pip install -e \".[dev]\"" >&2
  exit 1
fi

# Make Homebrew's native libs discoverable to weasyprint without clobbering any
# value the caller already set.
if command -v brew >/dev/null 2>&1; then
  BREW_LIB="$(brew --prefix)/lib"
  export DYLD_FALLBACK_LIBRARY_PATH="${BREW_LIB}${DYLD_FALLBACK_LIBRARY_PATH:+:$DYLD_FALLBACK_LIBRARY_PATH}"
fi

exec "$VENV_PY" -m pytest "$@"
