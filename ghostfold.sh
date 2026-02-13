#!/usr/bin/env bash

# Compatibility wrapper:
# - Preserves legacy option-only usage (`./ghostfold.sh --project_name ...`)
# - Routes execution through the packaged Typer CLI implementation.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

if [[ -d "${SCRIPT_DIR}/src" ]]; then
    export PYTHONPATH="${SCRIPT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
fi

if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: python3 command not found." >&2
    exit 1
fi

# Legacy mode: no explicit subcommand, only options.
if [[ $# -eq 0 || "${1}" == -* ]]; then
    exec python3 -m ghostfold.cli run "$@"
fi

# Pass explicit subcommands through unchanged.
exec python3 -m ghostfold.cli "$@"
