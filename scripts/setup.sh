#!/usr/bin/env bash
# stereohand project setup — run once after cloning.
#
#   ./scripts/setup.sh
#
# Creates the .venv, installs the package + dev tooling, and enables the git
# hooks. Idempotent: safe to re-run. Override extras with EXTRAS, e.g.
# EXTRAS="dev,demo".
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

if ! command -v uv >/dev/null 2>&1; then
  echo "FATAL: uv is not installed. See https://github.com/astral-sh/uv" >&2
  exit 1
fi

EXTRAS="${EXTRAS:-dev,demo}"

echo "==> Creating virtualenv (.venv) and installing .[$EXTRAS] ..."
uv venv
uv pip install -e ".[$EXTRAS]"

echo "==> Enabling git hooks (core.hooksPath .githooks) ..."
git config core.hooksPath .githooks

echo "==> Done."
