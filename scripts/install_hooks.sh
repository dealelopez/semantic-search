#!/bin/sh
# install_hooks.sh — instala los git hooks versionados en .git/hooks.
# Uso: scripts/install_hooks.sh
set -eu
TOP="$(git rev-parse --show-toplevel)"
ln -sf ../../scripts/pre-commit "$TOP/.git/hooks/pre-commit"
chmod +x "$TOP/scripts/pre-commit" "$TOP/scripts/check_no_pii.py"
echo "Hook pre-commit instalado (.git/hooks/pre-commit -> scripts/pre-commit)."
