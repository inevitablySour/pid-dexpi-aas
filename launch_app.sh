#!/usr/bin/env bash
# Double-click (or "Open With > Terminal") to launch the P&ID to DEXPI converter GUI.
# No command-line knowledge needed - this opens a browser tab automatically.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -x .venv/bin/python ]; then
    echo "First run: setting up the app (this only happens once, may take a minute)..."
    rm -rf .venv
    python3 -m venv .venv
    .venv/bin/pip install --quiet --upgrade pip
fi
# (re)install when pyproject.toml changed since the last install - e.g. after a git pull
if [ ! -f .venv/.installed ] || [ pyproject.toml -nt .venv/.installed ]; then
    .venv/bin/pip install --quiet -e ".[dev]"
    touch .venv/.installed
fi

# Streamlit prompts for an email on first run unless this file already exists - skip it
# so this script never blocks waiting for input when double-clicked.
mkdir -p "$HOME/.streamlit"
if [ ! -f "$HOME/.streamlit/credentials.toml" ]; then
    printf '[general]\nemail = ""\n' > "$HOME/.streamlit/credentials.toml"
fi

.venv/bin/streamlit run app.py "$@"
