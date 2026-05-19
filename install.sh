#!/usr/bin/env bash
# install.sh — Add this repo to PATH so cnn.py is callable directly.
# Run once from anywhere: bash /path/to/esp-computer-vision/install.sh

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MARKER="# esp-computer-vision PATH"
BASHRC="$HOME/.bashrc"

if grep -qF "$MARKER" "$BASHRC" 2>/dev/null; then
    echo "Already installed ($BASHRC already contains the PATH entry)."
else
    printf '\n%s\nexport PATH="%s:$PATH"\n' "$MARKER" "$REPO_DIR" >> "$BASHRC"
    echo "Added $REPO_DIR to PATH in $BASHRC."
fi

echo "Run 'source ~/.bashrc' or open a new terminal, then use: cnn.py <tool>"
