#!/usr/bin/env bash
# Render every *.html in this directory to a sibling *.pdf using headless Chrome.
# macOS: looks for Google Chrome.app. Linux: looks for `google-chrome` / `chromium`.
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

CHROME=""
if [[ "$(uname)" == "Darwin" ]]; then
    CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
elif command -v google-chrome &>/dev/null; then
    CHROME="$(command -v google-chrome)"
elif command -v chromium &>/dev/null; then
    CHROME="$(command -v chromium)"
fi

if [[ -z "$CHROME" || ! -x "$CHROME" ]]; then
    echo "❌ Google Chrome / Chromium not found. Install one of:"
    echo "    macOS: https://www.google.com/chrome/"
    echo "    Linux: apt install chromium-browser  (or equivalent)"
    exit 1
fi

shopt -s nullglob
for html in *.html; do
    pdf="${html%.html}.pdf"
    echo "→ $html → $pdf"
    "$CHROME" \
        --headless \
        --disable-gpu \
        --no-pdf-header-footer \
        --print-to-pdf="$DIR/$pdf" \
        --print-to-pdf-no-header \
        "file://$DIR/$html" 2>/dev/null
done

echo "✓ done"
