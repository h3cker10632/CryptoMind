#!/usr/bin/env bash
# Snapshot all CryptoMind persistent data. Safe to run while the app is live.
set -euo pipefail
cd "$(dirname "$0")"
DEST="backups/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$DEST"
# SQLite: use the online backup API so a live/mid-write DB copies consistently
# (a plain cp can catch a half-flushed WAL). Falls back to cp if sqlite3 missing.
if command -v sqlite3 >/dev/null 2>&1 && [ -f cryptomind.db ]; then
  sqlite3 cryptomind.db ".backup '$DEST/cryptomind.db'"
elif [ -f cryptomind.db ]; then
  cp cryptomind.db "$DEST/"
fi
# JSON/text state — copy whatever exists
for f in state.json settings.json tunables.json alerts.json api_token.txt; do
  [ -f "$f" ] && cp "$f" "$DEST/"
done
echo "Backed up to $DEST:"
ls -la "$DEST"
