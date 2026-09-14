#!/bin/bash
# CryptoMind supervisor — keeps the server alive across restarts & crashes.
# - "Restart server" button: process exits, this loop relaunches it (picking
#   up any code changes).
# - "Kill server" button: writes .shutdown marker, loop exits for real.
# - Crashes: relaunched after 2s (state.json restores everything).
cd "$(dirname "$0")"
rm -f .shutdown
export CRYPTOMIND_SUPERVISED=1
echo "[supervisor] starting CryptoMind"
while true; do
  python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
  code=$?
  if [ -f .shutdown ]; then
    rm -f .shutdown
    echo "[supervisor] shutdown requested — exiting for real (code $code)"
    break
  fi
  echo "[supervisor] server exited (code $code) — restarting in 2s"
  sleep 2
done
