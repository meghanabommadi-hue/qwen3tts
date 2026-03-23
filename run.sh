#!/usr/bin/env bash
# Qwen3TTS — single-process WebSocket server launcher
#
# Usage:
#   ./run.sh                          # 1 port at 9765
#   ./run.sh --ports 100              # 100 ports: 9765…9864
#   ./run.sh --ports 3 --port 9000   # ports 9000, 9001, 9002
#   ./run.sh --ctrl-port 9764        # + HTTP control API at :9764
#   ./run.sh --save-audio /tmp/wav   # save decoded WAV files

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

exec python -m qwen3tts.server "$@"
