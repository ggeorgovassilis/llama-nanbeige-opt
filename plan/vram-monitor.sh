#!/usr/bin/env bash
# Sample NVIDIA GPU VRAM while a command runs; report the peak.
# Usage: vram-monitor.sh [--tag NAME] -- <command...>
# Writes a timestamped log to plan/logs/<tag>-<stamp>.vram.log and prints the peak.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$REPO_DIR/plan/logs"
mkdir -p "$LOG_DIR"

TAG="run"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --tag) TAG="${2:?missing tag for --tag}"; shift 2 ;;
        --) shift; break ;;
        -*) echo "error: unknown option: $1" >&2; exit 2 ;;
        *) break ;;
    esac
done

if [[ $# -eq 0 ]]; then
    echo "usage: vram-monitor.sh [--tag NAME] -- <command...>" >&2
    exit 2
fi

STAMP="$(date +%Y%m%d-%H%M%S)"
LOG="$LOG_DIR/${TAG}-${STAMP}.vram.log"
: > "$LOG"
echo "# command: $*" >> "$LOG"
echo "# started: $(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG"
echo "# ts used_mib total_mib util_pct" >> "$LOG"

INTERVAL=0.5
PEAK=0

"$@" &
PID=$!

while kill -0 "$PID" 2>/dev/null; do
    sample="$(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu \
        --format=csv,noheader,nounits 2>/dev/null || true)"
    if [[ -n "$sample" && "$sample" != *"No devices"* ]]; then
        IFS=',' read -r used total util <<< "$sample"
        used="${used// /}"; total="${total// /}"; util="${util// /}"
        ts="$(date +%H:%M:%S.%3N)"
        echo "$ts $used $total $util" >> "$LOG"
        if [[ "$used" =~ ^[0-9]+$ ]] && (( used > PEAK )); then
            PEAK=$used
        fi
    fi
    sleep "$INTERVAL"
done

RC=0
wait "$PID" || RC=$?

echo "vram-monitor: peak = ${PEAK} MiB (log: $LOG)"
echo "vram-monitor: command exit = $RC"
exit "$RC"
