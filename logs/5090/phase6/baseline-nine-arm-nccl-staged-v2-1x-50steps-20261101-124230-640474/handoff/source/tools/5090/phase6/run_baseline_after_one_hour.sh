#!/usr/bin/env bash
set -euo pipefail

rtx5090_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
rtx5090_build=/workspace/Load-5090/logs/5090/phase6/baseline-nine-arm-nccl-staged-v2-1x-50steps-20261101-013331-414102/build

if [[ ! -d "$rtx5090_build" ]]; then
    printf 'Baseline build directory not found: %s\n' "$rtx5090_build" >&2
    exit 1
fi

cd -- "$rtx5090_repo"
printf '[%s] Waiting 3600 seconds before starting the 5090 baseline.\n' "$(date '+%F %T %Z')"
sleep 3600
printf '[%s] Starting the 5090 baseline.\n' "$(date '+%F %T %Z')"
exec bash tools/5090/phase6/run_baseline.sh --execute --baseline-build "$rtx5090_build"
