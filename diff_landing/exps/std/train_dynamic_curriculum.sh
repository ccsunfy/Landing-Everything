#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash diff_landing/exps/std/train_dynamic_curriculum.sh
#   bash diff_landing/exps/std/train_dynamic_curriculum.sh BPTT my_curr
#   bash diff_landing/exps/std/train_dynamic_curriculum.sh BPTT my_curr 3000000,3000000,4000000
#   bash diff_landing/exps/std/train_dynamic_curriculum.sh BPTT my_curr 3000000,3000000,4000000 0.55,0.70,0.82

ALG="${1:-BPTT}"
TAG="${2:-dynamic_curriculum}"
STEPS="${3:-3000000,3000000,4000000}"
SUCCESS="${4:-0.55,0.70,0.82}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_PY="${SCRIPT_DIR}/run.py"

python "${RUN_PY}" \
  -t 1 \
  -e dynamicLanding \
  -a "${ALG}" \
  -c "${TAG}" \
  --curriculum "dynamicLanding_s1,dynamicLanding_s2,dynamicLanding_s3" \
  --stage_steps "${STEPS}" \
  --stage_success "${SUCCESS}" \
  --check_steps 200000 \
  --min_success_samples 100
