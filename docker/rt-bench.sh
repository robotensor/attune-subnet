#!/usr/bin/env bash
# `rt-bench ...`: RoboTwin-Vector's harness from its checkout, under the simulator's interpreter.
set -euo pipefail
export PYTHONPATH="${ROBOTWIN_BENCH_ROOT:-/opt/robotensor/vector/RoboTwin-Vector}${PYTHONPATH:+:${PYTHONPATH}}"
exec "${ROBOTWIN_BENCH_PYTHON:-/opt/miniforge3/envs/robotwin/bin/python}" -m robotensor_bench.cli "$@"
