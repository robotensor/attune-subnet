#!/usr/bin/env bash
# Install the image's checkouts into its environments, as docs/VALIDATOR.md does. The image's build
# runs it, and pod-init again after ROBOTENSOR_PULL=1 has pulled them. Each environment already
# holds the versions of its lock file, so this adds the checkouts themselves, and anything they
# have come to need since the lock was frozen.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT=/opt/robotensor
ORCH="${ROOT}/vector/vector-orchestrator"

# simulator: vector-protocol only, without its dependencies (RoboTwin pins numpy)
uv pip install -q --python /opt/miniforge3/envs/robotwin/bin/python --no-deps \
    -e "${ORCH}/packages/vector-protocol"
# policy
uv pip install -q --python "${ROOT}/.venvs/vector-policy/bin/python" -c "${HERE}/policy.lock.txt" \
    --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match \
    -e "${ORCH}/packages/vector-runtime[model]" -e "${ORCH}/packages/vector-protocol"
# host
uv pip install -q --python "${ROOT}/.venvs/subnet/bin/python" -c "${HERE}/host.lock.txt" \
    -e "${ROOT}/robotensor-subnet" -e "${ORCH}" \
    -e "${ORCH}/packages/vector-protocol" -e "${ORCH}/packages/vector-runtime"
