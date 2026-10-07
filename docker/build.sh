#!/usr/bin/env bash
# Build the validator image from the checkouts as they are on this machine: tracked files and
# untracked ones git does not ignore, with their .git, so a pod can `git pull` them later.
#
#   bash docker/build.sh                                        # robotensor-validator:<date>-<sha>
#   IMAGE=docker.io/me/robotensor-validator bash docker/build.sh --push
#
#   ROBOTWIN=... ORCHESTRATOR=...   the other two checkouts (default: found beside this one)
#   ASSETS=...                      unzipped RoboTwin assets to copy in (default: the checkout's
#                                   assets/, when it has them; otherwise the build downloads 15 GB)
#   GPU_PATH=reference CUDA_ARCH="8.0;8.6;8.9;9.0+PTX"   the pre-Blackwell stack (Dockerfile)
#
# Anything else on the command line goes to `docker buildx build`.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBNET="$(cd "${HERE}/.." && pwd)"
first_dir() { local d; for d in "$@"; do [[ -d "${d}/.git" ]] && { (cd "${d}" && pwd); return; }; done; }
ROBOTWIN="${ROBOTWIN:-$(first_dir "${SUBNET}/../vector/RoboTwin-Vector" "${SUBNET}/../RoboTwin-Vector")}"
ORCHESTRATOR="${ORCHESTRATOR:-$(first_dir "${SUBNET}/../vector/vector-orchestrator" "${SUBNET}/../vector-orchestrator")}"
: "${ROBOTWIN:?no RoboTwin-Vector checkout beside this one: set ROBOTWIN}"
: "${ORCHESTRATOR:?no vector-orchestrator checkout beside this one: set ORCHESTRATOR}"
ASSETS="${ASSETS:-${ROBOTWIN}/assets}"
IMAGE="${IMAGE:-robotensor-validator}"
TAG="${TAG:-$(date -u +%Y%m%d)-$(git -C "${SUBNET}" rev-parse --short HEAD)}"
log() { printf '\033[95m[build]\033[0m %s\n' "$*"; }

CONTEXT="$(mktemp -d)"
trap 'rm -rf "${CONTEXT}"' EXIT
for src in "${ROBOTWIN}" "${ORCHESTRATOR}" "${SUBNET}"; do
    name="$(basename "${src}")"
    [[ -z "$(git -C "${src}" status --porcelain)" ]] ||
        log "${name}: uncommitted changes go into the image, and its runs will record a dirty commit"
    ahead="$(git -C "${src}" rev-list --count '@{upstream}..HEAD' 2>/dev/null || echo 0)"
    (( ahead == 0 )) ||
        log "${name}: ${ahead} commit(s) not pushed: in the image, but ROBOTENSOR_PULL=1 cannot fast-forward past them until they are"
    mkdir -p "${CONTEXT}/${name}"
    git -C "${src}" ls-files -z --cached --others --exclude-standard |
        tar -C "${src}" --null --ignore-failed-read -T - -cf - 2>/dev/null | tar -C "${CONTEXT}/${name}" -xf -
    tar -C "${src}" -cf - .git | tar -C "${CONTEXT}/${name}" -xf -
done

args=(-f "${CONTEXT}/$(basename "${SUBNET}")/docker/Dockerfile" -t "${IMAGE}:${TAG}")
[[ -n "${GPU_PATH:-}" ]] && args+=(--build-arg "GPU_PATH=${GPU_PATH}")
[[ -n "${CUDA_ARCH:-}" ]] && args+=(--build-arg "CUDA_ARCH=${CUDA_ARCH}")
if [[ -d "${ASSETS}/objects" && -d "${ASSETS}/embodiments" && -d "${ASSETS}/background_texture" ]]; then
    log "assets from ${ASSETS}"
    args+=(--build-context "robotwin-assets=${ASSETS}")
else
    log "no unzipped assets at ${ASSETS}: the build downloads them (15 GB)"
fi
log "building ${IMAGE}:${TAG}"
docker buildx build "${args[@]}" "$@" "${CONTEXT}"
log "built ${IMAGE}:${TAG}"
