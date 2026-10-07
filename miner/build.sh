#!/usr/bin/env bash
# Build the miner's package, robotensor-attune: the `attune` command with `init` and `miner`
# only, and none of the validator. `attune miner check` is vector-runtime's own check,
# copied from the vector-orchestrator checkout beside this one, and the build refuses a tensor
# manifest that is not the one the orchestrator's spec.json pins, so a miner's check is the
# validator's.
#
#   bash miner/build.sh                  # the wheel, in miner/dist/
#   bash miner/build.sh --stage DIR      # only lay the package's sources out in DIR
#   ORCHESTRATOR=/path/to/vector-orchestrator bash miner/build.sh
#
# Then publish it: twine upload miner/dist/robotensor_attune-<version>-py3-none-any.whl
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBNET="$(cd "${HERE}/.." && pwd)"
first_dir() { local d; for d in "$@"; do [[ -d "${d}/.git" ]] && { (cd "${d}" && pwd); return; }; done; }
ORCHESTRATOR="${ORCHESTRATOR:-$(first_dir "${SUBNET}/../vector/vector-orchestrator" "${SUBNET}/../vector-orchestrator")}"
: "${ORCHESTRATOR:?no vector-orchestrator checkout beside this one: set ORCHESTRATOR}"
RUNTIME="${ORCHESTRATOR}/packages/vector-runtime/src/vector_runtime"
log() { printf '\033[95m[miner]\033[0m %s\n' "$*"; }

# What the miner's commands import, and nothing else.
SUBNET_FILES=(__init__.py miner_cli.py miner.py hub.py chain.py init.py
    protocol/__init__.py protocol/commitment.py)
RUNTIME_FILES=(__init__.py check.py header.py vector_v1.1.json)

stage() {
    local out="$1"
    rm -rf "${out}"
    mkdir -p "${out}/src/robotensor/protocol" "${out}/src/vector_runtime"
    cp "${HERE}/pyproject.toml" "${SUBNET}/LICENSE" "${out}/"
    for f in "${SUBNET_FILES[@]}"; do cp "${SUBNET}/src/robotensor/${f}" "${out}/src/robotensor/${f}"; done
    for f in "${RUNTIME_FILES[@]}"; do cp "${RUNTIME}/${f}" "${out}/src/vector_runtime/${f}"; done
}

# The manifest is the architecture's tensors; the spec pins its sha256, and the validator holds
# every submission to the pinned one.
[[ -z "$(git -C "${ORCHESTRATOR}" status --porcelain -- packages/vector-runtime/src/vector_runtime spec.json)" ]] ||
    { log "${ORCHESTRATOR} has uncommitted changes to the weights check or spec.json"; exit 1; }
python3 - "${ORCHESTRATOR}/spec.json" "${RUNTIME}" <<'EOF'
import hashlib, json, pathlib, sys
model = json.loads(pathlib.Path(sys.argv[1]).read_text())["model"]
manifest = pathlib.Path(sys.argv[2]) / f"{model['architecture']}.json"
if not manifest.is_file():
    sys.exit(f"spec.json pins {model['architecture']}, and vector-runtime has no {manifest.name}")
if hashlib.sha256(manifest.read_bytes()).hexdigest() != model["tensors_sha256"]:
    sys.exit(f"{manifest.name} is not the manifest spec.json pins (model.tensors_sha256)")
EOF
log "weights check from vector-orchestrator $(git -C "${ORCHESTRATOR}" log --oneline -1)"

if [[ "${1:-}" == "--stage" ]]; then
    stage "${2:?--stage needs a directory}"
    exit 0
fi

STAGE="$(mktemp -d)"
trap 'rm -rf "${STAGE}"' EXIT
stage "${STAGE}"
uv build --wheel --out-dir "${HERE}/dist" "${STAGE}"
log "built $(ls -t "${HERE}"/dist/*.whl | head -1)"
