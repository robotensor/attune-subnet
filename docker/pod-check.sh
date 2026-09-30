#!/usr/bin/env bash
# Can this pod run a duel? The things a rented machine gets wrong, in under a minute: the GPUs, the
# driver's graphics libraries, /dev/shm, SAPIEN rendering, CuRobo's kernels, torch in the policy
# env, and the benchmark checkout. `robotensor doctor --config ...` checks the rest (chain, wallet,
# Hub token, disk, clock) once there is a config.
#
#   pod-check              # the quick checks
#   pod-check --selftest   # and rt-bench selftest: demo make, demo verify, a replay (about 70 s)
set -uo pipefail
SIM="${ROBOTWIN_BENCH_PYTHON:-/opt/miniforge3/envs/robotwin/bin/python}"
RT="${ROBOTWIN_BENCH_ROOT:-/opt/robotensor/vector/RoboTwin-Vector}"
POLICY=/opt/robotensor/.venvs/vector-policy/bin/python
failed=0
report() { printf '  [%-4s] %s%s\n' "$1" "$2" "${3:+: $3}"; [[ "$1" == FAIL ]] && failed=1; }
last() { tail -n 1 <<<"$1"; }

echo "pod-check: $(hostname)"
gpus="$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>&1)"
if [[ $? -eq 0 && -n "${gpus}" ]]; then
    report ok gpu "$(wc -l <<<"${gpus}") x $(sed -n 1p <<<"${gpus}")"
else
    report FAIL gpu "nvidia-smi: $(last "${gpus}")"
fi

if ldconfig -p | grep -q 'libGLX_nvidia\.so\.0' && ldconfig -p | grep -q 'libEGL_nvidia\.so\.0'; then
    report ok "graphics libraries" "libGLX_nvidia, libEGL_nvidia"
else
    report FAIL "graphics libraries" "the driver's Vulkan/EGL libraries were not mounted: the container needs NVIDIA_DRIVER_CAPABILITIES=all (the image sets it), and a host whose container toolkit allows graphics (--runtime=nvidia, or graphics in nvidia-container-runtime/config.toml)"
fi

shm_gb=$(( $(df -k --output=size /dev/shm | tail -n 1) / 1024 / 1024 ))
if (( shm_gb >= 1 )); then report ok "/dev/shm" "${shm_gb} GB"; else report warn "/dev/shm" "under 1 GB: start the container with --shm-size=16g or --ipc=host"; fi

out="$(cd "${RT}" && "${SIM}" scripts/test_render.py 2>&1)"
if grep -q "Render Well" <<<"${out}"; then report ok render "SAPIEN renders"; else report FAIL render "$(last "${out}") (check the NVIDIA Vulkan driver)"; fi

# torch first, as CuRobo imports it: the kernels link libc10 and friends from torch's own lib/.
out="$("${SIM}" -c 'import torch; from curobo.curobolib import kinematics_fused_cu, geom_cu, lbfgs_step_cu, line_search_cu, tensor_step_cu' 2>&1)"
if [[ $? -eq 0 ]]; then report ok curobo "prebuilt kernels import"; else report FAIL curobo "$(last "${out}")"; fi

out="$("${POLICY}" -c 'import torch; torch.ones(1, device="cuda"); print(torch.__version__, torch.cuda.get_device_name(0))' 2>&1)"
if [[ $? -eq 0 ]]; then report ok "policy torch" "$(last "${out}")"; else report FAIL "policy torch" "$(last "${out}")"; fi

out="$(vector-orchestrator check 2>&1)"
if [[ $? -eq 0 ]]; then
    report ok benchmark "$(jq -r '"harness \(.source_sha256[0:12]) at \(.commit[0:7])\(if (.commit | endswith("-dirty")) then ", uncommitted changes" else "" end)"' <<<"${out}")"
else
    report FAIL benchmark "$(last "${out}")"
fi

if [[ "${1:-}" == "--selftest" ]]; then
    out="$(rt-bench selftest 2>&1)"
    if [[ $? -eq 0 ]]; then report ok selftest "$(last "${out}")"; else report FAIL selftest "$(last "${out}")"; fi
fi

if (( failed )); then echo "Something above must be fixed before this pod can run a duel."; exit 1; fi
echo "This pod can run the simulator and the policy."
