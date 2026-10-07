# syntax=docker/dockerfile:1.7
# A validator host in one image: the simulator, policy and host environments, RoboTwin's assets
# and the three checkouts, so that a rented GPU pod (vast.ai, lium.io) is ready when it starts
# instead of an hour of installs and downloads later.
#
# Built by docker/build.sh, which stages the checkouts side by side as the build context:
#
#   RoboTwin-Vector/   vector-orchestrator/   attune-subnet/
#
# Everything lives under /opt, laid out as a hand-built host lays out /root/robotensor, and
# /root/robotensor and /root/miniforge3 are links to it: lium mounts the pod's volume over /root,
# which would hide anything the image put there, and pod-init puts the links back.

ARG UBUNTU_VERSION=24.04
# install_robotwin.sh's GPU path, and the compute capabilities CuRobo's kernels are built for: a
# docker build has no GPU to ask. blackwell (torch 2.8.0+cu128) runs on Ampere through Blackwell;
# reference (torch 2.4.1+cu121, RoboTwin's pin) runs below 10.0 only, and takes "8.0;8.6;8.9;9.0+PTX".
ARG GPU_PATH=blackwell
ARG CUDA_ARCH="8.0;8.6;8.9;9.0;10.0;12.0+PTX"

# ---- base: what every stage and the pod need from the distribution --------------------------
FROM ubuntu:${UBUNTU_VERSION} AS base
ENV DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
# libegl1 and libvulkan1: SAPIEN renders through the NVIDIA Vulkan driver; libgl1 and the X
# libraries: opencv-python and open3d link them.
RUN apt-get update && apt-get install -y -q --no-install-recommends \
        ca-certificates curl git unzip bzip2 python3 \
        libegl1 libvulkan1 libgl1 libglib2.0-0t64 libgomp1 libsm6 libxext6 libxrender1 libx11-6 libxfixes3 \
        openssh-server tini tmux htop jq rsync less vim-tiny procps iproute2 \
    && rm -rf /var/lib/apt/lists/* /etc/ssh/ssh_host_*
COPY --from=ghcr.io/astral-sh/uv:0.12.10 /uv /uvx /usr/local/bin/

# ---- sim: the simulator env, by the harness's own installer ---------------------------------
FROM base AS sim
ARG GPU_PATH
ARG CUDA_ARCH
ENV CONDA_ROOT=/opt/miniforge3 CONDA_PKGS_DIRS=/root/.cache/conda-pkgs
WORKDIR /opt/robotensor/vector/RoboTwin-Vector
# Only what install_robotwin.sh reads, so a commit that leaves them alone reuses this layer.
COPY RoboTwin-Vector/robotensor_bench/scripts/ robotensor_bench/scripts/
COPY RoboTwin-Vector/robotensor_bench/requirements.txt robotensor_bench/
COPY RoboTwin-Vector/scripts/requirements.txt RoboTwin-Vector/scripts/test_render.py scripts/
COPY RoboTwin-Vector/envs/_base_task.py envs/
RUN --mount=type=cache,target=/root/.cache/pip --mount=type=cache,target=/root/.cache/conda-pkgs \
    ROBOTWIN_GPU_PATH="${GPU_PATH}" ROBOTWIN_CUDA_ARCH="${CUDA_ARCH}" \
    ROBOTWIN_SKIP_RENDER_CHECK=1 ROBOTWIN_SKIP_ASSETS=1 \
    bash robotensor_bench/scripts/install_robotwin.sh
# The CUDA toolkit is there to build CuRobo, whose kernels link only libcudart; torch brings its
# own CUDA libraries. Drop the profilers, the static libraries and the math libraries (4 GB), and
# conda's package cache, which it refills if it ever installs again.
RUN env=/opt/miniforge3/envs/robotwin \
    && rm -rf "${env}"/nsight-compute-* "${env}/nsightee_plugins" \
    && find "${env}/targets" -name '*.a' -delete \
    && for lib in cublas nvblas cusparse cufft cusolver curand npp nvjpeg; do \
           rm -f "${env}/targets/x86_64-linux/lib/lib${lib}"*; done \
    && rm -rf /opt/miniforge3/pkgs

# ---- policy and host: the two uv venvs, at the versions of their lock files -----------------
FROM base AS policy
COPY attune-subnet/docker/policy.lock.txt /tmp/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv -q --python /usr/bin/python3.12 /opt/robotensor/.venvs/vector-policy \
    && uv pip install -q --python /opt/robotensor/.venvs/vector-policy/bin/python \
        --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match \
        -r /tmp/policy.lock.txt

FROM base AS host
COPY attune-subnet/docker/host.lock.txt /tmp/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv -q --python /usr/bin/python3.12 /opt/robotensor/.venvs/subnet \
    && uv pip install -q --python /opt/robotensor/.venvs/subnet/bin/python -r /tmp/host.lock.txt

# ---- robotwin-assets: TianxingChen/RoboTwin2.0, unzipped -------------------------------------
# build.sh replaces this stage with a local assets directory when one is there
# (--build-context robotwin-assets=RoboTwin-Vector/assets), which saves a 15 GB download.
FROM base AS assets-download
RUN --mount=type=cache,target=/root/.cache/uv --mount=type=cache,target=/zips \
    uvx -q --from "huggingface_hub>=1.0" hf download TianxingChen/RoboTwin2.0 \
        background_texture.zip embodiments.zip objects.zip --repo-type dataset --local-dir /zips \
    && mkdir /out && for zip in background_texture embodiments objects; do \
           unzip -q "/zips/${zip}.zip" -d /out && rm "/zips/${zip}.zip"; done

FROM scratch AS robotwin-assets
COPY --from=assets-download /out/ /

# ---- the image ------------------------------------------------------------------------------
FROM base
# graphics: the NVIDIA container toolkit mounts the driver's Vulkan and EGL libraries only when
# asked, and SAPIEN cannot render without them.
ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=all \
    CONDA_ROOT=/opt/miniforge3 \
    ROBOTWIN_BENCH_ROOT=/opt/robotensor/vector/RoboTwin-Vector \
    ROBOTWIN_BENCH_PYTHON=/opt/miniforge3/envs/robotwin/bin/python
ARG RT=/opt/robotensor/vector/RoboTwin-Vector

# Assets first: 16 GB that never change. Split into layers of 1 to 6 GB: registries refuse layers
# over 10 GB (GHCR), and a pull fetches layers in parallel.
COPY --link --from=robotwin-assets objects/ ${RT}/assets/objects/
COPY --link --from=robotwin-assets background_texture/seen/[0-4]* ${RT}/assets/background_texture/seen/
COPY --link --from=robotwin-assets background_texture/seen/[5-9]* ${RT}/assets/background_texture/seen/
COPY --link --from=robotwin-assets background_texture/unseen/ ${RT}/assets/background_texture/unseen/
COPY --link --from=robotwin-assets embodiments/ ${RT}/assets/embodiments/

# The environments. install_robotwin.sh installs CuRobo editable from envs/curobo, and writes the
# NVIDIA Vulkan and EGL manifests under /usr/share when the system has none.
COPY --link --from=sim /opt/miniforge3 /opt/miniforge3
COPY --link --from=sim ${RT}/envs/curobo ${RT}/envs/curobo
COPY --from=sim /usr/share/vulkan/icd.d/ /usr/share/vulkan/icd.d/
COPY --from=sim /usr/share/glvnd/egl_vendor.d/ /usr/share/glvnd/egl_vendor.d/
COPY --link --from=policy /opt/robotensor/.venvs/vector-policy /opt/robotensor/.venvs/vector-policy
COPY --link --from=host /opt/robotensor/.venvs/subnet /opt/robotensor/.venvs/subnet

# The checkouts, last: they change most.
COPY RoboTwin-Vector/ ${RT}/
COPY vector-orchestrator/ /opt/robotensor/vector/vector-orchestrator/
COPY attune-subnet/ /opt/robotensor/attune-subnet/
RUN --mount=type=cache,target=/root/.cache/uv \
    bash /opt/robotensor/attune-subnet/docker/install-checkouts.sh \
    && (cd "${RT}" && /opt/miniforge3/envs/robotwin/bin/python scripts/update_embodiment_config_path.py >/dev/null) \
    && ln -s /opt/miniforge3/envs/robotwin /opt/robotensor/.venvs/robotwin-bench \
    && ln -s /opt/robotensor /root/robotensor && ln -s /opt/miniforge3 /root/miniforge3 \
    && for cmd in pod-init pod-check rt-bench; do \
           ln -s "/opt/robotensor/attune-subnet/docker/${cmd}.sh" "/usr/local/bin/${cmd}"; done \
    && for cmd in attune robotensor vector-orchestrator vector-runtime; do \
           ln -s "/opt/robotensor/.venvs/subnet/bin/${cmd}" "/usr/local/bin/${cmd}"; done \
    && printf 'export %s\n' CONDA_ROOT="${CONDA_ROOT}" ROBOTWIN_BENCH_ROOT="${ROBOTWIN_BENCH_ROOT}" \
        ROBOTWIN_BENCH_PYTHON="${ROBOTWIN_BENCH_PYTHON}" > /etc/profile.d/robotensor.sh

WORKDIR /opt/robotensor
EXPOSE 22
# tini reaps the orphans a long run leaves; pod-init failing must not stop the pod, or there is
# nothing to log into to see why.
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["bash", "-c", "pod-init; exec sleep infinity"]
