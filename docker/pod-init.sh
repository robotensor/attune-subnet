#!/usr/bin/env bash
# Make a pod of the validator image usable: the image's CMD runs it at start. vast.ai's SSH and
# Jupyter launch modes replace the CMD, so there it goes in the template's on-start script.
# Idempotent: running it again changes nothing that is already so.
#
#   PUBLIC_KEY / SSH_PUBLIC_KEY   authorised for root, and sshd started if nothing serves port 22
#   ROBOTENSOR_PULL=1             fast-forward the three checkouts first, then reinstall them
set -uo pipefail
log() { printf '\033[95m[pod-init]\033[0m %s\n' "$*"; }

# lium mounts the pod's volume over /root, which hides the links the image made there.
for link in /root/robotensor:/opt/robotensor /root/miniforge3:/opt/miniforge3; do
    if [[ ! -e "${link%%:*}" && ! -L "${link%%:*}" ]]; then
        ln -s "${link#*:}" "${link%%:*}"
        log "linked ${link%%:*} -> ${link#*:}"
    fi
done

# SSH sessions start from /etc/environment, not from the container's environment, so a token the
# pod was started with (HF_TOKEN, ...) would be missing from every shell a person opens.
env_file=/etc/environment
tmp="$(mktemp)"
grep -E '^PATH=' "${env_file}" > "${tmp}" 2>/dev/null || true
while IFS= read -r -d '' kv; do
    case "${kv%%=*}" in
        PATH | HOME | PWD | OLDPWD | SHLVL | _ | HOSTNAME | TERM | SSH_* | LS_COLORS | *[!A-Za-z0-9_]*) continue ;;
    esac
    [[ "${kv}" == *$'\n'* || "${kv}" == *'"'* ]] && continue
    printf '%s="%s"\n' "${kv%%=*}" "${kv#*=}" >> "${tmp}"
done < <(env -0)
install -m 600 "${tmp}" "${env_file}" && rm -f "${tmp}"

key="${PUBLIC_KEY:-${SSH_PUBLIC_KEY:-}}"
if [[ -n "${key}" ]]; then
    install -d -m 700 /root/.ssh
    touch /root/.ssh/authorized_keys && chmod 600 /root/.ssh/authorized_keys
    grep -qxF "${key}" /root/.ssh/authorized_keys || printf '%s\n' "${key}" >> /root/.ssh/authorized_keys
fi
# The image ships no host keys, so no two pods share them.
ssh-keygen -A >/dev/null
mkdir -p /run/sshd
if ! ss -Hltn 'sport = :22' | grep -q .; then
    /usr/sbin/sshd && log "sshd listening on 22"
fi

if [[ "${ROBOTENSOR_PULL:-0}" == "1" ]]; then
    for repo in /opt/robotensor/vector/RoboTwin-Vector /opt/robotensor/vector/vector-orchestrator \
        /opt/robotensor/robotensor-subnet; do
        git -C "${repo}" pull -q --ff-only && log "pulled ${repo}: $(git -C "${repo}" log --oneline -1)" \
            || log "could not fast-forward ${repo}; left at $(git -C "${repo}" log --oneline -1)"
    done
    bash /opt/robotensor/robotensor-subnet/docker/install-checkouts.sh && log "checkouts reinstalled"
fi

log "ready: $(nvidia-smi -L 2>/dev/null | wc -l) GPU(s); run pod-check to see that the simulator renders"
