#!/usr/bin/env bash
# The whole subnet loop on a local chain, with two vector_v1.1 submissions on the Hub:
#
#   1. a localnet with the subnet, the owner (validator) and two miners   (localnet.sh, localnet_setup.py)
#   2. miner1 commits KING, then miner2 commits CHALLENGER                  (robotensor miner commit)
#   3. the validator takes both in; KING, the older, takes the empty throne by genesis; CHALLENGER
#      duels it, seeded from a block after its commitment               (validator run --once, twice)
#   4. the validator sets weights: the champions' hotkeys get the lane's share, the rest burns
#
#   KING=owner/name@<sha> CHALLENGER=owner/name@<sha> bash scripts/rehearsal.sh
#
# Needs config/localnet.toml's environments (simulator, policy) and HF_TOKEN with read access.
# A duel (160 units a side) takes hours on one GPU; the genesis (one side) about half as long.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY=${PY:-/root/robotensor/.venvs/subnet/bin/python}
CONFIG=config/localnet.toml
: "${KING:?set KING to owner/name@<sha> of the first submission}"
: "${CHALLENGER:?set CHALLENGER to owner/name@<sha> of the second}"

bash scripts/localnet.sh up
"${PY}" scripts/localnet_setup.py
NETUID=$(sed -n 's/^netuid = //p' "${CONFIG}")

commit() {
    "${PY}" -m robotensor.miner commit --repo "${2%@*}" --revision "${2#*@}" \
        --netuid "${NETUID}" --network ws://127.0.0.1:9944 \
        --wallet.name "$1" --wallet.hotkey default --wallet.path var/wallets
}
commit miner1 "${KING}"
commit miner2 "${CHALLENGER}"

# First step: intake, and the genesis of the oldest entry on the empty throne.
"${PY}" -m robotensor.validator --config "${CONFIG}" run --once --no-weights
# Second step: the duel of the challenger against the king.
"${PY}" -m robotensor.validator --config "${CONFIG}" run --once --no-weights
"${PY}" -m robotensor.validator --config "${CONFIG}" status
"${PY}" -m robotensor.validator --config "${CONFIG}" weights
