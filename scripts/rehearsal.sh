#!/usr/bin/env bash
# The whole subnet loop on a local chain, with the published test models:
#
#   1. a localnet with the subnet, the owner (validator) and two miners   (localnet.sh, localnet_setup.py)
#   2. miner2 commits robotensor/vector-update1@<sha> on chain               (robotensor-miner commit)
#   3. the validator takes it in, crowns robotensor/vector-base by genesis, duels the commitment against
#      it seeded from a block after the commitment, publishes the signed record   (validator run --once, twice)
#   4. the validator sets weights: the champion's hotkey gets the lane's share, the rest burns
#
#   UPDATE1_SHA=<sha> bash scripts/rehearsal.sh
#
# Needs config/localnet.toml's environments (simulator, policy) and HF_TOKEN with read access.
# A `light` duel (48 units a side) takes hours on one GPU; the genesis (16 units, one side) less.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY=${PY:-/root/robotensor/.venvs/subnet/bin/python}
CONFIG=config/localnet.toml
: "${UPDATE1_SHA:?set UPDATE1_SHA to the commit of robotensor/vector-update1}"

bash scripts/localnet.sh up
"${PY}" scripts/localnet_setup.py
NETUID=$(sed -n 's/^netuid = //p' "${CONFIG}")

"${PY}" -m robotensor_subnet.miner commit --repo robotensor/vector-update1 --revision "${UPDATE1_SHA}" \
    --netuid "${NETUID}" --network ws://127.0.0.1:9944 \
    --wallet.name miner2 --wallet.hotkey default --wallet.path var/wallets

# First step: intake, and the genesis of the baseline on the empty throne.
"${PY}" -m robotensor_subnet.validator --config "${CONFIG}" run --once --no-weights
# Second step: the duel of miner2's commitment against the king.
"${PY}" -m robotensor_subnet.validator --config "${CONFIG}" run --once --no-weights
"${PY}" -m robotensor_subnet.validator --config "${CONFIG}" status
"${PY}" -m robotensor_subnet.validator --config "${CONFIG}" weights
