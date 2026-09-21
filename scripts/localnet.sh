#!/usr/bin/env bash
# A local subtensor chain in Docker, for rehearsing the whole subnet loop without TAO.
#
#   bash scripts/localnet.sh up      # start it (fast blocks: 250 ms), RPC on ws://127.0.0.1:9944
#   bash scripts/localnet.sh down    # stop and remove it
#   bash scripts/localnet.sh logs
#
# Then `python scripts/localnet_setup.py` creates the subnet, the wallets and the registrations and
# writes config/localnet.toml's netuid. Nothing here touches finney or testnet.
set -euo pipefail
NAME=${LOCALNET_NAME:-robotensor-localnet}
IMAGE=${LOCALNET_IMAGE:-ghcr.io/opentensor/subtensor-localnet:devnet-ready}
case "${1:-up}" in
  up)
    if docker ps --format '{{.Names}}' | grep -qx "${NAME}"; then
      echo "${NAME} is running"
    else
      docker rm -f "${NAME}" >/dev/null 2>&1 || true
      docker run -d --name "${NAME}" -p 9944:9944 -p 9945:9945 "${IMAGE}" >/dev/null
      echo "started ${NAME} (${IMAGE}); waiting for its RPC"
    fi
    for _ in $(seq 1 60); do
      if curl -s -m 2 -H 'Content-Type: application/json' \
          -d '{"id":1,"jsonrpc":"2.0","method":"chain_getHeader","params":[]}' \
          http://127.0.0.1:9944 | grep -q '"number"'; then
        echo "ws://127.0.0.1:9944 answers"; exit 0
      fi
      sleep 2
    done
    echo "the localnet did not answer in 120 s; see: docker logs ${NAME}" >&2; exit 1 ;;
  down) docker rm -f "${NAME}" ;;
  logs) docker logs --tail 100 -f "${NAME}" ;;
  *) echo "usage: $0 up|down|logs" >&2; exit 2 ;;
esac
