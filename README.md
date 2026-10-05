# robotensor-subnet

Bittensor subnet for robot foundation models. Miners fine-tune a policy and submit its weights;
validators score submissions in simulation and set weights on chain.

The first competition is **Robotensor Vector**: a Vector policy (`vector_v1.1`) is
shown one demonstration of a task and must complete the same task in a different scene.

## How it works

| | |
|---|---|
| Submission | A Hugging Face model repo containing only `model.safetensors` (and an optional README) |
| Commitment | `vector:<owner>/<repo>@<sha>` on chain |
| Scoring | King of the hill. Each new submission duels the current king on 160 units (16 tasks × 10), seeded from the finalized block when the duel starts, which must come after the commitment. The challenger takes the crown when its average success rate beats the king's by 3+ points. |
| Rewards | The lane's share goes to the 4 most recent champions: 40% / 30% / 20% / 10%, newest first. |

Submissions are weights only. Nothing a miner uploads is executed or unpickled: the safetensors
header is checked against the pinned architecture before any tensor is read.

## Components

| Repo | Role |
|---|---|
| `robotensor-subnet` | Chain side: commitments, seeds, weights, validator loop, miner CLI |
| [`vector-orchestrator`](https://github.com/robotensor/vector-orchestrator) | Duel engine, result store, `vector-runtime` and `vector-protocol` |
| [`RoboTwin-Vector`](https://github.com/robotensor/RoboTwin-Vector) | RoboTwin 2.0 fork with the level gate and the benchmark harness |

## Mining

```bash
pip install "robotensor[vector]"

robotensor miner check --dir submission/

export HF_TOKEN=...
robotensor miner submit --dir submission/ --repo <you>/vector-mine \
    --network finney --netuid <N> --wallet.name miner --wallet.hotkey default
```

`submit` uploads to a private repo, commits on chain, then makes the repo public. The earliest
commitment of a given set of weights wins, so commit before publishing. Check your entry with
`robotensor miner status --hotkey <ss58> --network finney --netuid <N>`.

## Validating

One GPU host with three Python environments:

| Env | Python | Contents |
|---|---|---|
| simulator | 3.10 | RoboTwin-Vector (`robotensor_bench/scripts/install_robotwin.sh`), `vector-protocol` |
| policy | 3.12 | torch 2.8.0+cu128, `vector-runtime[model]`, `vector-protocol` |
| host | 3.12 | `robotensor`, `vector-orchestrator`, `vector-runtime` |

```bash
git clone -b vector https://github.com/robotensor/RoboTwin-Vector.git
git clone https://github.com/robotensor/vector-orchestrator.git
git clone https://github.com/robotensor/robotensor-subnet.git

uv venv --python 3.12 .venvs/subnet
uv pip install --python .venvs/subnet/bin/python -e robotensor-subnet -e vector-orchestrator \
    -e vector-orchestrator/packages/vector-protocol -e vector-orchestrator/packages/vector-runtime
```

Point `[vector]` in `config/<network>.toml` at the environments and your wallet:

```toml
network = "finney"   # or "test"
netuid = 0
wallet = { name = "validator", hotkey = "default" }

[vector]
policy_python = "/abs/.venvs/vector-policy/bin/python"
simulator_python = "/abs/.venvs/robotwin/bin/python"
simulator_root = "/abs/RoboTwin-Vector"
workers = 1   # units per GPU; one unit can use up to 34 GB
```

```bash
robotensor doctor --config config/<network>.toml
export HF_TOKEN=...
robotensor validator --config config/<network>.toml run
```

`status`, `weights --dry-run` and `duel --challenger owner/name@sha` are available as well.
Interrupted duels resume from `var/<network>/`.

For rented GPU pods (vast.ai, lium.io), `docker/build.sh` builds an image with all three
environments, the assets and the checkouts. Run `pod-check --selftest` on a new pod.

## Development

```bash
uv venv --python 3.12 .venv && uv pip install -e ".[dev]"
ruff check . && ruff format --check .
pytest -q -m "not chain and not gpu"
```

`scripts/localnet.sh up` and `scripts/localnet_setup.py` run the full loop against a local
subtensor.

## License

Apache-2.0
