<p align="center">
  <img src="assets/header.png" alt="Attune: the open frontier for adaptive robot intelligence" width="100%">
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue.svg" alt="License: Apache-2.0"></a>
  <img src="https://img.shields.io/badge/python-3.11%2B-blue.svg" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/bittensor-subnet-black.svg" alt="Bittensor subnet">
</p>

<p align="center">
  <a href="#how-it-works">How it works</a> ·
  <a href="#mining">Mining</a> ·
  <a href="#validating">Validating</a> ·
  <a href="#development">Development</a>
</p>

---

**Attune** is a Bittensor subnet for robot foundation models, built by [Robotensor](https://www.robotensor.ai). Miners submit policy weights;
validators score them in simulation and set weights on chain.

The first competition is **Vector**: a policy (`vector_v1.1`) is shown one
demonstration of a task and must complete the same task in a different scene.

## How it works

| | |
|---|---|
| **Submission** | A Hugging Face model repo containing only `model.safetensors` (and an optional README). |
| **Commitment** | `vector:<owner>/<repo>@<commit>.<digest>` on chain: the revision and the sha256 of `model.safetensors`, in base64url. `attune miner commit` builds it; repo ids can be up to 49 characters long. |
| **Queue** | The validator queues a commitment as soon as it reads it from the chain, oldest first. **One submission per hotkey**: once a hotkey is queued, its later commitments are refused. |
| **Scoring** | King of the hill. Each queued submission duels the current king on 160 units (16 tasks × 10), seeded from the finalized block when the duel starts. That block must come after the commitment. The challenger takes the crown when its average success rate beats the king's by 3+ points. |
| **Rewards** | The lane's share goes to the 4 most recent champions: 40% / 30% / 20% / 10%, newest first. |

### Repositories and the kings' weights

A miner's repo is gated, with the validator's Hugging Face account let in and nobody else. The
validator checks it once, when the entry's turn comes, and downloads the weights then.

- **Challenger not downloadable.** If the validator can't download the challenger's repo at its
  turn (deleted, private, gated without the validator let in, or the revision is gone), the entry is
  dropped as `missing` and the validator duels the next one. The hotkey's one submission is used up.
- **Hub not answering.** A Hub that doesn't answer is asked again, three times in all at least 90
  seconds apart, before the entry is dropped as `missing`.
- **Kings.** From the download on, the duel runs on the validator's copy, and nothing asks the miner
  to keep the repo up. The four newest crowned models - the four that are paid - are kept by the
  validator (`var/<network>/vector/kings/`) and mirrored to a public model repo (`kings_repo`), so
  anyone can take the weights a crown was won with. A new king pushes the oldest of the four out.

Submissions are weights only. Nothing a miner uploads is executed or unpickled: the safetensors
header is checked against the pinned architecture before any tensor is read. Weights that don't
hash to the committed digest are refused. Weights identical to an earlier commitment's are refused
as duplicates, and the earlier commitment keeps them.

## Components

| Repo | Role |
|---|---|
| [`attune-subnet`](https://github.com/robotensor/attune-subnet) | Chain side: commitments, queue, seeds, weights, validator loop, miner CLI |
| [`vector-orchestrator`](https://github.com/robotensor/vector-orchestrator) | Duel engine, result store, `vector-runtime` and `vector-protocol` |
| [`RoboTwin-Vector`](https://github.com/robotensor/RoboTwin-Vector) | RoboTwin 2.0 fork with the level gate and the benchmark harness |

## Mining

A submission is a `vector_v1.1` weights file: `model.safetensors`, with exactly the tensors the
architecture pins (`vector_runtime/vector_v1.1.json` in the orchestrator).

```bash
pip install robotensor-attune

attune miner check --dir submission/

export HF_TOKEN=...
attune miner submit --dir submission/ --repo <you>/vector-mine \
    --network finney --netuid <N> --wallet.name miner --wallet.hotkey default
```

`submit` creates the repo gated (manual approval), lets the validator's account in, uploads, then
commits on chain. Nobody else can download your weights; if you are crowned, the validator publishes
them in its kings repo. The repo must be downloadable by the validator at the one moment its turn
comes: a repo it can't download then is dropped as missing.

Check your entry with:

```bash
attune miner status --hotkey <ss58> --network finney --netuid <N>
```

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
git clone https://github.com/robotensor/attune-subnet.git

uv venv --python 3.12 .venvs/subnet
uv pip install --python .venvs/subnet/bin/python -e attune-subnet -e vector-orchestrator \
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
kings_repo = "<validator-account>/vector-kings"   # public model repo the four kings are mirrored to
```

```bash
export HF_TOKEN=...
attune validator --config config/<network>.toml run
```

`status`, `weights --dry-run` and `duel --challenger owner/name@sha` are also available.
Interrupted duels resume from `var/<network>/`. A validator started on another host with the
same `mirror` takes the competition up from that dataset (the index, the king and who committed
what; not the events or clips), and will not start on an empty store while the dataset cannot be
read.

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

### Releasing the miner package

Miners install `robotensor-attune` from PyPI: the `attune` command with `init` and `miner`
only, and none of the validator. `miner/build.sh` lays out the miner's modules and copies
vector-runtime's weights check from the vector-orchestrator checkout beside this one; it refuses
a tensor manifest that is not the one `spec.json` pins. The version is `__version__` in
`src/robotensor/__init__.py`.

```bash
bash miner/build.sh                     # miner/dist/robotensor_attune-<version>-py3-none-any.whl
twine upload miner/dist/robotensor_attune-<version>-py3-none-any.whl
```

Rebuild and upload whenever the miner's commands or the architecture change. Validators do not
install it: their install from source has the same commands.

## License

[Apache-2.0](LICENSE)
