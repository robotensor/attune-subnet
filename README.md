# robotensor-subnet

The Robotensor Bittensor subnet: miners fine-tune robot foundation models and submit their
**weights**; validators score them in simulation and set weights on chain. One subnet, one lane per
competition; the first is Vector (Vector policy).

Work in progress: the protocol (commitments, seeds, the weight vector) first, then the miner and the
validator.

## Development

```bash
uv venv --python 3.12 .venv && uv pip install -e ".[dev]"
ruff check . && ruff format --check .
pytest -q
```
