# FineTune on Apple Macbook

Repository for experimenting with fine-tuning / LoRA adapters for large language models using Apple's Silicon supported Macbooks.

This repo contains a notebook and helper modules used to prepare data and run lightweight fine-tuning (LoRA-style) experiments on a sharded safetensors model. It bundles example data and a local model snapshot for quick testing.

## Quick start

Prerequisites

- Python 3.10 or newer
- M2 Pro or above (Tested on M3 Max)

Recommended minimal dependencies (examples — install in a virtualenv):

```bash
pip install uv #install uv global
uv sync # install all the packages and create .venv
```

The notebook walks through loading the model shards, tokenization, preparing the `datasets`-compatible datasets from the local JSONL files, configuring LoRA/PEFT training, running a short training job, and saving adapters/checkpoints.

## Troubleshooting

- Out of memory: reduce batch size, use gradient accumulation, enable 8-bit load or mixed precision, or test with a smaller model.

## License & Attribution

This repository is a research/experiment scaffold. Check any included model files and third-party code for their individual licenses before reuse.

[Apple MLX Github](https://github.com/ml-explore/mlx)
[Apple MLX Repo](https://github.com/ml-explore/mlx-examples)