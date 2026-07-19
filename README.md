# Gemma 4 Sparse Autoencoder Training

[![CI](https://github.com/lamm-mit/gemma-sae/actions/workflows/ci.yml/badge.svg)](https://github.com/lamm-mit/gemma-sae/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB.svg)](https://www.python.org/)

A reproducible, single-node pipeline for training sparse autoencoders on the residual
stream of **Gemma 4 E4B**.

All expensive or state-changing work runs through the tested command-line interface.
The [analysis notebook](notebooks/analyze_gemma4_sae.ipynb) is deliberately read-only:
it turns completed artifacts into publication figures without loading Gemma, starting
training, or uploading anything. The project separates data collection, SAE
optimization, evaluation, downstream intervention, feature mining, and release so every
artifact is auditable and SAE optimization is exactly resumable from checkpoints.

No pretrained SAE is bundled with this source repository. A defensible Gemma 4 SAE
requires a real activation collection and training run; the release command packages and
uploads that checkpoint only after the required evaluation artifacts exist.

## Why train a new SAE?

Gemma 4's small model is called `E4B`. It has a 2,560-dimensional residual stream, 42
text decoder layers, approximately 4.5B effective parameters, and approximately 8B total
parameters because of per-layer embeddings. Google's currently released
[Gemma Scope 2](https://ai.google.dev/gemma/docs/gemma_scope) SAEs target Gemma 3, not
Gemma 4. Reusing one of those dictionaries for Gemma 4 would be invalid because the
architecture, weights, and activation distribution differ.

The default trainer uses **BatchTopK**. It directly controls average L0 while permitting a
variable number of active features per token. The
[BatchTopK paper](https://arxiv.org/abs/2412.06410) reports a better
reconstruction–sparsity frontier than per-token TopK and performance comparable to
JumpReLU on its experiments. The repository also follows core practices from
[Scaling and evaluating sparse autoencoders](https://arxiv.org/abs/2406.04093):
an overcomplete dictionary, unit-norm decoder directions, explicit sparsity, dead-feature
tracking, and held-out reconstruction metrics.

## Pipeline

```text
FineWeb text or UltraChat messages
    │
    ▼
packed Gemma tokens ── frozen Gemma 4 E4B ── layer-20 residual activations
    │                                             │
    │ token contexts                              ▼
    └────────────────────────────── memory-mapped NumPy shards
                                                  │
                                                  ▼
                                   resumable BatchTopK training
                                                  │
                              ┌───────────────────┴──────────────────┐
                              ▼                                      ▼
                   held-out SAE metrics                    top activating contexts
                              │
                              ▼
                   live-model loss recovery
```

The model and SAE do not need to fit in memory together. Activation collection runs
Gemma; training then consumes cached shards without loading Gemma.

## What is implemented

- pinned Gemma 4 E4B base and instruction-tuned checkpoints
- CUDA, Apple MPS, and CPU runtime selection with backend-appropriate model dtypes
- a full-scale CUDA/BF16 configuration sized for NVIDIA DGX Spark
- pinned FineWeb text and UltraChat message corpora; message lists use Gemma's chat template
- robust discovery of the 42-layer text decoder stack
- a forward hook that stores only the chosen layer rather than all hidden states
- streaming text packing from Hugging Face Datasets
- fixed-size, memory-mapped activation shards
- token IDs and local token windows for later interpretation
- SHA-256 hashes for every activation, token, and context shard
- deterministic whole-shard validation splits with training-only normalization statistics
- software, runtime, Git revision, model revision, dataset revision, and config provenance
- BatchTopK training with:
  - an explicit target L0;
  - unit-norm decoder columns;
  - removal of decoder gradients parallel to feature directions;
  - an exponential-moving-average inference threshold;
  - the paper's top-k auxiliary residual loss for dead latents;
  - dead-feature tracking and optional residual-based resampling;
  - warmup plus cosine learning-rate decay;
  - gradient clipping, atomic checkpoints, and exact optimizer/RNG resume
- held-out MSE, fraction of variance explained, cosine similarity, L0, and live-feature
  metrics
- downstream language-model loss recovery from live SAE and mean-ablation interventions
- automatic or explicit feature selection and top-context mining
- prompt-level explanations with per-token SAE feature IDs, strengths, and known contexts
- inference-only `safetensors` release bundles, model cards, and SHA-256 checksums
- an explicit Hugging Face publisher configured for the `lamm-mit` organization
- checkpoint/config/activation compatibility checks before resume or evaluation

## Installation

Python 3.11+ is required. CUDA is fastest; MPS and CPU are supported. CPU collection of
the full E4B checkpoint requires substantial system memory and is slow.

```bash
git clone https://github.com/lamm-mit/gemma-sae.git
cd gemma-sae
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

For the optional 4-bit collection path:

```bash
python -m pip install -e ".[quantization]"
```

For Jupyter:

```bash
python -m pip install -e ".[notebook]"
GEMMA4_SAE_CONFIG=configs/e4b_layer20_batchtopk_dgx_spark.yaml \
  jupyter lab notebooks/analyze_gemma4_sae.ipynb
```

Use `configs/e4b_layer20_batchtopk.yaml` instead when analyzing the smaller pilot run.
Set `GEMMA4_SAE_EXPLANATION=runs/prompt-paris.json` as well to render a saved prompt's
token-by-feature heatmap.

For development:

```bash
python -m pip install -e ".[dev]"
pytest
ruff check .
```

### Native DGX Spark installation

DGX Spark uses an ARM64 Grace CPU and GB10 Blackwell GPU. The primary path is native:
reuse the CUDA-enabled PyTorch supplied by DGX OS instead of replacing it with a generic
wheel. First verify that the host Python sees CUDA:

```bash
python3 -c "import torch; print(torch.__version__, torch.version.cuda); \
print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

If that prints `True` and an NVIDIA GB10 device:

```bash
git clone https://github.com/lamm-mit/gemma-sae.git
cd gemma-sae
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --upgrade-strategy only-if-needed -e ".[dev,notebook]"
python -m pip check
pytest -q
ruff check .
```

The `--system-site-packages` flag is intentional: it keeps the NVIDIA CUDA-enabled
PyTorch build visible inside the virtual environment. Re-run the CUDA check after
installation.

Docker is optional. It is useful only when the host Python does not expose the NVIDIA
PyTorch build, or when an immutable NVIDIA-tested CUDA/PyTorch environment is required
for reproducibility. It is not required by this project. After exporting `HF_TOKEN` as
shown below, the fallback is:

```bash
docker run --rm -it \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  -e HF_TOKEN \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$PWD:/workspace/gemma-sae" \
  -w /workspace/gemma-sae \
  nvcr.io/nvidia/pytorch:25.12-py3 \
  bash
```

Inside that container, run `python -m pip install -e ".[dev,notebook]"`.

Accept any access terms shown on the
[`google/gemma-4-E4B`](https://huggingface.co/google/gemma-4-E4B) model page. Read the
token without placing it in shell history or source control:

```bash
read -rsp "Hugging Face token: " HF_TOKEN
echo
export HF_TOKEN
```

## Quick start

The base and IT configurations are **pilot runs**, not smoke tests:

```bash
CONFIG=configs/e4b_layer20_batchtopk.yaml  # or e4b_it_layer20_batchtopk.yaml
gemma4-sae doctor --config "$CONFIG"
gemma4-sae collect --config "$CONFIG"
gemma4-sae verify --config "$CONFIG"
gemma4-sae train --config "$CONFIG"
gemma4-sae evaluate --config "$CONFIG"
gemma4-sae fidelity --config "$CONFIG"
gemma4-sae explain --config "$CONFIG" --text "Paris is the capital of France."
gemma4-sae mine --config "$CONFIG"
gemma4-sae publish --config "$CONFIG" --checkpoint latest --dry-run
```

### Full DGX Spark run

The checked-in `e4b_layer20_batchtopk_dgx_spark.yaml` configuration is the primary
single-layer research run:

- CUDA and BF16 are mandatory, so it fails instead of silently falling back to CPU;
- 50 million FineWeb activation tokens;
- 262,144-row shards;
- a 40,960-feature dictionary (16× expansion);
- target L0 of 64;
- 4,096-vector training batches, matching the BatchTopK reference experiments;
- the paper's 512-latent auxiliary loss with coefficient 1/32;
- 25,000 optimization steps, 102.4 million optimizer examples, and 10 checkpoints;
- 256-sequence independent language-model fidelity evaluation.

Version 0.2 changed the training objective after a high dead-feature fraction exposed
that the original trainer lacked the BatchTopK auxiliary loss. Pre-0.2 checkpoints are
preserved negative-result artifacts and are intentionally incompatible with the corrected
primary run. Existing verified activation shards remain valid and should be reused.

It stores about 261 GB of activation, token, and context arrays. Ten FP32
SAE-plus-Adam checkpoints add an estimated 25 GB. Reserve at least 450 GB free; 600 GB is
more comfortable after the model cache, container or environment, logs, and release
artifacts.

The safest default is a writable directory in the current user's home:

```bash
export SAE_DATA="$HOME/gemma-sae-data"
mkdir -p "$SAE_DATA"/{activations,runs}
ln -s "$SAE_DATA/activations" activations
ln -s "$SAE_DATA/runs" runs
```

Create those links only in a fresh clone where `activations` and `runs` do not already
exist. Both names are ignored by Git. Use another volume only after confirming its real
mount point and write permission with `df -h <path>` and `test -w <path>`.

Use a persistent terminal because activation collection is not resumable:

```bash
cd "$HOME/src/gemma-sae"  # or wherever the repository was cloned
source .venv/bin/activate
tmux new -s gemma-sae
```

Inside `tmux`:

```bash
export CONFIG=configs/e4b_layer20_batchtopk_dgx_spark.yaml
mkdir -p logs
set -o pipefail

gemma4-sae doctor --config "$CONFIG" | tee logs/doctor.json
```

Before continuing, confirm `backend` is `cuda`, `model_dtype` is `bfloat16`, the
accelerator is GB10, `warnings` is empty, and the reported filesystem has sufficient
free space.

Run the pipeline:

```bash
gemma4-sae collect --config "$CONFIG" 2>&1 | tee logs/collect.log
gemma4-sae verify --config "$CONFIG" 2>&1 | tee logs/verify.log
gemma4-sae train --config "$CONFIG" 2>&1 | tee logs/train.log
gemma4-sae evaluate \
  --config "$CONFIG" \
  --checkpoint latest \
  --max-batches 1000000 \
  2>&1 | tee logs/evaluate.log
gemma4-sae fidelity \
  --config "$CONFIG" \
  --checkpoint latest \
  2>&1 | tee logs/fidelity.log
gemma4-sae mine \
  --config "$CONFIG" \
  --checkpoint latest \
  --n-features 64 \
  --top-contexts 40 \
  --max-batches 4096 \
  2>&1 | tee logs/mine.log
gemma4-sae publish \
  --config "$CONFIG" \
  --checkpoint latest \
  --dry-run
```

Detach from `tmux` with `Ctrl-B`, then `D`; reconnect with
`tmux attach -t gemma-sae`.

Resume an interrupted training run:

```bash
gemma4-sae train --config "$CONFIG" --resume 2>&1 | tee -a logs/train.log
```

Mine particular feature IDs:

```bash
gemma4-sae mine \
  --config "$CONFIG" \
  --features 41 928 12007 \
  --top-contexts 40
```

### Explain a new prompt

After training, apply the SAE to a new prompt by running Gemma once, capturing the same
layer-20 residual stream, and encoding each token with the trained dictionary:

```bash
gemma4-sae explain \
  --config "$CONFIG" \
  --checkpoint latest \
  --text "Paris is the capital of France." \
  --top-features 5 \
  --top-prompt-features 20 \
  --output runs/prompt-paris.json
```

The JSON report includes:

- every token and its position;
- the number of threshold-active SAE features for that token;
- the strongest feature IDs and activation strengths per token;
- the strongest features across the whole prompt and the token positions where they fire;
- examples already available in `feature_reports/features.json`;
- an exact suggested `gemma4-sae mine --features ...` command for gathering held-out
  contexts for the newly observed feature IDs.

For example, a token row has this shape:

```json
{
  "position": 1,
  "token_id": 6044,
  "token": "Paris",
  "text": "Paris",
  "special": false,
  "active_feature_count": 61,
  "top_features": [
    {"feature_id": 12007, "activation": 8.41, "known_contexts": []},
    {"feature_id": 928, "activation": 6.73, "known_contexts": []}
  ]
}
```

The numbers above illustrate the schema; real IDs and activations come from the trained
checkpoint. If `known_contexts` is empty, run the report's suggested mining command and
then rerun `explain`. The prompt is printed and, when `--output` is used, stored verbatim;
review sensitive inputs before saving or sharing reports.

After reviewing the generated release bundle, model card, metrics, data terms, and any
mined contexts, upload the inference-only release. The checked-in base and IT
configurations target separate repositories in the verified lowercase `lamm-mit`
namespace:

```bash
# HF_TOKEN must have write permission to the lamm-mit organization.
gemma4-sae publish \
  --config "$CONFIG" \
  --checkpoint latest \
  --public
```

Omitting `--public` respects the safer `publication.private: true` staging default.
Publishing refuses to proceed unless run metadata, validation metrics, held-out
evaluation, and live-model fidelity are present. Optimizer state and mined text contexts
are excluded; contexts can be included only by changing the explicit publication setting
after privacy and license review. The checked-in configurations also refuse publication
when fewer than 90% of dictionary features activate on the held-out evaluation scan;
change that explicit threshold only with a documented scientific rationale.

## Pilot experiment

| Setting | Value |
|---|---:|
| Model | `google/gemma-4-E4B` (base) |
| Hook point | decoder layer 20 output |
| Residual dimension | discovered from model config; expected 2,560 |
| Activation tokens | 5,000,000 |
| Activation storage | about 25.6 GB plus contexts and NumPy headers |
| Dictionary width | 20,480 (8× expansion) |
| Target mean L0 | 64 |
| Training batch | 512 activation vectors |
| Training steps | 50,000 |
| Tokens sampled by optimizer | 25.6 million, with repeated shuffled passes |
| Validation data | deterministic 2% whole-shard holdout |

Each BF16/FP16 activation vector requires `2 × 2,560 = 5,120` bytes. Plan disk
capacity before increasing `max_activation_tokens`.

## DGX Spark primary experiment

| Setting | Value |
|---|---:|
| Configuration | `configs/e4b_layer20_batchtopk_dgx_spark.yaml` |
| Backend / model dtype | CUDA / BF16 |
| Gemma collection batch | 4 sequences × 512 tokens |
| Activation tokens | 50,000,000 |
| Complete activation store estimate | 261.2 GB |
| Dictionary width | 40,960 (16× expansion) |
| Target mean L0 | 64 |
| Training batch | 4,096 activation vectors |
| Auxiliary dead-latent objective | top-k 512, coefficient 1/32 |
| Training steps | 25,000 |
| Optimizer activation examples | 102.4 million |
| Checkpoint cadence | 2,500 steps (10 estimated checkpoints) |
| Estimated checkpoint storage | 25.2 GB |
| Independent fidelity sequences | 256 |

### Suggested run sizes

| Run | Activation tokens | Expansion | Steps | Purpose |
|---|---:|---:|---:|---|
| Smoke | 100k | 2× | 1k | verify the system |
| Pilot | 5M | 8× | 50k | tune L0, width, and learning rate |
| DGX Spark primary | 50M | 16× | 25k at batch 4,096 | serious single-layer feature research |
| Large single layer | 100M | 16×+ | budget 100M+ optimizer examples | wider frontier and stability study |
| Scope-style suite | 100M+ per site | multiple widths | sweep | many layers and hook sites |

The final row is a substantial compute and storage project. "Proper" does not mean that
one hyperparameter setting is universally correct: recent SAE work shows that feature
recovery can be sensitive to L0, width, data, and random seed. Run multiple seeds and
report the full reconstruction–sparsity frontier.

Training is genuinely expensive. On a laptop, the CPU/MPS paths are valuable for smoke
tests, development, and analysis, but a serious 50–100M-token single-layer study normally
belongs on a CUDA machine with enough disk for activation shards. Caching decouples Gemma
forward passes from SAE optimization, so the costly model can be released from memory
before training and failed optimization runs can reuse the same immutable activations.

## Configuration

All experiment-defining values live in one YAML file. Unknown keys fail fast.

### Model

- `model_id`: base E4B is preferred for general representation research; E4B-IT is
  appropriate for assistant-specific behavior.
- `revision`: exact Hugging Face model commit. Publication configurations must pin it.
- `backend`: `auto` selects CUDA, then MPS, then CPU; an explicit backend fails rather
  than silently falling back.
- `layer_index`: zero-based text decoder layer.
- `dtype`: `auto` chooses CUDA BF16/FP16, MPS FP16, or CPU FP32.
- `load_in_4bit`: useful for access, but quantization changes activation geometry. Do not
  mix quantized and BF16 shards, and do not present a quantized SAE as a BF16 SAE.
- `sequence_length`: packed token-block length.
- `inference_batch_size`: Gemma forward-pass batch size.

### Data

The default source is the streaming
[`HuggingFaceFW/fineweb`](https://huggingface.co/datasets/HuggingFaceFW/fineweb)
`sample-10BT` configuration. FineWeb is released under ODC-By; preserve its attribution
when distributing derived datasets or context examples.

The instruction-tuned configuration uses the MIT-licensed
[`HuggingFaceH4/ultrachat_200k`](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k)
`messages` column. Conversation roles and contents are rendered with the pinned E4B-IT
tokenizer's chat template before packing.

- `max_activation_tokens`: number of non-special token activations to keep.
- `tokens_per_shard`: 65,536 rows produces roughly 336 MB activation files.
- `context_radius`: token IDs retained on each side of the activating token.
- `activation_dir`: immutable output location for one model/layer/data run.
- `hash_shards`: compute and record SHA-256 for every generated array.

Collection refuses to write over an existing manifest. This is deliberate: silently
appending a different model, layer, precision, or dataset would invalidate the SAE.

### SAE

- `expansion_factor`: dictionary width divided by residual dimension.
- `target_l0`: average selected latents per training token.
- `validation_fraction`: fraction of complete shards reserved before normalization or
  training, preventing adjacent-token leakage.
- `dead_after_steps`: inactivity window before a feature is considered dead.
- `auxiliary_top_k`: number of dead latents per sample used to reconstruct the main
  residual.
- `auxiliary_loss_coefficient`: weight on that auxiliary objective; the BatchTopK
  reference value is 1/32.
- `resample_dead_features`: optional experimental residual-based reinitialization. It is
  disabled in the primary paper-aligned configuration because the auxiliary objective is
  the primary anti-collapse mechanism.

Run an L0 sweep rather than assuming 64 is correct, for example 32, 64, 128, and 256.
Compare fraction of variance explained, downstream fidelity, feature coherence, and sparse
probing—not reconstruction alone.

### Evaluation

The `evaluation` section defines an independently pinned corpus for live-model fidelity.
The base configuration uses Wikitext-103 test text; the IT configuration uses UltraChat
`test_sft` conversations. `gemma4-sae fidelity` reports baseline cross-entropy, SAE
cross-entropy, mean-ablation cross-entropy, cross-entropy increase, and loss recovered.

### Publication

- `hf_repo_id`: destination model repository. The base default is
  `lamm-mit/gemma-4-e4b-layer20-batchtopk-sae`; IT uses a separate repository.
- `private`: creates a private staging repository unless `--public` is passed.
- `include_feature_reports`: defaults to false because activating text may carry privacy
  or licensing risk.

## Artifact layout

```text
activations/gemma-4-e4b/layer-20/
├── manifest.json
├── shard-000000.activations.npy
├── shard-000000.tokens.npy
├── shard-000000.contexts.npy
└── ...

runs/e4b-layer20-batchtopk/
├── resolved_config.json
├── train_metrics.jsonl
├── run_metadata.json
├── validation_metrics.json
├── evaluation.json
├── fidelity.json
├── checkpoints/
│   ├── latest.json
│   └── step-00050000.pt
├── release/
│   └── step-00050000/
│       ├── README.md
│       ├── sae_weights.safetensors
│       ├── sae_config.json
│       ├── activation_manifest.json
│       └── checksums.json
└── feature_reports/
    └── features.json
```

`manifest.json` stores the exact model, layer, precision, dataset, seed, dimensions,
token count, shard row counts, and per-file hashes. Checkpoints store the activation
manifest and project-configuration hashes and refuse to run against a mismatched artifact.
Release bundles omit the optimizer/RNG checkpoint and retain only inference weights,
normalization, aggregate evidence, provenance, and checksums.

## Reading the metrics

- **Normalized MSE** measures reconstruction error after subtracting the activation mean
  and dividing by a global RMS.
- **Fraction variance explained (FVE)** is `1 - SSE / activation_energy`.
- **Mean L0** is the number of active features per token under the learned inference
  threshold. It may differ slightly from training target L0.
- **Active feature fraction** is the fraction of dictionary features that fired in the
  held-out evaluation sample.

A good MSE does not prove interpretability. Inspect top contexts, negative examples,
cross-dataset behavior, seed stability, and causal effects.

## Publication standard

The code path is designed for reproducible research; a single pilot run is not a
publication result. Follow the complete
[publication protocol](docs/PUBLICATION_PROTOCOL.md) and complete the
[SAE release card](docs/SAE_RELEASE_CARD_TEMPLATE.md). At minimum, a paper or SAE release
still needs:

1. multiple seeds, widths, L0 values, layers, and data mixtures;
2. feature-density distributions and dead-feature curves over training;
3. automated and human feature-coherence evaluation;
4. sparse probing and feature-specific steering/ablation on held-out tasks;
5. cross-seed feature stability or matching analysis;
6. total compute, wall-clock, energy/hardware, and failed-run disclosure;
7. safety, privacy, and dataset-license review before releasing activating text contexts.

SAEs provide a useful decomposition, not a guaranteed catalogue of the model's uniquely
"true" concepts. Treat feature names as hypotheses supported by evidence.

## References

- [Source repository](https://github.com/lamm-mit/gemma-sae)
- [NVIDIA DGX Spark user guide](https://docs.nvidia.com/dgx/dgx-spark/)
- [NVIDIA container runtime for DGX Spark](https://docs.nvidia.com/dgx/dgx-spark/nvidia-container-runtime-for-docker.html)
- [Gemma 4 model card](https://ai.google.dev/gemma/docs/core/model_card_4)
- [Gemma 4 E4B weights](https://huggingface.co/google/gemma-4-E4B)
- [Gemma Scope](https://ai.google.dev/gemma/docs/gemma_scope)
- [Gemma Scope technical report](https://arxiv.org/abs/2408.05147)
- [Scaling and evaluating sparse autoencoders](https://arxiv.org/abs/2406.04093)
- [BatchTopK Sparse Autoencoders](https://arxiv.org/abs/2412.06410)
- [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features/)
