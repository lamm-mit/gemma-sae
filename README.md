# Gemma 4 Sparse Autoencoder Training

A reproducible, single-node pipeline for training sparse autoencoders on the residual
stream of **Gemma 4 E4B**.

All expensive or state-changing work runs through the tested command-line interface.
The [analysis notebook](notebooks/analyze_gemma4_sae.ipynb) is deliberately read-only:
it turns completed artifacts into publication figures without loading Gemma, starting
training, or uploading anything. The project separates data collection, SAE
optimization, evaluation, downstream intervention, feature mining, and release so every
expensive stage is resumable and auditable.

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
  - dead-feature tracking and residual-based resampling;
  - warmup plus cosine learning-rate decay;
  - gradient clipping, atomic checkpoints, and exact optimizer/RNG resume
- held-out MSE, fraction of variance explained, cosine similarity, L0, and live-feature
  metrics
- downstream language-model loss recovery from live SAE and mean-ablation interventions
- automatic or explicit feature selection and top-context mining
- inference-only `safetensors` release bundles, model cards, and SHA-256 checksums
- an explicit Hugging Face publisher configured for the `lamm-mit` organization
- checkpoint/config/activation compatibility checks before resume or evaluation

## Installation

Python 3.11+ is required. CUDA is fastest; MPS and CPU are supported. CPU collection of
the full E4B checkpoint requires substantial system memory and is slow.

```bash
git clone <this-repository>
cd gemma-4-sae-training
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

For the optional 4-bit collection path:

```bash
pip install -e ".[quantization]"
```

For Jupyter:

```bash
pip install -e ".[notebook]"
jupyter lab notebooks/analyze_gemma4_sae.ipynb
```

For development:

```bash
pip install -e ".[dev]"
pytest
ruff check .
```

Accept any access terms shown on the
[`google/gemma-4-E4B`](https://huggingface.co/google/gemma-4-E4B) model page, then
authenticate without putting a token in source control:

```bash
export HF_TOKEN=hf_...
```

## Quick start

The checked-in configurations are **pilot runs**, not smoke tests:

```bash
CONFIG=configs/e4b_layer20_batchtopk.yaml  # or e4b_it_layer20_batchtopk.yaml
gemma4-sae doctor --config "$CONFIG"
gemma4-sae collect --config "$CONFIG"
gemma4-sae verify --config "$CONFIG"
gemma4-sae train --config "$CONFIG"
gemma4-sae evaluate --config "$CONFIG"
gemma4-sae fidelity --config "$CONFIG"
gemma4-sae mine --config "$CONFIG"
gemma4-sae publish --config "$CONFIG" --checkpoint latest --dry-run
```

Resume an interrupted training run:

```bash
gemma4-sae train --config configs/e4b_layer20_batchtopk.yaml --resume
```

Mine particular feature IDs:

```bash
gemma4-sae mine \
  --config configs/e4b_layer20_batchtopk.yaml \
  --features 41 928 12007 \
  --top-contexts 40
```

After reviewing the generated release bundle, model card, metrics, data terms, and any
mined contexts, upload the inference-only release. The checked-in base and IT
configurations target separate repositories in the verified lowercase `lamm-mit`
namespace:

```bash
export HF_TOKEN=hf_...  # must have write permission to lamm-mit
gemma4-sae publish \
  --config configs/e4b_layer20_batchtopk.yaml \
  --checkpoint latest \
  --public
```

Omitting `--public` respects the safer `publication.private: true` staging default.
Publishing refuses to proceed unless run metadata, validation metrics, held-out
evaluation, and live-model fidelity are present. Optimizer state and mined text contexts
are excluded; contexts can be included only by changing the explicit publication setting
after privacy and license review.

## Default experiment

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

### Suggested run sizes

| Run | Activation tokens | Expansion | Steps | Purpose |
|---|---:|---:|---:|---|
| Smoke | 100k | 2× | 1k | verify the system |
| Pilot (default) | 5M | 8× | 50k | tune L0, width, and learning rate |
| Serious single layer | 50–100M | 16× | 200k+ | feature research |
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
- `resample_every_steps`: cadence for reinitializing dead directions from large residuals.

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

- [Gemma 4 model card](https://ai.google.dev/gemma/docs/core/model_card_4)
- [Gemma 4 E4B weights](https://huggingface.co/google/gemma-4-E4B)
- [Gemma Scope](https://ai.google.dev/gemma/docs/gemma_scope)
- [Gemma Scope technical report](https://arxiv.org/abs/2408.05147)
- [Scaling and evaluating sparse autoencoders](https://arxiv.org/abs/2406.04093)
- [BatchTopK Sparse Autoencoders](https://arxiv.org/abs/2412.06410)
- [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features/)
