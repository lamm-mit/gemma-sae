# Gemma 4 Sparse Autoencoder Training

A reproducible, single-node pipeline for training sparse autoencoders on the residual
stream of **Gemma 4 E4B**.

This repository is the scalable counterpart to the course's
`LLM Interpretability with a Sparse Autoencoder - Gemma 4 E4B.ipynb` notebook. The
notebook teaches the idea on a few thousand activations; this project separates data
collection, SAE optimization, evaluation, and feature mining so each stage is resumable
and auditable.

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
FineWeb stream
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
```

The model and SAE do not need to fit in memory together. Activation collection runs
Gemma; training then consumes cached shards without loading Gemma.

## What is implemented

- Gemma 4 E4B base and instruction-tuned checkpoint validation
- robust discovery of the 42-layer text decoder stack
- a forward hook that stores only the chosen layer rather than all hidden states
- streaming text packing from Hugging Face Datasets
- fixed-size, memory-mapped activation shards
- token IDs and local token windows for later interpretation
- online activation mean and global RMS statistics
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
- automatic or explicit feature selection and top-context mining

## Installation

Python 3.11+ and a CUDA machine are strongly recommended.

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

The checked-in configuration is a **pilot run**, not a smoke test:

```bash
gemma4-sae-collect --config configs/e4b_layer20_batchtopk.yaml
gemma4-sae-train --config configs/e4b_layer20_batchtopk.yaml
gemma4-sae-evaluate --config configs/e4b_layer20_batchtopk.yaml
gemma4-sae-mine --config configs/e4b_layer20_batchtopk.yaml
```

Resume an interrupted training run:

```bash
gemma4-sae-train --config configs/e4b_layer20_batchtopk.yaml --resume
```

Mine particular feature IDs:

```bash
gemma4-sae-mine \
  --config configs/e4b_layer20_batchtopk.yaml \
  --features 41 928 12007 \
  --top-contexts 40
```

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
| Validation rows | 2% of each shard, excluded from training |

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

## Configuration

All experiment-defining values live in one YAML file. Unknown keys fail fast.

### Model

- `model_id`: base E4B is preferred for representation research.
- `layer_index`: zero-based text decoder layer.
- `dtype`: use `bfloat16` where supported.
- `load_in_4bit`: useful for access, but quantization changes activation geometry. Do not
  mix quantized and BF16 shards, and do not present a quantized SAE as a BF16 SAE.
- `sequence_length`: packed token-block length.
- `inference_batch_size`: Gemma forward-pass batch size.

### Data

The default source is the streaming
[`HuggingFaceFW/fineweb`](https://huggingface.co/datasets/HuggingFaceFW/fineweb)
`sample-10BT` configuration. FineWeb is released under ODC-By; preserve its attribution
when distributing derived datasets or context examples.

- `max_activation_tokens`: number of non-special token activations to keep.
- `tokens_per_shard`: 65,536 rows produces roughly 336 MB activation files.
- `context_radius`: token IDs retained on each side of the activating token.
- `activation_dir`: immutable output location for one model/layer/data run.

Collection refuses to write over an existing manifest. This is deliberate: silently
appending a different model, layer, precision, or dataset would invalidate the SAE.

### SAE

- `expansion_factor`: dictionary width divided by residual dimension.
- `target_l0`: average selected latents per training token.
- `validation_fraction`: tail rows reserved independently in every shard.
- `dead_after_steps`: inactivity window before a feature is considered dead.
- `resample_every_steps`: cadence for reinitializing dead directions from large residuals.

Run an L0 sweep rather than assuming 64 is correct, for example 32, 64, 128, and 256.
Compare fraction of variance explained, downstream fidelity, feature coherence, and sparse
probing—not reconstruction alone.

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
├── validation_metrics.json
├── evaluation.json
├── checkpoints/
│   ├── latest.json
│   └── step-00050000.pt
└── feature_reports/
    └── features.json
```

`manifest.json` stores the exact model, layer, precision, dataset, seed, dimensions,
normalization statistics, token count, and shard row counts.

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

## Research gaps before making strong claims

This repository provides a sound training base, but a publishable Gemma 4 SAE release
should additionally include:

1. multiple seeds, widths, L0 values, layers, and data mixtures;
2. downstream loss-recovered or KL metrics from replacing Gemma's live activation with
   the SAE reconstruction;
3. feature-density distributions and dead-feature curves over training;
4. automated and human feature-coherence evaluation;
5. sparse probing and steering/ablation tests on held-out datasets;
6. checkpoint and dataset hashes, hardware details, total FLOPs, and wall-clock cost;
7. safety and privacy review of any released activating text contexts.

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
