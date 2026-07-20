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
- corpus-driven feature selection with leakage-resistant development/validation splits
- resumable, checkpoint-bound feature labeling with held-out automatic scoring
- OpenAI, Anthropic, OpenAI-compatible, and local Transformers labeling backends
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
jupyter lab notebooks/analyze_gemma4_sae.ipynb
```

The notebook's first code cell defaults to
`HF_REPO_ID = "lamm-mit/gemma-4-e4b-layer20-batchtopk-sae"` and therefore loads the
verified inference release directly from Hugging Face. Set `HF_REPO_ID = ""` to use
`LOCAL_CONFIG_PATH` for a local training run. Set `EXPLANATION_JSON` there as well to
render a saved prompt's token-by-feature heatmap. `DEVICE = "auto"` selects CUDA, then
Apple MPS, then CPU; set it explicitly to `cuda`, `mps`, or `cpu` to require that
backend. `ANALYSIS_BATCH_SIZE` controls local activation-analysis memory. Analyses
requiring unpublished activation shards remain clearly unavailable in Hub mode.

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
gemma4-sae label --config "$CONFIG" --provider transformers --model "ORG/INSTRUCT-MODEL"
gemma4-sae publish --config "$CONFIG" --checkpoint latest --dry-run
```

### Full DGX Spark 16× baseline run

The checked-in `e4b_layer20_batchtopk_dgx_spark.yaml` configuration is the original
16× single-layer baseline. It produced the shared 50-million-token activation cache used
by the subsequent width sweep. The selected 12× release candidate and its complete
finalization workflow are documented below.

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
  --random-contexts 40 \
  --max-batches 4096 \
  2>&1 | tee logs/mine.log
gemma4-sae publish \
  --config "$CONFIG" \
  --checkpoint latest \
  --dry-run
```

`fidelity` runs three live-model passes and displays separate `tqdm` progress bars for
the baseline, SAE reconstruction, and mean-ablation phases.

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
  --top-contexts 40 \
  --random-contexts 40
```

### Develop labels from a prompt corpus

Use `develop-labels` when the goal is to label features that matter for a particular
scientific domain or prompt distribution. The command:

1. reads a local `.jsonl`, `.json`, or `.txt` corpus;
2. runs the development records through Gemma and the trained SAE;
3. ranks all SAE features by corpus coverage, activation mass, or frequency;
4. runs a second pass to collect strong development contexts and held-out validation
   contexts for the selected features;
5. generates and scores labels through the same checkpoint-bound registry used by
   `label`.

The repository includes a small, synthetic workflow example at
[`examples/science_label_corpus.jsonl`](examples/science_label_corpus.jsonl). It contains
explicit `development` and `validation` records across several scientific fields. It is a
software demonstration with a small set of general-prose controls, not a benchmark or
adequate evidence for a publication claim.

After training has finished:

```bash
cd ~/gemma-sae
git pull --ff-only origin main
source .venv/bin/activate
python -m pip install -e ".[notebook]"

export CONFIG=configs/e4b_layer20_batchtopk_dgx_spark_12x_l064.yaml
export RUN_DIR=runs/e4b-layer20-batchtopk-dgx-50m-12x-l064-auxk512-seed17
read -rsp "OpenAI API key: " OPENAI_API_KEY
echo
export OPENAI_API_KEY

gemma4-sae develop-labels \
  --config "$CONFIG" \
  --checkpoint latest \
  --corpus examples/science_label_corpus.jsonl \
  --text-column text \
  --n-features 64 \
  --ranking coverage \
  --train-contexts 8 \
  --heldout-contexts 4 \
  --provider openai \
  --model gpt-5.6 \
  --max-output-tokens 25000 \
  --acknowledge-external-data \
  2>&1 | tee logs/develop-science-labels.log
```

This performs **two Gemma corpus passes**, followed by approximately two labeling-model
calls per previously unlabeled feature when automatic scoring is enabled. Run it in
`tmux` on the DGX. Existing registry entries are skipped, so the workflow can safely be
rerun or expanded with a larger corpus. `--dry-run` still performs corpus selection and
writes the evidence report, but does not call a labeling provider. The four-per-class
held-out setting above is for the 60-record demonstration; use 8–12 or more per class
with a substantially larger validation corpus for a scientific study.

OpenAI reasoning tokens and visible structured output share `--max-output-tokens`.
The OpenAI provider therefore defaults to 25,000; other providers default to 1,024.
If the Responses API returns `status=incomplete`, the CLI reports
`incomplete_details.reason`, retries according to `--retries`, and keeps every feature
already written to the registry. To resume without repeating the two Gemma corpus passes,
run `gemma4-sae label --report <the-written-corpus-report>/features.json` with the same
labeling and validation arguments. Existing feature IDs are skipped automatically.

JSONL records use this shape:

```json
{"id":"paper-0001","split":"development","text":"First scientific passage..."}
{"id":"paper-0002","split":"validation","text":"Independent held-out passage..."}
```

If no `split` field is present, the command creates a deterministic document-level split
using `--validation-fraction` and the SAE seed. An explicit, frozen split is preferable
for a paper. Feature ranking uses only development records; validation records are used
only as held-out positive and negative contexts. Label evidence identifies the exact
target token inside each context window, and corpus validation keeps at most one example
per document in each class; negative controls come from documents in which the feature
never fired. A feature cannot receive `auto_validated` status unless the
requested number of held-out positive and negative examples are both available. The
default `coverage` score is:

```text
mean_active_activation × sqrt(token_frequency × document_frequency)
```

Use `--ranking activation-mass` to prioritize total activation contribution or
`--ranking frequency` to prioritize how often features fire. Selection scores, corpus
SHA-256, split policy, checkpoint identity, context partitions, and runtime metadata are
stored under:

```text
runs/.../corpus_reports/<corpus-name>-<corpus-sha>-<analysis-sha>/features.json
```

Validated labels are merged into `runs/.../feature_labels/labels.json`; subsequent
`gemma4-sae explain` calls load them automatically. The label describes a feature of the
SAE checkpoint, but its validation evidence is specific to the corpus domain. Do not
generalize a science-corpus validation score to unrelated prose without a separate test.

### Label mined features once

`mine` collects strong activations, randomly sampled active examples, and
zero-activation controls. `label` turns that evidence into reusable natural-language
interpretations and, by default, tests each interpretation on blinded held-out positive
and negative contexts. The registry is written atomically after every feature:

```text
runs/.../feature_labels/labels.json
```

The registry is bound to the exact SAE checkpoint SHA-256, model revision, layer,
training configuration, and activation manifest. An interrupted invocation is safe to
rerun: existing labels are skipped unless `--overwrite` is passed.

With OpenAI's Responses API, choose any structured-output-capable model available to the
API project. The request uses the documented
[JSON-schema Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs)
contract:

```bash
read -rsp "OpenAI API key: " OPENAI_API_KEY
echo
export OPENAI_API_KEY

gemma4-sae label \
  --config "$CONFIG" \
  --checkpoint latest \
  --provider openai \
  --model gpt-5.6 \
  --max-output-tokens 25000 \
  --acknowledge-external-data
```

With Anthropic, the request uses the documented
[`output_config.format`](https://platform.claude.com/docs/en/build-with-claude/structured-outputs)
contract:

```bash
read -rsp "Anthropic API key: " ANTHROPIC_API_KEY
echo
export ANTHROPIC_API_KEY

gemma4-sae label \
  --config "$CONFIG" \
  --checkpoint latest \
  --provider anthropic \
  --model "YOUR-CLAUDE-MODEL-ID" \
  --acknowledge-external-data
```

For a Hugging Face causal instruction model running locally on CUDA, MPS, or CPU:

```bash
gemma4-sae label \
  --config "$CONFIG" \
  --checkpoint latest \
  --provider transformers \
  --model "ORG/INSTRUCT-MODEL" \
  --device cuda \
  --dtype bfloat16
```

For vLLM, LM Studio, or another server implementing OpenAI-compatible chat completions
and JSON-schema response formats:

```bash
gemma4-sae label \
  --config "$CONFIG" \
  --checkpoint latest \
  --provider openai-compatible \
  --model "local-model-name" \
  --base-url http://127.0.0.1:8000/v1
```

Use a different scorer model to reduce self-evaluation bias:

```bash
gemma4-sae label \
  --config "$CONFIG" \
  --provider openai \
  --model gpt-5.6 \
  --scorer-provider anthropic \
  --scorer-model "YOUR-CLAUDE-MODEL-ID" \
  --acknowledge-external-data
```

External backends require `--acknowledge-external-data` because mined dataset text leaves
the machine. API keys are read only from environment variables and are never written to
artifacts. Use `--api-key-env` or `--scorer-api-key-env` for non-default variable names.
Localhost OpenAI-compatible servers and local Transformers do not require the
acknowledgement.

Each feature receives a status of `candidate`, `auto_validated`, or `uninterpretable`.
Automatic validation measures held-out balanced accuracy and activation-rank correlation;
it is not a substitute for human review or causal intervention. `--no-score` explicitly
creates candidate-only labels. The registry records provider/model identifiers, response
IDs, token usage,
prompt/schema hashes, evidence hashes, validation metrics, and thresholds, but does not
store held-out scorer text. Exact train/held-out splits are preserved separately under
`feature_labels/evidence/` so the scoring run remains locally reproducible. Those raw-text
snapshots are intentionally excluded from release bundles.

This is done once **per feature per SAE checkpoint**, not once per prompt. Labeling all
40,960 features would require substantial mining and roughly two model calls per feature.
A practical workflow labels 64–500 scientifically interesting features first, then grows
the same registry incrementally when new prompts reveal useful unlabeled features.

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

After publication, the same operation is portable and does not require the training
checkpoint, activation cache, or original YAML:

```bash
python -m pip install "gemma4-sae @ git+https://github.com/lamm-mit/gemma-sae.git"

gemma4-sae explain \
  --sae-repo lamm-mit/gemma-4-e4b-layer20-batchtopk-sae \
  --device auto \
  --text "Paris is the capital of France." \
  --output prompt-paris.json
```

`--sae-repo` accepts either a Hugging Face model repository or a local downloaded release
directory. The loader verifies `checksums.json`, the resolved configuration, activation
manifest, source-checkpoint identity, and any included feature-label registry before
inference. Use `--sae-revision <commit>` to pin the Hub artifact, `--local-files-only`
for an already-cached snapshot, and `--device cuda|mps|cpu|auto` to choose the execution
backend. The exact released Gemma revision is always retained.

The JSON report includes:

- every token and its position;
- the number of threshold-active SAE features for that token;
- the strongest feature IDs and activation strengths per token;
- the strongest features across the whole prompt and the token positions where they fire;
- examples already available in `feature_reports/features.json`;
- reusable interpretations and validation metrics from `feature_labels/labels.json`;
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
    {
      "feature_id": 12007,
      "activation": 8.41,
      "known_contexts": [],
      "interpretation": {
        "label": "French places and civic geography",
        "confidence": "medium",
        "status": "auto_validated"
      }
    },
    {"feature_id": 928, "activation": 6.73, "known_contexts": []}
  ]
}
```

The numbers above illustrate the schema; real IDs and activations come from the trained
checkpoint. If `known_contexts` is empty, run the report's suggested mining command and
then run `label`. All later `explain` calls automatically load the compatible registry;
use `--no-labels` only when raw feature IDs are desired. The prompt is printed and, when
`--output` is used, stored verbatim; review sensitive inputs before saving or sharing
reports.

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
after privacy and license review. A compatible feature-label registry is included
automatically without its held-out scorer text. The checked-in configurations also refuse
publication
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

## DGX Spark 16× baseline experiment

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

### Overnight utilization sweep

The checked-in overnight sweep reuses the immutable 50M-token activation cache and trains
controlled 8× and 12× width variants at the same target L0, optimizer-example budget,
auxiliary objective, and seed as the completed 16× baseline:

```bash
bash scripts/dgx_overnight_width_sweep.sh \
  2>&1 | tee logs/overnight-width-sweep.log
```

The runs are sequential so they do not compete for DGX Spark memory. Each is resumable
from its own checkpoint and receives a full held-out cached-activation evaluation. The
script verifies activation headers once, never recollects activations, and prints an
8×/12×/16× comparison of FVE, cosine similarity, mean L0, active-feature fraction, and
the configured 90% utilization gate. Checkpoints are saved every 5,000 steps to limit
the two new runs to roughly 16 GB of checkpoint storage.

Each run displays separate tqdm bars for the normalization scan, 25,000 optimization
steps, and held-out evaluation batches. The training bar shows live MSE, FVE, L0, dead
fraction, throughput, and ETA; tqdm-safe checkpoint and metric lines remain usable in
the corresponding `tee` logs.

Activation readers advise Linux to evict clean pages after each completed memory-mapped
shard. This matters on DGX Spark because the GB10 GPU and CPU share physical memory:
a normalization pass over hundreds of gigabytes can otherwise leave almost all RAM in
file cache and make a subsequent CUDA allocation fail even when `nvidia-smi` shows no
compute process.

Do not run live-model fidelity or develop labels for every sweep member. In the morning,
select the narrowest model that clears the utilization gate without an unacceptable
reconstruction drop, then run fidelity and checkpoint-bound labeling for that winner.
The earlier 16× label registry cannot be transferred to a new checkpoint.

### Selected 12× release candidate

The completed controlled sweep selected the 12× checkpoint. All three runs used the same
Gemma revision, layer, 50-million-token FineWeb activation cache, target L0, optimizer
example budget, auxiliary objective, and random seed. Fidelity used the same pinned
Wikitext-103 test split and 256 sequences.

| Width | Features | Held-out FVE | Cosine | Mean L0 | Active features | Loss recovered | Decision |
|---:|---:|---:|---:|---:|---:|---:|---|
| 8× | 20,480 | 0.8391 | 0.9147 | 64.07 | 100.00% | 0.7907 | passes utilization gate |
| **12×** | **30,720** | **0.8415** | **0.9160** | **64.15** | **100.00%** | **0.8030** | **selected release candidate** |
| 16× | 40,960 | 0.8403 | 0.9151 | 64.09 | 81.09% | 0.8099 | fails 90% utilization gate |

The 12× run has the strongest FVE and cosine similarity of the three, preserves more
language-model loss than 8×, and activates every feature in the one-million-example
held-out scan. The 16× run has slightly higher loss recovery but fails the configured
feature-utilization gate. This is a release-candidate decision for this controlled sweep,
not a publication-level claim: a paper still requires multiple seeds and the analyses in
the publication protocol.

### Finalize, publish, and test the selected 12× release

The following is the complete post-training workflow. Run it from the repository on the
DGX Spark. It deliberately targets the canonical public repository
`lamm-mit/gemma-4-e4b-layer20-batchtopk-sae`, overriding the experiment-specific staging
repository in the 12× YAML.

First update the checkout, install the current CLI, and select the exact run:

```bash
cd ~/gemma-sae
git pull --ff-only origin main
source .venv/bin/activate
python -m pip install --upgrade-strategy only-if-needed -e ".[notebook]"

mkdir -p logs
set -o pipefail

export CONFIG="configs/e4b_layer20_batchtopk_dgx_spark_12x_l064.yaml"
export RUN_DIR="runs/e4b-layer20-batchtopk-dgx-50m-12x-l064-auxk512-seed17"
export HF_REPO_ID="lamm-mit/gemma-4-e4b-layer20-batchtopk-sae"
export RELEASE_DIR="$RUN_DIR/release/step-00025000"
export PYTHONUNBUFFERED=1
```

If `evaluation.json` or `fidelity.json` is not already present, create it before
labeling. These commands are safe to omit when the files from the completed runs shown
above already exist:

```bash
test -f "$RUN_DIR/evaluation.json" || gemma4-sae evaluate \
  --config "$CONFIG" \
  --checkpoint latest \
  --max-batches 1000000 \
  2>&1 | tee logs/evaluate-12x-final.log

test -f "$RUN_DIR/fidelity.json" || gemma4-sae fidelity \
  --config "$CONFIG" \
  --checkpoint latest \
  2>&1 | tee logs/fidelity-12x-final.log
```

Develop the initial 64 labels for the selected checkpoint. This small included corpus
demonstrates the complete machinery but does not by itself establish publication-quality
scientific interpretations. The command is resumable; existing compatible feature IDs
are skipped:

```bash
read -rsp "OpenAI API key: " OPENAI_API_KEY
echo
export OPENAI_API_KEY

gemma4-sae develop-labels \
  --config "$CONFIG" \
  --checkpoint latest \
  --corpus examples/science_label_corpus.jsonl \
  --text-column text \
  --n-features 64 \
  --ranking coverage \
  --train-contexts 8 \
  --heldout-contexts 4 \
  --provider openai \
  --model gpt-5.6 \
  --max-output-tokens 25000 \
  --retries 2 \
  --acknowledge-external-data \
  2>&1 | tee logs/develop-science-labels-12x.log
```

Confirm that the checkpoint, aggregate evaluations, and checkpoint-bound label registry
exist, then print the release gates and label statuses:

```bash
test -f "$RUN_DIR/checkpoints/latest.json"
test -f "$RUN_DIR/evaluation.json"
test -f "$RUN_DIR/fidelity.json"
test -f "$RUN_DIR/feature_labels/labels.json"

python - <<'PY'
import json
import os
from collections import Counter
from pathlib import Path

run = Path(os.environ["RUN_DIR"])
evaluation = json.loads((run / "evaluation.json").read_text())["metrics"]
fidelity = json.loads((run / "fidelity.json").read_text())["metrics"]
labels = json.loads(
    (run / "feature_labels" / "labels.json").read_text()
)["labels"]

print("FVE:", evaluation["fraction_variance_explained"])
print("cosine:", evaluation["mean_cosine_similarity"])
print("mean L0:", evaluation["mean_l0"])
print("active feature fraction:", evaluation["active_feature_fraction"])
print("loss recovered:", fidelity["loss_recovered"])
print("labels:", len(labels))
print("label statuses:", dict(Counter(item["status"] for item in labels)))

assert evaluation["active_feature_fraction"] >= 0.90
assert len(labels) >= 64
print("Configured release gates passed.")
PY
```

Build the inference-only release without making a network change. The dry run writes the
local bundle, computes checksums, and reports missing evidence or failed quality gates:

```bash
gemma4-sae publish \
  --config "$CONFIG" \
  --checkpoint latest \
  --repo-id "$HF_REPO_ID" \
  --public \
  --dry-run \
  2>&1 | tee logs/publish-dry-run-12x.json

python - <<'PY'
import json
from pathlib import Path

report = json.loads(Path("logs/publish-dry-run-12x.json").read_text())
assert report["missing_required_evidence"] == []
assert report["quality_failures"] == []
print("Dry-run publication checks passed:", report["release_dir"])
PY
```

Verify every local release checksum and the expected 12× metadata before uploading:

```bash
python - <<'PY'
import json
import os
from pathlib import Path
from gemma4_sae.release import verify_release_bundle

release = Path(os.environ["RELEASE_DIR"])
checksums = verify_release_bundle(release)
sae = json.loads((release / "sae_config.json").read_text())
metadata = json.loads((release / "release_metadata.json").read_text())
labels = json.loads((release / "feature_labels.json").read_text())["labels"]

assert sae["n_features"] == 30_720
assert sae["checkpoint_step"] == 25_000
assert metadata["hf_repo_id"] == os.environ["HF_REPO_ID"]
assert metadata["contains_feature_labels"] is True
assert len(labels) >= 64

print("Verified files:", len(checksums))
print("Features:", sae["n_features"])
print("Checkpoint step:", sae["checkpoint_step"])
print("Included labels:", len(labels))
PY
```

Authenticate with a Hugging Face token that can write to the `lamm-mit` organization,
confirm the account, and perform the public upload:

```bash
read -rsp "Hugging Face write token: " HF_TOKEN
echo
export HF_TOKEN

python - <<'PY'
import os
from huggingface_hub import HfApi

identity = HfApi(token=os.environ["HF_TOKEN"]).whoami()
print("Authenticated Hugging Face account:", identity["name"])
PY

gemma4-sae publish \
  --config "$CONFIG" \
  --checkpoint latest \
  --repo-id "$HF_REPO_ID" \
  --public \
  2>&1 | tee logs/publish-12x.json
```

Resolve and save the immutable Hub commit SHA rather than relying on a moving `main`
revision:

```bash
python - <<'PY' | tee logs/hf-12x-revision.txt
import os
from huggingface_hub import HfApi

info = HfApi(token=os.environ["HF_TOKEN"]).model_info(
    os.environ["HF_REPO_ID"]
)
print(info.sha)
PY

export HF_REVISION="$(tail -n 1 logs/hf-12x-revision.txt)"
test -n "$HF_REVISION"
echo "Published revision: $HF_REVISION"
```

Download that exact Hub revision through the normal release loader. This independently
checks every downloaded file against `checksums.json` and confirms that the portable
bundle contains the expected SAE and labels:

```bash
python - <<'PY'
import json
import os
from gemma4_sae.release import resolve_release_bundle, verify_release_bundle

release = resolve_release_bundle(
    os.environ["HF_REPO_ID"],
    revision=os.environ["HF_REVISION"],
)
checksums = verify_release_bundle(release)
sae = json.loads((release / "sae_config.json").read_text())
labels = json.loads((release / "feature_labels.json").read_text())["labels"]

assert sae["n_features"] == 30_720
assert sae["checkpoint_step"] == 25_000
assert len(labels) >= 64

print("Verified Hub snapshot:", release)
print("Verified files:", len(checksums))
print("Included labels:", len(labels))
PY
```

Finally, run a real prompt through Gemma and the SAE loaded from the immutable Hub
release, then inspect the strongest prompt-level features and any reusable labels:

```bash
gemma4-sae explain \
  --sae-repo "$HF_REPO_ID" \
  --sae-revision "$HF_REVISION" \
  --device cuda \
  --text "A catalyst lowers the activation energy without changing the reaction equilibrium." \
  --top-features 5 \
  --top-prompt-features 20 \
  --output runs/hub-smoke-test-12x.json

python - <<'PY'
import json
from pathlib import Path

report = json.loads(Path("runs/hub-smoke-test-12x.json").read_text())
print("Model:", report["model_id"])
print("Layer:", report["layer_index"])
print("Mean prompt L0:", report["mean_inference_l0"])
print("Labeled prompt-feature fraction:", report["labeled_prompt_feature_fraction"])

for feature in report["prompt_features"][:10]:
    interpretation = feature.get("interpretation") or {}
    print(
        feature["feature_id"],
        f'{feature["max_activation"]:.3f}',
        interpretation.get("label", "unlabeled"),
    )
PY
```

On any CUDA, Apple Silicon, or CPU analysis machine, install the package and open the
notebook:

```bash
git clone https://github.com/lamm-mit/gemma-sae.git
cd gemma-sae
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[notebook]"
jupyter lab notebooks/analyze_gemma4_sae.ipynb
```

The notebook already sets
`HF_REPO_ID = "lamm-mit/gemma-4-e4b-layer20-batchtopk-sae"` inside its first code cell.
For a frozen paper analysis, paste the saved commit SHA into `HF_REPO_REVISION` in that
same cell. Leave `DEVICE = "auto"` to select CUDA, then MPS, then CPU. The Hub bundle is
sufficient for prompt explanations, released metrics, label analysis, and publication
figures; analyses over the original 50-million-token activation cache still require
those local unpublished shards.

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

For corpus-driven feature selection or a future domain-specific SAE, useful Hugging Face
sources include:

| Dataset | Best use | Important caveat |
|---|---|---|
| [`allenai/peS2o`](https://huggingface.co/datasets/allenai/peS2o) | broad scientific-paper language across STEM fields | large; dataset is ODC-By, and source-document terms still require review |
| [`HuggingFaceFW/fineweb-edu`](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | educational explanations and general-domain controls | web-derived content; retain attribution and review source rights |
| [`HuggingFaceTB/finemath`](https://huggingface.co/datasets/HuggingFaceTB/finemath) | mathematical exposition and notation | web-derived and specialized; pair with general text |
| [`wikimedia/wikipedia`](https://huggingface.co/datasets/wikimedia/wikipedia) | broad reference prose and matched non-paper controls | GFDL/CC BY-SA attribution and share-alike requirements apply |
| [`allenai/sciq`](https://huggingface.co/datasets/allenai/sciq) | science questions, answers, and supporting passages | CC BY-NC 3.0; non-commercial restriction |
| [`allenai/dolma`](https://huggingface.co/datasets/allenai/dolma) | a documented general/science mixture, including peS2o | multi-source corpus; review each original source's terms |

Pin the exact dataset revision, sample by document rather than isolated sentence, remove
duplicates before splitting, retain stable IDs, and keep validation documents disjoint
from feature selection. For a science-focused comparison, start with a separately trained
70–80% general / 20–30% scientific mixture rather than continuing the existing FineWeb
checkpoint in place. Preserve the current FineWeb SAE as the baseline.

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
├── feature_labels/
│   ├── labels.json
│   └── evidence/
│       └── feature-00000041.json
├── checkpoints/
│   ├── latest.json
│   └── step-00050000.pt
├── release/
│   └── step-00050000/
│       ├── README.md
│       ├── sae_weights.safetensors
│       ├── sae_config.json
│       ├── activation_manifest.json
│       ├── feature_labels.json
│       └── checksums.json
└── feature_reports/
    └── features.json
```

`manifest.json` stores the exact model, layer, precision, dataset, seed, dimensions,
token count, shard row counts, and per-file hashes. Checkpoints store the activation
manifest and project-configuration hashes and refuse to run against a mismatched artifact.
Release bundles omit the optimizer/RNG checkpoint and retain only inference weights,
normalization, aggregate evidence, provenance, reusable feature labels, and checksums.
Raw mined contexts remain excluded by default; the label registry contains descriptions,
provenance, and aggregate validation metrics rather than held-out scorer text. Local
`feature_labels/evidence/` snapshots are not copied into the release.

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
