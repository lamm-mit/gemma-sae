# Publication Protocol

This protocol separates a reproducible software artifact from a defensible scientific
claim. Passing the repository test suite is necessary but not sufficient for publication.

## 1. Freeze the study before the main run

Record:

- target checkpoint (`E4B` base, `E4B-it`, or both) and exact revision;
- hook site and the reason for selecting it;
- corpus, split, message formatting, and exact dataset revision;
- primary metrics and exclusion criteria;
- width, target L0, learning-rate, and random-seed grid;
- criteria for feature coherence and causal validation;
- stopping and failed-run policies.

Do not select only favorable seeds or checkpoints after inspecting results.

For the preregistered DGX Spark primary run, start from
`configs/e4b_layer20_batchtopk_dgx_spark.yaml`, record any change as a new configuration,
and retain the output of `gemma4-sae doctor` with the experiment artifacts.

## 2. Validate the pipeline with a smoke run

Create a dedicated 100k-token smoke configuration. Training and other state-changing
stages run through the CLI; the notebook remains a read-only analysis client.

```bash
gemma4-sae collect --config <smoke.yaml>
gemma4-sae verify --config <smoke.yaml>
gemma4-sae train --config <smoke.yaml>
gemma4-sae evaluate --config <smoke.yaml>
gemma4-sae fidelity --config <smoke.yaml>
gemma4-sae mine --config <smoke.yaml>
gemma4-sae publish --config <smoke.yaml> --dry-run
pytest
ruff check .
```

The smoke run validates code and memory assumptions. It must not be used for feature
claims.

## 3. Main experimental grid

For a single-layer study, a reasonable minimum is:

- at least three random seeds;
- at least three target L0 values spanning the plausible region;
- at least two dictionary widths;
- one preregistered primary configuration plus the full sweep;
- an untrained or dense reconstruction baseline where relevant.

A multi-layer or base-versus-IT paper should run the same grid and evaluation datasets for
every comparison. Do not compare models using different token counts or corpus formats
without a controlled ablation.

## 4. Data hygiene

- Pin dataset revisions.
- Preserve licenses and attribution.
- Verify every activation store with full hashes.
- Use whole-shard validation, as implemented here.
- Compute normalization from training shards only.
- Keep the live-model fidelity corpus independent from SAE training activations.
- Report filtering, truncation, packing, special-token removal, and chat-template details.
- Review top activating contexts for personal or sensitive information before release.

## 5. Optimization reporting

For every run, retain:

- resolved YAML and its SHA-256;
- activation manifest and its SHA-256;
- repository commit;
- model and dataset revisions;
- software/runtime metadata;
- optimizer and scheduler states;
- training metrics JSONL;
- all planned seeds, including failed or diverged runs;
- token count, wall-clock, device type, peak memory, and checkpoint cadence.

Plot training/validation MSE, FVE, L0, inference threshold, dead-feature fraction,
auxiliary dead-latent loss, any resampling events, and learning rate.

The preregistered BatchTopK objective must include the auxiliary reconstruction term
described in the BatchTopK paper: each sample's top activations among currently dead
latents reconstruct the main SAE residual. Record its top-k and coefficient. Treat a
persistently high dead-feature fraction as a failed optimization run even when
reconstruction FVE is high; do not select or publish such a checkpoint as the primary
SAE. The checked-in release gate requires at least 90% of features to activate on the
held-out evaluation scan; any change to that threshold must be preregistered and justified.

## 6. Required SAE-only evaluation

Report on held-out shards:

- normalized MSE;
- fraction of variance explained;
- cosine similarity;
- mean and distribution of L0;
- activation-frequency distribution;
- live/dead feature fraction;
- results as a reconstruction–sparsity frontier, not one isolated point.

Include uncertainty across seeds.

## 7. Required downstream fidelity

Use `gemma4-sae fidelity` on the pinned independent evaluation corpus and report:

- baseline cross-entropy;
- cross-entropy with the SAE reconstruction inserted;
- cross-entropy under mean ablation;
- SAE cross-entropy increase;
- loss recovered.

Run enough sequences to report a confidence interval. The CLI stores per-sequence losses
and deterministic bootstrap confidence intervals in `fidelity.json`.

## 8. Feature-quality evidence

Feature names are hypotheses. For a sampled and preregistered feature set:

- show top activating contexts and random negative contexts;
- use held-out contexts not seen during SAE optimization;
- blind human raters to model/condition where possible;
- report inter-rater agreement and the labeling rubric;
- measure automated explanation performance only with a fixed evaluator and prompt;
- test feature specificity, not just sensitivity;
- inspect token-position, template, language, and formatting confounds.

## 9. Causal evidence

For claims that a feature contributes to behavior:

- intervene at the trained hook site;
- compare ablation and positive/negative steering;
- use dose-response curves;
- include random-direction and matched-norm controls;
- report changes in target and non-target behaviors;
- distinguish correlation, causal contribution, and sufficiency.

The repository's aggregate SAE reconstruction intervention validates dictionary fidelity.
Feature-specific behavioral interventions should be implemented for the study's
preregistered tasks.

## 10. Stability and multiplicity

SAEs trained with different seeds can learn different decompositions. Report:

- decoder-direction matching or another explicit feature alignment method;
- match-rate and similarity distributions;
- stability of named features;
- multiple-comparison correction for large feature searches;
- whether a feature was selected before or after examining the evaluation task.

## 11. Base versus instruction-tuned comparisons

Use the base model for broad pretraining representations and the IT model for
assistant-conditioned behaviors. A fair comparison requires:

- the same layer semantics and dimensionality;
- matched token budgets;
- both raw-text and chat-formatted corpus controls, if the research question permits;
- explicit treatment of chat control tokens;
- separate dictionaries—never reuse or merge activation shards across checkpoints.

## 12. Release checklist

Release:

- code commit and environment specification;
- exact configurations for every reported run;
- manifests and checksums;
- checkpoints or a documented reason they cannot be released;
- aggregate metrics and plotting code;
- feature reports after privacy review;
- model/data licenses and attribution;
- limitations, negative results, and compute statement.

Build the inference-only release locally before any network write:

```bash
gemma4-sae publish --config <primary.yaml> --checkpoint latest --dry-run
```

The dry run reports missing required evidence. After review, publish privately for
internal validation or pass `--public` for the final `lamm-mit` upload. The publisher
refuses a real upload without run metadata, validation metrics, held-out SAE evaluation,
and live-model fidelity. Mined contexts remain excluded unless the configuration
explicitly opts in after privacy and license review.

Use precise language: an SAE is a useful learned decomposition of an activation space,
not proof that it recovered a unique set of concepts intrinsically used by the model.
