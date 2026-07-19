# SAE Release Card

Complete every field before releasing weights or making feature claims.

## Identification

- Release name:
- Repository commit:
- Checkpoint SHA-256:
- License:
- Contact:
- Release date:
- Hugging Face repository:
- Public or private:

## Source model

- Model ID and revision:
- Base or instruction tuned:
- Model license:
- Hook site:
- Residual dimension:
- Model precision used for activation collection:
- Quantization, if any:

## Activation corpus

- Dataset ID/config/revision:
- Dataset license:
- Split:
- Text or message/chat format:
- Chat-template revision:
- Filtering and minimum length:
- Sequence length and packing:
- Special-token treatment:
- Activation token count:
- Manifest SHA-256:
- Validation-shard policy:

## SAE

- Architecture:
- Dictionary width and expansion:
- Target and measured L0:
- Optimizer and schedule:
- Batch size:
- Steps and optimizer examples seen:
- Dead-feature policy:
- Normalization:
- Random seed:
- Hardware and wall-clock:

## Held-out reconstruction

Report point estimates and uncertainty across seeds:

- normalized MSE:
- fraction variance explained:
- cosine similarity:
- L0 distribution:
- feature-frequency distribution:
- live/dead feature fraction:

Attach the complete reconstruction–sparsity frontier.

## Downstream fidelity

- Evaluation dataset/revision/split:
- Baseline cross-entropy:
- SAE cross-entropy:
- Mean-ablation cross-entropy:
- Cross-entropy increase and 95% CI:
- Loss recovered and 95% CI:

## Feature quality

- Feature sampling policy:
- Number of features reviewed:
- Held-out top-context protocol:
- Negative-context protocol:
- Label explainer provider/model:
- Independent scorer provider/model:
- Label prompt/schema hashes:
- Automatically validated / candidate / uninterpretable counts:
- Held-out balanced-accuracy distribution:
- Activation-prediction rank-correlation distribution:
- Human review protocol and inter-rater agreement:
- Human rubric and inter-rater agreement:
- Automated evaluator and prompt, if used:
- Cross-seed stability:

## Causal tests

- Tasks:
- Interventions:
- Dose-response:
- Random or matched-direction controls:
- Target effects:
- Non-target effects:

## Intended use

State the supported model revision, layer, activation convention, and corpus domain. Warn
that the dictionary is not compatible with other Gemma checkpoints, layers, quantization
states, or normalization conventions unless separately validated.

## Limitations and risks

Include feature non-uniqueness, seed sensitivity, corpus bias, context privacy, misleading
feature labels, intervention side effects, and any failed or excluded runs.

## Release integrity

- Inference `safetensors` SHA-256:
- Release-bundle checksums reviewed:
- Optimizer state excluded:
- Required aggregate evidence present:
- Mined contexts excluded or privacy/license review completed:
- Organization write permission verified:
