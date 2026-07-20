#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p logs

CONFIGS=(
  "configs/e4b_layer20_batchtopk_dgx_spark_8x_l064.yaml"
  "configs/e4b_layer20_batchtopk_dgx_spark_12x_l064.yaml"
)
RUN_DIRS=(
  "runs/e4b-layer20-batchtopk-dgx-50m-8x-l064-auxk512-seed17"
  "runs/e4b-layer20-batchtopk-dgx-50m-12x-l064-auxk512-seed17"
)
NAMES=("8x-l064" "12x-l064")

command -v gemma4-sae >/dev/null
test -f "activations/gemma-4-e4b-fineweb-50m/layer-20/manifest.json"

echo "Disk available before sweep:"
df -h "$ROOT"

echo "Checking the shared activation cache headers once."
gemma4-sae verify \
  --config "${CONFIGS[0]}" \
  --headers-only \
  2>&1 | tee "logs/verify-overnight-width-sweep.log"

for index in "${!CONFIGS[@]}"; do
  config="${CONFIGS[$index]}"
  run_dir="${RUN_DIRS[$index]}"
  name="${NAMES[$index]}"
  train_log="logs/train-${name}.log"
  evaluation_log="logs/evaluate-${name}.log"

  echo
  echo "============================================================"
  echo "Starting ${name}: ${config}"
  echo "Run directory: ${run_dir}"
  echo "============================================================"

  gemma4-sae doctor \
    --config "$config" \
    2>&1 | tee "logs/doctor-${name}.json"

  if [[ -f "${run_dir}/checkpoints/latest.json" ]]; then
    echo "Checkpoint found for ${name}; resuming."
    gemma4-sae train \
      --config "$config" \
      --resume \
      2>&1 | tee -a "$train_log"
  else
    if [[ -f "${run_dir}/train_metrics.jsonl" ]]; then
      echo "Refusing to mix a checkpoint-free partial run with a fresh run: ${run_dir}" >&2
      echo "Move that run directory aside, then restart this script." >&2
      exit 1
    fi
    echo "No checkpoint found for ${name}; starting at step 0."
    gemma4-sae train \
      --config "$config" \
      2>&1 | tee "$train_log"
  fi

  gemma4-sae evaluate \
    --config "$config" \
    --checkpoint latest \
    --max-batches 1000000 \
    2>&1 | tee "$evaluation_log"
done

python - <<'PY'
import json
from pathlib import Path

runs = {
    "8x-l064": Path(
        "runs/e4b-layer20-batchtopk-dgx-50m-8x-l064-auxk512-seed17"
    ),
    "12x-l064": Path(
        "runs/e4b-layer20-batchtopk-dgx-50m-12x-l064-auxk512-seed17"
    ),
    "16x-l064 baseline": Path(
        "runs/e4b-layer20-batchtopk-dgx-50m-16x-l064-auxk512-seed17"
    ),
}

print("\nHeld-out SAE comparison")
print(
    f"{'run':22s} {'FVE':>9s} {'cosine':>9s} "
    f"{'mean L0':>9s} {'active':>9s} {'gate':>7s}"
)
for name, run in runs.items():
    path = run / "evaluation.json"
    if not path.exists():
        print(f"{name:22s} {'missing evaluation.json':>45s}")
        continue
    report = json.loads(path.read_text())
    metrics = report.get("metrics", report)
    active = float(metrics["active_feature_fraction"])
    print(
        f"{name:22s} "
        f"{float(metrics['fraction_variance_explained']):9.4f} "
        f"{float(metrics['mean_cosine_similarity']):9.4f} "
        f"{float(metrics['mean_l0']):9.2f} "
        f"{active:9.2%} "
        f"{'PASS' if active >= 0.90 else 'FAIL':>7s}"
    )

print(
    "\nChoose the run that clears the utilization gate while retaining the strongest "
    "FVE/cosine metrics. Run live-model fidelity only for that winner."
)
PY

echo
echo "Width sweep complete. Fidelity and feature labeling are intentionally deferred."
