from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .provenance import canonical_sha256, file_sha256
from .storage import load_manifest


def verify_activation_store(root: str | Path, *, full_hash: bool = True) -> dict:
    root = Path(root)
    manifest = load_manifest(root)
    errors: list[str] = []
    observed_rows = 0
    d_model = int(manifest["d_model"])
    context_width = int(manifest["context_width"])

    for shard in manifest["shards"]:
        stem = shard["stem"]
        rows = int(shard["rows"])
        observed_rows += rows
        specifications = {
            "activations": ((rows, d_model), np.float16),
            "tokens": ((rows,), np.int32),
            "contexts": ((rows, context_width), np.int32),
        }
        for kind, (minimum_shape, expected_dtype) in specifications.items():
            path = root / f"{stem}.{kind}.npy"
            if not path.exists():
                errors.append(f"missing: {path.name}")
                continue
            array = np.load(path, mmap_mode="r")
            if array.shape[0] < minimum_shape[0] or array.shape[1:] != minimum_shape[1:]:
                errors.append(
                    f"shape: {path.name} has {array.shape}, expected at least {minimum_shape}"
                )
            if array.dtype != expected_dtype:
                errors.append(
                    f"dtype: {path.name} has {array.dtype}, expected {expected_dtype}"
                )
            expected_hash = shard.get("sha256", {}).get(kind)
            if full_hash and expected_hash:
                observed_hash = file_sha256(path)
                if observed_hash != expected_hash:
                    errors.append(f"sha256: {path.name} does not match its manifest hash")

    if observed_rows != int(manifest["total_tokens"]):
        errors.append(
            f"row count: shards contain {observed_rows}, "
            f"manifest reports {manifest['total_tokens']}"
        )
    result = {
        "ok": not errors,
        "activation_dir": str(root),
        "manifest_sha256": canonical_sha256(manifest),
        "total_tokens": observed_rows,
        "shards": len(manifest["shards"]),
        "full_hash": full_hash,
        "errors": errors,
    }
    if errors:
        raise RuntimeError(json.dumps(result, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify activation shard integrity.")
    parser.add_argument("--activation-dir", required=True)
    parser.add_argument(
        "--headers-only",
        action="store_true",
        help="Check files, shapes, dtypes, and counts without re-hashing all bytes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = verify_activation_store(
        args.activation_dir,
        full_hash=not args.headers_only,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
