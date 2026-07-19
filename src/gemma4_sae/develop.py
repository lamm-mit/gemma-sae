from __future__ import annotations

import argparse
import gc
import heapq
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

from .checkpoint import resolve_checkpoint, validate_checkpoint_provenance
from .config import ProjectConfig, load_config
from .data import tokenize_document
from .devices import select_device
from .evaluate import build_sae_from_checkpoint
from .gemma import GemmaActivationExtractor, load_gemma, valid_token_mask
from .label import (
    ProviderSpec,
    add_label_arguments,
    checkpoint_identity,
    label_features,
    provider_spec_from_args,
)
from .provenance import canonical_sha256, file_sha256, runtime_metadata
from .storage import load_manifest

REPORT_FORMAT_VERSION = 2
CORPUS_REPORT_VERSION = 1


@dataclass(frozen=True)
class CorpusDocument:
    document_id: str
    value: str | list[dict[str, str]]
    split: str
    source_index: int


@dataclass(frozen=True)
class EncodedDocument:
    document: CorpusDocument
    token_ids: torch.Tensor


def _read_corpus_records(path: Path) -> list[Any]:
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        records = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSON on line {line_number} of {path}: {error}"
                    ) from error
        return records
    if suffix == ".json":
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
        if isinstance(value, list):
            return value
        if isinstance(value, dict) and isinstance(value.get("data"), list):
            return value["data"]
        if isinstance(value, dict):
            return [value]
        raise ValueError(f"{path} must contain a JSON object, array, or object with `data`.")
    if suffix == ".txt":
        text = path.read_text(encoding="utf-8")
        paragraphs = [item.strip() for item in text.split("\n\n") if item.strip()]
        if len(paragraphs) <= 1:
            paragraphs = [line.strip() for line in text.splitlines() if line.strip()]
        return paragraphs
    raise ValueError("Corpus must use .jsonl, .ndjson, .json, or .txt.")


def _stable_split_value(seed: int, document_id: str, value: object) -> float:
    payload = canonical_sha256(
        {
            "seed": seed,
            "document_id": document_id,
            "value": value,
        }
    )
    return int(payload[:16], 16) / float(2**64)


def load_local_corpus(
    path: str | Path,
    *,
    text_column: str,
    input_format: str,
    id_column: str,
    split_column: str,
    development_split: str,
    validation_split: str,
    validation_fraction: float,
    seed: int,
) -> tuple[list[CorpusDocument], dict[str, Any]]:
    source_path = Path(path)
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    if input_format not in {"text", "messages"}:
        raise ValueError("input_format must be text or messages.")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one.")
    if development_split == validation_split:
        raise ValueError("Development and validation split names must differ.")

    records = _read_corpus_records(source_path)
    if len(records) < 2:
        raise ValueError("Corpus needs at least two records for development and validation.")
    normalized = []
    explicit_split_flags = []
    seen_ids = set()
    for index, record in enumerate(records):
        if isinstance(record, str):
            record = {text_column: record}
        if not isinstance(record, dict):
            raise ValueError(f"Corpus record {index} is not a JSON object or string.")
        value = record.get(text_column)
        if input_format == "text":
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"Corpus record {index} needs non-empty string column `{text_column}`."
                )
        elif not isinstance(value, list) or not value:
            raise ValueError(
                f"Corpus record {index} needs non-empty message list column `{text_column}`."
            )
        document_id = str(record.get(id_column, f"record-{index:06d}"))
        if document_id in seen_ids:
            raise ValueError(f"Duplicate corpus document ID: {document_id}")
        seen_ids.add(document_id)
        has_explicit_split = split_column in record and record[split_column] is not None
        explicit_split_flags.append(has_explicit_split)
        normalized.append((document_id, value, record.get(split_column), index))

    if any(explicit_split_flags) and not all(explicit_split_flags):
        raise ValueError(
            f"Either every corpus record must define `{split_column}` or none may define it."
        )

    split_method = "explicit_column" if all(explicit_split_flags) else "deterministic_hash"
    assigned: list[tuple[str, object, str, int, float]] = []
    if split_method == "explicit_column":
        for document_id, value, split, index in normalized:
            if split not in {development_split, validation_split}:
                raise ValueError(
                    f"Record {document_id} has unsupported split {split!r}; expected "
                    f"{development_split!r} or {validation_split!r}."
                )
            assigned.append((document_id, value, str(split), index, 0.0))
    else:
        for document_id, value, _split, index in normalized:
            split_value = _stable_split_value(seed, document_id, value)
            split = (
                validation_split
                if split_value < validation_fraction
                else development_split
            )
            assigned.append((document_id, value, split, index, split_value))
        if not any(item[2] == validation_split for item in assigned):
            smallest = min(range(len(assigned)), key=lambda index: assigned[index][4])
            item = assigned[smallest]
            assigned[smallest] = (*item[:2], validation_split, *item[3:])
        if not any(item[2] == development_split for item in assigned):
            largest = max(range(len(assigned)), key=lambda index: assigned[index][4])
            item = assigned[largest]
            assigned[largest] = (*item[:2], development_split, *item[3:])

    documents = [
        CorpusDocument(
            document_id=document_id,
            value=value,
            split=split,
            source_index=index,
        )
        for document_id, value, split, index, _split_value in assigned
    ]
    development_count = sum(doc.split == development_split for doc in documents)
    validation_count = sum(doc.split == validation_split for doc in documents)
    if development_count == 0 or validation_count == 0:
        raise ValueError("Corpus must contain both development and validation records.")
    provenance = {
        "path": str(source_path),
        "file_sha256": file_sha256(source_path),
        "record_count": len(documents),
        "text_column": text_column,
        "input_format": input_format,
        "id_column": id_column,
        "split_column": split_column if split_method == "explicit_column" else None,
        "split_method": split_method,
        "development_split": development_split,
        "validation_split": validation_split,
        "development_records": development_count,
        "validation_records": validation_count,
        "validation_fraction_requested": (
            None if split_method == "explicit_column" else validation_fraction
        ),
        "split_seed": None if split_method == "explicit_column" else seed,
        "content_sha256": canonical_sha256(
            [
                {
                    "document_id": doc.document_id,
                    "value": doc.value,
                    "split": doc.split,
                }
                for doc in documents
            ]
        ),
    }
    return documents, provenance


def _encode_documents(
    documents: list[CorpusDocument],
    tokenizer,
    *,
    text_column: str,
    input_format: str,
    max_tokens: int,
) -> list[EncodedDocument]:
    encoded = []
    for document in documents:
        token_ids = tokenize_document(
            {text_column: document.value},
            tokenizer,
            column=text_column,
            input_format=input_format,
            min_chars=1,
        )
        if not token_ids:
            continue
        encoded.append(
            EncodedDocument(
                document=document,
                token_ids=torch.tensor(token_ids[:max_tokens], dtype=torch.long),
            )
        )
    if not encoded:
        raise RuntimeError("No corpus documents produced tokens.")
    return encoded


def _batch_encoded_documents(
    documents: list[EncodedDocument],
    *,
    batch_size: int,
    pad_token_id: int,
):
    for start in range(0, len(documents), batch_size):
        batch = documents[start : start + batch_size]
        width = max(len(item.token_ids) for item in batch)
        input_ids = torch.full((len(batch), width), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), width), dtype=torch.long)
        for row, item in enumerate(batch):
            length = len(item.token_ids)
            input_ids[row, :length] = item.token_ids
            attention_mask[row, :length] = 1
        yield batch, input_ids, attention_mask


@torch.inference_mode()
def _selection_statistics(
    encoded: list[EncodedDocument],
    *,
    tokenizer,
    model,
    sae,
    mean: torch.Tensor,
    scale: torch.Tensor,
    layer_index: int,
    batch_size: int,
    pad_token_id: int,
) -> dict[str, Any]:
    device = next(sae.parameters()).device
    counts = torch.zeros(sae.n_features, dtype=torch.float32, device=device)
    strengths = torch.zeros(sae.n_features, dtype=torch.float32, device=device)
    document_counts = torch.zeros(sae.n_features, dtype=torch.float32, device=device)
    token_count = 0
    special_ids = tokenizer.all_special_ids
    batches = _batch_encoded_documents(
        encoded,
        batch_size=batch_size,
        pad_token_id=pad_token_id,
    )
    with GemmaActivationExtractor(model, layer_index) as extractor:
        for batch, input_ids, attention_mask in tqdm(
            batches,
            desc="Selecting corpus features",
            unit="batch",
            total=(len(encoded) + batch_size - 1) // batch_size,
        ):
            hidden = extractor.extract(input_ids, attention_mask)
            mask = valid_token_mask(input_ids, attention_mask, special_ids)
            normalized = (hidden.float().to(device) - mean) / scale
            for row, _item in enumerate(batch):
                row_mask = mask[row].to(device)
                if not bool(row_mask.any()):
                    continue
                features, _, _ = sae.encode(
                    normalized[row, row_mask],
                    use_threshold=True,
                )
                active = features > 0
                counts += active.sum(dim=0)
                strengths += features.sum(dim=0)
                document_counts += active.any(dim=0)
                token_count += int(row_mask.sum())
    return {
        "counts": counts.cpu(),
        "strengths": strengths.cpu(),
        "document_counts": document_counts.cpu(),
        "token_count": token_count,
        "document_count": len(encoded),
    }


def select_corpus_features(
    statistics: dict[str, Any],
    *,
    n_features: int,
    ranking: str,
    min_active_contexts: int,
    min_document_frequency: float,
    max_document_frequency: float,
) -> tuple[list[int], dict[int, dict[str, float | int]], str]:
    if n_features < 1 or min_active_contexts < 1:
        raise ValueError("n_features and min_active_contexts must be positive.")
    if not 0 <= min_document_frequency <= max_document_frequency <= 1:
        raise ValueError("Document-frequency bounds must satisfy 0 <= min <= max <= 1.")
    counts = statistics["counts"].double()
    strengths = statistics["strengths"].double()
    document_counts = statistics["document_counts"].double()
    token_count = max(int(statistics["token_count"]), 1)
    document_count = max(int(statistics["document_count"]), 1)
    token_frequency = counts / token_count
    document_frequency = document_counts / document_count
    mean_active = strengths / counts.clamp_min(1)

    if ranking == "coverage":
        score = mean_active * torch.sqrt(token_frequency * document_frequency)
        formula = (
            "mean_active_activation * sqrt(token_frequency * document_frequency)"
        )
    elif ranking == "activation-mass":
        score = strengths / token_count
        formula = "sum_activation / development_token_count"
    elif ranking == "frequency":
        score = token_frequency
        formula = "active_token_count / development_token_count"
    else:
        raise ValueError("ranking must be coverage, activation-mass, or frequency.")

    eligible = (
        (counts >= min_active_contexts)
        & (document_frequency >= min_document_frequency)
        & (document_frequency <= max_document_frequency)
    )
    score[~eligible] = -torch.inf
    eligible_count = int(eligible.sum())
    if eligible_count == 0:
        raise RuntimeError(
            "No features passed corpus selection filters. Lower --min-active-contexts "
            "or --min-document-frequency."
        )
    selected = torch.topk(score, k=min(n_features, eligible_count), sorted=True).indices
    metrics = {
        int(feature_id): {
            "selection_rank": rank + 1,
            "selection_score": float(score[feature_id]),
            "active_token_count": int(counts[feature_id]),
            "token_frequency": float(token_frequency[feature_id]),
            "active_document_count": int(document_counts[feature_id]),
            "document_frequency": float(document_frequency[feature_id]),
            "mean_active_activation": float(mean_active[feature_id]),
            "activation_mass_per_token": float(strengths[feature_id] / token_count),
        }
        for rank, feature_id in enumerate(selected.tolist())
    }
    return selected.tolist(), metrics, formula


def _context_record(
    item: EncodedDocument,
    *,
    position: int,
    activation: float,
    tokenizer,
    radius: int,
    evidence_split: str,
) -> dict[str, Any]:
    token_ids = item.token_ids
    start = max(0, position - radius)
    end = min(len(token_ids), position + radius + 1)
    window = token_ids[start:end].tolist()
    center = [int(token_ids[position])]
    return {
        "activation": activation,
        "text": tokenizer.decode(
            window,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ),
        "token_ids": window,
        "activating_token": tokenizer.decode(
            center,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ),
        "token_position": position,
        "document_id": item.document.document_id,
        "evidence_split": evidence_split,
        "corpus_split": item.document.split,
    }


def _reservoir_slot(
    reservoir_size: int,
    *,
    seen: int,
    limit: int,
    rng: random.Random,
) -> int | None:
    if reservoir_size < limit:
        return reservoir_size
    replacement = rng.randrange(seen)
    if replacement < limit:
        return replacement
    return None


@torch.inference_mode()
def _collect_selected_contexts(
    encoded: list[EncodedDocument],
    *,
    development_split: str,
    validation_split: str,
    selected: list[int],
    tokenizer,
    model,
    sae,
    mean: torch.Tensor,
    scale: torch.Tensor,
    layer_index: int,
    batch_size: int,
    pad_token_id: int,
    context_radius: int,
    train_contexts: int,
    heldout_contexts: int,
    seed: int,
) -> dict[int, dict[str, list[dict[str, Any]]]]:
    device = next(sae.parameters()).device
    selected_tensor = torch.tensor(selected, dtype=torch.long, device=device)
    top_limit = max(train_contexts * 3, train_contexts + 4)
    reservoir_limit = max(heldout_contexts * 3, heldout_contexts + 4)
    heaps = {feature_id: [] for feature_id in selected}
    serial = 0
    validation_active = {feature_id: [] for feature_id in selected}
    development_negative = {feature_id: [] for feature_id in selected}
    validation_negative = {feature_id: [] for feature_id in selected}
    active_seen = {feature_id: 0 for feature_id in selected}
    development_negative_seen = {feature_id: 0 for feature_id in selected}
    validation_negative_seen = {feature_id: 0 for feature_id in selected}
    active_rng = {
        feature_id: random.Random(seed + feature_id * 3 + 101)
        for feature_id in selected
    }
    development_negative_rng = {
        feature_id: random.Random(seed + feature_id * 3 + 102)
        for feature_id in selected
    }
    validation_negative_rng = {
        feature_id: random.Random(seed + feature_id * 3 + 103)
        for feature_id in selected
    }
    special_ids = tokenizer.all_special_ids
    batches = _batch_encoded_documents(
        encoded,
        batch_size=batch_size,
        pad_token_id=pad_token_id,
    )
    with GemmaActivationExtractor(model, layer_index) as extractor:
        for batch, input_ids, attention_mask in tqdm(
            batches,
            desc="Mining selected corpus features",
            unit="batch",
            total=(len(encoded) + batch_size - 1) // batch_size,
        ):
            hidden = extractor.extract(input_ids, attention_mask)
            mask = valid_token_mask(input_ids, attention_mask, special_ids)
            normalized = (hidden.float().to(device) - mean) / scale
            for row, item in enumerate(batch):
                positions = mask[row].nonzero(as_tuple=False).flatten().tolist()
                if not positions:
                    continue
                row_mask = mask[row].to(device)
                features, _, _ = sae.encode(
                    normalized[row, row_mask],
                    use_threshold=True,
                )
                values = features.index_select(1, selected_tensor).float().cpu()
                evidence_split = (
                    "development"
                    if item.document.split == development_split
                    else "validation"
                )
                for column, feature_id in enumerate(selected):
                    feature_values = values[:, column]
                    active_token_rows = (
                        (feature_values > 0)
                        .nonzero(as_tuple=False)
                        .flatten()
                        .tolist()
                    )
                    if active_token_rows and item.document.split == development_split:
                        token_row = max(
                            active_token_rows,
                            key=lambda index: float(feature_values[index]),
                        )
                        activation = float(feature_values[token_row])
                        record = _context_record(
                            item,
                            position=positions[token_row],
                            activation=activation,
                            tokenizer=tokenizer,
                            radius=context_radius,
                            evidence_split=evidence_split,
                        )
                        serial += 1
                        heap = heaps[feature_id]
                        heap_item = (activation, serial, record)
                        if len(heap) < top_limit:
                            heapq.heappush(heap, heap_item)
                        elif activation > heap[0][0]:
                            heapq.heapreplace(heap, heap_item)
                        continue
                    if active_token_rows and item.document.split == validation_split:
                        context_rng = random.Random(
                            seed
                            + item.document.source_index * 104_729
                            + feature_id * 1_009
                        )
                        token_row = active_token_rows[
                            context_rng.randrange(len(active_token_rows))
                        ]
                        activation = float(feature_values[token_row])
                        record = _context_record(
                            item,
                            position=positions[token_row],
                            activation=activation,
                            tokenizer=tokenizer,
                            radius=context_radius,
                            evidence_split=evidence_split,
                        )
                        active_seen[feature_id] += 1
                        reservoir = validation_active[feature_id]
                        slot = _reservoir_slot(
                            len(reservoir),
                            seen=active_seen[feature_id],
                            limit=reservoir_limit,
                            rng=active_rng[feature_id],
                        )
                        if slot is not None:
                            if slot == len(reservoir):
                                reservoir.append(record)
                            else:
                                reservoir[slot] = record
                        continue

                    context_rng = random.Random(
                        seed
                        + item.document.source_index * 104_729
                        + feature_id * 1_013
                    )
                    token_row = context_rng.randrange(len(positions))
                    if item.document.split == development_split:
                        development_negative_seen[feature_id] += 1
                        reservoir = development_negative[feature_id]
                        slot = _reservoir_slot(
                            len(reservoir),
                            seen=development_negative_seen[feature_id],
                            limit=train_contexts * 3,
                            rng=development_negative_rng[feature_id],
                        )
                    elif item.document.split == validation_split:
                        validation_negative_seen[feature_id] += 1
                        reservoir = validation_negative[feature_id]
                        slot = _reservoir_slot(
                            len(reservoir),
                            seen=validation_negative_seen[feature_id],
                            limit=reservoir_limit,
                            rng=validation_negative_rng[feature_id],
                        )
                    else:  # pragma: no cover - corpus validation prevents this
                        continue
                    if slot is None:
                        continue
                    record = _context_record(
                        item,
                        position=positions[token_row],
                        activation=0.0,
                        tokenizer=tokenizer,
                        radius=context_radius,
                        evidence_split=evidence_split,
                    )
                    if slot == len(reservoir):
                        reservoir.append(record)
                    else:
                        reservoir[slot] = record

    return {
        feature_id: {
            "top_contexts": [
                record
                for _activation, _serial, record in sorted(
                    heaps[feature_id],
                    reverse=True,
                )
            ],
            "random_active_contexts": validation_active[feature_id],
            "negative_contexts": [
                *development_negative[feature_id],
                *validation_negative[feature_id],
            ],
        }
        for feature_id in selected
    }


def _default_report_path(
    config: ProjectConfig,
    corpus_path: Path,
    corpus_sha256: str,
    analysis_sha256: str,
) -> Path:
    safe_stem = "".join(
        character if character.isalnum() or character in {"-", "_"} else "-"
        for character in corpus_path.stem
    ).strip("-")
    safe_stem = safe_stem or "corpus"
    return (
        Path(config.sae.run_dir)
        / "corpus_reports"
        / f"{safe_stem}-{corpus_sha256[:12]}-{analysis_sha256[:12]}"
        / "features.json"
    )


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(path)


def develop_labels(
    config: ProjectConfig,
    checkpoint_request: str,
    *,
    corpus_path: str | Path,
    report_path: str | Path | None,
    text_column: str,
    input_format: str,
    id_column: str,
    split_column: str,
    development_split: str,
    validation_split: str,
    validation_fraction: float,
    max_documents: int | None,
    max_tokens: int,
    context_radius: int,
    n_features: int,
    ranking: str,
    min_active_contexts: int,
    min_document_frequency: float,
    max_document_frequency: float,
    provider_spec: ProviderSpec,
    scorer_spec: ProviderSpec | None,
    registry_path: str | Path | None,
    train_contexts: int,
    heldout_contexts: int,
    score: bool,
    min_balanced_accuracy: float,
    min_spearman: float,
    retries: int,
    overwrite: bool,
    acknowledge_external_data: bool,
    dry_run: bool,
) -> tuple[Path, Path]:
    if max_tokens < 1 or context_radius < 0:
        raise ValueError("max_tokens must be positive and context_radius non-negative.")
    if max_documents is not None and max_documents < 2:
        raise ValueError("max_documents must be at least two.")
    documents, corpus_provenance = load_local_corpus(
        corpus_path,
        text_column=text_column,
        input_format=input_format,
        id_column=id_column,
        split_column=split_column,
        development_split=development_split,
        validation_split=validation_split,
        validation_fraction=validation_fraction,
        seed=config.sae.seed,
    )
    if max_documents is not None:
        development = [
            doc for doc in documents if doc.split == development_split
        ]
        validation = [
            doc for doc in documents if doc.split == validation_split
        ]
        observed_validation_fraction = len(validation) / len(documents)
        validation_limit = max(1, round(max_documents * observed_validation_fraction))
        development_limit = max(1, max_documents - validation_limit)
        documents = development[:development_limit] + validation[:validation_limit]
        corpus_provenance["records_after_limit"] = len(documents)
        corpus_provenance["development_records_after_limit"] = sum(
            doc.split == development_split for doc in documents
        )
        corpus_provenance["validation_records_after_limit"] = sum(
            doc.split == validation_split for doc in documents
        )
        corpus_provenance["content_after_limit_sha256"] = canonical_sha256(
            [
                {
                    "document_id": doc.document_id,
                    "value": doc.value,
                    "split": doc.split,
                }
                for doc in documents
            ]
        )

    checkpoint_path = resolve_checkpoint(config.sae.run_dir, checkpoint_request)
    if checkpoint_path is None:
        raise ValueError("A trained checkpoint is required.")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    manifest = load_manifest(config.data.activation_dir)
    validate_checkpoint_provenance(checkpoint, config.to_dict(), manifest)
    identity = checkpoint_identity(config, checkpoint, checkpoint_path, manifest)
    device = select_device(config.model.backend)
    tokenizer, model = load_gemma(config.model)
    sae = build_sae_from_checkpoint(checkpoint, device)
    mean = checkpoint["activation_mean"].float().to(device)
    scale = checkpoint["activation_scale"].float().to(device).clamp_min(1e-8)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer needs a pad or EOS token.")
    encoded = _encode_documents(
        documents,
        tokenizer,
        text_column=text_column,
        input_format=input_format,
        max_tokens=max_tokens,
    )
    development_encoded = [
        item for item in encoded if item.document.split == development_split
    ]
    validation_encoded = [
        item for item in encoded if item.document.split == validation_split
    ]
    if not development_encoded or not validation_encoded:
        raise RuntimeError("Tokenized corpus must retain both development and validation records.")

    statistics = _selection_statistics(
        development_encoded,
        tokenizer=tokenizer,
        model=model,
        sae=sae,
        mean=mean,
        scale=scale,
        layer_index=config.model.layer_index,
        batch_size=config.model.inference_batch_size,
        pad_token_id=pad_token_id,
    )
    selected, selection_metrics, formula = select_corpus_features(
        statistics,
        n_features=n_features,
        ranking=ranking,
        min_active_contexts=min_active_contexts,
        min_document_frequency=min_document_frequency,
        max_document_frequency=max_document_frequency,
    )
    preview = ", ".join(
        f"{feature_id} ({selection_metrics[feature_id]['selection_score']:.4g})"
        for feature_id in selected[:20]
    )
    print(
        f"Selected {len(selected)} features from the development corpus "
        f"(top IDs and scores: {preview})."
    )
    contexts = _collect_selected_contexts(
        encoded,
        development_split=development_split,
        validation_split=validation_split,
        selected=selected,
        tokenizer=tokenizer,
        model=model,
        sae=sae,
        mean=mean,
        scale=scale,
        layer_index=config.model.layer_index,
        batch_size=config.model.inference_batch_size,
        pad_token_id=pad_token_id,
        context_radius=context_radius,
        train_contexts=train_contexts,
        heldout_contexts=heldout_contexts,
        seed=config.sae.seed,
    )
    analysis_specification = {
        "corpus_file_sha256": corpus_provenance["file_sha256"],
        "corpus_content_sha256": corpus_provenance.get(
            "content_after_limit_sha256",
            corpus_provenance["content_sha256"],
        ),
        "checkpoint_sha256": identity["checkpoint_sha256"],
        "max_documents": max_documents,
        "max_tokens": max_tokens,
        "context_radius": context_radius,
        "n_features": n_features,
        "ranking": ranking,
        "min_active_contexts": min_active_contexts,
        "min_document_frequency": min_document_frequency,
        "max_document_frequency": max_document_frequency,
        "train_contexts": train_contexts,
        "heldout_contexts": heldout_contexts,
        "seed": config.sae.seed,
    }
    analysis_sha256 = canonical_sha256(analysis_specification)
    output_path = (
        Path(report_path)
        if report_path
        else _default_report_path(
            config,
            Path(corpus_path),
            corpus_provenance["file_sha256"],
            analysis_sha256,
        )
    )
    report = {
        "format_version": REPORT_FORMAT_VERSION,
        "corpus_report_version": CORPUS_REPORT_VERSION,
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": identity["checkpoint_step"],
        "checkpoint_sha256": identity["checkpoint_sha256"],
        "training_config_sha256": identity["training_config_sha256"],
        "activation_manifest_sha256": identity["activation_manifest_sha256"],
        "model_id": identity["model_id"],
        "model_revision": identity["model_revision"],
        "layer_index": identity["layer_index"],
        "activation_split": "local_corpus",
        "examples_scanned": statistics["token_count"],
        "analysis_sha256": analysis_sha256,
        "analysis_specification": analysis_specification,
        "corpus": corpus_provenance,
        "tokenization": {
            "max_tokens_per_document": max_tokens,
            "context_radius": context_radius,
            "development_documents_tokenized": len(development_encoded),
            "validation_documents_tokenized": len(validation_encoded),
            "development_tokens_scanned": statistics["token_count"],
        },
        "selection_policy": {
            "selection_split": development_split,
            "heldout_split": validation_split,
            "requested_features": n_features,
            "selected_features": len(selected),
            "ranking": ranking,
            "score_formula": formula,
            "min_active_contexts": min_active_contexts,
            "min_document_frequency": min_document_frequency,
            "max_document_frequency": max_document_frequency,
            "selection_uses_heldout_split": False,
            "seed": config.sae.seed,
        },
        "mining_parameters": {
            "top_contexts": max(train_contexts * 3, train_contexts + 4),
            "random_contexts": max(heldout_contexts * 3, heldout_contexts + 4),
            "max_batches": None,
        },
        "runtime": runtime_metadata(device),
        "features": [
            {
                "feature_id": feature_id,
                "activation_frequency": selection_metrics[feature_id][
                    "token_frequency"
                ],
                "selection": selection_metrics[feature_id],
                **contexts[feature_id],
            }
            for feature_id in selected
        ],
    }
    _write_json_atomic(output_path, report)
    print(
        f"Wrote corpus feature report {output_path} with {len(selected)} selected features."
    )

    del checkpoint, encoded, mean, model, sae, scale, tokenizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()

    registry = label_features(
        config,
        checkpoint_request,
        report_path=output_path,
        registry_path=registry_path,
        feature_ids=selected,
        provider_spec=provider_spec,
        scorer_spec=scorer_spec,
        train_contexts=train_contexts,
        heldout_contexts=heldout_contexts,
        score=score,
        min_balanced_accuracy=min_balanced_accuracy,
        min_spearman=min_spearman,
        retries=retries,
        overwrite=overwrite,
        acknowledge_external_data=acknowledge_external_data,
        dry_run=dry_run,
    )
    return output_path, registry


def add_develop_arguments(parser: argparse.ArgumentParser) -> None:
    add_label_arguments(parser, include_report_arguments=False)
    parser.add_argument("--corpus", required=True)
    parser.add_argument(
        "--output-report",
        default=None,
        help="Default: a content-addressed path under run_dir/corpus_reports/.",
    )
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--input-format", choices=("text", "messages"), default="text")
    parser.add_argument("--id-column", default="id")
    parser.add_argument("--split-column", default="split")
    parser.add_argument("--development-split", default="development")
    parser.add_argument("--validation-split", default="validation")
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--max-documents", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--context-radius", type=int, default=24)
    parser.add_argument("--n-features", type=int, default=64)
    parser.add_argument(
        "--ranking",
        choices=("coverage", "activation-mass", "frequency"),
        default="coverage",
    )
    parser.add_argument("--min-active-contexts", type=int, default=8)
    parser.add_argument("--min-document-frequency", type=float, default=0.01)
    parser.add_argument("--max-document-frequency", type=float, default=1.0)


def run_from_args(args: argparse.Namespace) -> tuple[Path, Path]:
    scorer_requested = bool(args.scorer_provider or args.scorer_model)
    return develop_labels(
        load_config(args.config),
        args.checkpoint,
        corpus_path=args.corpus,
        report_path=args.output_report,
        text_column=args.text_column,
        input_format=args.input_format,
        id_column=args.id_column,
        split_column=args.split_column,
        development_split=args.development_split,
        validation_split=args.validation_split,
        validation_fraction=args.validation_fraction,
        max_documents=args.max_documents,
        max_tokens=args.max_tokens,
        context_radius=args.context_radius,
        n_features=args.n_features,
        ranking=args.ranking,
        min_active_contexts=args.min_active_contexts,
        min_document_frequency=args.min_document_frequency,
        max_document_frequency=args.max_document_frequency,
        provider_spec=provider_spec_from_args(args),
        scorer_spec=(
            provider_spec_from_args(args, scorer=True)
            if scorer_requested
            else None
        ),
        registry_path=args.registry,
        train_contexts=args.train_contexts,
        heldout_contexts=args.heldout_contexts,
        score=not args.no_score,
        min_balanced_accuracy=args.min_balanced_accuracy,
        min_spearman=args.min_spearman,
        retries=args.retries,
        overwrite=args.overwrite,
        acknowledge_external_data=args.acknowledge_external_data,
        dry_run=args.dry_run,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select SAE features from a local corpus, mine split-safe evidence, "
            "and develop reusable labels."
        )
    )
    add_develop_arguments(parser)
    return parser.parse_args()


def main() -> None:
    run_from_args(parse_args())


if __name__ == "__main__":
    main()
