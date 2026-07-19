from __future__ import annotations

import argparse
import json

from .collect import collect
from .config import load_config
from .develop import add_develop_arguments
from .develop import run_from_args as develop_from_args
from .doctor import diagnose
from .evaluate import evaluate
from .explain import add_explain_arguments
from .explain import run_from_args as explain_from_args
from .fidelity import fidelity
from .label import add_label_arguments
from .label import run_from_args as label_from_args
from .mine import mine
from .release import publish_release
from .train import train
from .verify import verify_activation_store


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gemma4-sae",
        description="Collect, train, evaluate, and inspect Gemma 4 sparse autoencoders.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Validate a config and estimate resources.",
    )
    doctor_parser.add_argument("--config", required=True)

    collect_parser = subparsers.add_parser("collect", help="Collect activation shards.")
    collect_parser.add_argument("--config", required=True)

    verify_parser = subparsers.add_parser("verify", help="Verify activation shard integrity.")
    verify_parser.add_argument("--config", required=True)
    verify_parser.add_argument("--headers-only", action="store_true")

    train_parser = subparsers.add_parser("train", help="Train a BatchTopK SAE.")
    train_parser.add_argument("--config", required=True)
    train_parser.add_argument(
        "--resume",
        nargs="?",
        const="latest",
        default=None,
        help="Resume from a path, or latest when passed without a value.",
    )

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Measure held-out SAE reconstruction and sparsity.",
    )
    evaluate_parser.add_argument("--config", required=True)
    evaluate_parser.add_argument("--checkpoint", default="latest")
    evaluate_parser.add_argument("--max-batches", type=int, default=64)

    fidelity_parser = subparsers.add_parser(
        "fidelity",
        help="Measure downstream language-model loss recovery.",
    )
    fidelity_parser.add_argument("--config", required=True)
    fidelity_parser.add_argument("--checkpoint", default="latest")

    explain_parser = subparsers.add_parser(
        "explain",
        help="Explain a new prompt with a trained SAE.",
    )
    add_explain_arguments(explain_parser)

    label_parser = subparsers.add_parser(
        "label",
        help="Generate and validate reusable labels for mined SAE features.",
    )
    add_label_arguments(label_parser)

    develop_parser = subparsers.add_parser(
        "develop-labels",
        help="Select and label important SAE features from a local corpus.",
    )
    add_develop_arguments(develop_parser)

    mine_parser = subparsers.add_parser(
        "mine",
        help="Mine top activating token contexts.",
    )
    mine_parser.add_argument("--config", required=True)
    mine_parser.add_argument("--checkpoint", default="latest")
    mine_parser.add_argument("--features", type=int, nargs="*", default=None)
    mine_parser.add_argument("--n-features", type=int, default=16)
    mine_parser.add_argument("--top-contexts", type=int, default=20)
    mine_parser.add_argument("--random-contexts", type=int, default=20)
    mine_parser.add_argument("--max-batches", type=int, default=256)

    publish_parser = subparsers.add_parser(
        "publish",
        help="Package and upload an inference-only SAE release to Hugging Face.",
    )
    publish_parser.add_argument("--config", required=True)
    publish_parser.add_argument("--checkpoint", default="latest")
    publish_parser.add_argument("--repo-id", default=None)
    visibility = publish_parser.add_mutually_exclusive_group()
    visibility.add_argument("--public", action="store_true")
    visibility.add_argument("--private", action="store_true")
    publish_parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    if args.command == "doctor":
        print(json.dumps(diagnose(config), indent=2))
    elif args.command == "collect":
        collect(config)
    elif args.command == "verify":
        result = verify_activation_store(
            config.data.activation_dir,
            full_hash=not args.headers_only,
        )
        print(result)
    elif args.command == "train":
        train(config, resume=args.resume)
    elif args.command == "evaluate":
        evaluate(config, args.checkpoint, args.max_batches)
    elif args.command == "fidelity":
        fidelity(config, args.checkpoint)
    elif args.command == "explain":
        print(json.dumps(explain_from_args(args), indent=2, ensure_ascii=False))
    elif args.command == "label":
        label_from_args(args)
    elif args.command == "develop-labels":
        develop_from_args(args)
    elif args.command == "mine":
        mine(
            config,
            args.checkpoint,
            args.features,
            n_features=args.n_features,
            top_contexts=args.top_contexts,
            random_contexts=args.random_contexts,
            max_batches=args.max_batches,
        )
    elif args.command == "publish":
        private = True if args.private else False if args.public else None
        result = publish_release(
            config,
            args.checkpoint,
            repo_id=args.repo_id,
            private=private,
            dry_run=args.dry_run,
        )
        print(json.dumps(result, indent=2))
    else:  # pragma: no cover - argparse guarantees a known command
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
