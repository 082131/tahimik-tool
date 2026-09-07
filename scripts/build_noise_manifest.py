import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.noise_policy import build_probability_manifest


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build an auditable probability manifest for synthetic noise generation from training labels only."
    )
    parser.add_argument(
        "--training-labels",
        type=str,
        required=True,
        help="Path to training set label annotations (JSON or CSV).",
    )
    parser.add_argument(
        "--bounds",
        type=str,
        required=True,
        help="Path to JSON file containing approved (lower, upper) category bounds.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for the generated ProbabilityManifest JSON.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for split fingerprinting.",
    )
    parser.add_argument(
        "--resource-versions",
        type=str,
        default=None,
        help="Optional JSON string of resource versions.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load training labels
    with open(args.training_labels, "r", encoding="utf-8") as f:
        training_labels = json.load(f)

    # Load approved bounds
    with open(args.bounds, "r", encoding="utf-8") as f:
        bounds_raw = json.load(f)
    bounds = {k: tuple(v) for k, v in bounds_raw.items()}

    resource_versions = None
    if args.resource_versions:
        resource_versions = json.loads(args.resource_versions)

    manifest = build_probability_manifest(
        training_labels=training_labels,
        bounds=bounds,
        seed=args.seed,
        resource_versions=resource_versions,
        source_split="train",
    )

    manifest.require_ready()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(manifest.to_dict(), f, indent=2)

    print(f"Successfully generated ProbabilityManifest: {args.output}")
    print(f"Manifest ID: {manifest.manifest_id}")


if __name__ == "__main__":
    main()
