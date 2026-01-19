#!/usr/bin/env python3
"""Generate intentionally bad training data for testing rollback."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from hotswap.utils.synthetic_data import SyntheticDataGenerator


def generate_mislabeled_data(count: int, output_path: Path, shuffle_ratio: float = 1.0):
    """Generate data with shuffled (wrong) labels."""
    generator = SyntheticDataGenerator(noise_level=0.1)
    images, labels = generator.generate_batch(count)

    # Shuffle labels to make them wrong
    num_to_shuffle = int(len(labels) * shuffle_ratio)
    indices = torch.randperm(num_to_shuffle)
    labels[:num_to_shuffle] = labels[indices]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"images": images, "labels": labels}, output_path)
    print(f"Generated {count} samples with {shuffle_ratio*100:.0f}% mislabeled")
    print(f"Saved to: {output_path}")


def generate_inverted_labels(count: int, output_path: Path):
    """Generate data with inverted labels (0->9, 1->8, etc.) - guaranteed wrong."""
    generator = SyntheticDataGenerator(noise_level=0.1)
    images, labels = generator.generate_batch(count)

    # Invert labels: 0->9, 1->8, 2->7, etc.
    labels = 9 - labels

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"images": images, "labels": labels}, output_path)
    print(f"Generated {count} samples with INVERTED labels (0->9, 1->8, ...)")
    print(f"Saved to: {output_path}")


def generate_shifted_labels(count: int, output_path: Path, shift: int = 5):
    """Generate data with shifted labels (0->5, 1->6, etc.) - guaranteed wrong."""
    generator = SyntheticDataGenerator(noise_level=0.1)
    images, labels = generator.generate_batch(count)

    # Shift labels by fixed amount
    labels = (labels + shift) % 10

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"images": images, "labels": labels}, output_path)
    print(f"Generated {count} samples with labels shifted by +{shift}")
    print(f"Saved to: {output_path}")


def generate_random_noise(count: int, output_path: Path):
    """Generate pure random noise with random labels."""
    images = torch.rand(count, 1, 28, 28)  # Pure noise
    labels = torch.randint(0, 10, (count,))  # Random labels

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"images": images, "labels": labels}, output_path)
    print(f"Generated {count} pure noise samples")
    print(f"Saved to: {output_path}")


def generate_constant_data(count: int, output_path: Path, constant_label: int = 5):
    """Generate data where all labels are the same (model learns nothing useful)."""
    generator = SyntheticDataGenerator(noise_level=0.1)
    images, _ = generator.generate_batch(count)
    labels = torch.full((count,), constant_label, dtype=torch.long)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"images": images, "labels": labels}, output_path)
    print(f"Generated {count} samples all labeled as {constant_label}")
    print(f"Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate bad training data")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Output path")
    parser.add_argument("--count", "-n", type=int, default=500, help="Number of samples")
    parser.add_argument(
        "--type", "-t",
        choices=["mislabeled", "noise", "constant", "inverted", "shifted"],
        default="inverted",
        help="Type of bad data (inverted recommended for testing rollback)"
    )
    parser.add_argument("--shuffle-ratio", type=float, default=1.0, help="Ratio of labels to shuffle (for mislabeled)")
    parser.add_argument("--shift", type=int, default=5, help="Label shift amount (for shifted)")
    args = parser.parse_args()

    if args.type == "mislabeled":
        generate_mislabeled_data(args.count, args.output, args.shuffle_ratio)
    elif args.type == "noise":
        generate_random_noise(args.count, args.output)
    elif args.type == "constant":
        generate_constant_data(args.count, args.output)
    elif args.type == "inverted":
        generate_inverted_labels(args.count, args.output)
    elif args.type == "shifted":
        generate_shifted_labels(args.count, args.output, args.shift)


if __name__ == "__main__":
    main()
