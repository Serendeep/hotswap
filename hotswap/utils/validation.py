"""Validation dataset for shadow mode comparison."""

import hashlib
from pathlib import Path

import torch

from .synthetic_data import SyntheticDataGenerator


class ValidationDataset:
    """Fixed validation dataset for consistent shadow mode comparison.

    Instead of comparing on random inputs (which gives meaningless results),
    we use a fixed validation set to measure actual model agreement on
    the same inputs.
    """

    def __init__(
        self,
        size: int = 100,
        cache_path: Path | None = None,
        seed: int = 42,
    ):
        self.size = size
        self.cache_path = cache_path
        self.seed = seed
        self._images: torch.Tensor | None = None
        self._labels: torch.Tensor | None = None

    def _generate(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate validation data with fixed seed."""
        import random

        # Save and restore random state for reproducibility
        state = random.getstate()
        torch.manual_seed(self.seed)
        random.seed(self.seed)

        generator = SyntheticDataGenerator(
            noise_level=0.1,
            intensity_variation=0.1,
            shift_range=1,
        )
        images, labels = generator.generate_batch(self.size)

        # Restore random state
        random.setstate(state)

        return images, labels

    def load(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Load or generate the validation dataset."""
        if self._images is not None:
            return self._images, self._labels

        # Try to load from cache
        if self.cache_path and self.cache_path.exists():
            data = torch.load(self.cache_path, weights_only=True)
            self._images = data["images"]
            self._labels = data["labels"]
            return self._images, self._labels

        # Generate new dataset
        self._images, self._labels = self._generate()

        # Save to cache
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"images": self._images, "labels": self._labels}, self.cache_path)

        return self._images, self._labels

    def get_batch(self, batch_size: int = 10) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a batch from the validation set."""
        images, labels = self.load()
        indices = torch.randperm(len(images))[:batch_size]
        return images[indices], labels[indices]

    def get_all(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the entire validation set."""
        return self.load()


def compare_models_on_validation(
    model1: torch.nn.Module,
    model2: torch.nn.Module,
    validation: ValidationDataset | None = None,
) -> dict:
    """Compare two models on a validation dataset.

    Returns:
        Dictionary with comparison metrics:
        - agreement_rate: How often both models predict the same class
        - model1_accuracy: Model 1 accuracy on validation labels
        - model2_accuracy: Model 2 accuracy on validation labels
        - accuracy_difference: Absolute difference in accuracy
    """
    if validation is None:
        validation = ValidationDataset()

    images, labels = validation.get_all()

    model1.eval()
    model2.eval()

    with torch.no_grad():
        out1 = model1(images)
        out2 = model2(images)

    pred1 = out1.argmax(dim=-1)
    pred2 = out2.argmax(dim=-1)

    agreement = (pred1 == pred2).float().mean().item()
    acc1 = (pred1 == labels).float().mean().item()
    acc2 = (pred2 == labels).float().mean().item()

    return {
        "agreement_rate": agreement,
        "model1_accuracy": acc1,
        "model2_accuracy": acc2,
        "accuracy_difference": abs(acc1 - acc2),
        "model2_better": acc2 > acc1,
    }
