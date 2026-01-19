"""Synthetic MNIST-like data generator for testing."""

import random
from pathlib import Path

import torch


# Simple 7x7 digit templates (binary patterns)
DIGIT_TEMPLATES = {
    0: [
        [0, 1, 1, 1, 1, 1, 0],
        [1, 1, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 1, 1],
        [0, 1, 1, 1, 1, 1, 0],
    ],
    1: [
        [0, 0, 0, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 1],
    ],
    2: [
        [0, 1, 1, 1, 1, 1, 0],
        [1, 1, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 1, 1, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 1, 1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1],
    ],
    3: [
        [0, 1, 1, 1, 1, 1, 0],
        [1, 1, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 1, 1],
        [0, 0, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 1, 1],
        [0, 1, 1, 1, 1, 1, 0],
    ],
    4: [
        [0, 0, 0, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 1, 0],
        [0, 1, 1, 0, 1, 1, 0],
        [1, 1, 0, 0, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 1, 1, 0],
    ],
    5: [
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 1, 1],
        [0, 1, 1, 1, 1, 1, 0],
    ],
    6: [
        [0, 0, 1, 1, 1, 1, 0],
        [0, 1, 1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 0],
        [1, 1, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 1, 1],
        [0, 1, 1, 1, 1, 1, 0],
    ],
    7: [
        [1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0],
    ],
    8: [
        [0, 1, 1, 1, 1, 1, 0],
        [1, 1, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 1, 1],
        [0, 1, 1, 1, 1, 1, 0],
        [1, 1, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 1, 1],
        [0, 1, 1, 1, 1, 1, 0],
    ],
    9: [
        [0, 1, 1, 1, 1, 1, 0],
        [1, 1, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 1, 1],
        [0, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 1, 1, 0],
        [0, 1, 1, 1, 1, 0, 0],
    ],
}


class SyntheticDataGenerator:
    """Generates synthetic MNIST-like data for testing."""

    def __init__(
        self,
        noise_level: float = 0.1,
        intensity_variation: float = 0.2,
        shift_range: int = 2,
    ):
        self.noise_level = noise_level
        self.intensity_variation = intensity_variation
        self.shift_range = shift_range

    def generate_digit(self, digit: int) -> torch.Tensor:
        """Generate a single 28x28 digit image.

        Args:
            digit: Digit to generate (0-9)

        Returns:
            Tensor of shape (1, 28, 28) with values in [0, 1]
        """
        # Get template
        template = torch.tensor(DIGIT_TEMPLATES[digit], dtype=torch.float32)

        # Upscale 7x7 -> 28x28
        template = template.unsqueeze(0).unsqueeze(0)  # (1, 1, 7, 7)
        image = torch.nn.functional.interpolate(
            template, size=(28, 28), mode="nearest"
        ).squeeze()

        # Apply intensity variation
        intensity = 0.8 + random.uniform(-self.intensity_variation, self.intensity_variation)
        image = image * intensity

        # Apply random shift
        if self.shift_range > 0:
            shift_x = random.randint(-self.shift_range, self.shift_range)
            shift_y = random.randint(-self.shift_range, self.shift_range)
            image = torch.roll(image, shifts=(shift_y, shift_x), dims=(0, 1))

        # Add noise
        noise = torch.rand_like(image) * self.noise_level
        image = image + noise

        # Clamp to [0, 1]
        image = torch.clamp(image, 0, 1)

        return image.unsqueeze(0)  # (1, 28, 28)

    def generate_batch(self, count: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate a batch of synthetic digits.

        Args:
            count: Number of samples to generate

        Returns:
            Tuple of (images, labels) where:
            - images: Tensor of shape (count, 1, 28, 28)
            - labels: Tensor of shape (count,) with digit labels
        """
        images = []
        labels = []

        for _ in range(count):
            digit = random.randint(0, 9)
            image = self.generate_digit(digit)
            images.append(image)
            labels.append(digit)

        return torch.stack(images), torch.tensor(labels, dtype=torch.long)


def generate_synthetic_batch(
    count: int,
    output_path: str | Path | None = None,
    noise_level: float = 0.1,
    intensity_variation: float = 0.2,
    shift_range: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a batch of synthetic MNIST-like data.

    Args:
        count: Number of samples to generate
        output_path: Optional path to save the data as .pt file
        noise_level: Amount of random noise to add
        intensity_variation: Variation in digit intensity
        shift_range: Max pixels to shift the digit

    Returns:
        Tuple of (images, labels) tensors
    """
    generator = SyntheticDataGenerator(
        noise_level=noise_level,
        intensity_variation=intensity_variation,
        shift_range=shift_range,
    )

    images, labels = generator.generate_batch(count)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"images": images, "labels": labels}, output_path)

    return images, labels
