"""Pytest fixtures for hotswap tests."""

import tempfile
from pathlib import Path

import pytest
import torch

from hotswap.core.base_model import BaseModel
from hotswap.core.inference import InferenceEngine
from hotswap.core.registry import ModelRegistry
from hotswap.core.shadow import ShadowModeCollector
from hotswap.models.mnist_cnn import MNISTClassifier
from hotswap.utils.synthetic_data import generate_synthetic_batch


class SimpleModel(BaseModel):
    """Simple model for testing."""

    def __init__(self, output_value: int = 0):
        super().__init__()
        self.output_value = output_value
        self.linear = torch.nn.Linear(10, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Return fixed logits based on output_value
        batch_size = x.shape[0]
        logits = torch.zeros(batch_size, 10)
        logits[:, self.output_value] = 1.0
        return logits


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def registry(temp_dir):
    """Create a temporary model registry."""
    db_path = temp_dir / "test_registry.db"
    return ModelRegistry(db_path=db_path)


@pytest.fixture
def engine():
    """Create an inference engine."""
    return InferenceEngine()


@pytest.fixture
def simple_model():
    """Create a simple test model."""
    return SimpleModel(output_value=5)


@pytest.fixture
def mnist_model():
    """Create an MNIST classifier."""
    return MNISTClassifier()


@pytest.fixture
def shadow_collector():
    """Create a shadow mode collector."""
    return ShadowModeCollector(
        auto_promote_threshold=0.95,
        auto_rollback_threshold=0.85,
        min_samples_for_decision=10,
    )


@pytest.fixture
def sample_input():
    """Create sample input tensor."""
    return torch.randn(4, 1, 28, 28)


@pytest.fixture
def training_data(temp_dir):
    """Create synthetic training data."""
    data_path = temp_dir / "train_data.pt"
    images, labels = generate_synthetic_batch(count=100, output_path=data_path)
    return data_path


@pytest.fixture
def checkpoint_dir(temp_dir):
    """Create checkpoint directory."""
    checkpoint_dir = temp_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    return checkpoint_dir
