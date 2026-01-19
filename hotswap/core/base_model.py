"""Abstract base model class for hotswappable models."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


class BaseModel(ABC, nn.Module):
    """Abstract base class for all hotswappable models.

    Models must implement the forward method and can optionally
    override save/load for custom serialization.
    """

    def __init__(self):
        super().__init__()
        self._model_id: str | None = None
        self._version: int | None = None

    @property
    def model_id(self) -> str | None:
        return self._model_id

    @model_id.setter
    def model_id(self, value: str):
        self._model_id = value

    @property
    def version(self) -> int | None:
        return self._version

    @version.setter
    def version(self, value: int):
        self._version = value

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        pass

    def save(self, path: str | Path) -> None:
        """Save model state dict to path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str | Path, **kwargs: Any) -> "BaseModel":
        """Load model from state dict at path."""
        model = cls(**kwargs)
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Run inference with no_grad context."""
        self.eval()
        with torch.no_grad():
            return self.forward(x)

    def get_num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
