"""Simple MNIST CNN classifier for demonstration."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.base_model import BaseModel


class MNISTClassifier(BaseModel):
    """Simple CNN classifier for MNIST-like 28x28 grayscale images.

    Architecture:
    - Conv2d(1, 32, 3) -> ReLU -> MaxPool2d(2)
    - Conv2d(32, 64, 3) -> ReLU -> MaxPool2d(2)
    - Flatten -> Linear(64*7*7, 128) -> ReLU -> Linear(128, 10)
    """

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, 1, 28, 28)

        Returns:
            Logits tensor of shape (batch, 10)
        """
        # Ensure input is 4D
        if x.dim() == 3:
            x = x.unsqueeze(1)  # Add channel dimension

        # Conv block 1: 28x28 -> 14x14
        x = self.pool(F.relu(self.conv1(x)))

        # Conv block 2: 14x14 -> 7x7
        x = self.pool(F.relu(self.conv2(x)))

        # Flatten and FC layers
        x = x.view(-1, 64 * 7 * 7)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def predict_class(self, x: torch.Tensor) -> torch.Tensor:
        """Predict class labels.

        Args:
            x: Input tensor of shape (batch, 1, 28, 28)

        Returns:
            Class indices tensor of shape (batch,)
        """
        logits = self.predict(x)
        return logits.argmax(dim=-1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Predict class probabilities.

        Args:
            x: Input tensor of shape (batch, 1, 28, 28)

        Returns:
            Probability tensor of shape (batch, 10)
        """
        logits = self.predict(x)
        return F.softmax(logits, dim=-1)
