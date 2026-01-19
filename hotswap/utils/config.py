"""Configuration management."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Config:
    """Application configuration."""

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000

    # Paths
    data_dir: Path = field(default_factory=lambda: Path("data"))
    models_dir: Path = field(default_factory=lambda: Path("models"))
    checkpoint_dir: Path = field(default_factory=lambda: Path("models/checkpoints"))
    registry_db: Path = field(default_factory=lambda: Path("models/registry.db"))

    # Training defaults
    default_epochs: int = 10
    default_batch_size: int = 32
    default_lr: float = 0.001

    # Shadow mode settings
    shadow_auto_promote_threshold: float = 0.95
    shadow_auto_rollback_threshold: float = 0.85
    shadow_min_samples: int = 100

    # Watcher settings
    watcher_debounce_seconds: float = 2.0
    watcher_patterns: list[str] = field(default_factory=lambda: [".pt", ".pth", ".csv"])

    # Auto behaviors
    auto_shadow_on_ready: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "host": self.host,
            "port": self.port,
            "data_dir": str(self.data_dir),
            "models_dir": str(self.models_dir),
            "checkpoint_dir": str(self.checkpoint_dir),
            "registry_db": str(self.registry_db),
            "default_epochs": self.default_epochs,
            "default_batch_size": self.default_batch_size,
            "default_lr": self.default_lr,
            "shadow_auto_promote_threshold": self.shadow_auto_promote_threshold,
            "shadow_auto_rollback_threshold": self.shadow_auto_rollback_threshold,
            "shadow_min_samples": self.shadow_min_samples,
            "watcher_debounce_seconds": self.watcher_debounce_seconds,
            "watcher_patterns": self.watcher_patterns,
            "auto_shadow_on_ready": self.auto_shadow_on_ready,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        """Create Config from dictionary."""
        config = cls()
        for key, value in data.items():
            if hasattr(config, key):
                if key.endswith("_dir") or key == "registry_db":
                    setattr(config, key, Path(value))
                else:
                    setattr(config, key, value)
        return config


# Global config instance
_config: Config | None = None


def get_config() -> Config:
    """Get global configuration instance."""
    global _config
    if _config is None:
        _config = Config()
    return _config


def set_config(config: Config) -> None:
    """Set global configuration instance."""
    global _config
    _config = config
