"""Shadow mode metrics collector for model comparison."""

import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

import torch


@dataclass
class ShadowMetrics:
    """Aggregated shadow mode metrics."""
    total_samples: int = 0
    agreements: int = 0
    disagreements: int = 0
    total_active_latency_ms: float = 0.0
    total_shadow_latency_ms: float = 0.0
    started_at: float = field(default_factory=time.time)
    errors: int = 0

    @property
    def agreement_rate(self) -> float:
        """Calculate agreement rate between models."""
        if self.total_samples == 0:
            return 0.0
        return self.agreements / self.total_samples

    @property
    def avg_active_latency_ms(self) -> float:
        """Average active model latency."""
        if self.total_samples == 0:
            return 0.0
        return self.total_active_latency_ms / self.total_samples

    @property
    def avg_shadow_latency_ms(self) -> float:
        """Average shadow model latency."""
        if self.total_samples == 0:
            return 0.0
        return self.total_shadow_latency_ms / self.total_samples

    @property
    def latency_difference_ms(self) -> float:
        """Difference in average latency (shadow - active)."""
        return self.avg_shadow_latency_ms - self.avg_active_latency_ms

    @property
    def elapsed_seconds(self) -> float:
        """Time since shadow mode started."""
        return time.time() - self.started_at

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_samples": self.total_samples,
            "agreements": self.agreements,
            "disagreements": self.disagreements,
            "agreement_rate": round(self.agreement_rate, 4),
            "avg_active_latency_ms": round(self.avg_active_latency_ms, 2),
            "avg_shadow_latency_ms": round(self.avg_shadow_latency_ms, 2),
            "latency_difference_ms": round(self.latency_difference_ms, 2),
            "elapsed_seconds": round(self.elapsed_seconds, 1),
            "errors": self.errors,
        }


class ShadowModeCollector:
    """Thread-safe collector for shadow mode comparison metrics.

    Collects metrics comparing active and shadow model outputs
    to determine if the shadow model should be promoted.
    """

    def __init__(
        self,
        auto_promote_threshold: float = 0.95,
        auto_rollback_threshold: float = 0.85,
        min_samples_for_decision: int = 100,
    ):
        self._lock = Lock()
        self._metrics = ShadowMetrics()
        self.auto_promote_threshold = auto_promote_threshold
        self.auto_rollback_threshold = auto_rollback_threshold
        self.min_samples_for_decision = min_samples_for_decision

    def record(
        self,
        active_output: torch.Tensor,
        shadow_output: torch.Tensor,
        active_latency_ms: float,
        shadow_latency_ms: float,
    ) -> None:
        """Record a comparison between active and shadow outputs."""
        try:
            # Compare predictions (argmax for classification)
            active_pred = active_output.argmax(dim=-1)
            shadow_pred = shadow_output.argmax(dim=-1)
            agrees = torch.equal(active_pred, shadow_pred)

            with self._lock:
                self._metrics.total_samples += 1
                if agrees:
                    self._metrics.agreements += 1
                else:
                    self._metrics.disagreements += 1
                self._metrics.total_active_latency_ms += active_latency_ms
                self._metrics.total_shadow_latency_ms += shadow_latency_ms
        except Exception:
            with self._lock:
                self._metrics.errors += 1

    def get_metrics(self) -> dict[str, Any]:
        """Get current metrics as a dictionary."""
        with self._lock:
            return self._metrics.to_dict()

    def get_decision(self) -> str:
        """Get recommendation based on current metrics.

        Returns:
            'promote': Shadow model should be promoted
            'rollback': Shadow model should be discarded
            'wait': Need more samples
            'manual': Requires manual decision (between thresholds)
        """
        with self._lock:
            if self._metrics.total_samples < self.min_samples_for_decision:
                return "wait"

            rate = self._metrics.agreement_rate
            if rate >= self.auto_promote_threshold:
                return "promote"
            elif rate < self.auto_rollback_threshold:
                return "rollback"
            else:
                return "manual"

    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._metrics = ShadowMetrics()

    @property
    def total_samples(self) -> int:
        """Get total sample count."""
        with self._lock:
            return self._metrics.total_samples

    @property
    def agreement_rate(self) -> float:
        """Get current agreement rate."""
        with self._lock:
            return self._metrics.agreement_rate

    def should_auto_promote(self) -> bool:
        """Check if shadow should be auto-promoted."""
        return self.get_decision() == "promote"

    def should_auto_rollback(self) -> bool:
        """Check if shadow should be auto-rolled back."""
        return self.get_decision() == "rollback"
