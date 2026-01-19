"""Thread-safe inference engine with shadow mode support."""

import time
from typing import Any, Callable

import torch
from readerwriterlock import rwlock

from .base_model import BaseModel
from .shadow import ShadowModeCollector


class InferenceEngine:
    """Thread-safe inference engine supporting hot-swapping and shadow mode.

    Uses a reader-writer lock to allow concurrent reads while ensuring
    atomic model swaps. Shadow mode runs both active and shadow models
    in parallel, returning active results while collecting comparison metrics.
    """

    def __init__(self):
        self._rw_lock = rwlock.RWLockFair()
        self._active_model: BaseModel | None = None
        self._shadow_model: BaseModel | None = None
        self._shadow_collector: ShadowModeCollector | None = None
        self._shadow_enabled = False
        self._prediction_count = 0
        self._on_prediction_callbacks: list[Callable[[dict], None]] = []

    @property
    def active_model(self) -> BaseModel | None:
        """Get the active model (read-locked)."""
        with self._rw_lock.gen_rlock():
            return self._active_model

    @property
    def shadow_model(self) -> BaseModel | None:
        """Get the shadow model (read-locked)."""
        with self._rw_lock.gen_rlock():
            return self._shadow_model

    @property
    def shadow_enabled(self) -> bool:
        """Check if shadow mode is enabled."""
        with self._rw_lock.gen_rlock():
            return self._shadow_enabled

    @property
    def shadow_collector(self) -> ShadowModeCollector | None:
        """Get the shadow metrics collector."""
        with self._rw_lock.gen_rlock():
            return self._shadow_collector

    @property
    def prediction_count(self) -> int:
        """Get total prediction count."""
        return self._prediction_count

    def load_model(self, model: BaseModel) -> None:
        """Load a model as the active model (write-locked)."""
        with self._rw_lock.gen_wlock():
            self._active_model = model
            self._active_model.eval()

    def predict(self, x: torch.Tensor) -> dict[str, Any]:
        """Run inference, optionally with shadow mode comparison.

        Returns:
            Dictionary with 'prediction', 'latency_ms', and optionally
            'shadow_comparison' if shadow mode is enabled.
        """
        with self._rw_lock.gen_rlock():
            if self._active_model is None:
                raise RuntimeError("No active model loaded")

            result: dict[str, Any] = {}

            # Run active model
            start_time = time.perf_counter()
            active_output = self._active_model.predict(x)
            active_latency = (time.perf_counter() - start_time) * 1000

            result["prediction"] = active_output
            result["latency_ms"] = active_latency

            # Run shadow model if enabled
            if self._shadow_enabled and self._shadow_model is not None:
                start_time = time.perf_counter()
                shadow_output = self._shadow_model.predict(x)
                shadow_latency = (time.perf_counter() - start_time) * 1000

                # Record comparison
                if self._shadow_collector is not None:
                    self._shadow_collector.record(
                        active_output=active_output,
                        shadow_output=shadow_output,
                        active_latency_ms=active_latency,
                        shadow_latency_ms=shadow_latency,
                    )

                result["shadow_comparison"] = {
                    "shadow_latency_ms": shadow_latency,
                    "agreement": torch.equal(
                        active_output.argmax(dim=-1), shadow_output.argmax(dim=-1)
                    ),
                }

            self._prediction_count += 1

            # Notify callbacks
            for callback in self._on_prediction_callbacks:
                try:
                    callback(result)
                except Exception:
                    pass  # Don't let callback errors affect inference

            return result

    def enable_shadow(
        self,
        shadow_model: BaseModel,
        collector: ShadowModeCollector | None = None,
    ) -> None:
        """Enable shadow mode with the given model (write-locked)."""
        with self._rw_lock.gen_wlock():
            self._shadow_model = shadow_model
            self._shadow_model.eval()
            self._shadow_collector = collector or ShadowModeCollector()
            self._shadow_enabled = True

    def disable_shadow(self) -> ShadowModeCollector | None:
        """Disable shadow mode and return the collector (write-locked)."""
        with self._rw_lock.gen_wlock():
            collector = self._shadow_collector
            self._shadow_model = None
            self._shadow_collector = None
            self._shadow_enabled = False
            return collector

    def swap_model(self, new_model: BaseModel) -> None:
        """Atomically swap the active model (write-locked)."""
        with self._rw_lock.gen_wlock():
            new_model.eval()
            self._active_model = new_model

    def promote_shadow(self) -> bool:
        """Promote shadow model to active (write-locked).

        Returns:
            True if promotion succeeded, False if no shadow model exists.
        """
        with self._rw_lock.gen_wlock():
            if self._shadow_model is None:
                return False

            self._active_model = self._shadow_model
            self._shadow_model = None
            self._shadow_collector = None
            self._shadow_enabled = False
            return True

    def add_prediction_callback(self, callback: Callable[[dict], None]) -> None:
        """Add a callback to be called after each prediction."""
        self._on_prediction_callbacks.append(callback)

    def remove_prediction_callback(self, callback: Callable[[dict], None]) -> None:
        """Remove a prediction callback."""
        if callback in self._on_prediction_callbacks:
            self._on_prediction_callbacks.remove(callback)

    def get_status(self) -> dict[str, Any]:
        """Get engine status."""
        with self._rw_lock.gen_rlock():
            status = {
                "has_active_model": self._active_model is not None,
                "shadow_enabled": self._shadow_enabled,
                "has_shadow_model": self._shadow_model is not None,
                "prediction_count": self._prediction_count,
            }

            if self._active_model is not None:
                status["active_model_id"] = self._active_model.model_id
                status["active_model_version"] = self._active_model.version

            if self._shadow_model is not None:
                status["shadow_model_id"] = self._shadow_model.model_id
                status["shadow_model_version"] = self._shadow_model.version

            if self._shadow_collector is not None:
                status["shadow_metrics"] = self._shadow_collector.get_metrics()

            return status
