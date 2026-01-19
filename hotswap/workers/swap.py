"""Swap coordinator for orchestrating shadow mode and model promotion."""

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Type

from ..core.base_model import BaseModel
from ..core.inference import InferenceEngine
from ..core.registry import ModelRegistry, ModelStatus
from ..core.shadow import ShadowModeCollector
from ..utils.validation import ValidationDataset, compare_models_on_validation

logger = logging.getLogger(__name__)


class SwapState(str, Enum):
    """Swap coordinator state."""
    IDLE = "idle"
    SHADOW_MODE = "shadow_mode"
    PROMOTING = "promoting"


@dataclass
class SwapStatus:
    """Current swap coordinator status."""
    state: SwapState = SwapState.IDLE
    shadow_model_id: str | None = None
    shadow_model_version: int | None = None
    started_at: float | None = None
    metrics: dict[str, Any] | None = None
    decision: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "shadow_model_id": self.shadow_model_id,
            "shadow_model_version": self.shadow_model_version,
            "started_at": self.started_at,
            "elapsed_seconds": round(time.time() - self.started_at, 1) if self.started_at else None,
            "metrics": self.metrics,
            "decision": self.decision,
        }


class SwapCoordinator:
    """Orchestrates shadow mode validation and model promotion.

    Coordinates the flow:
    1. New model becomes READY
    2. Start shadow mode
    3. Collect comparison metrics
    4. Auto-promote or rollback based on thresholds

    Can also be controlled manually via API.
    """

    def __init__(
        self,
        engine: InferenceEngine,
        registry: ModelRegistry,
        model_class: Type[BaseModel],
        auto_shadow_on_ready: bool = True,
        auto_promote_threshold: float = 0.95,
        auto_rollback_threshold: float = 0.85,
        min_samples: int = 100,
        check_interval: float = 1.0,
        use_validation_mode: bool = True,
        validation_size: int = 100,
    ):
        self.engine = engine
        self.registry = registry
        self.model_class = model_class
        self.auto_shadow_on_ready = auto_shadow_on_ready
        self.auto_promote_threshold = auto_promote_threshold
        self.auto_rollback_threshold = auto_rollback_threshold
        self.min_samples = min_samples
        self.check_interval = check_interval
        self.use_validation_mode = use_validation_mode

        self._status = SwapStatus()
        self._lock = threading.Lock()
        self._monitor_thread: threading.Thread | None = None
        self._stop_monitor = threading.Event()
        self._on_swap_callbacks: list[Callable[[str, str], None]] = []
        self._validation = ValidationDataset(size=validation_size)
        self._validation_result: dict[str, Any] | None = None

    def get_status(self) -> dict[str, Any]:
        """Get current swap status."""
        with self._lock:
            if self._status.state == SwapState.SHADOW_MODE:
                if self._validation_result:
                    self._status.metrics = self._validation_result
                    self._status.decision = self._get_validation_decision()
                else:
                    collector = self.engine.shadow_collector
                    if collector:
                        self._status.metrics = collector.get_metrics()
                        self._status.decision = collector.get_decision()
            return self._status.to_dict()

    def run_validation(self) -> dict[str, Any]:
        """Run validation comparison between active and shadow models.

        Returns:
            Validation metrics including agreement rate and accuracy comparison.
        """
        active_model = self.engine.active_model
        shadow_model = self.engine.shadow_model

        if not active_model or not shadow_model:
            return {"error": "Both active and shadow models required"}

        result = compare_models_on_validation(
            active_model, shadow_model, self._validation
        )

        self._validation_result = result
        logger.info(
            f"Validation: agreement={result['agreement_rate']:.1%}, "
            f"active_acc={result['model1_accuracy']:.1%}, "
            f"shadow_acc={result['model2_accuracy']:.1%}, "
            f"shadow_better={result['model2_better']}"
        )
        return result

    def _get_validation_decision(self) -> str:
        """Get decision based on validation results."""
        if not self._validation_result:
            return "wait"

        result = self._validation_result
        shadow_acc = result["model2_accuracy"]
        active_acc = result["model1_accuracy"]
        agreement = result["agreement_rate"]

        # Log for debugging
        logger.info(
            f"Decision check: shadow_acc={shadow_acc:.1%}, active_acc={active_acc:.1%}, "
            f"agreement={agreement:.1%}"
        )

        # If shadow model is significantly worse, rollback
        if shadow_acc < active_acc - 0.10:
            return "rollback"

        # If shadow model is at least as good, promote
        if shadow_acc >= active_acc - 0.05:
            return "promote"

        # In between - needs manual decision
        return "manual"

    def start_shadow(self, model_id: str) -> bool:
        """Start shadow mode with the specified model.

        Returns:
            True if shadow mode started successfully.
        """
        with self._lock:
            if self._status.state != SwapState.IDLE:
                logger.warning("Cannot start shadow: not in IDLE state")
                return False

            # Load the model
            record = self.registry.get(model_id)
            if not record:
                logger.error(f"Model {model_id} not found")
                return False

            try:
                model = self.model_class.load(record.checkpoint_path)
                model.model_id = record.id
                model.version = record.version
            except Exception as e:
                logger.error(f"Failed to load model {model_id}: {e}")
                return False

            # Create collector and enable shadow mode
            collector = ShadowModeCollector(
                auto_promote_threshold=self.auto_promote_threshold,
                auto_rollback_threshold=self.auto_rollback_threshold,
                min_samples_for_decision=self.min_samples,
            )

            self.engine.enable_shadow(model, collector)

            # Update registry status
            self.registry.update_status(model_id, ModelStatus.SHADOW)

            # Reset validation result
            self._validation_result = None

            # Update coordinator status
            self._status = SwapStatus(
                state=SwapState.SHADOW_MODE,
                shadow_model_id=record.id,
                shadow_model_version=record.version,
                started_at=time.time(),
            )

            # Start monitor thread
            self._stop_monitor.clear()
            self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._monitor_thread.start()

            logger.info(f"Started shadow mode with model {model_id} v{record.version}")
            return True

    def _monitor_loop(self) -> None:
        """Background thread to check metrics and auto-promote/rollback."""
        # If using validation mode, run validation immediately
        if self.use_validation_mode:
            time.sleep(0.5)  # Brief delay to ensure models are loaded
            self.run_validation()
            decision = self._get_validation_decision()

            if decision == "promote":
                logger.info("Validation passed - auto-promoting shadow model")
                self.promote()
                return
            elif decision == "rollback":
                logger.info("Validation failed - auto-rolling back shadow model")
                self.cancel()
                return
            else:
                logger.info("Validation inconclusive - waiting for manual decision")
                return

        # Legacy mode: use per-prediction agreement
        while not self._stop_monitor.is_set():
            time.sleep(self.check_interval)

            collector = self.engine.shadow_collector
            if not collector:
                continue

            decision = collector.get_decision()

            if decision == "promote":
                logger.info("Auto-promoting shadow model")
                self.promote()
                break
            elif decision == "rollback":
                logger.info("Auto-rolling back shadow model")
                self.cancel()
                break

    def promote(self) -> bool:
        """Promote shadow model to active.

        Returns:
            True if promotion succeeded.
        """
        with self._lock:
            if self._status.state != SwapState.SHADOW_MODE:
                logger.warning("Cannot promote: not in SHADOW_MODE state")
                return False

            self._status.state = SwapState.PROMOTING
            self._stop_monitor.set()

            shadow_id = self._status.shadow_model_id
            if not shadow_id:
                return False

            # Promote in engine
            if not self.engine.promote_shadow():
                return False

            # Update registry
            self.registry.set_active(shadow_id)

            # Notify callbacks
            old_active = self.registry.get_active()
            old_id = old_active.id if old_active else None
            for callback in self._on_swap_callbacks:
                try:
                    callback(old_id or "", shadow_id)
                except Exception as e:
                    logger.error(f"Swap callback error: {e}")

            logger.info(f"Promoted model {shadow_id} to active")

            # Reset status
            self._status = SwapStatus()
            return True

    def cancel(self) -> bool:
        """Cancel shadow mode without promoting.

        Returns:
            True if cancellation succeeded.
        """
        with self._lock:
            if self._status.state != SwapState.SHADOW_MODE:
                logger.warning("Cannot cancel: not in SHADOW_MODE state")
                return False

            self._stop_monitor.set()

            shadow_id = self._status.shadow_model_id

            # Disable shadow mode
            self.engine.disable_shadow()

            # Archive the shadow model
            if shadow_id:
                self.registry.update_status(shadow_id, ModelStatus.ARCHIVED)

            logger.info(f"Cancelled shadow mode for model {shadow_id}")

            # Reset status
            self._status = SwapStatus()
            return True

    def force_swap(self, model_id: str) -> bool:
        """Force swap to a specific model, bypassing shadow mode.

        Returns:
            True if swap succeeded.
        """
        with self._lock:
            # Cancel any ongoing shadow mode
            if self._status.state == SwapState.SHADOW_MODE:
                self._stop_monitor.set()
                self.engine.disable_shadow()

            # Load the model
            record = self.registry.get(model_id)
            if not record:
                logger.error(f"Model {model_id} not found")
                return False

            try:
                model = self.model_class.load(record.checkpoint_path)
                model.model_id = record.id
                model.version = record.version
            except Exception as e:
                logger.error(f"Failed to load model {model_id}: {e}")
                return False

            # Swap the model
            self.engine.swap_model(model)

            # Update registry
            self.registry.set_active(model_id)

            logger.info(f"Force swapped to model {model_id} v{record.version}")

            # Reset status
            self._status = SwapStatus()
            return True

    def on_model_ready(self, model_id: str) -> None:
        """Called when a new model becomes READY.

        If auto_shadow_on_ready is True and there's an active model,
        starts shadow mode with the new model.
        """
        if not self.auto_shadow_on_ready:
            return

        with self._lock:
            if self._status.state != SwapState.IDLE:
                logger.debug("Already in shadow mode, skipping auto-shadow")
                return

        # Check if we have an active model
        active = self.registry.get_active()
        if active:
            logger.info(f"Auto-starting shadow mode for new model {model_id}")
            self.start_shadow(model_id)
        else:
            # No active model, just activate this one
            logger.info(f"No active model, activating {model_id} directly")
            self.force_swap(model_id)

    def add_on_swap_callback(self, callback: Callable[[str, str], None]) -> None:
        """Add callback for when a swap occurs (old_id, new_id)."""
        self._on_swap_callbacks.append(callback)

    def shutdown(self) -> None:
        """Shutdown the coordinator."""
        self._stop_monitor.set()
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5.0)
