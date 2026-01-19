"""Background training worker."""

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Type

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from ..core.base_model import BaseModel
from ..core.registry import ModelRegistry, ModelStatus

logger = logging.getLogger(__name__)


class TrainingStatus(str, Enum):
    """Training job status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TrainingJob:
    """Training job metadata."""
    id: str
    data_path: str
    status: TrainingStatus = TrainingStatus.PENDING
    progress: float = 0.0
    current_epoch: int = 0
    total_epochs: int = 0
    current_loss: float | None = None
    model_id: str | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    completed_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "data_path": self.data_path,
            "status": self.status.value,
            "progress": round(self.progress, 2),
            "current_epoch": self.current_epoch,
            "total_epochs": self.total_epochs,
            "current_loss": round(self.current_loss, 4) if self.current_loss else None,
            "model_id": self.model_id,
            "error": self.error,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
        }


class TrainingWorker:
    """Background worker for training models.

    Runs training jobs in a thread pool and provides progress updates.
    Automatically registers trained models with the registry.
    """

    def __init__(
        self,
        model_class: Type[BaseModel],
        registry: ModelRegistry,
        checkpoint_dir: str | Path = "models/checkpoints",
        max_workers: int = 1,
        default_epochs: int = 5,
        default_batch_size: int = 32,
        default_lr: float = 0.001,
    ):
        self.model_class = model_class
        self.registry = registry
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.default_epochs = default_epochs
        self.default_batch_size = default_batch_size
        self.default_lr = default_lr

        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._jobs: dict[str, TrainingJob] = {}
        self._futures: dict[str, Future] = {}
        self._lock = threading.Lock()
        self._on_complete_callbacks: list[Callable[[TrainingJob], None]] = []
        self._on_progress_callbacks: list[Callable[[TrainingJob], None]] = []

    def train(
        self,
        data_path: str | Path,
        epochs: int | None = None,
        batch_size: int | None = None,
        lr: float | None = None,
    ) -> str:
        """Submit a training job.

        Returns:
            Job ID for tracking progress.
        """
        import uuid

        job_id = str(uuid.uuid4())[:8]
        job = TrainingJob(
            id=job_id,
            data_path=str(data_path),
            total_epochs=epochs or self.default_epochs,
        )

        with self._lock:
            self._jobs[job_id] = job

        future = self._executor.submit(
            self._train_worker,
            job,
            epochs or self.default_epochs,
            batch_size or self.default_batch_size,
            lr or self.default_lr,
        )
        self._futures[job_id] = future

        logger.info(f"Submitted training job {job_id} for {data_path}")
        return job_id

    def _train_worker(
        self,
        job: TrainingJob,
        epochs: int,
        batch_size: int,
        lr: float,
    ) -> None:
        """Worker function for training a model."""
        try:
            job.status = TrainingStatus.RUNNING
            self._notify_progress(job)

            # Load data
            data = torch.load(job.data_path, weights_only=True)
            if isinstance(data, dict):
                x_data = data["images"]
                y_data = data["labels"]
            elif isinstance(data, tuple):
                x_data, y_data = data
            else:
                raise ValueError("Expected dict with 'images'/'labels' or tuple")

            dataset = TensorDataset(x_data, y_data)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            # Create model
            model = self.model_class()
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=lr)

            # Register as training
            record = self.registry.register(
                checkpoint_path="pending",
                status=ModelStatus.TRAINING,
                training_data_path=job.data_path,
            )
            job.model_id = record.id

            # Training loop
            model.train()
            for epoch in range(epochs):
                job.current_epoch = epoch + 1
                epoch_loss = 0.0
                num_batches = 0

                for batch_x, batch_y in dataloader:
                    optimizer.zero_grad()
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                    num_batches += 1

                job.current_loss = epoch_loss / num_batches
                job.progress = (epoch + 1) / epochs
                self._notify_progress(job)

                logger.debug(f"Job {job.id} epoch {epoch+1}/{epochs} loss={job.current_loss:.4f}")

            # Save checkpoint
            checkpoint_path = self.checkpoint_dir / f"model_v{record.version}.pt"
            model.save(checkpoint_path)

            # Update registry
            model.model_id = record.id
            model.version = record.version

            # Update checkpoint path and status
            with self.registry._get_connection() as conn:
                conn.execute(
                    "UPDATE models SET checkpoint_path = ?, status = ?, updated_at = ? WHERE id = ?",
                    (str(checkpoint_path), ModelStatus.READY.value,
                     __import__('datetime').datetime.now().isoformat(), record.id),
                )

            self.registry.update_metrics(
                record.id,
                {"final_loss": job.current_loss, "epochs": epochs},
            )

            job.status = TrainingStatus.COMPLETED
            job.completed_at = time.time()
            logger.info(f"Training job {job.id} completed, model {record.id}")

        except Exception as e:
            logger.error(f"Training job {job.id} failed: {e}")
            job.status = TrainingStatus.FAILED
            job.error = str(e)
            job.completed_at = time.time()

        finally:
            self._notify_complete(job)

    def get_job(self, job_id: str) -> TrainingJob | None:
        """Get job by ID."""
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(self) -> list[TrainingJob]:
        """List all jobs."""
        with self._lock:
            return list(self._jobs.values())

    def get_active_job(self) -> TrainingJob | None:
        """Get currently running job."""
        with self._lock:
            for job in self._jobs.values():
                if job.status == TrainingStatus.RUNNING:
                    return job
            return None

    def cancel(self, job_id: str) -> bool:
        """Cancel a pending job."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job and job.status == TrainingStatus.PENDING:
                future = self._futures.get(job_id)
                if future and future.cancel():
                    job.status = TrainingStatus.CANCELLED
                    return True
            return False

    def add_on_complete_callback(self, callback: Callable[[TrainingJob], None]) -> None:
        """Add callback for when training completes."""
        self._on_complete_callbacks.append(callback)

    def add_on_progress_callback(self, callback: Callable[[TrainingJob], None]) -> None:
        """Add callback for training progress updates."""
        self._on_progress_callbacks.append(callback)

    def _notify_complete(self, job: TrainingJob) -> None:
        """Notify completion callbacks."""
        for callback in self._on_complete_callbacks:
            try:
                callback(job)
            except Exception as e:
                logger.error(f"Completion callback error: {e}")

    def _notify_progress(self, job: TrainingJob) -> None:
        """Notify progress callbacks."""
        for callback in self._on_progress_callbacks:
            try:
                callback(job)
            except Exception as e:
                logger.error(f"Progress callback error: {e}")

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the worker."""
        self._executor.shutdown(wait=wait)
