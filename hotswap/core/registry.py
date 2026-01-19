"""Model registry with SQLite storage for versioning and metadata."""

import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Generator


class ModelStatus(str, Enum):
    """Model lifecycle status."""
    TRAINING = "training"
    READY = "ready"
    ACTIVE = "active"
    SHADOW = "shadow"
    ARCHIVED = "archived"


@dataclass
class ModelRecord:
    """Model metadata record."""
    id: str
    version: int
    status: ModelStatus
    checkpoint_path: str
    created_at: datetime
    updated_at: datetime
    metrics: dict | None = None
    training_data_path: str | None = None

    @classmethod
    def from_row(cls, row: tuple) -> "ModelRecord":
        """Create ModelRecord from database row."""
        import json
        return cls(
            id=row[0],
            version=row[1],
            status=ModelStatus(row[2]),
            checkpoint_path=row[3],
            created_at=datetime.fromisoformat(row[4]),
            updated_at=datetime.fromisoformat(row[5]),
            metrics=json.loads(row[6]) if row[6] else None,
            training_data_path=row[7],
        )


class ModelRegistry:
    """SQLite-backed model registry for version tracking."""

    def __init__(self, db_path: str | Path = "models/registry.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager for database connections."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS models (
                    id TEXT PRIMARY KEY,
                    version INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    checkpoint_path TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    metrics TEXT,
                    training_data_path TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_status ON models(status)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_version ON models(version)
            """)

    def register(
        self,
        checkpoint_path: str | Path,
        status: ModelStatus = ModelStatus.READY,
        metrics: dict | None = None,
        training_data_path: str | None = None,
    ) -> ModelRecord:
        """Register a new model version."""
        import json

        model_id = str(uuid.uuid4())
        version = self._get_next_version()
        now = datetime.now().isoformat()

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO models (id, version, status, checkpoint_path, created_at, updated_at, metrics, training_data_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    model_id,
                    version,
                    status.value,
                    str(checkpoint_path),
                    now,
                    now,
                    json.dumps(metrics) if metrics else None,
                    training_data_path,
                ),
            )

        return ModelRecord(
            id=model_id,
            version=version,
            status=status,
            checkpoint_path=str(checkpoint_path),
            created_at=datetime.fromisoformat(now),
            updated_at=datetime.fromisoformat(now),
            metrics=metrics,
            training_data_path=training_data_path,
        )

    def _get_next_version(self) -> int:
        """Get the next version number."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT MAX(version) FROM models")
            row = cursor.fetchone()
            return (row[0] or 0) + 1

    def get(self, model_id: str) -> ModelRecord | None:
        """Get model by ID."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM models WHERE id = ?", (model_id,)
            )
            row = cursor.fetchone()
            return ModelRecord.from_row(row) if row else None

    def get_active(self) -> ModelRecord | None:
        """Get the currently active model."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM models WHERE status = ? ORDER BY version DESC LIMIT 1",
                (ModelStatus.ACTIVE.value,),
            )
            row = cursor.fetchone()
            return ModelRecord.from_row(row) if row else None

    def get_shadow(self) -> ModelRecord | None:
        """Get the current shadow model."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM models WHERE status = ? ORDER BY version DESC LIMIT 1",
                (ModelStatus.SHADOW.value,),
            )
            row = cursor.fetchone()
            return ModelRecord.from_row(row) if row else None

    def get_latest_ready(self) -> ModelRecord | None:
        """Get the latest ready model."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM models WHERE status = ? ORDER BY version DESC LIMIT 1",
                (ModelStatus.READY.value,),
            )
            row = cursor.fetchone()
            return ModelRecord.from_row(row) if row else None

    def list_models(
        self, status: ModelStatus | None = None, limit: int = 100
    ) -> list[ModelRecord]:
        """List models, optionally filtered by status."""
        try:
            with self._get_connection() as conn:
                if status:
                    cursor = conn.execute(
                        "SELECT * FROM models WHERE status = ? ORDER BY version DESC LIMIT ?",
                        (status.value, limit),
                    )
                else:
                    cursor = conn.execute(
                        "SELECT * FROM models ORDER BY version DESC LIMIT ?", (limit,)
                    )
                return [ModelRecord.from_row(row) for row in cursor.fetchall()]
        except sqlite3.OperationalError:
            # Table might have been deleted, reinitialize
            self._init_db()
            return []

    def update_status(self, model_id: str, status: ModelStatus) -> None:
        """Update model status."""
        now = datetime.now().isoformat()
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE models SET status = ?, updated_at = ? WHERE id = ?",
                (status.value, now, model_id),
            )

    def set_active(self, model_id: str) -> None:
        """Set a model as active, archiving the previous active model."""
        with self._get_connection() as conn:
            # Archive current active
            conn.execute(
                "UPDATE models SET status = ?, updated_at = ? WHERE status = ?",
                (ModelStatus.ARCHIVED.value, datetime.now().isoformat(), ModelStatus.ACTIVE.value),
            )
            # Set new active
            conn.execute(
                "UPDATE models SET status = ?, updated_at = ? WHERE id = ?",
                (ModelStatus.ACTIVE.value, datetime.now().isoformat(), model_id),
            )

    def update_metrics(self, model_id: str, metrics: dict) -> None:
        """Update model metrics."""
        import json
        now = datetime.now().isoformat()
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE models SET metrics = ?, updated_at = ? WHERE id = ?",
                (json.dumps(metrics), now, model_id),
            )

    def delete(self, model_id: str) -> None:
        """Delete a model record (does not delete checkpoint file)."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM models WHERE id = ?", (model_id,))
