"""Pydantic schemas for API requests and responses."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Request for prediction endpoint."""
    data: list[list[list[float]]] = Field(
        ...,
        description="Input tensor as nested lists (batch, height, width) or (batch, channel, height, width)"
    )


class PredictResponse(BaseModel):
    """Response from prediction endpoint."""
    predictions: list[int] = Field(..., description="Predicted class indices")
    probabilities: list[list[float]] | None = Field(None, description="Class probabilities")
    latency_ms: float = Field(..., description="Inference latency in milliseconds")
    model_version: int | None = Field(None, description="Model version used")
    shadow_comparison: dict[str, Any] | None = Field(None, description="Shadow mode comparison if enabled")


class ModelInfo(BaseModel):
    """Model information."""
    id: str
    version: int
    status: str
    checkpoint_path: str
    created_at: datetime
    updated_at: datetime
    metrics: dict[str, Any] | None = None
    training_data_path: str | None = None


class TrainRequest(BaseModel):
    """Request for training endpoint."""
    data_path: str = Field(..., description="Path to training data file")
    epochs: int | None = Field(None, description="Number of training epochs")
    batch_size: int | None = Field(None, description="Training batch size")
    learning_rate: float | None = Field(None, description="Learning rate")


class TrainResponse(BaseModel):
    """Response from training endpoint."""
    job_id: str = Field(..., description="Training job ID")
    status: str = Field(..., description="Job status")
    message: str = Field(..., description="Status message")


class TrainingJobInfo(BaseModel):
    """Training job information."""
    id: str
    data_path: str
    status: str
    progress: float
    current_epoch: int
    total_epochs: int
    current_loss: float | None = None
    model_id: str | None = None
    error: str | None = None


class ShadowMetricsResponse(BaseModel):
    """Response with shadow mode metrics."""
    enabled: bool = Field(..., description="Whether shadow mode is enabled")
    shadow_model_id: str | None = Field(None, description="Shadow model ID")
    shadow_model_version: int | None = Field(None, description="Shadow model version")
    metrics: dict[str, Any] | None = Field(None, description="Collected metrics")
    decision: str | None = Field(None, description="Recommendation: promote, rollback, wait, manual")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    has_active_model: bool = Field(..., description="Whether an active model is loaded")
    active_model_version: int | None = Field(None, description="Active model version")
    shadow_enabled: bool = Field(..., description="Whether shadow mode is enabled")
    prediction_count: int = Field(..., description="Total predictions made")
    training_active: bool = Field(..., description="Whether training is in progress")


class SwapRequest(BaseModel):
    """Request for model swap."""
    model_id: str = Field(..., description="Model ID to swap to")


class SwapResponse(BaseModel):
    """Response from swap operation."""
    success: bool
    message: str
    active_model_id: str | None = None
    active_model_version: int | None = None


class ShadowStartRequest(BaseModel):
    """Request to start shadow mode."""
    model_id: str = Field(..., description="Model ID to run in shadow mode")


class StatusResponse(BaseModel):
    """Overall system status."""
    engine: dict[str, Any]
    swap_coordinator: dict[str, Any]
    training: dict[str, Any] | None = None
    watcher_active: bool = False
    config: dict[str, Any] | None = None
