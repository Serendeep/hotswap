"""API route definitions."""

import logging
from pathlib import Path
from typing import Any

import torch
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from .schemas import (
    PredictRequest,
    PredictResponse,
    ModelInfo,
    TrainRequest,
    TrainResponse,
    TrainingJobInfo,
    ShadowMetricsResponse,
    HealthResponse,
    SwapRequest,
    SwapResponse,
    ShadowStartRequest,
    StatusResponse,
)
from ..core.inference import InferenceEngine
from ..core.registry import ModelRegistry, ModelStatus
from ..workers.trainer import TrainingWorker
from ..workers.swap import SwapCoordinator
from ..workers.watcher import DataWatcher

logger = logging.getLogger(__name__)

router = APIRouter()

# These will be injected by the app factory
engine: InferenceEngine | None = None
registry: ModelRegistry | None = None
trainer: TrainingWorker | None = None
coordinator: SwapCoordinator | None = None
watcher: DataWatcher | None = None

# WebSocket connections for live updates
ws_connections: list[WebSocket] = []


async def broadcast_update(event: str, data: dict[str, Any]) -> None:
    """Broadcast update to all connected WebSocket clients."""
    message = {"event": event, "data": data}
    disconnected = []
    for ws in ws_connections:
        try:
            await ws.send_json(message)
        except Exception:
            disconnected.append(ws)
    for ws in disconnected:
        ws_connections.remove(ws)


# Health and Status endpoints

@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Health check endpoint."""
    status = engine.get_status() if engine else {}
    training_job = trainer.get_active_job() if trainer else None

    return HealthResponse(
        status="healthy",
        has_active_model=status.get("has_active_model", False),
        active_model_version=status.get("active_model_version"),
        shadow_enabled=status.get("shadow_enabled", False),
        prediction_count=status.get("prediction_count", 0),
        training_active=training_job is not None,
    )


@router.get("/status", response_model=StatusResponse)
def get_status() -> StatusResponse:
    """Get detailed system status."""
    engine_status = engine.get_status() if engine else {}
    swap_status = coordinator.get_status() if coordinator else {}
    training_job = trainer.get_active_job() if trainer else None

    return StatusResponse(
        engine=engine_status,
        swap_coordinator=swap_status,
        training=training_job.to_dict() if training_job else None,
        watcher_active=watcher.is_running if watcher else False,
    )


# Prediction endpoint

@router.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """Run inference on input data."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Inference engine not initialized")

    if engine.active_model is None:
        raise HTTPException(status_code=503, detail="No active model loaded")

    try:
        # Convert input to tensor
        data = torch.tensor(request.data, dtype=torch.float32)

        # Ensure proper shape (batch, channel, height, width)
        if data.dim() == 3:
            data = data.unsqueeze(1)  # Add channel dimension
        elif data.dim() == 2:
            data = data.unsqueeze(0).unsqueeze(0)  # Add batch and channel

        # Run inference
        result = engine.predict(data)

        # Get predictions
        output = result["prediction"]
        predictions = output.argmax(dim=-1).tolist()
        probabilities = torch.softmax(output, dim=-1).tolist()

        return PredictResponse(
            predictions=predictions,
            probabilities=probabilities,
            latency_ms=result["latency_ms"],
            model_version=engine.active_model.version,
            shadow_comparison=result.get("shadow_comparison"),
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Model management endpoints

@router.get("/models", response_model=list[ModelInfo])
def list_models(status: str | None = None, limit: int = 100) -> list[ModelInfo]:
    """List all registered models."""
    if registry is None:
        raise HTTPException(status_code=503, detail="Registry not initialized")

    model_status = ModelStatus(status) if status else None
    records = registry.list_models(status=model_status, limit=limit)

    return [
        ModelInfo(
            id=r.id,
            version=r.version,
            status=r.status.value,
            checkpoint_path=r.checkpoint_path,
            created_at=r.created_at,
            updated_at=r.updated_at,
            metrics=r.metrics,
            training_data_path=r.training_data_path,
        )
        for r in records
    ]


@router.get("/models/{model_id}", response_model=ModelInfo)
def get_model(model_id: str) -> ModelInfo:
    """Get model by ID."""
    if registry is None:
        raise HTTPException(status_code=503, detail="Registry not initialized")

    record = registry.get(model_id)
    if not record:
        raise HTTPException(status_code=404, detail="Model not found")

    return ModelInfo(
        id=record.id,
        version=record.version,
        status=record.status.value,
        checkpoint_path=record.checkpoint_path,
        created_at=record.created_at,
        updated_at=record.updated_at,
        metrics=record.metrics,
        training_data_path=record.training_data_path,
    )


@router.post("/models/{model_id}/activate", response_model=SwapResponse)
def activate_model(model_id: str) -> SwapResponse:
    """Force activate a specific model."""
    if coordinator is None:
        raise HTTPException(status_code=503, detail="Swap coordinator not initialized")

    success = coordinator.force_swap(model_id)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to activate model")

    active = registry.get_active() if registry else None
    return SwapResponse(
        success=True,
        message=f"Model {model_id} activated",
        active_model_id=active.id if active else None,
        active_model_version=active.version if active else None,
    )


# Training endpoints

@router.post("/train", response_model=TrainResponse)
def start_training(request: TrainRequest) -> TrainResponse:
    """Start a training job."""
    if trainer is None:
        raise HTTPException(status_code=503, detail="Training worker not initialized")

    # Verify data file exists
    data_path = Path(request.data_path)
    if not data_path.exists():
        raise HTTPException(status_code=400, detail=f"Data file not found: {request.data_path}")

    job_id = trainer.train(
        data_path=request.data_path,
        epochs=request.epochs,
        batch_size=request.batch_size,
        lr=request.learning_rate,
    )

    return TrainResponse(
        job_id=job_id,
        status="submitted",
        message=f"Training job {job_id} submitted",
    )


@router.get("/train/{job_id}", response_model=TrainingJobInfo)
def get_training_job(job_id: str) -> TrainingJobInfo:
    """Get training job status."""
    if trainer is None:
        raise HTTPException(status_code=503, detail="Training worker not initialized")

    job = trainer.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Training job not found")

    return TrainingJobInfo(
        id=job.id,
        data_path=job.data_path,
        status=job.status.value,
        progress=job.progress,
        current_epoch=job.current_epoch,
        total_epochs=job.total_epochs,
        current_loss=job.current_loss,
        model_id=job.model_id,
        error=job.error,
    )


@router.get("/train", response_model=list[TrainingJobInfo])
def list_training_jobs() -> list[TrainingJobInfo]:
    """List all training jobs."""
    if trainer is None:
        raise HTTPException(status_code=503, detail="Training worker not initialized")

    jobs = trainer.list_jobs()
    return [
        TrainingJobInfo(
            id=j.id,
            data_path=j.data_path,
            status=j.status.value,
            progress=j.progress,
            current_epoch=j.current_epoch,
            total_epochs=j.total_epochs,
            current_loss=j.current_loss,
            model_id=j.model_id,
            error=j.error,
        )
        for j in jobs
    ]


# Shadow mode endpoints

@router.post("/shadow/start", response_model=SwapResponse)
def start_shadow(request: ShadowStartRequest) -> SwapResponse:
    """Start shadow mode with specified model."""
    if coordinator is None:
        raise HTTPException(status_code=503, detail="Swap coordinator not initialized")

    success = coordinator.start_shadow(request.model_id)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to start shadow mode")

    return SwapResponse(
        success=True,
        message=f"Shadow mode started with model {request.model_id}",
    )


@router.get("/shadow/metrics", response_model=ShadowMetricsResponse)
def get_shadow_metrics() -> ShadowMetricsResponse:
    """Get current shadow mode metrics."""
    if coordinator is None or engine is None:
        raise HTTPException(status_code=503, detail="Services not initialized")

    status = coordinator.get_status()

    return ShadowMetricsResponse(
        enabled=engine.shadow_enabled,
        shadow_model_id=status.get("shadow_model_id"),
        shadow_model_version=status.get("shadow_model_version"),
        metrics=status.get("metrics"),
        decision=status.get("decision"),
    )


@router.post("/shadow/validate")
def run_validation() -> dict:
    """Run validation comparison between active and shadow models."""
    if coordinator is None:
        raise HTTPException(status_code=503, detail="Swap coordinator not initialized")

    if not engine.shadow_enabled:
        raise HTTPException(status_code=400, detail="Shadow mode not active")

    result = coordinator.run_validation()
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return {
        "success": True,
        "validation": result,
        "decision": coordinator._get_validation_decision(),
    }


@router.post("/shadow/promote", response_model=SwapResponse)
def promote_shadow() -> SwapResponse:
    """Promote shadow model to active."""
    if coordinator is None:
        raise HTTPException(status_code=503, detail="Swap coordinator not initialized")

    success = coordinator.promote()
    if not success:
        raise HTTPException(status_code=400, detail="Failed to promote shadow model")

    active = registry.get_active() if registry else None
    return SwapResponse(
        success=True,
        message="Shadow model promoted to active",
        active_model_id=active.id if active else None,
        active_model_version=active.version if active else None,
    )


@router.post("/shadow/cancel", response_model=SwapResponse)
def cancel_shadow() -> SwapResponse:
    """Cancel shadow mode."""
    if coordinator is None:
        raise HTTPException(status_code=503, detail="Swap coordinator not initialized")

    success = coordinator.cancel()
    if not success:
        raise HTTPException(status_code=400, detail="Failed to cancel shadow mode")

    return SwapResponse(
        success=True,
        message="Shadow mode cancelled",
    )


# WebSocket endpoint for live updates

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """WebSocket endpoint for real-time updates."""
    await websocket.accept()
    ws_connections.append(websocket)

    try:
        # Send initial status
        if engine and coordinator:
            await websocket.send_json({
                "event": "status",
                "data": {
                    "engine": engine.get_status(),
                    "swap": coordinator.get_status(),
                }
            })

        # Keep connection alive and handle messages
        while True:
            data = await websocket.receive_text()
            # Handle ping/pong or other messages if needed
            if data == "ping":
                await websocket.send_text("pong")

    except WebSocketDisconnect:
        if websocket in ws_connections:
            ws_connections.remove(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if websocket in ws_connections:
            ws_connections.remove(websocket)
