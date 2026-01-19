"""FastAPI application factory."""

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from . import routes
from ..core.inference import InferenceEngine
from ..core.registry import ModelRegistry, ModelStatus
from ..models.mnist_cnn import MNISTClassifier
from ..workers.trainer import TrainingWorker, TrainingStatus
from ..workers.swap import SwapCoordinator
from ..workers.watcher import DataWatcher
from ..utils.config import Config, get_config

logger = logging.getLogger(__name__)


def create_app(
    config: Config | None = None,
    enable_watcher: bool = True,
    watch_path: str | Path | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        config: Application configuration. Uses default if not provided.
        enable_watcher: Whether to enable the file watcher.
        watch_path: Path to watch for new data files.

    Returns:
        Configured FastAPI application.
    """
    config = config or get_config()
    watch_path = Path(watch_path) if watch_path else config.data_dir

    # Initialize components
    engine = InferenceEngine()
    registry = ModelRegistry(db_path=config.registry_db)
    trainer = TrainingWorker(
        model_class=MNISTClassifier,
        registry=registry,
        checkpoint_dir=config.checkpoint_dir,
        default_epochs=config.default_epochs,
        default_batch_size=config.default_batch_size,
        default_lr=config.default_lr,
    )
    coordinator = SwapCoordinator(
        engine=engine,
        registry=registry,
        model_class=MNISTClassifier,
        auto_shadow_on_ready=config.auto_shadow_on_ready,
        auto_promote_threshold=config.shadow_auto_promote_threshold,
        auto_rollback_threshold=config.shadow_auto_rollback_threshold,
        min_samples=config.shadow_min_samples,
    )

    # Set up watcher callback
    def on_new_data(data_path: Path) -> None:
        """Callback when new data is detected."""
        logger.info(f"New data detected: {data_path}")
        trainer.train(data_path=data_path)

    watcher = DataWatcher(
        watch_path=watch_path,
        on_new_data=on_new_data,
        patterns=config.watcher_patterns,
        debounce_seconds=config.watcher_debounce_seconds,
    ) if enable_watcher else None

    # Event loop reference for thread-safe callbacks
    _loop: asyncio.AbstractEventLoop | None = None

    async def _safe_broadcast(event: str, data: dict) -> None:
        """Safely broadcast without raising exceptions."""
        try:
            await routes.broadcast_update(event, data)
        except Exception as e:
            logger.debug(f"Broadcast failed: {e}")

    def _schedule_broadcast(event: str, data: dict) -> None:
        """Schedule a broadcast from a background thread."""
        if _loop is not None and _loop.is_running():
            asyncio.run_coroutine_threadsafe(_safe_broadcast(event, data), _loop)

    # Set up training completion callback
    def on_training_complete(job) -> None:
        """Callback when training completes."""
        if job.status == TrainingStatus.COMPLETED and job.model_id:
            logger.info(f"Training completed, triggering shadow mode for {job.model_id}")
            coordinator.on_model_ready(job.model_id)
            _schedule_broadcast("training_complete", job.to_dict())

    trainer.add_on_complete_callback(on_training_complete)

    # Set up progress callback for WebSocket updates
    def on_training_progress(job) -> None:
        """Callback for training progress updates."""
        _schedule_broadcast("training_progress", job.to_dict())

    trainer.add_on_progress_callback(on_training_progress)

    # Inject dependencies into routes
    routes.engine = engine
    routes.registry = registry
    routes.trainer = trainer
    routes.coordinator = coordinator
    routes.watcher = watcher

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        """Application lifespan handler."""
        nonlocal _loop
        _loop = asyncio.get_running_loop()
        logger.info("Starting hotswap server...")

        # Load active model if exists
        active = registry.get_active()
        if active:
            try:
                model = MNISTClassifier.load(active.checkpoint_path)
                model.model_id = active.id
                model.version = active.version
                engine.load_model(model)
                logger.info(f"Loaded active model v{active.version}")
            except Exception as e:
                logger.warning(f"Failed to load active model: {e}")

        # Start file watcher
        if watcher and enable_watcher:
            watcher.start()
            logger.info(f"File watcher started on {watch_path}")

        yield

        # Cleanup
        logger.info("Shutting down...")
        if watcher:
            watcher.stop()
        coordinator.shutdown()
        trainer.shutdown()

    app = FastAPI(
        title="Hotswap ML",
        description="Hotswappable ML model system with shadow mode validation",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include API routes
    app.include_router(routes.router, prefix="/api")

    # Serve dashboard static files
    dashboard_path = Path(__file__).parent.parent.parent / "dashboard"
    if dashboard_path.exists():
        app.mount("/static", StaticFiles(directory=str(dashboard_path)), name="static")

        @app.get("/dashboard")
        @app.get("/dashboard/")
        async def serve_dashboard() -> FileResponse:
            """Serve the dashboard HTML."""
            return FileResponse(dashboard_path / "index.html")

    @app.get("/")
    async def root() -> dict:
        """Root endpoint."""
        return {
            "name": "Hotswap ML",
            "version": "0.1.0",
            "dashboard": "/dashboard",
            "api": "/api",
            "docs": "/docs",
        }

    return app
