"""CLI entry point for the hotswap system."""

import logging
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(
    name="hotswap",
    help="Hotswappable ML model system with shadow mode validation",
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@app.command()
def serve(
    port: int = typer.Option(8000, "--port", "-p", help="Server port"),
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Server host"),
    watch: Optional[Path] = typer.Option(None, "--watch", "-w", help="Directory to watch for new data"),
    no_watch: bool = typer.Option(False, "--no-watch", help="Disable file watcher"),
    reload: bool = typer.Option(False, "--reload", "-r", help="Enable auto-reload for development"),
) -> None:
    """Start the hotswap server."""
    import uvicorn

    from .api.app import create_app
    from .utils.config import Config

    config = Config(port=port, host=host)
    watch_path = watch or config.data_dir

    if no_watch:
        typer.echo(f"Starting server on {host}:{port} (watcher disabled)")
    else:
        typer.echo(f"Starting server on {host}:{port}, watching {watch_path}")

    # Create app
    application = create_app(
        config=config,
        enable_watcher=not no_watch,
        watch_path=watch_path,
    )

    # Run with uvicorn
    uvicorn.run(
        application,
        host=host,
        port=port,
        reload=reload,
    )


@app.command("generate-data")
def generate_data(
    output: Path = typer.Option(..., "--output", "-o", help="Output file path"),
    count: int = typer.Option(1000, "--count", "-n", help="Number of samples to generate"),
    noise: float = typer.Option(0.1, "--noise", help="Noise level (0-1)"),
) -> None:
    """Generate synthetic MNIST-like training data."""
    from .utils.synthetic_data import generate_synthetic_batch

    typer.echo(f"Generating {count} synthetic samples...")

    images, labels = generate_synthetic_batch(
        count=count,
        output_path=output,
        noise_level=noise,
    )

    typer.echo(f"Generated {len(labels)} samples")
    typer.echo(f"  Images shape: {images.shape}")
    typer.echo(f"  Labels shape: {labels.shape}")
    typer.echo(f"  Saved to: {output}")


@app.command()
def train(
    data: Path = typer.Option(..., "--data", "-d", help="Path to training data file"),
    epochs: int = typer.Option(5, "--epochs", "-e", help="Number of training epochs"),
    batch_size: int = typer.Option(32, "--batch-size", "-b", help="Batch size"),
    lr: float = typer.Option(0.001, "--lr", help="Learning rate"),
) -> None:
    """Manually trigger training on a data file."""
    import time

    from .core.registry import ModelRegistry
    from .models.mnist_cnn import MNISTClassifier
    from .workers.trainer import TrainingWorker, TrainingStatus

    if not data.exists():
        typer.echo(f"Error: Data file not found: {data}", err=True)
        raise typer.Exit(1)

    registry = ModelRegistry()
    trainer = TrainingWorker(
        model_class=MNISTClassifier,
        registry=registry,
    )

    typer.echo(f"Starting training on {data}...")
    job_id = trainer.train(
        data_path=data,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
    )

    # Wait for completion with progress
    with typer.progressbar(length=100, label="Training") as progress:
        last_progress = 0
        while True:
            job = trainer.get_job(job_id)
            if not job:
                break

            current = int(job.progress * 100)
            if current > last_progress:
                progress.update(current - last_progress)
                last_progress = current

            if job.status in (TrainingStatus.COMPLETED, TrainingStatus.FAILED, TrainingStatus.CANCELLED):
                break

            time.sleep(0.5)

        # Fill to 100%
        if last_progress < 100:
            progress.update(100 - last_progress)

    job = trainer.get_job(job_id)
    if job and job.status == TrainingStatus.COMPLETED:
        typer.echo(f"\nTraining completed!")
        typer.echo(f"  Model ID: {job.model_id}")
        typer.echo(f"  Final loss: {job.current_loss:.4f}")
    elif job and job.status == TrainingStatus.FAILED:
        typer.echo(f"\nTraining failed: {job.error}", err=True)
        raise typer.Exit(1)

    trainer.shutdown()


@app.command("models")
def list_models_cmd(
    status: Optional[str] = typer.Option(None, "--status", "-s", help="Filter by status"),
    limit: int = typer.Option(20, "--limit", "-n", help="Maximum number of models to show"),
) -> None:
    """List registered models."""
    from .core.registry import ModelRegistry, ModelStatus

    registry = ModelRegistry()

    model_status = ModelStatus(status) if status else None
    models = registry.list_models(status=model_status, limit=limit)

    if not models:
        typer.echo("No models found.")
        return

    typer.echo(f"{'ID':<36} {'Ver':<5} {'Status':<10} {'Created':<20} {'Loss':<10}")
    typer.echo("-" * 85)

    for m in models:
        loss = m.metrics.get("final_loss", "-") if m.metrics else "-"
        if isinstance(loss, float):
            loss = f"{loss:.4f}"
        typer.echo(
            f"{m.id:<36} v{m.version:<4} {m.status.value:<10} "
            f"{m.created_at.strftime('%Y-%m-%d %H:%M'):<20} {loss:<10}"
        )


@app.command()
def swap(
    model_id: str = typer.Option(..., "--model-id", "-m", help="Model ID to swap to"),
) -> None:
    """Force swap to a specific model version."""
    import httpx

    try:
        response = httpx.post(
            "http://localhost:8000/api/models/{}/activate".format(model_id),
            timeout=30.0,
        )
        response.raise_for_status()
        data = response.json()

        if data.get("success"):
            typer.echo(f"Swapped to model {model_id}")
            if data.get("active_model_version"):
                typer.echo(f"  Active version: v{data['active_model_version']}")
        else:
            typer.echo(f"Swap failed: {data.get('message')}", err=True)

    except httpx.ConnectError:
        typer.echo("Error: Could not connect to server. Is it running?", err=True)
        raise typer.Exit(1)
    except httpx.HTTPStatusError as e:
        typer.echo(f"Error: {e.response.text}", err=True)
        raise typer.Exit(1)


@app.command()
def status() -> None:
    """Get current system status."""
    import httpx

    try:
        response = httpx.get("http://localhost:8000/api/status", timeout=10.0)
        response.raise_for_status()
        data = response.json()

        typer.echo("=== Hotswap Status ===\n")

        # Engine status
        engine = data.get("engine", {})
        typer.echo("Inference Engine:")
        typer.echo(f"  Active model: {'Yes' if engine.get('has_active_model') else 'No'}")
        if engine.get("active_model_version"):
            typer.echo(f"  Model version: v{engine['active_model_version']}")
        typer.echo(f"  Predictions: {engine.get('prediction_count', 0)}")
        typer.echo(f"  Shadow mode: {'Enabled' if engine.get('shadow_enabled') else 'Disabled'}")

        # Swap coordinator
        swap = data.get("swap_coordinator", {})
        typer.echo(f"\nSwap Coordinator:")
        typer.echo(f"  State: {swap.get('state', 'unknown')}")
        if swap.get("shadow_model_id"):
            typer.echo(f"  Shadow model: {swap['shadow_model_id']}")
            if swap.get("metrics"):
                metrics = swap["metrics"]
                typer.echo(f"  Agreement rate: {metrics.get('agreement_rate', 0) * 100:.1f}%")
                typer.echo(f"  Samples: {metrics.get('total_samples', 0)}")
            if swap.get("decision"):
                typer.echo(f"  Decision: {swap['decision']}")

        # Training
        training = data.get("training")
        if training:
            typer.echo(f"\nTraining:")
            typer.echo(f"  Job ID: {training.get('id')}")
            typer.echo(f"  Status: {training.get('status')}")
            typer.echo(f"  Progress: {training.get('progress', 0) * 100:.0f}%")

        # Watcher
        typer.echo(f"\nFile Watcher: {'Active' if data.get('watcher_active') else 'Inactive'}")

    except httpx.ConnectError:
        typer.echo("Error: Could not connect to server. Is it running?", err=True)
        raise typer.Exit(1)
    except httpx.HTTPStatusError as e:
        typer.echo(f"Error: {e.response.text}", err=True)
        raise typer.Exit(1)


@app.command()
def shadow(
    start: Optional[str] = typer.Option(None, "--start", help="Start shadow mode with model ID"),
    promote: bool = typer.Option(False, "--promote", help="Promote shadow model"),
    cancel: bool = typer.Option(False, "--cancel", help="Cancel shadow mode"),
    metrics: bool = typer.Option(False, "--metrics", help="Show shadow metrics"),
) -> None:
    """Manage shadow mode."""
    import httpx

    base_url = "http://localhost:8000/api/shadow"

    try:
        if start:
            response = httpx.post(f"{base_url}/start", json={"model_id": start}, timeout=30.0)
            response.raise_for_status()
            typer.echo(f"Shadow mode started with model {start}")

        elif promote:
            response = httpx.post(f"{base_url}/promote", timeout=30.0)
            response.raise_for_status()
            data = response.json()
            typer.echo("Shadow model promoted to active")
            if data.get("active_model_version"):
                typer.echo(f"  Active version: v{data['active_model_version']}")

        elif cancel:
            response = httpx.post(f"{base_url}/cancel", timeout=30.0)
            response.raise_for_status()
            typer.echo("Shadow mode cancelled")

        elif metrics:
            response = httpx.get(f"{base_url}/metrics", timeout=10.0)
            response.raise_for_status()
            data = response.json()

            if not data.get("enabled"):
                typer.echo("Shadow mode is not active")
                return

            typer.echo("=== Shadow Mode Metrics ===\n")
            typer.echo(f"Shadow model: {data.get('shadow_model_id')}")
            typer.echo(f"Version: v{data.get('shadow_model_version')}")

            m = data.get("metrics", {})
            typer.echo(f"\nSamples: {m.get('total_samples', 0)}")
            typer.echo(f"Agreement rate: {m.get('agreement_rate', 0) * 100:.1f}%")
            typer.echo(f"Agreements: {m.get('agreements', 0)}")
            typer.echo(f"Disagreements: {m.get('disagreements', 0)}")
            typer.echo(f"Avg active latency: {m.get('avg_active_latency_ms', 0):.2f}ms")
            typer.echo(f"Avg shadow latency: {m.get('avg_shadow_latency_ms', 0):.2f}ms")
            typer.echo(f"\nDecision: {data.get('decision', 'unknown')}")

        else:
            typer.echo("Use --start, --promote, --cancel, or --metrics")

    except httpx.ConnectError:
        typer.echo("Error: Could not connect to server. Is it running?", err=True)
        raise typer.Exit(1)
    except httpx.HTTPStatusError as e:
        typer.echo(f"Error: {e.response.text}", err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
