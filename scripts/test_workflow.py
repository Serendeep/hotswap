#!/usr/bin/env python3
"""
Automated test script for the complete Hotswap workflow.

This script tests the entire system without requiring a running server.
It directly uses the Python APIs to verify the workflow.

Usage:
    python scripts/test_workflow.py
"""

import sys
import time
import tempfile
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from hotswap.core.inference import InferenceEngine
from hotswap.core.registry import ModelRegistry, ModelStatus
from hotswap.models.mnist_cnn import MNISTClassifier
from hotswap.workers.trainer import TrainingWorker, TrainingStatus
from hotswap.workers.swap import SwapCoordinator
from hotswap.utils.synthetic_data import generate_synthetic_batch


def print_header(text: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}\n")


def print_step(step: int, text: str) -> None:
    print(f"\n[Step {step}] {text}")
    print("-" * 40)


def print_ok(text: str) -> None:
    print(f"  ✓ {text}")


def print_fail(text: str) -> None:
    print(f"  ✗ {text}")


def main():
    print_header("Hotswap Workflow Test")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Initialize components
        print_step(1, "Initialize Components")

        registry = ModelRegistry(db_path=tmpdir / "registry.db")
        print_ok("ModelRegistry created")

        engine = InferenceEngine()
        print_ok("InferenceEngine created")

        trainer = TrainingWorker(
            model_class=MNISTClassifier,
            registry=registry,
            checkpoint_dir=tmpdir / "checkpoints",
            default_epochs=2,
            default_batch_size=32,
        )
        print_ok("TrainingWorker created")

        coordinator = SwapCoordinator(
            engine=engine,
            registry=registry,
            model_class=MNISTClassifier,
            auto_shadow_on_ready=False,  # Manual control for test
            min_samples=10,
            auto_promote_threshold=0.5,  # Low threshold for testing
        )
        print_ok("SwapCoordinator created")

        # Generate training data
        print_step(2, "Generate Training Data")

        data_path1 = tmpdir / "batch1.pt"
        images, labels = generate_synthetic_batch(count=200, output_path=data_path1)
        print_ok(f"Generated 200 samples: {images.shape}")

        data_path2 = tmpdir / "batch2.pt"
        images, labels = generate_synthetic_batch(count=200, output_path=data_path2)
        print_ok(f"Generated 200 more samples: {images.shape}")

        # Train first model
        print_step(3, "Train First Model")

        job_id1 = trainer.train(data_path=data_path1, epochs=2)
        print_ok(f"Training job submitted: {job_id1}")

        while True:
            job = trainer.get_job(job_id1)
            if job.status == TrainingStatus.COMPLETED:
                print_ok(f"Training completed, loss: {job.current_loss:.4f}")
                break
            elif job.status == TrainingStatus.FAILED:
                print_fail(f"Training failed: {job.error}")
                return 1
            time.sleep(0.1)

        model1_id = job.model_id

        # Activate first model
        print_step(4, "Activate First Model")

        coordinator.force_swap(model1_id)
        assert engine.active_model is not None
        print_ok(f"Model {model1_id[:8]}... activated")

        # Run some predictions
        sample = torch.randn(1, 1, 28, 28)
        result = engine.predict(sample)
        print_ok(f"Prediction works, latency: {result['latency_ms']:.2f}ms")

        # Train second model
        print_step(5, "Train Second Model")

        job_id2 = trainer.train(data_path=data_path2, epochs=2)
        print_ok(f"Training job submitted: {job_id2}")

        while True:
            job = trainer.get_job(job_id2)
            if job.status == TrainingStatus.COMPLETED:
                print_ok(f"Training completed, loss: {job.current_loss:.4f}")
                break
            elif job.status == TrainingStatus.FAILED:
                print_fail(f"Training failed: {job.error}")
                return 1
            time.sleep(0.1)

        model2_id = job.model_id

        # Start shadow mode
        print_step(6, "Start Shadow Mode")

        coordinator.start_shadow(model2_id)
        assert engine.shadow_enabled
        print_ok(f"Shadow mode started with model {model2_id[:8]}...")

        # Run predictions with shadow comparison
        print_step(7, "Run Predictions with Shadow Comparison")

        agreements = 0
        total = 20
        for i in range(total):
            sample = torch.randn(1, 1, 28, 28)
            result = engine.predict(sample)

            if result.get("shadow_comparison", {}).get("agreement"):
                agreements += 1

        collector = engine.shadow_collector
        metrics = collector.get_metrics()
        print_ok(f"Ran {total} predictions")
        print_ok(f"Agreement rate: {metrics['agreement_rate']*100:.1f}%")
        print_ok(f"Avg active latency: {metrics['avg_active_latency_ms']:.2f}ms")
        print_ok(f"Avg shadow latency: {metrics['avg_shadow_latency_ms']:.2f}ms")
        print_ok(f"Decision: {collector.get_decision()}")

        # Promote shadow model
        print_step(8, "Promote Shadow Model")

        success = coordinator.promote()
        assert success
        assert not engine.shadow_enabled
        print_ok("Shadow model promoted to active")

        # Verify new model is active
        active = registry.get_active()
        assert active.id == model2_id
        print_ok(f"Active model is now {model2_id[:8]}...")

        # List all models
        print_step(9, "Final Model Registry")

        models = registry.list_models()
        print(f"\n  {'Version':<8} {'Status':<12} {'ID':<36}")
        print(f"  {'-'*56}")
        for m in models:
            print(f"  v{m.version:<7} {m.status.value:<12} {m.id}")

        # Cleanup
        trainer.shutdown()
        coordinator.shutdown()

        print_header("Promotion Test Passed!")
        return 0


def test_rollback_workflow():
    """Test that a bad model gets rolled back and archived."""
    print_header("Hotswap Rollback Test")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Initialize components
        print_step(1, "Initialize Components")

        registry = ModelRegistry(db_path=tmpdir / "registry.db")
        print_ok("ModelRegistry created")

        engine = InferenceEngine()
        print_ok("InferenceEngine created")

        trainer = TrainingWorker(
            model_class=MNISTClassifier,
            registry=registry,
            checkpoint_dir=tmpdir / "checkpoints",
            default_epochs=5,
        )
        print_ok("TrainingWorker created")

        coordinator = SwapCoordinator(
            engine=engine,
            registry=registry,
            model_class=MNISTClassifier,
            auto_shadow_on_ready=False,
            use_validation_mode=True,
        )
        print_ok("SwapCoordinator created (validation mode)")

        # Generate GOOD training data
        print_step(2, "Generate Training Data")

        good_data_path = tmpdir / "good.pt"
        images, labels = generate_synthetic_batch(count=500, output_path=good_data_path)
        print_ok(f"Good data: {len(labels)} samples with correct labels")

        # Generate BAD training data (inverted labels)
        bad_data_path = tmpdir / "bad.pt"
        from hotswap.utils.synthetic_data import SyntheticDataGenerator
        generator = SyntheticDataGenerator(noise_level=0.1)
        images, labels = generator.generate_batch(500)
        labels = 9 - labels  # Invert: 0->9, 1->8, 2->7, etc.
        torch.save({"images": images, "labels": labels}, bad_data_path)
        print_ok(f"Bad data: {len(labels)} samples with INVERTED labels (0->9, 1->8, ...)")

        # Train good model
        print_step(3, "Train Good Model")

        job_id1 = trainer.train(data_path=good_data_path, epochs=5)
        print_ok(f"Training job submitted: {job_id1}")

        while True:
            job = trainer.get_job(job_id1)
            if job.status == TrainingStatus.COMPLETED:
                print_ok(f"Training completed, loss: {job.current_loss:.4f}")
                break
            elif job.status == TrainingStatus.FAILED:
                print_fail(f"Training failed: {job.error}")
                return 1
            time.sleep(0.1)

        good_model_id = job.model_id

        # Activate good model
        print_step(4, "Activate Good Model")

        coordinator.force_swap(good_model_id)
        assert engine.active_model is not None
        print_ok(f"Good model {good_model_id[:8]}... activated")

        # Train bad model
        print_step(5, "Train Bad Model (inverted labels)")

        job_id2 = trainer.train(data_path=bad_data_path, epochs=5)
        print_ok(f"Training job submitted: {job_id2}")

        while True:
            job = trainer.get_job(job_id2)
            if job.status == TrainingStatus.COMPLETED:
                print_ok(f"Training completed, loss: {job.current_loss:.4f}")
                break
            elif job.status == TrainingStatus.FAILED:
                print_fail(f"Training failed: {job.error}")
                return 1
            time.sleep(0.1)

        bad_model_id = job.model_id

        # Start shadow mode - should auto-rollback
        print_step(6, "Start Shadow Mode (expect auto-rollback)")

        coordinator.start_shadow(bad_model_id)
        print_ok(f"Shadow mode started with bad model {bad_model_id[:8]}...")

        # Wait for validation to run
        time.sleep(1.5)

        # Check validation result
        print_step(7, "Verify Rollback")

        bad_record = registry.get(bad_model_id)
        if bad_record.status == ModelStatus.ARCHIVED:
            print_ok(f"Bad model correctly ARCHIVED")
        else:
            print_fail(f"Expected ARCHIVED, got {bad_record.status}")
            return 1

        active = registry.get_active()
        if active.id == good_model_id:
            print_ok(f"Good model still ACTIVE")
        else:
            print_fail(f"Wrong active model")
            return 1

        # Shadow mode should be disabled
        if not engine.shadow_enabled:
            print_ok("Shadow mode disabled after rollback")
        else:
            print_fail("Shadow mode still enabled")
            return 1

        # List all models
        print_step(8, "Final Model Registry")

        models = registry.list_models()
        print(f"\n  {'Version':<8} {'Status':<12} {'ID':<36}")
        print(f"  {'-'*56}")
        for m in models:
            status_str = m.status.value
            if m.id == good_model_id:
                status_str += " (good)"
            elif m.id == bad_model_id:
                status_str += " (bad)"
            print(f"  v{m.version:<7} {status_str:<20} {m.id}")

        # Cleanup
        trainer.shutdown()
        coordinator.shutdown()

        print_header("Rollback Test Passed!")
        return 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test hotswap workflows")
    parser.add_argument(
        "--test", "-t",
        choices=["promotion", "rollback", "all"],
        default="all",
        help="Which test to run"
    )
    args = parser.parse_args()

    if args.test == "promotion":
        sys.exit(main())
    elif args.test == "rollback":
        sys.exit(test_rollback_workflow())
    else:
        # Run both
        result1 = main()
        if result1 != 0:
            sys.exit(result1)
        result2 = test_rollback_workflow()
        sys.exit(result2)
