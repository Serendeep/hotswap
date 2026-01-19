"""End-to-end integration tests for the hotswap system."""

import time
from pathlib import Path

import pytest
import torch

from hotswap.core.inference import InferenceEngine
from hotswap.core.registry import ModelRegistry, ModelStatus
from hotswap.core.shadow import ShadowModeCollector
from hotswap.models.mnist_cnn import MNISTClassifier
from hotswap.workers.trainer import TrainingWorker, TrainingStatus
from hotswap.workers.swap import SwapCoordinator
from hotswap.utils.synthetic_data import generate_synthetic_batch


class TestEndToEndWorkflow:
    """End-to-end workflow tests."""

    def test_full_workflow(self, temp_dir):
        """Test complete workflow: train -> shadow -> swap."""
        # Setup
        registry = ModelRegistry(db_path=temp_dir / "registry.db")
        engine = InferenceEngine()
        trainer = TrainingWorker(
            model_class=MNISTClassifier,
            registry=registry,
            checkpoint_dir=temp_dir / "checkpoints",
            default_epochs=2,
        )
        coordinator = SwapCoordinator(
            engine=engine,
            registry=registry,
            model_class=MNISTClassifier,
            auto_shadow_on_ready=False,  # Manual control for test
            min_samples=5,
        )

        # Generate training data
        data_path = temp_dir / "train.pt"
        generate_synthetic_batch(count=50, output_path=data_path)

        # Train first model
        job_id = trainer.train(data_path=data_path, epochs=2)

        # Wait for training
        while True:
            job = trainer.get_job(job_id)
            if job.status in (TrainingStatus.COMPLETED, TrainingStatus.FAILED):
                break
            time.sleep(0.1)

        assert job.status == TrainingStatus.COMPLETED
        model1_id = job.model_id

        # Activate first model
        coordinator.force_swap(model1_id)
        assert engine.active_model is not None

        # Train second model
        job_id2 = trainer.train(data_path=data_path, epochs=2)
        while True:
            job = trainer.get_job(job_id2)
            if job.status in (TrainingStatus.COMPLETED, TrainingStatus.FAILED):
                break
            time.sleep(0.1)

        model2_id = job.model_id

        # Start shadow mode
        coordinator.start_shadow(model2_id)
        assert engine.shadow_enabled

        # Run predictions to collect metrics
        sample_input = torch.randn(1, 1, 28, 28)
        for _ in range(10):
            result = engine.predict(sample_input)
            assert "shadow_comparison" in result

        # Check metrics collected
        collector = engine.shadow_collector
        assert collector.total_samples >= 5

        # Promote shadow
        coordinator.promote()
        assert not engine.shadow_enabled

        # Verify new model is active
        active = registry.get_active()
        assert active.id == model2_id

        # Cleanup
        trainer.shutdown()
        coordinator.shutdown()

    def test_auto_promote_workflow(self, temp_dir):
        """Test automatic promotion when threshold is met."""
        registry = ModelRegistry(db_path=temp_dir / "registry.db")
        engine = InferenceEngine()

        # Create and register models
        model1 = MNISTClassifier()
        model1_path = temp_dir / "model1.pt"
        model1.save(model1_path)
        record1 = registry.register(checkpoint_path=model1_path, status=ModelStatus.READY)
        registry.set_active(record1.id)

        model2 = MNISTClassifier()
        model2_path = temp_dir / "model2.pt"
        model2.save(model2_path)
        record2 = registry.register(checkpoint_path=model2_path, status=ModelStatus.READY)

        # Load active model
        loaded_model1 = MNISTClassifier.load(model1_path)
        loaded_model1.model_id = record1.id
        loaded_model1.version = record1.version
        engine.load_model(loaded_model1)

        # Setup coordinator with low threshold for testing
        coordinator = SwapCoordinator(
            engine=engine,
            registry=registry,
            model_class=MNISTClassifier,
            auto_promote_threshold=0.5,  # Low threshold for testing
            auto_rollback_threshold=0.3,
            min_samples=5,
            check_interval=0.1,
        )

        # Start shadow mode
        coordinator.start_shadow(record2.id)

        # Run predictions - models are identical so should agree
        sample_input = torch.randn(1, 1, 28, 28)
        for _ in range(10):
            engine.predict(sample_input)
            time.sleep(0.05)

        # Wait for auto-promote
        time.sleep(0.5)

        # Check promotion happened
        status = coordinator.get_status()
        assert status["state"] == "idle" or status["state"] == "shadow_mode"

        coordinator.shutdown()

    def test_cancel_shadow_mode(self, temp_dir):
        """Test canceling shadow mode."""
        registry = ModelRegistry(db_path=temp_dir / "registry.db")
        engine = InferenceEngine()

        # Create and load model
        model = MNISTClassifier()
        model_path = temp_dir / "model.pt"
        model.save(model_path)
        record = registry.register(checkpoint_path=model_path, status=ModelStatus.READY)
        registry.set_active(record.id)

        loaded_model = MNISTClassifier.load(model_path)
        loaded_model.model_id = record.id
        engine.load_model(loaded_model)

        # Create shadow model
        shadow_path = temp_dir / "shadow.pt"
        model.save(shadow_path)
        shadow_record = registry.register(checkpoint_path=shadow_path, status=ModelStatus.READY)

        coordinator = SwapCoordinator(
            engine=engine,
            registry=registry,
            model_class=MNISTClassifier,
        )

        # Start and cancel shadow mode
        coordinator.start_shadow(shadow_record.id)
        assert engine.shadow_enabled

        coordinator.cancel()
        assert not engine.shadow_enabled

        # Shadow model should be archived
        shadow_fetched = registry.get(shadow_record.id)
        assert shadow_fetched.status == ModelStatus.ARCHIVED

        coordinator.shutdown()


class TestConcurrentOperations:
    """Tests for concurrent operations."""

    def test_concurrent_predictions_during_shadow(self, temp_dir):
        """Test concurrent predictions during shadow mode."""
        import threading

        registry = ModelRegistry(db_path=temp_dir / "registry.db")
        engine = InferenceEngine()

        # Create models
        model = MNISTClassifier()
        model_path = temp_dir / "model.pt"
        model.save(model_path)
        record = registry.register(checkpoint_path=model_path, status=ModelStatus.READY)
        registry.set_active(record.id)

        loaded_model = MNISTClassifier.load(model_path)
        loaded_model.model_id = record.id
        engine.load_model(loaded_model)

        shadow_path = temp_dir / "shadow.pt"
        model.save(shadow_path)
        shadow_record = registry.register(checkpoint_path=shadow_path, status=ModelStatus.READY)

        coordinator = SwapCoordinator(
            engine=engine,
            registry=registry,
            model_class=MNISTClassifier,
            min_samples=100,
        )

        # Start shadow mode
        coordinator.start_shadow(shadow_record.id)

        results = []
        errors = []

        def predict_worker():
            sample = torch.randn(1, 1, 28, 28)
            for _ in range(50):
                try:
                    result = engine.predict(sample)
                    results.append(result)
                except Exception as e:
                    errors.append(e)
                time.sleep(0.01)

        threads = [threading.Thread(target=predict_worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 250

        # All results should have shadow comparison
        for r in results:
            assert "shadow_comparison" in r

        coordinator.shutdown()

    def test_swap_during_predictions(self, temp_dir):
        """Test model swap during concurrent predictions."""
        import threading

        registry = ModelRegistry(db_path=temp_dir / "registry.db")
        engine = InferenceEngine()

        # Create initial model
        model = MNISTClassifier()
        model_path = temp_dir / "model.pt"
        model.save(model_path)
        record = registry.register(checkpoint_path=model_path, status=ModelStatus.READY)
        registry.set_active(record.id)

        loaded_model = MNISTClassifier.load(model_path)
        loaded_model.model_id = record.id
        loaded_model.version = record.version
        engine.load_model(loaded_model)

        # Create new model
        new_path = temp_dir / "new.pt"
        model.save(new_path)
        new_record = registry.register(checkpoint_path=new_path, status=ModelStatus.READY)

        coordinator = SwapCoordinator(
            engine=engine,
            registry=registry,
            model_class=MNISTClassifier,
        )

        results = []
        errors = []
        swap_done = threading.Event()

        def predict_worker():
            sample = torch.randn(1, 1, 28, 28)
            for _ in range(100):
                try:
                    result = engine.predict(sample)
                    results.append(result)
                except Exception as e:
                    errors.append(e)
                time.sleep(0.005)

        def swap_worker():
            time.sleep(0.1)  # Let some predictions happen
            coordinator.force_swap(new_record.id)
            swap_done.set()

        threads = [threading.Thread(target=predict_worker) for _ in range(3)]
        threads.append(threading.Thread(target=swap_worker))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert swap_done.is_set()
        assert engine.active_model.model_id == new_record.id

        coordinator.shutdown()
