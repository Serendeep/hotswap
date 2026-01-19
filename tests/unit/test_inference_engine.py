"""Unit tests for the InferenceEngine."""

import threading
import time

import pytest
import torch

from hotswap.core.inference import InferenceEngine
from hotswap.core.shadow import ShadowModeCollector


class TestInferenceEngine:
    """Tests for InferenceEngine."""

    def test_load_model(self, engine, simple_model, sample_input):
        """Test loading a model."""
        engine.load_model(simple_model)

        assert engine.active_model is not None
        assert not engine.shadow_enabled

    def test_predict_without_model(self, engine, sample_input):
        """Test prediction fails without a model."""
        with pytest.raises(RuntimeError, match="No active model"):
            engine.predict(sample_input)

    def test_predict_with_model(self, engine, simple_model, sample_input):
        """Test prediction with a loaded model."""
        engine.load_model(simple_model)

        result = engine.predict(sample_input)

        assert "prediction" in result
        assert "latency_ms" in result
        assert result["prediction"].shape[0] == sample_input.shape[0]

    def test_enable_shadow_mode(self, engine, simple_model):
        """Test enabling shadow mode."""
        engine.load_model(simple_model)

        shadow_model = type(simple_model)(output_value=3)
        collector = ShadowModeCollector()
        engine.enable_shadow(shadow_model, collector)

        assert engine.shadow_enabled
        assert engine.shadow_model is not None
        assert engine.shadow_collector is not None

    def test_shadow_mode_prediction(self, engine, simple_model, sample_input):
        """Test prediction in shadow mode collects comparison."""
        engine.load_model(simple_model)

        shadow_model = type(simple_model)(output_value=3)
        collector = ShadowModeCollector()
        engine.enable_shadow(shadow_model, collector)

        result = engine.predict(sample_input)

        assert "shadow_comparison" in result
        assert collector.total_samples == 1

    def test_disable_shadow_mode(self, engine, simple_model):
        """Test disabling shadow mode."""
        engine.load_model(simple_model)
        shadow_model = type(simple_model)(output_value=3)
        engine.enable_shadow(shadow_model)

        collector = engine.disable_shadow()

        assert not engine.shadow_enabled
        assert engine.shadow_model is None
        assert collector is not None

    def test_promote_shadow(self, engine, simple_model):
        """Test promoting shadow model to active."""
        engine.load_model(simple_model)
        shadow_model = type(simple_model)(output_value=7)
        shadow_model.model_id = "shadow-id"
        engine.enable_shadow(shadow_model)

        success = engine.promote_shadow()

        assert success
        assert not engine.shadow_enabled
        assert engine.active_model.model_id == "shadow-id"

    def test_swap_model(self, engine, simple_model):
        """Test atomic model swap."""
        engine.load_model(simple_model)

        new_model = type(simple_model)(output_value=9)
        new_model.model_id = "new-id"
        engine.swap_model(new_model)

        assert engine.active_model.model_id == "new-id"

    def test_concurrent_reads(self, engine, simple_model, sample_input):
        """Test concurrent reads don't block each other."""
        engine.load_model(simple_model)

        results = []
        errors = []

        def read_worker():
            try:
                for _ in range(10):
                    result = engine.predict(sample_input)
                    results.append(result)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=read_worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 50

    def test_swap_during_reads(self, engine, simple_model, sample_input):
        """Test model swap during concurrent reads."""
        engine.load_model(simple_model)

        results = []
        errors = []
        swap_done = threading.Event()

        def read_worker():
            try:
                for _ in range(20):
                    result = engine.predict(sample_input)
                    results.append(result)
                    time.sleep(0.01)
            except Exception as e:
                errors.append(e)

        def swap_worker():
            time.sleep(0.05)  # Let some reads happen
            new_model = type(simple_model)(output_value=8)
            engine.swap_model(new_model)
            swap_done.set()

        threads = [threading.Thread(target=read_worker) for _ in range(3)]
        threads.append(threading.Thread(target=swap_worker))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert swap_done.is_set()

    def test_prediction_count(self, engine, simple_model, sample_input):
        """Test prediction counter."""
        engine.load_model(simple_model)

        for _ in range(5):
            engine.predict(sample_input)

        assert engine.prediction_count == 5

    def test_get_status(self, engine, simple_model):
        """Test status reporting."""
        engine.load_model(simple_model)
        simple_model.model_id = "test-id"
        simple_model.version = 1

        status = engine.get_status()

        assert status["has_active_model"] is True
        assert status["shadow_enabled"] is False
        assert "prediction_count" in status
