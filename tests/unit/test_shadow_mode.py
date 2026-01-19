"""Unit tests for ShadowModeCollector."""

import pytest
import torch

from hotswap.core.shadow import ShadowModeCollector, ShadowMetrics


class TestShadowMetrics:
    """Tests for ShadowMetrics dataclass."""

    def test_agreement_rate_empty(self):
        """Test agreement rate with no samples."""
        metrics = ShadowMetrics()
        assert metrics.agreement_rate == 0.0

    def test_agreement_rate_calculation(self):
        """Test agreement rate calculation."""
        metrics = ShadowMetrics(total_samples=100, agreements=95, disagreements=5)
        assert metrics.agreement_rate == 0.95

    def test_avg_latency_empty(self):
        """Test average latency with no samples."""
        metrics = ShadowMetrics()
        assert metrics.avg_active_latency_ms == 0.0
        assert metrics.avg_shadow_latency_ms == 0.0

    def test_avg_latency_calculation(self):
        """Test average latency calculation."""
        metrics = ShadowMetrics(
            total_samples=10,
            total_active_latency_ms=100.0,
            total_shadow_latency_ms=120.0,
        )
        assert metrics.avg_active_latency_ms == 10.0
        assert metrics.avg_shadow_latency_ms == 12.0
        assert metrics.latency_difference_ms == 2.0

    def test_to_dict(self):
        """Test serialization to dictionary."""
        metrics = ShadowMetrics(
            total_samples=50,
            agreements=45,
            disagreements=5,
        )
        data = metrics.to_dict()

        assert data["total_samples"] == 50
        assert data["agreements"] == 45
        assert data["agreement_rate"] == 0.9


class TestShadowModeCollector:
    """Tests for ShadowModeCollector."""

    def test_record_agreement(self, shadow_collector):
        """Test recording agreeing predictions."""
        active = torch.tensor([[0.1, 0.9, 0.0]])
        shadow = torch.tensor([[0.2, 0.8, 0.0]])

        shadow_collector.record(active, shadow, 1.0, 1.2)

        metrics = shadow_collector.get_metrics()
        assert metrics["total_samples"] == 1
        assert metrics["agreements"] == 1
        assert metrics["disagreements"] == 0

    def test_record_disagreement(self, shadow_collector):
        """Test recording disagreeing predictions."""
        active = torch.tensor([[0.9, 0.1, 0.0]])  # Class 0
        shadow = torch.tensor([[0.1, 0.9, 0.0]])  # Class 1

        shadow_collector.record(active, shadow, 1.0, 1.2)

        metrics = shadow_collector.get_metrics()
        assert metrics["total_samples"] == 1
        assert metrics["agreements"] == 0
        assert metrics["disagreements"] == 1

    def test_batch_recording(self, shadow_collector):
        """Test recording batch predictions."""
        # Batch of 4, 3 agree, 1 disagrees
        active = torch.tensor([
            [0.1, 0.9, 0.0],  # Class 1
            [0.9, 0.1, 0.0],  # Class 0
            [0.0, 0.1, 0.9],  # Class 2
            [0.1, 0.9, 0.0],  # Class 1
        ])
        shadow = torch.tensor([
            [0.2, 0.8, 0.0],  # Class 1 (agree)
            [0.8, 0.2, 0.0],  # Class 0 (agree)
            [0.0, 0.2, 0.8],  # Class 2 (agree)
            [0.9, 0.05, 0.05],  # Class 0 (disagree)
        ])

        shadow_collector.record(active, shadow, 1.0, 1.5)

        # Note: batch is treated as one sample in current implementation
        metrics = shadow_collector.get_metrics()
        assert metrics["total_samples"] == 1

    def test_decision_wait(self, shadow_collector):
        """Test decision is 'wait' with few samples."""
        active = torch.tensor([[0.1, 0.9, 0.0]])
        shadow = torch.tensor([[0.2, 0.8, 0.0]])

        for _ in range(5):  # Less than min_samples (10)
            shadow_collector.record(active, shadow, 1.0, 1.2)

        assert shadow_collector.get_decision() == "wait"

    def test_decision_promote(self):
        """Test auto-promote decision."""
        collector = ShadowModeCollector(
            auto_promote_threshold=0.95,
            auto_rollback_threshold=0.85,
            min_samples_for_decision=10,
        )

        active = torch.tensor([[0.1, 0.9, 0.0]])
        shadow = torch.tensor([[0.2, 0.8, 0.0]])

        # Record 10 agreements
        for _ in range(10):
            collector.record(active, shadow, 1.0, 1.2)

        assert collector.get_decision() == "promote"
        assert collector.should_auto_promote() is True

    def test_decision_rollback(self):
        """Test auto-rollback decision."""
        collector = ShadowModeCollector(
            auto_promote_threshold=0.95,
            auto_rollback_threshold=0.85,
            min_samples_for_decision=10,
        )

        active_agree = torch.tensor([[0.1, 0.9, 0.0]])
        shadow_agree = torch.tensor([[0.2, 0.8, 0.0]])
        active_disagree = torch.tensor([[0.9, 0.1, 0.0]])
        shadow_disagree = torch.tensor([[0.1, 0.9, 0.0]])

        # Record 8 agreements, 2 disagreements (80% agreement)
        for _ in range(8):
            collector.record(active_agree, shadow_agree, 1.0, 1.2)
        for _ in range(2):
            collector.record(active_disagree, shadow_disagree, 1.0, 1.2)

        assert collector.get_decision() == "rollback"
        assert collector.should_auto_rollback() is True

    def test_decision_manual(self):
        """Test manual decision zone."""
        collector = ShadowModeCollector(
            auto_promote_threshold=0.95,
            auto_rollback_threshold=0.85,
            min_samples_for_decision=10,
        )

        active_agree = torch.tensor([[0.1, 0.9, 0.0]])
        shadow_agree = torch.tensor([[0.2, 0.8, 0.0]])
        active_disagree = torch.tensor([[0.9, 0.1, 0.0]])
        shadow_disagree = torch.tensor([[0.1, 0.9, 0.0]])

        # Record 9 agreements, 1 disagreement (90% agreement)
        for _ in range(9):
            collector.record(active_agree, shadow_agree, 1.0, 1.2)
        collector.record(active_disagree, shadow_disagree, 1.0, 1.2)

        assert collector.get_decision() == "manual"

    def test_reset(self, shadow_collector):
        """Test resetting metrics."""
        active = torch.tensor([[0.1, 0.9, 0.0]])
        shadow = torch.tensor([[0.2, 0.8, 0.0]])

        for _ in range(5):
            shadow_collector.record(active, shadow, 1.0, 1.2)

        assert shadow_collector.total_samples == 5

        shadow_collector.reset()

        assert shadow_collector.total_samples == 0

    def test_thread_safety(self, shadow_collector):
        """Test thread-safe recording."""
        import threading

        active = torch.tensor([[0.1, 0.9, 0.0]])
        shadow = torch.tensor([[0.2, 0.8, 0.0]])

        def record_worker():
            for _ in range(100):
                shadow_collector.record(active, shadow, 1.0, 1.2)

        threads = [threading.Thread(target=record_worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert shadow_collector.total_samples == 500
