"""Unit tests for ModelRegistry."""

import pytest

from hotswap.core.registry import ModelRegistry, ModelStatus


class TestModelRegistry:
    """Tests for ModelRegistry."""

    def test_register_model(self, registry, temp_dir):
        """Test registering a new model."""
        checkpoint = temp_dir / "model.pt"
        checkpoint.touch()

        record = registry.register(
            checkpoint_path=checkpoint,
            status=ModelStatus.READY,
            metrics={"loss": 0.1},
        )

        assert record.id is not None
        assert record.version == 1
        assert record.status == ModelStatus.READY
        assert record.metrics == {"loss": 0.1}

    def test_version_auto_increment(self, registry, temp_dir):
        """Test version auto-incrementing."""
        checkpoint = temp_dir / "model.pt"
        checkpoint.touch()

        record1 = registry.register(checkpoint_path=checkpoint)
        record2 = registry.register(checkpoint_path=checkpoint)
        record3 = registry.register(checkpoint_path=checkpoint)

        assert record1.version == 1
        assert record2.version == 2
        assert record3.version == 3

    def test_get_model(self, registry, temp_dir):
        """Test getting a model by ID."""
        checkpoint = temp_dir / "model.pt"
        checkpoint.touch()

        record = registry.register(checkpoint_path=checkpoint)
        fetched = registry.get(record.id)

        assert fetched is not None
        assert fetched.id == record.id
        assert fetched.version == record.version

    def test_get_nonexistent(self, registry):
        """Test getting a non-existent model."""
        result = registry.get("nonexistent-id")
        assert result is None

    def test_list_models(self, registry, temp_dir):
        """Test listing all models."""
        checkpoint = temp_dir / "model.pt"
        checkpoint.touch()

        for _ in range(5):
            registry.register(checkpoint_path=checkpoint)

        models = registry.list_models()

        assert len(models) == 5

    def test_list_models_by_status(self, registry, temp_dir):
        """Test filtering models by status."""
        checkpoint = temp_dir / "model.pt"
        checkpoint.touch()

        registry.register(checkpoint_path=checkpoint, status=ModelStatus.READY)
        registry.register(checkpoint_path=checkpoint, status=ModelStatus.READY)
        registry.register(checkpoint_path=checkpoint, status=ModelStatus.TRAINING)

        ready = registry.list_models(status=ModelStatus.READY)
        training = registry.list_models(status=ModelStatus.TRAINING)

        assert len(ready) == 2
        assert len(training) == 1

    def test_update_status(self, registry, temp_dir):
        """Test updating model status."""
        checkpoint = temp_dir / "model.pt"
        checkpoint.touch()

        record = registry.register(checkpoint_path=checkpoint, status=ModelStatus.READY)
        registry.update_status(record.id, ModelStatus.ACTIVE)

        fetched = registry.get(record.id)
        assert fetched.status == ModelStatus.ACTIVE

    def test_set_active(self, registry, temp_dir):
        """Test setting a model as active."""
        checkpoint = temp_dir / "model.pt"
        checkpoint.touch()

        # Create first model and set active
        record1 = registry.register(checkpoint_path=checkpoint, status=ModelStatus.READY)
        registry.set_active(record1.id)

        # Create second model and set active
        record2 = registry.register(checkpoint_path=checkpoint, status=ModelStatus.READY)
        registry.set_active(record2.id)

        # First should be archived, second should be active
        fetched1 = registry.get(record1.id)
        fetched2 = registry.get(record2.id)

        assert fetched1.status == ModelStatus.ARCHIVED
        assert fetched2.status == ModelStatus.ACTIVE

    def test_get_active(self, registry, temp_dir):
        """Test getting the active model."""
        checkpoint = temp_dir / "model.pt"
        checkpoint.touch()

        # No active model initially
        assert registry.get_active() is None

        record = registry.register(checkpoint_path=checkpoint, status=ModelStatus.READY)
        registry.set_active(record.id)

        active = registry.get_active()
        assert active is not None
        assert active.id == record.id

    def test_get_shadow(self, registry, temp_dir):
        """Test getting the shadow model."""
        checkpoint = temp_dir / "model.pt"
        checkpoint.touch()

        record = registry.register(checkpoint_path=checkpoint, status=ModelStatus.SHADOW)

        shadow = registry.get_shadow()
        assert shadow is not None
        assert shadow.id == record.id

    def test_update_metrics(self, registry, temp_dir):
        """Test updating model metrics."""
        checkpoint = temp_dir / "model.pt"
        checkpoint.touch()

        record = registry.register(checkpoint_path=checkpoint)
        registry.update_metrics(record.id, {"accuracy": 0.95, "loss": 0.05})

        fetched = registry.get(record.id)
        assert fetched.metrics == {"accuracy": 0.95, "loss": 0.05}

    def test_delete(self, registry, temp_dir):
        """Test deleting a model record."""
        checkpoint = temp_dir / "model.pt"
        checkpoint.touch()

        record = registry.register(checkpoint_path=checkpoint)
        registry.delete(record.id)

        assert registry.get(record.id) is None

    def test_list_limit(self, registry, temp_dir):
        """Test list limit parameter."""
        checkpoint = temp_dir / "model.pt"
        checkpoint.touch()

        for _ in range(10):
            registry.register(checkpoint_path=checkpoint)

        models = registry.list_models(limit=5)
        assert len(models) == 5

    def test_models_ordered_by_version_desc(self, registry, temp_dir):
        """Test models are returned in descending version order."""
        checkpoint = temp_dir / "model.pt"
        checkpoint.touch()

        for _ in range(5):
            registry.register(checkpoint_path=checkpoint)

        models = registry.list_models()
        versions = [m.version for m in models]

        assert versions == [5, 4, 3, 2, 1]
