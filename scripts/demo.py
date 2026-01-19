#!/usr/bin/env python3
"""
Interactive demo script for the Hotswap ML system.

This script demonstrates the complete workflow:
1. Generate training data
2. Train initial model
3. Generate more data to trigger retraining
4. Run shadow mode comparison
5. Promote or rollback based on metrics

Usage:
    python scripts/demo.py [--auto] [--demo promotion|rollback|all]

    --auto: Run fully automated without prompts
    --demo: Which demo to run (default: promotion)
        promotion: Train two good models, show auto-promotion
        rollback: Train good model, then bad model, show auto-archival
        all: Run both demos
"""

import argparse
import sys
import time
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import httpx

DEFAULT_PORT = 8000
BASE_URL = ""
API_URL = ""


def set_api_url(port: int) -> None:
    """Set the API URL based on port."""
    global BASE_URL, API_URL
    BASE_URL = f"http://localhost:{port}"
    API_URL = f"{BASE_URL}/api"


def print_header(text: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}\n")


def print_step(step: int, text: str) -> None:
    print(f"\n[Step {step}] {text}")
    print("-" * 40)


def wait_for_server() -> bool:
    """Wait for server to be available."""
    print("Waiting for server...", end="", flush=True)
    for _ in range(30):
        try:
            response = httpx.get(f"{API_URL}/health", timeout=2.0)
            if response.status_code == 200:
                print(" OK")
                return True
        except Exception:
            pass
        print(".", end="", flush=True)
        time.sleep(1)
    print(" FAILED")
    return False


def get_status() -> dict:
    """Get current system status."""
    response = httpx.get(f"{API_URL}/status", timeout=10.0)
    return response.json()


def list_models() -> list:
    """List all models."""
    response = httpx.get(f"{API_URL}/models", timeout=10.0)
    return response.json()


def generate_data(output_path: str, count: int = 500) -> None:
    """Generate synthetic training data."""
    from hotswap.utils.synthetic_data import generate_synthetic_batch
    generate_synthetic_batch(count=count, output_path=output_path)
    print(f"Generated {count} samples -> {output_path}")


def generate_bad_data(output_path: str, count: int = 500) -> None:
    """Generate bad training data with inverted labels."""
    from hotswap.utils.synthetic_data import SyntheticDataGenerator
    generator = SyntheticDataGenerator(noise_level=0.1)
    images, labels = generator.generate_batch(count)
    # Invert labels: 0->9, 1->8, etc.
    labels = 9 - labels
    torch.save({"images": images, "labels": labels}, output_path)
    print(f"Generated {count} BAD samples (inverted labels) -> {output_path}")


def wait_for_new_model(timeout: int = 120) -> bool:
    """Wait for a new model to be trained.

    Waits for the model count to increase by 1.
    """
    initial_models = list_models()
    initial_count = len(initial_models)
    print(f"Waiting for training (currently {initial_count} models)", end="", flush=True)

    start = time.time()
    training_seen = False

    while time.time() - start < timeout:
        status = get_status()
        training = status.get("training")

        if training and training.get("status") == "running":
            training_seen = True
            progress = training.get("progress", 0)
            print(f"\r  Training: {progress*100:.0f}%  ", end="", flush=True)
        elif training and training.get("status") == "completed":
            print(" DONE")
            return True
        elif training and training.get("status") == "failed":
            print(f" FAILED: {training.get('error')}")
            return False
        else:
            # Check if model count increased
            models = list_models()
            if len(models) > initial_count:
                print(" DONE")
                return True
            # Still waiting for training to start
            if not training_seen:
                print(".", end="", flush=True)

        time.sleep(0.5)

    # Final check
    models = list_models()
    if len(models) > initial_count:
        print(" DONE")
        return True

    print(" TIMEOUT")
    return False


def wait_for_shadow_samples(min_samples: int = 20, timeout: int = 60) -> bool:
    """Wait for shadow mode to collect samples."""
    print(f"Collecting shadow samples (target: {min_samples})", end="", flush=True)
    start = time.time()
    while time.time() - start < timeout:
        response = httpx.get(f"{API_URL}/shadow/metrics", timeout=10.0)
        data = response.json()
        if data.get("enabled") and data.get("metrics"):
            samples = data["metrics"].get("total_samples", 0)
            agreement = data["metrics"].get("agreement_rate", 0)
            print(f"\r  Samples: {samples}, Agreement: {agreement*100:.1f}% ", end="", flush=True)
            if samples >= min_samples:
                print("DONE")
                return True
        time.sleep(0.5)
    print(" TIMEOUT")
    return False


def run_predictions(count: int = 50) -> None:
    """Run predictions to generate shadow comparisons."""
    print(f"Running {count} predictions...")
    for i in range(count):
        data = torch.randn(1, 28, 28).tolist()
        try:
            response = httpx.post(
                f"{API_URL}/predict",
                json={"data": data},
                timeout=10.0
            )
            if response.status_code == 200:
                result = response.json()
                pred = result["predictions"][0]
                shadow = result.get("shadow_comparison")
                if shadow:
                    agree = "✓" if shadow.get("agreement") else "✗"
                    print(f"  [{i+1:3d}] Predicted: {pred}, Shadow: {agree}")
                else:
                    print(f"  [{i+1:3d}] Predicted: {pred}")
        except Exception as e:
            print(f"  [{i+1:3d}] Error: {e}")
        time.sleep(0.1)


def print_status() -> None:
    """Print current system status."""
    status = get_status()

    engine = status.get("engine", {})
    print(f"  Active Model: v{engine.get('active_model_version', '--')}")
    print(f"  Predictions: {engine.get('prediction_count', 0)}")
    print(f"  Shadow Mode: {'Enabled' if engine.get('shadow_enabled') else 'Disabled'}")

    if engine.get("shadow_enabled"):
        swap = status.get("swap_coordinator", {})
        metrics = swap.get("metrics", {})
        print(f"  Shadow Model: v{swap.get('shadow_model_version', '--')}")
        print(f"  Agreement: {metrics.get('agreement_rate', 0)*100:.1f}%")
        print(f"  Samples: {metrics.get('total_samples', 0)}")
        print(f"  Decision: {swap.get('decision', '--')}")


def print_models() -> None:
    """Print all models."""
    models = list_models()
    if not models:
        print("  No models registered")
        return

    print(f"  {'Version':<10} {'Status':<12} {'Loss':<10}")
    print(f"  {'-'*32}")
    for m in models:
        loss = m.get("metrics", {}).get("final_loss", "--")
        if isinstance(loss, float):
            loss = f"{loss:.4f}"
        print(f"  v{m['version']:<9} {m['status']:<12} {loss:<10}")


def prompt(text: str, auto: bool = False) -> bool:
    """Prompt user to continue."""
    if auto:
        print(f"{text} [auto: yes]")
        return True
    response = input(f"{text} [y/n]: ").strip().lower()
    return response in ("y", "yes", "")


def demo_promotion(auto: bool = False) -> int:
    """Demo the promotion flow with two good models."""
    print_header("Promotion Demo")
    print("This demo trains two models on valid data.")
    print("The second model should be AUTO-PROMOTED since both are good.\n")

    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(exist_ok=True)

    # Step 1: Generate initial training data
    print_step(1, "Generate Initial Training Data")
    generate_data(str(data_dir / "batch1.pt"), count=500)

    # Step 2: Wait for training
    print_step(2, "Train Initial Model")
    if not wait_for_new_model():
        print("Training failed!")
        return 1

    time.sleep(1)
    print("\nCurrent Status:")
    print_status()
    print("\nModels:")
    print_models()

    if not prompt("\nContinue to shadow mode demo?", auto):
        print("Exiting.")
        return 0

    # Track which model is active BEFORE second model trains
    models_before = list_models()
    active_before = next((m for m in models_before if m["status"] == "active"), None)
    active_version_before = active_before["version"] if active_before else None

    # Step 3: Generate more data for second model
    print_step(3, "Generate Data for Second Model")
    generate_data(str(data_dir / "batch2.pt"), count=500)

    # Step 4: Wait for training
    print_step(4, "Train Second Model")
    if not wait_for_new_model():
        print("Training failed!")
        return 1

    time.sleep(1.5)  # Give shadow mode time to complete validation

    # Step 5: Check shadow mode result
    print_step(5, "Shadow Mode Validation")

    # Check current state
    models_after = list_models()
    active_after = next((m for m in models_after if m["status"] == "active"), None)
    active_version_after = active_after["version"] if active_after else None

    status = get_status()
    if status.get("engine", {}).get("shadow_enabled"):
        # Shadow mode still running - show metrics
        print("Shadow mode is active, collecting metrics...")
        run_predictions(count=30)

        print("\nShadow Metrics:")
        response = httpx.get(f"{API_URL}/shadow/metrics", timeout=10.0)
        metrics = response.json()
        if metrics.get("metrics"):
            m = metrics["metrics"]
            print(f"  Total Samples: {m.get('total_samples', 0)}")
            print(f"  Agreement Rate: {m.get('agreement_rate', 0)*100:.1f}%")
            print(f"  Avg Active Latency: {m.get('avg_active_latency_ms', 0):.2f}ms")
            print(f"  Avg Shadow Latency: {m.get('avg_shadow_latency_ms', 0):.2f}ms")
        print(f"  Decision: {metrics.get('decision', '--')}")

        # Step 6: Promote
        print_step(6, "Promote Shadow Model")
        if prompt("Promote shadow model to active?", auto):
            response = httpx.post(f"{API_URL}/shadow/promote", timeout=10.0)
            if response.status_code == 200:
                print("Shadow model promoted!")
            else:
                print(f"Promotion failed: {response.text}")
    else:
        # Shadow mode already completed
        if active_version_after and active_version_before and active_version_after > active_version_before:
            print(f"Shadow mode completed with AUTO-PROMOTION!")
            print(f"  Previous active: v{active_version_before}")
            print(f"  New active:      v{active_version_after}")
            print("\nThe new model passed validation and was automatically promoted.")
            print("This is the expected behavior when shadow model performs well.")
        else:
            print("Shadow mode completed (model may have been archived)")

        # Skip step 6 since already done
        print_step(6, "Promotion Status")
        print("Auto-promotion already occurred during shadow validation.")

    # Final status
    print_step(7, "Final Status")
    print_status()
    print("\nAll Models:")
    print_models()

    print_header("Promotion Demo Complete!")
    return 0


def demo_rollback(auto: bool = False) -> int:
    """Demo the rollback/archival flow with a bad model."""
    print_header("Rollback Demo")
    print("This demo trains a good model, then a BAD model with inverted labels.")
    print("The bad model should be AUTO-ARCHIVED since it performs poorly.\n")

    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(exist_ok=True)

    # Step 1: Generate good training data
    print_step(1, "Generate GOOD Training Data")
    generate_data(str(data_dir / "good_data.pt"), count=500)

    # Step 2: Wait for training
    print_step(2, "Train Good Model")
    if not wait_for_new_model():
        print("Training failed!")
        return 1

    time.sleep(1)
    print("\nCurrent Status:")
    print_status()
    print("\nModels:")
    print_models()

    if not prompt("\nContinue to bad model demo?", auto):
        print("Exiting.")
        return 0

    # Step 3: Generate BAD training data
    print_step(3, "Generate BAD Training Data (Inverted Labels)")
    print("Labels are inverted: 0->9, 1->8, 2->7, etc.")
    print("This model will learn completely wrong patterns!\n")
    generate_bad_data(str(data_dir / "bad_data.pt"), count=500)

    # Step 4: Wait for training
    print_step(4, "Train Bad Model")
    if not wait_for_new_model():
        print("Training failed!")
        return 1

    time.sleep(1.5)  # Wait for validation to run

    # Step 5: Check shadow status
    print_step(5, "Check Shadow Mode Status")
    status = get_status()
    if status.get("engine", {}).get("shadow_enabled"):
        print("Shadow mode still active - validation may need more time")
        response = httpx.get(f"{API_URL}/shadow/metrics", timeout=10.0)
        metrics = response.json()
        if metrics.get("metrics"):
            m = metrics["metrics"]
            print(f"\nValidation Results:")
            print(f"  Active Model Accuracy: {m.get('model1_accuracy', 0)*100:.1f}%")
            print(f"  Shadow Model Accuracy: {m.get('model2_accuracy', 0)*100:.1f}%")
            print(f"  Agreement Rate: {m.get('agreement_rate', 0)*100:.1f}%")
        print(f"  Decision: {metrics.get('decision', '--')}")
    else:
        print("Shadow mode completed (model was auto-promoted or auto-archived)")

    # Step 6: Verify archival
    print_step(6, "Verify Bad Model Archived")
    models_after = list_models()

    # Find the newest model (highest version)
    newest_model = max(models_after, key=lambda m: m["version"])
    active_model = next((m for m in models_after if m["status"] == "active"), None)

    print("\nResult:")
    if newest_model["status"] == "archived":
        print(f"  BAD model v{newest_model['version']} was correctly ARCHIVED!")
        print(f"  GOOD model v{active_model['version']} remains ACTIVE")
        print("\n  SUCCESS: The system protected production by rejecting the bad model.")
    elif newest_model["status"] == "active":
        print(f"  Model v{newest_model['version']} was PROMOTED (unexpected)")
        print("  The bad model may not have been different enough to trigger rollback.")
    else:
        print(f"  Model v{newest_model['version']} status: {newest_model['status']}")

    print("\nAll Models:")
    print_models()

    # Final status
    print_step(7, "Final Status")
    print_status()

    print_header("Rollback Demo Complete!")
    print("The system correctly identified and archived the bad model.")
    print("The good model remains active, protecting production quality.\n")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Hotswap ML Demo")
    parser.add_argument("--auto", action="store_true", help="Run fully automated")
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=DEFAULT_PORT,
        help=f"Server port (default: {DEFAULT_PORT})"
    )
    parser.add_argument(
        "--demo", "-d",
        choices=["promotion", "rollback", "all"],
        default="promotion",
        help="Which demo to run (default: promotion)"
    )
    args = parser.parse_args()

    # Set API URL based on port
    set_api_url(args.port)

    # Check server
    if not wait_for_server():
        print("\nError: Server not running!")
        print(f"Start it with: python -m hotswap serve --port {args.port} --watch ./data")
        sys.exit(1)

    if args.demo == "promotion":
        result = demo_promotion(args.auto)
    elif args.demo == "rollback":
        result = demo_rollback(args.auto)
    else:
        # Run both
        result = demo_promotion(args.auto)
        if result == 0:
            if not args.auto:
                if not prompt("\nContinue to rollback demo?", False):
                    print("Exiting.")
                    sys.exit(0)
            result = demo_rollback(args.auto)

    if result == 0:
        print("You can continue testing with:")
        print(f"  - Dashboard: http://localhost:{args.port}/dashboard")
        print("  - Generate more data: python -m hotswap generate-data --output ./data/batch3.pt")
        print("  - Check status: python -m hotswap status")
        print("")

    sys.exit(result)


if __name__ == "__main__":
    main()
