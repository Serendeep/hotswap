#!/usr/bin/env python3
"""Run predictions against the hotswap server."""

import argparse
import sys
import time

import httpx
import torch


def main():
    parser = argparse.ArgumentParser(description="Run predictions against hotswap server")
    parser.add_argument("count", type=int, nargs="?", default=1, help="Number of predictions (default: 1)")
    parser.add_argument("--delay", type=float, default=0.1, help="Delay between predictions in seconds")
    parser.add_argument("--quiet", "-q", action="store_true", help="Only show summary")
    args = parser.parse_args()

    base_url = "http://localhost:8000/api"

    # Check server health
    try:
        response = httpx.get(f"{base_url}/health", timeout=5.0)
        if response.status_code != 200:
            print("Server not healthy")
            sys.exit(1)

        health = response.json()
        if not health.get("has_active_model"):
            print("No active model loaded!")
            print("\nTrain a model first:")
            print("  python -m hotswap generate-data --output ./data/batch1.pt --count 500")
            print("  # Wait for training to complete, then retry")
            sys.exit(1)

    except httpx.ConnectError:
        print("Cannot connect to server. Is it running?")
        print("\nStart the server:")
        print("  python -m hotswap serve --port 8000 --watch ./data")
        sys.exit(1)

    predictions = []
    agreements = 0
    shadow_count = 0
    total_latency = 0.0

    for i in range(args.count):
        data = torch.randn(1, 28, 28).tolist()

        try:
            response = httpx.post(
                f"{base_url}/predict",
                json={"data": data},
                timeout=10.0
            )
            result = response.json()

            if "detail" in result:
                print(f"[{i+1:4d}] Server error: {result['detail']}")
                continue

            if "predictions" not in result:
                print(f"[{i+1:4d}] Unexpected response: {result}")
                continue

            pred = result["predictions"][0]
            latency = result["latency_ms"]
            total_latency += latency
            predictions.append(pred)

            shadow = result.get("shadow_comparison")
            if shadow:
                shadow_count += 1
                if shadow.get("agreement"):
                    agreements += 1
                agree_str = "✓" if shadow.get("agreement") else "✗"
            else:
                agree_str = "-"

            if not args.quiet:
                print(f"[{i+1:4d}] Prediction: {pred}  Latency: {latency:6.2f}ms  Shadow: {agree_str}")

        except Exception as e:
            print(f"[{i+1:4d}] Error: {e}")

        if args.delay > 0 and i < args.count - 1:
            time.sleep(args.delay)

    # Summary
    print(f"\n{'='*40}")
    print(f"Total predictions: {len(predictions)}")
    print(f"Average latency: {total_latency / len(predictions):.2f}ms" if predictions else "N/A")

    if shadow_count > 0:
        print(f"Shadow comparisons: {shadow_count}")
        print(f"Agreement rate: {agreements / shadow_count * 100:.1f}%")


if __name__ == "__main__":
    main()
