# Hotswap ML System - Usage Guide

A hotswappable ML model system with shadow mode validation for zero-downtime model updates.

## Quick Start

```bash
# Activate virtual environment (required for all commands)
source .venv/bin/activate
```

---

## 1. Reset & Clean State

Before testing, reset to a clean state:

```bash
# Remove all data, models, and registry
./scripts/reset.sh
```

---

## 2. Standalone Workflow Test

Test the complete workflow without running a server:

```bash
python scripts/test_workflow.py
```

This tests:
- Component initialization
- Data generation
- Model training
- Shadow mode comparison
- Model promotion

You can run specific tests:

```bash
# Run only the promotion test
python scripts/test_workflow.py --test promotion

# Run only the rollback test
python scripts/test_workflow.py --test rollback

# Run both tests (default)
python scripts/test_workflow.py --test all
```

---

## 3. Testing Rollback & Archival

The system automatically archives models that fail validation. To test this:

### Automated Rollback Test

```bash
python scripts/test_workflow.py --test rollback
```

This test:
1. Creates training data with correct labels
2. Creates "bad" data with inverted labels (0→9, 1→8, etc.)
3. Trains a good model and activates it
4. Trains a bad model on inverted labels
5. Starts shadow mode with the bad model
6. Verifies the bad model gets **ARCHIVED** (not promoted)
7. Verifies the good model stays **ACTIVE**

### Manual Rollback Testing

#### Generate Bad Training Data

```bash
# Inverted labels (0→9, 1→8, etc.) - recommended for testing rollback
python scripts/generate_bad_data.py --output ./data/bad.pt --count 500 --type inverted

# Shifted labels (each label +5 mod 10)
python scripts/generate_bad_data.py --output ./data/bad.pt --count 500 --type shifted --shift 5

# Randomly shuffled labels
python scripts/generate_bad_data.py --output ./data/bad.pt --count 500 --type mislabeled

# Pure random noise images
python scripts/generate_bad_data.py --output ./data/bad.pt --count 500 --type noise

# All labels set to same value (model learns nothing)
python scripts/generate_bad_data.py --output ./data/bad.pt --count 500 --type constant
```

#### Test Rollback with Server Running

```bash
# Terminal 1: Start server
python -m hotswap serve --port 8000

# Terminal 2: Train a good model first
python -m hotswap generate-data --output ./data/good.pt --count 500
python -m hotswap train --data ./data/good.pt --epochs 5
# Wait for training, then activate the model
python -m hotswap models  # Get the model ID
python -m hotswap swap --model-id <good-model-id>

# Now train a bad model
python scripts/generate_bad_data.py --output ./data/bad.pt --count 500 --type inverted
python -m hotswap train --data ./data/bad.pt --epochs 5
# Wait for training
python -m hotswap models  # Get the bad model ID

# Start shadow mode with the bad model
python -m hotswap shadow --start <bad-model-id>

# The bad model should be automatically archived
python -m hotswap models  # Check status - bad model should be ARCHIVED
```

### How Rollback Decision Works

When shadow mode starts, the system runs validation:
- Compares accuracy on a fixed 100-sample validation set
- **Promote**: Shadow accuracy ≥ (Active accuracy - 5%)
- **Rollback**: Shadow accuracy < (Active accuracy - 10%)
- **Manual**: In between - requires human decision

A model trained on inverted labels learns the wrong patterns, so its validation accuracy is much lower than the good model, triggering automatic archival.

---

## 4. Interactive Demo (with Server)

### Terminal 1: Start the Server

```bash
python -m hotswap serve --port 8000 --watch ./data
```

### Terminal 2: Run the Demo

```bash
# Promotion demo - trains two good models, shows auto-promotion
python scripts/demo.py --demo promotion

# Rollback demo - trains good model, then bad model, shows auto-archival
python scripts/demo.py --demo rollback

# Run both demos
python scripts/demo.py --demo all

# Fully automated (no prompts)
python scripts/demo.py --demo promotion --auto
python scripts/demo.py --demo rollback --auto
python scripts/demo.py --demo all --auto
```

### Terminal 3: Open Dashboard

```bash
xdg-open http://localhost:8000/dashboard
# Or navigate to: http://localhost:8000/dashboard
```

---

## 5. Manual CLI Testing

### Generate Training Data

```bash
python -m hotswap generate-data --output ./data/batch1.pt --count 1000
python -m hotswap generate-data --output ./data/batch2.pt --count 500 --noise 0.2
```

### Train a Model Manually

```bash
python -m hotswap train --data ./data/batch1.pt --epochs 5
```

### Check System Status

```bash
python -m hotswap status
```

### List Registered Models

```bash
python -m hotswap models
python -m hotswap models --status ready
python -m hotswap models --status active
```

### Shadow Mode Management

```bash
# View shadow metrics
python -m hotswap shadow --metrics

# Start shadow mode with a specific model
python -m hotswap shadow --start <model-id>

# Promote shadow model to active
python -m hotswap shadow --promote

# Cancel shadow mode
python -m hotswap shadow --cancel
```

### Force Swap to a Model

```bash
python -m hotswap swap --model-id <model-id>
```

---

## 6. API Testing

### Health Check

```bash
curl http://localhost:8000/api/health
```

### System Status

```bash
curl http://localhost:8000/api/status
```

### List Models

```bash
curl http://localhost:8000/api/models
```

### Run Prediction

```bash
python -c "
import httpx
import torch

data = torch.randn(1, 28, 28).tolist()
response = httpx.post('http://localhost:8000/api/predict', json={'data': data})
print(response.json())
"
```

### Shadow Mode via API

```bash
# Get shadow metrics
curl http://localhost:8000/api/shadow/metrics

# Start shadow mode
curl -X POST http://localhost:8000/api/shadow/start \
  -H "Content-Type: application/json" \
  -d '{"model_id": "<model-id>"}'

# Promote shadow model
curl -X POST http://localhost:8000/api/shadow/promote

# Cancel shadow mode
curl -X POST http://localhost:8000/api/shadow/cancel
```

### Activate a Model

```bash
curl -X POST http://localhost:8000/api/models/<model-id>/activate
```

### Trigger Training via API

```bash
curl -X POST http://localhost:8000/api/train \
  -H "Content-Type: application/json" \
  -d '{"data_path": "./data/batch1.pt", "epochs": 3}'
```

---

## 7. Unit & Integration Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run only unit tests
python -m pytest tests/unit/ -v

# Run only integration tests
python -m pytest tests/integration/ -v

# Run specific test file
python -m pytest tests/unit/test_inference_engine.py -v

# Run with coverage (requires pytest-cov)
python -m pytest tests/ --cov=hotswap --cov-report=html
```

---

## 8. File Watcher Demo

The server automatically watches the `./data/` directory for new `.pt` files:

```bash
# Terminal 1: Start server with watcher
python -m hotswap serve --port 8000 --watch ./data

# Terminal 2: Drop data files to trigger training
python -m hotswap generate-data --output ./data/new_batch.pt --count 500

# Watch the server logs - training starts automatically!
# After training, shadow mode begins if there's an active model
```

---

## 9. Complete End-to-End Workflow

```bash
# 1. Reset everything
./scripts/reset.sh

# 2. Start server
python -m hotswap serve --port 8000 --watch ./data &

# 3. Wait for server to start
sleep 2

# 4. Generate first batch - creates initial model
python -m hotswap generate-data --output ./data/batch1.pt --count 1000

# 5. Wait for training (check status)
sleep 10
python -m hotswap status

# 6. Generate second batch - triggers shadow mode
python -m hotswap generate-data --output ./data/batch2.pt --count 1000

# 7. Wait and check shadow metrics
sleep 10
python -m hotswap shadow --metrics

# 8. Run predictions to collect comparison data
python -c "
import httpx
import torch
for i in range(50):
    data = torch.randn(1, 28, 28).tolist()
    httpx.post('http://localhost:8000/api/predict', json={'data': data})
print('Done')
"

# 9. Check final status
python -m hotswap shadow --metrics
python -m hotswap models

# 10. Promote manually if needed
python -m hotswap shadow --promote
```

---

## 10. Dashboard Features

Open `http://localhost:8000/dashboard` to see:

- **Active Model Card**: Current production model version and prediction count
- **Shadow Model Card**: Shadow model status with promote/cancel buttons
- **Training Card**: Real-time training progress
- **Shadow Metrics**: Agreement rate, latency comparison, decision recommendation
- **Model History**: All registered models with version, status, and activation buttons
- **Manual Controls**: Start shadow mode with any ready model

The dashboard updates in real-time via WebSocket.

---

## 11. Server Options

```bash
python -m hotswap serve --help

Options:
  --port, -p      Server port (default: 8000)
  --host, -h      Server host (default: 0.0.0.0)
  --watch, -w     Directory to watch for new data
  --no-watch      Disable file watcher
  --reload, -r    Enable auto-reload for development
```

---

## Project Structure

```
hotswap-test/
├── hotswap/              # Main package
│   ├── core/             # InferenceEngine, Registry, Shadow
│   ├── workers/          # Watcher, Trainer, SwapCoordinator
│   ├── models/           # MNISTClassifier
│   ├── api/              # FastAPI routes
│   └── utils/            # Config, synthetic data
├── dashboard/            # Web UI (HTML/CSS/JS)
├── scripts/              # Demo and test scripts
│   ├── reset.sh          # Clean all data
│   ├── demo.py           # Interactive demo
│   └── test_workflow.py  # Standalone test
├── tests/                # pytest tests
├── data/                 # Training data (watched)
└── models/               # Checkpoints and registry
```

---

## Troubleshooting

**ModuleNotFoundError**: Activate the venv first
```bash
source .venv/bin/activate
```

**Server not responding**: Check if port is in use
```bash
lsof -i :8000
```

**Training stuck**: Check server logs for errors

**Shadow mode not starting**: Ensure there's an active model first
