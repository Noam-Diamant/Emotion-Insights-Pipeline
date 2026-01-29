# Emotion Detection Pipeline - Replication Run

Self-contained environment for training and evaluating Transformer-based emotion detection models.

## Overview

The `project_part_b.py` script trains three Transformer models (BERT, ELECTRA, RoBERTa) for 6-class emotion classification with hyperparameter search and model compression.

**Modes:**
- **Training Mode** (default): Full hyperparameter search, model comparison, and compression
- **Inference Mode**: Run predictions using a pre-trained model

---

## Quick Start

### Environment Setup

**Prerequisites:**

Install `uv` (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Verify GPU access:
```bash
nvidia-smi
```

**Setup:**

Run the automated setup script:
```bash
cd /dsi/fetaya-lab/noam_diamant/projects/Unlearning_with_SAE/Emotion-Insights-Pipeline/replication_run
./setup_env.sh
```

This creates a virtual environment (`.venv`), installs PyTorch with CUDA 12.1, and all dependencies.

**Activate and run:**
```bash
source .venv/bin/activate
python project_part_b.py
```

**Verify Installation:**

Check CUDA is working:
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

Expected output:
```
CUDA available: True
Device: NVIDIA TITAN RTX
```

Monitor GPU usage during training:
```bash
watch -n 1 nvidia-smi
```

### Training Mode

```bash
python project_part_b.py
```

**Requirements:** `data/train.csv` and `data/validation.csv` with columns:
- `text`: Raw text for classification
- `label`: Integer (0-5) for emotions: sadness, joy, love, anger, fear, surprise

**Output:** All console logs saved to `results/pipeline_output.txt` automatically.

### Inference Mode

Edit `project_part_b.py`:
```python
RUN_INFERENCE_ONLY = True
TEST_FILE = "./data/test.csv"
BEST_MODEL_WEIGHTS = "./hp_models/bert_batch_size16_dropout_rate0.1_lr2e-05_weight_decay0.0.pt"
```

Then run:
```bash
python project_part_b.py
```

**Output:** `data/predictions.csv` with predicted labels.

---

## What It Does

### Training Pipeline

1. **Load & Preprocess**: Clean text (lowercase, remove URLs/HTML/contractions)
2. **Hyperparameter Search**: Grid search over 16 configs per model (48 total)
   - Parameters: `dropout_rate`, `learning_rate`, `batch_size`, `weight_decay`
   - Models: BERT, ELECTRA, RoBERTa
3. **Compression**: Apply pruning (30%) and quantization (int8) to best model
4. **Results**: Generate comparison tables, plots, confusion matrices

### Training Details

- **Optimizer**: AdamW with class-weighted CrossEntropyLoss
- **Early Stopping**: 3 epochs patience on validation loss
- **Max Epochs**: Configurable via `NUM_EPOCHS` parameter (default: 30)
- **Architecture**: Frozen Transformer + Classification head (hidden → 128 → 6)

---

## Configuration

Edit these parameters in `project_part_b.py`:

```python
# Data paths
TRAIN_FILE = "./data/train.csv"
VALIDATION_FILE = "./data/validation.csv"

# Output folders
RESULTS_FOLDER = "./results"
SAVE_MODELS_FOLDER = "./hp_models"

# Hyperparameter grid
PARAM_GRID = {
    "dropout_rate": [0.1, 0.3],
    "lr": [2e-5, 3e-5],
    "batch_size": [16, 32],
    "weight_decay": [0.0, 0.01],
}

# Settings
MAX_LENGTH = 128          # Maximum sequence length
NUM_CLASSES = 6           # Number of emotion classes
NUM_EPOCHS = 30           # Maximum epochs per model configuration
```

**Quick test** with fewer models:
```python
hp_summary_bert = hyperparameter_search(
    ...
    max_models=2,  # Train only 2 configs
)
```

---

## Output Structure

### `./results/` folder:
- `pipeline_output.txt`: Complete console output (auto-captured)
- `model_comparison.json` / `.csv`: All model comparisons
- `hp_results_<model>.json`: Hyperparameter search results
- `compressions_results_<model>.json`: Compression results
- `*_best_model_report.txt`: Classification reports
- `*.png`: Training plots and confusion matrices

### `./hp_models/` folder:
- `<model>_<params>.pt`: Trained model weights
- `prune_<model>_<params>.pt`: Pruned models
- `quantize_<model>_<params>.pt`: Quantized models

---

## Performance

- **Runtime**: ~6-12 hours with GPU for full pipeline (48 configs + compression)
- **GPU**: Automatically detected and used (CUDA required)
- **Fallback**: CPU if no GPU (not recommended)

---

## Dependencies

| Package | Purpose |
|---------|---------|
| torch | Deep learning framework with CUDA |
| transformers | BERT, ELECTRA, RoBERTa models |
| numpy | Numerical computing |
| pandas | Data manipulation |
| scikit-learn | Metrics and evaluation |
| matplotlib | Plotting and visualization |

---

## Troubleshooting

### CUDA not available

**Check PyTorch CUDA version:**
```bash
python -c "import torch; print(torch.version.cuda)"
```

**If `None`**, reinstall PyTorch with CUDA:
```bash
uv pip uninstall torch
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### File not found errors
Run from the correct directory:
```bash
cd /path/to/Emotion-Insights-Pipeline/replication_run
python project_part_b.py
```

### Out of GPU memory
Reduce batch size in `PARAM_GRID`:
```python
"batch_size": [8, 16],  # Instead of [16, 32]
```

### Virtual environment not activated
Make sure to activate:
```bash
source .venv/bin/activate
which python  # Should show: .../replication_run/.venv/bin/python
```

### Inference errors
- Don't rename model weight files (script extracts config from filename)
- Verify test CSV format matches train/validation
- Check `NUM_CLASSES = 6` matches your model

---

## Clean Up

Remove the environment:
```bash
deactivate
rm -rf .venv
```

Start fresh:
```bash
rm -rf .venv
./setup_env.sh
```

---

## Folder Structure

```
replication_run/
├── README.md              # This file
├── setup_env.sh           # Automated setup script
├── project_part_b.py      # Main script
├── data/                  # Input data (create this)
│   ├── train.csv
│   ├── validation.csv
│   └── test.csv
├── results/               # Generated outputs
└── hp_models/             # Saved model weights
```

---

## Notes

- All outputs are isolated from parent directory
- Output capture is automatic (no need for manual redirection)
- Model filenames encode hyperparameters (don't rename)
- Pruned models are retrained; quantized models are not
