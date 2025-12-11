# PT-BR to LIBRAS Text2Gloss - Quick Start Guide

## Installation

```bash
# Navigate to project directory
cd libras-text2gloss

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Download Portuguese spaCy model
python -m spacy download pt_core_news_lg
```

## Data Preparation

### Option 1: Use Sample Data (for testing)

The system will automatically generate sample data if no dataset is found.

```bash
python scripts/preprocess_data.py
```

### Option 2: Use Real LIBRAS Dataset

1. Download VLibrasBD dataset
2. Place CSV file in `data/raw/vlibrasbd.csv`
3. Run preprocessing:

```bash
python scripts/preprocess_data.py \
  --input data/raw/vlibrasbd.csv \
  --output data/processed/ \
  --split 0.8 0.1 0.1
```

## Training

### Train Syntax-Aware Model

```bash
python scripts/train_mt5_syntax.py \
  --config configs/training_config.yaml \
  --model_config configs/model_config.yaml \
  --output models/checkpoints/syntax_aware
```

### Debug Mode (small dataset)

```bash
python scripts/train_mt5_syntax.py --debug
```

## Evaluation

```bash
python scripts/evaluate.py \
  --model_syntax models/checkpoints/syntax_aware/best_model \
  --test_data data/processed/test.json \
  --output results/evaluation.json
```

## Inference

### Interactive Mode

```bash
python scripts/inference.py \
  --model models/checkpoints/syntax_aware/best_model \
  --interactive
```

### Single Translation

```bash
python scripts/inference.py \
  --model models/checkpoints/syntax_aware/best_model \
  --text "Bom dia, como você está?"
```

## Project Structure

```
libras-text2gloss/
├── configs/              # Configuration files
├── data/                 # Datasets
│   ├── raw/             # Raw data
│   └── processed/       # Preprocessed data
├── models/              # Model implementations
│   └── checkpoints/     # Trained models
├── scripts/             # Training/evaluation scripts
├── utils/               # Utility modules
└── notebooks/           # Jupyter notebooks
```

## Configuration

Edit `configs/training_config.yaml` to customize:
- Learning rate
- Batch size
- Number of epochs
- Early stopping patience

Edit `configs/model_config.yaml` to customize:
- Model size (mt5-small, mt5-base, mt5-large)
- Syntax features
- Generation parameters

## Troubleshooting

### CUDA Out of Memory

Reduce batch size in `configs/training_config.yaml`:
```yaml
batch:
  train_batch_size: 8  # Reduce from 16
  gradient_accumulation_steps: 8  # Increase to maintain effective batch size
```

### spaCy Model Not Found

```bash
python -m spacy download pt_core_news_lg
```

### Dataset Not Found

The system will use sample data automatically. To use real data, ensure the CSV file is in the correct location.

## Next Steps

1. Train baseline model for comparison
2. Experiment with different model sizes
3. Try multilingual pre-training
4. Analyze results in notebooks
