#!/usr/bin/env python3
"""
Demo script to showcase the PT-BR to LIBRAS translation system.

This script demonstrates:
1. Data preprocessing
2. Model training (debug mode)
3. Evaluation
4. Inference examples
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 80)
print("PT-BR to LIBRAS Text2Gloss Translation System - DEMO")
print("=" * 80)
print()

# Step 1: Data Preprocessing
print("STEP 1: Data Preprocessing")
print("-" * 80)
print("Preprocessing sample LIBRAS data...")
print()

from utils.data_loader import LibrasDataLoader

loader = LibrasDataLoader()
data = loader.load_libras_csv()

print(f"✓ Loaded {len(data)} samples")
print(f"\nSample data:")
for i in range(3):
    print(f"  {i+1}. PT-BR: {data[i]['source']}")
    print(f"     LIBRAS: {data[i]['target']}\n")

# Split data
train_data, dev_data, test_data = loader.split_data(data)
loader.save_processed_data(train_data, dev_data, test_data)

print(f"✓ Data split: Train={len(train_data)}, Dev={len(dev_data)}, Test={len(test_data)}")
print()

# Step 2: Syntax Processing Demo
print("\nSTEP 2: Syntax Processing Demo")
print("-" * 80)
print("Extracting syntactic features from Portuguese...")
print()

from utils.syntax_processor import PortugueseSyntaxProcessor

processor = PortugueseSyntaxProcessor()
sample_text = "O menino está jogando bola no parque."

tokens, dep_tags = processor.process_sentence(sample_text)

print(f"Input: {sample_text}")
print(f"Tokens: {tokens}")

# Get tag names
reverse_mapper = {v: k for k, v in processor.tag_mapper.items()}
tag_names = [reverse_mapper[tag_id] for tag_id in dep_tags]
print(f"Syntax Tags: {tag_names}")
print()

# Step 3: Tokenization Demo
print("\nSTEP 3: Tokenization Demo")
print("-" * 80)
print("Tokenizing with mT5...")
print()

from utils.tokenizer_utils import MT5TokenizerWrapper

tokenizer = MT5TokenizerWrapper(model_name="google/mt5-small")

pt_text = "Bom dia, como você está?"
encoding = tokenizer.encode(pt_text, add_language_token='pt', max_length=32)

print(f"Input: {pt_text}")
print(f"Token IDs shape: {encoding['input_ids'].shape}")
print(f"Subword tokens: {tokenizer.tokenize(pt_text)}")
print()

# Step 4: Metrics Demo
print("\nSTEP 4: Metrics Evaluation Demo")
print("-" * 80)
print("Computing translation metrics...")
print()

from utils.metrics import TranslationMetrics

metrics_calc = TranslationMetrics()

predictions = [
    "MENINO JOGAR BOLA PARQUE",
    "PROFESSORA ENSINAR MATEMATICA",
]

references = [
    "MENINO JOGAR BOLA PARQUE",
    "PROFESSORA ENSINAR MATEMATICA ALUNO",
]

metrics = metrics_calc.compute_all_metrics(predictions, references, measure_time=False)

print(f"sacreBLEU: {metrics['bleu']:.2f}")
print(f"TER: {metrics['ter']:.2f}")
print(f"ROUGE-L F1: {metrics['rouge_l_f1']:.4f}")
print(f"METEOR: {metrics['meteor']:.4f}")
print()

# Step 5: Model Architecture Demo
print("\nSTEP 5: Model Architecture Demo")
print("-" * 80)
print("Initializing mT5 models...")
print()

from models.mt5_syntax_aware import MT5SyntaxAware, MT5Baseline
import torch

# Initialize models
print("Loading Syntax-Aware Model...")
model_syntax = MT5SyntaxAware(model_name="google/mt5-small", num_syntax_tags=50)
num_params_syntax = sum(p.numel() for p in model_syntax.parameters()) / 1e6
print(f"✓ Syntax-Aware Model: {num_params_syntax:.1f}M parameters")

print("\nLoading Baseline Model...")
model_baseline = MT5Baseline(model_name="google/mt5-small")
num_params_baseline = sum(p.numel() for p in model_baseline.parameters()) / 1e6
print(f"✓ Baseline Model: {num_params_baseline:.1f}M parameters")
print()

# Step 6: Forward Pass Demo
print("\nSTEP 6: Forward Pass Demo")
print("-" * 80)
print("Testing model forward pass...")
print()

batch_size, seq_len = 2, 10
input_ids = torch.randint(0, 1000, (batch_size, seq_len))
syntax_ids = torch.randint(0, 50, (batch_size, seq_len))
labels = torch.randint(0, 1000, (batch_size, seq_len))

outputs = model_syntax(input_ids=input_ids, syntax_ids=syntax_ids, labels=labels)

print(f"Input shape: {input_ids.shape}")
print(f"Syntax IDs shape: {syntax_ids.shape}")
print(f"Output logits shape: {outputs.logits.shape}")
print(f"Loss: {outputs.loss.item():.4f}")
print()

# Step 7: Back-Translation Demo
print("\nSTEP 7: Back-Translation Augmentation Demo")
print("-" * 80)
print("Demonstrating data augmentation...")
print()

from utils.back_translation import BackTranslationAugmenter

augmenter = BackTranslationAugmenter(source_lang="pt", pivot_lang="en")

original_text = "O menino joga bola no parque."
back_translated = augmenter.back_translate(original_text)

print(f"Original: {original_text}")
print(f"Back-translated: {back_translated}")
print()

# Summary
print("\n" + "=" * 80)
print("DEMO COMPLETED SUCCESSFULLY!")
print("=" * 80)
print()
print("Summary of Demonstrated Features:")
print("  ✓ Data loading and preprocessing")
print("  ✓ Syntax extraction with spaCy (Portuguese)")
print("  ✓ mT5 tokenization with special tokens")
print("  ✓ Evaluation metrics (BLEU, TER, ROUGE-L, METEOR)")
print("  ✓ Syntax-aware mT5 model architecture")
print("  ✓ Baseline mT5 model for comparison")
print("  ✓ Back-translation data augmentation")
print()
print("Next Steps:")
print("  1. Train models: python scripts/train_mt5_syntax.py --debug")
print("  2. Evaluate: python scripts/evaluate.py --model_syntax <path>")
print("  3. Inference: python scripts/inference.py --model <path> --interactive")
print()
print("=" * 80)
