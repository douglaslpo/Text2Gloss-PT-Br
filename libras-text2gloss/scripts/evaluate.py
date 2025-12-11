#!/usr/bin/env python3
"""
Evaluation script for comparing syntax-aware and baseline models.

This script evaluates both models on the test set and compares their performance.
"""

import os
import sys
import json
import yaml
import torch
import argparse
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.mt5_syntax_aware import MT5SyntaxAware, MT5Baseline
from utils.tokenizer_utils import MT5TokenizerWrapper
from utils.syntax_processor import PortugueseSyntaxProcessor
from utils.data_loader import LibrasDataLoader, create_dataloaders
from utils.metrics import TranslationMetrics


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate models")
    parser.add_argument(
        "--model_syntax",
        type=str,
        required=True,
        help="Path to syntax-aware model checkpoint"
    )
    parser.add_argument(
        "--model_baseline",
        type=str,
        default=None,
        help="Path to baseline model checkpoint (optional)"
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default="data/processed/test.json",
        help="Path to test data"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/evaluation.json",
        help="Output file for results"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for evaluation"
    )
    
    return parser.parse_args()


def evaluate_model(model, test_loader, tokenizer, device, use_syntax=True):
    """Evaluate a single model."""
    model.eval()
    all_predictions = []
    all_references = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Generating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            syntax_ids = None
            if use_syntax and 'syntax_ids' in batch:
                syntax_ids = batch['syntax_ids'].to(device)
            
            # Generate predictions
            if use_syntax and syntax_ids is not None:
                generated_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    syntax_ids=syntax_ids,
                    max_length=128,
                    num_beams=5,
                    early_stopping=True
                )
            else:
                generated_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=128,
                    num_beams=5,
                    early_stopping=True
                )
            
            # Decode
            predictions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            references = tokenizer.batch_decode(labels, skip_special_tokens=True)
            
            all_predictions.extend(predictions)
            all_references.extend(references)
    
    return all_predictions, all_references


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = MT5TokenizerWrapper.from_pretrained(args.model_syntax)
    
    # Load syntax processor
    print("Loading syntax processor...")
    syntax_processor = PortugueseSyntaxProcessor()
    
    # Load test data
    print("Loading test data...")
    with open(args.test_data, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    print(f"Test set size: {len(test_data)}\n")
    
    # Create test dataloader
    from torch.utils.data import DataLoader
    from utils.data_loader import LibrasDataset
    
    # Initialize metrics
    metrics_calculator = TranslationMetrics()
    
    results = {}
    
    # Evaluate syntax-aware model
    print("=" * 60)
    print("Evaluating Syntax-Aware Model")
    print("=" * 60)
    
    print(f"Loading model from {args.model_syntax}...")
    model_syntax = MT5SyntaxAware.from_pretrained(args.model_syntax)
    model_syntax = model_syntax.to(device)
    
    test_dataset_syntax = LibrasDataset(
        test_data, tokenizer, syntax_processor, use_syntax=True
    )
    test_loader_syntax = DataLoader(
        test_dataset_syntax, batch_size=args.batch_size, shuffle=False
    )
    
    predictions_syntax, references = evaluate_model(
        model_syntax, test_loader_syntax, tokenizer, device, use_syntax=True
    )
    
    metrics_syntax = metrics_calculator.compute_all_metrics(
        predictions_syntax, references
    )
    
    print("\nSyntax-Aware Model Results:")
    metrics_calculator.print_metrics(metrics_syntax)
    
    results['syntax_aware'] = metrics_syntax
    
    # Evaluate baseline model if provided
    if args.model_baseline:
        print("\n" + "=" * 60)
        print("Evaluating Baseline Model")
        print("=" * 60)
        
        print(f"Loading model from {args.model_baseline}...")
        model_baseline = MT5Baseline.from_pretrained(args.model_baseline)
        model_baseline = model_baseline.to(device)
        
        test_dataset_baseline = LibrasDataset(
            test_data, tokenizer, None, use_syntax=False
        )
        test_loader_baseline = DataLoader(
            test_dataset_baseline, batch_size=args.batch_size, shuffle=False
        )
        
        predictions_baseline, _ = evaluate_model(
            model_baseline, test_loader_baseline, tokenizer, device, use_syntax=False
        )
        
        metrics_baseline = metrics_calculator.compute_all_metrics(
            predictions_baseline, references
        )
        
        print("\nBaseline Model Results:")
        metrics_calculator.print_metrics(metrics_baseline)
        
        results['baseline'] = metrics_baseline
        
        # Print comparison
        print("\n" + "=" * 60)
        print("COMPARISON")
        print("=" * 60)
        print(f"\n{'Metric':<20} {'Syntax-Aware':<15} {'Baseline':<15} {'Improvement':<15}")
        print("-" * 65)
        
        for metric in ['bleu', 'ter', 'rouge_l_f1', 'meteor']:
            syntax_val = metrics_syntax.get(metric, 0.0)
            baseline_val = metrics_baseline.get(metric, 0.0)
            
            if metric == 'ter':
                # Lower is better for TER
                improvement = baseline_val - syntax_val
                improvement_str = f"{improvement:+.2f}"
            else:
                improvement = syntax_val - baseline_val
                improvement_str = f"{improvement:+.2f}"
            
            print(f"{metric:<20} {syntax_val:<15.2f} {baseline_val:<15.2f} {improvement_str:<15}")
        
        print("=" * 65 + "\n")
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to {output_path}")
    
    # Save sample predictions
    samples_path = output_path.parent / "sample_predictions.txt"
    with open(samples_path, 'w', encoding='utf-8') as f:
        f.write("SAMPLE PREDICTIONS\n")
        f.write("=" * 80 + "\n\n")
        
        for i in range(min(10, len(references))):
            f.write(f"Example {i+1}:\n")
            f.write(f"  Reference:        {references[i]}\n")
            f.write(f"  Syntax-Aware:     {predictions_syntax[i]}\n")
            if args.model_baseline:
                f.write(f"  Baseline:         {predictions_baseline[i]}\n")
            f.write("\n")
    
    print(f"Sample predictions saved to {samples_path}")


if __name__ == "__main__":
    main()
