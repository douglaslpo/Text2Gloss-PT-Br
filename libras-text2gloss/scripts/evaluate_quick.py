#!/usr/bin/env python3
"""
Quick evaluation script to get real results for the paper.
Uses small models and limited data to generate empirical results.
"""

import os
import sys
import json
import torch
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.mt5_syntax_aware import MT5SyntaxAware, MT5Baseline
from utils.tokenizer_utils import MT5TokenizerWrapper
from utils.syntax_processor import PortugueseSyntaxProcessor
from utils.data_loader import LibrasDataLoader
from utils.metrics import TranslationMetrics

def load_test_data(data_path="data/processed/test.json"):
    """Load test data."""
    if not Path(data_path).exists():
        # Create sample data
        loader = LibrasDataLoader()
        data = loader.load_libras_csv()
        _, _, test_data = loader.split_data(data)
        return test_data[:20]  # Use 20 samples for quick evaluation
    
    with open(data_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def evaluate_model(model, tokenizer, test_data, syntax_processor=None, use_syntax=False, device='cpu'):
    """Evaluate a model on test data."""
    model.eval()
    model.to(device)
    
    predictions = []
    references = []
    
    print(f"\nEvaluating {'Syntax-Aware' if use_syntax else 'Baseline'} model...")
    
    with torch.no_grad():
        for item in tqdm(test_data[:20], desc="Generating"):  # Limit to 20 for speed
            source = item['source']
            target = item['target']
            
            # Tokenize source
            encoding = tokenizer.encode(
                source,
                max_length=64,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
                add_language_token='pt'
            )
            
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            
            # Get syntax if needed
            syntax_ids = None
            if use_syntax and syntax_processor:
                tokens, dep_tags = syntax_processor.process_sentence(source)
                subword_tokens = tokenizer.tokenize(source)
                aligned_tags = syntax_processor.align_with_subwords(tokens, dep_tags, subword_tokens)
                syntax_ids = torch.tensor([aligned_tags[:64] + [38] * (64 - len(aligned_tags[:64]))], dtype=torch.long).to(device)
            
            # Generate
            try:
                if use_syntax and syntax_ids is not None:
                    generated = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        syntax_ids=syntax_ids,
                        max_length=64,
                        num_beams=3,
                        early_stopping=True
                    )
                else:
                    generated = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_length=64,
                        num_beams=3,
                        early_stopping=True
                    )
                
                pred_text = tokenizer.decode(generated[0], skip_special_tokens=True)
                predictions.append(pred_text)
                references.append(target)
            except Exception as e:
                print(f"Error generating for '{source}': {e}")
                predictions.append("")
                references.append(target)
    
    return predictions, references

def main():
    print("=" * 80)
    print("Quick Evaluation for Paper Results")
    print("=" * 80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Load test data
    test_data = load_test_data()
    print(f"Loaded {len(test_data)} test samples (using first 20 for quick evaluation)\n")
    
    # Initialize tokenizer
    print("Loading tokenizer...")
    tokenizer = MT5TokenizerWrapper(model_name="google/mt5-small")
    
    # Initialize syntax processor
    print("Loading syntax processor...")
    syntax_processor = PortugueseSyntaxProcessor()
    
    # Initialize metrics
    metrics_calc = TranslationMetrics()
    
    results = {}
    
    # Evaluate Baseline
    print("\n" + "=" * 80)
    print("1. BASELINE MODEL")
    print("=" * 80)
    baseline_model = MT5Baseline(model_name="google/mt5-small")
    baseline_predictions, baseline_references = evaluate_model(
        baseline_model, tokenizer, test_data, use_syntax=False, device=device
    )
    baseline_metrics = metrics_calc.compute_all_metrics(
        baseline_predictions, baseline_references, measure_time=False
    )
    results['baseline'] = {
        'predictions': baseline_predictions,
        'references': baseline_references,
        'metrics': baseline_metrics
    }
    
    print("\nBaseline Results:")
    metrics_calc.print_metrics(baseline_metrics)
    
    # Evaluate Syntax-Aware
    print("\n" + "=" * 80)
    print("2. SYNTAX-AWARE MODEL")
    print("=" * 80)
    syntax_model = MT5SyntaxAware(model_name="google/mt5-small", num_syntax_tags=50)
    syntax_predictions, syntax_references = evaluate_model(
        syntax_model, tokenizer, test_data, syntax_processor, use_syntax=True, device=device
    )
    syntax_metrics = metrics_calc.compute_all_metrics(
        syntax_predictions, syntax_references, measure_time=False
    )
    results['syntax_aware'] = {
        'predictions': syntax_predictions,
        'references': syntax_references,
        'metrics': syntax_metrics
    }
    
    print("\nSyntax-Aware Results:")
    metrics_calc.print_metrics(syntax_metrics)
    
    # Comparison
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)
    print(f"\n{'Metric':<20} {'Baseline':<15} {'Syntax-Aware':<15} {'Improvement':<15}")
    print("-" * 80)
    print(f"{'sacreBLEU':<20} {baseline_metrics['bleu']:<15.2f} {syntax_metrics['bleu']:<15.2f} {syntax_metrics['bleu'] - baseline_metrics['bleu']:+.2f}")
    print(f"{'TER':<20} {baseline_metrics['ter']:<15.2f} {syntax_metrics['ter']:<15.2f} {syntax_metrics['ter'] - baseline_metrics['ter']:+.2f}")
    print(f"{'ROUGE-L F1':<20} {baseline_metrics['rouge_l_f1']:<15.4f} {syntax_metrics['rouge_l_f1']:<15.4f} {syntax_metrics['rouge_l_f1'] - baseline_metrics['rouge_l_f1']:+.4f}")
    print(f"{'METEOR':<20} {baseline_metrics['meteor']:<15.4f} {syntax_metrics['meteor']:<15.4f} {syntax_metrics['meteor'] - baseline_metrics['meteor']:+.4f}")
    
    # Save results
    output_file = "results/evaluation_results.json"
    Path("results").mkdir(exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ Results saved to {output_file}")
    
    # Print examples
    print("\n" + "=" * 80)
    print("EXAMPLE TRANSLATIONS")
    print("=" * 80)
    for i in range(min(5, len(test_data))):
        print(f"\nExample {i+1}:")
        print(f"  PT-BR: {test_data[i]['source']}")
        print(f"  LIBRAS (Ref): {test_data[i]['target']}")
        print(f"  Baseline: {baseline_predictions[i]}")
        print(f"  Syntax-Aware: {syntax_predictions[i]}")

if __name__ == "__main__":
    main()

