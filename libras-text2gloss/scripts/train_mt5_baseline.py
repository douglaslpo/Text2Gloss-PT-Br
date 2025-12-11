#!/usr/bin/env python3
"""
Training script for baseline mT5 model (without syntax).

This script trains the standard mT5 model without syntax-aware embeddings
for comparison with the syntax-aware version.
"""

import os
import sys
import yaml
import torch
import wandb
from pathlib import Path
from tqdm import tqdm
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.mt5_syntax_aware import MT5Baseline
from utils.tokenizer_utils import MT5TokenizerWrapper
from utils.data_loader import LibrasDataLoader, create_dataloaders
from utils.metrics import TranslationMetrics


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train baseline mT5 model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_config.yaml",
        help="Path to training config"
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default="configs/model_config.yaml",
        help="Path to model config"
    )
    parser.add_argument(
        "--data_config",
        type=str,
        default="configs/data_config.yaml",
        help="Path to data config"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/checkpoints/baseline",
        help="Output directory for checkpoints"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode with small dataset"
    )
    
    return parser.parse_args()


def load_configs(args):
    """Load all configuration files."""
    with open(args.config, 'r') as f:
        train_config = yaml.safe_load(f)
    
    with open(args.model_config, 'r') as f:
        model_config = yaml.safe_load(f)
    
    with open(args.data_config, 'r') as f:
        data_config = yaml.safe_load(f)
    
    return train_config, model_config, data_config


def setup_wandb(train_config, run_name="baseline"):
    """Initialize Weights & Biases logging."""
    # Check if wandb is enabled and if report_to includes 'wandb'
    report_to = train_config.get('logging', {}).get('report_to', [])
    if 'wandb' in report_to and 'wandb' in train_config.get('logging', {}):
        try:
            wandb_config = train_config['logging']['wandb']
            wandb.init(
                project=wandb_config['project'],
                entity=wandb_config.get('entity'),
                name=f"{run_name}_{wandb_config.get('name', '')}",
                tags=wandb_config.get('tags', []) + ['baseline'],
                config=train_config,
                mode='offline'  # Use offline mode if no API key
            )
        except Exception as e:
            print(f"⚠️  WandB não disponível: {e}. Continuando sem logging.")
            # Disable wandb in config
            if 'report_to' in train_config.get('logging', {}):
                train_config['logging']['report_to'] = [r for r in report_to if r != 'wandb']


def train_epoch(model, train_loader, optimizer, scheduler, device, epoch, config):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = len(train_loader)
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(progress_bar):
        # Move to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass (no syntax)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        
        # Gradient accumulation
        if (batch_idx + 1) % config['batch']['gradient_accumulation_steps'] == 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config['batch']['max_grad_norm']
            )
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})
        
        # Log to wandb
        if 'wandb' in config['logging']['report_to']:
            wandb.log({
                'train/loss': loss.item(),
                'train/learning_rate': scheduler.get_last_lr()[0]
            })
    
    avg_loss = total_loss / num_batches
    return avg_loss


def evaluate(model, eval_loader, tokenizer, device, metrics_calculator):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_references = []
    
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Compute loss
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            total_loss += outputs.loss.item()
            
            # Generate predictions
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=128,
                num_beams=5
            )
            
            # Decode
            predictions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            references = tokenizer.batch_decode(labels, skip_special_tokens=True)
            
            all_predictions.extend(predictions)
            all_references.extend(references)
    
    # Compute metrics
    metrics = metrics_calculator.compute_all_metrics(
        all_predictions,
        all_references,
        measure_time=False
    )
    
    metrics['eval_loss'] = total_loss / len(eval_loader)
    
    return metrics


def main():
    """Main training function."""
    args = parse_args()
    
    # Load configs
    train_config, model_config, data_config = load_configs(args)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize wandb
    if not args.debug:
        setup_wandb(train_config, run_name="baseline")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = MT5TokenizerWrapper(model_name=model_config['model']['name'])
    
    # Load data
    print("Loading data...")
    data_loader = LibrasDataLoader(config_path=args.data_config)
    
    # Check if processed data exists
    processed_dir = Path(data_config['output']['processed_dir'])
    if (processed_dir / "train.json").exists():
        print("Loading preprocessed data...")
        train_data, dev_data, test_data = data_loader.load_processed_data()
    else:
        print("Processing raw data...")
        raw_data = data_loader.load_libras_csv()
        train_data, dev_data, test_data = data_loader.split_data(raw_data)
        data_loader.save_processed_data(train_data, dev_data, test_data)
    
    # Debug mode
    if args.debug:
        train_data = train_data[:100]
        dev_data = dev_data[:50]
        print(f"Debug mode: using {len(train_data)} train, {len(dev_data)} dev samples")
    
    # Create dataloaders (without syntax)
    print("Creating dataloaders...")
    train_loader, dev_loader, test_loader = create_dataloaders(
        train_data,
        dev_data,
        test_data,
        tokenizer,
        syntax_processor=None,
        batch_size=train_config['batch']['train_batch_size'],
        use_syntax=False  # No syntax for baseline
    )
    
    # Initialize baseline model
    print("Initializing baseline model...")
    model = MT5Baseline(model_name=model_config['model']['name'])
    
    # Resize token embeddings
    model.resize_token_embeddings(tokenizer.vocab_size)
    model = model.to(device)
    
    print(f"Model has {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config['optimizer']['learning_rate'],
        weight_decay=train_config['optimizer']['weight_decay']
    )
    
    # Setup scheduler
    num_training_steps = len(train_loader) * train_config['training']['phase2_libras']['epochs']
    num_warmup_steps = train_config['scheduler']['warmup_steps']
    
    from transformers import get_linear_schedule_with_warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Initialize metrics
    metrics_calculator = TranslationMetrics()
    
    # Training loop
    best_bleu = 0.0
    patience_counter = 0
    max_patience = train_config['training']['phase2_libras']['early_stopping_patience']
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nStarting training (BASELINE - No Syntax)...")
    print("=" * 60)
    
    for epoch in range(1, train_config['training']['phase2_libras']['epochs'] + 1):
        print(f"\nEpoch {epoch}/{train_config['training']['phase2_libras']['epochs']}")
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, device, epoch, train_config
        )
        
        print(f"Train Loss: {train_loss:.4f}")
        
        # Evaluate
        if epoch % train_config['evaluation']['eval_steps'] == 0:
            eval_metrics = evaluate(model, dev_loader, tokenizer, device, metrics_calculator)
            
            print(f"Eval Loss: {eval_metrics['eval_loss']:.4f}")
            print(f"BLEU: {eval_metrics['bleu']:.2f}")
            print(f"TER: {eval_metrics['ter']:.2f}")
            print(f"ROUGE-L: {eval_metrics['rouge_l_f1']:.4f}")
            print(f"METEOR: {eval_metrics['meteor']:.4f}")
            
            # Log to wandb
            if not args.debug and 'wandb' in train_config['logging']['report_to']:
                wandb.log({
                    'eval/loss': eval_metrics['eval_loss'],
                    'eval/bleu': eval_metrics['bleu'],
                    'eval/ter': eval_metrics['ter'],
                    'eval/rouge_l': eval_metrics['rouge_l_f1'],
                    'eval/meteor': eval_metrics['meteor'],
                    'epoch': epoch
                })
            
            # Save best model
            if eval_metrics['bleu'] > best_bleu:
                best_bleu = eval_metrics['bleu']
                patience_counter = 0
                
                print(f"New best BLEU: {best_bleu:.2f}! Saving model...")
                model.save_pretrained(output_dir / "best_model")
                tokenizer.save_pretrained(output_dir / "best_model")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= max_patience:
                print(f"Early stopping after {epoch} epochs")
                break
        
        # Save checkpoint
        if epoch % train_config['checkpointing']['save_steps'] == 0:
            checkpoint_dir = output_dir / f"checkpoint-{epoch}"
            model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Best BLEU: {best_bleu:.2f}")
    print(f"Model saved to: {output_dir / 'best_model'}")
    
    if not args.debug:
        wandb.finish()


if __name__ == "__main__":
    main()
