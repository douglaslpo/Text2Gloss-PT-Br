#!/usr/bin/env python3
"""
Interactive inference script for PT-BR to LIBRAS translation.

This script allows interactive testing of trained models.
"""

import os
import sys
import torch
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.mt5_syntax_aware import MT5SyntaxAware
from utils.tokenizer_utils import MT5TokenizerWrapper
from utils.syntax_processor import PortugueseSyntaxProcessor


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Interactive inference")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model"
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Single text to translate (optional)"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=5,
        help="Number of beams for beam search"
    )
    
    return parser.parse_args()


def translate(text, model, tokenizer, syntax_processor, device, num_beams=5):
    """Translate a single text."""
    # Tokenize
    encoding = tokenizer.encode(
        text,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt',
        add_language_token='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Get syntax features
    tokens, dep_tags = syntax_processor.process_sentence(text)
    subword_tokens = tokenizer.tokenize(text)
    aligned_tags = syntax_processor.align_with_subwords(tokens, dep_tags, subword_tokens)
    
    # Pad to max length
    syntax_ids = aligned_tags + [syntax_processor.tag_mapper['PAD']] * (128 - len(aligned_tags))
    syntax_ids = syntax_ids[:128]
    syntax_ids = torch.tensor([syntax_ids], dtype=torch.long).to(device)
    
    # Generate
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            syntax_ids=syntax_ids,
            max_length=128,
            num_beams=num_beams,
            early_stopping=True
        )
    
    # Decode
    translation = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    return translation


def interactive_mode(model, tokenizer, syntax_processor, device, num_beams):
    """Run interactive translation mode."""
    print("\n" + "=" * 60)
    print("PT-BR to LIBRAS Interactive Translation")
    print("=" * 60)
    print("\nEnter Portuguese text to translate to LIBRAS glosses.")
    print("Type 'quit' or 'exit' to stop.\n")
    
    while True:
        try:
            text = input("PT-BR> ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            if not text:
                continue
            
            translation = translate(
                text, model, tokenizer, syntax_processor, device, num_beams
            )
            
            print(f"LIBRAS> {translation}\n")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}\n")


def main():
    """Main inference function."""
    args = parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.model}...")
    model = MT5SyntaxAware.from_pretrained(args.model)
    model = model.to(device)
    model.eval()
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = MT5TokenizerWrapper.from_pretrained(args.model)
    
    # Load syntax processor
    print("Loading syntax processor...")
    syntax_processor = PortugueseSyntaxProcessor()
    
    print("Ready!\n")
    
    # Single text mode
    if args.text:
        print(f"Input:  {args.text}")
        translation = translate(
            args.text, model, tokenizer, syntax_processor, device, args.num_beams
        )
        print(f"Output: {translation}")
    
    # Interactive mode
    elif args.interactive:
        interactive_mode(model, tokenizer, syntax_processor, device, args.num_beams)
    
    else:
        print("Please specify either --text or --interactive")
        print("Example: python scripts/inference.py --model models/checkpoints/best_model --interactive")


if __name__ == "__main__":
    main()
