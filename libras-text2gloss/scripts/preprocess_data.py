#!/usr/bin/env python3
"""
Data preprocessing script for LIBRAS datasets.

This script loads raw data, applies preprocessing, and saves train/dev/test splits.
"""

import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.data_loader import LibrasDataLoader


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Preprocess LIBRAS data")
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/vlibrasbd.csv",
        help="Path to input CSV file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/",
        help="Output directory for processed data"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/data_config.yaml",
        help="Path to data config"
    )
    parser.add_argument(
        "--split",
        nargs=3,
        type=float,
        default=[0.8, 0.1, 0.1],
        help="Train/dev/test split ratios"
    )
    
    return parser.parse_args()


def main():
    """Main preprocessing function."""
    args = parse_args()
    
    print("=" * 60)
    print("LIBRAS Data Preprocessing")
    print("=" * 60)
    
    # Initialize data loader
    print(f"\nLoading configuration from {args.config}...")
    loader = LibrasDataLoader(config_path=args.config)
    
    # Update split ratios if provided
    if args.split:
        loader.splits_config['train'] = args.split[0]
        loader.splits_config['dev'] = args.split[1]
        loader.splits_config['test'] = args.split[2]
        print(f"Using split ratios: {args.split[0]:.1%} / {args.split[1]:.1%} / {args.split[2]:.1%}")
    
    # Load raw data
    print(f"\nLoading raw data from {args.input}...")
    data = loader.load_libras_csv()
    print(f"Loaded {len(data)} samples")
    
    # Show sample
    print("\nSample data:")
    for i in range(min(3, len(data))):
        print(f"  {i+1}. Source: {data[i]['source'][:60]}...")
        print(f"     Target: {data[i]['target']}")
    
    # Split data
    print("\nSplitting data...")
    train_data, dev_data, test_data = loader.split_data(data)
    
    print(f"  Train: {len(train_data)} samples ({len(train_data)/len(data)*100:.1f}%)")
    print(f"  Dev:   {len(dev_data)} samples ({len(dev_data)/len(data)*100:.1f}%)")
    print(f"  Test:  {len(test_data)} samples ({len(test_data)/len(data)*100:.1f}%)")
    
    # Save processed data
    print(f"\nSaving processed data to {args.output}...")
    loader.save_processed_data(train_data, dev_data, test_data, output_dir=args.output)
    
    print("\n" + "=" * 60)
    print("Preprocessing completed successfully!")
    print("=" * 60)
    
    print(f"\nProcessed files:")
    print(f"  - {args.output}/train.json")
    print(f"  - {args.output}/dev.json")
    print(f"  - {args.output}/test.json")
    
    print("\nYou can now train the model using:")
    print("  python scripts/train_mt5_syntax.py")


if __name__ == "__main__":
    main()
