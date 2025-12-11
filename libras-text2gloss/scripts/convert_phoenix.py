#!/usr/bin/env python3
"""
Convert Phoenix-2014T text files to CSV format for the data loader.
"""

import pandas as pd
import os
from pathlib import Path

def convert_phoenix_to_csv(data_dir="data"):
    """Convert spoken and gloss text files to CSV."""
    data_path = Path(data_dir)
    output_dir = data_path / "raw"
    output_dir.mkdir(exist_ok=True)
    
    splits = ['train', 'dev', 'test']
    
    for split in splits:
        spoken_file = data_path / f"spoken_{split}.txt"
        gloss_file = data_path / f"glosses_{split}.txt"
        
        if not spoken_file.exists() or not gloss_file.exists():
            print(f"Skipping {split}: files not found")
            continue
            
        print(f"Converting {split} set...")
        
        with open(spoken_file, 'r', encoding='utf-8') as f:
            spoken_lines = [line.strip() for line in f]
            
        with open(gloss_file, 'r', encoding='utf-8') as f:
            gloss_lines = [line.strip() for line in f]
            
        assert len(spoken_lines) == len(gloss_lines), f"Mismatch in {split} lines"
        
        df = pd.DataFrame({
            'source': spoken_lines,
            'target': gloss_lines
        })
        
        output_file = output_dir / f"phoenix_{split}.csv"
        df.to_csv(output_file, index=False)
        print(f"Saved {len(df)} samples to {output_file}")

if __name__ == "__main__":
    convert_phoenix_to_csv()
