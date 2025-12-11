#!/usr/bin/env python3
"""
Prepare Phoenix data as processed JSONs for the data loader.
"""

import pandas as pd
import json
from pathlib import Path
import os

def prepare_json():
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    splits = ['train', 'dev', 'test']
    
    for split in splits:
        csv_file = raw_dir / f"phoenix_{split}.csv"
        if not csv_file.exists():
            print(f"File not found: {csv_file}")
            continue
            
        df = pd.read_csv(csv_file)
        data = df.to_dict('records')
        
        # Ensure strings
        for item in data:
            item['source'] = str(item['source'])
            item['target'] = str(item['target'])
            
        output_file = processed_dir / f"{split}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        print(f"Saved {len(data)} samples to {output_file}")

if __name__ == "__main__":
    prepare_json()
