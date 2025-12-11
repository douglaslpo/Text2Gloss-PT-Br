"""
Data Loader for LIBRAS Text2Gloss Translation

Handles loading and preprocessing of LIBRAS datasets with support for
multilingual training and syntax-aware features.
"""

import pandas as pd
import json
import yaml
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random
import numpy as np

from .syntax_processor import PortugueseSyntaxProcessor
from .tokenizer_utils import MT5TokenizerWrapper


class LibrasDataset(Dataset):
    """
    PyTorch Dataset for LIBRAS text-to-gloss translation.
    """
    
    def __init__(
        self,
        data: List[Dict[str, str]],
        tokenizer: MT5TokenizerWrapper,
        syntax_processor: Optional[PortugueseSyntaxProcessor] = None,
        max_source_length: int = 128,
        max_target_length: int = 128,
        use_syntax: bool = True,
    ):
        """
        Initialize dataset.
        
        Args:
            data: List of dictionaries with 'source' and 'target' keys
            tokenizer: mT5 tokenizer wrapper
            syntax_processor: Portuguese syntax processor
            max_source_length: Maximum source sequence length
            max_target_length: Maximum target sequence length
            use_syntax: Whether to extract syntax features
        """
        self.data = data
        self.tokenizer = tokenizer
        self.syntax_processor = syntax_processor
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.use_syntax = use_syntax
        
        # Preprocess syntax if enabled
        if self.use_syntax and self.syntax_processor is not None:
            self._preprocess_syntax()
    
    def _preprocess_syntax(self):
        """Preprocess syntax for all samples."""
        print("Preprocessing syntax features...")
        
        sources = [item['source'] for item in self.data]
        tokens_list, dep_tags_list = self.syntax_processor.process_batch(sources)
        
        # Store in data
        for i, (tokens, dep_tags) in enumerate(zip(tokens_list, dep_tags_list)):
            self.data[i]['syntax_tokens'] = tokens
            self.data[i]['syntax_tags'] = dep_tags
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single item.
        
        Returns:
            Dictionary with input_ids, attention_mask, labels, and optionally syntax_ids
        """
        item = self.data[idx]
        source = item['source']
        target = item['target']
        
        # Tokenize source
        source_encoding = self.tokenizer.encode(
            source,
            max_length=self.max_source_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            add_language_token='pt'
        )
        
        # Tokenize target
        target_encoding = self.tokenizer.encode(
            target,
            max_length=self.max_target_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            add_language_token='libras'
        )
        
        result = {
            'input_ids': source_encoding['input_ids'].squeeze(0),
            'attention_mask': source_encoding['attention_mask'].squeeze(0),
            'labels': target_encoding['input_ids'].squeeze(0),
        }
        
        # Add syntax features if available
        if self.use_syntax and 'syntax_tags' in item:
            # Get subword tokens from tokenizer
            subword_tokens = self.tokenizer.tokenize(source)
            
            # Align syntax tags with subwords
            aligned_tags = self.syntax_processor.align_with_subwords(
                item['syntax_tokens'],
                item['syntax_tags'],
                subword_tokens
            )
            
            # Pad to max_source_length
            syntax_ids = aligned_tags + [self.syntax_processor.tag_mapper['PAD']] * (
                self.max_source_length - len(aligned_tags)
            )
            syntax_ids = syntax_ids[:self.max_source_length]
            
            result['syntax_ids'] = torch.tensor(syntax_ids, dtype=torch.long)
        
        return result


class LibrasDataLoader:
    """
    Data loader for LIBRAS datasets with preprocessing.
    """
    
    def __init__(self, config_path: str = "configs/data_config.yaml"):
        """
        Initialize data loader.
        
        Args:
            config_path: Path to data configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.datasets_config = self.config['datasets']
        self.splits_config = self.config['splits']
        self.preprocessing_config = self.config['preprocessing']
    
    def load_libras_csv(self, dataset_name: str = 'libras') -> List[Dict[str, str]]:
        """
        Load LIBRAS dataset from CSV.
        
        Args:
            dataset_name: Name of dataset in config
            
        Returns:
            List of dictionaries with 'source' and 'target'
        """
        dataset_config = self.datasets_config[dataset_name]
        csv_path = dataset_config['path']
        
        if not Path(csv_path).exists():
            print(f"Warning: Dataset not found at {csv_path}")
            print("Using sample data for demonstration...")
            return self._create_sample_data()
        
        # Load CSV
        df = pd.read_csv(csv_path)
        
        # Get column names
        source_col = dataset_config['columns']['source']
        target_col = dataset_config['columns']['target']
        
        # Convert to list of dicts
        data = []
        for _, row in df.iterrows():
            data.append({
                'source': str(row[source_col]),
                'target': str(row[target_col])
            })
        
        # Apply preprocessing
        data = self._preprocess_data(data)
        
        return data
    
    def _create_sample_data(self, num_samples: int = 100) -> List[Dict[str, str]]:
        """Create sample data for demonstration."""
        samples = [
            ("O menino está jogando bola no parque.", "MENINO JOGAR BOLA PARQUE"),
            ("A professora ensina matemática para os alunos.", "PROFESSORA ENSINAR MATEMATICA ALUNO"),
            ("Eu vou ao mercado amanhã de manhã.", "EU IR MERCADO AMANHA MANHA"),
            ("Bom dia, como você está?", "BOM DIA COMO VOCE ESTAR"),
            ("Meu nome é João.", "MEU NOME JOAO"),
            ("Qual é o seu telefone?", "QUAL SEU TELEFONE"),
            ("Eu gosto de estudar português.", "EU GOSTAR ESTUDAR PORTUGUES"),
            ("A casa é grande e bonita.", "CASA GRANDE BONITO"),
            ("Nós vamos viajar nas férias.", "NOS IR VIAJAR FERIAS"),
            ("Ele trabalha na escola.", "ELE TRABALHAR ESCOLA"),
        ]
        
        # Repeat to get desired number
        data = []
        for i in range(num_samples):
            source, target = samples[i % len(samples)]
            data.append({'source': source, 'target': target})
        
        return data
    
    def _preprocess_data(self, data: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Apply preprocessing to data."""
        cleaned_data = []
        
        for item in data:
            source = item['source'].strip()
            target = item['target'].strip()
            
            # Apply filters
            if len(source.split()) < self.preprocessing_config['min_length']:
                continue
            if len(source.split()) > self.preprocessing_config['max_length']:
                continue
            if not source or not target:
                continue
            
            cleaned_data.append({'source': source, 'target': target})
        
        # Remove duplicates if configured
        if self.preprocessing_config['remove_duplicates']:
            seen = set()
            unique_data = []
            for item in cleaned_data:
                key = (item['source'], item['target'])
                if key not in seen:
                    seen.add(key)
                    unique_data.append(item)
            cleaned_data = unique_data
        
        return cleaned_data
    
    def split_data(
        self, 
        data: List[Dict[str, str]]
    ) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
        """
        Split data into train/dev/test.
        
        Args:
            data: Full dataset
            
        Returns:
            Tuple of (train_data, dev_data, test_data)
        """
        # Shuffle if configured
        if self.splits_config['shuffle']:
            random.shuffle(data)
        
        # Calculate split sizes
        total = len(data)
        train_size = int(total * self.splits_config['train'])
        dev_size = int(total * self.splits_config['dev'])
        
        # Split
        train_data = data[:train_size]
        dev_data = data[train_size:train_size + dev_size]
        test_data = data[train_size + dev_size:]
        
        return train_data, dev_data, test_data
    
    def save_processed_data(
        self,
        train_data: List[Dict[str, str]],
        dev_data: List[Dict[str, str]],
        test_data: List[Dict[str, str]],
        output_dir: str = "data/processed/"
    ):
        """Save processed data to JSON files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save each split
        with open(output_path / "train.json", 'w', encoding='utf-8') as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)
        
        with open(output_path / "dev.json", 'w', encoding='utf-8') as f:
            json.dump(dev_data, f, ensure_ascii=False, indent=2)
        
        with open(output_path / "test.json", 'w', encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)
        
        print(f"Saved processed data to {output_dir}")
        print(f"  Train: {len(train_data)} samples")
        print(f"  Dev: {len(dev_data)} samples")
        print(f"  Test: {len(test_data)} samples")
    
    def load_processed_data(
        self, 
        data_dir: str = "data/processed/"
    ) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
        """Load previously processed data."""
        data_path = Path(data_dir)
        
        with open(data_path / "train.json", 'r', encoding='utf-8') as f:
            train_data = json.load(f)
        
        with open(data_path / "dev.json", 'r', encoding='utf-8') as f:
            dev_data = json.load(f)
        
        with open(data_path / "test.json", 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        return train_data, dev_data, test_data


def create_dataloaders(
    train_data: List[Dict[str, str]],
    dev_data: List[Dict[str, str]],
    test_data: List[Dict[str, str]],
    tokenizer: MT5TokenizerWrapper,
    syntax_processor: Optional[PortugueseSyntaxProcessor] = None,
    batch_size: int = 16,
    use_syntax: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for train/dev/test.
    
    Args:
        train_data: Training data
        dev_data: Development data
        test_data: Test data
        tokenizer: mT5 tokenizer
        syntax_processor: Syntax processor
        batch_size: Batch size
        use_syntax: Whether to use syntax features
        
    Returns:
        Tuple of (train_loader, dev_loader, test_loader)
    """
    train_dataset = LibrasDataset(
        train_data, tokenizer, syntax_processor, use_syntax=use_syntax
    )
    dev_dataset = LibrasDataset(
        dev_data, tokenizer, syntax_processor, use_syntax=use_syntax
    )
    test_dataset = LibrasDataset(
        test_data, tokenizer, syntax_processor, use_syntax=use_syntax
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    
    return train_loader, dev_loader, test_loader


if __name__ == "__main__":
    # Test data loader
    print("Testing LIBRAS Data Loader\n")
    
    loader = LibrasDataLoader()
    data = loader.load_libras_csv()
    
    print(f"Loaded {len(data)} samples")
    print(f"\nSample data:")
    for i in range(min(3, len(data))):
        print(f"  {i+1}. Source: {data[i]['source']}")
        print(f"     Target: {data[i]['target']}\n")
    
    # Split data
    train, dev, test = loader.split_data(data)
    print(f"Split sizes: Train={len(train)}, Dev={len(dev)}, Test={len(test)}")
