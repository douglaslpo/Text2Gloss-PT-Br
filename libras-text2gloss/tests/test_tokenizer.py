"""
Unit tests for tokenizer utilities.
"""

import unittest
import sys
from pathlib import Path
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.tokenizer_utils import MT5TokenizerWrapper


class TestTokenizerUtils(unittest.TestCase):
    """Test cases for MT5TokenizerWrapper."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.tokenizer = MT5TokenizerWrapper(model_name="google/mt5-small")
    
    def test_initialization(self):
        """Test tokenizer initialization."""
        self.assertIsNotNone(self.tokenizer.tokenizer)
        self.assertGreater(self.tokenizer.vocab_size, 0)
    
    def test_special_tokens(self):
        """Test that special tokens are added."""
        # Check that language tokens exist
        self.assertIsNotNone(self.tokenizer.pt_token_id)
        self.assertIsNotNone(self.tokenizer.libras_token_id)
        
        # Check they are different
        self.assertNotEqual(self.tokenizer.pt_token_id, self.tokenizer.libras_token_id)
    
    def test_encode_single(self):
        """Test encoding a single text."""
        text = "Bom dia, como você está?"
        encoding = self.tokenizer.encode(text, max_length=32)
        
        # Check output format
        self.assertIn('input_ids', encoding)
        self.assertIn('attention_mask', encoding)
        
        # Check tensor shapes
        self.assertIsInstance(encoding['input_ids'], torch.Tensor)
        self.assertEqual(encoding['input_ids'].shape[1], 32)
    
    def test_encode_batch(self):
        """Test encoding multiple texts."""
        texts = [
            "Bom dia",
            "Como você está?",
            "Eu vou ao mercado"
        ]
        
        encoding = self.tokenizer.encode(texts, max_length=32)
        
        # Check batch dimension
        self.assertEqual(encoding['input_ids'].shape[0], len(texts))
        self.assertEqual(encoding['input_ids'].shape[1], 32)
    
    def test_encode_with_language_token(self):
        """Test encoding with language token."""
        text = "Bom dia"
        
        encoding = self.tokenizer.encode(text, add_language_token='pt', max_length=32)
        
        # Decode to check if language token is present
        decoded = self.tokenizer.decode(encoding['input_ids'][0])
        self.assertIn('pt', decoded.lower())
    
    def test_decode(self):
        """Test decoding token IDs."""
        text = "Bom dia"
        encoding = self.tokenizer.encode(text, max_length=32)
        decoded = self.tokenizer.decode(encoding['input_ids'][0])
        
        # Should contain original text (may have minor differences)
        self.assertIsInstance(decoded, str)
        self.assertGreater(len(decoded), 0)
    
    def test_batch_decode(self):
        """Test batch decoding."""
        texts = ["Bom dia", "Boa tarde"]
        encoding = self.tokenizer.encode(texts, max_length=32)
        decoded = self.tokenizer.batch_decode(encoding['input_ids'])
        
        # Should return list of strings
        self.assertIsInstance(decoded, list)
        self.assertEqual(len(decoded), len(texts))
    
    def test_tokenize(self):
        """Test tokenization."""
        text = "Bom dia"
        tokens = self.tokenizer.tokenize(text)
        
        # Should return list of tokens
        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)
    
    def test_vocab_properties(self):
        """Test vocabulary properties."""
        # Check pad token ID
        self.assertIsNotNone(self.tokenizer.pad_token_id)
        
        # Check EOS token ID
        self.assertIsNotNone(self.tokenizer.eos_token_id)
        
        # Check vocab size
        self.assertGreater(self.tokenizer.vocab_size, 1000)


if __name__ == '__main__':
    unittest.main()
