"""
Unit tests for syntax processor.
"""

import unittest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.syntax_processor import PortugueseSyntaxProcessor, PT_SYNTAX_MAPPER


class TestSyntaxProcessor(unittest.TestCase):
    """Test cases for PortugueseSyntaxProcessor."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.processor = PortugueseSyntaxProcessor()
        cls.processor.load_model()
    
    def test_process_sentence(self):
        """Test processing a single sentence."""
        sentence = "O menino joga bola."
        tokens, dep_tags = self.processor.process_sentence(sentence)
        
        # Check that we get tokens
        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)
        
        # Check that we get dependency tags
        self.assertIsInstance(dep_tags, list)
        self.assertEqual(len(tokens), len(dep_tags))
        
        # Check that tags are valid integers
        for tag in dep_tags:
            self.assertIsInstance(tag, int)
            self.assertGreaterEqual(tag, 0)
    
    def test_process_batch(self):
        """Test batch processing."""
        sentences = [
            "O menino joga bola.",
            "A professora ensina matemática.",
            "Eu vou ao mercado."
        ]
        
        tokens_list, dep_tags_list = self.processor.process_batch(
            sentences, show_progress=False
        )
        
        # Check correct number of results
        self.assertEqual(len(tokens_list), len(sentences))
        self.assertEqual(len(dep_tags_list), len(sentences))
        
        # Check each result
        for tokens, dep_tags in zip(tokens_list, dep_tags_list):
            self.assertGreater(len(tokens), 0)
            self.assertEqual(len(tokens), len(dep_tags))
    
    def test_tag_mapper(self):
        """Test that tag mapper contains expected tags."""
        # Check essential tags
        essential_tags = ['ROOT', 'nsubj', 'obj', 'det', 'PAD', 'OTHER']
        
        for tag in essential_tags:
            self.assertIn(tag, PT_SYNTAX_MAPPER)
            self.assertIsInstance(PT_SYNTAX_MAPPER[tag], int)
    
    def test_align_with_subwords(self):
        """Test alignment of tags with subword tokens."""
        tokens = ['O', 'menino', 'joga']
        dep_tags = [PT_SYNTAX_MAPPER['det'], PT_SYNTAX_MAPPER['nsubj'], PT_SYNTAX_MAPPER['ROOT']]
        subword_tokens = ['▁O', '▁menino', '▁jo', 'ga']
        
        aligned_tags = self.processor.align_with_subwords(tokens, dep_tags, subword_tokens)
        
        # Should have same length as subword tokens
        self.assertEqual(len(aligned_tags), len(subword_tokens))
        
        # All should be valid tag IDs
        for tag in aligned_tags:
            self.assertIsInstance(tag, int)
    
    def test_get_tag_statistics(self):
        """Test tag statistics calculation."""
        dep_tags_list = [
            [PT_SYNTAX_MAPPER['ROOT'], PT_SYNTAX_MAPPER['nsubj']],
            [PT_SYNTAX_MAPPER['ROOT'], PT_SYNTAX_MAPPER['obj']],
        ]
        
        stats = self.processor.get_tag_statistics(dep_tags_list)
        
        # Should be a dictionary
        self.assertIsInstance(stats, dict)
        
        # Should have counts for ROOT
        self.assertEqual(stats['ROOT'], 2)
        self.assertEqual(stats['nsubj'], 1)
        self.assertEqual(stats['obj'], 1)


if __name__ == '__main__':
    unittest.main()
