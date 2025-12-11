"""
Unit tests for metrics module.
"""

import unittest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.metrics import TranslationMetrics


class TestMetrics(unittest.TestCase):
    """Test cases for TranslationMetrics."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.metrics = TranslationMetrics()
        
        # Sample data
        cls.predictions = [
            "MENINO JOGAR BOLA PARQUE",
            "PROFESSORA ENSINAR MATEMATICA",
            "EU IR MERCADO AMANHA"
        ]
        
        cls.references = [
            "MENINO JOGAR BOLA PARQUE",
            "PROFESSORA ENSINAR MATEMATICA ALUNO",
            "EU IR MERCADO AMANHA MANHA"
        ]
    
    def test_compute_bleu(self):
        """Test BLEU score computation."""
        bleu_scores = self.metrics.compute_bleu(
            self.predictions,
            [[ref] for ref in self.references]
        )
        
        # Check that we get a score
        self.assertIn('bleu', bleu_scores)
        self.assertIsInstance(bleu_scores['bleu'], float)
        
        # BLEU should be between 0 and 100
        self.assertGreaterEqual(bleu_scores['bleu'], 0.0)
        self.assertLessEqual(bleu_scores['bleu'], 100.0)
        
        # Should have individual BLEU scores
        self.assertIn('bleu_1', bleu_scores)
        self.assertIn('bleu_2', bleu_scores)
    
    def test_compute_ter(self):
        """Test TER score computation."""
        ter_score = self.metrics.compute_ter(
            self.predictions,
            [[ref] for ref in self.references]
        )
        
        # Check that we get a score
        self.assertIsInstance(ter_score, float)
        
        # TER should be non-negative
        self.assertGreaterEqual(ter_score, 0.0)
    
    def test_compute_rouge_l(self):
        """Test ROUGE-L score computation."""
        rouge_scores = self.metrics.compute_rouge_l(
            self.predictions,
            self.references
        )
        
        # Check that we get scores
        self.assertIn('rouge_l_f1', rouge_scores)
        self.assertIsInstance(rouge_scores['rouge_l_f1'], float)
        
        # ROUGE-L should be between 0 and 1
        self.assertGreaterEqual(rouge_scores['rouge_l_f1'], 0.0)
        self.assertLessEqual(rouge_scores['rouge_l_f1'], 1.0)
    
    def test_compute_meteor(self):
        """Test METEOR score computation."""
        meteor_score = self.metrics.compute_meteor(
            self.predictions,
            self.references
        )
        
        # Check that we get a score
        self.assertIsInstance(meteor_score, float)
        
        # METEOR should be between 0 and 1
        self.assertGreaterEqual(meteor_score, 0.0)
        self.assertLessEqual(meteor_score, 1.0)
    
    def test_compute_all_metrics(self):
        """Test computing all metrics at once."""
        all_metrics = self.metrics.compute_all_metrics(
            self.predictions,
            self.references,
            measure_time=True
        )
        
        # Check that all metrics are present
        self.assertIn('bleu', all_metrics)
        self.assertIn('ter', all_metrics)
        self.assertIn('rouge_l_f1', all_metrics)
        self.assertIn('meteor', all_metrics)
        self.assertIn('generation_time', all_metrics)
        
        # All should be numeric
        for key, value in all_metrics.items():
            self.assertIsInstance(value, (int, float))
    
    def test_perfect_match(self):
        """Test metrics with perfect predictions."""
        perfect_preds = self.references.copy()
        
        metrics = self.metrics.compute_all_metrics(
            perfect_preds,
            self.references,
            measure_time=False
        )
        
        # BLEU should be 100 for perfect match
        self.assertAlmostEqual(metrics['bleu'], 100.0, delta=1.0)
        
        # TER should be 0 for perfect match
        self.assertAlmostEqual(metrics['ter'], 0.0, delta=1.0)
        
        # ROUGE-L should be 1.0 for perfect match
        self.assertAlmostEqual(metrics['rouge_l_f1'], 1.0, delta=0.1)


if __name__ == '__main__':
    unittest.main()
