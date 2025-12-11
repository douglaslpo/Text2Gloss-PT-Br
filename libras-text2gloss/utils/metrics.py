"""
Evaluation Metrics for Text2Gloss Translation

Implements standard MT evaluation metrics:
- sacreBLEU
- TER (Translation Edit Rate)
- ROUGE-L F1
- METEOR
"""

import sacrebleu
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
from nltk import word_tokenize
import nltk
from typing import List, Dict
import time
import numpy as np

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


class TranslationMetrics:
    """
    Compute evaluation metrics for text-to-gloss translation.
    """
    
    def __init__(self):
        """Initialize metrics calculators."""
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
    
    def compute_bleu(
        self, 
        predictions: List[str], 
        references: List[List[str]]
    ) -> Dict[str, float]:
        """
        Compute sacreBLEU score.
        
        Args:
            predictions: List of predicted translations
            references: List of reference translations (each can have multiple refs)
            
        Returns:
            Dictionary with BLEU scores
        """
        # Ensure references is list of lists
        if references and isinstance(references[0], str):
            references = [[ref] for ref in references]
        
        # Transpose references for sacreBLEU format
        refs_transposed = list(zip(*references))
        
        bleu = sacrebleu.corpus_bleu(predictions, refs_transposed)
        
        return {
            'bleu': bleu.score,
            'bleu_1': bleu.precisions[0] if len(bleu.precisions) > 0 else 0.0,
            'bleu_2': bleu.precisions[1] if len(bleu.precisions) > 1 else 0.0,
            'bleu_3': bleu.precisions[2] if len(bleu.precisions) > 2 else 0.0,
            'bleu_4': bleu.precisions[3] if len(bleu.precisions) > 3 else 0.0,
        }
    
    def compute_ter(
        self, 
        predictions: List[str], 
        references: List[List[str]]
    ) -> float:
        """
        Compute TER (Translation Edit Rate).
        
        Args:
            predictions: List of predicted translations
            references: List of reference translations
            
        Returns:
            TER score (lower is better)
        """
        # Ensure references is list of lists
        if references and isinstance(references[0], str):
            references = [[ref] for ref in references]
        
        # Transpose references
        refs_transposed = list(zip(*references))
        
        ter = sacrebleu.corpus_ter(predictions, refs_transposed)
        
        return ter.score
    
    def compute_rouge_l(
        self, 
        predictions: List[str], 
        references: List[str]
    ) -> Dict[str, float]:
        """
        Compute ROUGE-L F1 score.
        
        Args:
            predictions: List of predicted translations
            references: List of reference translations (single reference per prediction)
            
        Returns:
            Dictionary with ROUGE-L scores
        """
        rouge_scores = []
        
        for pred, ref in zip(predictions, references):
            score = self.rouge_scorer.score(ref, pred)
            rouge_scores.append(score['rougeL'].fmeasure)
        
        return {
            'rouge_l_f1': np.mean(rouge_scores),
            'rouge_l_precision': np.mean([
                self.rouge_scorer.score(ref, pred)['rougeL'].precision
                for pred, ref in zip(predictions, references)
            ]),
            'rouge_l_recall': np.mean([
                self.rouge_scorer.score(ref, pred)['rougeL'].recall
                for pred, ref in zip(predictions, references)
            ]),
        }
    
    def compute_meteor(
        self, 
        predictions: List[str], 
        references: List[str],
        language: str = 'pt'
    ) -> float:
        """
        Compute METEOR score.
        
        Args:
            predictions: List of predicted translations
            references: List of reference translations
            language: Language code for METEOR
            
        Returns:
            METEOR score
        """
        meteor_scores = []
        
        for pred, ref in zip(predictions, references):
            # Tokenize
            pred_tokens = word_tokenize(pred.lower())
            ref_tokens = word_tokenize(ref.lower())
            
            # Compute METEOR
            try:
                score = meteor_score([ref_tokens], pred_tokens)
                meteor_scores.append(score)
            except Exception as e:
                print(f"Error computing METEOR: {e}")
                meteor_scores.append(0.0)
        
        return np.mean(meteor_scores)
    
    def compute_all_metrics(
        self,
        predictions: List[str],
        references: List[str],
        measure_time: bool = True
    ) -> Dict[str, float]:
        """
        Compute all metrics at once.
        
        Args:
            predictions: List of predicted translations
            references: List of reference translations
            measure_time: Whether to measure generation time
            
        Returns:
            Dictionary with all metric scores
        """
        start_time = time.time()
        
        # Ensure references is in correct format
        if isinstance(references[0], str):
            refs_list = [[ref] for ref in references]
        else:
            refs_list = references
        
        # Compute metrics
        bleu_scores = self.compute_bleu(predictions, refs_list)
        ter_score = self.compute_ter(predictions, refs_list)
        rouge_scores = self.compute_rouge_l(predictions, references if isinstance(references[0], str) else [r[0] for r in references])
        meteor_score_val = self.compute_meteor(predictions, references if isinstance(references[0], str) else [r[0] for r in references])
        
        metrics = {
            **bleu_scores,
            'ter': ter_score,
            **rouge_scores,
            'meteor': meteor_score_val,
        }
        
        if measure_time:
            metrics['generation_time'] = time.time() - start_time
        
        return metrics
    
    def print_metrics(self, metrics: Dict[str, float]):
        """
        Pretty print metrics.
        
        Args:
            metrics: Dictionary of metric scores
        """
        print("\n" + "=" * 60)
        print("EVALUATION METRICS")
        print("=" * 60)
        
        # Main metrics
        print(f"\nsacreBLEU:     {metrics.get('bleu', 0.0):.2f}")
        print(f"TER:           {metrics.get('ter', 0.0):.2f}")
        print(f"ROUGE-L F1:    {metrics.get('rouge_l_f1', 0.0):.4f}")
        print(f"METEOR:        {metrics.get('meteor', 0.0):.4f}")
        
        # Additional BLEU scores
        if 'bleu_1' in metrics:
            print(f"\nBLEU-1:        {metrics['bleu_1']:.2f}")
            print(f"BLEU-2:        {metrics['bleu_2']:.2f}")
            print(f"BLEU-3:        {metrics['bleu_3']:.2f}")
            print(f"BLEU-4:        {metrics['bleu_4']:.2f}")
        
        # Time
        if 'generation_time' in metrics:
            print(f"\nGeneration Time: {metrics['generation_time']:.2f}s")
        
        print("=" * 60 + "\n")


def test_metrics():
    """Test metrics with sample data."""
    print("Testing Translation Metrics\n")
    
    # Sample predictions and references
    predictions = [
        "MENINO JOGAR BOLA PARQUE",
        "PROFESSORA ENSINAR MATEMATICA ALUNO",
        "EU IR MERCADO AMANHA",
    ]
    
    references = [
        "MENINO JOGAR BOLA PARQUE",
        "PROFESSORA ENSINAR MATEMATICA PARA ALUNO",
        "EU IR MERCADO AMANHA MANHA",
    ]
    
    metrics = TranslationMetrics()
    
    # Compute all metrics
    scores = metrics.compute_all_metrics(predictions, references)
    
    # Print results
    metrics.print_metrics(scores)
    
    # Test with multiple references
    print("\nTesting with multiple references:")
    multi_refs = [
        ["MENINO JOGAR BOLA PARQUE", "MENINO BOLA JOGAR PARQUE"],
        ["PROFESSORA ENSINAR MATEMATICA PARA ALUNO", "PROFESSORA MATEMATICA ENSINAR ALUNO"],
        ["EU IR MERCADO AMANHA MANHA", "EU MERCADO IR AMANHA"],
    ]
    
    scores_multi = metrics.compute_all_metrics(predictions, multi_refs)
    metrics.print_metrics(scores_multi)


if __name__ == "__main__":
    test_metrics()
