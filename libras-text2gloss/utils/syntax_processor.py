"""
Syntax Processor for Portuguese Text

Extracts syntactic dependency information from Portuguese sentences using spaCy.
This module is adapted for Brazilian Portuguese and follows the same approach
as the original German implementation.
"""

import spacy
from typing import List, Dict, Tuple
import numpy as np
from tqdm import tqdm


# Portuguese dependency tag mapping
PT_SYNTAX_MAPPER = {
    'ROOT': 0,
    'nsubj': 1,      # nominal subject
    'obj': 2,        # object
    'iobj': 3,       # indirect object
    'det': 4,        # determiner
    'amod': 5,       # adjectival modifier
    'advmod': 6,     # adverbial modifier
    'aux': 7,        # auxiliary
    'case': 8,       # case marking
    'cc': 9,         # coordinating conjunction
    'conj': 10,      # conjunct
    'mark': 11,      # marker
    'nmod': 12,      # nominal modifier
    'nummod': 13,    # numeric modifier
    'obl': 14,       # oblique nominal
    'punct': 15,     # punctuation
    'acl': 16,       # clausal modifier of noun
    'advcl': 17,     # adverbial clause modifier
    'ccomp': 18,     # clausal complement
    'xcomp': 19,     # open clausal complement
    'appos': 20,     # appositional modifier
    'compound': 21,  # compound
    'fixed': 22,     # fixed multiword expression
    'flat': 23,      # flat multiword expression
    'cop': 24,       # copula
    'csubj': 25,     # clausal subject
    'dep': 26,       # unspecified dependency
    'expl': 27,      # expletive
    'nsubj:pass': 28,  # passive nominal subject
    'obl:agent': 29,   # agent modifier
    'orphan': 30,      # orphan
    'parataxis': 31,   # parataxis
    'vocative': 32,    # vocative
    'discourse': 33,   # discourse element
    'dislocated': 34,  # dislocated elements
    'reparandum': 35,  # overridden disfluency
    'goeswith': 36,    # goes with
    'list': 37,        # list
    'PAD': 38,         # padding token
    'OTHER': 39,       # other/unknown
}


class PortugueseSyntaxProcessor:
    """
    Extracts syntactic dependency information from Portuguese text.
    """
    
    def __init__(self, model_name: str = "pt_core_news_lg", batch_size: int = 1000):
        """
        Initialize the syntax processor.
        
        Args:
            model_name: spaCy model name for Portuguese
            batch_size: Batch size for processing multiple sentences
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.nlp = None
        self.tag_mapper = PT_SYNTAX_MAPPER
        
    def load_model(self):
        """Load the spaCy model."""
        if self.nlp is None:
            print(f"Loading spaCy model: {self.model_name}")
            try:
                self.nlp = spacy.load(self.model_name)
            except OSError:
                print(f"Model {self.model_name} not found. Downloading...")
                import subprocess
                subprocess.run(["python", "-m", "spacy", "download", self.model_name])
                self.nlp = spacy.load(self.model_name)
            
            # Disable unnecessary components for speed
            self.to_disable = [
                comp for comp in self.nlp.component_names 
                if comp not in ['tok2vec', 'parser', 'tagger']
            ]
            print(f"Loaded spaCy model with components: {self.nlp.component_names}")
    
    def process_sentence(self, sentence: str) -> Tuple[List[str], List[int]]:
        """
        Process a single sentence and extract dependency tags.
        
        Args:
            sentence: Input sentence in Portuguese
            
        Returns:
            Tuple of (tokens, dependency_tag_ids)
        """
        if self.nlp is None:
            self.load_model()
        
        with self.nlp.select_pipes(disable=self.to_disable):
            doc = self.nlp(sentence)
        
        tokens = [token.text for token in doc]
        dep_tags = [
            self.tag_mapper.get(token.dep_, self.tag_mapper['OTHER']) 
            for token in doc
        ]
        
        return tokens, dep_tags
    
    def process_batch(
        self, 
        sentences: List[str], 
        show_progress: bool = True
    ) -> Tuple[List[List[str]], List[List[int]]]:
        """
        Process multiple sentences in batch.
        
        Args:
            sentences: List of input sentences
            show_progress: Whether to show progress bar
            
        Returns:
            Tuple of (list of token lists, list of dependency tag lists)
        """
        if self.nlp is None:
            self.load_model()
        
        all_tokens = []
        all_dep_tags = []
        
        # Process in batches for efficiency
        iterator = range(0, len(sentences), self.batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Processing syntax")
        
        for i in iterator:
            batch = sentences[i:i + self.batch_size]
            
            with self.nlp.select_pipes(disable=self.to_disable):
                docs = list(self.nlp.pipe(batch))
            
            for doc in docs:
                tokens = [token.text for token in doc]
                dep_tags = [
                    self.tag_mapper.get(token.dep_, self.tag_mapper['OTHER']) 
                    for token in doc
                ]
                all_tokens.append(tokens)
                all_dep_tags.append(dep_tags)
        
        return all_tokens, all_dep_tags
    
    def align_with_subwords(
        self, 
        tokens: List[str], 
        dep_tags: List[int], 
        subword_tokens: List[str]
    ) -> List[int]:
        """
        Align dependency tags with subword tokens from tokenizer.
        
        When a word is split into multiple subwords, the same dependency tag
        is repeated for all subwords of that word.
        
        Args:
            tokens: Original word tokens from spaCy
            dep_tags: Dependency tags for each word
            subword_tokens: Subword tokens from tokenizer
            
        Returns:
            List of dependency tags aligned with subword tokens
        """
        aligned_tags = []
        subword_idx = 0
        
        for word_idx, word in enumerate(tokens):
            # Find how many subwords correspond to this word
            word_lower = word.lower()
            num_subwords = 0
            
            # Count subwords that belong to this word
            temp_idx = subword_idx
            reconstructed = ""
            while temp_idx < len(subword_tokens):
                subword = subword_tokens[temp_idx].replace('▁', '').lower()
                reconstructed += subword
                num_subwords += 1
                temp_idx += 1
                
                if reconstructed == word_lower:
                    break
                if len(reconstructed) > len(word_lower):
                    # Mismatch, use single subword
                    num_subwords = 1
                    break
            
            # Assign the same dependency tag to all subwords
            for _ in range(num_subwords):
                if subword_idx < len(subword_tokens):
                    aligned_tags.append(dep_tags[word_idx])
                    subword_idx += 1
        
        # Pad if necessary
        while len(aligned_tags) < len(subword_tokens):
            aligned_tags.append(self.tag_mapper['PAD'])
        
        return aligned_tags[:len(subword_tokens)]
    
    def get_tag_statistics(self, dep_tags_list: List[List[int]]) -> Dict[str, int]:
        """
        Get statistics about dependency tag usage.
        
        Args:
            dep_tags_list: List of dependency tag sequences
            
        Returns:
            Dictionary mapping tag names to counts
        """
        tag_counts = {tag: 0 for tag in self.tag_mapper.keys()}
        reverse_mapper = {v: k for k, v in self.tag_mapper.items()}
        
        for dep_tags in dep_tags_list:
            for tag_id in dep_tags:
                tag_name = reverse_mapper.get(tag_id, 'OTHER')
                tag_counts[tag_name] += 1
        
        return tag_counts


def test_syntax_processor():
    """Test the syntax processor with sample Portuguese sentences."""
    processor = PortugueseSyntaxProcessor()
    
    test_sentences = [
        "O menino está jogando bola no parque.",
        "A professora ensina matemática para os alunos.",
        "Eu vou ao mercado amanhã de manhã.",
        "Bom dia, como você está?",
    ]
    
    print("Testing Portuguese Syntax Processor\n")
    print("=" * 60)
    
    for sentence in test_sentences:
        tokens, dep_tags = processor.process_sentence(sentence)
        
        print(f"\nSentence: {sentence}")
        print(f"Tokens: {tokens}")
        
        # Get tag names
        reverse_mapper = {v: k for k, v in PT_SYNTAX_MAPPER.items()}
        tag_names = [reverse_mapper[tag_id] for tag_id in dep_tags]
        print(f"Dependency Tags: {tag_names}")
        print("-" * 60)


if __name__ == "__main__":
    test_syntax_processor()
