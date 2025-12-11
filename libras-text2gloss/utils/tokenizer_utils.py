"""
Tokenizer Utilities for mT5

Handles tokenization for multilingual T5 models with special tokens
for different sign languages.
"""

from transformers import MT5Tokenizer, AutoTokenizer
from typing import List, Dict, Optional, Union
import torch


class MT5TokenizerWrapper:
    """
    Wrapper around HuggingFace mT5 tokenizer with special tokens
    for sign language translation.
    """
    
    def __init__(
        self, 
        model_name: str = "google/mt5-base",
        add_special_tokens: bool = True
    ):
        """
        Initialize the tokenizer.
        
        Args:
            model_name: HuggingFace model name
            add_special_tokens: Whether to add custom special tokens
        """
        self.model_name = model_name
        self.tokenizer = MT5Tokenizer.from_pretrained(model_name)
        
        if add_special_tokens:
            self._add_special_tokens()
    
    def _add_special_tokens(self):
        """Add special tokens for different sign languages."""
        special_tokens = {
            'additional_special_tokens': [
                '<pt>',      # Portuguese
                '<libras>',  # LIBRAS
                '<asl>',     # American Sign Language
                '<dgs>',     # German Sign Language
            ]
        }
        
        num_added = self.tokenizer.add_special_tokens(special_tokens)
        print(f"Added {num_added} special tokens to tokenizer")
        
        # Store token IDs for easy access
        self.pt_token_id = self.tokenizer.convert_tokens_to_ids('<pt>')
        self.libras_token_id = self.tokenizer.convert_tokens_to_ids('<libras>')
        self.asl_token_id = self.tokenizer.convert_tokens_to_ids('<asl>')
        self.dgs_token_id = self.tokenizer.convert_tokens_to_ids('<dgs>')
    
    def encode(
        self,
        text: Union[str, List[str]],
        max_length: int = 128,
        padding: str = "max_length",
        truncation: bool = True,
        return_tensors: Optional[str] = "pt",
        add_language_token: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text or list of texts
            max_length: Maximum sequence length
            padding: Padding strategy
            truncation: Whether to truncate
            return_tensors: Return format ('pt' for PyTorch)
            add_language_token: Language token to prepend ('pt', 'libras', etc.)
            
        Returns:
            Dictionary with input_ids and attention_mask
        """
        # Add language token if specified
        if add_language_token:
            lang_token = f"<{add_language_token}>"
            if isinstance(text, str):
                text = f"{lang_token} {text}"
            else:
                text = [f"{lang_token} {t}" for t in text]
        
        encoding = self.tokenizer(
            text,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors
        )
        
        return encoding
    
    def decode(
        self,
        token_ids: Union[List[int], torch.Tensor],
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True
    ) -> str:
        """
        Decode token IDs to text.
        
        Args:
            token_ids: Token IDs to decode
            skip_special_tokens: Whether to skip special tokens
            clean_up_tokenization_spaces: Whether to clean up spaces
            
        Returns:
            Decoded text
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        
        text = self.tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces
        )
        
        return text
    
    def batch_decode(
        self,
        token_ids: Union[List[List[int]], torch.Tensor],
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True
    ) -> List[str]:
        """
        Decode batch of token IDs to texts.
        
        Args:
            token_ids: Batch of token IDs
            skip_special_tokens: Whether to skip special tokens
            clean_up_tokenization_spaces: Whether to clean up spaces
            
        Returns:
            List of decoded texts
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        
        texts = self.tokenizer.batch_decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces
        )
        
        return texts
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into subword tokens.
        
        Args:
            text: Input text
            
        Returns:
            List of subword tokens
        """
        return self.tokenizer.tokenize(text)
    
    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """Convert tokens to IDs."""
        return self.tokenizer.convert_tokens_to_ids(tokens)
    
    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        """Convert IDs to tokens."""
        return self.tokenizer.convert_ids_to_tokens(ids)
    
    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.tokenizer)
    
    @property
    def pad_token_id(self) -> int:
        """Get padding token ID."""
        return self.tokenizer.pad_token_id
    
    @property
    def eos_token_id(self) -> int:
        """Get end-of-sequence token ID."""
        return self.tokenizer.eos_token_id
    
    @property
    def bos_token_id(self) -> int:
        """Get beginning-of-sequence token ID."""
        return self.tokenizer.bos_token_id if hasattr(self.tokenizer, 'bos_token_id') else self.tokenizer.pad_token_id
    
    def save_pretrained(self, save_directory: str):
        """Save tokenizer to directory."""
        self.tokenizer.save_pretrained(save_directory)
    
    @classmethod
    def from_pretrained(cls, model_path: str):
        """Load tokenizer from directory."""
        instance = cls(model_name=model_path, add_special_tokens=False)
        return instance


def test_tokenizer():
    """Test the tokenizer with sample sentences."""
    print("Testing mT5 Tokenizer Wrapper\n")
    print("=" * 60)
    
    tokenizer = MT5TokenizerWrapper()
    
    # Test Portuguese sentences
    pt_sentences = [
        "O menino está jogando bola.",
        "Bom dia, como você está?",
    ]
    
    # Test LIBRAS glosses
    libras_glosses = [
        "MENINO JOGAR BOLA",
        "BOM DIA COMO VOCE ESTAR",
    ]
    
    print("\n1. Encoding Portuguese sentences:")
    for sent in pt_sentences:
        encoding = tokenizer.encode(sent, add_language_token='pt', max_length=32)
        print(f"\nSentence: {sent}")
        print(f"Token IDs: {encoding['input_ids'][0][:15]}...")
        print(f"Tokens: {tokenizer.tokenize(sent)[:10]}...")
    
    print("\n\n2. Encoding LIBRAS glosses:")
    for gloss in libras_glosses:
        encoding = tokenizer.encode(gloss, add_language_token='libras', max_length=32)
        print(f"\nGloss: {gloss}")
        print(f"Token IDs: {encoding['input_ids'][0][:15]}...")
        decoded = tokenizer.decode(encoding['input_ids'][0])
        print(f"Decoded: {decoded}")
    
    print("\n\n3. Batch encoding:")
    batch_encoding = tokenizer.encode(
        pt_sentences,
        add_language_token='pt',
        max_length=32
    )
    print(f"Batch shape: {batch_encoding['input_ids'].shape}")
    print(f"Attention mask shape: {batch_encoding['attention_mask'].shape}")
    
    print("\n\n4. Vocabulary info:")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"PAD token ID: {tokenizer.pad_token_id}")
    print(f"EOS token ID: {tokenizer.eos_token_id}")
    print(f"PT token ID: {tokenizer.pt_token_id}")
    print(f"LIBRAS token ID: {tokenizer.libras_token_id}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    test_tokenizer()
