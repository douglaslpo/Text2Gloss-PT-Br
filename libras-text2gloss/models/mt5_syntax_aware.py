"""
Syntax-Aware mT5 Model for Text2Gloss Translation

Implements mT5 with syntax-aware embeddings for Portuguese to LIBRAS translation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import MT5ForConditionalGeneration, MT5Config
from typing import Optional, Dict


class MT5SyntaxAware(nn.Module):
    """
    mT5 model with syntax-aware embeddings.
    
    This model extends the standard mT5 by injecting syntactic dependency
    information into the encoder embeddings.
    """
    
    def __init__(
        self,
        model_name: str = "google/mt5-base",
        num_syntax_tags: int = 50,
        syntax_embedding_dim: Optional[int] = None,
        freeze_encoder: bool = False,
        freeze_decoder: bool = False,
    ):
        """
        Initialize the syntax-aware mT5 model.
        
        Args:
            model_name: HuggingFace model name or path
            num_syntax_tags: Number of dependency tags
            syntax_embedding_dim: Dimension of syntax embeddings (defaults to model hidden size)
            freeze_encoder: Whether to freeze encoder weights
            freeze_decoder: Whether to freeze decoder weights
        """
        super().__init__()
        
        # Load pre-trained mT5
        self.mt5 = MT5ForConditionalGeneration.from_pretrained(model_name)
        self.config = self.mt5.config
        
        # Set syntax embedding dimension
        if syntax_embedding_dim is None:
            syntax_embedding_dim = self.config.d_model
        
        self.syntax_embedding_dim = syntax_embedding_dim
        
        # Create syntax embedding layer
        self.syntax_embeddings = nn.Embedding(
            num_embeddings=num_syntax_tags,
            embedding_dim=syntax_embedding_dim
        )
        
        # Initialize syntax embeddings
        nn.init.normal_(self.syntax_embeddings.weight, mean=0.0, std=0.02)
        
        # Optionally freeze encoder/decoder
        if freeze_encoder:
            for param in self.mt5.encoder.parameters():
                param.requires_grad = False
        
        if freeze_decoder:
            for param in self.mt5.decoder.parameters():
                param.requires_grad = False
    
    def combine_embeddings(
        self,
        word_embeds: torch.Tensor,
        syntax_embeds: torch.Tensor,
        method: str = "sum"
    ) -> torch.Tensor:
        """
        Combine word and syntax embeddings.
        
        Args:
            word_embeds: Word embeddings [batch_size, seq_len, hidden_size]
            syntax_embeds: Syntax embeddings [batch_size, seq_len, hidden_size]
            method: Combination method ('sum', 'concat', 'attention')
            
        Returns:
            Combined embeddings
        """
        if method == "sum":
            # L2 normalize and sum (as in original paper)
            word_norm = F.normalize(word_embeds, p=2, dim=-1)
            syntax_norm = F.normalize(syntax_embeds, p=2, dim=-1)
            combined = word_norm + syntax_norm
            
        elif method == "concat":
            # Concatenate (requires projection layer)
            combined = torch.cat([word_embeds, syntax_embeds], dim=-1)
            
        elif method == "attention":
            # Attention-based combination (more complex)
            # For now, fallback to sum
            word_norm = F.normalize(word_embeds, p=2, dim=-1)
            syntax_norm = F.normalize(syntax_embeds, p=2, dim=-1)
            combined = word_norm + syntax_norm
        
        else:
            raise ValueError(f"Unknown combination method: {method}")
        
        return combined
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        syntax_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with syntax-aware embeddings.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            syntax_ids: Syntax tag IDs [batch_size, seq_len]
            labels: Target labels [batch_size, target_len]
            decoder_input_ids: Decoder input IDs
            decoder_attention_mask: Decoder attention mask
            return_dict: Whether to return dict
            
        Returns:
            Model outputs with loss, logits, etc.
        """
        # Get word embeddings from mT5
        word_embeds = self.mt5.encoder.embed_tokens(input_ids)
        
        # If syntax IDs provided, combine with syntax embeddings
        if syntax_ids is not None:
            syntax_embeds = self.syntax_embeddings(syntax_ids)
            encoder_inputs_embeds = self.combine_embeddings(
                word_embeds, syntax_embeds, method="sum"
            )
        else:
            encoder_inputs_embeds = word_embeds
        
        # Forward through mT5 with custom embeddings
        outputs = self.mt5(
            input_ids=None,  # We provide embeddings directly
            attention_mask=attention_mask,
            inputs_embeds=encoder_inputs_embeds,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            return_dict=return_dict,
        )
        
        return outputs
    
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        syntax_ids: Optional[torch.Tensor] = None,
        max_length: int = 128,
        num_beams: int = 5,
        early_stopping: bool = True,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate translations using beam search.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            syntax_ids: Syntax tag IDs
            max_length: Maximum generation length
            num_beams: Number of beams for beam search
            early_stopping: Whether to stop early
            **kwargs: Additional generation arguments
            
        Returns:
            Generated token IDs
        """
        # Get encoder embeddings
        word_embeds = self.mt5.encoder.embed_tokens(input_ids)
        
        if syntax_ids is not None:
            syntax_embeds = self.syntax_embeddings(syntax_ids)
            encoder_inputs_embeds = self.combine_embeddings(
                word_embeds, syntax_embeds, method="sum"
            )
        else:
            encoder_inputs_embeds = word_embeds
        
        # Encode
        encoder_outputs = self.mt5.encoder(
            input_ids=None,
            attention_mask=attention_mask,
            inputs_embeds=encoder_inputs_embeds,
            return_dict=True,
        )
        
        # Generate
        generated_ids = self.mt5.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=early_stopping,
            **kwargs
        )
        
        return generated_ids
    
    def resize_token_embeddings(self, new_num_tokens: int):
        """Resize token embeddings (for adding special tokens)."""
        self.mt5.resize_token_embeddings(new_num_tokens)
    
    def save_pretrained(self, save_directory: str):
        """Save model to directory."""
        self.mt5.save_pretrained(save_directory)
        
        # Save syntax embeddings separately
        torch.save(
            self.syntax_embeddings.state_dict(),
            f"{save_directory}/syntax_embeddings.pt"
        )
    
    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        num_syntax_tags: int = 50,
        **kwargs
    ):
        """Load model from directory."""
        instance = cls(
            model_name=model_path,
            num_syntax_tags=num_syntax_tags,
            **kwargs
        )
        
        # Load syntax embeddings if available
        syntax_path = f"{model_path}/syntax_embeddings.pt"
        try:
            instance.syntax_embeddings.load_state_dict(
                torch.load(syntax_path)
            )
        except FileNotFoundError:
            print(f"Syntax embeddings not found at {syntax_path}, using initialized weights")
        
        return instance


class MT5Baseline(nn.Module):
    """
    Baseline mT5 model without syntax (for comparison).
    """
    
    def __init__(
        self,
        model_name: str = "google/mt5-base",
        freeze_encoder: bool = False,
        freeze_decoder: bool = False,
    ):
        """
        Initialize baseline mT5 model.
        
        Args:
            model_name: HuggingFace model name
            freeze_encoder: Whether to freeze encoder
            freeze_decoder: Whether to freeze decoder
        """
        super().__init__()
        
        self.mt5 = MT5ForConditionalGeneration.from_pretrained(model_name)
        self.config = self.mt5.config
        
        if freeze_encoder:
            for param in self.mt5.encoder.parameters():
                param.requires_grad = False
        
        if freeze_decoder:
            for param in self.mt5.decoder.parameters():
                param.requires_grad = False
    
    def forward(self, **kwargs):
        """Forward pass (standard mT5)."""
        return self.mt5(**kwargs)
    
    def generate(self, **kwargs):
        """Generate translations."""
        return self.mt5.generate(**kwargs)
    
    def resize_token_embeddings(self, new_num_tokens: int):
        """Resize token embeddings."""
        self.mt5.resize_token_embeddings(new_num_tokens)
    
    def save_pretrained(self, save_directory: str):
        """Save model."""
        self.mt5.save_pretrained(save_directory)
    
    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs):
        """Load model."""
        return cls(model_name=model_path, **kwargs)


if __name__ == "__main__":
    # Test model initialization
    print("Testing mT5 Syntax-Aware Model\n")
    
    print("1. Loading syntax-aware model...")
    model_syntax = MT5SyntaxAware(model_name="google/mt5-small", num_syntax_tags=50)
    print(f"   Model loaded with {sum(p.numel() for p in model_syntax.parameters())/1e6:.1f}M parameters")
    
    print("\n2. Loading baseline model...")
    model_baseline = MT5Baseline(model_name="google/mt5-small")
    print(f"   Model loaded with {sum(p.numel() for p in model_baseline.parameters())/1e6:.1f}M parameters")
    
    print("\n3. Testing forward pass...")
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    syntax_ids = torch.randint(0, 50, (batch_size, seq_len))
    labels = torch.randint(0, 1000, (batch_size, seq_len))
    
    outputs = model_syntax(input_ids=input_ids, syntax_ids=syntax_ids, labels=labels)
    print(f"   Loss: {outputs.loss.item():.4f}")
    print(f"   Logits shape: {outputs.logits.shape}")
    
    print("\n✓ Model tests passed!")
