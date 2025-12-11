"""
Back-translation data augmentation for LIBRAS.

This module provides back-translation functionality to augment training data
by translating PT-BR → EN → PT-BR or using other pivot languages.
"""

from typing import List, Dict, Optional
import random
from tqdm import tqdm


class BackTranslationAugmenter:
    """
    Data augmentation using back-translation.
    
    Note: This is a placeholder implementation. For production use,
    you would need to integrate with translation APIs like:
    - Google Translate API
    - DeepL API
    - MarianMT models from HuggingFace
    """
    
    def __init__(
        self,
        source_lang: str = "pt",
        pivot_lang: str = "en",
        use_api: bool = False,
        api_key: Optional[str] = None
    ):
        """
        Initialize back-translation augmenter.
        
        Args:
            source_lang: Source language code
            pivot_lang: Pivot language for back-translation
            use_api: Whether to use translation API
            api_key: API key for translation service
        """
        self.source_lang = source_lang
        self.pivot_lang = pivot_lang
        self.use_api = use_api
        self.api_key = api_key
        
        if use_api:
            self._init_api()
        else:
            self._init_local_model()
    
    def _init_api(self):
        """Initialize translation API."""
        # Placeholder for API initialization
        print("Warning: API translation not implemented. Using local model.")
        self._init_local_model()
    
    def _init_local_model(self):
        """Initialize local translation model."""
        try:
            from transformers import MarianMTModel, MarianTokenizer
            
            # Load PT → EN model
            model_name_pt_en = f"Helsinki-NLP/opus-mt-{self.source_lang}-{self.pivot_lang}"
            self.tokenizer_pt_en = MarianTokenizer.from_pretrained(model_name_pt_en)
            self.model_pt_en = MarianMTModel.from_pretrained(model_name_pt_en)
            
            # Load EN → PT model
            model_name_en_pt = f"Helsinki-NLP/opus-mt-{self.pivot_lang}-{self.source_lang}"
            self.tokenizer_en_pt = MarianTokenizer.from_pretrained(model_name_en_pt)
            self.model_en_pt = MarianMTModel.from_pretrained(model_name_en_pt)
            
            print(f"Loaded MarianMT models for {self.source_lang}↔{self.pivot_lang}")
            
        except Exception as e:
            print(f"Warning: Could not load translation models: {e}")
            print("Back-translation will use simple paraphrasing instead.")
            self.model_pt_en = None
            self.model_en_pt = None
    
    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        """
        Translate text from source to target language.
        
        Args:
            text: Input text
            src_lang: Source language
            tgt_lang: Target language
            
        Returns:
            Translated text
        """
        if self.model_pt_en is None:
            # Fallback: simple paraphrasing
            return self._simple_paraphrase(text)
        
        # Select appropriate model
        if src_lang == self.source_lang and tgt_lang == self.pivot_lang:
            tokenizer = self.tokenizer_pt_en
            model = self.model_pt_en
        elif src_lang == self.pivot_lang and tgt_lang == self.source_lang:
            tokenizer = self.tokenizer_en_pt
            model = self.model_en_pt
        else:
            return text
        
        # Translate
        inputs = tokenizer(text, return_tensors="pt", padding=True)
        translated = model.generate(**inputs)
        translation = tokenizer.decode(translated[0], skip_special_tokens=True)
        
        return translation
    
    def _simple_paraphrase(self, text: str) -> str:
        """
        Simple paraphrasing as fallback when translation models unavailable.
        
        This is a very basic implementation that makes minor modifications.
        """
        # Simple word replacements for Portuguese
        replacements = {
            'muito': 'bastante',
            'bastante': 'muito',
            'grande': 'enorme',
            'pequeno': 'diminuto',
            'bonito': 'belo',
            'feio': 'horrível',
            'bom': 'ótimo',
            'ruim': 'péssimo',
            'rápido': 'veloz',
            'lento': 'devagar',
        }
        
        words = text.split()
        modified = False
        
        for i, word in enumerate(words):
            word_lower = word.lower()
            if word_lower in replacements and random.random() < 0.3:
                words[i] = replacements[word_lower]
                modified = True
        
        return ' '.join(words) if modified else text
    
    def back_translate(self, text: str) -> str:
        """
        Perform back-translation: source → pivot → source.
        
        Args:
            text: Input text in source language
            
        Returns:
            Back-translated text
        """
        # Translate to pivot language
        pivot_text = self.translate(text, self.source_lang, self.pivot_lang)
        
        # Translate back to source language
        back_translated = self.translate(pivot_text, self.pivot_lang, self.source_lang)
        
        return back_translated
    
    def augment_dataset(
        self,
        data: List[Dict[str, str]],
        augmentation_ratio: float = 0.5,
        show_progress: bool = True
    ) -> List[Dict[str, str]]:
        """
        Augment dataset using back-translation.
        
        Args:
            data: Original dataset
            augmentation_ratio: Ratio of data to augment (0.0 to 1.0)
            show_progress: Whether to show progress bar
            
        Returns:
            Augmented dataset (original + back-translated)
        """
        num_to_augment = int(len(data) * augmentation_ratio)
        
        # Randomly select samples to augment
        samples_to_augment = random.sample(data, num_to_augment)
        
        augmented_data = []
        
        iterator = samples_to_augment
        if show_progress:
            iterator = tqdm(samples_to_augment, desc="Back-translating")
        
        for item in iterator:
            # Back-translate source text
            back_translated_source = self.back_translate(item['source'])
            
            # Keep original target (gloss doesn't change)
            augmented_data.append({
                'source': back_translated_source,
                'target': item['target']
            })
        
        # Combine original and augmented data
        combined_data = data + augmented_data
        
        # Shuffle
        random.shuffle(combined_data)
        
        print(f"Augmented dataset: {len(data)} → {len(combined_data)} samples")
        
        return combined_data


def test_back_translation():
    """Test back-translation augmentation."""
    print("Testing Back-Translation Augmentation\n")
    print("=" * 60)
    
    # Sample data
    data = [
        {"source": "O menino está jogando bola no parque.", "target": "MENINO JOGAR BOLA PARQUE"},
        {"source": "A professora ensina matemática.", "target": "PROFESSORA ENSINAR MATEMATICA"},
        {"source": "Eu vou ao mercado amanhã.", "target": "EU IR MERCADO AMANHA"},
    ]
    
    print(f"Original dataset: {len(data)} samples\n")
    
    # Initialize augmenter
    augmenter = BackTranslationAugmenter(source_lang="pt", pivot_lang="en")
    
    # Test single back-translation
    print("Testing single back-translation:")
    original = data[0]['source']
    back_translated = augmenter.back_translate(original)
    print(f"  Original:        {original}")
    print(f"  Back-translated: {back_translated}\n")
    
    # Test dataset augmentation
    print("Testing dataset augmentation:")
    augmented = augmenter.augment_dataset(data, augmentation_ratio=0.5, show_progress=False)
    print(f"  Augmented dataset: {len(augmented)} samples")
    
    print("\nSample augmented data:")
    for i, item in enumerate(augmented[:5]):
        print(f"  {i+1}. Source: {item['source']}")
        print(f"     Target: {item['target']}\n")
    
    print("=" * 60)


if __name__ == "__main__":
    test_back_translation()
