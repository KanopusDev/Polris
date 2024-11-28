import sentencepiece as spm
from typing import List, Dict, Optional
import pickle
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from functools import lru_cache

class EnhancedCodeTokenizer:
    def __init__(self, vocab_size: int = 3500, model_path: Optional[str] = None):  # Reduced from 32000
        self.vocab_size = vocab_size
        self.model_path = model_path
        self.sp_model = None
        self.special_tokens = {
            'PAD': '[PAD]',
            'UNK': '[UNK]',
            'BOS': '[BOS]',
            'EOS': '[EOS]',
            'MASK': '[MASK]',
            'SEP': '[SEP]',
            # Language-specific tokens
            'PYTHON_START': '<PYTHON>',
            'JAVA_START': '<JAVA>',
            'CPP_START': '<CPP>',
            # Context tokens
            'CONTEXT_START': '<CTX>',
            'CONTEXT_END': '</CTX>',
            'USER_START': '<USER>',
            'ASSISTANT_START': '<ASSISTANT>'
        }
        self.cache_dir = Path('tokenizer_cache')
        self.cache_dir.mkdir(exist_ok=True)
        self._setup_logging()

    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('tokenizer.log'),
                logging.StreamHandler()
            ]
        )

    def train(self, text_data: List[str], num_threads: int = 8):
        """Train the tokenizer on code data using SentencePiece."""
        if not text_data:
            raise ValueError("No training data provided")
            
        logging.info(f"Training tokenizer on {len(text_data)} samples")
        
        # Clean and filter data
        cleaned_data = []
        for text in text_data:
            if isinstance(text, str) and text.strip():
                # Normalize whitespace and clean the text
                text = ' '.join(text.split())
                if len(text) >= 10:  # Skip very short snippets
                    cleaned_data.append(text)
        
        if not cleaned_data:
            raise ValueError("No valid training data after cleaning")
        
        # Prepare training data
        train_path = self.cache_dir / 'train.txt'
        with open(train_path, 'w', encoding='utf-8') as f:
            for text in cleaned_data:
                f.write(f"{text}\n")

        # Calculate actual vocab size accounting for special tokens
        effective_vocab_size = self.vocab_size - len(self.special_tokens)
        if effective_vocab_size <= 0:
            raise ValueError("Vocab size too small to accommodate special tokens")

        # Train SentencePiece model with adjusted parameters
        model_prefix = str(self.cache_dir / 'code_tokenizer')
        try:
            spm.SentencePieceTrainer.train(
                input=str(train_path),
                model_prefix=model_prefix,
                vocab_size=effective_vocab_size,
                model_type='bpe',
                character_coverage=0.9995,  # Reduced from 1.0
                num_threads=num_threads,
                split_digits=True,
                split_by_whitespace=True,
                max_sentence_length=8192,  # Reduced from 16384
                byte_fallback=True,
                user_defined_symbols=list(self.special_tokens.values()),
                pad_id=0,
                unk_id=1,
                bos_id=2,
                eos_id=3,
                control_symbols=['[PAD]', '[UNK]', '[BOS]', '[EOS]'],
                input_sentence_size=100000,  # Limit training data size
                shuffle_input_sentence=True
            )
            
            self.sp_model = spm.SentencePieceProcessor()
            self.sp_model.load(f"{model_prefix}.model")
            self._save_model()
            
        except Exception as e:
            logging.error(f"SentencePiece training failed: {str(e)}")
            raise

    @lru_cache(maxsize=100000)
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs with caching."""
        if not text:
            return []

        if add_special_tokens:
            text = f"{self.special_tokens['BOS']} {text} {self.special_tokens['EOS']}"

        try:
            return self.sp_model.encode_as_ids(text)
        except Exception as e:
            logging.error(f"Error encoding text: {e}")
            return [self.sp_model.piece_to_id(self.special_tokens['UNK'])]

    def encode_batch(self, texts: List[str], num_threads: int = 4) -> List[List[int]]:
        """Parallel batch encoding."""
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            return list(executor.map(self.encode, texts))

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text."""
        try:
            text = self.sp_model.decode_ids(token_ids)
            if skip_special_tokens:
                for special_token in self.special_tokens.values():
                    text = text.replace(special_token, '')
            return text.strip()
        except Exception as e:
            logging.error(f"Error decoding tokens: {e}")
            return ""

    def _save_model(self):
        """Save tokenizer model and configuration."""
        model_path = self.cache_dir / 'tokenizer.pkl'
        config = {
            'vocab_size': self.vocab_size,
            'special_tokens': self.special_tokens,
            'model_path': str(self.cache_dir / 'code_tokenizer.model')
        }
        with open(model_path, 'wb') as f:
            pickle.dump(config, f)

    @classmethod
    def load(cls, model_path: str):
        """Load pre-trained tokenizer."""
        with open(model_path, 'rb') as f:
            config = pickle.load(f)
        
        tokenizer = cls(vocab_size=config['vocab_size'])
        tokenizer.special_tokens = config['special_tokens']
        tokenizer.sp_model = spm.SentencePieceProcessor()
        tokenizer.sp_model.load(config['model_path'])
        return tokenizer

    def get_language_tokens(self, language: str) -> List[int]:
        """Get language-specific tokens."""
        token_key = f'{language.upper()}_START'
        if token_key in self.special_tokens:
            return [self.sp_model.piece_to_id(self.special_tokens[token_key])]
        return []