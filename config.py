from dataclasses import dataclass
from typing import Optional
from pydantic import BaseModel, validator

class ModelConfig(BaseModel):
    layers: int = 8  # Reduced for faster training
    hidden_size: int = 256  # Reduced for memory efficiency
    quantization: str = 'int8'
    vocab_size: int = 32000
    
    @validator('layers')
    def validate_layers(cls, v):
        if v <= 0:
            raise ValueError('Number of layers must be positive')
        return v
    
    @validator('hidden_size')
    def validate_hidden_size(cls, v):
        if v <= 0 or v % 64 != 0:
            raise ValueError('Hidden size must be positive and divisible by 64')
        return v

    @validator('vocab_size')
    def validate_vocab_size(cls, v):
        if v <= 0:
            raise ValueError('Vocabulary size must be positive')
        return v

class TrainingConfig(BaseModel):
    # Model config
    model: ModelConfig = ModelConfig()
    
    # Training parameters (moved from nested structure)
    batch_size: int = 8  # Increased for better training dynamics
    gradient_accumulation_steps: int = 16
    learning_rate: float = 3e-4
    max_seq_length: int = 512  # Reduced for memory efficiency
    epochs: int = 10
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    logging_steps: int = 100
    save_steps: int = 1000
    
    @validator('batch_size', 'gradient_accumulation_steps')
    def validate_positive(cls, v):
        if v <= 0:
            raise ValueError('Value must be positive')
        return v
        
    def validate(self):
        """Validate the complete configuration"""
        if self.batch_size * self.gradient_accumulation_steps > 512:
            raise ValueError('Effective batch size too large')