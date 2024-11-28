import torch
import torch.nn as nn
import math
from typing import Optional

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0)]

class EnhancedCodeTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 512, nhead: int = 8,
                 num_layers: int = 6, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4*d_model,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers)
        self.decoder = nn.Linear(d_model, vocab_size)
        
        # Initialize parameters
        self._init_parameters()

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src: torch.Tensor, 
                src_mask: Optional[torch.Tensor] = None,
                src_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer(src, src_mask, src_padding_mask)
        return self.decoder(output)

    def generate(self, prompt: torch.Tensor, max_length: int = 100, 
                temperature: float = 1.0, top_k: int = 50) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            for _ in range(max_length):
                output = self(prompt)
                next_token_logits = output[:, -1, :] / temperature
                top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                probs = torch.softmax(top_k_logits, dim=-1)
                next_token = top_k_indices[0][torch.multinomial(probs[0], 1)]
                prompt = torch.cat([prompt, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
        return prompt

    def generate_with_context(self, 
                            prompt: torch.Tensor,
                            conversation_context: str,
                            max_length: int = 100,
                            temperature: float = 0.8,
                            top_p: float = 0.9,
                            top_k: int = 50) -> torch.Tensor:
        """Generate response considering conversation context."""
        self.eval()
        with torch.no_grad():
            # Prepare context-aware input
            context_tokens = self.tokenizer.encode(conversation_context)
            prompt_with_context = torch.cat([
                torch.tensor(context_tokens).unsqueeze(0),
                prompt
            ], dim=1).to(self.device)

            # Initialize attention mask for context
            attention_mask = torch.ones_like(prompt_with_context)
            
            generated_tokens = []
            past_key_values = None
            
            for _ in range(max_length):
                # Forward pass with cached key/values for efficiency
                outputs = self(prompt_with_context,
                             attention_mask=attention_mask,
                             past_key_values=past_key_values,
                             use_cache=True)
                
                next_token_logits = outputs.logits[:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float('inf')
                
                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = -float('inf')
                
                # Sample next token
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Stop if we predict EOS
                if next_token.item() == self.tokenizer.special_tokens['EOS']:
                    break
                    
                generated_tokens.append(next_token.item())
                prompt_with_context = torch.cat([prompt_with_context, next_token], dim=1)
                attention_mask = torch.cat([attention_mask, attention_mask.new_ones((1, 1))], dim=1)
                past_key_values = outputs.past_key_values
                
            return torch.tensor(generated_tokens)

    def init_cache(self):
        """Initialize cache for faster inference."""
        self.key_value_cache = {}
        
    def clear_cache(self):
        """Clear the key-value cache."""
        self.key_value_cache = {}

    