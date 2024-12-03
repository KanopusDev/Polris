import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

class CPUOptimizedTransformer(nn.Module):
    def __init__(self, layers=16, hidden_size=768, quantization_level='int8', vocab_size=32000):
        super().__init__()
        self.layers = layers
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        
        # Initialize transformer components
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()
        
        # Add output projection for vocabulary
        self.output_projection = nn.Linear(hidden_size, vocab_size)
        
        # Add memory format optimization
        self = self.to(memory_format=torch.channels_last)
        
        # Quantize model if specified
        if quantization_level == 'int8':
            self.quantize_model()
    
    def _build_encoder(self):
        return nn.ModuleList([
            self._create_optimized_encoder_layer() 
            for _ in range(self.layers)
        ])
    
    def _create_optimized_encoder_layer(self):
        """Create CPU-friendly encoder layer"""
        class SimplifiedAttention(nn.Module):
            def __init__(self, d_model, nhead):
                super().__init__()
                self.d_k = d_model // nhead
                self.nhead = nhead
                self.scaling = self.d_k ** -0.5
                # Single linear layer for all projections
                self.qkv_proj = nn.Linear(d_model, 3 * d_model)
                self.out_proj = nn.Linear(d_model, d_model)
                
            def forward(self, x):
                batch_size = x.size(0)
                # Single matrix multiplication for Q, K, V projections
                qkv = self.qkv_proj(x)
                qkv = qkv.reshape(batch_size, -1, 3, self.nhead, self.d_k).permute(2, 0, 3, 1, 4)
                q, k, v = qkv[0], qkv[1], qkv[2]
                
                # Scaled dot-product attention
                scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
                attn = F.softmax(scores, dim=-1)
                x = torch.matmul(attn, v)
                x = x.transpose(1, 2).reshape(batch_size, -1, self.nhead * self.d_k)
                return self.out_proj(x)

        class SimplifiedEncoderLayer(nn.Module):
            def __init__(self, d_model, nhead, dim_feedforward):
                super().__init__()
                self.self_attn = SimplifiedAttention(d_model, nhead)
                self.linear1 = nn.Linear(d_model, dim_feedforward)
                self.linear2 = nn.Linear(dim_feedforward, d_model)
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(0.1)
                
            def forward(self, x):
                # Self attention and residual
                x = x + self.self_attn(x)
                # FFN and residual
                x = x + self.dropout(self.linear2(self.relu(self.linear1(x))))
                return x

        return SimplifiedEncoderLayer(
            d_model=self.hidden_size,
            nhead=8,
            dim_feedforward=self.hidden_size * 4
        )
    
    def _build_decoder(self):
        return nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=self.hidden_size,
                nhead=8,
                dim_feedforward=self.hidden_size * 4,
                batch_first=True
            ) for _ in range(self.layers)
        ])
    
    def quantize_model(self):
        """Apply dynamic quantization for CPU compatibility"""
        try:
            # Apply dynamic quantization to linear layers
            self.encoder = torch.quantization.quantize_dynamic(
                self.encoder,
                {nn.Linear},  # Quantize only linear layers
                dtype=torch.qint8
            )
            self.decoder = torch.quantization.quantize_dynamic(
                self.decoder,
                {nn.Linear},
                dtype=torch.qint8
            )
        except Exception as e:
            print(f"Quantization failed: {str(e)}. Continuing with unquantized model.")
    
    def _calibrate_model(self, num_batches=10):
        """Calibrate model with dummy data"""
        self.eval()
        with torch.no_grad():
            for _ in range(num_batches):
                # Create dummy batch with correct shape
                dummy_input = torch.randn(
                    2,  # batch_size
                    32,  # sequence_length
                    self.hidden_size,  # embedding_dim
                    device='cpu'
                )
                _ = self(dummy_input)
    
    def _run_calibration(self, num_batches=100):
        """Run calibration with proper observer updates"""
        self.eval()
        with torch.no_grad():
            for i in range(num_batches):
                dummy_input = torch.randn(4, 32, self.hidden_size)
                _ = self(dummy_input)
                
                # Update observers after each forward pass
                if i % 10 == 0:  # Update less frequently for efficiency
                    for module in self.modules():
                        if hasattr(module, 'activation_post_process'):
                            module.activation_post_process.calculate_qparams()

    def forward(self, src, tgt=None):
        """
        Forward pass with shape handling
        Args:
            src: Input tensor of shape (batch_size, seq_len) or (batch_size, seq_len, hidden_size)
            tgt: Optional target tensor for training
        """
        # Ensure inputs are in correct memory format
        if src.dim() >= 4:
            src = src.contiguous(memory_format=torch.channels_last)
        
        # Handle input shape
        if len(src.shape) == 2:
            # Add embedding dimension
            src = src.unsqueeze(-1).expand(-1, -1, self.hidden_size)
        
        # Validate input shape
        batch_size, seq_len, hidden_size = src.shape
        if hidden_size != self.hidden_size:
            raise ValueError(f"Input hidden size {hidden_size} doesn't match model hidden size {self.hidden_size}")

        # Process through encoder
        encoder_output = src
        for enc_layer in self.encoder:
            encoder_output = enc_layer(encoder_output)
        
        # Project to vocabulary size
        logits = self.output_projection(encoder_output)
        return logits  # Shape: [batch_size, seq_len, vocab_size]