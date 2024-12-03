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
        
        # Add embedding layer
        self.embedding = nn.Linear(1, hidden_size)  # Convert scalar inputs to hidden dim
        
        # Initialize transformer components
        self.encoder = self._build_encoder()
        
        # Add output projection for vocabulary
        self.output_projection = nn.Linear(hidden_size, vocab_size)
        
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
    
    def quantize_model(self):
        """Apply dynamic quantization for CPU compatibility"""
        try:
            # Quantize embedding
            self.embedding = torch.quantization.quantize_dynamic(
                self.embedding,
                {nn.Linear},
                dtype=torch.qint8
            )
            
            # Quantize encoder
            self.encoder = torch.quantization.quantize_dynamic(
                self.encoder,
                {nn.Linear},
                dtype=torch.qint8
            )
            
            # Quantize output projection
            self.output_projection = torch.quantization.quantize_dynamic(
                self.output_projection,
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
        # Handle input shape - expect (batch_size, seq_len)
        if len(src.shape) == 2:
            src = src.unsqueeze(-1)  # Add feature dimension
            
        # Convert to hidden dimension
        x = self.embedding(src)
        
        # Process through encoder
        for enc_layer in self.encoder:
            x = enc_layer(x)
        
        # Project to vocabulary size
        logits = self.output_projection(x)
        return logits