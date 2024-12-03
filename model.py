import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class CPUOptimizedTransformer(nn.Module):
    def __init__(self, layers=16, hidden_size=768, quantization_level='int8'):
        super().__init__()
        self.layers = layers
        self.hidden_size = hidden_size
        
        # Initialize transformer components with CPU-specific settings
        torch.set_num_threads(4)  # Optimize CPU thread usage
        torch.set_num_interop_threads(1)  # Optimize interop threads
        
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()
        
        # Quantize model if specified using standard PyTorch quantization
        if quantization_level == 'int8':
            self.quantize_model()
    
    def _build_encoder(self):
        return nn.ModuleList([
            self._create_optimized_encoder_layer() 
            for _ in range(self.layers)
        ])
    
    def _create_optimized_encoder_layer(self):
        # Use PyTorch's native optimizations
        layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=8,
            dim_feedforward=self.hidden_size * 4,
            batch_first=True,
            norm_first=True  # Better performance on CPU
        )
        # Enable fast path for better CPU performance
        layer.norm1.eps = 1e-5
        layer.norm2.eps = 1e-5
        return layer
    
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
        """Standard PyTorch quantization"""
        self.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        torch.quantization.prepare(self, inplace=True)
        torch.quantization.convert(self, inplace=True)
    
    def forward(self, src, tgt=None):
        # Encoder
        encoder_output = src
        for enc_layer in self.encoder:
            encoder_output = enc_layer(encoder_output)
            
        # Decoder (if in training mode)
        if self.training and tgt is not None:
            decoder_output = tgt
            for dec_layer in self.decoder:
                decoder_output = dec_layer(decoder_output, encoder_output)
            return decoder_output
            
        return encoder_output