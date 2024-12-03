import torch
import torch.nn as nn
from intel_extension_for_pytorch import cpu
import bitsandbytes as bnb

class CPUOptimizedTransformer(nn.Module):
    def __init__(self, layers=16, hidden_size=768, quantization_level='int8'):
        super().__init__()
        self.layers = layers
        self.hidden_size = hidden_size
        
        # Initialize transformer components
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()
        
        # Quantize model if specified
        if quantization_level == 'int8':
            self.quantize_model()
    
    def _build_encoder(self):
        return nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.hidden_size,
                nhead=8,
                dim_feedforward=self.hidden_size * 4,
                batch_first=True
            ) for _ in range(self.layers)
        ])
    
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
        """Convert model to 8-bit quantization"""
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