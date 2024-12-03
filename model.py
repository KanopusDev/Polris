import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

class CPUOptimizedTransformer(nn.Module):
    def __init__(self, layers=16, hidden_size=768, quantization_level='int8'):
        super().__init__()
        self.layers = layers
        self.hidden_size = hidden_size
        
        # Initialize transformer components
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()
        
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
        """Create quantization-friendly encoder layer without LayerNorm"""
        class QuantizableFeedForward(nn.Module):
            def __init__(self, d_model, dim_feedforward):
                super().__init__()
                self.linear1 = nn.Linear(d_model, dim_feedforward)
                self.linear2 = nn.Linear(dim_feedforward, d_model)
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(0.1)
            
            def forward(self, x):
                return self.dropout(self.linear2(self.relu(self.linear1(x))))

        class QuantizableAttention(nn.Module):
            def __init__(self, d_model, nhead):
                super().__init__()
                self.d_k = d_model // nhead
                self.nhead = nhead
                self.scaling = self.d_k ** -0.5  # Pre-compute scaling factor
                self.query = nn.Linear(d_model, d_model)
                self.key = nn.Linear(d_model, d_model)
                self.value = nn.Linear(d_model, d_model)
                self.out = nn.Linear(d_model, d_model)
                
            def forward(self, x):
                batch_size = x.size(0)
                
                # Linear transformations
                q = self.query(x).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
                k = self.key(x).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
                v = self.value(x).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
                
                # Scaled dot-product attention
                scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
                attn = F.softmax(scores, dim=-1)
                
                # Apply attention to values
                x = torch.matmul(attn, v)
                x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.nhead * self.d_k)
                return self.out(x)

        class QuantizableEncoderLayer(nn.Module):
            def __init__(self, d_model, nhead, dim_feedforward):
                super().__init__()
                self.self_attn = QuantizableAttention(d_model, nhead)
                self.feed_forward = QuantizableFeedForward(d_model, dim_feedforward)
                self.dropout1 = nn.Dropout(0.1)
                self.dropout2 = nn.Dropout(0.1)
                
            def forward(self, x):
                # Self attention
                att_output = self.dropout1(self.self_attn(x))
                x = x + att_output  # Residual connection
                
                # Feed forward
                ff_output = self.dropout2(self.feed_forward(x))
                x = x + ff_output  # Residual connection
                return x

        return QuantizableEncoderLayer(
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
        """Implement basic quantization for CPU"""
        # Ensure model is in training mode before preparing
        self.train()
        
        # Basic quantization config
        self.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        
        # Prepare and calibrate
        torch.quantization.prepare(self, inplace=True)
        self._run_calibration(num_batches=10)
        
        # Convert to quantized model
        self.eval()
        torch.quantization.convert(self, inplace=True)
    
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
        
        # Handle decoder if in training mode
        if self.training and tgt is not None:
            if len(tgt.shape) == 2:
                tgt = tgt.unsqueeze(-1).expand(-1, -1, self.hidden_size)
            decoder_output = tgt
            for dec_layer in self.decoder:
                decoder_output = dec_layer(
                    decoder_output,
                    encoder_output
                )
            return decoder_output
        
        return encoder_output