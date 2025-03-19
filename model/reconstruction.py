import torch
from torch import nn
import torch.nn.functional as F

from model.block import SpatioTemporalTransformerBlock
from model.embedding import TimeEmbedding, DataEmbedding, PositionEmbedding

class Reconstruction(nn.Module):
    """
    Reconstruction module to generate the final output from decomposed components
    """
    def __init__(self, window_size, model_dim, ff_dim, atten_dim, feature_num, time_num, block_num, head_num, dropout):
        super(Reconstruction, self).__init__()
        self.time_embed = TimeEmbedding(model_dim, time_num)
        self.data_embedding = DataEmbedding(model_dim, feature_num)
        self.position_embedding = PositionEmbedding(model_dim)
        
        # Embeddings for high and low frequency components
        self.high_freq_embedding = nn.Linear(feature_num, model_dim)
        self.low_freq_embedding = nn.Linear(feature_num, model_dim)
        
        # Cross-attention for component integration
        self.decoder_blocks = nn.ModuleList()
        for i in range(block_num):
            dp = 0 if i == block_num - 1 else dropout
            self.decoder_blocks.append(
                SpatioTemporalTransformerBlock(window_size, model_dim, ff_dim, atten_dim, head_num, dp)
            )
        
        # Output projection
        self.fc1 = nn.Linear(model_dim, model_dim // 2, bias=True)
        self.fc2 = nn.Linear(model_dim // 2, feature_num, bias=True)
        
        # Component gating mechanism
        self.high_freq_gate = nn.Sequential(
            nn.Linear(model_dim, 1),
            nn.Sigmoid()
        )
        self.low_freq_gate = nn.Sequential(
            nn.Linear(model_dim, 1),
            nn.Sigmoid()
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.norm_out = nn.LayerNorm(feature_num)

    def forward(self, noise, high_freq, low_freq, time):
        """
        Reconstruct signal from noise and frequency components
        
        Args:
            noise: Noisy input signal (batch_size, window_size, feature_dim)
            high_freq: High frequency component (batch_size, window_size, feature_dim)
            low_freq: Low frequency component (batch_size, window_size, feature_dim)
            time: Time features (batch_size, window_size, time_features)
            
        Returns:
            Reconstructed signal
        """
        batch_size, seq_len, feature_dim = noise.shape
        
        # Embed all inputs
        noise_embed = self.data_embedding(noise)
        high_freq_embed = self.high_freq_embedding(high_freq)
        low_freq_embed = self.low_freq_embedding(low_freq)
        time_embed = self.time_embed(time)
        pos_embed = self.position_embedding(noise)
        
        # Apply gating to control contribution of each component
        high_freq_importance = self.high_freq_gate(high_freq_embed)
        low_freq_importance = self.low_freq_gate(low_freq_embed)
        
        # Normalize gates to sum to 1
        gate_sum = high_freq_importance + low_freq_importance
        high_freq_importance = high_freq_importance / (gate_sum + 1e-8)
        low_freq_importance = low_freq_importance / (gate_sum + 1e-8)
        
        # Apply gates
        high_freq_embed = high_freq_embed * high_freq_importance
        low_freq_embed = low_freq_embed * low_freq_importance
        
        # Combine embeddings
        x = noise_embed + pos_embed + time_embed + high_freq_embed + low_freq_embed
        x = self.norm1(x)
        
        # Process through transformer blocks
        for block in self.decoder_blocks:
            x = block(x)
        
        # Final processing
        x = self.norm2(x)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        
        # Residual connection with input components
        x = x + high_freq + low_freq
        x = self.norm_out(x)
        
        return x