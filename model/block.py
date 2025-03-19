import torch
import torch.nn.functional as F
from torch import nn

from model.attention import OrdinaryAttention, MixAttention, TimeAwareAttention, ComponentAttention


class TemporalTransformerBlock(nn.Module):
    """
    Transformer block for temporal dimension processing
    
    Args:
        model_dim: Dimension of model features
        ff_dim: Dimension of feed-forward network
        atten_dim: Dimension of attention heads
        head_num: Number of attention heads
        dropout: Dropout rate
    """
    def __init__(self, model_dim, ff_dim, atten_dim, head_num, dropout):
        super(TemporalTransformerBlock, self).__init__()
        # Self-attention layer
        self.attention = OrdinaryAttention(model_dim, atten_dim, head_num, dropout, True)
        
        # Feed-forward network as 1D convolutions
        self.conv1 = nn.Conv1d(in_channels=model_dim, out_channels=ff_dim, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=ff_dim, out_channels=model_dim, kernel_size=1)
        
        # Initialize weights
        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_in", nonlinearity="leaky_relu")
        nn.init.kaiming_normal_(self.conv2.weight, mode="fan_in", nonlinearity="leaky_relu")
        
        # Regularization
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(model_dim)
        
        # Activation function
        self.activation = F.gelu

    def forward(self, x):
        """
        Forward pass of temporal transformer block
        
        Args:
            x: Input tensor (batch_size, seq_len, model_dim)
            
        Returns:
            Processed tensor
        """
        # Self-attention with residual connection and layer norm
        x = self.attention(x)
        
        # Save residual for second residual connection
        residual = x.clone()
        
        # Feed-forward network with permutation for conv1d
        x = self.activation(self.conv1(x.permute(0, 2, 1)))
        x = self.dropout(self.conv2(x).permute(0, 2, 1))
        
        # Add residual and apply layer norm
        return self.norm(x + residual)


class SpatialTransformerBlock(nn.Module):
    """
    Transformer block for spatial/feature dimension processing
    
    Args:
        window_size: Size of sliding window
        model_dim: Dimension of model features
        ff_dim: Dimension of feed-forward network
        atten_dim: Dimension of attention heads
        head_num: Number of attention heads
        dropout: Dropout rate
    """
    def __init__(self, window_size, model_dim, ff_dim, atten_dim, head_num, dropout):
        super(SpatialTransformerBlock, self).__init__()
        self.window_size = window_size
        
        # Self-attention layer operating on feature dimension
        self.attention = OrdinaryAttention(window_size, atten_dim, head_num, dropout, True)
        
        # Feed-forward network as 1D convolutions
        self.conv1 = nn.Conv1d(in_channels=model_dim, out_channels=ff_dim, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=ff_dim, out_channels=model_dim, kernel_size=1)
        
        # Initialize weights
        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_in", nonlinearity="leaky_relu")
        nn.init.kaiming_normal_(self.conv2.weight, mode="fan_in", nonlinearity="leaky_relu")
        
        # Regularization
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(model_dim)
        
        # Activation function
        self.activation = F.gelu
        
    def forward(self, x):
        """
        Forward pass of spatial transformer block
        
        Args:
            x: Input tensor (batch_size, seq_len, model_dim)
            
        Returns:
            Processed tensor
        """
        # Permute to operate on feature dimension
        x = x.permute(0, 2, 1)  # [batch, model_dim, seq_len]
        
        # Self-attention with residual connection and layer norm
        x = self.attention(x)
        
        # Permute back
        x = x.permute(0, 2, 1)  # [batch, seq_len, model_dim]
        
        # Save residual for second residual connection
        residual = x.clone()
        
        # Feed-forward network with permutation for conv1d
        x = self.activation(self.conv1(x.permute(0, 2, 1)))
        x = self.dropout(self.conv2(x).permute(0, 2, 1))
        
        # Add residual and apply layer norm
        return self.norm(x + residual)


class SpatioTemporalTransformerBlock(nn.Module):
    """
    Combined spatio-temporal transformer block
    
    Args:
        window_size: Size of sliding window
        model_dim: Dimension of model features
        ff_dim: Dimension of feed-forward network
        atten_dim: Dimension of attention heads
        head_num: Number of attention heads
        dropout: Dropout rate
    """
    def __init__(self, window_size, model_dim, ff_dim, atten_dim, head_num, dropout):
        super(SpatioTemporalTransformerBlock, self).__init__()
        
        # Temporal and spatial transformer blocks
        self.temporal_block = TemporalTransformerBlock(model_dim, ff_dim, atten_dim, head_num, dropout)
        self.spatial_block = SpatialTransformerBlock(window_size, model_dim, ff_dim, atten_dim, head_num, dropout)
        
        # Integration layer
        self.conv1 = nn.Conv1d(in_channels=2 * model_dim, out_channels=ff_dim, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=ff_dim, out_channels=model_dim, kernel_size=1)
        
        # Initialize weights
        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_in", nonlinearity="leaky_relu")
        nn.init.kaiming_normal_(self.conv2.weight, mode="fan_in", nonlinearity="leaky_relu")
        
        # Regularization
        self.dropout = nn.Dropout(dropout)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(2 * model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        
        # Activation function
        self.activation = F.gelu
        
    def forward(self, x):
        """
        Forward pass of spatio-temporal transformer block
        
        Args:
            x: Input tensor (batch_size, seq_len, model_dim)
            
        Returns:
            Processed tensor
        """
        # Process in temporal and spatial dimensions
        temporal_x = self.temporal_block(x)
        spatial_x = self.spatial_block(x)
        
        # Concatenate results
        x = torch.cat([temporal_x, spatial_x], dim=-1)
        x = self.norm1(x)
        
        # Integration with feed-forward network
        x = self.activation(self.conv1(x.permute(0, 2, 1)))
        x = self.dropout(self.conv2(x).permute(0, 2, 1))
        
        # Final layer norm
        return self.norm2(x)


class ComponentAwareBlock(nn.Module):
    """
    Transformer block with awareness of frequency components
    
    Args:
        window_size: Size of sliding window
        model_dim: Dimension of model features
        ff_dim: Dimension of feed-forward network
        atten_dim: Dimension of attention heads
        head_num: Number of attention heads
        dropout: Dropout rate
    """
    def __init__(self, window_size, model_dim, ff_dim, atten_dim, head_num, dropout):
        super(ComponentAwareBlock, self).__init__()
        
        # Component attention for integration
        self.component_attention = ComponentAttention(model_dim, atten_dim, head_num, dropout)
        
        # Standard transformer processing
        self.transformer = SpatioTemporalTransformerBlock(
            window_size, model_dim, ff_dim, atten_dim, head_num, dropout
        )
        
    def forward(self, x, high_freq, low_freq):
        """
        Forward pass with frequency component conditioning
        
        Args:
            x: Input tensor
            high_freq: High frequency component
            low_freq: Low frequency component
            
        Returns:
            Component-aware processed tensor
        """
        # Integrate components
        x = self.component_attention(x, high_freq, low_freq)
        
        # Process with transformer
        return self.transformer(x)


class DecompositionBlock(nn.Module):
    """
    Block for signal decomposition
    
    Args:
        model_dim: Dimension of model features
        ff_dim: Dimension of feed-forward network
        atten_dim: Dimension of attention heads
        feature_num: Number of input features
        head_num: Number of attention heads
        dropout: Dropout rate
    """
    def __init__(self, model_dim, ff_dim, atten_dim, feature_num, head_num, dropout):
        super(DecompositionBlock, self).__init__()
        
        # Mixed attention for decomposition
        self.mixed_attention = MixAttention(model_dim, atten_dim, head_num, dropout, False)
        self.ordinary_attention = OrdinaryAttention(model_dim, atten_dim, head_num, dropout, True)
        
        # Feed-forward network for processing
        self.conv1 = nn.Conv1d(in_channels=model_dim, out_channels=ff_dim, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=ff_dim, out_channels=model_dim, kernel_size=1)
        
        # Initialize weights
        nn.init.kaiming_normal_(self.conv1.weight, mode="fan_in", nonlinearity="leaky_relu")
        nn.init.kaiming_normal_(self.conv2.weight, mode="fan_in", nonlinearity="leaky_relu")
        
        # Output projections
        self.fc1 = nn.Linear(model_dim, ff_dim, bias=True)
        self.fc2 = nn.Linear(ff_dim, feature_num, bias=True)
        
        # Regularization
        self.dropout = nn.Dropout(dropout)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        
        # Activation function
        self.activation = F.gelu
        
    def forward(self, trend, time):
        """
        Forward pass for decomposition
        
        Args:
            trend: Trend component
            time: Time features
            
        Returns:
            Stable component and updated trend
        """
        # Apply mixed attention for time-aware processing
        stable = self.mixed_attention(trend, time, trend, time, trend, time)
        
        # Further refine with self-attention
        stable = self.ordinary_attention(stable)
        
        # Save residual
        residual = stable.clone()
        
        # Process with feed-forward network
        stable = self.activation(self.conv1(stable.permute(0, 2, 1)))
        stable = self.dropout(self.conv2(stable).permute(0, 2, 1))
        stable = self.norm1(stable + residual)
        
        # Update trend by subtracting stable component
        trend = self.norm2(trend - stable)
        
        # Project stable component to feature space
        stable = self.fc2(self.activation(self.fc1(stable)))
        
        return stable, trend


class WaveletHHTFusionBlock(nn.Module):
    """
    Fusion block for wavelet and HHT decompositions
    
    Args:
        model_dim: Dimension of model features
        feature_num: Number of input features
        fusion_type: Type of fusion ('concat', 'attention', or 'weighted')
        dropout: Dropout rate
    """
    def __init__(self, model_dim, feature_num, fusion_type='attention', dropout=0.1):
        super(WaveletHHTFusionBlock, self).__init__()
        self.fusion_type = fusion_type
        
        # Projections for wavelet and HHT components
        self.wavelet_proj = nn.Linear(feature_num, model_dim)
        self.hht_proj = nn.Linear(feature_num, model_dim)
        
        # Fusion with attention
        if fusion_type == 'attention':
            self.fusion_attn = OrdinaryAttention(model_dim, model_dim // 8, 4, dropout)
            
        # Weighted fusion
        elif fusion_type == 'weighted':
            self.wavelet_weight = nn.Parameter(torch.ones(1))
            self.hht_weight = nn.Parameter(torch.ones(1))
            
        # Output projection
        self.out_proj = nn.Linear(model_dim, feature_num)
        
        # Regularization
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(feature_num)
        
    def forward(self, wavelet_comp, hht_comp):
        """
        Forward pass for fusion
        
        Args:
            wavelet_comp: Wavelet decomposition component
            hht_comp: HHT decomposition component
            
        Returns:
            Fused component
        """
        # Project components to model dimension
        wavelet_feat = self.wavelet_proj(wavelet_comp)
        hht_feat = self.hht_proj(hht_comp)
        
        # Apply fusion based on type
        if self.fusion_type == 'concat':
            # Simple concatenation and projection
            fused = wavelet_feat + hht_feat
            
        elif self.fusion_type == 'attention':
            # Attention-based fusion
            concat = (wavelet_feat + hht_feat) / 2
            fused = self.fusion_attn(concat)
            
        elif self.fusion_type == 'weighted':
            # Weighted combination
            w_weight = torch.sigmoid(self.wavelet_weight)
            h_weight = torch.sigmoid(self.hht_weight)
            weights_sum = w_weight + h_weight
            
            fused = (w_weight * wavelet_feat + h_weight * hht_feat) / weights_sum
            
        # Project back to feature space
        out = self.out_proj(fused)
        out = self.dropout(out)
        
        # Residual connection with average of inputs
        residual = (wavelet_comp + hht_comp) / 2
        return self.norm(out + residual)