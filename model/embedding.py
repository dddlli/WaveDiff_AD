import math
import torch
from torch import nn
import torch.nn.functional as F

class PositionEmbedding(nn.Module):
    """
    Sinusoidal position embedding
    
    Args:
        model_dim: Dimension of model features
        max_len: Maximum sequence length
    """
    def __init__(self, model_dim, max_len=5000):
        super(PositionEmbedding, self).__init__()
        
        # Create position encoding matrix
        pe = torch.zeros(max_len, model_dim)
        pe.require_grad = False  # Fixed encoding, not learned
        
        # Compute position encodings
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, model_dim, 2).float() * (-math.log(10000.0) / model_dim)
        )
        
        # Sinusoidal pattern
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add batch dimension
        
        # Register as buffer (not a parameter)
        self.register_buffer("pe", pe)
        self.norm = nn.LayerNorm(model_dim)
        
    def forward(self, x):
        """
        Get position embeddings for input
        
        Args:
            x: Input tensor (batch_size, seq_len, dim)
            
        Returns:
            Position embeddings
        """
        # Extract embeddings for the current sequence length
        pe = self.pe[:, :x.size(1), :]
        return self.norm(pe)


class TimeEmbedding(nn.Module):
    """
    Time feature embedding
    
    Args:
        model_dim: Dimension of model features
        time_num: Number of time features
    """
    def __init__(self, model_dim, time_num):
        super(TimeEmbedding, self).__init__()
        
        # Project time features to model dimension
        self.conv = nn.Conv1d(in_channels=time_num, out_channels=model_dim, kernel_size=1)
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_in", nonlinearity="leaky_relu")
        
        # Add learnable frequency modulators
        self.freq_mod = nn.Parameter(torch.ones(time_num, 1))
        self.phase_mod = nn.Parameter(torch.zeros(time_num, 1))
        
        self.norm = nn.LayerNorm(model_dim)
        
    def forward(self, x):
        """
        Get time embeddings for input
        
        Args:
            x: Time features (batch_size, seq_len, time_num)
            
        Returns:
            Time embeddings
        """
        batch_size, seq_len, time_num = x.shape
        
        # Apply frequency and phase modulation
        # This enhances the model's ability to capture different periodicities
        x_mod = x.clone()
        for i in range(time_num):
            x_mod[:, :, i] = torch.sin(self.freq_mod[i] * x[:, :, i] + self.phase_mod[i])
            
        # Apply convolution to project to model dimension
        x = self.conv(x_mod.permute(0, 2, 1)).permute(0, 2, 1)
        return self.norm(x)


class DataEmbedding(nn.Module):
    """
    Data feature embedding
    
    Args:
        model_dim: Dimension of model features
        feature_num: Number of input features
    """
    def __init__(self, model_dim, feature_num):
        super(DataEmbedding, self).__init__()
        
        # Project features to model dimension
        self.conv = nn.Conv1d(in_channels=feature_num, out_channels=model_dim, kernel_size=1)
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_in", nonlinearity="leaky_relu")
        
        # Additional projection for feature interactions
        self.feature_interaction = nn.Sequential(
            nn.Linear(feature_num, feature_num),
            nn.GELU(),
            nn.Linear(feature_num, feature_num)
        )
        
    def forward(self, x):
        """
        Get data embeddings for input
        
        Args:
            x: Input features (batch_size, seq_len, feature_num)
            
        Returns:
            Data embeddings
        """
        # Apply feature interaction module
        x_interaction = self.feature_interaction(x)
        x = x + x_interaction
        
        # Project to model dimension
        x = self.conv(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x


class ComponentEmbedding(nn.Module):
    """
    Embedding for frequency components (high/low)
    
    Args:
        model_dim: Dimension of model features
        feature_num: Number of input features
        component_type: Type of component ('high' or 'low')
    """
    def __init__(self, model_dim, feature_num, component_type='high'):
        super(ComponentEmbedding, self).__init__()
        self.component_type = component_type
        
        # Projection to model dimension
        self.conv = nn.Conv1d(in_channels=feature_num, out_channels=model_dim, kernel_size=1)
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_in", nonlinearity="leaky_relu")
        
        # Component-specific processing
        if component_type == 'high':
            # High frequency component gets multi-scale processing
            self.multi_scale = nn.ModuleList([
                nn.Conv1d(in_channels=feature_num, out_channels=feature_num, 
                          kernel_size=k, padding=k//2)
                for k in [3, 5, 7]
            ])
        else:
            # Low frequency component gets smoother processing
            self.smooth = nn.Sequential(
                nn.Conv1d(in_channels=feature_num, out_channels=feature_num, 
                          kernel_size=5, padding=2),
                nn.GELU(),
                nn.Conv1d(in_channels=feature_num, out_channels=feature_num, 
                          kernel_size=5, padding=2)
            )
            
        self.norm = nn.LayerNorm(model_dim)
        
    def forward(self, x):
        """
        Get component embeddings for input
        
        Args:
            x: Input features (batch_size, seq_len, feature_num)
            
        Returns:
            Component embeddings
        """
        batch_size, seq_len, feature_num = x.shape
        
        # Apply component-specific processing
        if self.component_type == 'high':
            # Multi-scale processing for high frequency
            x_perm = x.permute(0, 2, 1)  # [batch, feature, seq]
            multi_scale_out = [conv(x_perm) for conv in self.multi_scale]
            x_processed = torch.stack(multi_scale_out).mean(0)
            x = x_processed.permute(0, 2, 1)  # [batch, seq, feature]
        else:
            # Smooth processing for low frequency
            x_perm = x.permute(0, 2, 1)  # [batch, feature, seq]
            x_processed = self.smooth(x_perm)
            x = x_processed.permute(0, 2, 1)  # [batch, seq, feature]
            
        # Project to model dimension
        x = self.conv(x.permute(0, 2, 1)).permute(0, 2, 1)
        return self.norm(x)


class TimeStepEmbedding(nn.Module):
    """
    Embedding for diffusion timesteps
    
    Args:
        model_dim: Dimension of model features
        max_steps: Maximum number of diffusion steps
    """
    def __init__(self, model_dim, max_steps=1000):
        super(TimeStepEmbedding, self).__init__()
        
        # Sinusoidal embeddings for timesteps
        half_dim = model_dim // 2
        exponent = -math.log(10000) / (half_dim - 1)
        self.register_buffer(
            'freq_bands',
            torch.exp(torch.arange(half_dim, dtype=torch.float32) * exponent)
        )
        
        # Projection layers
        self.proj1 = nn.Linear(model_dim, model_dim * 2)
        self.proj2 = nn.Linear(model_dim * 2, model_dim)
        
        self.norm = nn.LayerNorm(model_dim)
        
    def forward(self, timesteps):
        """
        Get timestep embeddings
        
        Args:
            timesteps: Diffusion timesteps (batch_size,)
            
        Returns:
            Timestep embeddings
        """
        # Expand timesteps and compute frequencies
        t_freq = timesteps.float().unsqueeze(-1) * self.freq_bands.unsqueeze(0)
        
        # Get sinusoidal encodings
        embeddings = torch.cat([torch.sin(t_freq), torch.cos(t_freq)], dim=-1)
        
        # Project through MLP
        embeddings = self.proj1(embeddings)
        embeddings = F.gelu(embeddings)
        embeddings = self.proj2(embeddings)
        
        return self.norm(embeddings)