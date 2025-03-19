import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalPositionEmbeddings(nn.Module):
    """
    Sinusoidal position embeddings for diffusion timesteps
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        """
        Args:
            time: Timesteps (batch_size,)
            
        Returns:
            Time embeddings (batch_size, dim)
        """
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class TimeRouting(nn.Module):
    """
    Time-dependent routing mechanism for component weighting
    """
    def __init__(self, time_steps, hidden_dim):
        super().__init__()
        self.time_steps = time_steps
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Routing weights generator
        self.routing_weights = nn.Sequential(
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, t):
        """
        Generate routing weights based on diffusion timestep
        
        Args:
            t: Timestep tensor (batch_size,)
            
        Returns:
            Routing weights for high and low frequency components
        """
        # Normalize time to [0, 1]
        t_norm = t.float() / self.time_steps
        
        # Get time embeddings
        t_emb = self.time_mlp(t_norm)
        
        # Generate routing weights
        weights = self.routing_weights(t_emb)
        
        return {
            'high_freq_weight': weights[:, 0],
            'low_freq_weight': weights[:, 1]
        }


class ConditionalDenoisingNetwork(nn.Module):
    """
    Denoising network conditioned on time and signal components
    """
    def __init__(self, feature_dim, window_size, time_steps, hidden_dim=256):
        super().__init__()
        self.feature_dim = feature_dim
        self.window_size = window_size
        self.time_steps = time_steps
        
        # Time embedding
        self.time_embedder = SinusoidalPositionEmbeddings(hidden_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Component condition processing
        self.high_freq_encoder = nn.Linear(feature_dim, hidden_dim)
        self.low_freq_encoder = nn.Linear(feature_dim, hidden_dim)
        
        # Time-based routing
        self.time_router = TimeRouting(time_steps, hidden_dim)
        
        # Combined processing
        self.signal_encoder = nn.Linear(feature_dim, hidden_dim)
        
        # U-Net style for multiple scale processing
        self.down1 = nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1)
        self.down2 = nn.Conv1d(hidden_dim, hidden_dim, 3, stride=2, padding=1)
        self.down3 = nn.Conv1d(hidden_dim, hidden_dim, 3, stride=2, padding=1)
        
        self.mid_block = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim*2, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim*2, hidden_dim, 3, padding=1)
        )
        
        self.up3 = nn.ConvTranspose1d(hidden_dim*2, hidden_dim, 4, stride=2, padding=1)
        self.up2 = nn.ConvTranspose1d(hidden_dim*2, hidden_dim, 4, stride=2, padding=1)
        self.up1 = nn.Conv1d(hidden_dim*2, hidden_dim, 3, padding=1)
        
        # Final projection
        self.final = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, feature_dim, 1)
        )
        
        # Normalization layers
        self.norm1 = nn.GroupNorm(8, hidden_dim)
        self.norm2 = nn.GroupNorm(8, hidden_dim)
        self.norm3 = nn.GroupNorm(8, hidden_dim)
        self.norm_mid = nn.GroupNorm(8, hidden_dim)
        self.norm_up3 = nn.GroupNorm(8, hidden_dim)
        self.norm_up2 = nn.GroupNorm(8, hidden_dim)
        self.norm_up1 = nn.GroupNorm(8, hidden_dim)
    
    def forward(self, x, timesteps, high_freq, low_freq):
        """
        Denoise signal conditioned on time and decomposition components
        
        Args:
            x: Noisy input signal (batch_size, window_size, feature_dim)
            timesteps: Diffusion timesteps (batch_size,)
            high_freq: High frequency component (batch_size, window_size, feature_dim)
            low_freq: Low frequency component (batch_size, window_size, feature_dim)
            
        Returns:
            Denoised signal prediction
        """
        batch_size = x.shape[0]
        
        # Time embeddings
        t_emb = self.time_embedder(timesteps)
        t_emb = self.time_mlp(t_emb)  # (batch_size, hidden_dim)
        
        # Get routing weights
        routing = self.time_router(timesteps)
        high_weight = routing['high_freq_weight'].view(batch_size, 1, 1)
        low_weight = routing['low_freq_weight'].view(batch_size, 1, 1)
        
        # Process components with routing weights
        high_emb = self.high_freq_encoder(high_freq) * high_weight
        low_emb = self.low_freq_encoder(low_freq) * low_weight
        
        # Process input signal
        x_emb = self.signal_encoder(x)
        
        # Add time embeddings and conditions
        t_emb = t_emb.unsqueeze(1).repeat(1, self.window_size, 1)
        h = x_emb + t_emb + high_emb + low_emb
        
        # U-Net forward path
        h = h.permute(0, 2, 1)  # (batch, channels, seq_len)
        
        h1 = F.gelu(self.norm1(self.down1(h)))
        h2 = F.gelu(self.norm2(self.down2(h1)))
        h3 = F.gelu(self.norm3(self.down3(h2)))
        
        # Middle
        h_mid = F.gelu(self.norm_mid(self.mid_block(h3) + h3))
        
        # U-Net backward path with skip connections
        h_up3 = torch.cat([h_mid, h3], dim=1)
        h_up3 = F.gelu(self.norm_up3(self.up3(h_up3)))
        
        h_up2 = torch.cat([h_up3, h2], dim=1)
        h_up2 = F.gelu(self.norm_up2(self.up2(h_up2)))
        
        h_up1 = torch.cat([h_up2, h1], dim=1)
        h_up1 = F.gelu(self.norm_up1(self.up1(h_up1)))
        
        # Final output
        output = self.final(h_up1)
        output = output.permute(0, 2, 1)  # (batch, seq_len, channels)
        
        return output