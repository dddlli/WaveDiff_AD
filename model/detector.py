import torch
from torch import nn
import torch.nn.functional as F

from model.decomposition.wavelet import LearnableWaveletTransform
from model.decomposition.hht import HilbertHuangTransform
from model.decomposition.signal_decomp import AdaptiveFusion
from model.diffusion.diffusion import Diffusion 
from model.diffusion.denoise import ConditionalDenoisingNetwork
from model.reconstruction import Reconstruction

class AnomalyDetection(nn.Module):
    def __init__(self, 
                 time_steps=1000, 
                 beta_start=0.0001, 
                 beta_end=0.02, 
                 window_size=64, 
                 model_dim=512, 
                 ff_dim=2048, 
                 atten_dim=64, 
                 feature_num=51, 
                 time_num=5, 
                 block_num=2, 
                 head_num=8, 
                 dropout=0.6, 
                 device='cpu', 
                 wavelet_level=3, 
                 hht_imfs=5, 
                 fusion_type='weighted', 
                 t=500):
        super(AnomalyDetection, self).__init__()
        
        self.device = device
        self.window_size = window_size
        self.feature_num = feature_num
        self.t = t
        self.time_steps = time_steps
        
        # Decomposition modules
        self.wavelet = LearnableWaveletTransform(
            feature_dim=feature_num,
            level=wavelet_level, 
            wavelet_init='db4', 
            mode='zero'
        )
        
        self.hht = HilbertHuangTransform(
            max_imfs=hht_imfs,
            sift_threshold=0.05
        )
        
        self.fusion = AdaptiveFusion(
            feature_dim=feature_num,
            fusion_type=fusion_type,
            window_size=window_size
        )
        
        # Diffusion process
        self.diffusion = Diffusion(
            time_steps=time_steps,
            beta_start=beta_start,
            beta_end=beta_end,
            device=device
        )
        
        # Denoising network
        self.denoiser = ConditionalDenoisingNetwork(
            feature_dim=feature_num,
            window_size=window_size,
            time_steps=time_steps,
            hidden_dim=model_dim
        )
        
        # Reconstruction module
        self.reconstruction = Reconstruction(
            window_size=window_size,
            model_dim=model_dim,
            ff_dim=ff_dim,
            atten_dim=atten_dim,
            feature_num=feature_num,
            time_num=time_num,
            block_num=block_num,
            head_num=head_num,
            dropout=dropout
        )
        
        # Loss functions
        self.mse_loss = nn.MSELoss(reduction='mean')
        
    def forward(self, data, time, p=0):
        """
        Forward pass of the WHDiff model with dimension handling
        """
        batch_size = data.shape[0]
        
        # Disturbance for robustness
        disturb = torch.rand(batch_size, self.feature_num) * p
        disturb = disturb.unsqueeze(1).repeat(1, self.window_size, 1).float().to(self.device)
        data_disturbed = data + disturb
        
        # Decompose signal using wavelet transform
        wavelet_out = self.wavelet(data_disturbed)
        
        # Decompose signal using HHT (especially for non-stationary trends)
        # 这里使用浅拷贝来避免修改原始wavelet_out
        hht_input = {'low_freq': wavelet_out['low_freq'].clone()}
        hht_out = self.hht(hht_input['low_freq'])
        
        # 获取各组件的维度信息
        wav_high_shape = wavelet_out['high_freq'].shape
        hht_high_shape = hht_out['high_freq'].shape
        wav_low_shape = wavelet_out['low_freq'].shape
        hht_low_shape = hht_out['low_freq'].shape
        
        # Fuse decomposition results with energy consistency
        fusion_out = self.fusion(wavelet_out, hht_out, original_signal=data_disturbed)
        high_freq = fusion_out['high_freq']
        low_freq = fusion_out['low_freq']
        energy_loss = fusion_out['energy_loss']
        
        # 确保high_freq和low_freq的长度一致
        min_len = min(high_freq.shape[1], low_freq.shape[1])
        high_freq = high_freq[:, :min_len, :]
        low_freq = low_freq[:, :min_len, :]
        
        # 调整data_disturbed的长度以匹配组件
        data_disturbed_adj = data_disturbed[:, :min_len, :]
        
        # Apply diffusion process
        timesteps = torch.full((batch_size,), self.t, device=self.device)
        noise = torch.randn_like(data_disturbed_adj).to(self.device)
        noisy_data = self.diffusion.q_sample(data_disturbed_adj, low_freq, timesteps, noise)
        
        # Apply denoising network
        denoised = self.denoiser(noisy_data, timesteps, high_freq, low_freq)
        
        # 确保time的长度与noisy_data匹配
        time_adj = time[:, :min_len, :] if time.shape[1] > min_len else time
        
        # Final reconstruction
        recon = self.reconstruction(noisy_data, high_freq, low_freq, time_adj)
        
        # Calculate losses
        recon_loss = self.mse_loss(recon, data_disturbed_adj)
        denoise_loss = self.mse_loss(denoised, data_disturbed_adj)
        decomp_loss = self.mse_loss(high_freq + low_freq, data_disturbed_adj)
        
        # Combined losses
        total_loss = 0.3 * recon_loss + 0.3 * denoise_loss + 0.2 * decomp_loss
        if energy_loss is not None:
            total_loss = total_loss + 0.2 * energy_loss
        
        losses = {
            'total': total_loss,
            'recon': recon_loss,
            'denoise': denoise_loss,
            'decomp': decomp_loss,
            'energy': energy_loss
        }
        
        # Remove disturbance from outputs
        if p > 0:
            # 避免除零，确保分母不为零
            component_sum = high_freq + low_freq
            component_sum_safe = component_sum + 1e-10
            high_freq = high_freq - disturb[:, :min_len, :] * (high_freq / component_sum_safe)
            low_freq = low_freq - disturb[:, :min_len, :] * (low_freq / component_sum_safe)
            recon = recon - disturb[:, :min_len, :]
        
        return high_freq, low_freq, recon, losses
    
    def compute_anomaly_score(self, data, time):
        """
        Compute anomaly score based on reconstruction error and component analysis
        
        Args:
            data: Input time series data
            time: Time features
            
        Returns:
            Anomaly scores and components
        """
        with torch.no_grad():
            # Get decomposition and reconstruction
            high_freq, low_freq, recon, _ = self.forward(data, time, p=0)
            
            # Reconstruction error (MSE)
            recon_error = torch.mean((data - recon)**2, dim=-1)
            
            # High-frequency component error (sensitive to local anomalies)
            high_freq_error = torch.mean((data - (low_freq + recon))**2, dim=-1)
            
            # Low-frequency component error (sensitive to trend anomalies)
            low_freq_error = torch.mean((data - (high_freq + recon))**2, dim=-1)
            
            # Component weights for final score
            alpha = 0.6  # Weight for reconstruction error
            beta = 0.25  # Weight for high frequency error
            gamma = 0.15  # Weight for low frequency error
            
            # Combined anomaly score
            anomaly_score = alpha * recon_error + beta * high_freq_error + gamma * low_freq_error
            
            # Return all components for detailed analysis
            return {
                'score': anomaly_score,
                'recon_error': recon_error,
                'high_freq_error': high_freq_error,
                'low_freq_error': low_freq_error,
                'high_freq': high_freq,
                'low_freq': low_freq,
                'recon': recon
            }