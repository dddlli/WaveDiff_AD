import torch
import torch.nn as nn
import numpy as np
import pywt
from pytorch_wavelets import DWT1DForward, DWT1DInverse

class WaveletTransform(nn.Module):
    """
    Time series wavelet transform using PyTorch
    """
    def __init__(self, wavelet='db4', level=3, mode='zero'):
        """
        Args:
            wavelet: Wavelet type (default: 'db4')
            level: Decomposition level (default: 3)
            mode: Padding mode (default: 'zero')
        """
        super(WaveletTransform, self).__init__()
        self.wavelet = wavelet
        self.level = level
        self.mode = mode
        self.dwt1d = DWT1DForward(wave=wavelet, J=level, mode=mode)
        self.idwt1d = DWT1DInverse(wave=wavelet, mode=mode)
    
    def decompose(self, signal):
        """
        Decompose signal using wavelet transform
        
        Args:
            signal: Input signal (batch_size, sequence_length, features)
        
        Returns:
            Dictionary with approximation and detail coefficients
        """
        # Get device
        device = signal.device
        batch_size, seq_len, features = signal.shape
        
        # Ensure wavelet modules are on the correct device
        self.to(device)
        
        # Transpose to format expected by pytorch_wavelets (batch_size, features, sequence_length)
        signal_transpose = signal.permute(0, 2, 1)
        
        # Initialize storage for coefficients in original format
        approx = torch.zeros((batch_size, seq_len, features), device=device)
        details = [torch.zeros((batch_size, seq_len, features), device=device) for _ in range(self.level)]
        
        # Apply wavelet decomposition to entire batch
        coeffs_ll, coeffs_details = self.dwt1d(signal_transpose)
        
        # Convert approximation coefficients back to original format
        approx_len = coeffs_ll.shape[2]
        approx[:, :approx_len, :] = coeffs_ll.permute(0, 2, 1)
        
        # Convert detail coefficients back to original format
        for i, detail_coeff in enumerate(coeffs_details):
            detail_len = detail_coeff.shape[2]
            details[i][:, :detail_len, :] = detail_coeff.permute(0, 2, 1)
        
        # Create high frequency component (sum of all details)
        high_freq = torch.zeros_like(signal)
        for detail in details:
            high_freq = high_freq + detail
        
        # Low frequency is the approximation
        low_freq = approx
        
        return {
            'approx': approx,
            'details': details,
            'high_freq': high_freq,
            'low_freq': low_freq
        }
    
    def reconstruct(self, components):
        """
        Reconstruct signal from components
        
        Args:
            components: Dictionary with approximation and detail coefficients
        
        Returns:
            Reconstructed signal
        """
        # Get components
        approx = components['approx']
        details = components.get('details', [])
        
        batch_size, seq_len, features = approx.shape
        device = approx.device
        
        # Ensure wavelet modules are on the correct device
        self.to(device)
        
        # Convert to format expected by pytorch_wavelets (batch_size, features, seq_len)
        approx_transpose = approx.permute(0, 2, 1)
        detail_transposes = [detail.permute(0, 2, 1) for detail in details]
        
        # Apply inverse transform
        reconstructed = self.idwt1d((approx_transpose, detail_transposes))
        
        # Convert back to original format
        result = reconstructed.permute(0, 2, 1)
        
        return result
    
    def forward(self, x):
        """
        Forward pass: decompose and reconstruct
        
        Args:
            x: Input signal
        
        Returns:
            Reconstructed signal
        """
        decomp = self.decompose(x)
        return self.reconstruct(decomp)


class LearnableWaveletTransform(nn.Module):
    """
    Wavelet transform with learnable filters
    """
    def __init__(self, feature_dim, level=3, wavelet_init='db4', mode='zero'):
        super(LearnableWaveletTransform, self).__init__()
        self.feature_dim = feature_dim
        self.level = level
        self.wavelet_name = wavelet_init
        self.mode = mode
        
        # Initialize standard wavelet transform
        self.standard_wavelet = WaveletTransform(wavelet=wavelet_init, level=level, mode=mode)
        
        # Learnable filter parameters
        self.filter_params = nn.Parameter(torch.ones(level + 1))
        
        # Learnable scale factors for different frequency bands
        self.high_freq_scale = nn.Parameter(torch.ones(1))
        self.low_freq_scale = nn.Parameter(torch.ones(1))
        
        # Importance weights for different frequency bands
        self.importance = nn.Parameter(torch.ones(level + 1))
    
    def forward(self, x):
        """
        Apply wavelet transform with dimension consistency
        """
        # Move to same device as input
        device = x.device
        self.to(device)
        
        # Get original dimensions
        batch_size, seq_len, feature_dim = x.shape
        
        # Use standard wavelet transform as base
        decomp = self.standard_wavelet.decompose(x)
        
        # Extract components
        approx = decomp['approx']
        details = decomp['details']
        
        # Ensure all components have the same feature dimension
        if approx.shape[2] != feature_dim:
            approx_proj = nn.Linear(approx.shape[2], feature_dim).to(device)
            approx = approx_proj(approx)
        
        for i in range(len(details)):
            if details[i].shape[2] != feature_dim:
                detail_proj = nn.Linear(details[i].shape[2], feature_dim).to(device)
                details[i] = detail_proj(details[i])
        
        approx = approx * (self.importance[0] * self.low_freq_scale)
        details = [detail * (self.importance[i+1] * self.high_freq_scale) for i, detail in enumerate(details)]
        
        high_freq = torch.zeros((batch_size, seq_len, feature_dim), device=device)
        for detail in details:
            if detail.shape[1] < seq_len:
                pad = torch.zeros((batch_size, seq_len - detail.shape[1], feature_dim), device=device)
                detail_padded = torch.cat([detail, pad], dim=1)
                high_freq = high_freq + detail_padded
            elif detail.shape[1] > seq_len:
                high_freq = high_freq + detail[:, :seq_len, :]
            else:
                high_freq = high_freq + detail
        
        low_freq = approx
        if low_freq.shape[1] < seq_len:
            pad = torch.zeros((batch_size, seq_len - low_freq.shape[1], feature_dim), device=device)
            low_freq = torch.cat([low_freq, pad], dim=1)
        elif low_freq.shape[1] > seq_len:
            low_freq = low_freq[:, :seq_len, :]
        
        return {
            'approx': approx,
            'details': details,
            'high_freq': high_freq,
            'low_freq': low_freq
        }
    
    def get_filter_parameters(self):
        """
        Get current filter parameters
        
        Returns:
            Dictionary with filter parameters
        """
        return {
            'importance': self.importance.detach().cpu().numpy(),
            'high_freq_scale': self.high_freq_scale.item(),
            'low_freq_scale': self.low_freq_scale.item(),
            'filter_params': self.filter_params.detach().cpu().numpy()
        }