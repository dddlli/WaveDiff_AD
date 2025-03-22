import torch
import torch.nn as nn
import numpy as np
import pywt
from pytorch_wavelets import DWT1DForward, DWT1DInverse

class WaveletTransform(nn.Module):
    """
    Time series wavelet transform using PyTorch with enhanced subband processing
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
        
        # Create subband signal representations for each level
        subbands = []
        # Lowest frequency subband is the approximation
        subbands.append(approx)
        
        # Add detail subbands
        for detail in details:
            subbands.append(detail)
        
        # Create high frequency component (sum of all details)
        high_freq = torch.zeros_like(signal)
        for detail in details:
            high_freq = high_freq + detail
        
        # Low frequency is the approximation
        low_freq = approx
        
        return {
            'approx': approx,
            'details': details,
            'subbands': subbands,
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


class AdaptiveWaveletTransform(nn.Module):
    """
    Wavelet transform with dynamic level selection based on signal complexity
    """
    def __init__(self, feature_dim, max_level=5, min_level=2, wavelet_init='db4', mode='zero'):
        super(AdaptiveWaveletTransform, self).__init__()
        self.feature_dim = feature_dim
        self.max_level = max_level
        self.min_level = min_level
        self.wavelet = wavelet_init  # Added for cache compatibility
        self.level = max_level  # Added for cache compatibility
        self.mode = mode
        
        # Initialize array of wavelet transforms for different levels
        self.wavelet_transforms = nn.ModuleList([
            WaveletTransform(wavelet=wavelet_init, level=l, mode=mode) 
            for l in range(min_level, max_level+1)
        ])
        
        # Parameters for adaptive level selection
        self.complexity_threshold = nn.Parameter(torch.tensor(0.7))
        
    def forward(self, x, complexity_scores=None):
        """
        Apply wavelet transform with adaptive level selection
        
        Args:
            x: Input signal
            complexity_scores: Pre-computed complexity scores (AAPE) for each feature
            
        Returns:
            Dictionary with decomposition results
        """
        # Move to same device as input
        device = x.device
        self.to(device)
        
        batch_size, seq_len, feature_dim = x.shape
        
        # If complexity scores not provided, use default level
        if complexity_scores is None:
            default_level_idx = min(2, len(self.wavelet_transforms)-1)  # Use level 3 if available
            return self.wavelet_transforms[default_level_idx].decompose(x)
        
        # Initialize containers for combined results
        combined_approx = torch.zeros_like(x)
        combined_details = []
        for _ in range(self.max_level):
            combined_details.append(torch.zeros_like(x))
        
        combined_high_freq = torch.zeros_like(x)
        combined_low_freq = torch.zeros_like(x)
        
        # Process each feature with appropriate level based on complexity
        for feat_idx in range(feature_dim):
            # Get complexity score for this feature
            feat_complexity = complexity_scores[feat_idx].item()
            
            # Scale complexity to level
            level_scaled = self.min_level + round(feat_complexity * (self.max_level - self.min_level))
            level_idx = min(level_scaled - self.min_level, len(self.wavelet_transforms) - 1)
            level_idx = max(0, level_idx)  # Ensure valid index
            
            # Extract this feature
            feat_data = x[:, :, feat_idx:feat_idx+1]
            
            # Apply appropriate wavelet transform
            feat_decomp = self.wavelet_transforms[level_idx].decompose(feat_data)
            
            # Add to combined results
            combined_approx[:, :, feat_idx:feat_idx+1] = feat_decomp['approx']
            for i, detail in enumerate(feat_decomp['details']):
                if i < self.max_level:
                    combined_details[i][:, :, feat_idx:feat_idx+1] = detail
            
            combined_high_freq[:, :, feat_idx:feat_idx+1] = feat_decomp['high_freq']
            combined_low_freq[:, :, feat_idx:feat_idx+1] = feat_decomp['low_freq']
        
        # Trim details list to actual used levels
        actual_details = combined_details[:self.max_level]
        
        return {
            'approx': combined_approx,
            'details': actual_details,
            'high_freq': combined_high_freq,
            'low_freq': combined_low_freq
        }


class LearnableWaveletTransform(nn.Module):
    """
    Wavelet transform with learnable filters and complexity-aware adaptation
    """
    def __init__(self, feature_dim, level=3, wavelet_init='db4', mode='zero'):
        super(LearnableWaveletTransform, self).__init__()
        self.feature_dim = feature_dim
        # Required attributes for cache compatibility
        self.level = level
        self.wavelet = wavelet_init  # Changed from wavelet_name to wavelet for cache compatibility
        self.mode = mode
        
        # Initialize standard wavelet transform
        self.standard_wavelet = WaveletTransform(wavelet=wavelet_init, level=level, mode=mode)
        
        # Learnable filter parameters
        self.filter_params = nn.Parameter(torch.ones(level + 1))
        
        # Learnable scale factors for different frequency bands
        self.high_freq_scale = nn.Parameter(torch.ones(1))
        self.low_freq_scale = nn.Parameter(torch.ones(1))
        
        # Importance weights for different frequency bands - make sure it has level+1 elements
        self.importance = nn.Parameter(torch.ones(level + 1))
        
        # Adaptive complexity analyzer
        self.adaptive_transform = AdaptiveWaveletTransform(
            feature_dim, max_level=level+2, min_level=level-1, 
            wavelet_init=wavelet_init, mode=mode
        )

        # Energy conservation factor for IMF integration
        self.energy_conservation = nn.Parameter(torch.ones(1))
    
    def integrate_imf(self, wavelet_coeffs, imfs, complexity_scores):
        """
        Integrate IMFs into wavelet coefficients for composite wavelet basis
        
        Args:
            wavelet_coeffs: Dictionary with wavelet coefficients
            imfs: IMFs from HHT [batch, imfs, seq_len, features]
            complexity_scores: Feature complexity scores [features]
            
        Returns:
            Modified wavelet coefficients with IMF integration
        """
        # Check if imfs is None or invalid
        if imfs is None:
            return wavelet_coeffs
            
        # Check if imfs has the right shape (should be 4D tensor)
        if not isinstance(imfs, torch.Tensor) or len(imfs.shape) != 4:
            print(f"Warning: IMFs have invalid shape: {imfs if isinstance(imfs, int) else type(imfs)}")
            return wavelet_coeffs
            
        # Extract components
        details = wavelet_coeffs['details']
        
        batch_size, num_imfs, seq_len, features = imfs.shape
        device = imfs.device
        
        # Normalize complexity scores if provided
        if complexity_scores is not None:
            norm_complexity = torch.sigmoid(complexity_scores)
        else:
            # Default to 0.5 if no complexity scores
            norm_complexity = torch.ones(features, device=device) * 0.5
        
        # Get weights from importance parameter
        # Make sure we don't exceed the importance parameter size
        actual_imfs = min(num_imfs, len(self.importance))
        weights = torch.sigmoid(self.importance[:actual_imfs])
        weights = weights.view(1, -1, 1, 1)
        
        # Apply weights to IMFs
        weighted_imfs = imfs[:, :actual_imfs] * weights
        
        # Integrate IMFs into detail coefficients
        for i, detail in enumerate(details):
            if i < actual_imfs:
                # Calculate energy before modification
                detail_energy = torch.sum(detail**2)
                
                # Resize IMF to match detail size if needed
                if weighted_imfs.shape[2] != detail.shape[1]:
                    imf_resized = torch.nn.functional.interpolate(
                        weighted_imfs[:, i].permute(0, 2, 1), 
                        size=detail.shape[1]
                    ).permute(0, 2, 1)
                else:
                    imf_resized = weighted_imfs[:, i]
                
                # Integrate based on complexity
                complexity_factor = norm_complexity.view(1, 1, -1)
                detail_mod = detail * (1 - complexity_factor) + imf_resized * complexity_factor
                
                # Energy conservation
                mod_energy = torch.sum(detail_mod**2)
                if mod_energy > 0:
                    energy_ratio = torch.sqrt(detail_energy / (mod_energy + 1e-8))
                    detail_mod = detail_mod * energy_ratio * torch.sigmoid(self.energy_conservation)
                
                # Update detail
                details[i] = detail_mod
        
        # Update wavelet coefficients
        wavelet_coeffs['details'] = details
        return wavelet_coeffs

    # Add a custom method for cache system compatibility
    def decompose(self, signal, complexity_scores=None):
        """
        Wrapper method for forward to maintain compatibility with cache system
        """
        return self.forward(signal, None, complexity_scores)
    
    def forward(self, x, imfs=None, complexity_scores=None):
        """
        Apply wavelet transform with dimension consistency and complexity adaptation
        """
        # Move to same device as input
        device = x.device
        self.to(device)
        
        # Get original dimensions
        batch_size, seq_len, feature_dim = x.shape
        
        # Use adaptive wavelet transform if complexity scores provided
        if complexity_scores is not None:
            decomp = self.adaptive_transform(x, complexity_scores)
        else:
            # Use standard wavelet transform as base
            decomp = self.standard_wavelet.decompose(x)
        
        # Integrate IMFs if provided - with safe handling
        if imfs is not None:
            try:
                decomp = self.integrate_imf(decomp, imfs, complexity_scores)
            except Exception as e:
                print(f"Warning: Error integrating IMFs: {e}")
                # Continue without integration if there's an error
        
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
        
        # Apply importance weights
        approx = approx * (self.importance[0] * self.low_freq_scale)
        
        # Make sure we don't exceed the importance parameter size when weighting details
        for i, detail in enumerate(details):
            # Safe indexing - don't exceed importance size
            if i + 1 < len(self.importance):
                details[i] = detail * (self.importance[i+1] * self.high_freq_scale)
            else:
                # Use the last element if we go beyond bounds
                details[i] = detail * (self.importance[-1] * self.high_freq_scale)
        
        # Adaptive frequency partitioning based on complexity
        if complexity_scores is not None:
            avg_complexity = torch.mean(complexity_scores).item()
            partition_idx = max(1, min(self.level - 1, int(self.level * avg_complexity)))
        else:
            partition_idx = self.level // 2
        
        # Create high and low frequency components
        high_freq = torch.zeros((batch_size, seq_len, feature_dim), device=device)
        for i in range(partition_idx):
            if i < len(details):
                if details[i].shape[1] < seq_len:
                    pad = torch.zeros((batch_size, seq_len - details[i].shape[1], feature_dim), device=device)
                    detail_padded = torch.cat([details[i], pad], dim=1)
                    high_freq = high_freq + detail_padded
                elif details[i].shape[1] > seq_len:
                    high_freq = high_freq + details[i][:, :seq_len, :]
                else:
                    high_freq = high_freq + details[i]
        
        # Low frequency component
        low_freq = approx.clone()
        for i in range(partition_idx, len(details)):
            if i < len(details):
                if details[i].shape[1] < seq_len:
                    pad = torch.zeros((batch_size, seq_len - details[i].shape[1], feature_dim), device=device)
                    detail_padded = torch.cat([details[i], pad], dim=1)
                    low_freq = low_freq + detail_padded
                elif details[i].shape[1] > seq_len:
                    low_freq = low_freq + details[i][:, :seq_len, :]
                else:
                    low_freq = low_freq + details[i]
        
        # Ensure proper padding for low_freq
        if low_freq.shape[1] < seq_len:
            pad = torch.zeros((batch_size, seq_len - low_freq.shape[1], feature_dim), device=device)
            low_freq = torch.cat([low_freq, pad], dim=1)
        elif low_freq.shape[1] > seq_len:
            low_freq = low_freq[:, :seq_len, :]
        
        # Energy conservation
        original_energy = torch.sum(x**2, dim=1, keepdim=True)
        component_energy = torch.sum(high_freq**2, dim=1, keepdim=True) + torch.sum(low_freq**2, dim=1, keepdim=True)
        energy_ratio = torch.sqrt(original_energy / (component_energy + 1e-8))
        high_freq = high_freq * energy_ratio
        low_freq = low_freq * energy_ratio
        
        return {
            'approx': approx,
            'details': details,
            'high_freq': high_freq,
            'low_freq': low_freq,
            'partition_idx': partition_idx
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