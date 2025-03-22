import torch
import torch.nn as nn
import numpy as np
from PyEMD import EMD
import scipy.signal as signal

class HilbertHuangTransform(nn.Module):
    """
    Hilbert-Huang Transform implementation for non-stationary and non-linear time series analysis
    with enhanced preprocessing and adaptive IMF selection based on signal complexity
    """
    def __init__(self, max_imfs=5, sift_threshold=0.05, adaptive_mode=True):
        super(HilbertHuangTransform, self).__init__()
        self.max_imfs = max_imfs
        self.sift_threshold = sift_threshold
        self.adaptive_mode = adaptive_mode
        
        # Initialize EMD processor
        self.emd = EMD()
        # EMD parameters
        self.emd.FIXE = 0  # No fixed iterations for sifting
        self.emd.FIXE_H = 0  # No fixed iterations for sifting with threshold
        self.emd.MAX_ITERATION = 1000  # Max iterations overall
        self.emd.EPSILON = self.sift_threshold  # Threshold for stopping
        
        # Learnable parameters for IMF importance
        self.imf_importance = nn.Parameter(torch.ones(max_imfs))
        
        # Energy preservation parameters
        self.energy_conservation = nn.Parameter(torch.ones(1))
        
        # Frequency band boundaries for adaptive partitioning
        self.freq_boundaries = nn.Parameter(torch.linspace(0.2, 0.8, max_imfs-1))
        
    def empirical_mode_decomposition(self, signal_data, complexity=None):
        """
        Perform EMD on a single time series signal with adaptive IMF selection
        and energy preservation
        
        Args:
            signal_data: Single time series as numpy array
            complexity: Signal complexity score for adaptive processing
            
        Returns:
            Tuple of IMFs, residue, instantaneous frequency, amplitude, and phase
        """
        if len(signal_data) < 3:
            # Not enough data for EMD
            return np.zeros((self.max_imfs, len(signal_data))), signal_data, None, None, None
            
        try:
            # Calculate signal energy before decomposition
            original_energy = np.sum(signal_data**2)
            
            # Determine number of IMFs based on complexity if adaptive mode is on
            target_imfs = self.max_imfs
            if self.adaptive_mode and complexity is not None:
                # Scale complexity to number of IMFs (more complex = more IMFs)
                # Using a sigmoid function for smoother scaling
                scaling = 1 / (1 + np.exp(-10 * (complexity - 0.5)))
                target_imfs = 2 + int(scaling * (self.max_imfs - 2))
                target_imfs = min(max(2, target_imfs), self.max_imfs)
            
            # Extract IMFs
            imfs = self.emd.emd(signal_data, max_imf=target_imfs)
            
            if imfs.shape[0] == 0:
                # No IMFs could be extracted
                return np.zeros((self.max_imfs, len(signal_data))), signal_data, None, None, None
            
            # Calculate instantaneous attributes
            inst_freq = np.zeros((imfs.shape[0], imfs.shape[1]))
            inst_amp = np.zeros((imfs.shape[0], imfs.shape[1]))
            inst_phase = np.zeros((imfs.shape[0], imfs.shape[1]))
            
            for i in range(imfs.shape[0]):
                if np.allclose(imfs[i], 0):
                    continue
                    
                # Apply Hilbert transform
                analytic_signal = signal.hilbert(imfs[i])
                inst_amp[i] = np.abs(analytic_signal)
                
                # Calculate instantaneous phase
                phase = np.unwrap(np.angle(analytic_signal))
                inst_phase[i] = phase
                
                # Calculate instantaneous frequency (derivative of phase)
                inst_freq[i, 1:] = np.diff(phase) / (2.0 * np.pi)
                
            # Pad or truncate to required number of IMFs
            if imfs.shape[0] < self.max_imfs:
                # Pad with zeros
                padded_imfs = np.zeros((self.max_imfs, imfs.shape[1]))
                padded_imfs[:imfs.shape[0], :] = imfs
                
                # Also pad instantaneous attributes
                padded_freq = np.zeros((self.max_imfs, imfs.shape[1]))
                padded_freq[:imfs.shape[0], :] = inst_freq
                
                padded_amp = np.zeros((self.max_imfs, imfs.shape[1]))
                padded_amp[:imfs.shape[0], :] = inst_amp
                
                padded_phase = np.zeros((self.max_imfs, imfs.shape[1]))
                padded_phase[:imfs.shape[0], :] = inst_phase
                
                imfs = padded_imfs
                inst_freq = padded_freq
                inst_amp = padded_amp
                inst_phase = padded_phase
            elif imfs.shape[0] > self.max_imfs:
                # Truncate
                imfs = imfs[:self.max_imfs]
                inst_freq = inst_freq[:self.max_imfs]
                inst_amp = inst_amp[:self.max_imfs]
                inst_phase = inst_phase[:self.max_imfs]
                
            # Calculate residue (original signal - sum of IMFs)
            residue = signal_data - np.sum(imfs, axis=0)
            
            # Apply energy conservation
            decomposed_energy = np.sum(np.sum(imfs**2, axis=1)) + np.sum(residue**2)
            if decomposed_energy > 0:
                energy_ratio = np.sqrt(original_energy / decomposed_energy)
                imfs = imfs * energy_ratio
                residue = residue * energy_ratio
                inst_amp = inst_amp * energy_ratio
            
            return imfs, residue, inst_freq, inst_amp, inst_phase
            
        except Exception as e:
            print(f"EMD failed: {e}")
            return np.zeros((self.max_imfs, len(signal_data))), signal_data, None, None, None
    
    def hilbert_transform(self, imfs):
        """
        Apply Hilbert transform to IMFs to get instantaneous frequency and amplitude
        
        Args:
            imfs: Array of IMFs
            
        Returns:
            Instantaneous frequency and amplitude
        """
        num_imfs, signal_length = imfs.shape
        inst_freq = np.zeros_like(imfs)
        inst_amp = np.zeros_like(imfs)
        
        for i in range(num_imfs):
            # Skip if IMF is all zeros
            if np.allclose(imfs[i], 0):
                continue
                
            # Apply Hilbert transform
            analytic_signal = signal.hilbert(imfs[i])
            inst_amp[i] = np.abs(analytic_signal)
            
            # Calculate instantaneous phase
            inst_phase = np.unwrap(np.angle(analytic_signal))
            
            # Calculate instantaneous frequency (derivative of phase)
            inst_freq[i, 1:] = np.diff(inst_phase) / (2.0 * np.pi)
            
        return inst_freq, inst_amp
    
    def adaptive_frequency_partitioning(self, imfs, inst_freq, inst_amp, complexity=None):
        """
        Partition IMFs into high and low frequency components based on 
        instantaneous frequency characteristics and complexity
        
        Args:
            imfs: Array of IMFs
            inst_freq: Instantaneous frequency
            inst_amp: Instantaneous amplitude
            complexity: Signal complexity score
            
        Returns:
            High and low frequency components and boundary index
        """
        if inst_freq is None or inst_amp is None:
            # If attributes not available, use simple partitioning
            if complexity is not None:
                boundary = max(1, min(self.max_imfs - 1, int(self.max_imfs * (1 - complexity))))
            else:
                boundary = self.max_imfs // 2
                
            high_freq = np.sum(imfs[:boundary], axis=0)
            low_freq = np.sum(imfs[boundary:], axis=0)
            return high_freq, low_freq, boundary
        
        # Calculate dominant frequency for each IMF
        dominant_freq = np.zeros(imfs.shape[0])
        for i in range(imfs.shape[0]):
            if np.sum(inst_amp[i]) > 0:
                # Weighted average of instantaneous frequency by amplitude
                dominant_freq[i] = np.sum(inst_freq[i] * inst_amp[i]) / np.sum(inst_amp[i])
        
        # Sort IMFs by dominant frequency
        sorted_indices = np.argsort(dominant_freq)[::-1]  # Descending order
        
        # Determine boundary based on complexity and frequency boundaries
        if complexity is not None:
            # Convert learnable boundary parameters to numpy
            freq_boundaries_np = self.freq_boundaries.detach().cpu().numpy()
            
            # Find appropriate boundary based on complexity
            boundary_idx = np.searchsorted(freq_boundaries_np, complexity)
            boundary = boundary_idx + 1  # +1 because boundaries are between IMFs
            boundary = max(1, min(boundary, self.max_imfs - 1))
        else:
            boundary = self.max_imfs // 2
        
        # Apply importance weights (convert to numpy)
        importance_np = self.imf_importance.detach().cpu().numpy()
        weighted_imfs = imfs * importance_np.reshape(-1, 1)
        
        # Create high and low frequency components
        high_freq_imfs = np.zeros_like(imfs)
        low_freq_imfs = np.zeros_like(imfs)
        
        for i, idx in enumerate(sorted_indices):
            if i < boundary:
                high_freq_imfs[idx] = weighted_imfs[idx]
            else:
                low_freq_imfs[idx] = weighted_imfs[idx]
        
        high_freq = np.sum(high_freq_imfs, axis=0)
        low_freq = np.sum(low_freq_imfs, axis=0)
        
        return high_freq, low_freq, boundary
    
    # Add decompose method for cache system compatibility
    def decompose(self, signal, complexity_scores=None):
        """Wrapper method for forward function to maintain compatibility with cache system"""
        return self.forward(signal, complexity_scores)
    
    def forward(self, x, complexity_scores=None):
        """
        Apply HHT to decompose signal with adaptive processing
        
        Args:
            x: Input signal [batch_size, seq_len, features]
            complexity_scores: Signal complexity scores for adaptive processing [features]
            
        Returns:
            Dictionary with decomposition results
        """
        batch_size, seq_len, features = x.shape
        device = x.device
        
        # 确保输出长度与输入一致
        target_len = seq_len
        
        # Initialize tensors for results
        imfs = torch.zeros((batch_size, self.max_imfs, target_len, features), device=device)
        residue = torch.zeros((batch_size, target_len, features), device=device)
        high_freq = torch.zeros((batch_size, target_len, features), device=device)
        low_freq = torch.zeros((batch_size, target_len, features), device=device)
        
        # Process each batch and feature
        for b in range(batch_size):
            for f in range(features):
                # Extract signal and move to CPU for numpy processing
                signal_data = x[b, :, f].detach().cpu().numpy()
                
                # Get complexity for this feature if available
                feat_complexity = None
                if complexity_scores is not None and f < len(complexity_scores):
                    feat_complexity = complexity_scores[f].item()
                
                # Apply enhanced EMD with instantaneous attributes
                signal_imfs, signal_residue, inst_freq, inst_amp, inst_phase = self.empirical_mode_decomposition(
                    signal_data, complexity=feat_complexity
                )
                
                # Ensure length consistency
                imf_len = signal_imfs.shape[1] if signal_imfs.shape[0] > 0 else 0
                if imf_len > 0:
                    if imf_len > target_len:
                        # Truncate
                        signal_imfs = signal_imfs[:, :target_len]
                        signal_residue = signal_residue[:target_len] if len(signal_residue) > target_len else signal_residue
                        if inst_freq is not None:
                            inst_freq = inst_freq[:, :target_len]
                            inst_amp = inst_amp[:, :target_len]
                            inst_phase = inst_phase[:, :target_len]
                    elif imf_len < target_len:
                        # Pad
                        pad_imfs = np.zeros((signal_imfs.shape[0], target_len))
                        pad_imfs[:, :imf_len] = signal_imfs
                        signal_imfs = pad_imfs
                        
                        pad_residue = np.zeros(target_len)
                        if len(signal_residue) > 0:
                            pad_residue[:len(signal_residue)] = signal_residue
                        signal_residue = pad_residue
                        
                        if inst_freq is not None:
                            pad_freq = np.zeros((inst_freq.shape[0], target_len))
                            pad_freq[:, :imf_len] = inst_freq
                            inst_freq = pad_freq
                            
                            pad_amp = np.zeros((inst_amp.shape[0], target_len))
                            pad_amp[:, :imf_len] = inst_amp
                            inst_amp = pad_amp
                            
                            pad_phase = np.zeros((inst_phase.shape[0], target_len))
                            pad_phase[:, :imf_len] = inst_phase
                            inst_phase = pad_phase
                    
                    # Adaptive frequency partitioning using instantaneous attributes
                    feature_high, feature_low, boundary = self.adaptive_frequency_partitioning(
                        signal_imfs, inst_freq, inst_amp, feat_complexity
                    )
                    
                    # Store results
                    imfs[b, :, :, f] = torch.tensor(signal_imfs, device=device)
                    residue[b, :, f] = torch.tensor(signal_residue, device=device)
                    high_freq[b, :, f] = torch.tensor(feature_high, device=device)
                    low_freq[b, :, f] = torch.tensor(feature_low, device=device)
        
        # Apply energy conservation scaling
        energy_scaling = torch.sigmoid(self.energy_conservation)
        high_freq = high_freq * energy_scaling
        low_freq = low_freq * energy_scaling
        
        # Check for energy conservation
        original_energy = torch.sum(x**2, dim=1, keepdim=True)
        component_energy = torch.sum(high_freq**2, dim=1, keepdim=True) + torch.sum(low_freq**2, dim=1, keepdim=True)
        
        # Adjust for energy conservation
        energy_ratio = torch.sqrt(original_energy / (component_energy + 1e-8))
        high_freq = high_freq * energy_ratio
        low_freq = low_freq * energy_ratio
        
        # Apply learned importance weights to IMFs
        imf_weights = torch.sigmoid(self.imf_importance).view(1, -1, 1, 1)
        weighted_imfs = imfs * imf_weights
        
        return {
            'imfs': imfs,
            'weighted_imfs': weighted_imfs,
            'residue': residue,
            'high_freq': high_freq,
            'low_freq': low_freq,
        }