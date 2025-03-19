import torch
import torch.nn as nn
import numpy as np
from PyEMD import EMD
import scipy.signal as signal

class HilbertHuangTransform(nn.Module):
    """
    Hilbert-Huang Transform implementation for non-stationary and non-linear time series analysis
    """
    def __init__(self, max_imfs=5, sift_threshold=0.05):
        super(HilbertHuangTransform, self).__init__()
        self.max_imfs = max_imfs
        self.sift_threshold = sift_threshold
        # Initialize EMD processor
        self.emd = EMD()
        # EMD parameters
        self.emd.FIXE = 0  # No fixed iterations for sifting
        self.emd.FIXE_H = 0  # No fixed iterations for sifting with threshold
        self.emd.MAX_ITERATION = 1000  # Max iterations overall
        self.emd.EPSILON = self.sift_threshold  # Threshold for stopping
        
    def empirical_mode_decomposition(self, signal_data):
        """
        Perform EMD on a single time series signal
        
        Args:
            signal_data: Single time series as numpy array
            
        Returns:
            Tuple of IMFs and residue
        """
        if len(signal_data) < 3:
            # Not enough data for EMD
            return np.zeros((self.max_imfs, len(signal_data))), signal_data
            
        try:
            # Extract IMFs
            imfs = self.emd.emd(signal_data, max_imf=self.max_imfs)
            
            if imfs.shape[0] == 0:
                # No IMFs could be extracted
                return np.zeros((self.max_imfs, len(signal_data))), signal_data
                
            # Pad or truncate to required number of IMFs
            if imfs.shape[0] < self.max_imfs:
                # Pad with zeros
                padded_imfs = np.zeros((self.max_imfs, imfs.shape[1]))
                padded_imfs[:imfs.shape[0], :] = imfs
                imfs = padded_imfs
            elif imfs.shape[0] > self.max_imfs:
                # Truncate
                imfs = imfs[:self.max_imfs]
                
            # Calculate residue (original signal - sum of IMFs)
            residue = signal_data - np.sum(imfs, axis=0)
            
            return imfs, residue
            
        except Exception as e:
            # Fallback in case of EMD failure
            print(f"EMD failed: {e}")
            return np.zeros((self.max_imfs, len(signal_data))), signal_data
    
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
    
    def forward(self, x):
        """
        Apply HHT to decompose signal with flexible output length
        """
        batch_size, seq_len, features = x.shape
        device = x.device
        
        # 确保输出长度与输入一致
        target_len = seq_len
        
        # Initialize tensors for results
        imfs = torch.zeros((batch_size, self.max_imfs, target_len, features), device=device)
        residue = torch.zeros((batch_size, target_len, features), device=device)
        
        # Process each batch and feature
        for b in range(batch_size):
            for f in range(features):
                # Extract signal and move to CPU for numpy processing
                signal_data = x[b, :, f].detach().cpu().numpy()
                
                # Apply EMD with explicit length management
                try:
                    signal_imfs, signal_residue = self.empirical_mode_decomposition(signal_data)
                    
                    # Ensure length consistency
                    imf_len = signal_imfs.shape[1] if signal_imfs.shape[0] > 0 else 0
                    if imf_len > 0:
                        if imf_len > target_len:
                            # Truncate
                            signal_imfs = signal_imfs[:, :target_len]
                            signal_residue = signal_residue[:target_len] if len(signal_residue) > target_len else signal_residue
                        elif imf_len < target_len:
                            # Pad
                            pad_imfs = np.zeros((signal_imfs.shape[0], target_len))
                            pad_imfs[:, :imf_len] = signal_imfs
                            signal_imfs = pad_imfs
                            
                            pad_residue = np.zeros(target_len)
                            if len(signal_residue) > 0:
                                pad_residue[:len(signal_residue)] = signal_residue
                            signal_residue = pad_residue
                        
                        # Store results
                        num_imfs = min(signal_imfs.shape[0], self.max_imfs)
                        imfs[b, :num_imfs, :, f] = torch.tensor(signal_imfs[:num_imfs], device=device)
                        residue[b, :, f] = torch.tensor(signal_residue, device=device)
                except Exception as e:
                    print(f"EMD failed for batch {b}, feature {f}: {str(e)}")
                    # Keep zeros for this feature
        
        # Calculate high frequency component as sum of first half of IMFs
        high_freq_imfs = imfs[:, :self.max_imfs//2] if self.max_imfs > 1 else imfs
        high_freq = torch.sum(high_freq_imfs, dim=1)
        
        # Low frequency component is residue plus remaining IMFs
        low_freq_imfs = imfs[:, self.max_imfs//2:] if self.max_imfs > 1 else torch.zeros_like(imfs)
        low_freq = residue + torch.sum(low_freq_imfs, dim=1)
        
        return {
            'imfs': imfs,
            'residue': residue,
            'high_freq': high_freq,
            'low_freq': low_freq
        }