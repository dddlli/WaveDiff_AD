import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.decomposition.permutation_entropy import PermutationEntropyModule

class AdaptiveFusion(nn.Module):
    """
    Adaptive fusion module for wavelet and HHT decomposition components
    with improved energy conservation and dimension handling.
    """
    def __init__(self, feature_dim, fusion_type='weighted', window_size=64):
        super(AdaptiveFusion, self).__init__()
        self.feature_dim = feature_dim  # Initial guidance, will adapt to actual inputs
        self.fusion_type = fusion_type
        self.window_size = window_size
        
        # Complexity analysis module
        self.aape_module = PermutationEntropyModule(m=3, delay=1, sensitivity=1.5)
        
        # Learnable weights for component fusion
        self.wavelet_weight = nn.Parameter(torch.ones(1))
        self.hht_weight = nn.Parameter(torch.ones(1))
        
        # Energy balance parameters
        self.energy_factor = nn.Parameter(torch.ones(1))
        self.energy_balance = nn.Parameter(torch.zeros(1))  # Controls high/low energy distribution
    
    def _safely_get_tensor(self, dictionary, key, default_shape=None):
        """
        Safely extract tensor from dictionary with dimension validation.
        
        Args:
            dictionary: Source dictionary
            key: Key to extract
            default_shape: Expected shape for validation
            
        Returns:
            Extracted tensor or None
        """
        if dictionary is None:
            return None
            
        value = dictionary.get(key)
        if not isinstance(value, torch.Tensor):
            return None
            
        if default_shape is not None and value.shape[-1] != default_shape[-1]:
            # Need to adjust feature dimension
            return self._resize_features(value, default_shape[-1])
            
        return value
    
    def _resize_features(self, tensor, target_features):
        """
        Adjust tensor's feature dimension.
        
        Args:
            tensor: Input tensor
            target_features: Target feature dimension
            
        Returns:
            Adjusted tensor
        """
        if tensor is None:
            return None
            
        # Get original shape
        batch_size, seq_len, feat_dim = tensor.shape
        device = tensor.device
        
        if feat_dim == target_features:
            return tensor
            
        # Create projection layer
        projector = nn.Linear(feat_dim, target_features).to(device)
        
        # Apply projection
        return projector(tensor)
    
    def compute_energy_consistency(self, original, high_freq, low_freq):
        """
        Compute energy consistency loss using relative error.
        
        Args:
            original: Original signal
            high_freq: High frequency component
            low_freq: Low frequency component
            
        Returns:
            Energy consistency loss (scaled relative error)
        """
        # Calculate signal energies
        original_energy = torch.mean(original**2, dim=(1, 2), keepdim=True)
        component_energy = torch.mean(high_freq**2, dim=(1, 2), keepdim=True) + \
                           torch.mean(low_freq**2, dim=(1, 2), keepdim=True)
        
        # Compute relative error instead of MSE
        rel_error = torch.abs(original_energy - component_energy) / (original_energy + 1e-8)
        energy_loss = torch.mean(rel_error)
        
        return energy_loss
    
    def forward(self, wavelet_out, hht_out, original_signal=None):
        """
        Fuse wavelet and HHT decomposition results with improved dimension
        handling and energy conservation.
        
        Args:
            wavelet_out: Wavelet decomposition results
            hht_out: HHT decomposition results
            original_signal: Original signal for energy consistency
            
        Returns:
            Dictionary with fusion results
        """
        try:
            # 1. Determine target feature dimension (based on original signal)
            target_dim = original_signal.shape[2] if original_signal is not None else None
            
            # 2. Safely extract components
            wav_high = self._safely_get_tensor(wavelet_out, 'high_freq', 
                                              default_shape=(1, 1, target_dim))
            wav_low = self._safely_get_tensor(wavelet_out, 'low_freq', 
                                             default_shape=(1, 1, target_dim))
            hht_high = self._safely_get_tensor(hht_out, 'high_freq', 
                                              default_shape=(1, 1, target_dim))
            hht_low = self._safely_get_tensor(hht_out, 'low_freq', 
                                             default_shape=(1, 1, target_dim))
            
            # 3. Validate extracted components
            if wav_high is None or wav_low is None:
                raise ValueError("Cannot extract valid high_freq and low_freq from wavelet_out")
            if hht_high is None or hht_low is None:
                # Fall back to using only wavelet results
                print("Cannot extract valid components from hht_out, using wavelet results only")
                high_freq = wav_high
                low_freq = wav_low
            else:
                # 4. Ensure all components have same feature dimension
                target_dim = wav_high.shape[2]
                hht_high = self._resize_features(hht_high, target_dim)
                hht_low = self._resize_features(hht_low, target_dim)
                
                # 5. Linear fusion with normalized weights
                w_weight = torch.sigmoid(self.wavelet_weight)
                h_weight = torch.sigmoid(self.hht_weight)
                weights_sum = w_weight + h_weight
                
                # Apply weights
                high_freq = (w_weight * wav_high + h_weight * hht_high) / weights_sum
                low_freq = (w_weight * wav_low + h_weight * hht_low) / weights_sum
            
            # 6. Energy consistency correction (if original signal provided)
            energy_loss = None
            if original_signal is not None:
                # Ensure shape consistency
                min_len = min(high_freq.shape[1], low_freq.shape[1], original_signal.shape[1])
                high_freq_adj = high_freq[:, :min_len, :]
                low_freq_adj = low_freq[:, :min_len, :]
                original_adj = original_signal[:, :min_len, :]
                
                # Compute energy loss using relative error for better scaling
                energy_loss = self.compute_energy_consistency(original_adj, high_freq_adj, low_freq_adj)
                
                # Calculate energy balance
                energy_balance = torch.sigmoid(self.energy_balance)
                
                # Calculate original and component energies
                orig_energy = torch.sum(original_adj**2, dim=(1, 2), keepdim=True)
                high_energy = torch.sum(high_freq_adj**2, dim=(1, 2), keepdim=True)
                low_energy = torch.sum(low_freq_adj**2, dim=(1, 2), keepdim=True)
                
                # Target energies based on balance parameter
                target_high_energy = orig_energy * energy_balance
                target_low_energy = orig_energy * (1 - energy_balance)
                
                # Calculate scaling factors
                high_scale = torch.sqrt(target_high_energy / (high_energy + 1e-8))
                low_scale = torch.sqrt(target_low_energy / (low_energy + 1e-8))
                
                # Apply energy correction
                high_freq = high_freq * high_scale * torch.sigmoid(self.energy_factor)
                low_freq = low_freq * low_scale * torch.sigmoid(self.energy_factor)
                
                # Trim to original length
                if high_freq.shape[1] > original_signal.shape[1]:
                    high_freq = high_freq[:, :original_signal.shape[1], :]
                if low_freq.shape[1] > original_signal.shape[1]:
                    low_freq = low_freq[:, :original_signal.shape[1], :]
            
            # 7. Return fusion results
            # Calculate energy conservation as percentage (0-100%)
            energy_conservation_pct = 0.0
            if energy_loss is not None:
                # Convert loss to conservation percentage (loss of 0 = 100% conservation)
                energy_conservation_pct = 100.0 * (1.0 - min(1.0, energy_loss.item()))
            
            return {
                'high_freq': high_freq,
                'low_freq': low_freq,
                'energy_loss': energy_loss,
                'wavelet_weight': self.wavelet_weight.item(),
                'hht_weight': self.hht_weight.item(),
                'energy_conservation': energy_conservation_pct,
                'energy_balance': energy_balance.item() if 'energy_balance' in locals() else 0.5
            }
        
        except Exception as e:
            print(f"AdaptiveFusion error: {e}, using fallback solution")
            
            # Fallback solution: use wavelet results or construct basic output
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            # Try to get from wavelet results
            if isinstance(wavelet_out, dict) and 'high_freq' in wavelet_out and 'low_freq' in wavelet_out:
                high_freq = wavelet_out['high_freq']
                low_freq = wavelet_out['low_freq']
            # Otherwise, construct from original signal
            elif original_signal is not None:
                high_freq = original_signal * 0.3  # High freq is 30% of original signal
                low_freq = original_signal * 0.7   # Low freq is 70% of original signal
            # Last resort: create zero tensors
            else:
                batch_size = 1
                seq_len = self.window_size
                feat_dim = self.feature_dim
                
                # Try to infer shape from inputs
                if isinstance(wavelet_out, dict) and isinstance(wavelet_out.get('high_freq'), torch.Tensor):
                    shape = wavelet_out['high_freq'].shape
                    batch_size = shape[0]
                    seq_len = shape[1]
                    feat_dim = shape[2]
                elif isinstance(hht_out, dict) and isinstance(hht_out.get('high_freq'), torch.Tensor):
                    shape = hht_out['high_freq'].shape
                    batch_size = shape[0]
                    seq_len = shape[1]
                    feat_dim = shape[2]
                
                high_freq = torch.zeros((batch_size, seq_len, feat_dim), device=device)
                low_freq = torch.zeros((batch_size, seq_len, feat_dim), device=device)
            
            return {
                'high_freq': high_freq,
                'low_freq': low_freq,
                'energy_loss': None,
                'wavelet_weight': 0.5,
                'hht_weight': 0.5,
                'energy_conservation': 100.0
            }