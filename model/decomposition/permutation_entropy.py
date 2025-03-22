import numpy as np
from itertools import permutations
import torch
import torch.nn as nn

def compute_aape(ts, m=3, delay=1):
    """
    Compute Amplitude-Aware Permutation Entropy (AAPE)
    
    Args:
        ts: Time series data (numpy array)
        m: Embedding dimension (default: 3)
        delay: Time delay (default: 1)
        
    Returns:
        AAPE value normalized to [0,1]
    """
    n = len(ts)
    if n < m * delay:
        return 0.5  # Default value for very short series
    
    # Find all possible permutation patterns
    possible_patterns = list(permutations(range(m)))
    pattern_dict = {i: 0 for i in range(len(possible_patterns))}
    
    # Extract patterns and their amplitudes
    pattern_weights = {}
    total_weight = 0
    
    for i in range(n - (m-1)*delay):
        # Extract sequence
        sequence = np.array([ts[i + j*delay] for j in range(m)])
        
        # Determine permutation pattern
        sorted_idx = np.argsort(sequence)
        pattern = tuple(sorted_idx)
        pattern_idx = possible_patterns.index(pattern)
        
        # Calculate amplitude weight: enhanced with L2-norm and range for better sensitivity
        ampl_var = np.var(sequence) + 1e-10  # Add small constant to avoid zero weights
        ampl_range = np.max(sequence) - np.min(sequence)
        l2_norm = np.sqrt(np.sum(sequence**2)) / m
        
        # Combined amplitude weight
        weight = ampl_var * (1 + 0.5 * ampl_range) * l2_norm
        
        # Update pattern weights
        if pattern_idx not in pattern_weights:
            pattern_weights[pattern_idx] = 0
        pattern_weights[pattern_idx] += weight
        total_weight += weight
    
    # Normalize weights and calculate entropy
    aape = 0
    for pattern_idx in pattern_weights:
        prob = pattern_weights[pattern_idx] / total_weight
        aape -= prob * np.log2(prob)
    
    # Normalize by maximum entropy log2(m!)
    max_entropy = np.log2(np.math.factorial(m))
    if max_entropy > 0:
        aape /= max_entropy
    
    return aape


class PermutationEntropyModule(nn.Module):
    """
    PyTorch module for computing Amplitude-Aware Permutation Entropy (AAPE)
    
    This module provides functionality to compute AAPE for batches of time series data.
    AAPE is a measure of signal complexity that considers both order patterns and amplitudes.
    """
    def __init__(self, m=3, delay=1, sensitivity=1.0):
        """
        Initialize AAPE module
        
        Args:
            m: Embedding dimension (default: 3)
            delay: Time delay (default: 1)
            sensitivity: Scaling factor for amplitude sensitivity (default: 1.0)
        """
        super(PermutationEntropyModule, self).__init__()
        self.m = m
        self.delay = delay
        self.sensitivity = nn.Parameter(torch.tensor(sensitivity))
    
    def compute_aape_batch(self, x_np):
        """
        Compute AAPE for a batch of numpy time series
        
        Args:
            x_np: Batch of time series data [batch_size, seq_len, features]
            
        Returns:
            AAPE scores [features]
        """
        batch_size, seq_len, features = x_np.shape
        aape_scores = np.zeros(features)
        
        # Compute AAPE for each feature (averaging across batch)
        for f in range(features):
            feature_aape = 0
            for b in range(batch_size):
                feature_aape += compute_aape(x_np[b, :, f], self.m, self.delay)
            aape_scores[f] = feature_aape / batch_size
        
        return aape_scores
    
    def forward(self, x):
        """
        Compute AAPE for a PyTorch tensor of time series
        
        Args:
            x: Time series data [batch_size, seq_len, features]
            
        Returns:
            AAPE scores [features] and energy distribution tensor
        """
        # Convert to numpy for computation
        x_np = x.detach().cpu().numpy()
        
        # Compute AAPE for entire batch
        aape_scores = self.compute_aape_batch(x_np)
        aape_scores = torch.tensor(aape_scores, device=x.device)
        
        # Compute signal energy distribution for frequency bands
        energy = torch.mean(x**2, dim=1)  # Mean energy across time dimension
        energy_ratio = energy / (torch.sum(energy, dim=1, keepdim=True) + 1e-8)
        
        # Apply sensitivity scaling to entropy scores
        scaled_scores = aape_scores * torch.sigmoid(self.sensitivity)
        
        return scaled_scores, energy_ratio