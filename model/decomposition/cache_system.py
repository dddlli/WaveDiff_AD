import torch
import numpy as np
from functools import lru_cache
import hashlib

class WaveletCache:
    """
    Cache for wavelet transform operations to avoid redundant computations
    """
    def __init__(self, device='cpu', max_size=128):
        self.device = device
        self.max_size = max_size
        self.decomposition_cache = {}
        self.reconstruction_cache = {}
    
    def _get_tensor_hash(self, tensor):
        """
        Generate a hash key for a tensor
        
        Args:
            tensor: PyTorch tensor
            
        Returns:
            String hash key
        """
        # Get data as bytes
        data_bytes = tensor.detach().cpu().numpy().tobytes()
        # Generate hash
        tensor_hash = hashlib.md5(data_bytes).hexdigest()
        return tensor_hash
    
    def _get_wavelet_hash(self, wavelet):
        """
        Generate a hash key for a wavelet transform module
        
        Args:
            wavelet: Wavelet transform module
            
        Returns:
            String hash key
        """
        params = f"{wavelet.wavelet}_{wavelet.level}_{wavelet.mode}"
        return hashlib.md5(params.encode()).hexdigest()
    
    def decompose(self, wavelet, signal):
        """
        Get decomposition result from cache or compute and cache
        
        Args:
            wavelet: Wavelet transform module
            signal: Input signal tensor
            
        Returns:
            Decomposition result dictionary
        """
        # Generate keys
        tensor_key = self._get_tensor_hash(signal)
        wavelet_key = self._get_wavelet_hash(wavelet)
        cache_key = f"{wavelet_key}_{tensor_key}"
        
        # Check if in cache
        if cache_key in self.decomposition_cache:
            return self.decomposition_cache[cache_key]
        
        # Compute and cache
        result = wavelet.decompose_original(signal)
        
        # Manage cache size
        if len(self.decomposition_cache) >= self.max_size:
            # Remove oldest entry (could be improved with LRU)
            self.decomposition_cache.pop(next(iter(self.decomposition_cache)))
        
        self.decomposition_cache[cache_key] = result
        return result
    
    def reconstruct(self, wavelet, components):
        """
        Get reconstruction result from cache or compute and cache
        
        Args:
            wavelet: Wavelet transform module
            components: Component dictionary
            
        Returns:
            Reconstructed signal
        """
        # Generate keys
        approx_key = self._get_tensor_hash(components['approx'])
        details_key = "_".join([self._get_tensor_hash(detail) for detail in components['details']])
        wavelet_key = self._get_wavelet_hash(wavelet)
        cache_key = f"{wavelet_key}_{approx_key}_{details_key}"
        
        # Check if in cache
        if cache_key in self.reconstruction_cache:
            return self.reconstruction_cache[cache_key]
        
        # Compute and cache
        result = wavelet.reconstruct_original(components)
        
        # Manage cache size
        if len(self.reconstruction_cache) >= self.max_size:
            self.reconstruction_cache.pop(next(iter(self.reconstruction_cache)))
        
        self.reconstruction_cache[cache_key] = result
        return result


class FeatureCache:
    """
    Cache for learnable feature extraction
    """
    def __init__(self, device='cpu', max_size=128):
        self.device = device
        self.max_size = max_size
        self.feature_cache = {}
    
    def _get_tensor_hash(self, tensor):
        """Generate a hash key for a tensor"""
        data_bytes = tensor.detach().cpu().numpy().tobytes()
        tensor_hash = hashlib.md5(data_bytes).hexdigest()
        return tensor_hash
    
    def _get_model_hash(self, model):
        """Generate a hash based on model parameters"""
        param_str = ""
        for name, param in model.named_parameters():
            if param.requires_grad:
                param_str += f"{name}_{param.sum().item():.4f}_"
        return hashlib.md5(param_str.encode()).hexdigest()
    
    def get_features(self, model, x):
        """
        Get features from cache or compute and cache
        
        Args:
            model: Feature extraction model
            x: Input tensor
            
        Returns:
            Extracted features
        """
        # Generate keys
        tensor_key = self._get_tensor_hash(x)
        model_key = self._get_model_hash(model)
        cache_key = f"{model_key}_{tensor_key}"
        
        # Check if in cache
        if cache_key in self.feature_cache:
            return self.feature_cache[cache_key]
        
        # Compute and cache
        result = model.forward_original(x)
        
        # Manage cache size
        if len(self.feature_cache) >= self.max_size:
            self.feature_cache.pop(next(iter(self.feature_cache)))
        
        self.feature_cache[cache_key] = result
        return result


class CacheManager:
    """
    Global cache manager for various caching needs
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CacheManager, cls).__new__(cls)
            cls._instance.enabled = True
            cls._instance.wavelet_caches = {}
            cls._instance.feature_caches = {}
        return cls._instance
    
    def enable(self):
        """Enable caching"""
        self.enabled = True
    
    def disable(self):
        """Disable caching"""
        self.enabled = False
    
    def is_enabled(self):
        """Check if caching is enabled"""
        return self.enabled
    
    def get_wavelet_cache(self, device='cpu'):
        """
        Get wavelet cache for a specific device
        
        Args:
            device: Device string or object
            
        Returns:
            WaveletCache instance
        """
        device_str = str(device)
        if device_str not in self.wavelet_caches:
            self.wavelet_caches[device_str] = WaveletCache(device=device)
        return self.wavelet_caches[device_str]
    
    def get_feature_cache(self, device='cpu'):
        """
        Get feature cache for a specific device
        
        Args:
            device: Device string or object
            
        Returns:
            FeatureCache instance
        """
        device_str = str(device)
        if device_str not in self.feature_caches:
            self.feature_caches[device_str] = FeatureCache(device=device)
        return self.feature_caches[device_str]
    
    def clear_all_caches(self):
        """Clear all caches"""
        for cache in self.wavelet_caches.values():
            cache.decomposition_cache.clear()
            cache.reconstruction_cache.clear()
        
        for cache in self.feature_caches.values():
            cache.feature_cache.clear()