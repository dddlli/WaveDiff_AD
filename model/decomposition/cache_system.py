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
    
    def decompose(self, wavelet, signal, complexity_scores=None):
        """
        Get decomposition result from cache or compute and cache
        
        Args:
            wavelet: Wavelet transform module
            signal: Input signal tensor
            complexity_scores: Signal complexity scores (optional)
            
        Returns:
            Decomposition result dictionary
        """
        # Generate keys
        tensor_key = self._get_tensor_hash(signal)
        wavelet_key = self._get_wavelet_hash(wavelet)
        
        # Add complexity scores to cache key if provided
        complexity_key = ""
        if complexity_scores is not None:
            complexity_bytes = complexity_scores.detach().cpu().numpy().tobytes()
            complexity_key = "_" + hashlib.md5(complexity_bytes).hexdigest()
        
        cache_key = f"{wavelet_key}_{tensor_key}{complexity_key}"
        
        # Check if in cache
        if cache_key in self.decomposition_cache:
            return self.decomposition_cache[cache_key]
        
        # Compute and cache
        if hasattr(wavelet, 'decompose_original'):
            # Use original method if it exists
            if complexity_scores is None:
                result = wavelet.decompose_original(signal)
            else:
                result = wavelet.decompose_original(signal, complexity_scores)
        else:
            # Use default forward method
            if complexity_scores is None:
                result = wavelet.forward(signal)
            else:
                result = wavelet.forward(signal, complexity_scores)
        
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
        if hasattr(wavelet, 'reconstruct_original'):
            result = wavelet.reconstruct_original(components)
        else:
            result = wavelet.reconstruct(components)
        
        # Manage cache size
        if len(self.reconstruction_cache) >= self.max_size:
            self.reconstruction_cache.pop(next(iter(self.reconstruction_cache)))
        
        self.reconstruction_cache[cache_key] = result
        return result


class ComplexityCache:
    """
    Cache for complexity analysis results (AAPE)
    """
    def __init__(self, max_size=256):
        self.max_size = max_size
        self.complexity_cache = {}
    
    def _get_tensor_hash(self, tensor):
        """Generate a hash key for a tensor"""
        data_bytes = tensor.detach().cpu().numpy().tobytes()
        tensor_hash = hashlib.md5(data_bytes).hexdigest()
        return tensor_hash
    
    def get_complexity(self, module, signal):
        """
        Get complexity scores from cache or compute and cache
        
        Args:
            module: Complexity analysis module
            signal: Input signal
            
        Returns:
            Complexity scores
        """
        # Generate cache key
        tensor_key = self._get_tensor_hash(signal)
        module_key = f"{module.__class__.__name__}_{id(module)}"
        cache_key = f"{module_key}_{tensor_key}"
        
        # Check if in cache
        if cache_key in self.complexity_cache:
            return self.complexity_cache[cache_key]
        
        # Compute and cache
        result = module(signal)
        
        # Manage cache size
        if len(self.complexity_cache) >= self.max_size:
            self.complexity_cache.pop(next(iter(self.complexity_cache)))
        
        self.complexity_cache[cache_key] = result
        return result


class HHTCache:
    """
    Cache for HHT decomposition results
    """
    def __init__(self, device='cpu', max_size=128):
        self.device = device
        self.max_size = max_size
        self.decomposition_cache = {}
    
    def _get_tensor_hash(self, tensor):
        """Generate a hash key for a tensor"""
        data_bytes = tensor.detach().cpu().numpy().tobytes()
        tensor_hash = hashlib.md5(data_bytes).hexdigest()
        return tensor_hash
    
    def _get_module_hash(self, module):
        """Generate a hash for HHT module"""
        params = f"{module.max_imfs}_{module.sift_threshold}_{getattr(module, 'adaptive_mode', False)}"
        return hashlib.md5(params.encode()).hexdigest()
    
    def decompose(self, module, signal, complexity_scores=None):
        """
        Get HHT decomposition from cache or compute and cache
        
        Args:
            module: HHT module
            signal: Input signal
            complexity_scores: Signal complexity scores (optional)
            
        Returns:
            HHT decomposition result
        """
        # Generate cache key
        tensor_key = self._get_tensor_hash(signal)
        module_key = self._get_module_hash(module)
        
        # Add complexity to cache key if provided
        complexity_key = ""
        if complexity_scores is not None:
            complexity_bytes = complexity_scores.detach().cpu().numpy().tobytes()
            complexity_key = "_" + hashlib.md5(complexity_bytes).hexdigest()
        
        cache_key = f"{module_key}_{tensor_key}{complexity_key}"
        
        # Check if in cache
        if cache_key in self.decomposition_cache:
            return self.decomposition_cache[cache_key]
        
        # Compute and cache
        if complexity_scores is None:
            result = module(signal)
        else:
            result = module(signal, complexity_scores)
        
        # Manage cache size
        if len(self.decomposition_cache) >= self.max_size:
            self.decomposition_cache.pop(next(iter(self.decomposition_cache)))
        
        self.decomposition_cache[cache_key] = result
        return result


class FusionCache:
    """
    Cache for fusion results
    """
    def __init__(self, device='cpu', max_size=128):
        self.device = device
        self.max_size = max_size
        self.fusion_cache = {}
    
    def _get_tensor_hash(self, tensor):
        """Generate a hash key for a tensor"""
        data_bytes = tensor.detach().cpu().numpy().tobytes()
        tensor_hash = hashlib.md5(data_bytes).hexdigest()
        return tensor_hash
    
    def _get_dict_hash(self, tensor_dict, keys_to_hash):
        """Generate a hash for a dictionary of tensors"""
        combined_hash = ""
        for key in keys_to_hash:
            if key in tensor_dict:
                tensor = tensor_dict[key]
                if isinstance(tensor, torch.Tensor):
                    combined_hash += key + "_" + self._get_tensor_hash(tensor) + "_"
        return hashlib.md5(combined_hash.encode()).hexdigest()
    
    def _get_module_hash(self, module):
        """Generate a hash for fusion module"""
        params = f"{module.fusion_type}_{getattr(module, 'feature_dim', 0)}"
        return hashlib.md5(params.encode()).hexdigest()
    
    def fuse(self, module, wavelet_out, hht_out, original_signal=None):
        """
        Get fusion result from cache or compute and cache
        
        Args:
            module: Fusion module
            wavelet_out: Wavelet decomposition output
            hht_out: HHT decomposition output
            original_signal: Original signal (optional)
            
        Returns:
            Fusion result
        """
        # Generate cache key
        wavelet_key = self._get_dict_hash(wavelet_out, ['high_freq', 'low_freq'])
        hht_key = self._get_dict_hash(hht_out, ['high_freq', 'low_freq'])
        module_key = self._get_module_hash(module)
        
        original_key = ""
        if original_signal is not None:
            original_key = "_" + self._get_tensor_hash(original_signal)
        
        cache_key = f"{module_key}_{wavelet_key}_{hht_key}{original_key}"
        
        # Check if in cache
        if cache_key in self.fusion_cache:
            return self.fusion_cache[cache_key]
        
        # Compute and cache
        result = module(wavelet_out, hht_out, original_signal)
        
        # Manage cache size
        if len(self.fusion_cache) >= self.max_size:
            self.fusion_cache.pop(next(iter(self.fusion_cache)))
        
        self.fusion_cache[cache_key] = result
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
        if hasattr(model, 'forward_original'):
            result = model.forward_original(x)
        else:
            result = model(x)
        
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
            cls._instance.hht_caches = {}
            cls._instance.feature_caches = {}
            cls._instance.complexity_caches = {}
            cls._instance.fusion_caches = {}
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
    
    def get_hht_cache(self, device='cpu'):
        """
        Get HHT cache for a specific device
        
        Args:
            device: Device string or object
            
        Returns:
            HHTCache instance
        """
        device_str = str(device)
        if device_str not in self.hht_caches:
            self.hht_caches[device_str] = HHTCache(device=device)
        return self.hht_caches[device_str]
    
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
    
    def get_complexity_cache(self):
        """
        Get complexity analysis cache
        
        Returns:
            ComplexityCache instance
        """
        if 'default' not in self.complexity_caches:
            self.complexity_caches['default'] = ComplexityCache()
        return self.complexity_caches['default']
    
    def get_fusion_cache(self, device='cpu'):
        """
        Get fusion cache for a specific device
        
        Args:
            device: Device string or object
            
        Returns:
            FusionCache instance
        """
        device_str = str(device)
        if device_str not in self.fusion_caches:
            self.fusion_caches[device_str] = FusionCache(device=device)
        return self.fusion_caches[device_str]
    
    def clear_all_caches(self):
        """Clear all caches"""
        for cache in self.wavelet_caches.values():
            cache.decomposition_cache.clear()
            cache.reconstruction_cache.clear()
        
        for cache in self.hht_caches.values():
            cache.decomposition_cache.clear()
        
        for cache in self.feature_caches.values():
            cache.feature_cache.clear()
        
        for cache in self.complexity_caches.values():
            cache.complexity_cache.clear()
        
        for cache in self.fusion_caches.values():
            cache.fusion_cache.clear()


# Create helper functions to wrap cache operations
def cached_wavelet_decompose(wavelet_module, signal, complexity_scores=None, device='cpu'):
    """Cached wavelet decomposition wrapper function with enhanced error handling"""
    cache_manager = CacheManager()
    if not cache_manager.is_enabled():
        # Safe direct call without caching
        try:
            if hasattr(wavelet_module, 'decompose'):
                return wavelet_module.decompose(signal, complexity_scores)
            elif complexity_scores is None:
                return wavelet_module(signal)
            else:
                return wavelet_module(signal, None, complexity_scores)
        except Exception as e:
            print(f"Warning: Direct wavelet decomposition failed: {e}")
            # Final fallback - call forward directly
            if hasattr(wavelet_module, 'forward'):
                return wavelet_module.forward(signal, None, complexity_scores)
            return wavelet_module(signal)
    
    try:
        # Get cache with error handling
        wavelet_cache = cache_manager.get_wavelet_cache(device)
        
        # Try cached version
        try:
            return wavelet_cache.decompose(wavelet_module, signal, complexity_scores)
        except Exception as e:
            print(f"Warning: Cache retrieval failed: {e}")
            
            # Direct call as fallback
            if hasattr(wavelet_module, 'decompose'):
                return wavelet_module.decompose(signal, complexity_scores)
            elif complexity_scores is None:
                return wavelet_module(signal)
            else:
                return wavelet_module(signal, None, complexity_scores)
    except Exception as e:
        print(f"Warning: Cache system error: {e}")
        # Final fallback
        if hasattr(wavelet_module, 'forward'):
            return wavelet_module.forward(signal, None, complexity_scores)
        return wavelet_module(signal)

def cached_hht_decompose(hht_module, signal, complexity_scores=None, device='cpu'):
    """Cached HHT decomposition wrapper function with enhanced error handling"""
    cache_manager = CacheManager()
    if not cache_manager.is_enabled():
        # Safe direct call without caching
        try:
            if hasattr(hht_module, 'decompose'):
                return hht_module.decompose(signal, complexity_scores)
            elif complexity_scores is None:
                return hht_module(signal)
            else:
                return hht_module(signal, complexity_scores)
        except Exception as e:
            print(f"Warning: Direct HHT decomposition failed: {e}")
            # Final fallback
            return hht_module(signal)
    
    try:
        # Get cache with error handling
        hht_cache = cache_manager.get_hht_cache(device)
        
        # Try cached version
        try:
            return hht_cache.decompose(hht_module, signal, complexity_scores)
        except Exception as e:
            print(f"Warning: HHT cache retrieval failed: {e}")
            
            # Direct call as fallback
            if hasattr(hht_module, 'decompose'):
                return hht_module.decompose(signal, complexity_scores)
            elif complexity_scores is None:
                return hht_module(signal)
            else:
                return hht_module(signal, complexity_scores)
    except Exception as e:
        print(f"Warning: HHT cache system error: {e}")
        # Final fallback
        return hht_module(signal)

def cached_fusion(fusion_module, wavelet_out, hht_out, original_signal=None, device='cpu'):
    """Cached fusion wrapper function with enhanced error handling"""
    cache_manager = CacheManager()
    if not cache_manager.is_enabled():
        # Safe direct call without caching
        try:
            return fusion_module(wavelet_out, hht_out, original_signal)
        except Exception as e:
            print(f"Warning: Direct fusion failed: {e}")
            # Try alternative calling style if the first one fails
            if original_signal is not None:
                return fusion_module(wavelet_out, hht_out, original_signal=original_signal)
            return fusion_module(wavelet_out, hht_out)
    
    try:
        # Get cache with error handling
        fusion_cache = cache_manager.get_fusion_cache(device)
        
        # Try cached version
        try:
            return fusion_cache.fuse(fusion_module, wavelet_out, hht_out, original_signal)
        except Exception as e:
            print(f"Warning: Fusion cache retrieval failed: {e}")
            
            # Direct call as fallback
            try:
                return fusion_module(wavelet_out, hht_out, original_signal)
            except Exception as e2:
                print(f"Warning: Secondary fusion call failed: {e2}")
                # Try alternative calling style
                if original_signal is not None:
                    return fusion_module(wavelet_out, hht_out, original_signal=original_signal)
                return fusion_module(wavelet_out, hht_out)
    except Exception as e:
        print(f"Warning: Fusion cache system error: {e}")
        # Final fallback with minimal arguments
        return fusion_module(wavelet_out, hht_out)

def cached_complexity_analysis(complexity_module, signal):
    """Cached complexity analysis wrapper function with enhanced error handling"""
    cache_manager = CacheManager()
    if not cache_manager.is_enabled():
        # Safe direct call without caching
        try:
            return complexity_module(signal)
        except Exception as e:
            print(f"Warning: Direct complexity analysis failed: {e}")
            # Return default values as fallback
            return (torch.ones(signal.shape[2], device=signal.device) * 0.5, 
                   torch.ones(signal.shape[0], signal.shape[2], device=signal.device) / signal.shape[2])
    
    try:
        # Get cache with error handling
        complexity_cache = cache_manager.get_complexity_cache()
        
        # Try cached version
        try:
            return complexity_cache.get_complexity(complexity_module, signal)
        except Exception as e:
            print(f"Warning: Complexity cache retrieval failed: {e}")
            
            # Direct call as fallback
            try:
                return complexity_module(signal)
            except Exception as e2:
                print(f"Warning: Secondary complexity analysis failed: {e2}")
                # Return default values as final fallback
                return (torch.ones(signal.shape[2], device=signal.device) * 0.5, 
                       torch.ones(signal.shape[0], signal.shape[2], device=signal.device) / signal.shape[2])
    except Exception as e:
        print(f"Warning: Complexity cache system error: {e}")
        # Return default values
        return (torch.ones(signal.shape[2], device=signal.device) * 0.5, 
               torch.ones(signal.shape[0], signal.shape[2], device=signal.device) / signal.shape[2])