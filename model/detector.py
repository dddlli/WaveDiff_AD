import torch
from torch import nn
import torch.nn.functional as F
import sys

from model.decomposition.wavelet import LearnableWaveletTransform
from model.decomposition.hht import HilbertHuangTransform
from model.decomposition.signal_decomp import AdaptiveFusion
from model.decomposition.permutation_entropy import PermutationEntropyModule
from model.diffusion.diffusion import Diffusion 
from model.diffusion.denoise import ConditionalDenoisingNetwork
from model.reconstruction import Reconstruction

class AnomalyDetection(nn.Module):
    """
    Enhanced anomaly detection model with wavelet-HHT decomposition,
    diffusion process, and energy conservation constraints.
    """
    def __init__(self, 
                 time_steps=1000, 
                 beta_start=0.0001, 
                 beta_end=0.02, 
                 window_size=64, 
                 model_dim=512, 
                 ff_dim=2048, 
                 atten_dim=64, 
                 feature_num=51,  # Note: This parameter will be adjusted based on actual data
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
        
        # Print system information for debugging
        print(f"Python version: {sys.version}")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA devices: {torch.cuda.device_count()}")
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        
        # Store all parameters for later initialization
        self.wavelet_level = wavelet_level
        self.hht_imfs = hht_imfs
        self.fusion_type = fusion_type
        self.model_dim = model_dim
        self.ff_dim = ff_dim
        self.atten_dim = atten_dim
        self.time_num = time_num
        self.block_num = block_num
        self.head_num = head_num
        self.dropout = dropout
        
        # First call flag for dynamic initialization
        self.first_forward_call = True
        
        # Complexity analyzer - not dependent on specific feature dimension
        self.complexity_analyzer = PermutationEntropyModule(
            m=3, 
            delay=1,
            sensitivity=1.5
        )
        
        # Diffusion process - not dependent on specific feature dimension
        self.diffusion = Diffusion(
            time_steps=time_steps,
            beta_start=beta_start,
            beta_end=beta_end,
            device=device
        )
        
        # Loss function
        self.mse_loss = nn.MSELoss(reduction='mean')
        
    def _create_model_components(self, actual_feature_dim):
        """
        Create model components based on actual feature dimension.
        
        Args:
            actual_feature_dim: The actual feature dimension from input data
        """
        print(f"Creating model components with feature dimension: {actual_feature_dim}")
        
        # Wavelet transform
        self.wavelet = LearnableWaveletTransform(
            feature_dim=actual_feature_dim,
            level=self.wavelet_level, 
            wavelet_init='db4', 
            mode='zero'
        ).to(self.device)
        
        # HHT decomposition
        self.hht = HilbertHuangTransform(
            max_imfs=self.hht_imfs,
            sift_threshold=0.05,
            adaptive_mode=True
        ).to(self.device)
        
        # Adaptive fusion
        self.fusion = AdaptiveFusion(
            feature_dim=actual_feature_dim,
            fusion_type=self.fusion_type,
            window_size=self.window_size
        ).to(self.device)
        
        # Denoising network
        self.denoiser = ConditionalDenoisingNetwork(
            feature_dim=actual_feature_dim,
            window_size=self.window_size,
            time_steps=self.time_steps,
            hidden_dim=self.model_dim
        ).to(self.device)
        
        # Reconstruction module
        self.reconstruction = Reconstruction(
            window_size=self.window_size,
            model_dim=self.model_dim,
            ff_dim=self.ff_dim,
            atten_dim=self.atten_dim,
            feature_num=actual_feature_dim,
            time_num=self.time_num,
            block_num=self.block_num,
            head_num=self.head_num,
            dropout=self.dropout
        ).to(self.device)
        
    def forward(self, data, time, p=0):
        """
        Forward pass with dynamic feature dimension handling and loss scaling.
        
        Args:
            data: Input time series data [batch_size, seq_len, features]
            time: Time features [batch_size, seq_len, time_features]
            p: Disturbance factor for robustness training
            
        Returns:
            Tuple of (high_freq, low_freq, recon, losses)
        """
        # Get current batch feature dimension
        batch_size, seq_len, feature_dim = data.shape
        
        # Initialize components on first call
        if self.first_forward_call:
            self._create_model_components(feature_dim)
            self.first_forward_call = False
            print(f"Model initialized with feature dimension: {feature_dim}")
        
        # Create disturbance for robustness
        disturb = torch.rand(batch_size, feature_dim) * p
        disturb = disturb.unsqueeze(1).repeat(1, self.window_size, 1).float().to(self.device)
        data_disturbed = data + disturb
        
        # Compute signal complexity
        try:
            complexity_scores, energy_ratio = self.complexity_analyzer(data_disturbed)
        except Exception as e:
            print(f"Warning: Complexity analysis failed: {e}")
            complexity_scores = torch.ones(feature_dim, device=self.device) * 0.5
            energy_ratio = None
        
        # Wavelet decomposition
        try:
            wavelet_out = self.wavelet(data_disturbed, None, complexity_scores)
        except Exception as e:
            print(f"Warning: Wavelet transform failed: {e}")
            # Simple fallback: split original signal equally
            wavelet_out = {
                'high_freq': data_disturbed * 0.5,
                'low_freq': data_disturbed * 0.5
            }
        
        # HHT decomposition
        try:
            hht_out = self.hht(data_disturbed, complexity_scores)
        except Exception as e:
            print(f"Warning: HHT failed: {e}")
            # Use wavelet results as fallback
            hht_out = {
                'high_freq': wavelet_out['high_freq'].clone(),
                'low_freq': wavelet_out['low_freq'].clone()
            }
        
        # Fusion of wavelet and HHT results
        try:
            fusion_out = self.fusion(wavelet_out, hht_out, data_disturbed)
            high_freq = fusion_out['high_freq']
            low_freq = fusion_out['low_freq']
            energy_loss = fusion_out.get('energy_loss')
        except Exception as e:
            print(f"Warning: Fusion failed: {e}")
            # Fallback: use wavelet results directly
            high_freq = wavelet_out['high_freq']
            low_freq = wavelet_out['low_freq']
            energy_loss = None
        
        # Ensure shape consistency
        min_len = min(high_freq.shape[1], low_freq.shape[1])
        high_freq = high_freq[:, :min_len, :]
        low_freq = low_freq[:, :min_len, :]
        data_disturbed_adj = data_disturbed[:, :min_len, :]
        
        # Apply diffusion process
        timesteps = torch.full((batch_size,), self.t, device=self.device)
        noise = torch.randn_like(data_disturbed_adj).to(self.device)
        noisy_data = self.diffusion.q_sample(data_disturbed_adj, low_freq, timesteps, noise)
        
        # Apply denoising network
        try:
            denoised = self.denoiser(noisy_data, timesteps, high_freq, low_freq)
        except Exception as e:
            print(f"Warning: Denoising failed: {e}")
            denoised = noisy_data  # Use noisy data as fallback
        
        # Adjust time features
        time_adj = time[:, :min_len, :] if time.shape[1] > min_len else time
        
        # Final reconstruction
        try:
            recon = self.reconstruction(noisy_data, high_freq, low_freq, time_adj)
        except Exception as e:
            print(f"Warning: Reconstruction failed: {e}")
            # Fallback: use denoised data as reconstruction
            recon = denoised
        
        # Calculate losses with normalization
        recon_loss = self.mse_loss(recon, data_disturbed_adj)
        denoise_loss = self.mse_loss(denoised, data_disturbed_adj)
        decomp_loss = self.mse_loss(high_freq + low_freq, data_disturbed_adj)
        
        # Normalize losses by data points for better scaling
        data_points = batch_size * seq_len * feature_dim
        recon_loss = recon_loss / data_points
        denoise_loss = denoise_loss / data_points
        decomp_loss = decomp_loss / data_points
        
        # Combined losses with scaled energy loss
        total_loss = 0.3 * recon_loss + 0.3 * denoise_loss + 0.2 * decomp_loss
        if energy_loss is not None:
            # Scale energy loss to avoid dominating the total loss
            energy_scale = 0.01  # Reduce energy loss impact
            total_loss = total_loss + 0.2 * energy_loss * energy_scale
        
        # Loss collection
        losses = {
            'total': total_loss,
            'recon': recon_loss,
            'denoise': denoise_loss,
            'decomp': decomp_loss,
            'energy': energy_loss
        }
        
        # Remove disturbance (if applied)
        if p > 0:
            component_sum = high_freq + low_freq
            component_sum_safe = component_sum + 1e-10
            high_freq = high_freq - disturb[:, :min_len, :] * (high_freq / component_sum_safe)
            low_freq = low_freq - disturb[:, :min_len, :] * (low_freq / component_sum_safe)
            recon = recon - disturb[:, :min_len, :]
        
        return high_freq, low_freq, recon, losses
    
    def compute_anomaly_score(self, data, time):
        """
        Compute anomaly score with complexity-aware component weighting.
        
        Args:
            data: Input time series data
            time: Time features
            
        Returns:
            Dictionary with anomaly scores and components
        """
        # Initialize model on first call
        if self.first_forward_call:
            self._create_model_components(data.shape[2])
            self.first_forward_call = False
        
        with torch.no_grad():
            # Complexity analysis
            try:
                complexity_scores, energy_ratio = self.complexity_analyzer(data)
            except Exception as e:
                print(f"Warning: Complexity analysis failed in scoring: {e}")
                complexity_scores = torch.ones(data.shape[2], device=data.device) * 0.5
            
            # Decomposition and reconstruction
            high_freq, low_freq, recon, _ = self.forward(data, time, p=0)
            
            # Reconstruction error
            recon_error = torch.mean((data - recon)**2, dim=-1)
            
            # Component-specific errors
            high_freq_error = torch.mean((data - (low_freq + recon))**2, dim=-1)
            low_freq_error = torch.mean((data - (high_freq + recon))**2, dim=-1)
            
            # Complexity-based adaptive weights
            complexity_factor = 0.5  # Default value
            if complexity_scores is not None:
                try:
                    complexity_factor = torch.mean(complexity_scores).item()
                    complexity_factor = max(0.0, min(1.0, complexity_factor))  # Clip to [0,1]
                except:
                    pass
            
            # Set component weights
            alpha = 0.5  # Reconstruction error weight
            beta = 0.25 + 0.2 * complexity_factor  # High frequency error weight
            gamma = 0.25 - 0.2 * complexity_factor  # Low frequency error weight
            
            # Ensure weights sum to 1
            total = alpha + beta + gamma
            alpha /= total
            beta /= total
            gamma /= total
            
            # Calculate combined anomaly score
            anomaly_score = alpha * recon_error + beta * high_freq_error + gamma * low_freq_error
            
            return {
                'score': anomaly_score,
                'recon_error': recon_error,
                'high_freq_error': high_freq_error,
                'low_freq_error': low_freq_error,
                'high_freq': high_freq,
                'low_freq': low_freq,
                'recon': recon,
                'complexity': complexity_scores
            }