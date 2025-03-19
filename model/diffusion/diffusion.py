import torch
import numpy as np

class Diffusion:
    """
    Base diffusion process for forward noising and reverse denoising
    """
    def __init__(self, time_steps=1000, beta_start=0.0001, beta_end=0.02, device='cpu'):
        """
        Initialize diffusion process parameters
        
        Args:
            time_steps: Number of diffusion time steps
            beta_start: Starting noise level
            beta_end: Ending noise level
            device: Computation device
        """
        # Create beta schedule
        self.time_steps = time_steps
        self.betas = torch.linspace(beta_start, beta_end, time_steps).float().to(device)
        
        # Pre-compute diffusion parameters
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.one_minus_sqrt_alphas_cumprod = 1. - torch.sqrt(self.alphas_cumprod)
        
        # Additional parameters for sampling
        self.alphas_cumprod_prev = torch.cat([torch.ones(1).to(device), self.alphas_cumprod[:-1]])
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    @staticmethod
    def _extract(data, batch_t, shape):
        """
        Extract values from tensor at specific timesteps
        
        Args:
            data: Tensor with values to extract from
            batch_t: Batch of timesteps
            shape: Shape of output tensor
            
        Returns:
            Extracted values reshaped to match input
        """
        batch_size = batch_t.shape[0]
        out = torch.gather(data, -1, batch_t)
        return out.reshape(batch_size, *((1,) * (len(shape) - 1)))

    def q_sample(self, x_start, condition, batch_t, noise=None):
        """
        Forward diffusion process: add noise to data
        
        Args:
            x_start: Initial clean data
            condition: Conditioning information (e.g., low freq component)
            batch_t: Batch of timesteps
            noise: Optional pre-generated noise
            
        Returns:
            Noisy data at specified timestep
        """
        if noise is None:
            noise = torch.randn_like(x_start)
            
        # Get diffusion parameters for this timestep
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, batch_t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, batch_t, x_start.shape)
        one_minus_sqrt_alphas_cumprod_t = self._extract(self.one_minus_sqrt_alphas_cumprod, batch_t, x_start.shape)
        
        # Apply forward process formula with conditioning
        x_noisy = sqrt_alphas_cumprod_t * x_start + \
                  sqrt_one_minus_alphas_cumprod_t * noise + \
                  one_minus_sqrt_alphas_cumprod_t * condition
        
        return x_noisy
    
    def p_sample(self, model, x_t, t, condition_high, condition_low):
        """
        Sample from reverse diffusion process (single step)
        
        Args:
            model: Denoising model
            x_t: Noisy data at time t
            t: Current timestep
            condition_high: High frequency component for conditioning
            condition_low: Low frequency component for conditioning
            
        Returns:
            Sample from previous timestep (less noisy)
        """
        with torch.no_grad():
            batch_size = x_t.shape[0]
            device = x_t.device
            
            # Get model prediction
            model_output = model(x_t, t, condition_high, condition_low)
            
            # Extract appropriate parameters for this timestep
            betas_t = self._extract(self.betas, t, x_t.shape)
            alphas_cumprod_t = self._extract(self.alphas_cumprod, t, x_t.shape)
            sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
            
            # Compute mean for posterior distribution
            mean = (1 / torch.sqrt(self.alphas[t])) * (
                x_t - (betas_t / sqrt_one_minus_alphas_cumprod_t) * model_output
            )
            
            # Add condition
            mean = mean + (1 - alphas_cumprod_t) * condition_low
            
            # Sample from posterior
            noise = torch.randn_like(x_t) if t > 0 else torch.zeros_like(x_t)
            variance_t = self._extract(self.posterior_variance, t, x_t.shape)
            x_t_minus_1 = mean + torch.sqrt(variance_t) * noise
            
            return x_t_minus_1
    
    def p_sample_loop(self, model, shape, condition_high, condition_low, device='cpu'):
        """
        Full reverse diffusion sampling
        
        Args:
            model: Denoising model
            shape: Shape of data to generate
            condition_high: High frequency conditioning
            condition_low: Low frequency conditioning
            device: Computation device
            
        Returns:
            Generated sample
        """
        batch_size = shape[0]
        
        # Start from pure noise
        x_t = torch.randn(shape, device=device)
        
        # Iteratively denoise
        for t in reversed(range(self.time_steps)):
            t_batch = torch.full((batch_size,), t, device=device)
            x_t = self.p_sample(model, x_t, t_batch, condition_high, condition_low)
            
        return x_t
    
    def compute_loss(self, model, x_0, timesteps, condition_high, condition_low, noise=None):
        """
        Compute diffusion training loss
        
        Args:
            model: Denoising model
            x_0: Clean data
            timesteps: Batch of timesteps for training
            condition_high: High frequency conditioning
            condition_low: Low frequency conditioning
            noise: Optional pre-generated noise
            
        Returns:
            Loss value
        """
        if noise is None:
            noise = torch.randn_like(x_0)
            
        # Apply forward diffusion
        x_noisy = self.q_sample(x_0, condition_low, timesteps, noise)
        
        # Get model prediction
        predicted_noise = model(x_noisy, timesteps, condition_high, condition_low)
        
        # Simple MSE loss
        loss = torch.nn.functional.mse_loss(predicted_noise, noise)
        
        return loss