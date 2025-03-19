"""
Configuration parameters for time series anomaly detection with
adaptive decomposition-enhanced diffusion models.
"""

# Dataset-specific model parameters
MODEL_PARAMS = {
    'SMD': {
        'input_c': 38,
        'output_c': 38,
        'd_model': 256,
        'n_heads': 8,
        'd_ff': 512,
        'num_layers': 4,
        'dropout': 0.1
    },
    'SMAP': {
        'input_c': 25,
        'output_c': 25,
        'd_model': 256,
        'n_heads': 8,
        'd_ff': 512,
        'num_layers': 4,
        'dropout': 0.1
    },
    'MSL': {
        'input_c': 55,
        'output_c': 55,
        'd_model': 256,
        'n_heads': 8,
        'd_ff': 512,
        'num_layers': 4,
        'dropout': 0.1
    },
    'PSM': {
        'input_c': 25,
        'output_c': 25,
        'd_model': 256,
        'n_heads': 8,
        'd_ff': 512,
        'num_layers': 4,
        'dropout': 0.1
    },
    'SWaT': {
        'input_c': 51,
        'output_c': 51,
        'd_model': 256,
        'n_heads': 8,
        'd_ff': 512,
        'num_layers': 4,
        'dropout': 0.1
    }
}

# Default diffusion parameters
DIFFUSION_PARAMS = {
    'T': 500,
    'beta_0': 0.0001,
    'beta_T': 0.05
}

# Default decomposition parameters
DECOMPOSITION_PARAMS = {
    'wavelet_type': 'db4',   # Type of wavelet to use
    'wavelet_level': 3,      # Decomposition level for wavelet transform
    'hht_imfs': 3            # Number of IMFs to extract in HHT
}

# Default anomaly detection parameters
ANOMALY_PARAMS = {
    'anomaly_ratio': {
        'SMD': 0.5,
        'SMAP': 1.0,
        'MSL': 1.0,
        'PSM': 1.0,
        'SWaT': 0.5
    },
    'high_freq_weight': 0.6,  # Weight for high frequency error in anomaly score
    'low_freq_weight': 0.4    # Weight for low frequency error in anomaly score
}

# Default training parameters
TRAINING_PARAMS = {
    'lr': 1e-4,
    'batch_size': 256,
    'num_epochs': 100,
    'win_size': 100,
    'masking': 'rm',  # 'rm', 'mnr', or 'bm'
    'masking_k': 10,
    'only_generate_missing': 0  # Whether to only generate missing values during denoising
}

def get_config(dataset):
    """
    Get default configuration for a specific dataset.
    
    Args:
        dataset: Dataset name ('SMD', 'SMAP', 'MSL', 'PSM', 'SWaT')
        
    Returns:
        Dictionary with configuration parameters
    """
    if dataset not in MODEL_PARAMS:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    config = {
        **MODEL_PARAMS[dataset],
        **DIFFUSION_PARAMS,
        **DECOMPOSITION_PARAMS,
        'anomaly_ratio': ANOMALY_PARAMS['anomaly_ratio'][dataset],
        'high_freq_weight': ANOMALY_PARAMS['high_freq_weight'],
        'low_freq_weight': ANOMALY_PARAMS['low_freq_weight'],
        **TRAINING_PARAMS
    }
    
    return config