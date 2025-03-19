# Adaptive Time Series Anomaly Detection

An advanced unsupervised anomaly detection framework for multivariate time series that combines adaptive signal decomposition techniques with diffusion models.

## Overview

This framework implements a novel approach to time series anomaly detection that leverages the complementary strengths of wavelet transforms and Hilbert-Huang Transform (HHT). It employs a mixture of experts architecture where high-frequency and low-frequency components are processed by specialized models, guided by a time-aware routing mechanism.

Key features:
- **Adaptive signal decomposition**: Combines wavelet transforms and HHT for optimal handling of both periodic and non-stationary/non-linear signals
- **Frequency-specialized processing**: Different expert models handle high and low frequency components
- **Time-aware routing**: Dynamic selection of appropriate experts based on signal characteristics
- **Diffusion-based reconstruction**: Uses diffusion models for robust signal reconstruction
- **Unsupervised learning**: No need for labeled anomaly data during training

## Architecture

The system architecture consists of several key components:

1. **Adaptive Decomposition Module**: Dynamically decomposes time series into high and low frequency components, using either wavelet transforms, HHT, or a learned combination.

2. **Mixture of Experts**: Specialized models process different frequency components, with a router that determines which experts to use for each input.

3. **Diffusion Model**: Learns to denoise and reconstruct signals through a progressive denoising process, enabling anomaly detection based on reconstruction error.

4. **Time-Frequency Attention**: Integrated attention mechanisms that facilitate information exchange between temporal and frequency domains.

## Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.12+
- CUDA (optional, but recommended for faster training)

### Setup
1. Clone the repository:
```bash
git clone https://github.com/yourusername/adaptive-ts-anomaly-detection.git
cd adaptive-ts-anomaly-detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Configuration
Modify the parameters in `config/config.py` to suit your dataset and training preferences:

```python
# Basic configuration
parser.add_argument('--dataset', type=str, default='SMD', help='dataset name')
parser.add_argument('--data_dir', type=str, default='./dataset/', help='path of the data')
parser.add_argument('--model_dir', type=str, default='./checkpoint/', help='path of the checkpoint')
parser.add_argument('--output_dir', type=str, default='./output/', help='path of the output results')

# Training parameters 
parser.add_argument('--epochs', type=int, default=10, help='epoch of training')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')

# Decomposition parameters
parser.add_argument('--decomp_mode', type=str, default='adaptive', 
                    choices=['wavelet', 'hht', 'adaptive'], help='decomposition mode')
```

### Data Preparation
Prepare your dataset in the following format:
- `{dataset_name}_train_data.npy`: Training data array (n_samples, n_features)
- `{dataset_name}_train_date.npy`: Timestamps for training data
- `{dataset_name}_test_data.npy`: Test data array
- `{dataset_name}_test_date.npy`: Timestamps for test data
- `{dataset_name}_test_label.npy`: Binary anomaly labels for test data

Place these files in the `dataset/{dataset_name}/` directory.

### Training and Evaluation
Run the main script to train and evaluate the model:

```bash
python main.py --dataset SMD --epochs 10 --batch_size 32 --decomp_mode adaptive
```

For multiple iterations with different random seeds:
```bash
python main.py --dataset SMD --itr 5 --random_seed 42
```

## Key Components

### Wavelet Transform Module
The wavelet transform module provides efficient decomposition for periodic components and offers multi-resolution analysis of time series data.

```python
# Wavelet decomposition with learnable filters
wavelet_components = wavelet_module.forward(time_series)
high_freq = wavelet_components['high_freq']
low_freq = wavelet_components['low_freq']
```

### Hilbert-Huang Transform (HHT) Module
The HHT module handles non-stationary and non-linear data through Empirical Mode Decomposition (EMD) and Hilbert spectral analysis.

```python
# Adaptive HHT decomposition
hht_components = hht_module.forward(time_series)
spectral_features = hht_components['spectral_features']
```

### Mixture of Experts
The MoE architecture dynamically routes inputs to specialized expert models:

```python
# Route based on frequency characteristics
routing_weights = router(time_series, timestep)
# Process with appropriate experts
high_freq_output = high_freq_expert(high_freq_component)
low_freq_output = low_freq_expert(low_freq_component)
```

### Diffusion Model
The diffusion model enables high-quality signal reconstruction:

```python
# Forward diffusion (add noise)
noisy_signal = diffusion.q_sample(original_signal, timestep)
# Reverse diffusion (denoise)
reconstructed_signal = diffusion.p_sample_loop(noise_predictor, noise_shape, condition)
```

## Output and Visualization

The system generates comprehensive visualizations and metrics:

1. **Anomaly Detection Results**: Time series plots with highlighted anomalies
2. **Decomposition Visualization**: Original signal with separated frequency components
3. **Reconstruction Analysis**: Original vs. reconstructed signals with error plots
4. **Evaluation Metrics**: Precision, recall, F1-score, and more
5. **Routing Analysis**: Visualizations of how different experts are utilized

All visualizations and metrics are saved to the specified output directory.

## Evaluation Metrics

The system provides multiple evaluation approaches:
- **Point-Adjustment Metrics**: Accounts for slight timing discrepancies in anomaly detection
- **Range-Based Metrics**: Evaluates detection performance on anomaly segments
- **Standard Metrics**: Precision, recall, F1-score, AUC, and AUPR

## Citation

If you use this framework in your research, please cite:

```
@article{adaptive_ts_anomaly_detection,
  title={Adaptive Time Series Anomaly Detection Through Signal Decomposition and Diffusion Models},
  author={Your Name},
  journal={arXiv preprint},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This work is inspired by and builds upon several research papers:
- "Drift doesn't Matter: Dynamic Decomposition with Diffusion Reconstruction for Unstable Multivariate Time Series Anomaly Detection"
- "Denoising Diffusion Probabilistic Models" by Ho et al.
- "The Empirical Mode Decomposition and the Hilbert spectrum for nonlinear and non-stationary time series analysis" by Huang et al.