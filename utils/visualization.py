import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, auc

# Set style
sns.set_palette("deep")


def plot_decomposition(signal, decomp_result, title=None, figsize=(15, 10)):
    """
    Plot original signal and its decomposed components
    
    Args:
        signal: Original signal [channels, time_steps]
        decomp_result: Dictionary with decomposition results
        title: Plot title
        figsize: Figure size
    """
    # Convert tensors to numpy arrays
    if isinstance(signal, torch.Tensor):
        signal = signal.detach().cpu().numpy()
    
    high_freq = decomp_result['high_freq']
    if isinstance(high_freq, torch.Tensor):
        high_freq = high_freq.detach().cpu().numpy()
    
    low_freq = decomp_result['low_freq']
    if isinstance(low_freq, torch.Tensor):
        low_freq = low_freq.detach().cpu().numpy()
    
    # Ensure shapes are correct
    if len(signal.shape) > 2:
        signal = signal.squeeze(0)
    if len(high_freq.shape) > 2:
        high_freq = high_freq.squeeze(0)
    if len(low_freq.shape) > 2:
        low_freq = low_freq.squeeze(0)
    
    # Determine number of channels
    if len(signal.shape) == 1:
        channels = 1
        signal = signal.reshape(1, -1)
        high_freq = high_freq.reshape(1, -1)
        low_freq = low_freq.reshape(1, -1)
    else:
        channels = signal.shape[0]
    
    # Create figure
    fig, axes = plt.subplots(channels, 3, figsize=figsize, sharex=True)
    
    # Add title
    if title:
        fig.suptitle(title, fontsize=16)
    
    # Adjust for single channel case
    if channels == 1:
        axes = axes.reshape(1, -1)
    
    # Plot each channel
    for i in range(channels):
        # Original signal
        axes[i, 0].plot(signal[i], label='Original')
        axes[i, 0].set_title(f'Original Signal (Channel {i+1})')
        
        # High frequency component
        axes[i, 1].plot(high_freq[i], color='orangered', label='High Freq')
        axes[i, 1].set_title(f'High Frequency Component (Channel {i+1})')
        
        # Low frequency component
        axes[i, 2].plot(low_freq[i], color='forestgreen', label='Low Freq')
        axes[i, 2].set_title(f'Low Frequency Component (Channel {i+1})')
        
        # Add legends
        for j in range(3):
            axes[i, j].legend()
            axes[i, j].set_xlabel('Time')
            axes[i, j].set_ylabel('Amplitude')
    
    plt.tight_layout()
    if title:
        plt.subplots_adjust(top=0.9)
    
    return fig


def plot_signal_reconstruction(original, reconstructed, errors=None, anomaly_scores=None, 
                               threshold=None, labels=None, figsize=(15, 10)):
    """
    Plot original signal, reconstruction, errors, and anomaly scores
    
    Args:
        original: Original signal [channels, time_steps]
        reconstructed: Reconstructed signal [channels, time_steps]
        errors: Reconstruction errors [channels, time_steps]
        anomaly_scores: Anomaly scores [time_steps]
        threshold: Anomaly threshold
        labels: True anomaly labels [time_steps]
        figsize: Figure size
    """
    # Convert tensors to numpy arrays
    if isinstance(original, torch.Tensor):
        original = original.detach().cpu().numpy()
    if isinstance(reconstructed, torch.Tensor):
        reconstructed = reconstructed.detach().cpu().numpy()
    if isinstance(errors, torch.Tensor):
        errors = errors.detach().cpu().numpy()
    if isinstance(anomaly_scores, torch.Tensor):
        anomaly_scores = anomaly_scores.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    
    # Ensure shapes are correct
    if len(original.shape) > 2:
        original = original.squeeze(0)
    if len(reconstructed.shape) > 2:
        reconstructed = reconstructed.squeeze(0)
    
    # Determine number of channels
    if len(original.shape) == 1:
        channels = 1
        original = original.reshape(1, -1)
        reconstructed = reconstructed.reshape(1, -1)
        if errors is not None:
            if len(errors.shape) == 1:
                errors = errors.reshape(1, -1)
    else:
        channels = original.shape[0]
    
    # Determine number of plots
    if anomaly_scores is not None:
        n_plots = 3
    elif errors is not None:
        n_plots = 2
    else:
        n_plots = 1
    
    # Create figure
    fig, axes = plt.subplots(channels + (1 if anomaly_scores is not None else 0), n_plots, 
                         figsize=figsize, sharex=True)
    
    # Adjust for single channel case
    if channels == 1 and anomaly_scores is None:
        axes = axes.reshape(1, -1)
    
    # Plot each channel
    for i in range(channels):
        # Original vs reconstructed
        axes[i, 0].plot(original[i], label='Original')
        axes[i, 0].plot(reconstructed[i], label='Reconstructed', alpha=0.7)
        axes[i, 0].set_title(f'Original vs Reconstructed (Channel {i+1})')
        axes[i, 0].legend()
        
        # Plot errors if available
        if errors is not None:
            axes[i, 1].plot(errors[i], color='coral', label='Error')
            axes[i, 1].set_title(f'Reconstruction Error (Channel {i+1})')
            axes[i, 1].legend()
    
    # Plot anomaly scores if available
    if anomaly_scores is not None:
        ax = axes[-1, -1] if channels > 1 else axes[-1]
        ax.plot(anomaly_scores, color='red', label='Anomaly Score')
        
        if threshold is not None:
            ax.axhline(y=threshold, color='black', linestyle='--', label='Threshold')
        
        if labels is not None:
            # Highlight anomaly regions
            anomaly_idx = np.where(labels == 1)[0]
            if len(anomaly_idx) > 0:
                # Find continuous segments
                diff = np.diff(anomaly_idx)
                seg_idx = np.where(diff > 1)[0] + 1
                segs = np.split(anomaly_idx, seg_idx)
                
                # Highlight each segment
                for seg in segs:
                    if len(seg) > 0:
                        ax.axvspan(seg[0], seg[-1], alpha=0.2, color='red')
        
        ax.set_title('Anomaly Scores')
        ax.legend()
    
    plt.tight_layout()
    
    return fig


def plot_performance_curves(labels, scores, figsize=(15, 5)):
    """
    Plot ROC and Precision-Recall curves for anomaly detection
    
    Args:
        labels: True anomaly labels
        scores: Anomaly scores
        figsize: Figure size
    """
    # Ensure inputs are numpy arrays
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    if isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().numpy()
    
    # Flatten arrays
    labels = labels.flatten()
    scores = scores.flatten()
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # ROC curve
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    
    ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('Receiver Operating Characteristic (ROC)')
    ax1.legend(loc="lower right")
    
    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(labels, scores)
    pr_auc = auc(recall, precision)
    
    ax2.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.legend(loc="lower left")
    
    plt.tight_layout()
    
    return fig


def plot_diffusion_process(original, diffusion_steps, figsize=(15, 10)):
    """
    Plot diffusion process steps
    
    Args:
        original: Original signal [channels, time_steps]
        diffusion_steps: List of diffusion steps [steps, channels, time_steps]
        figsize: Figure size
    """
    # Convert tensors to numpy arrays
    if isinstance(original, torch.Tensor):
        original = original.detach().cpu().numpy()
    
    # Convert diffusion steps to numpy
    np_diffusion_steps = []
    for step in diffusion_steps:
        if isinstance(step, torch.Tensor):
            np_diffusion_steps.append(step.detach().cpu().numpy())
        else:
            np_diffusion_steps.append(step)
    
    # Ensure shapes are correct
    if len(original.shape) > 2:
        original = original.squeeze(0)
    
    # Determine number of channels
    if len(original.shape) == 1:
        channels = 1
        original = original.reshape(1, -1)
    else:
        channels = original.shape[0]
    
    # Select a subset of steps to display
    num_steps = len(np_diffusion_steps)
    step_indices = np.linspace(0, num_steps-1, min(num_steps, 5)).astype(int)
    selected_steps = [np_diffusion_steps[i] for i in step_indices]
    
    # Create figure
    fig, axes = plt.subplots(channels, len(selected_steps) + 1, figsize=figsize, sharex=True)
    
    # Adjust for single channel case
    if channels == 1:
        axes = axes.reshape(1, -1)
    
    # Plot each channel
    for i in range(channels):
        # Original signal
        axes[i, 0].plot(original[i], label='Original')
        axes[i, 0].set_title(f'Original (Channel {i+1})')
        
        # Diffusion steps
        for j, (step_idx, step) in enumerate(zip(step_indices, selected_steps)):
            step_data = step.squeeze(0)[i] if len(step.shape) > 2 else step[i]
            axes[i, j+1].plot(step_data, label=f't={step_idx}')
            axes[i, j+1].set_title(f'Step {step_idx} (Channel {i+1})')
    
    plt.tight_layout()
    
    return fig