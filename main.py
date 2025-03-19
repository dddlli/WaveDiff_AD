import argparse
import torch
import numpy as np
import os
from solver import Solver
from utils.seed import setSeed

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Wavelet-HHT Diffusion Model for Time Series Anomaly Detection')

    # Dataset settings
    parser.add_argument('--dataset', type=str, default='SWaT', help='Dataset name')
    parser.add_argument('--data_dir', type=str, default='./dataset/', help='Path of the data')
    parser.add_argument('--model_dir', type=str, default='./checkpoint/', help='Path for saving model checkpoints')

    # Training parameters
    parser.add_argument('--itr', type=int, default=5, help='Number of evaluation iterations')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--patience', type=int, default=3, help='Patience for early stopping')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')

    # Data processing parameters
    parser.add_argument('--period', type=int, default=1440, help='Approximate period of time series')
    parser.add_argument('--train_rate', type=float, default=0.8, help='Proportion of data for training')
    parser.add_argument('--window_size', type=int, default=64, help='Size of sliding window')

    # Model architecture
    parser.add_argument('--model_dim', type=int, default=512, help='Dimension of model hidden layers')
    parser.add_argument('--ff_dim', type=int, default=2048, help='Dimension of feedforward network')
    parser.add_argument('--atten_dim', type=int, default=64, help='Dimension of attention')
    parser.add_argument('--block_num', type=int, default=2, help='Number of transformer blocks')
    parser.add_argument('--head_num', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate')

    # Diffusion parameters
    parser.add_argument('--time_steps', type=int, default=1000, help='Number of diffusion time steps')
    parser.add_argument('--beta_start', type=float, default=0.0001, help='Start value for beta schedule')
    parser.add_argument('--beta_end', type=float, default=0.02, help='End value for beta schedule')
    parser.add_argument('--t', type=int, default=500, help='Time step for adding noise')
    parser.add_argument('--p', type=float, default=10.00, help='Peak value of trend disturbance')

    # Decomposition parameters
    parser.add_argument('--wavelet_level', type=int, default=3, help='Wavelet decomposition level')
    parser.add_argument('--hht_imfs', type=int, default=5, help='Number of IMFs for HHT')
    parser.add_argument('--fusion_type', type=str, default='weighted', help='Fusion type: weighted, attention, or tensor')
    parser.add_argument('--d', type=int, default=30, help='Shift of period for decomposition')

    # Evaluation parameters
    parser.add_argument('--q', type=float, default=0.01, help='Initial anomaly probability for SPOT')

    # System parameters
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
    parser.add_argument('--use_amp', action='store_true', help='Use Automatic Mixed Precision')

    # Parse arguments
    args = parser.parse_args()
    config = vars(args)
    
    # Set random seed and GPU
    setSeed(config['random_seed'])
    if torch.cuda.is_available():
        torch.cuda.set_device(config['gpu_id'])
    
    # Print configuration
    print('='*40 + ' CONFIGURATION ' + '='*40)
    for k, v in config.items():
        print(f'{k:20s}: {v}')
    print('='*90)

    # Create experiment and run
    results = []
    for ii in range(config['itr']):
        print(f'\nIteration {ii+1}/{config["itr"]}')
        exp = Solver(config)
        
        print('Training model...')
        exp.train()
        
        print('Testing model...')
        metrics = exp.test()
        results.append(metrics)
    
    # Average results across iterations
    if config['itr'] > 1:
        f1_scores = [res['f1_score'] for res in results]
        precision_scores = [res['precision'] for res in results]
        recall_scores = [res['recall'] for res in results]
        
        print('\n' + '='*40 + ' AVERAGE RESULTS ' + '='*40)
        print(f'Average F1 Score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}')
        print(f'Average Precision: {np.mean(precision_scores):.4f} ± {np.std(precision_scores):.4f}')
        print(f'Average Recall: {np.mean(recall_scores):.4f} ± {np.std(recall_scores):.4f}')
        print('='*90)