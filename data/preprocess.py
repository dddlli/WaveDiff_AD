import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os


def getTimeEmbedding(time):
    """
    Create time embeddings from timestamps
    
    Args:
        time: Array of timestamps
        
    Returns:
        Time embeddings with cyclic features
    """
    df = pd.DataFrame(time, columns=['time'])
    df['time'] = pd.to_datetime(df['time'])

    # Cyclic encoding of time features, normalized to [-0.5, 0.5]
    df['minute'] = df['time'].apply(lambda row: row.minute / 59 - 0.5)
    df['hour'] = df['time'].apply(lambda row: row.hour / 23 - 0.5)
    df['weekday'] = df['time'].apply(lambda row: row.weekday() / 6 - 0.5)
    df['day'] = df['time'].apply(lambda row: row.day / 30 - 0.5)
    df['month'] = df['time'].apply(lambda row: row.month / 12 - 0.5)
    df['year_day'] = df['time'].apply(lambda row: row.dayofyear / 365 - 0.5)
    
    # Optional: Add sine/cosine encoding for better cyclic representation
    for col in ['minute', 'hour', 'weekday', 'day', 'month', 'year_day']:
        # Convert back to [0, 1] range for sine/cosine
        val = df[col] + 0.5
        df[f'{col}_sin'] = np.sin(2 * np.pi * val)
        df[f'{col}_cos'] = np.cos(2 * np.pi * val)

    # Extract time embeddings
    time_cols = ['minute', 'hour', 'weekday', 'day', 'month']
    return df[time_cols].values


def getStable(data, w=1440):
    """
    Extract stable (high frequency) and trend (low frequency) components
    
    Args:
        data: Input time series data
        w: Window size for trend extraction (default: 1440 - one day for minute data)
        
    Returns:
        Tuple of (trend-adjusted data, stable component)
    """
    # Use rolling median for robust trend extraction
    trend = pd.DataFrame(data).rolling(w, center=True).median().values
    
    # Fill NaN values at the edges
    trend_filled = trend.copy()
    for i in range(w//2):
        trend_filled[i] = trend[w//2]
        trend_filled[-(i+1)] = trend[-(w//2+1)]
    
    # Extract stable component (data - trend)
    stable = data - trend_filled
    
    # Return trimmed data if required
    if np.isnan(trend).any():
        return data[w//2:-w//2, :], stable[w//2:-w//2, :]
    else:
        return data, stable


def getData(path='./dataset/', dataset='SWaT', period=1440, train_rate=0.8, 
            normalization='standard', fill_method='forward'):
    """
    Load and preprocess time series data
    
    Args:
        path: Path to dataset directory
        dataset: Dataset name
        period: Approximate period of time series (for trend extraction)
        train_rate: Proportion of normal data to use for training
        normalization: Normalization method ('standard', 'minmax', or None)
        fill_method: Method for filling missing values ('forward', 'zero', or 'mean')
        
    Returns:
        Dictionary with preprocessed data splits
    """
    # Load data
    try:
        # Try to load using numpy files
        init_data = np.load(os.path.join(path, dataset, f'{dataset}_train_data.npy'))
        init_time = getTimeEmbedding(np.load(os.path.join(path, dataset, f'{dataset}_train_date.npy')))
        test_data = np.load(os.path.join(path, dataset, f'{dataset}_test_data.npy'))
        test_time = getTimeEmbedding(np.load(os.path.join(path, dataset, f'{dataset}_test_date.npy')))
        test_label = np.load(os.path.join(path, dataset, f'{dataset}_test_label.npy'))
    except FileNotFoundError:
        # Alternative: try to load using CSV files
        train_df = pd.read_csv(os.path.join(path, dataset, f'{dataset}_train.csv'))
        test_df = pd.read_csv(os.path.join(path, dataset, f'{dataset}_test.csv'))
        
        # Extract timestamps
        time_col = [col for col in train_df.columns if 'time' in col.lower() or 'date' in col.lower()][0]
        init_time = getTimeEmbedding(train_df[time_col].values)
        test_time = getTimeEmbedding(test_df[time_col].values)
        
        # Extract data and labels
        feature_cols = [col for col in train_df.columns if col != time_col and 'label' not in col.lower()]
        label_col = [col for col in test_df.columns if 'label' in col.lower()][0] if 'label' in ' '.join(test_df.columns.tolist()).lower() else None
        
        init_data = train_df[feature_cols].values
        test_data = test_df[feature_cols].values
        test_label = test_df[label_col].values if label_col else np.zeros(len(test_df))
        
    # Apply normalization
    if normalization == 'standard':
        scaler = StandardScaler()
        scaler.fit(init_data)
        init_data = pd.DataFrame(scaler.transform(init_data))
        test_data = pd.DataFrame(scaler.transform(test_data))
    elif normalization == 'minmax':
        scaler = MinMaxScaler()
        scaler.fit(init_data)
        init_data = pd.DataFrame(scaler.transform(init_data))
        test_data = pd.DataFrame(scaler.transform(test_data))
    else:
        init_data = pd.DataFrame(init_data)
        test_data = pd.DataFrame(test_data)
    
    # Fill missing values
    if fill_method == 'forward':
        init_data = init_data.fillna(method='ffill').fillna(method='bfill').values
        test_data = test_data.fillna(method='ffill').fillna(method='bfill').values
    elif fill_method == 'zero':
        init_data = init_data.fillna(0).values
        test_data = test_data.fillna(0).values
    elif fill_method == 'mean':
        init_data = init_data.fillna(init_data.mean()).values
        test_data = test_data.fillna(test_data.mean()).values
    
    # Get stable components
    init_data, init_stable = getStable(init_data, w=period)
    init_time = init_time[:len(init_data)]
    init_label = np.zeros((len(init_data), 1))
    
    # For test data, we don't want to use future information for trend extraction
    # So we use a causal filter or no filtering
    test_stable = np.zeros_like(test_data)  # Default: no decomposition for test data
    
    # Split normal data into train and validation
    train_size = int(train_rate * len(init_data))
    
    train_data = init_data[:train_size]
    train_time = init_time[:train_size]
    train_stable = init_stable[:train_size]
    train_label = init_label[:train_size]

    valid_data = init_data[train_size:]
    valid_time = init_time[train_size:]
    valid_stable = init_stable[train_size:]
    valid_label = init_label[train_size:]

    # Ensure test_label has the right shape
    if len(test_label.shape) == 1:
        test_label = test_label.reshape(-1, 1)

    # Prepare output dictionary
    data = {
        'train_data': train_data, 
        'train_time': train_time, 
        'train_stable': train_stable, 
        'train_label': train_label,
        'valid_data': valid_data, 
        'valid_time': valid_time, 
        'valid_stable': valid_stable, 
        'valid_label': valid_label,
        'init_data': init_data, 
        'init_time': init_time, 
        'init_stable': init_stable, 
        'init_label': init_label,
        'test_data': test_data, 
        'test_time': test_time, 
        'test_stable': test_stable, 
        'test_label': test_label
    }

    return data


def augment_time_series(data, time, stable, label, augmentation_methods=None):
    """
    Apply data augmentation to time series
    
    Args:
        data: Time series data
        time: Time features
        stable: Stable components
        label: Labels
        augmentation_methods: List of augmentation methods to apply
        
    Returns:
        Augmented data
    """
    if augmentation_methods is None:
        augmentation_methods = ['jitter', 'scaling', 'permutation']
    
    augmented_data = []
    augmented_time = []
    augmented_stable = []
    augmented_label = []
    
    # Original data
    augmented_data.append(data)
    augmented_time.append(time)
    augmented_stable.append(stable)
    augmented_label.append(label)
    
    # Apply augmentations
    for method in augmentation_methods:
        if method == 'jitter':
            # Add random noise
            noise_level = 0.01
            noise = np.random.normal(0, noise_level, data.shape)
            aug_data = data + noise
            aug_stable = stable + noise
            
            augmented_data.append(aug_data)
            augmented_time.append(time)
            augmented_stable.append(aug_stable)
            augmented_label.append(label)
            
        elif method == 'scaling':
            # Random scaling
            scaling_factor = np.random.uniform(0.8, 1.2)
            aug_data = data * scaling_factor
            aug_stable = stable * scaling_factor
            
            augmented_data.append(aug_data)
            augmented_time.append(time)
            augmented_stable.append(aug_stable)
            augmented_label.append(label)
            
        elif method == 'permutation':
            # Time-sliced permutation (for normal data only)
            if np.all(label == 0):
                window_size = min(100, len(data) // 10)
                num_segments = len(data) // window_size
                
                aug_data = data.copy()
                aug_stable = stable.copy()
                
                # Only permute a subset of segments
                num_permute = max(2, num_segments // 3)
                segments_to_permute = np.random.choice(num_segments, num_permute, replace=False)
                permuted_order = np.random.permutation(segments_to_permute)
                
                for i, j in enumerate(permuted_order):
                    orig_idx = segments_to_permute[i]
                    new_idx = segments_to_permute[j]
                    
                    start_i, end_i = orig_idx * window_size, (orig_idx + 1) * window_size
                    start_j, end_j = new_idx * window_size, (new_idx + 1) * window_size
                    
                    # Swap segments
                    aug_data[start_i:end_i], aug_data[start_j:end_j] = \
                        aug_data[start_j:end_j].copy(), aug_data[start_i:end_i].copy()
                    
                    aug_stable[start_i:end_i], aug_stable[start_j:end_j] = \
                        aug_stable[start_j:end_j].copy(), aug_stable[start_i:end_i].copy()
                
                augmented_data.append(aug_data)
                augmented_time.append(time)
                augmented_stable.append(aug_stable)
                augmented_label.append(label)
    
    # Concatenate all augmentations
    return (
        np.concatenate(augmented_data, axis=0),
        np.concatenate(augmented_time, axis=0),
        np.concatenate(augmented_stable, axis=0),
        np.concatenate(augmented_label, axis=0)
    )