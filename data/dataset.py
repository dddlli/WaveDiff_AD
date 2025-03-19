import torch
from torch.utils.data import Dataset as TorchDataset

class Dataset(TorchDataset):
    """
    Dataset class for time series data
    
    Args:
        data: Time series data
        time: Time features
        stable: Stable components (for supervised training)
        label: Anomaly labels
        window_size: Size of sliding window
    """
    def __init__(self, data, time, stable, label, window_size):
        self.data = data
        self.time = time
        self.stable = stable
        self.label = label
        self.window_size = window_size

    def __getitem__(self, index):
        """
        Get windowed time series sample
        
        Args:
            index: Sample index
            
        Returns:
            Tuple of (data, time, stable, label) for the window
        """
        data = self.data[index: index + self.window_size, :]
        time = self.time[index: index + self.window_size, :]
        stable = self.stable[index: index + self.window_size, :]
        label = self.label[index: index + self.window_size, :]

        return data, time, stable, label

    def __len__(self):
        """
        Get dataset length
        
        Returns:
            Number of possible windows
        """
        return len(self.data) - self.window_size + 1


class WaveletHHTDataset(TorchDataset):
    """
    Extended dataset class with preprocessing capabilities
    
    Args:
        data: Time series data
        time: Time features
        label: Anomaly labels
        window_size: Size of sliding window
        stride: Window stride for overlap control
        wavelet_transform: Wavelet transform function (optional)
        hht_transform: HHT transform function (optional)
    """
    def __init__(self, data, time, label, window_size, stride=1, 
                 wavelet_transform=None, hht_transform=None):
        self.data = data
        self.time = time
        self.label = label
        self.window_size = window_size
        self.stride = stride
        self.wavelet_transform = wavelet_transform
        self.hht_transform = hht_transform
        
        # Calculate length
        self.valid_indices = list(range(0, len(data) - window_size + 1, stride))

    def __getitem__(self, index):
        """
        Get windowed time series sample with optional decomposition
        
        Args:
            index: Sample index
            
        Returns:
            Dictionary with data and decomposition results
        """
        idx = self.valid_indices[index]
        data = self.data[idx: idx + self.window_size, :]
        time = self.time[idx: idx + self.window_size, :]
        label = self.label[idx: idx + self.window_size, :]
        
        # Create sample dictionary
        sample = {
            'data': data,
            'time': time,
            'label': label
        }
        
        # Apply wavelet transform if available
        if self.wavelet_transform is not None:
            sample['wavelet'] = self.wavelet_transform(data)
            
        # Apply HHT if available
        if self.hht_transform is not None:
            sample['hht'] = self.hht_transform(data)
            
        return sample

    def __len__(self):
        """
        Get dataset length
        
        Returns:
            Number of valid windows
        """
        return len(self.valid_indices)