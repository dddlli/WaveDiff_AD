import numpy as np
from math import floor, log


class SPOT:
    """
    This class implements the SPOT algorithm for anomaly detection.
    
    SPOT (Streaming Peaks Over Threshold) provides a fully automatic and parameter-free
    method for detecting anomalies in streaming data.
    """
    def __init__(self, q=1e-4):
        """
        Initialize SPOT with the risk level
        
        Args:
            q: Detection level (risk of false alarm)
        """
        self.q = q
        self.n = 0
        self.extreme_quantile = None
        self.data = None
        self.init_data = None
        self.peaks = None
        self.thresholds = None
        self.peaks_threshold = None
    
    def fit(self, init_data, data):
        """
        Fit the model to data
        
        Args:
            init_data: Initial batch of data for threshold initialization
            data: Data for anomaly detection
        """
        self.init_data = init_data
        self.data = data
        self.n = len(data)
        self.thresholds = np.zeros(self.n)
    
    def initialize(self, level=0.98, min_extremes=10, verbose=True):
        """
        Initialize the algorithm by setting initial threshold
        
        Args:
            level: Probability for the initial threshold  
            min_extremes: Minimum number of extremes required
            verbose: Whether to print information
        """
        if self.init_data is None:
            raise ValueError("SPOT is not fitted yet, please call fit() first")
            
        n_init = len(self.init_data)
        
        # Compute sample quantile
        init_threshold = np.quantile(self.init_data, level)
        
        # Extract peaks (excesses over threshold)
        peaks = self.init_data[self.init_data > init_threshold] - init_threshold
        
        # Ensure we have enough peaks
        if len(peaks) < min_extremes:
            if verbose:
                print(f"Not enough extremes found with level {level}. Adjusting...")
            
            # Try to find a suitable threshold with enough peaks
            sort_data = np.sort(self.init_data)
            valid_indices = max(min_extremes, int(0.02 * n_init))
            init_threshold = sort_data[min(n_init - valid_indices - 1, floor(0.98 * n_init))]
            peaks = self.init_data[self.init_data > init_threshold] - init_threshold
        
        self.peaks = peaks
        self.peaks_threshold = init_threshold
        
        # Fit generalized Pareto distribution (approximation)
        gamma = 0.1  # Shape parameter approximation
        sigma = np.mean(peaks)  # Scale parameter approximation
        
        # Compute the extreme quantile
        self.extreme_quantile = init_threshold + (sigma / gamma) * ((pow(self.q, -gamma) - 1))
        
        if verbose:
            print(f"Extreme quantile: {self.extreme_quantile}")
            print(f"Peaks threshold: {self.peaks_threshold}")
            print(f"Number of peaks: {len(self.peaks)}")
        
    def run(self, with_alarm=True):
        """
        Run the SPOT algorithm on the data
        
        Args:
            with_alarm: Whether to compute alarms
            
        Returns:
            Dictionary with results
        """
        if self.extreme_quantile is None:
            raise ValueError("SPOT is not initialized yet, please call initialize() first")
            
        # Apply constant threshold
        alarms = np.zeros(self.n)
        for i in range(self.n):
            self.thresholds[i] = self.extreme_quantile
            
            if with_alarm and self.data[i] > self.extreme_quantile:
                alarms[i] = 1
            
        return {
            'thresholds': self.thresholds,
            'alarms': alarms
        }