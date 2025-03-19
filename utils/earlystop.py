import numpy as np
import torch

class EarlyStop:
    """
    Early stopping for training neural networks
    
    Args:
        patience: Number of epochs to wait before stopping
        delta: Minimum change in validation loss to be considered improvement
        path: Path to save the best model
    """
    def __init__(self, patience=7, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        
    def __call__(self, val_loss, model):
        """
        Check if training should stop
        
        Args:
            val_loss: Current validation loss
            model: Model to save if improvement
        """
        score = -val_loss  # Higher score is better

        if self.best_score is None:
            # First epoch
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            # Not improved enough
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # Improved
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """
        Save model when validation loss decreases
        
        Args:
            val_loss: Current validation loss
            model: Model to save
        """
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss