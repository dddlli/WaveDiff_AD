import os
from time import time

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import Dataset
from data.preprocess import getData
from model.detector import AnomalyDetection
from utils.earlystop import EarlyStop
from utils.evaluate import evaluate

class Solver:
    """
    Experiment class for training and evaluation of WHDiff model
    """
    def __init__(self, config):
        self.__dict__.update(config)
        self._get_data()
        self._get_model()

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

    def _get_data(self):
        """
        Prepare datasets and dataloaders
        """
        data = getData(
            path=self.data_dir,
            dataset=self.dataset,
            period=self.period,
            train_rate=self.train_rate
        )

        self.feature_num = data['train_data'].shape[1]
        self.time_num = data['train_time'].shape[1]
        print('\nData shape: ')
        for k, v in data.items():
            print(k, ': ', v.shape)

        self.train_set = Dataset(
            data=data['train_data'],
            time=data['train_time'],
            stable=data['train_stable'],
            label=data['train_label'],
            window_size=self.window_size
        )
        self.valid_set = Dataset(
            data=data['valid_data'],
            time=data['valid_time'],
            stable=data['valid_stable'],
            label=data['valid_label'],
            window_size=self.window_size
        )
        self.init_set = Dataset(
            data=data['init_data'],
            time=data['init_time'],
            stable=data['init_stable'],
            label=data['init_label'],
            window_size=self.window_size
        )
        self.test_set = Dataset(
            data=data['test_data'],
            time=data['test_time'],
            stable=data['test_stable'],
            label=data['test_label'],
            window_size=self.window_size
        )

        self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, drop_last=False)
        self.valid_loader = DataLoader(self.valid_set, batch_size=self.batch_size, shuffle=False, drop_last=False)
        self.init_loader = DataLoader(self.init_set, batch_size=self.batch_size, shuffle=False, drop_last=False)
        self.test_loader = DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, drop_last=False)

    def _get_model(self):
        """
        Initialize model, optimizer and early stopping
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('\nDevice:', self.device)

        self.model = AnomalyDetection(
            time_steps=self.time_steps,
            beta_start=self.beta_start,
            beta_end=self.beta_end,
            window_size=self.window_size,
            model_dim=self.model_dim,
            ff_dim=self.ff_dim,
            atten_dim=self.atten_dim,
            feature_num=self.feature_num,
            time_num=self.time_num,
            block_num=self.block_num,
            head_num=self.head_num,
            dropout=self.dropout,
            device=self.device,
            wavelet_level=self.wavelet_level,
            hht_imfs=self.hht_imfs,
            fusion_type=self.fusion_type,
            t=self.t
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        self.early_stopping = EarlyStop(patience=self.patience, path=self.model_dir + self.dataset + '_model.pkl')
        self.criterion = nn.MSELoss(reduction='mean')

    def _process_one_batch(self, batch_data, batch_time, batch_stable, train=True):
        """
        Process one batch of data
        
        Args:
            batch_data: Input data
            batch_time: Time features
            batch_stable: Stable components (for supervised loss)
            train: Whether in training mode
            
        Returns:
            Loss if training, or components if evaluation
        """
        batch_data = batch_data.float().to(self.device)
        batch_time = batch_time.float().to(self.device)
        batch_stable = batch_stable.float().to(self.device)

        if train:
            high_freq, low_freq, recon, losses = self.model(batch_data, batch_time, self.p)
            return losses['total']
        else:
            high_freq, low_freq, recon, _ = self.model(batch_data, batch_time, 0.00)
            return high_freq, low_freq, recon

    def train(self):
        """
        Train the model
        """
        for e in range(self.epochs):
            start = time()

            self.model.train()
            train_loss = []
            for (batch_data, batch_time, batch_stable, _) in tqdm(self.train_loader):
                self.optimizer.zero_grad()
                loss = self._process_one_batch(batch_data, batch_time, batch_stable, train=True)
                train_loss.append(loss.item())
                loss.backward()
                self.optimizer.step()

            with torch.no_grad():
                self.model.eval()
                valid_loss = []
                for (batch_data, batch_time, batch_stable, _) in tqdm(self.valid_loader):
                    loss = self._process_one_batch(batch_data, batch_time, batch_stable, train=True)
                    valid_loss.append(loss.item())

            train_loss, valid_loss = np.average(train_loss), np.average(valid_loss)
            end = time()
            print(f'Epoch: {e} || Train Loss: {train_loss:.6f} Valid Loss: {valid_loss:.6f} || Cost: {end - start:.4f}')

            self.early_stopping(valid_loss, self.model)
            if self.early_stopping.early_stop:
                print(f'Early stopping at epoch {e}')
                break

        # Load the best model
        self.model.load_state_dict(torch.load(self.model_dir + self.dataset + '_model.pkl'))

    def test(self):
        """
        Evaluate the model on test data
        """
        # Load the best model
        self.model.load_state_dict(torch.load(self.model_dir + self.dataset + '_model.pkl'))

        with torch.no_grad():
            self.model.eval()
            
            # Collect initialization data for threshold setting
            init_src, init_rec = [], []
            for (batch_data, batch_time, batch_stable, batch_label) in tqdm(self.init_loader):
                _, _, recon = self._process_one_batch(batch_data, batch_time, batch_stable, train=False)
                init_src.append(batch_data.detach().cpu().numpy()[:, -1, :])
                init_rec.append(recon.detach().cpu().numpy()[:, -1, :])

            # Collect test data
            test_label, test_src, test_rec = [], [], []
            for (batch_data, batch_time, batch_stable, batch_label) in tqdm(self.test_loader):
                _, _, recon = self._process_one_batch(batch_data, batch_time, batch_stable, train=False)
                test_label.append(batch_label.detach().cpu().numpy()[:, -1, :])
                test_src.append(batch_data.detach().cpu().numpy()[:, -1, :])
                test_rec.append(recon.detach().cpu().numpy()[:, -1, :])

        # Concatenate results
        init_src = np.concatenate(init_src, axis=0)
        init_rec = np.concatenate(init_rec, axis=0)
        init_mse = (init_src - init_rec) ** 2

        test_label = np.concatenate(test_label, axis=0)
        test_src = np.concatenate(test_src, axis=0)
        test_rec = np.concatenate(test_rec, axis=0)
        test_mse = (test_src - test_rec) ** 2

        # Calculate anomaly scores
        init_score = np.mean(init_mse, axis=-1, keepdims=True)
        test_score = np.mean(test_mse, axis=-1, keepdims=True)

        # Evaluate with SPOT algorithm
        res = evaluate(init_score.reshape(-1), test_score.reshape(-1), test_label.reshape(-1), q=self.q)
        
        print("\n=============== " + self.dataset + " ===============")
        print(f"Precision: {res['precision']:.4f} || Recall: {res['recall']:.4f} || F1: {res['f1_score']:.4f}")
        print("=============== " + self.dataset + " ===============\n")
        
        return res