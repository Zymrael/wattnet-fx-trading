import natsort
import torch
import pandas as pd
from .pandas_utils  import mu_law_encode
from ..utils import one_hot


class FXDataHandler:
    def __init__(self,
                 fpath=None):
        if fpath:
            self.data = pd.read_pickle(f'{fpath}')
        self._is_split = False
        self._is_quantized = False
        self._is_tensor = False

    def assign_data(self, data):
        self.data = data

    def split_train_val_test(self, percentage_split):
        assert sum(percentage_split) == 100, f'Percentage splits sum {sum(percentage_split)} not equal to 100'
        split_d = []
        run_idx = 0
        for p in percentage_split:
            idx = p * len(self.data) // 100
            split_d.append(self.data.iloc[run_idx: run_idx + idx])
            run_idx += idx
        self.train = split_d[0]
        self.val = split_d[1]
        self.test = split_d[2]
        self._is_split = True

    def normalize(self, window=30):
        assert not self._is_split and not self._is_tensor, 'normalize before splitting and tensor transforms'
        if not window:
            self.data = (self.data - self.data.mean()) / self.data.std()
        else:
            self.data = (self.data - self.data.rolling(window, min_periods=1).mean()) / \
                        self.data.rolling(window, min_periods=1).std()
        self.data = self.data.dropna()

    def to_percentage_change(self, multiplier=1, diff_degree=1):
        self.data = multiplier * self.data.pct_change(diff_degree).dropna()
        if self._is_split:
            self.train = multiplier * self.train.pct_change(diff_degree).dropna()
            self.val = multiplier * self.val.pct_change(diff_degree).dropna()
            self.test = multiplier * self.test.pct_change(diff_degree).dropna()

    def encode(self,
               scheme='mu',
               n_bins=4,
               shift=True,
               norm=False,
               clip=True):

        if scheme == 'mu':
            half = n_bins // 2
            self.data = mu_law_encode(self.data, n_bins, norm)
            if self._is_split:
                self.train = mu_law_encode(self.train, n_bins, norm)
                self.val = mu_law_encode(self.val, n_bins, norm)
                self.test = mu_law_encode(self.test, n_bins, norm)
            if shift:
                self.data -= half
                if self._is_split:
                    self.train -= half
                    self.val -= half
                    self.test -= half
            if clip:
                self._clip(n_bins, shift)
            self._is_quantized = True

        else:
            raise NotImplementedError

    def _clip(self, n_bins, shift):
        half = n_bins // 2
        if shift:
            self.data.clip(-half, half - 1, inplace=True)
            if self._is_split:
                self.train.clip(-half, half - 1, inplace=True)
                self.val.clip(-half, half - 1, inplace=True)
                self.test.clip(-half, half - 1, inplace=True)
        else:
            self.data.clip(0, n_bins - 1, inplace=True)
            if self._is_split:
                self.train.clip(0, n_bins - 1, inplace=True)
                self.val.clip(0, n_bins - 1, inplace=True)
                self.test.clip(0, n_bins - 1, inplace=True)

    def to_tensor(self, device):
        if self._is_tensor: return
        if self._is_quantized:
            self.data = torch.IntTensor(self.data.values).to(device)
            if self._is_split:
                self.train = torch.IntTensor(self.train.values).to(device)
                self.val = torch.IntTensor(self.val.values).to(device)
                self.test = torch.IntTensor(self.test.values).to(device)
        else:
            self.data = torch.FloatTensor(self.data.values).to(device)
            if self._is_split:
                self.train = torch.FloatTensor(self.train.values).to(device)
                self.val = torch.FloatTensor(self.val.values).to(device)
                self.test = torch.FloatTensor(self.test.values).to(device)
        self._is_tensor = True

    def one_hot_transform(self, n_bins):
        assert self._is_tensor, f'one_hot only implemented for torch Tensors'
        self.data = one_hot(self.data, dim=n_bins)
        if self._is_split:
            self.train = one_hot(self.train, dim=n_bins)
            self.val = one_hot(self.val, dim=n_bins)
            self.test = one_hot(self.test, dim=n_bins)

    @property
    def datasets(self):
        if self._is_split:
            return self.train, self.val, self.test
        else:
            return self.data

    def rolling_serve(self, index:int, window:int, stride:int):

        yield data