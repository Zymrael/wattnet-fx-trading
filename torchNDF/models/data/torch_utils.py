import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from ..utils import seq_normalization

class Hook():
    """Generic hook for nn.Modules with input/output"""
    def __init__(self, module, backward=False):
        if backward==False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output
    def close(self):
        self.hook.remove()

class RegularFinancialData(Dataset):
    """Class of torch datasets for loading the financial data. When using with dataloaders,
    set shuffle=False to sample via shifting (consecutive samples share `seq_len - 1` data points).
    Especially useful for forecasting validation dataloaders.
    Data shape is:
        `(n_samples, seq_len, n_feat)` = `(N, H, C)`
    """
    def __init__(self, dataset, forecast_length):
        """
        Args:
            dataset: the torch dataset
        """
        self._data = dataset
        self.fl = forecast_length

    def __len__(self):
        """ Returns length of the dataset. Length calculated as len(data) - forecast length
        to avoid sampling time series shorter than forecast length. """
        return len(self._data) - self.fl

    def __getitem__(self, idx):
        batch = self._data[idx:idx + self.fl, :]
        return batch, idx


def ensemble_predictions(ensemble:list, data:torch.Tensor, device:torch.device, mode:str='min', softmax=True):
    """Returns predictions of a voting or min ensemble of models. Can be used for single models
       if length of ensemble is 1"""
    preds = torch.LongTensor([]).to(device)
    for model in ensemble:
        yhat = model(data)
        if softmax: yhat = nn.Softmax(1)(yhat)
        _, pred = torch.max(yhat, 1)
        preds = torch.cat((preds, pred.unsqueeze(0)))
    if mode=='min': predictions, _ = preds.cpu().min(0)
    else: predictions, _ = torch.mode(preds.cpu(), 0)
    return predictions
