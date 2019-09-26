import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .utils import seq_normalization

def cross_entropy_accuracy(preds, targets):
    """Accuracy for regular CrossEntropyLoss"""
    _, preds = torch.max(preds, dim=1)
    acc = 100*(preds == targets).float().mean()
    return acc

def weighted_cross_entropy_accuracy(preds:torch.Tensor, targets:torch.Tensor, weights:torch.Tensor):
    """Accuracy function for unbalanced classes"""
    preds, targets, weights = preds.cpu(), targets.cpu(), weights.cpu()
    _, preds = torch.max(preds, dim=1)
    weighted_preds = copy.deepcopy(preds).to(dtype=torch.float)
    weighted_targets = copy.deepcopy(targets).to(dtype=torch.float)
    weighted_preds.apply_(lambda x: weights[int(x)])
    weighted_targets.apply_(lambda x: weights[int(x)])
    are_equal = (preds == targets).to(dtype=torch.float)
    acc = 100 * torch.sum(weighted_preds * are_equal)/torch.sum(weighted_targets)
    return acc

class BaselinePerfContainer:
    """Simple container of performance metrics"""
    def __init__(self):
        self.returns = 0
        self.return_list = []
        self.volatility = 0.
        self.activity = 0
        self.days_held = 0
        self.pos_acc = 0.
        self.opt_acc = 0.
    def __repr__(self):
        return f'Returns {round(self.returns, 2)}\n' + \
               f'Standard dev of returns {round(self.volatility, 4)}\n' + \
               f'Cumulative sum of tenors {self.days_held}\n' + \
               f'Number of Buys {self.activity}\n' + \
               f'Positive return accuracy {round(self.pos_acc, 2)}\n' + \
               f'Optimal return accuracy {round(self.opt_acc, 2)}\n'

def positive_return_accuracy(preds:np.array, returns:np.array):
    """Computes accuracy against positive return tenor actions"""
    count = 0
    for i in range(len(returns[0])):
        if returns[int(preds[i]),i] >= 0:
            count += 1
    return 100*count/len(returns[0])

def returns_and_activity(returns: torch.Tensor, predictions: torch.Tensor = None, baseline: str = None,
                         input_data: torch.Tensor = None, confidence_multipliers: torch.Tensor = None,
                         frictional_check: bool = False, frictional_act: torch.Tensor = None):
    """Calculate trading returns of model given `predictions` or returns of one of the following baselines:
        `random`: uniform random action
        `expert`: expert positive return trades computed from `input_data` labels
        confidence_multipliers, if given as input, are used to weigh model returns"""
    options = ['last_seen', 'expert', 'random', None]
    assert (predictions is not None) | (baseline is not None), \
        'If not using a standard model please choose a baseline'
    assert baseline in options, f'{baseline} not supported'

    if baseline == 'random':
        # random action each step
        predictions = np.random.choice(np.arange(0, 91), len(input_data), replace=True)
        predictions = torch.Tensor(predictions).to(dtype=torch.long)

    container = BaselinePerfContainer()
    for idx, act in enumerate(predictions):
        # if action not `Hold`
        if act != 0:
            act = int(act)
            if frictional_check:
                if frictional_act[idx] != act: continue
            if confidence_multipliers is not None:
                container.return_list.append(confidence_multipliers[idx].item() * returns[act, idx])
            else:
                container.return_list.append(returns[act, idx])
            container.days_held += act
        else:
            container.return_list.append(0.)
    container.return_list = np.array(container.return_list)

    container.activity = torch.numel(predictions.nonzero())
    container.volatility = container.return_list.std()
    container.returns = container.return_list.sum()
    _, optimal_actions = returns.max(0)
    container.opt_acc = 100 * (predictions == optimal_actions).to(dtype=torch.float).mean().item()
    container.pos_acc = positive_return_accuracy(predictions, returns)
    return container

def ensemble_single_model_returns(ensemble:list, dataloader:DataLoader, returns:torch.Tensor, device:torch.device):
    """Returns for each model in an ensemble"""
    data, _ = next(iter(dataloader))
    performances = []
    for model in ensemble:
        data[:, :-1, :-1] = seq_normalization(data[:, :-1, :-1])
        yhat = model(data[:, :-1, :-1].to(device))
        probs = nn.Softmax(1)(yhat)
        _, predictions = torch.max(probs, 1)
        return_model = 0
        for idx, el in enumerate(predictions):
            # if action not `Hold`
            if el != 0:
                return_model += returns[el, idx]
        performances.append(return_model)
    return performances