import torch
import torch.nn as nn
import torch.nn.functional as F

def seq_normalization(x:torch.Tensor, col_idx:int=145):
    """Normalizes a Tensor across its temporal (1) dimension `N. H, W` up to `col_idx`
    in the `W` dimension"""
    epsilon = 1e-10
    m_x = x[:,:,:col_idx]
    x[:,:,:col_idx] = (m_x - m_x.mean(1).unsqueeze(1))/(m_x.std(1).unsqueeze(1)+epsilon)
    return x

def masked_temperature_softmax(x:torch.Tensor, idx:int, T:float=0.01):
    """Temperature scaled softmax on torch.Tensor masked by indices `idx`"""
    return torch.exp(x[idx, :] / T) / torch.exp(x[idx, :] / T).sum()

def scaled_cross_entropy(preds:torch.Tensor, labels, order_penalty=0.2, margin=2, margin_penalty=0.1):
    """
    Computes 2D soft cross entropy loss between with asymmetric scaling `preds` and `labels`.

    Args:
        preds (torch.Tensor): shape `N, C, H, W`
        labels (torch.Tensor): shape `N, H, W`
        order_penalty int: percentage penalty to predictions with index bigger than ground truth
                           label
        margin int: maximum distance without penalty between ground truth index and prediction index
        margin_penalty int: percentage penalty to predictions with index outside `margin`

    Returns:
        loss (torch.Tensor):
    """
    loss = 0
    # loop through samples
    for i in range(preds.size(0)):
        # loop through `H` dim
        for j in range(preds.size(2)):
            # loop through `W` dim
            for k in range(preds.size(3)):
                # weight vector of length num. classes `C`
                w = preds.new_ones(preds.size(1))
                positive_label = labels[i, j, k].data
                w[positive_label:] += order_penalty
                if positive_label > margin:
                    w[:positive_label - margin] += margin_penalty
                if positive_label < preds.size(1) - margin:
                    w[positive_label + 2:] += margin_penalty
                loss += F.cross_entropy(preds[None, i, :, j, k], labels[None, i, j, k,], weight=w)
    loss /= torch.numel(labels)
    return loss

def expert_guided_loss(yhat: torch.Tensor, returns: torch.Tensor, index: torch.Tensor):
    """Compute CrossEntropyLoss between `yhat` and optimal return actions given `returns`"""
    probs = nn.Softmax(1)(yhat)
    action = torch.argmax(probs, 1)
    model_return = returns[action, index].sum(0)
    optimal_action = returns[:, index].argmax(0)
    optimal_return, _ = (returns[:, index].max(0))
    optimal_return = optimal_return.sum()
    loss = nn.CrossEntropyLoss()(yhat, optimal_action)
    return loss, model_return, optimal_return

def risk_penalty(x:torch.Tensor, returns:torch.Tensor):
    """Linearly scaling penalty for long tenor returns"""
    x = torch.sign(x) * (torch.abs(x) * torch.linspace(1, 0.5, returns.size(0)).unsqueeze(1).type_as(x))
    return x

def expert_risk_aware_loss(yhat: torch.Tensor, returns: torch.Tensor, index: torch.Tensor):
    """Compute CrossEntropyLoss between `yhat` and optimal return actions given `returns`"""
    probs = nn.Softmax(1)(yhat)
    action = torch.argmax(probs, 1)
    model_return = returns[action, index].sum(0)
    mask = (returns[:, index] > 0)
    optimal_action = mask.argmax(0)
    optimal_return, _ = (returns[optimal_action, index].max(0))
    optimal_return = optimal_return.sum()
    loss = nn.CrossEntropyLoss()(yhat, optimal_action)
    return loss, model_return, optimal_return

def probabilistic_expert_guided_loss(yhat:torch.Tensor, returns:torch.Tensor, index:torch.Tensor):
    """Compute CrossEntropyLoss between `yhat` and optimal return actions given `returns`"""
    action = torch.argmax(yhat, 1)
    model_return = returns[action, index].sum(0)
    optimal_distrib = masked_temperature_softmax(returns.transpose(0, 1), index) # returns: `W, H` -> `H, W`
    expert_sampled_action = torch.distributions.Multinomial(1, probs=optimal_distrib).sample()
    optimal_action = expert_sampled_action.argmax(0)
    optimal_return, _ = (returns[:, index].max(0))
    optimal_return = optimal_return.sum()
    loss = nn.CrossEntropyLoss()(yhat, optimal_action)
    return loss, model_return, optimal_return

def one_hot(input_data:torch.Tensor, dim:int):
    """ Turns input_data of shape `H, W` with integer entries into a tensor of shape `H, C, W` where
    `C` is the one-hot encoding dimension. """
    res = []
    n_channels = input_data.size(1)
    offset = input_data.min()
    length = dim
    for channel_idx in range(n_channels):
        channel_one_hot = []
        channel = input_data[:, channel_idx]
        for entry in channel:
            one_hot_x = torch.zeros(length)
            one_hot_x[entry+offset] = 1
            channel_one_hot.append(one_hot_x)
        channel_one_hot = torch.cat(channel_one_hot)
        channel_one_hot = channel_one_hot.reshape(-1, length)
        res.append(channel_one_hot.unsqueeze(2))
    res = torch.cat(res, dim=2)
    return res