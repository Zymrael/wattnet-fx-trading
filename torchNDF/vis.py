import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def plot_cumsum_explained_variance(pca:PCA):
    """Plots cumulative sum of explained variance as a function of the number
    of PCA factors """
    c = pca.explained_variance_ratio_
    plt.plot(np.cumsum(c))
    plt.ylabel('explained variance')
    plt.xlabel('number factors')

def plot_trade_mode_embeddings(embeddings:torch.Tensor, label_idxs:torch.Tensor, dims_to_plot:int=10,
                               save_fpath:str=None):
    """Plots first `dims_to_plot` of embeddings labelled by model tenor actions"""
    plt.figure(figsize=(30,75))
    # action ranges (by tenor groups): 0 -> Hold, 1:91 -> Buy
    action_ranges = [slice(0,1), slice(10,30), slice(30,60), slice(60,91)]
    for dim in range(dims_to_plot):
        plt.subplot(10,3,1+dim)
        for i in range(4):
            action_idxs = np.concatenate(label_idxs[action_ranges[i]])
            plt.scatter(embeddings[action_idxs, dim],
                        embeddings[action_idxs,dim+1], s=12.8, alpha=0.6);
            plt.xlabel(f'Embedding dimension: {dim}')
            plt.ylabel(f'Embedding dimension: {dim+1}')
        plt.legend(['Hold', 'Buy_short', 'Buy_med', 'Buy_long'])
    if save_fpath: plt.savefig(f'{save_fpath}.jpg', dpi=200)

def plot_activity(predictions:np.array, model_name:str, curr_name:str, expert_labels:np.array, returns:np.array,
                  save_fpath:str=None):
    """Plots trading activity of experts and model"""
    n_preds = len(predictions)
    assert n_preds == len(expert_labels) == len(returns[0]), 'Number of predictions has to match number' + \
                                                             '# expert labels and size(1) for the return gradient'
    plt.figure(figsize=(20, 20))
    ax = plt.subplot(211)
    ax.scatter(list(range(n_preds)),
           expert_labels, alpha=1, color='r', edgecolor='black' , s=0.7);
    im = ax.matshow(returns, alpha=0.6, cmap='RdYlGn', aspect='auto', origin='lower', extent=[0,n_preds,-0.1,90.1],
                  vmin=-0.08, vmax=0.08)
    plt.colorbar(im)
    ax.set_yticks(np.arange(0, 90, 10))
    ax.set_yticklabels(['Hold', 'Buy<10d', 'Buy<20d', 'Buy<30d', 'Buy<40d', 'Buy<50d', 'Buy<60d', \
                        'Buy<70d', 'Buy<80d', 'Buy<90d'])
    ax.set_xlabel('Trading Days')
    plt.title(f'DTCC (oracle) `{curr_name}`, background: {curr_name} return gradient, 1 day tenor (bottom) to 90 days (top)')

    ax = plt.subplot(212)
    ax.scatter(list(range(n_preds)),
             predictions.cpu().numpy(), alpha=1, color='y', edgecolor='black', s=0.7)
    im = ax.matshow(returns, alpha=0.6, cmap='RdYlGn', aspect='auto', origin='lower', extent=[0,n_preds,-0.1,90.1],
                  vmin=-0.08, vmax=0.08)
    plt.colorbar(im)
    ax.set_yticks(np.arange(0, 90, 10))
    ax.set_yticklabels(['Hold', 'Buy<10d', 'Buy<20d', 'Buy<30d', 'Buy<40d', 'Buy<50d', 'Buy<60d', \
                        'Buy<70d', 'Buy<80d', 'Buy<90d'])
    ax.set_xlabel('Trading Days')
    plt.title(f'{model_name} on `{curr_name}`, background: {curr_name} return gradient, 1 day tenor (bottom) to 90 days (top)')
    if save_fpath: plt.savefig(f'{save_fpath}.jpg', dpi=200)

def plot_explaining_currency(gradients: torch.Tensor, sequence_idxs: torch.Tensor,
                             spots: np.array, volatility: np.array, mode: str = 'min',
                             n_explaining_currencies: int = 6):
    plt.figure(figsize=(30, 10))

    # `N, H, W` obtain index of minimum or maximum gradients
    # w.r.t input spot rates

    # average across sequence and batch samples
    gradients = gradients.mean(0).mean(1).cpu().numpy()

    # `N`
    # sort indices by gradient, then slice bottom `n_explaining_currencies`
    # or top `n_explaining_currencies` (at the start of the sorted arr)
    if mode == 'min':
        idx = np.argsort(gradients[:TECH_START_IDX])[:n_explaining_currencies]
    elif mode == 'max':
        idx = np.argsort(gradients[:TECH_START_IDX])[-n_explaining_currencies:]
    elif mode == 'abs':
        gradients = abs(gradients)
        idx = np.argsort(gradients[:TECH_START_IDX])[-n_explaining_currencies:]
    for i in range(n_explaining_currencies):
        plt.subplot(2, 3, 1 + i)
        plt.xticks([])
        plt.yticks([])
        curr_idx = idx[i].item()
        curr_n = spots.columns[curr_idx]
        # single currency time series
        ts = spots.iloc[:, curr_idx]
        plt.title(rf'{curr_n} $\rho$: {corr[i]}', fontsize=30)
        ts.plot()
        plt.pcolor(spots.index, [ts.min(), ts.max() + 0.5],
                   abs(volatility.iloc[:, curr_idx]).values[np.newaxis], alpha=0.3, cmap='Greens')
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=20)
        for el in sequence_idxs:
            el = el.cpu().item()
            # plot dot where tenor action is taken
            plt.scatter(spots.index[30 + el], ts[30 + el], s=20, color='black', zorder=4, marker='p')
            ts[el:30 + el].plot(color='red', alpha=0.4)