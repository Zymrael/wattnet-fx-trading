import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .mlp import MLP
from .modules import *
from ..utils import expert_guided_loss
from ..metrics import cross_entropy_accuracy


class WATTNet(nn.Module):
    def __init__(self, in_dim: int = 1132, out_dim: int = 91, w_dim: int = 128, emb_dim: int = 8,
                 dilation_depth: int = 4, dropout_prob: float = 0.2, n_repeat: int = 2):
        """
        Args:
            w_dim: spatial compression dimension carried out by a 2-layer MLP.
                           When more memory/data is available, increasing w_dim can yield better performance
            emb_dim: embedding dimension of scalar values for each of the `w_dim` left after compression.
                     Higher embedding dimension increases accuracy of the spatial attention module at the cost
                     of increased memory requirement. BEWARE: w_dim * emb_dim > 1e4 can get *VERY* costly in terms
                     of GPU memory, especially with big batches.
            dilation_depth: number of temporal-spatial blocks. Dilation for temporal dilated convolution is doubled
                            each time.
            n_repeat: number of repeats of #`dilation_depth` of temporal-spatial layers. Useful to increase model depth
                      with short sequences without running into situations where the dilated kernel becomes wider than the
                      sequence itself.
        """
        super().__init__()
        self.w_dim = w_dim
        self.emb_dim = emb_dim
        self.dilation_depth = dilation_depth
        self.n_layers = dilation_depth * n_repeat
        self.dilations = [2 ** i for i in range(1, dilation_depth + 1)] * n_repeat

        ltransf_dim = w_dim * emb_dim
        self.attblocks = nn.ModuleList([AttentionBlock(in_channels=w_dim,
                                                       key_size=ltransf_dim,
                                                       value_size=ltransf_dim)
                                        for _ in self.dilations])

        self.resblocks = nn.ModuleList([GatedBlock(dilation=d, w_dim=w_dim)
                                        for d in self.dilations])

        self.emb_conv = nn.Conv2d(1, emb_dim, kernel_size=1)
        self.dec_conv = nn.Conv2d(w_dim, w_dim, kernel_size=(1, emb_dim), groups=w_dim)

        # feature compression: when more memory/data is available, increasing w_dim can yield
        # better performance
        self.preMLP = MLP(in_dim, w_dim, out_softmax=False)

        # post fully-connected head not always necessary. When sequence length perfectly aligns
        # with the number of time points lost to high dilation, (i.e single latent output by
        # alternating TCN and attention modules) the single latent can be used directly
        self.postMLP = MLP(5 * w_dim, out_dim, [512], \
                           out_softmax=False, drop_probability=dropout_prob)

    def forward(self, x_in):
        """
        Args:
            x_in: 'N, C, H, W' where `N` is the batch dimension, `C` the one-hot
                  embedding dimension, `H` is the temporal dimension, `W` is the
                  second dimension of the timeseries (e.g timeseries for different FX pairs)
        Returns:
        """
        x_in = self.preMLP(x_in.squeeze(1))
        x_in = x_in.unsqueeze(1)

        if self.emb_dim > 1: x_in = self.emb_conv(x_in)

        # swap `W` dim to channel dimension for grouped convolutions
        # `N, W, H, C`
        x_in = x_in.transpose(1, 3)

        skip_connections = []
        for i in range(len(self.resblocks)):
            x_in = self.resblocks[i](x_in)
            x_att_list = []
            # slicing across `H`, temporal dimension
            for k in range(x_in.size(2)):
                # `C` embedding message passing using self-att
                x_att = self.attblocks[i](x_in[:, :, k, :])
                # `N, W, C` -> `N, W, 1, C`
                x_att = x_att.unsqueeze(2)
                x_att_list.append(x_att)
                # `N, W, 1, C` -> `N, W, H, C`
            x_in = torch.cat(x_att_list, dim=2)
        # `N, W, H, C` ->  `N, W, H, 1`
        if self.emb_dim > 1: x_in = self.dec_conv(x_in)
        # `N, W, H, 1` ->  `N, 1, H, W`
        x_out = x_in.transpose(1, 3)
        # `N, 1, H, W` ->  `N, H, W`
        x_out = x_out[:, 0, :, :]

        x_out = x_out.reshape(x_out.size(0), -1)
        x_out = self.postMLP(x_out)
        return x_out

    def fit(self,
            epochs,
            trainloader,
            valloader,
            opt,
            sched,
            device,
            log_interval=10000,
            dropout_schedule=None,
            dropout_interval=1000,
            early_stop_loss=2.,
            tr_returns=None,
            val_returns=None
            ):

        if valloader:
            x_val, val_idx = next(iter(valloader))
            x_val_clip = x_val[:, :-1, :].unsqueeze(1).to(device)
            y_val = val_returns.argmax(0)
            val_opt_rets, _ = val_returns.max(0)
            val_opt_rets = val_opt_rets.sum().item()

        for e in range(epochs):
            drop_idx, run_loss, run_rets, run_opt_rets = 0, 0., 0., 0.
            iterator = iter(trainloader)
            if sched: sched.step()
            for i in range(len(iterator)):
                opt.zero_grad()
                x, idx = next(iterator)
                x = x.to(device)
                x_clip = x[:, :-1, :].unsqueeze(1)
                yhat = self(x_clip)
                loss, iter_rets, iter_opt_rets = expert_guided_loss(yhat, tr_returns, idx)
                run_loss += loss.item()
                # early stopping check:
                if run_loss / (i + 1) < early_stop_loss:
                    print(f'Early stopping...')
                    return None

                run_rets += iter_rets.item()
                run_opt_rets += iter_opt_rets.item()
                loss.backward()
                opt.step()

                if i % log_interval == 0:
                    print(f'Epoch: {e}')
                    print(f'Training Loss: {np.round(run_loss / (i + 1), 2)}')
                    print(f'Avg train returns: {np.round(run_rets / (i + 1), 2)}')
                    print(f'Avg train optimal returns: {np.round(run_opt_rets / (i + 1), 2)} \n')

                    if valloader:
                        yhat = self(x_val_clip)
                        val_loss = nn.CrossEntropyLoss()(yhat, y_val).item()
                        probs = nn.Softmax(1)(yhat)
                        val_act = probs.argmax(1)
                        val_mod_rets = (val_returns[val_act, val_idx]).sum().item()

                        val_acc = cross_entropy_accuracy(probs, y_val).item()

                        print(f'Validation Loss: {np.round(val_loss, 2)}')
                        print(f'Validation Accuracy: {np.round(val_acc, 2)} %')
                        print(f'Avg val returns: {np.round(val_mod_rets, 2)}')
                        print(f'Avg val optimal returns: {np.round(val_opt_rets, 2)} \n')

            if e % dropout_interval and dropout_schedule:
                drop_idx += 1
                if drop_idx < len(dropout_schedule):
                    self.preMLP.drop_probability = dropout_schedule[drop_idx]
                    self.postMLP.drop_probability = dropout_schedule[drop_idx]