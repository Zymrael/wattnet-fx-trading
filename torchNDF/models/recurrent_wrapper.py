import numpy as np
import torch
import torch.nn as nn
from ..utils import expert_guided_loss
from ..metrics import cross_entropy_accuracy
from .mlp import MLP


class RecurrentWrapper(nn.Module):
    """Wrapper for recurrent models (GRU - LSTM)"""
    def __init__(self,
                 seq_len: int,
                 rec_cell_type: str,
                 in_dim: int,
                 latent_dim: int,
                 n_recurrent_layers: int,
                 mlp_in_dim: int=90,
                 mlp_out_dim: int=19,
                 mlp_layers=[128, 128],
                 dropout_prob: int=0.2
                 ):

        super().__init__()
        self.seq_len = seq_len
        self.rec_cell_type = rec_cell_type
        self._set_recurrent_layers(in_dim, latent_dim, n_recurrent_layers, dropout_prob)
        self.MLP = MLP(mlp_in_dim, mlp_out_dim, mlp_layers, drop_probability=dropout_prob, \
                       hidden_activation='leaky_relu', out_softmax=False)

    def _set_recurrent_layers(self, in_dim, ld, nl, dp):
        if self.rec_cell_type == 'LSTM':
            self.recurrent_layers = nn.LSTM(in_dim, ld, num_layers=nl, dropout=dp)
        elif self.rec_cell_type == 'GRU':
            self.recurrent_layers = nn.GRU(in_dim, ld, num_layers=nl, dropout=dp)
        else:
            print('f{self.rec_cell_type} not supported')

    def forward(self, x_in):
        x_in = x_in.transpose(0, 1)
        x_in, _ = self.recurrent_layers(x_in)
        # last latent
        x_in = x_in[-1, :, :]
        x_out = self.MLP(x_in)
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
            early_stop_loss=2.,
            dropout_interval=1000,
            tr_returns=None,
            val_returns=None
            ):
        if valloader:
            x_val, val_idx = next(iter(valloader))
            x_val_clip = x_val[:, :-1, :].to(device)
            y_val = val_returns.argmax(0)
            val_opt_rets, _ = val_returns.max(0)
            val_opt_rets = val_opt_rets.sum().item()

        for e in range(epochs):
            drop_idx, run_rets, run_loss, run_opt_rets = 0, 0., 0., 0.
            iterator = iter(trainloader)
            sched.step()
            for i in range(len(iterator)):
                opt.zero_grad()
                x, idx = next(iterator)
                x = x.to(device)
                x_clip = x[:, :-1, :]
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
                        val_acc = cross_entropy_accuracy(probs, y_val).item()
                        val_act = probs.argmax(1)
                        val_mod_rets = (val_returns[val_act, val_idx]).sum().item()

                        print(f'Validation Loss: {np.round(val_loss, 2)}')
                        print(f'Validation Accuracy: {np.round(val_acc, 2)}')
                        print(f'Avg val returns: {np.round(val_mod_rets, 2)}')
                        print(f'Avg val optimal returns: {np.round(val_opt_rets, 2)} \n')
            if e % dropout_interval and dropout_schedule:
                drop_idx += 1
                if drop_idx < len(dropout_schedule):
                    self.MLP.drop_probability = dropout_schedule[drop_idx]
                    self.recurrent_layers.drop_probability = dropout_schedule[drop_idx]