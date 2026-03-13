import torch
import torch.nn as nn
import numpy as np
from functools import partial  # Used for binding hook layer names
import random
import torch.nn.functional as F
from src.model_module.brain_decoder import register_model
from src.data_module.augmentation import add_noise, ChannelMasking, TimeMasking, random_shift, Mixup


class ResidualBiGRU(nn.Module):
    def __init__(self, hidden_size, n_layers=1, bidir=True):
        super(ResidualBiGRU, self).__init__()

        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.gru = nn.GRU(
            hidden_size,
            hidden_size,
            n_layers,
            batch_first=True,
            bidirectional=bidir,
        )
        dir_factor = 2 if bidir else 1
        self.fc1 = nn.Linear(
            hidden_size * dir_factor, hidden_size * dir_factor * 2
        )
        self.ln1 = nn.LayerNorm(hidden_size * dir_factor * 2)
        self.fc2 = nn.Linear(hidden_size * dir_factor * 2, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        for name, param in self.gru.named_parameters():
            if 'weight_ih' in name:
                nn.init.orthogonal_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)

    def forward(self, x, h=None):
        res, new_h = self.gru(x, h)
        # res.shape = (batch_size, sequence_size, 2*hidden_size)
        res = self.fc1(res)
        res = self.ln1(res)
        res = nn.functional.relu(res)
        res = self.fc2(res)
        res = self.ln2(res)
        res = nn.functional.relu(res)
        # skip connection
        res = res + x

        return res, new_h


@register_model('MultiResGRU')
class MultiResidualBiGRU(nn.Module):
    def __init__(self, cfg):
        super(MultiResidualBiGRU, self).__init__()
        self.cfg = cfg
        self.task = cfg.dataset.task
        self.channel_dict = np.load(cfg.dataset.channel_dict_path, allow_pickle=True)
        self.channel_name = cfg.dataset.channel_index_name

        # -------------------------- Layer output collection configuration --------------------------
        self.collect_activations = False  # Collection switch
        self.activations = {}  # Store layer outputs (key=layer name, value=output data)

        # Channel selection logic (unchanged)
        if self.channel_name == 'all':
            self.selected_channels = list(range(cfg.dataset.input_channels))
        elif isinstance(self.channel_name, int):
            self.selected_channels = [self.channel_name]
        elif isinstance(self.channel_name, str) and len(self.channel_name) == 1:
            self.selected_channels = self.channel_dict[self.channel_name]
        elif isinstance(self.channel_name, str) and len(self.channel_name) > 1:
            split_channels = self.channel_name.split('-')
            selected_channels = []
            for i in range(len(split_channels)):
                if split_channels[i].isalpha() or split_channels[i][0].isalpha():
                    selected_channels.append(self.channel_dict[split_channels[i]])
                else:
                    selected_channels.append(int(split_channels[i]))
            self.selected_channels = selected_channels

        self.input_size = len(self.selected_channels)
        self.hidden_size = cfg.model.hidden_size
        self.n_layers = cfg.model.n_layers
        self.bidir = cfg.model.bidir
        self.cfg = cfg
        self.fc_in = nn.Linear(self.input_size, self.hidden_size)  # Input layer (named embedding)
        self.ln = nn.LayerNorm(self.hidden_size)
        self.res_bigrus = nn.ModuleList(
            [
                ResidualBiGRU(self.hidden_size, n_layers=1, bidir=self.bidir)
                for _ in range(self.n_layers)
            ]
        )
        self.target_size = cfg.dataset.target_size[cfg.dataset.task]
        self.fc_out = nn.Linear(self.hidden_size, self.target_size)
        self.loss_fn = nn.BCEWithLogitsLoss()

        # -------------------------- Simplified Hook registration --------------------------
        # Input layer (named embedding)
        self.fc_in.register_forward_hook(partial(self.hook_fn, layer_name="embedding"))
        # Outputs of GRU, fc1, fc2 for each ResidualBiGRU
        for idx, res_bigru in enumerate(self.res_bigrus):
            res_bigru.gru.register_forward_hook(partial(self.hook_fn, layer_name=f"res_bigru_{idx}_gru"))
            res_bigru.fc2.register_forward_hook(partial(self.hook_fn, layer_name=f"res_bigru_{idx}_fc"))

    # -------------------------- Hook function --------------------------
    def hook_fn(self, module, input, output, layer_name):
        """Collect only specified layer outputs, compatible with GRU tuple outputs"""
        if self.collect_activations:
            if isinstance(output, tuple):
                # GRU returns tuple: (sequence output, hidden state), only store sequence output (or store as needed)
                self.activations[layer_name] = output[0].detach().cpu().numpy()  # Take sequence output
                # If need to store hidden states simultaneously: self.activations[layer_name] = (output[0].detach().cpu().numpy(), output[1].detach().cpu().numpy())
            else:
                # Linear layer output (single tensor)
                self.activations[layer_name] = output.detach().cpu().numpy()

    # -------------------------- Control interfaces --------------------------
    def start_collect_activations(self):
        self.collect_activations = True
        self.activations = {}  # Clear historical data

    def stop_collect_activations(self):
        self.collect_activations = False

    def get_activations(self):
        return self.activations.copy()

    def apply_augmentation(self, x):
        shift_ratio = self.cfg.augmentation.shift_ratio
        augmentations = [
            (lambda x: random_shift(x, shift_ratio), self.cfg.augmentation.random_shift_prob),
            (add_noise(), self.cfg.augmentation.add_noise_prob),
            (ChannelMasking(), self.cfg.augmentation.ChannelMasking_prob),
            (TimeMasking(), self.cfg.augmentation.TimeMasking_prob),
        ]
        for aug_func, prob in augmentations:
            if random.random() < prob:
                x = aug_func(x)
            else:
                x = x
        return x

    def apply_tta(self, x):
        x = random_shift(x)
        return x

    def forward(self, ieeg_raw_data, labels=None, **kwargs):
        # Clear historical data before collection
        if self.collect_activations:
            self.activations = {}

        if type(ieeg_raw_data) == dict:
            ieeg_raw_data, labels = ieeg_raw_data["ieeg_raw_data"], ieeg_raw_data["labels"]
        x = ieeg_raw_data[:, self.selected_channels]
        if labels is not None:
            if type(labels) != torch.LongTensor:
                labels = labels.long()
            one_hot_labels = F.one_hot(labels, num_classes=self.target_size).float()

        x = x.float()

        if self.training:
            x = self.apply_augmentation(x)
            p = random.random()
            if p < self.cfg.augmentation.mixup_prob:
                x, one_hot_labels = Mixup(alpha=0.5)(x, one_hot_labels)

            x = x.permute(0, 2, 1)

            x = self.fc_in(x)  # embedding layer
            h = [None for _ in range(self.n_layers)]
            x = self.ln(x)
            x = nn.functional.relu(x)
            new_h = []
            for i, res_bigru in enumerate(self.res_bigrus):
                x, new_hi = res_bigru(x, h[i])
                new_h.append(new_hi)
            prediction = self.fc_out(x.mean(1))
            loss1 = self.loss_fn(prediction, one_hot_labels)
            loss = loss1
            return {"loss"  : loss,
                    "labels": labels,
                    "logits": prediction}
        else:
            all_logits = []
            with torch.no_grad():
                x = x.float()
                for idx in range(self.cfg.model.tta_times):
                    if idx == 0:
                        x_aug = x.clone()
                    else:
                        x_aug = self.apply_tta(x.clone())

                    x_aug = x_aug.permute(0, 2, 1)
                    x_aug = self.fc_in(x_aug)  # embedding layer
                    h = [None for _ in range(self.n_layers)]
                    x_aug = self.ln(x_aug)
                    x_aug = nn.functional.relu(x_aug)
                    new_h = []
                    for i, res_bigru in enumerate(self.res_bigrus):
                        x_aug, new_hi = res_bigru(x_aug, h[i])
                        new_h.append(new_hi)
                    prediction = self.fc_out(x_aug.mean(1))
                    all_logits.append(prediction)
            avg_logits = torch.mean(torch.stack(all_logits), dim=0)
            loss = self.loss_fn(avg_logits, one_hot_labels) if labels is not None else None

            return {
                "loss"  : loss,
                "labels": labels,
                "logits": avg_logits,
            }