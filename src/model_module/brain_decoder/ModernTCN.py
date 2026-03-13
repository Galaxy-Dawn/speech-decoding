import einops
import torch
from omegaconf import ListConfig
from torch import nn
import torch.nn.functional as F
from src.model_module.brain_decoder import register_model
from src.data_module.augmentation import add_noise, ChannelMasking, TimeMasking, random_shift, Mixup
import random
import numpy as np
from functools import partial  # Added: used for binding layer names to hooks


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


# forecast task head
class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0):
        super(Flatten_Head, self).__init__()

        self.individual = individual
        self.n_vars = n_vars

        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:, i, :, :])  # z: [bs x d_model * patch_num]
                z = self.linears[i](z)  # z: [bs x target_window]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)  # x: [bs x nvars x target_window]
        else:
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x)
        return x


class LayerNorm(nn.Module):

    def __init__(self, channels, eps=1e-6, data_format="channels_last"):
        super(LayerNorm, self).__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        B, M, D, N = x.shape
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(B * M, N, D)
        x = self.norm(x)
        x = x.reshape(B, M, N, D)
        x = x.permute(0, 1, 3, 2)
        return x


def get_conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias):
    return nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=bias)


def get_bn(channels):
    return nn.BatchNorm1d(channels)


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1, bias=False):
    if padding is None:
        padding = kernel_size // 2
    result = nn.Sequential()
    result.add_module('conv', get_conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias))
    result.add_module('bn', get_bn(out_channels))
    return result


def fuse_bn(conv, bn):
    kernel = conv.weight
    running_mean = bn.running_mean
    running_var = bn.running_var
    gamma = bn.weight
    beta = bn.bias
    eps = bn.eps
    std = (running_var + eps).sqrt()
    t = (gamma / std).reshape(-1, 1, 1)
    return kernel * t, beta - running_mean * gamma / std


class ReparamLargeKernelConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, groups,
                 small_kernel,
                 small_kernel_merged=False, nvars=7):
        super(ReparamLargeKernelConv, self).__init__()
        self.kernel_size = kernel_size
        self.small_kernel = small_kernel
        # We assume the conv does not change the feature map size, so padding = k//2. Otherwise, you may configure padding as you wish, and change the padding of small_conv accordingly.
        padding = kernel_size // 2
        if small_kernel_merged:
            self.lkb_reparam = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, dilation=1, groups=groups, bias=True)
        else:
            self.lkb_origin = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                      stride=stride, padding=padding, dilation=1, groups=groups, bias=False)
            if small_kernel is not None:
                assert small_kernel <= kernel_size, 'The kernel size for re-param cannot be larger than the large kernel!'
                self.small_conv = conv_bn(in_channels=in_channels, out_channels=out_channels,
                                          kernel_size=small_kernel,
                                          stride=stride, padding=small_kernel // 2, groups=groups, dilation=1,
                                          bias=False)

    def forward(self, inputs):

        if hasattr(self, 'lkb_reparam'):
            out = self.lkb_reparam(inputs)
        else:
            out = self.lkb_origin(inputs)
            if hasattr(self, 'small_conv'):
                out += self.small_conv(inputs)
        return out

    def PaddingTwoEdge1d(self, x, pad_length_left, pad_length_right, pad_values=0):

        D_out, D_in, ks = x.shape
        if pad_values == 0:
            pad_left = torch.zeros(D_out, D_in, pad_length_left)
            pad_right = torch.zeros(D_out, D_in, pad_length_right)
        else:
            pad_left = torch.ones(D_out, D_in, pad_length_left) * pad_values
            pad_right = torch.ones(D_out, D_in, pad_length_right) * pad_values
        x = torch.cat([pad_left, x], dim=-1)
        x = torch.cat([x, pad_right], dim=-1)
        return x

    def get_equivalent_kernel_bias(self):
        eq_k, eq_b = fuse_bn(self.lkb_origin.conv, self.lkb_origin.bn)
        if hasattr(self, 'small_conv'):
            small_k, small_b = fuse_bn(self.small_conv.conv, self.small_conv.bn)
            eq_b += small_b
            eq_k += self.PaddingTwoEdge1d(small_k, (self.kernel_size - self.small_kernel) // 2,
                                          (self.kernel_size - self.small_kernel) // 2, 0)
        return eq_k, eq_b

    def merge_kernel(self):
        eq_k, eq_b = self.get_equivalent_kernel_bias()
        self.lkb_reparam = nn.Conv1d(in_channels=self.lkb_origin.conv.in_channels,
                                     out_channels=self.lkb_origin.conv.out_channels,
                                     kernel_size=self.lkb_origin.conv.kernel_size, stride=self.lkb_origin.conv.stride,
                                     padding=self.lkb_origin.conv.padding, dilation=self.lkb_origin.conv.dilation,
                                     groups=self.lkb_origin.conv.groups, bias=True)
        self.lkb_reparam.weight.data = eq_k
        self.lkb_reparam.bias.data = eq_b
        self.__delattr__('lkb_origin')
        if hasattr(self, 'small_conv'):
            self.__delattr__('small_conv')


class Block(nn.Module):
    def __init__(self, large_size, small_size, dmodel, dff, nvars, small_kernel_merged=False, drop=0.1):
        super(Block, self).__init__()
        self.dw = ReparamLargeKernelConv(in_channels=nvars * dmodel, out_channels=nvars * dmodel,
                                         kernel_size=large_size, stride=1, groups=nvars * dmodel,
                                         small_kernel=small_size, small_kernel_merged=small_kernel_merged, nvars=nvars)
        self.norm = nn.BatchNorm1d(dmodel)

        # convffn1
        self.ffn1pw1 = nn.Conv1d(in_channels=nvars * dmodel, out_channels=nvars * dff, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=nvars)
        self.ffn1act = nn.GELU()
        self.ffn1pw2 = nn.Conv1d(in_channels=nvars * dff, out_channels=nvars * dmodel, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=nvars)
        self.ffn1drop1 = nn.Dropout(drop)
        self.ffn1drop2 = nn.Dropout(drop)

        # convffn2
        self.ffn2pw1 = nn.Conv1d(in_channels=nvars * dmodel, out_channels=nvars * dff, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=dmodel)
        self.ffn2act = nn.GELU()
        self.ffn2pw2 = nn.Conv1d(in_channels=nvars * dff, out_channels=nvars * dmodel, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=dmodel)
        self.ffn2drop1 = nn.Dropout(drop)
        self.ffn2drop2 = nn.Dropout(drop)

        self.ffn_ratio = dff // dmodel

    def forward(self, x):
        input = x
        B, M, D, N = x.shape
        x = x.reshape(B, M * D, N)
        x = self.dw(x)
        x = x.reshape(B, M, D, N)
        x = x.reshape(B * M, D, N)
        x = self.norm(x)
        x = x.reshape(B, M, D, N)
        x = x.reshape(B, M * D, N)

        x = self.ffn1drop1(self.ffn1pw1(x))
        x = self.ffn1act(x)
        x = self.ffn1drop2(self.ffn1pw2(x))
        x = x.reshape(B, M, D, N)

        x = x.permute(0, 2, 1, 3)
        x = x.reshape(B, D * M, N)
        x = self.ffn2drop1(self.ffn2pw1(x))
        x = self.ffn2act(x)
        x = self.ffn2drop2(self.ffn2pw2(x))
        x = x.reshape(B, D, M, N)
        x = x.permute(0, 2, 1, 3)

        x = input + x
        return x


class Stage(nn.Module):
    def __init__(self, ffn_ratio, num_blocks, large_size, small_size, dmodel, nvars,
                 small_kernel_merged=False, drop=0.1):

        super(Stage, self).__init__()
        d_ffn = dmodel * ffn_ratio
        blks = []
        for i in range(num_blocks):
            blk = Block(large_size=large_size, small_size=small_size, dmodel=dmodel, dff=d_ffn, nvars=nvars,
                        small_kernel_merged=small_kernel_merged, drop=drop)
            blks.append(blk)

        self.blocks = nn.ModuleList(blks)

    def forward(self, x):

        for blk in self.blocks:
            x = blk(x)

        return x


class ConvBNReLU(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            groups=1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                              padding=1, groups=groups, bias=False)
        self.norm = nn.BatchNorm1d(out_channels, eps=1e-5)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class ModernTCN(nn.Module):
    def __init__(self, patch_size, patch_stride, downsample_ratio, ffn_ratio, num_blocks, large_size,
                 small_size, dims,
                 nvars, small_kernel_merged=False, backbone_dropout=0.1, head_dropout=0.1, use_multi_scale=True,
                 revin=True, affine=True,
                 subtract_last=False, seq_len=512, c_in=7, individual=False, target_window=96, class_drop=0.,
                 class_num=10):

        super(ModernTCN, self).__init__()

        self.class_drop = class_drop
        self.class_num = class_num

        # -------------------------- Added: Layer output collection configuration --------------------------
        self.collect_activations = False  # Collection switch (disabled by default)
        self.activations = {}  # Dictionary to store layer outputs (key: layer name, value: numpy array)

        # RevIN (remove hook registration)
        self.revin = revin
        if self.revin:
            self.revin_layer = RevIN(nvars, affine=affine, subtract_last=subtract_last)

        # stem layer & down sampling layers
        self.downsample_layers = nn.ModuleList()
        self.stem = nn.Sequential(
            nn.Conv1d(1, dims[0], kernel_size=patch_size, stride=patch_stride),
            nn.BatchNorm1d(dims[0])
        )
        self.stem.register_forward_hook(partial(self.hook_fn, layer_name="downsample_layer_0"))

        self.downsample_layers.append(self.stem)

        self.num_stage = len(num_blocks)
        if self.num_stage > 1:
            for i in range(self.num_stage - 1):
                downsample_layer = nn.Sequential(
                    nn.BatchNorm1d(dims[i]),
                    nn.Conv1d(dims[i], dims[i + 1], kernel_size=downsample_ratio, stride=downsample_ratio),
                )
                self.downsample_layers.append(downsample_layer)
                self.downsample_layers[-1].register_forward_hook(
                    partial(self.hook_fn, layer_name=f"downsample_layer_{i + 1}"))

        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.downsample_ratio = downsample_ratio
        self.num_stage = len(num_blocks)
        self.stages = nn.ModuleList()
        for stage_idx in range(self.num_stage):
            layer = Stage(ffn_ratio, num_blocks[stage_idx], large_size[stage_idx], small_size[stage_idx],
                          dmodel=dims[stage_idx],
                          nvars=c_in, small_kernel_merged=small_kernel_merged, drop=backbone_dropout)
            self.stages.append(layer)
            # Only register hooks for Block (remove Stage-level hook)
            for block_idx, block in enumerate(layer.blocks):
                block.register_forward_hook(partial(self.hook_fn, layer_name=f"stage_{stage_idx}_block_{block_idx}"))

        # head (remove all head-related hooks)
        patch_num = seq_len // patch_stride
        self.n_vars = c_in
        self.individual = individual
        d_model = dims[self.num_stage - 1]

        if use_multi_scale:
            self.head_nf = d_model * patch_num
            self.head = Flatten_Head(self.individual, self.n_vars, self.head_nf, target_window,
                                     head_dropout=head_dropout)
        else:
            if patch_num % pow(downsample_ratio, (self.num_stage - 1)) == 0:
                self.head_nf = d_model * patch_num // pow(downsample_ratio, (self.num_stage - 1))
            else:
                self.head_nf = d_model * (patch_num // pow(downsample_ratio, (self.num_stage - 1)) + 1)

            self.head = Flatten_Head(self.individual, self.n_vars, self.head_nf, target_window,
                                     head_dropout=head_dropout)

        self.act_class = F.gelu
        self.class_dropout = nn.Dropout(self.class_drop)
        self.head_class = nn.Linear(c_in * d_model, self.class_num)

    # -------------------------- Added: Hook function (collects layer outputs) --------------------------
    def hook_fn(self, module, input, output, layer_name):
        """
        Forward hook: Only stores layer outputs when collect_activations=True
        - Automatically handles tuple outputs (e.g., series_decomp)
        - Converts outputs to numpy arrays and moves to CPU to avoid GPU memory leaks
        """
        if self.collect_activations:
            if isinstance(output, tuple):
                # Handle multi-output layers (e.g., series_decomp returns (res, moving_mean))
                self.activations[layer_name] = [o.detach().cpu().numpy() for o in output]
            else:
                self.activations[layer_name] = output.detach().cpu().numpy()

    def _calculate_max_layers(self, n_timesteps):
        """Calculate the maximum allowed number of pooling layers"""
        max_layers = 0
        current_T = n_timesteps
        while current_T >= 2:
            current_T = current_T // 2
            max_layers += 1
        return max_layers

    def forward_feature(self, x, te=None):
        B, M, L = x.shape

        x = x.unsqueeze(-2)

        for i in range(self.num_stage):
            B, M, D, N = x.shape
            x = x.reshape(B * M, D, N)
            if i == 0:
                if self.patch_size != self.patch_stride:
                    # stem layer padding
                    pad_len = self.patch_size - self.patch_stride
                    pad = x[:, :, -1:].repeat(1, 1, pad_len)
                    x = torch.cat([x, pad], dim=-1)
            else:
                if N % self.downsample_ratio != 0:
                    pad_len = self.downsample_ratio - (N % self.downsample_ratio)
                    x = torch.cat([x, x[:, :, -pad_len:]], dim=-1)
            x = self.downsample_layers[i](x)
            _, D_, N_ = x.shape
            x = x.reshape(B, M, D_, N_)
            x = self.stages[i](x)
        return x

    def forward(self, x):
        # Clear historical data before collection
        if self.collect_activations:
            self.activations = {}
        x = self.forward_feature(x, te=None)
        x = self.act_class(x)
        x = self.class_dropout(x)
        x = einops.reduce(x, 'b c d s -> b c d', reduction='mean')
        x = einops.rearrange(x, 'b c d -> b (c d)')
        x = self.head_class(x)
        return x

    def structural_reparam(self):
        for m in self.modules():
            if hasattr(m, 'merge_kernel'):
                m.merge_kernel()


@register_model("ModernTCN")
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        # hyper param
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.configs = configs

        self.downsample_ratio = configs.model.downsample_ratio
        self.ffn_ratio = configs.model.ffn_ratio
        self.num_blocks = configs.model.num_blocks
        self.large_size = configs.model.large_size
        self.small_size = configs.model.small_size
        self.dims = configs.model.dims
        self.channel_dict = np.load(configs.dataset.channel_dict_path, allow_pickle=True)
        self.channel_name = configs.dataset.channel_index_name

        if self.channel_name == 'all':
            self.selected_channels = list(range(configs.dataset.input_channels))
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

        self.nvars = len(self.selected_channels)

        self.small_kernel_merged = configs.model.small_kernel_merged
        self.drop_backbone = configs.model.dropout
        self.drop_head = configs.model.head_dropout
        self.use_multi_scale = configs.model.use_multi_scale
        self.revin = configs.model.revin
        self.affine = configs.model.affine
        self.subtract_last = configs.model.subtract_last

        self.seq_len = configs.dataset.seq_len[configs.dataset.task]
        self.c_in = self.nvars
        self.individual = configs.model.individual

        self.kernel_size = configs.model.kernel_size
        self.patch_size = configs.model.patch_size
        self.patch_stride = configs.model.patch_stride

        # classification
        self.class_dropout = configs.model.class_dropout
        self.class_num = configs.dataset.target_size[configs.dataset.task]

        self.model = ModernTCN(patch_size=self.patch_size, patch_stride=self.patch_stride,
                               downsample_ratio=self.downsample_ratio, ffn_ratio=self.ffn_ratio,
                               num_blocks=self.num_blocks,
                               large_size=self.large_size, small_size=self.small_size, dims=self.dims,
                               nvars=self.nvars, small_kernel_merged=self.small_kernel_merged,
                               backbone_dropout=self.drop_backbone, head_dropout=self.drop_head,
                               use_multi_scale=self.use_multi_scale, revin=self.revin, affine=self.affine,
                               subtract_last=self.subtract_last, seq_len=self.seq_len, c_in=self.c_in,
                               individual=self.individual,
                               class_drop=self.class_dropout, class_num=self.class_num)

        # -------------------------- Added: Interfaces to control layer output collection --------------------------
        self.collect_activations = False

    # Enable layer output collection
    def start_collect_activations(self):
        self.collect_activations = True
        self.model.collect_activations = True
        self.model.activations = {}  # Clear historical data

    # Disable layer output collection
    def stop_collect_activations(self):
        self.collect_activations = False
        self.model.collect_activations = False

    # Get collected layer outputs (returns a copy of the dictionary)
    def get_activations(self):
        return self.model.activations.copy()

    def apply_augmentation(self, x):
        shift_ratio = self.configs.augmentation.shift_ratio
        augmentations = [
            (lambda x: random_shift(x, shift_ratio), self.configs.augmentation.random_shift_prob),
            (add_noise(), self.configs.augmentation.add_noise_prob),
            (ChannelMasking(), self.configs.augmentation.ChannelMasking_prob),
            (TimeMasking(), self.configs.augmentation.TimeMasking_prob),
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
        if type(ieeg_raw_data) == dict:
            ieeg_raw_data, labels = ieeg_raw_data["ieeg_raw_data"], ieeg_raw_data["labels"]
        x = ieeg_raw_data[:, self.selected_channels]
        if labels is not None:
            if type(labels) != torch.LongTensor:
                labels = labels.long()
            one_hot_labels = F.one_hot(labels, num_classes=self.class_num).float()

        if self.training:
            x = self.apply_augmentation(x)
            p = random.random()
            if p < self.configs.augmentation.mixup_prob:
                x, one_hot_labels = Mixup(alpha=0.5)(x, one_hot_labels)
            x = x.float()
            prediction = self.model(x)
            loss1 = self.loss_fn(prediction, one_hot_labels)
            loss = loss1
            return {"loss"  : loss,
                    "labels": labels,
                    "logits": prediction}
        else:
            all_logits = []
            with torch.no_grad():
                x = x.float()
                for idx in range(self.configs.model.tta_times):
                    if idx == 0:
                        x_aug = x.clone()
                    else:
                        x_aug = self.apply_tta(x.clone())
                    prediction = self.model(x_aug)
                    all_logits.append(prediction)
            avg_logits = torch.mean(torch.stack(all_logits), dim=0)
            loss = self.loss_fn(avg_logits, one_hot_labels) if labels is not None else None

            return {
                "loss"  : loss,
                "labels": labels,
                "logits": avg_logits,
            }