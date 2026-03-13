import os
import pickle
import re
import pandas as pd
from torch import nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, trunc_normal_
from torch.utils.data import DataLoader
from pathlib import Path
import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from src.data_module.dataset.simple_dataset import SimpleDataset
import numpy as np
from omegaconf import DictConfig
from src.model_module.brain_decoder import ModelFactory
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from functools import partial
from typing import Dict, List
from src.utils.get_act import get_act

NORM_EPS = 1e-5

def filter_state_dict(state_dict):
    """
    Filter out keys in state_dict that end with .total_params and .total_ops
    """
    filtered_state_dict = {}
    for key, value in state_dict.items():
        if not (key.startswith('pool.') or key.startswith('fc.')):
            filtered_state_dict[key] = value
    return filtered_state_dict


def select_channel_for_subject(cfg, input, channel_name, channel_dict=None):
    auditory_cortex_channels = {
    'S5': {'initial': [13, 14, 15, 76, 77, 78, 79, 80], 'final': [14, 15, 77, 78, 79, 80]},
    'S6': {'initial': [117, 118, 122], 'final': [117, 118, 122]},
    'S7': {'initial': [83, 84, 85, 92, 93, 94], 'final': [83, 84, 85, 92, 93, 94]},
    'S11': {'initial': [10, 38], 'final': [12]},
    'S12': {'initial': [52, 53, 54, 55, 56, 57, 58], 'final': [53, 54, 55, 56, 57, 58]},
    }
    if "use_auditory_cortex_channels" not in cfg.dataset:
        use_auditory_cortex_channels = False
    else:
        use_auditory_cortex_channels = cfg.dataset.use_auditory_cortex_channels

    if use_auditory_cortex_channels:
        phoneme_type = cfg.dataset.task.split('_')[-1]
        selected_channels = auditory_cortex_channels[cfg.dataset.id][phoneme_type]
    else:
        if channel_name == 'all':
            return input, None
        elif isinstance(channel_name, int):
            return input[:, channel_name].unsqueeze(1), [channel_name]
        elif isinstance(channel_name, str) and len(channel_name) == 1:
            return input[:, channel_dict[channel_name]], channel_dict[channel_name]
        elif isinstance(channel_name, str) and len(channel_name) > 1:
            split_channels = channel_name.split('-')
            selected_channels = []
            for i in range(len(split_channels)):
                if split_channels[i].isalpha() or split_channels[i][0].isalpha():
                    selected_channels.append(channel_dict[split_channels[i]])
                else:
                    selected_channels.append(int(split_channels[i]))

    return input[:, selected_channels], selected_channels


def collate_fn(instances):
    ieeg_raw_data, labels = tuple([instance[key] for instance in instances]
                                 for key in ("ieeg_raw_data", "labels"))
    ieeg_raw_data = torch.stack(ieeg_raw_data)
    labels = torch.tensor(labels)

    batch = {
        "ieeg_raw_data": ieeg_raw_data,
        "labels"       : labels,
        "return_loss"  : True,
    }
    return batch


def merge_pre_bn(module, pre_bn_1, pre_bn_2=None):
    """ Merge pre BN to reduce inference runtime.
    """
    weight = module.weight.data
    if module.bias is None:
        zeros = torch.zeros(module.out_channels, device=weight.device).type(weight.type())
        module.bias = nn.Parameter(zeros)
    bias = module.bias.data
    if pre_bn_2 is None:
        assert pre_bn_1.track_running_stats is True, "Unsupport bn_module.track_running_stats is False"
        assert pre_bn_1.affine is True, "Unsupport bn_module.affine is False"

        scale_invstd = pre_bn_1.running_var.add(pre_bn_1.eps).pow(-0.5)
        extra_weight = scale_invstd * pre_bn_1.weight
        extra_bias = pre_bn_1.bias - pre_bn_1.weight * pre_bn_1.running_mean * scale_invstd
    else:
        assert pre_bn_1.track_running_stats is True, "Unsupport bn_module.track_running_stats is False"
        assert pre_bn_1.affine is True, "Unsupport bn_module.affine is False"

        assert pre_bn_2.track_running_stats is True, "Unsupport bn_module.track_running_stats is False"
        assert pre_bn_2.affine is True, "Unsupport bn_module.affine is False"

        scale_invstd_1 = pre_bn_1.running_var.add(pre_bn_1.eps).pow(-0.5)
        scale_invstd_2 = pre_bn_2.running_var.add(pre_bn_2.eps).pow(-0.5)

        extra_weight = scale_invstd_1 * pre_bn_1.weight * scale_invstd_2 * pre_bn_2.weight
        extra_bias = scale_invstd_2 * pre_bn_2.weight *(pre_bn_1.bias - pre_bn_1.weight * pre_bn_1.running_mean * scale_invstd_1 - pre_bn_2.running_mean) + pre_bn_2.bias

    if isinstance(module, nn.Linear):
        extra_bias = weight @ extra_bias
        weight.mul_(extra_weight.view(1, weight.size(1)).expand_as(weight))
    elif isinstance(module, nn.Conv2d):
        assert weight.shape[2] == 1 and weight.shape[3] == 1
        weight = weight.reshape(weight.shape[0], weight.shape[1])
        extra_bias = weight @ extra_bias
        weight.mul_(extra_weight.view(1, weight.size(1)).expand_as(weight))
        weight = weight.reshape(weight.shape[0], weight.shape[1], 1, 1)
    bias.add_(extra_bias)

    module.weight.data = weight
    module.bias.data = bias


class ConvBNAct(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            groups=1,
            act_type="ReLU"
    ):
        super(ConvBNAct, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                              padding=(kernel_size-1)//2, groups=groups, bias=False)
        self.norm = nn.BatchNorm2d(out_channels, eps=NORM_EPS)
        self.act = get_act(act_type)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class PatchEmbed(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1):
        super(PatchEmbed, self).__init__()
        norm_layer = partial(nn.BatchNorm2d, eps=NORM_EPS)
        if stride == 2:
            self.avgpool = nn.AvgPool2d((2, 2), stride=2, ceil_mode=True, count_include_pad=False)
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
            self.norm = norm_layer(out_channels)
        elif in_channels != out_channels:
            self.avgpool = nn.Identity()
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
            self.norm = norm_layer(out_channels)
        else:
            self.avgpool = nn.Identity()
            self.conv = nn.Identity()
            self.norm = nn.Identity()

    def forward(self, x):
        return self.norm(self.conv(self.avgpool(x)))


class MHCA(nn.Module):
    """
    Multi-Head Convolutional Attention
    """
    def __init__(self, out_channels, head_dim, act_type="ReLU", cnn_dropout=0):
        super(MHCA, self).__init__()
        norm_layer = partial(nn.BatchNorm2d, eps=NORM_EPS)
        self.group_conv3x3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                                       padding=1, groups=out_channels // head_dim, bias=False)
        self.norm = norm_layer(out_channels)
        self.act = get_act(act_type)
        self.dropout = nn.Dropout(p=cnn_dropout)
        self.projection = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.group_conv3x3(x)
        out = self.norm(out)
        out = self.act(out)
        out = self.dropout(out)
        out = self.projection(out)
        return out


class Mlp(nn.Module):
    def __init__(self, in_features, out_features=None, mlp_ratio=None, drop=0., bias=True, act_type="ReLU"):
        super().__init__()
        out_features = out_features or in_features
        hidden_dim = _make_divisible(in_features * mlp_ratio, 32)
        self.conv1 = nn.Conv2d(in_features, hidden_dim, kernel_size=1, bias=bias)
        self.act = get_act(act_type)
        self.conv2 = nn.Conv2d(hidden_dim, out_features, kernel_size=1, bias=bias)
        self.drop = nn.Dropout(drop)

    def merge_bn(self, pre_norm):
        merge_pre_bn(self.conv1, pre_norm)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.conv2(x)
        return x


class NCB(nn.Module):
    """
    Next Convolution Block
    """
    def __init__(self, in_channels, out_channels, stride=1, path_dropout=0.0, act_type="ReLU",
                 cnn_dropout=0.0, mlp_dropout=0.0, head_dim=32, mlp_ratio=3):
        super(NCB, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        norm_layer = partial(nn.BatchNorm2d, eps=NORM_EPS)
        assert out_channels % head_dim == 0

        self.patch_embed = PatchEmbed(in_channels, out_channels, stride)
        self.mhca = MHCA(out_channels, head_dim, act_type, cnn_dropout)
        self.cnn_path_dropout = DropPath(path_dropout)

        self.norm = norm_layer(out_channels)

        self.mlp = Mlp(out_channels, mlp_ratio=mlp_ratio, drop=mlp_dropout, bias=True, act_type=act_type)
        self.mlp_path_dropout = DropPath(path_dropout)
        self.is_bn_merged = False

    def merge_bn(self):
        if not self.is_bn_merged:
            self.mlp.merge_bn(self.norm)
            self.is_bn_merged = True

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.cnn_path_dropout(self.mhca(x))
        if not torch.onnx.is_in_onnx_export() and not self.is_bn_merged:
            out = self.norm(x)
        else:
            out = x
        x = x + self.mlp_path_dropout(self.mlp(out))
        return x


class NeuroSketchEncoder(nn.Module):
    def __init__(self, stem_chs, depths, act_type, path_dropout, attn_drop=0.0, cnn_dropout=0.0, mlp_dropout=0.0,
                 strides=[1, 2, 2, 2], sr_ratios=[8, 4, 2, 1], head_dim=32, mix_block_ratio=0.75, mlp_ratio=3,
                 use_checkpoint=False):
        super(NeuroSketchEncoder, self).__init__()
        self.use_checkpoint = use_checkpoint
        self.depths = depths
        self.stage_out_channels = [[96] * (depths[0] - 1) + [128],
                                   [192] * (depths[1] - 1) + [256],
                                   [384] * (depths[2] - 1) + [512],
                                   [768] * (depths[3] - 1) + [1024]]

        self.stage_block_types = [[NCB] * depths[0],
                                  [NCB] * (depths[1]),
                                  [NCB] * (depths[2]),
                                  [NCB] * (depths[3])]

        self.stem = nn.Sequential(
            ConvBNAct(1, stem_chs[0], kernel_size=3, stride=2, act_type=act_type),
            ConvBNAct(stem_chs[0], stem_chs[1], kernel_size=3, stride=1, act_type=act_type),
            ConvBNAct(stem_chs[1], stem_chs[2], kernel_size=3, stride=1, act_type=act_type),
            ConvBNAct(stem_chs[2], stem_chs[2], kernel_size=3, stride=2, act_type=act_type),
        )
        input_channel = stem_chs[-1]
        features = []
        idx = 0
        dpr = [x.item() for x in torch.linspace(0, path_dropout, sum(depths))]  # stochastic depth decay rule
        for stage_id in range(len(depths)):
            numrepeat = depths[stage_id]
            output_channels = self.stage_out_channels[stage_id]
            block_types = self.stage_block_types[stage_id]
            for block_id in range(numrepeat):
                if strides[stage_id] == 2 and block_id == 0:
                    stride = 2
                else:
                    stride = 1
                output_channel = output_channels[block_id]
                block_type = block_types[block_id]
                if block_type is NCB:
                    layer = NCB(input_channel,
                                output_channel,
                                stride=stride,
                                path_dropout=dpr[idx + block_id],
                                cnn_dropout=cnn_dropout,
                                mlp_dropout=mlp_dropout,
                                head_dim=head_dim,
                                mlp_ratio=mlp_ratio)
                    features.append(layer)
                else:
                    raise ValueError(f"Unknown block type: {block_type}")
                input_channel = output_channel
            idx += numrepeat
        self.features = nn.Sequential(*features)
        self.norm = nn.BatchNorm2d(output_channel, eps=NORM_EPS)

    def merge_bn(self):
        self.eval()
        for idx, module in self.named_modules():
            if isinstance(module, NCB):
                module.merge_bn()

    def forward_features(self, x):
        all_layer_outs = {}
        # Process each layer in stem (order exactly matches original stem)
        stem_x = x
        for stem_layer_id, stem_layer in enumerate(self.stem):
            stem_x = stem_layer(stem_x)
            all_layer_outs[f"stem_layer_{stem_layer_id}"] = stem_x
        stem_out = stem_x  # Preserve original stem final output

        # Process each layer in all stages (order exactly matches original features)
        stage_x = stem_out
        stage_outs = []  # Preserve original stage final outputs
        current_stage_id = 0
        current_stage_layer_count = 0

        for layer_idx, layer in enumerate(self.features):
            # Record current layer output (key contains stage and inner index)
            stage_id = current_stage_id
            stage_inner_idx = current_stage_layer_count
            all_layer_outs[f"stage{stage_id}_layer_{stage_inner_idx}"] = stage_x

            if self.use_checkpoint:
                stage_x = checkpoint.checkpoint(layer, stage_x)
            else:
                stage_x = layer(stage_x)

            # Update stage counter and record stage final output
            current_stage_layer_count += 1
            if current_stage_layer_count == self.depths[current_stage_id]:
                stage_outs.append(stage_x)
                current_stage_id += 1
                current_stage_layer_count = 0

        # Final norm output
        final_feat = self.norm(stage_x)

        return {
            "stem_out"      : stem_out,
            "stage_outs"    : stage_outs,
            "final_feat"    : final_feat,
            "all_layer_outs": all_layer_outs  # Still able to extract each layer output
        }

    def forward(self, x):
        x = self.forward_features(x)
        return x


class NeuroSketch(nn.Module):
    def __init__(self, cfg: DictConfig):
        super(NeuroSketch, self).__init__()
        self.cfg = cfg
        self.encoder = NeuroSketchEncoder(stem_chs=[64, 32, 64],
                                 depths=[4, 4, 4, 4],
                                 act_type=cfg.encoder.act_type,
                                 path_dropout=cfg.encoder.drop_path_rate,
                                 cnn_dropout=cfg.encoder.cnn_dropout,
                                 mlp_dropout=cfg.encoder.mlp_dropout,
                                 head_dim=cfg.encoder.head_dim,
                                 mlp_ratio=cfg.encoder.mlp_ratio)
    def extract_features(self, x):
        encoder_outs = self.encoder.forward_features(x)
        return encoder_outs


def inference_model_cnn_2d(device, ieeg, label, ckpt_path, batch_size):
    ieeg = np.swapaxes(ieeg, 0, 1)
    # Store outputs of all individual layers (key names: stem_layer_xx / stagex_layer_xx)
    all_layer_features_dict = {}
    # Load model configuration and weights
    checkpoint_dir = Path(ckpt_path).parent
    cfg_path = checkpoint_dir / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Configuration file {cfg_path} not found")
    cfg = OmegaConf.load(cfg_path)
    channel_dict = np.load(cfg.dataset.channel_dict_path, allow_pickle=True)
    channel_name = cfg.dataset.channel_index_name
    model = NeuroSketch(cfg).to(device)
    state_dict = torch.load(ckpt_path, map_location="cpu")
    state_dict = filter_state_dict(state_dict)
    model.load_state_dict(state_dict)
    model.eval()  # Inference mode

    # --------------- Key: Get all layer names (extracted from one model inference, ensuring consistency with model) ---------------
    # Run once with a dummy mini-batch to get all layer key names
    dummy_x = torch.randn(1, 1, 3, 3).to(device)  # Minimum size dummy input
    with torch.no_grad():
        dummy_outs = model.extract_features(dummy_x)
        all_layer_names = list(dummy_outs["all_layer_outs"].keys())  # List of all layer names
    # Initialize feature storage list for each layer (corresponding to layer names)
    layer_features = {layer_name: [] for layer_name in all_layer_names}

    # Build dataset (keep original logic)
    dataset = [
        {
            'ieeg_raw_data': torch.tensor(d, dtype=torch.float32),
            'labels': torch.tensor(l, dtype=torch.long),
        }
        for d, l in zip(ieeg, label)
    ]

    # Data loader (keep sample order)
    loader = DataLoader(
        SimpleDataset(dataset),
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=False,
    )

    # Iterate through batches for processing (input processing logic remains unchanged)
    for batch in tqdm(loader, desc="Processing batches (per layer)"):
        x = batch['ieeg_raw_data'].to(device)  # Input shape: [B, C, L]
        x, _ = select_channel_for_subject(cfg, x, channel_name, channel_dict)

        bs, ch, l = x.size()

        # Adjust input length (pad to multiple of 3)
        remainder = l % 3
        l_new = l + (3 - remainder) if remainder != 0 else l
        if l_new != l:
            x = F.pad(x, (0, l_new - l))

        # Input shape transformation (adapt to model 2D convolution input)
        x = x.view(bs, ch, l_new // 3, 3)
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(bs, ch * 3, l_new // 3)
        x = torch.unsqueeze(x, dim=1)  # Final shape: [B, 1, H, W]

        # Get all model intermediate features (key: all_layer_outs)
        with torch.no_grad():  # Disable gradient calculation for faster inference
            intermediate_outs = model.extract_features(x)
            layer_outs = intermediate_outs["all_layer_outs"]  # Dictionary of all individual layer outputs

        # --------------- Collect each layer's output (split by sample, maintain order)---------------
        for layer_name in all_layer_names:
            # Current layer's batch output: [B, C, H, W]
            batch_out = layer_outs[layer_name].detach().cpu().numpy()
            # Split by sample dimension, add one by one to corresponding list (ensure consistent order)
            for b in range(bs):
                layer_features[layer_name].append(batch_out[b])

    # --------------- Convert to numpy arrays, organize into output dictionary ---------------
    for layer_name in all_layer_names:
        all_layer_features_dict[layer_name] = np.array(layer_features[layer_name])

    return all_layer_features_dict


def inference_model_cnn_1d(
        device: torch.device,
        ieeg: np.ndarray,
        label: np.ndarray,
        ckpt_path: str,
        batch_size: int = 32
) -> Dict[str, np.ndarray]:
    """
    Perform 1D CNN model inference and collect output features from all layers
    Added: 4D tensor automatically merges first and second dimensions to 3D tensor (B, M*D, N)

    Args:
        device: Inference device (cpu/cuda)
        ieeg: Input data, shape=[N, L, C] (N samples, L time steps, C channels)
        label: Label data, shape=[N]
        ckpt_path: Model weight path
        batch_size: Batch size

    Returns:
        all_layer_features_dict: Layer feature dictionary, key=layer name, value=feature array (N, ...)
    """
    ieeg = np.swapaxes(ieeg, 0, 1)
    # -------------------------- Load model configuration and weights --------------------------
    checkpoint_dir = Path(ckpt_path).parent
    cfg_path = checkpoint_dir / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Configuration file {cfg_path} not found")
    cfg = OmegaConf.load(cfg_path)
    cfg.model.tta_times = 1
    # Initialize model
    model = ModelFactory(cfg.model.name)(cfg).to(device)
    state_dict = torch.load(ckpt_path, map_location="cpu")
    state_dict = filter_state_dict(state_dict)
    model.load_state_dict(state_dict)
    model.eval()  # Inference mode

    # -------------------------- Build dataset and dataloader --------------------------
    dataset = [
        {
            'ieeg_raw_data': torch.tensor(d, dtype=torch.float32),
            'labels': torch.tensor(l, dtype=torch.long),
        }
        for d, l in zip(ieeg, label)
    ]

    loader = DataLoader(
        SimpleDataset(dataset),
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        pin_memory=True if device.type == 'cuda' else False
    )

    # -------------------------- Initialize layer feature storage --------------------------
    all_layer_features_dict = {}
    layer_features: Dict[str, List[np.ndarray]] = {}  # Temporarily store layer outputs for each sample
    all_layer_names = None  # Lazy initialization of layer names

    # -------------------------- Batch inference and feature collection --------------------------
    with torch.no_grad():
        for batch in tqdm(loader, desc="Processing batches (collecting layer features)"):
            x = batch['ieeg_raw_data'].to(device)
            model.start_collect_activations()
            _ = model(x)

            # 4. Get layer outputs for current batch
            batch_activations = model.get_activations()
            model.stop_collect_activations()

            # 5. Initialize layer names (first collection)
            if all_layer_names is None:
                all_layer_names = list(batch_activations.keys())
                for layer_name in all_layer_names:
                    layer_features[layer_name] = []

            # 6. Split and store layer outputs by sample (added 4D tensor processing logic)
            for layer_name in all_layer_names:
                layer_out = batch_activations[layer_name]
                # ========== Core modification: Handle 4D tensors ==========
                # Case 1: Layer output is 4D numpy array (B, M, D, N)
                if isinstance(layer_out, np.ndarray) and len(layer_out.shape) == 4:
                    B, M, D, N = layer_out.shape
                    # Merge first and second dimensions → (B, M*D, N)
                    layer_out = layer_out.reshape(B * M, D, N)
                # Case 2: Layer output is tuple containing 4D tensors (e.g., series_decomp)
                elif isinstance(layer_out, tuple):
                    new_layer_out = []
                    for out in layer_out:
                        if isinstance(out, np.ndarray) and len(out.shape) == 4:
                            B, M, D, N = out.shape
                            # Merge first and second dimensions → (B, M*D, N)
                            out = out.reshape(B * M, D, N)
                        new_layer_out.append(out)
                    layer_out = tuple(new_layer_out)
                # ========================================
                bs = layer_out.shape[0]
                # Split by sample
                if isinstance(layer_out, tuple):
                    # Split each element in the tuple by sample
                    layer_out_split = [
                        tuple(out[b] for out in layer_out)
                        for b in range(bs)
                    ]
                else:
                    # Directly split by sample dimension (layer_out.shape: [B, ...])
                    layer_out_split = [layer_out[b] for b in range(bs)]

                # Add to temporary list
                layer_features[layer_name].extend(layer_out_split)

    # -------------------------- Organize into final feature dictionary --------------------------
    for layer_name in all_layer_names:
        # Convert to numpy array (handle special case of tuple outputs)
        if isinstance(layer_features[layer_name][0], tuple):
            tuple_len = len(layer_features[layer_name][0])
            all_layer_features_dict[layer_name] = tuple(
                np.stack([item[i] for item in layer_features[layer_name]])
                for i in range(tuple_len)
            )
        else:
            # Directly stack all samples
            all_layer_features_dict[layer_name] = np.stack(layer_features[layer_name])
    return all_layer_features_dict


def inference_model_gru(
        device: torch.device,
        ieeg: np.ndarray,
        label: np.ndarray,
        ckpt_path: str,
        batch_size: int = 32
) -> Dict[str, np.ndarray]:

    ieeg = np.swapaxes(ieeg, 0, 1)
    # -------------------------- Load model configuration and weights --------------------------
    checkpoint_dir = Path(ckpt_path).parent
    cfg_path = checkpoint_dir / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Configuration file {cfg_path} not found")
    cfg = OmegaConf.load(cfg_path)
    cfg.model.tta_times = 1
    # Initialize model
    model = ModelFactory(cfg.model.name)(cfg).to(device)
    state_dict = torch.load(ckpt_path, map_location="cpu")
    state_dict = filter_state_dict(state_dict)
    model.load_state_dict(state_dict)
    model.eval()  # Inference mode

    # -------------------------- Build dataset and dataloader --------------------------
    dataset = [
        {
            'ieeg_raw_data': torch.tensor(d, dtype=torch.float32),
            'labels': torch.tensor(l, dtype=torch.long),
        }
        for d, l in zip(ieeg, label)
    ]

    loader = DataLoader(
        SimpleDataset(dataset),
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        pin_memory=True if device.type == 'cuda' else False
    )

    # -------------------------- Initialize layer feature storage --------------------------
    all_layer_features_dict = {}
    layer_features: Dict[str, List[np.ndarray]] = {}  # Temporarily store layer outputs for each sample
    all_layer_names = None  # Lazy initialization of layer names

    # -------------------------- Batch inference and feature collection --------------------------
    with torch.no_grad():
        for batch in tqdm(loader, desc="Processing batches (collecting layer features)"):
            x = batch['ieeg_raw_data'].to(device)
            model.start_collect_activations()
            _ = model(x)

            # 4. Get layer outputs for current batch
            batch_activations = model.get_activations()
            model.stop_collect_activations()

            # 5. Initialize layer names (first collection)
            if all_layer_names is None:
                all_layer_names = list(batch_activations.keys())
                for layer_name in all_layer_names:
                    layer_features[layer_name] = []

            # 6. Split and store layer outputs by sample (add 4D tensor processing logic)
            for layer_name in all_layer_names:
                layer_out = batch_activations[layer_name]
                layer_features[layer_name].extend(layer_out)

    # -------------------------- Organize into final feature dictionary --------------------------
    for layer_name in all_layer_names:
        # Convert to numpy arrays (handle special case of tuple outputs)
        if isinstance(layer_features[layer_name][0], tuple):
            tuple_len = len(layer_features[layer_name][0])
            all_layer_features_dict[layer_name] = tuple(
                np.stack([item[i] for item in layer_features[layer_name]])
                for i in range(tuple_len)
            )
        else:
            # Directly stack all samples
            all_layer_features_dict[layer_name] = np.stack(layer_features[layer_name])
    return all_layer_features_dict


def inference_model_transformer(device, ieeg, label, ckpt_path, batch_size):
    ieeg = np.swapaxes(ieeg, 0, 1)
    embedding_results_dict = {}
    attention_results_dict = {}
    # Add: Dictionary to store each layer's Attention output vectors
    attention_output_results_dict = {}

    # Load model configuration and weights
    checkpoint_dir = Path(ckpt_path).parent
    cfg_path = checkpoint_dir / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Configuration file {cfg_path} does not exist")

    cfg = OmegaConf.load(cfg_path)
    cfg.model.output_attention = True  # Force enable attention output
    channel_dict = np.load(cfg.dataset.channel_dict_path, allow_pickle=True)
    channel_name = cfg.dataset.channel_index_name
    model = ModelFactory(cfg.model.name)(cfg).to(device)
    state_dict = torch.load(ckpt_path, map_location="cpu")
    # state_dict = filter_state_dict(state_dict)
    model.load_state_dict(state_dict)
    model.eval()

    # Get key parameters
    num_encoder_layers = cfg.model.e_layers  # Number of encoder layers
    total_samples = ieeg.shape[0]  # Total number of samples
    patch_len_list = eval(cfg.model.patch_len_list)
    num_patches = len(patch_len_list)  # Number of patches (corresponding to intra-attention count)
    patch_names = [f"patch_{patch_len}" for patch_len in patch_len_list]
    # Determine if inter-attention exists (based on model configuration)
    per_layer_attn_count = num_patches + 1

    # Initialize storage structure
    sample_attention = [
        [[] for _ in range(num_encoder_layers)]
        for __ in range(num_patches)
    ]
    sample_embedding = [[] for _ in range(num_patches)]
    sample_attention_output = [
        [[] for _ in range(per_layer_attn_count)]
        for __ in range(num_encoder_layers)
    ]

    # Build dataset and dataloader
    sentence_dataset = [
        {
            'ieeg_raw_data': torch.tensor(d, dtype=torch.float32),
            'labels'       : torch.tensor(l, dtype=torch.long),
        }
        for d, l in zip(ieeg, label)
    ]
    loader = torch.utils.data.DataLoader(
        SimpleDataset(sentence_dataset),
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=False,  # Ensure sample order consistency
    )

    # Inference: Save embedding vectors, attention matrices, and each layer's Attention output vectors simultaneously
    for batch in tqdm(loader):
        inputs = batch['ieeg_raw_data'].to(device)  # [B, C, L]
        x, _ = select_channel_for_subject(cfg, inputs, channel_name, channel_dict)
        with torch.no_grad():
            x = x.float()
            if x.shape[-1] % 2 == 1:
                x = x[:, :, :-1]

            x = x.permute(0, 2, 1)

            # 1. Get embedding vectors (original logic)
            enc_out = model.enc_embedding(x)  # List[Tensor], length=num_patches, each element [B, seq_len, d_model]
            for patch_idx in range(num_patches):
                patch_emb = enc_out[patch_idx].detach().cpu().numpy()  # [B, seq_len, d_model]
                for b in range(patch_emb.shape[0]):
                    sample_embedding[patch_idx].append(patch_emb[b])  # Single sample: [seq_len, d_model]

            _, attns, all_x_out_history = model.encoder(enc_out, attn_mask=None)

            batch_size = x.shape[0]
            for e_layer in range(num_encoder_layers):
                layer_attention_outputs = all_x_out_history[e_layer]  # List[Tensor], length=per_layer_attn_count
                for attn_idx in range(per_layer_attn_count):
                    attn_output = layer_attention_outputs[attn_idx].detach().cpu().numpy()
                    for b in range(batch_size):
                        sample_attention_output[e_layer][attn_idx].append(attn_output[b])  # Single sample vector

            for e_layer in range(num_encoder_layers):
                e_layer_attn = attns[e_layer]  # List[Tensor], length=per_layer_attn_count
                for patch_idx in range(num_patches):
                    attn = e_layer_attn[patch_idx].detach().cpu().numpy()  # [B, n_heads, seq1, seq2]
                    attn_avg_heads = attn.mean(axis=1)  # Average multi-heads: [B, seq1, seq2]
                    for b in range(batch_size):
                        sample_attention[patch_idx][e_layer].append(attn_avg_heads[b])

    for patch_idx in range(num_patches):
        assert len(sample_embedding[patch_idx]) == total_samples, \
            f"patch_{patch_names[patch_idx]} embedding vector sample count mismatch: {len(sample_embedding[patch_idx])} vs {total_samples}"

    for patch_idx in range(num_patches):
        for e_layer in range(num_encoder_layers):
            assert len(sample_attention[patch_idx][e_layer]) == total_samples, \
                f"patch_{patch_names[patch_idx]} layer {e_layer} attention sample count mismatch"

    for e_layer in range(num_encoder_layers):
        for attn_idx in range(per_layer_attn_count):
            assert len(sample_attention_output[e_layer][attn_idx]) == total_samples, \
                f"layer {e_layer} attention {attn_idx} output vector sample count mismatch"

    for patch_idx in range(num_patches):
        sample_embedding[patch_idx] = np.array(sample_embedding[patch_idx])  # [total_samples, seq_len, d_model]
        embedding_results_dict[f"{patch_names[patch_idx]}_embedding"] = sample_embedding[patch_idx]

    for patch_idx in range(num_patches):
        for e_layer in range(num_encoder_layers):
            sample_attention[patch_idx][e_layer] = np.array(sample_attention[patch_idx][e_layer])  # [total_samples, seq1, seq2]
        attention_results_dict[f"{patch_names[patch_idx]}_attention"] = sample_attention[patch_idx]

    for e_layer in range(num_encoder_layers):
        for attn_idx in range(per_layer_attn_count):
            sample_attention_output[e_layer][attn_idx] = np.array(sample_attention_output[e_layer][attn_idx])
            if attn_idx < num_patches:
                key = f"layer_{e_layer}_{patch_names[attn_idx]}_intra_attention_output"
            else:
                key = f"layer_{e_layer}_inter_attention_output"
            attention_output_results_dict[key] = sample_attention_output[e_layer][attn_idx]

    return embedding_results_dict, attention_results_dict, attention_output_results_dict


def analyze_attention_embedding_svd(
        embedding_dict=None,
        attn_matrix_dict=None,
        attention_output_dict=None,
        tolerance=1e-5,
        use_gpu=True
):
    """
    Low-memory version Transformer rank analysis (fix list to array conversion issue)
    """

    # Helper function: Ensure data is numpy array
    def to_numpy_array(data):
        if isinstance(data, list):
            return np.array(data, dtype=np.float32)
        elif isinstance(data, torch.Tensor):
            return data.cpu().numpy().astype(np.float32)
        elif isinstance(data, np.ndarray):
            return data.astype(np.float32)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

    # Helper function: Count rank distribution
    def get_rank_distribution(rank_array):
        if len(rank_array) == 0:
            return {}
        counts = np.bincount(rank_array)
        return {int(rank): int(count) for rank, count in enumerate(counts) if count > 0}

    # Helper function: Parse attention_output_dict keys
    def parse_attention_output_key(key):
        parts = key.split("_")
        e_layer = int(parts[1])
        if "inter" in key:
            return {"type": "inter", "e_layer": e_layer, "patch_name": None}
        else:
            patch_name = "_".join(parts[2:4])
            return {"type": "intra", "e_layer": e_layer, "patch_name": patch_name}

    # Core: Batch SVD function (only returns singular values, saves memory)
    def batch_svd_get_singular_values(mat_batch):
        """
        Batch SVD: Only returns singular values S, does not store U/V
        """
        # Ensure input is numpy array and convert to float32
        mat_batch = to_numpy_array(mat_batch)

        # Convert to torch tensor (float32)
        mat_tensor = torch.from_numpy(mat_batch)
        if use_gpu and torch.cuda.is_available():
            mat_tensor = mat_tensor.cuda()

        # Only compute singular values (more efficient)
        S = torch.linalg.svdvals(mat_tensor)

        # Convert back to numpy float32
        S_np = S.cpu().numpy().astype(np.float32) if use_gpu else S.numpy().astype(np.float32)
        return S_np

    # Helper: Batch compute effective ranks
    def compute_batch_ranks(S_batch, tolerance):
        """
        Batch compute 95% variance rank and ε-rank (float32)
        """
        S_batch = S_batch.astype(np.float32)

        # Singular value ratios
        s_max = S_batch[:, 0:1]  # [B, 1]
        s_ratios = np.where(s_max >= 1e-10, S_batch / s_max, 0).astype(np.float32)

        # 95% variance rank
        s_squared = (S_batch ** 2).astype(np.float32)
        total_var = np.sum(s_squared, axis=1, keepdims=True)
        total_var = np.where(total_var < 1e-10, 1, total_var)
        cumulative_var = (np.cumsum(s_squared, axis=1) / total_var).astype(np.float32)
        rank_var95 = np.argmax(cumulative_var >= 0.95, axis=1) + 1
        all_below = np.all(cumulative_var < 0.95, axis=1)
        rank_var95[all_below] = S_batch.shape[1]

        # ε-rank
        rank_epsilon = np.sum(s_ratios >= tolerance, axis=1)

        return s_ratios, rank_var95, rank_epsilon

    analysis_results = {}
    if not attn_matrix_dict:
        raise ValueError("attn_matrix_dict cannot be empty")
    patch_names = [name.replace("_attention", "") for name in attn_matrix_dict.keys()]

    for patch_name in patch_names:
        print(f"\nProcessing {patch_name} ...")
        analysis_results[patch_name] = {
            "attention": {}, "embedding": {}, "attention_output": {}, "statistics": {}
        }
        attn_key = f"{patch_name}_attention"
        emb_key = f"{patch_name}_embedding"

        # -------------------------- 1. Process Attention matrices (only keep singular values and ranks)--------------------------
        # Fix: Convert to numpy array first
        attn_data = to_numpy_array(attn_matrix_dict[attn_key])
        num_layers, n_samples_attn, seq1, seq2 = attn_data.shape
        attn_results = {
            "singular_values"     : [], "singular_value_ratios": [],
            "effective_rank_var95": [], "effective_rank_epsilon": [],
            "num_layers"          : num_layers, "n_samples": n_samples_attn
        }

        for layer_idx in tqdm(range(num_layers), desc=f"{patch_name} Attention Layers"):
            attn_layer = attn_data[layer_idx]  # [n_samples, seq1, seq2]
            S_batch = batch_svd_get_singular_values(attn_layer)
            s_ratios, rank_var95, rank_epsilon = compute_batch_ranks(S_batch, tolerance)

            # Expand results list (only store necessary data)
            attn_results["singular_values"].extend(S_batch)
            attn_results["singular_value_ratios"].extend(s_ratios)
            attn_results["effective_rank_var95"].extend(rank_var95)
            attn_results["effective_rank_epsilon"].extend(rank_epsilon)

        # Convert to numpy arrays (float32, only store necessary data)
        for key in attn_results.keys():
            if key == "singular_values" or key == "singular_value_ratios":
                attn_results[key] = np.array(attn_results[key], dtype=np.float32)
            elif key not in ["num_layers", "n_samples"]:
                attn_results[key] = np.array(attn_results[key], dtype=int)
        analysis_results[patch_name]["attention"] = attn_results

        # -------------------------- 2. Process Embedding vectors (only keep singular values and ranks)--------------------------
        if embedding_dict and emb_key in embedding_dict:
            # Fix: Convert to numpy array first
            emb_data = to_numpy_array(embedding_dict[emb_key])
            n_samples_emb, seq_len, emb_dim = emb_data.shape
            S_batch = batch_svd_get_singular_values(emb_data)
            s_ratios, rank_var95, rank_epsilon = compute_batch_ranks(S_batch, tolerance)

            emb_results = {
                "singular_values"       : S_batch.astype(np.float32),
                "singular_value_ratios" : s_ratios.astype(np.float32),
                "effective_rank_var95"  : rank_var95,
                "effective_rank_epsilon": rank_epsilon,
                "num_layers"            : 0, "n_samples": n_samples_emb,
                "embedding_dim"         : emb_dim
            }
            analysis_results[patch_name]["embedding"] = emb_results
        else:
            analysis_results[patch_name]["embedding"] = None

        # -------------------------- 3. Process Attention output vectors (only keep singular values and ranks)--------------------------
        attn_output_results = {}
        if attention_output_dict:
            relevant_keys = []
            for key in attention_output_dict.keys():
                parsed = parse_attention_output_key(key)
                if (parsed["type"] == "intra" and parsed["patch_name"] == patch_name) or parsed["type"] == "inter":
                    relevant_keys.append(key)

            for key in relevant_keys:
                parsed = parse_attention_output_key(key)
                # Fix: Convert to numpy array first
                output_data = to_numpy_array(attention_output_dict[key])
                n_samples_out, seq_len_out, d_model = output_data.shape
                S_batch = batch_svd_get_singular_values(output_data)
                s_ratios, rank_var95, rank_epsilon = compute_batch_ranks(S_batch, tolerance)

                key_results = {
                    "singular_values"       : S_batch.astype(np.float32),
                    "singular_value_ratios" : s_ratios.astype(np.float32),
                    "effective_rank_var95"  : rank_var95,
                    "effective_rank_epsilon": rank_epsilon,
                    "e_layer"               : parsed["e_layer"],
                    "type"                  : parsed["type"],
                    "n_samples"             : n_samples_out,
                    "d_model"               : d_model
                }
                attn_output_results[key] = key_results
        analysis_results[patch_name]["attention_output"] = attn_output_results

        # -------------------------- 4. Rank distribution statistics (keep original logic)--------------------------
        stats = {"attention": {}, "embedding": None, "attention_output": {}}

        # 4.1 Attention matrix statistics
        attn_rank_var95 = np.array(attn_results["effective_rank_var95"]).reshape(num_layers, n_samples_attn)
        attn_rank_epsilon = np.array(attn_results["effective_rank_epsilon"]).reshape(num_layers, n_samples_attn)
        for layer_idx in range(num_layers):
            stats["attention"][layer_idx] = {
                "var95_rank_distribution"  : get_rank_distribution(attn_rank_var95[layer_idx]),
                "epsilon_rank_distribution": get_rank_distribution(attn_rank_epsilon[layer_idx])
            }

        # 4.2 Embedding statistics
        if embedding_dict and emb_key in embedding_dict:
            stats["embedding"] = {
                "var95_rank_distribution"  : get_rank_distribution(emb_results["effective_rank_var95"]),
                "epsilon_rank_distribution": get_rank_distribution(emb_results["effective_rank_epsilon"])
            }

        # 4.3 Attention output statistics
        for key, output_results in attn_output_results.items():
            parsed = parse_attention_output_key(key)
            stats_key = f"layer_{parsed['e_layer']}_{parsed['type']}"
            stats["attention_output"][stats_key] = {
                "var95_rank_distribution"  : get_rank_distribution(output_results["effective_rank_var95"]),
                "epsilon_rank_distribution": get_rank_distribution(output_results["effective_rank_epsilon"])
            }

        analysis_results[patch_name]["statistics"] = stats

    return analysis_results


def analyze_cnn2d_svd(layer_features_dict, tolerance=1e-3, use_gpu=True, entropy_alpha=1.0):
    """
    Low-memory version CNN layer feature SVD analysis (added Matrix-Based Entropy)
    """
    # -------------------------- Helper function definitions --------------------------
    # 2. Batch SVD (returns only singular values, low memory)
    def batch_svd_get_singular_values(feat_data):
        B, C, H, W = feat_data.shape
        mat_batch = feat_data.reshape(B, C, H * W).astype(np.float32)
        mat_tensor = torch.from_numpy(mat_batch)
        if use_gpu and torch.cuda.is_available():
            mat_tensor = mat_tensor.cuda()
        S = torch.linalg.svdvals(mat_tensor)  # Only compute singular values (efficient, saves memory)
        return S.cpu().numpy().astype(np.float32) if use_gpu else S.numpy().astype(np.float32)

    # 3. Batch compute SVD-derived metrics
    def compute_batch_metrics(S_batch, tolerance):
        s_max = S_batch[:, 0:1]
        s_ratios_batch = np.where(s_max >= 1e-10, S_batch / s_max, 0).astype(np.float32)

        s_squared_batch = (S_batch ** 2).astype(np.float32)
        total_var_batch = np.sum(s_squared_batch, axis=1, keepdims=True)
        total_var_batch = np.where(total_var_batch < 1e-10, 1, total_var_batch)
        cumulative_var_batch = (np.cumsum(s_squared_batch, axis=1) / total_var_batch).astype(np.float32)

        rank_var95_batch = np.argmax(cumulative_var_batch >= 0.95, axis=1) + 1
        rank_var95_batch[np.all(cumulative_var_batch < 0.95, axis=1)] = S_batch.shape[1]
        rank_epsilon_batch = np.sum(s_ratios_batch >= tolerance, axis=1)

        return s_ratios_batch, rank_var95_batch, rank_epsilon_batch

    # 4. Core addition: Compute Matrix-Based Entropy (paper formula 1)
    def compute_matrix_based_entropy(Z, alpha=entropy_alpha):
        """
        Z: [N, C] feature matrix (all batches concatenated, H/W averaged)
        alpha: Order of entropy (default 1.0 → Shannon entropy, commonly used in papers)
        Formula: S_α(Z) = (1/(1-α)) * log(Σ(λ_i^α)), λ_i = eigenvalue / trace(Gram matrix)
        """
        N, C = Z.shape
        if N == 0 or C == 0:
            return 0.0

        # Compute Gram matrix K = Z^T @ Z ([C,C]), float32 saves memory
        Z_tensor = torch.from_numpy(Z).float()
        if use_gpu and torch.cuda.is_available():
            Z_tensor = Z_tensor.cuda()
        K = Z_tensor.T @ Z_tensor  # Symmetric matrix, more efficient computation

        # Extract eigenvalues (non-negative), use eigvalsh for symmetric matrices
        eigenvalues = torch.linalg.eigvalsh(K).cpu().numpy().astype(np.float32)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]  # Filter out minimal values to avoid numerical errors

        # Normalize eigenvalues (λ_i = λ_i / tr(K))
        tr_K = np.sum(eigenvalues)
        if tr_K < 1e-10:
            return 0.0
        lambda_norm = eigenvalues / tr_K

        # Compute entropy (split into alpha=1 and other cases)
        if np.isclose(alpha, 1.0):
            # Simplified to Shannon entropy when alpha→1: -Σ(λ·logλ)
            entropy = -np.sum(lambda_norm * np.log(lambda_norm))
        else:
            # General formula
            lambda_alpha = np.power(lambda_norm, alpha)
            sum_alpha = np.sum(lambda_alpha)
            entropy = (1 / (1 - alpha)) * np.log(sum_alpha)

        return float(entropy)

    # -------------------------- Main logic --------------------------
    analysis_results = {}
    if not layer_features_dict:
        raise ValueError("layer_features_dict cannot be empty")

    for layer_name in tqdm(layer_features_dict.keys(), desc="Processing Layers"):
        analysis_results[layer_name] = {"svd_results": {}, "statistics": {}}

        # Parse layer information
        if layer_name.startswith("stem_layer_"):
            layer_info = {"type"           : "stem", "stem_layer_idx": int(layer_name.split("_")[-1]),
                          "stage_idx"      : None, "stage_inner_idx": None}
        elif layer_name.startswith("stage") and "_layer_" in layer_name:
            parts = layer_name.split("_")
            layer_info = {"type"           : "stage", "stem_layer_idx": None, "stage_idx": int(parts[0][-1]),
                          "stage_inner_idx": int(parts[-1])}
        else:
            raise ValueError(f"Unsupported layer name format: {layer_name}")

        # 1. Load features and perform batch SVD
        feat_data = layer_features_dict[layer_name].astype(np.float32)
        B, C, H, W = feat_data.shape
        S_batch = batch_svd_get_singular_values(feat_data)

        # 2. Compute SVD-derived metrics
        s_ratios_batch, rank_var95_batch, rank_epsilon_batch = compute_batch_metrics(S_batch, tolerance)

        # 3. Addition: Process H/W averaging + concat all batches → [N,C]
        avg_feat = feat_data.mean(axis=(2, 3))  # [B, C], average over H/W dimensions
        layer_total_feat = avg_feat  # Directly use if input contains all batches; if batched, use np.concatenate([...], axis=0)

        # 4. Addition: Compute Matrix-Based Entropy
        matrix_entropy = compute_matrix_based_entropy(layer_total_feat, alpha=entropy_alpha)

        # 5. Store results
        svd_results = {
            "singular_values"       : S_batch,
            "singular_value_ratios" : s_ratios_batch,
            "matrix_entropy"        : matrix_entropy,
            "effective_rank_var95"  : rank_var95_batch,
            "effective_rank_epsilon": rank_epsilon_batch,
            "n_samples"             : B,
            "channels"              : C,
            "height"                : H,
            "width"                 : W
        }
        analysis_results[layer_name]["svd_results"] = svd_results

        # Statistics (including added entropy value)
        stats = {
            "layer_info"               : layer_info
        }
        analysis_results[layer_name]["statistics"] = stats

    return analysis_results


def analyze_cnn1d_svd(layer_features_dict, tolerance=1e-3, use_gpu=True, entropy_alpha=1.0):
    """
    Low-memory version CNN 1D layer feature SVD analysis (adapted for [B,C,L] sequence features, added Matrix-Based Entropy)
    """
    # -------------------------- Helper function definitions --------------------------
    # 2. Batch SVD (returns only singular values, low memory, adapted for 1D sequences)
    def batch_svd_get_singular_values(feat_data):
        B, C, L = feat_data.shape  # 1D features: Batch × Channels × Seq_Length
        mat_batch = feat_data.reshape(B, C, L).astype(np.float32)  # No need to flatten H*W for 1D
        mat_tensor = torch.from_numpy(mat_batch)
        if use_gpu and torch.cuda.is_available():
            mat_tensor = mat_tensor.cuda()
        S = torch.linalg.svdvals(mat_tensor)  # Compute only singular values (efficient and memory-saving)
        return S.cpu().numpy().astype(np.float32) if use_gpu else S.numpy().astype(np.float32)

    # 3. Batch compute SVD-derived metrics (logic unchanged)
    def compute_batch_metrics(S_batch, tolerance):
        s_max = S_batch[:, 0:1]
        s_ratios_batch = np.where(s_max >= 1e-10, S_batch / s_max, 0).astype(np.float32)

        s_squared_batch = (S_batch ** 2).astype(np.float32)
        total_var_batch = np.sum(s_squared_batch, axis=1, keepdims=True)
        total_var_batch = np.where(total_var_batch < 1e-10, 1, total_var_batch)
        cumulative_var_batch = (np.cumsum(s_squared_batch, axis=1) / total_var_batch).astype(np.float32)

        rank_var95_batch = np.argmax(cumulative_var_batch >= 0.95, axis=1) + 1
        rank_var95_batch[np.all(cumulative_var_batch < 0.95, axis=1)] = S_batch.shape[1]
        rank_epsilon_batch = np.sum(s_ratios_batch >= tolerance, axis=1)

        return s_ratios_batch, rank_var95_batch, rank_epsilon_batch

    # 4. Core addition: Compute Matrix-Based Entropy (logic unchanged)
    def compute_matrix_based_entropy(Z, alpha=entropy_alpha):
        """
        Z: [N, C] feature matrix (all batches concatenated, sequence dimension L averaged)
        alpha: Order of entropy (default 1.0→Shannon entropy, commonly used in papers)
        """
        N, C = Z.shape
        if N == 0 or C == 0:
            return 0.0

        # Compute Gram matrix K = Z^T @ Z ([C,C]), float32 for memory efficiency
        Z_tensor = torch.from_numpy(Z).float()
        if use_gpu and torch.cuda.is_available():
            Z_tensor = Z_tensor.cuda()
        K = Z_tensor.T @ Z_tensor  # Symmetric matrix, more efficient computation

        # Extract eigenvalues (non-negative), use eigvalsh for symmetric matrix
        eigenvalues = torch.linalg.eigvalsh(K).cpu().numpy().astype(np.float32)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]  # Filter out extremely small values to avoid numerical errors

        # Normalize eigenvalues (λ_i = λ_i / tr(K))
        tr_K = np.sum(eigenvalues)
        if tr_K < 1e-10:
            return 0.0
        lambda_norm = eigenvalues / tr_K

        # Compute entropy (for alpha=1 and other cases)
        if np.isclose(alpha, 1.0):
            # When alpha→1, simplifies to Shannon entropy: -Σ(λ·logλ)
            entropy = -np.sum(lambda_norm * np.log(lambda_norm))
        else:
            # General formula
            lambda_alpha = np.power(lambda_norm, alpha)
            sum_alpha = np.sum(lambda_alpha)
            entropy = (1 / (1 - alpha)) * np.log(sum_alpha)

        return float(entropy)

    # -------------------------- Main logic --------------------------
    analysis_results = {}
    if not layer_features_dict:
        raise ValueError("layer_features_dict cannot be empty")

    for layer_name in tqdm(layer_features_dict.keys(), desc="Processing Layers (CNN 1D)"):
        analysis_results[layer_name] = {"svd_results": {}, "statistics": {}}

        # Parse layer information (adapted for new 1D layer name format)
        if layer_name == "stem":
            layer_info = {"type": "stem", "stem_layer_idx": 0, "stage_idx": None, "block_idx": None}
        elif layer_name.startswith("downsample_layer_"):
            layer_idx = int(layer_name.split("_")[-1])
            layer_info = {"type": "downsample", "downsample_layer_idx": layer_idx, "stage_idx": None, "block_idx": None}
        elif layer_name.startswith("stage_") and "_block_" in layer_name:
            parts = layer_name.split("_")
            stage_idx = int(parts[1])
            block_idx = int(parts[3])
            layer_info = {"type": "stage_block", "stage_idx": stage_idx, "block_idx": block_idx}
        else:
            # Compatible with other layer names, no error
            layer_info = {"type": "unknown", "layer_name": layer_name}

        # 1. Load features and process (compatible with tuple-type outputs)
        feat_data = layer_features_dict[layer_name].astype(np.float32)
        # Handle tuple-type outputs (e.g., (res, moving_mean) returned by series_decomp)
        if isinstance(feat_data, tuple):
            feat_data = feat_data[0]  # Take the first element (residual) for analysis
        B, C, L = feat_data.shape  # 1D feature shape: Batch × Channels × Seq_Length

        # Skip layers with empty features or abnormal dimensions
        if B == 0 or C == 0 or L == 0:
            print(f"Warning: {layer_name} has abnormal feature dimensions, skipping")
            continue

        # 2. Batch SVD
        S_batch = batch_svd_get_singular_values(feat_data)

        # 3. Compute SVD-derived metrics
        s_ratios_batch, rank_var95_batch, rank_epsilon_batch = compute_batch_metrics(S_batch, tolerance)

        # 4. Process sequence dimension averaging → [N,C]
        avg_feat = feat_data.mean(axis=-1)  # Average over 1D sequence dimension L (axis=2)
        layer_total_feat = avg_feat  # Directly use if input contains all batches

        # 5. Compute Matrix-Based Entropy
        matrix_entropy = compute_matrix_based_entropy(layer_total_feat, alpha=entropy_alpha)

        # 6. Store results (adapted for 1D features, replace height/width with seq_length)
        svd_results = {
            "singular_values": S_batch,
            "singular_value_ratios": s_ratios_batch,
            "matrix_entropy": matrix_entropy,
            "effective_rank_var95": rank_var95_batch,
            "effective_rank_epsilon": rank_epsilon_batch,
            "n_samples": B,
            "channels": C,
            "seq_length": L  # 1D feature: sequence length (replace height/width in 2D)
        }
        analysis_results[layer_name]["svd_results"] = svd_results

        # Statistics (including layer type parsing results)
        stats = {
            "layer_info": layer_info
        }
        analysis_results[layer_name]["statistics"] = stats

    return analysis_results


def analyze_gru_svd(layer_features_dict, tolerance=1e-3, use_gpu=True, entropy_alpha=1.0):
    """
    Low-memory version GRU layer feature SVD analysis (adapted for [B,C,L] sequence features, added Matrix-Based Entropy)
    """
    # -------------------------- Helper function definitions --------------------------
    # 2. Batch SVD (returns only singular values, low memory, adapted for 1D sequences)
    def batch_svd_get_singular_values(feat_data):
        B, L, C = feat_data.shape  # 1D features: Batch × Channels × Seq_Length
        mat_batch = feat_data.reshape(B, C, L).astype(np.float32)  # No need to flatten H*W for 1D
        mat_tensor = torch.from_numpy(mat_batch)
        if use_gpu and torch.cuda.is_available():
            mat_tensor = mat_tensor.cuda()
        S = torch.linalg.svdvals(mat_tensor)  # Compute only singular values (efficient and memory-saving)
        return S.cpu().numpy().astype(np.float32) if use_gpu else S.numpy().astype(np.float32)

    # 3. Batch compute SVD-derived metrics (logic unchanged)
    def compute_batch_metrics(S_batch, tolerance):
        s_max = S_batch[:, 0:1]
        s_ratios_batch = np.where(s_max >= 1e-10, S_batch / s_max, 0).astype(np.float32)

        s_squared_batch = (S_batch ** 2).astype(np.float32)
        total_var_batch = np.sum(s_squared_batch, axis=1, keepdims=True)
        total_var_batch = np.where(total_var_batch < 1e-10, 1, total_var_batch)
        cumulative_var_batch = (np.cumsum(s_squared_batch, axis=1) / total_var_batch).astype(np.float32)

        rank_var95_batch = np.argmax(cumulative_var_batch >= 0.95, axis=1) + 1
        rank_var95_batch[np.all(cumulative_var_batch < 0.95, axis=1)] = S_batch.shape[1]
        rank_epsilon_batch = np.sum(s_ratios_batch >= tolerance, axis=1)

        return s_ratios_batch, rank_var95_batch, rank_epsilon_batch

    # 4. Core addition: Compute Matrix-Based Entropy (logic unchanged)
    def compute_matrix_based_entropy(Z, alpha=entropy_alpha):
        """
        Z: [N, C] feature matrix (all batches concatenated, sequence dimension L averaged)
        alpha: Order of entropy (default 1.0→Shannon entropy, commonly used in papers)
        """
        N, C = Z.shape
        if N == 0 or C == 0:
            return 0.0

        # Compute Gram matrix K = Z^T @ Z ([C,C]), float32 for memory efficiency
        Z_tensor = torch.from_numpy(Z).float()
        if use_gpu and torch.cuda.is_available():
            Z_tensor = Z_tensor.cuda()
        K = Z_tensor.T @ Z_tensor  # Symmetric matrix, more efficient computation

        # Extract eigenvalues (non-negative), use eigvalsh for symmetric matrix
        eigenvalues = torch.linalg.eigvalsh(K).cpu().numpy().astype(np.float32)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]  # Filter out extremely small values to avoid numerical errors

        # Normalize eigenvalues (λ_i = λ_i / tr(K))
        tr_K = np.sum(eigenvalues)
        if tr_K < 1e-10:
            return 0.0
        lambda_norm = eigenvalues / tr_K

        # Compute entropy (for alpha=1 and other cases)
        if np.isclose(alpha, 1.0):
            # When alpha→1, simplifies to Shannon entropy: -Σ(λ·logλ)
            entropy = -np.sum(lambda_norm * np.log(lambda_norm))
        else:
            # General formula
            lambda_alpha = np.power(lambda_norm, alpha)
            sum_alpha = np.sum(lambda_alpha)
            entropy = (1 / (1 - alpha)) * np.log(sum_alpha)

        return float(entropy)

    # -------------------------- Main logic --------------------------
    analysis_results = {}
    if not layer_features_dict:
        raise ValueError("layer_features_dict cannot be empty")

    for layer_name in tqdm(layer_features_dict.keys(), desc="Processing Layers (GRU)"):
        analysis_results[layer_name] = {"svd_results": {}, "statistics": {}}
        # Parse layer information
        if layer_name == "embedding":
            layer_info = {
                "type"               : "embedding",
                "embedding_layer_idx": 0,
                "stage_idx"          : None,
                "block_idx"          : None,
                "sub_layer_type"     : None
            }
        elif re.match(r"res_bigru_\d+_(gru|fc)", layer_name):
            # Match formats like res_bigru_0_gru, res_bigru_1_fc
            parts = layer_name.split("_")
            block_idx = int(parts[2])  # Extract index after res_bigru
            sub_layer_type = parts[3]  # Extract sub-layer type (gru/fc)
            layer_info = {
                "type"               : "residual_bigru",
                "embedding_layer_idx": None,
                "stage_idx"          : None,
                "block_idx"          : block_idx,
                "sub_layer_type"     : sub_layer_type
            }
        else:
            # Compatible with other possible layer names (e.g., output layer, etc.)
            layer_info = {
                "type"               : "unknown",
                "embedding_layer_idx": None,
                "stage_idx"          : None,
                "block_idx"          : None,
                "sub_layer_type"     : None,
                "layer_name"         : layer_name
            }

        # 1. Load features and process (compatible with tuple-type outputs)
        feat_data = layer_features_dict[layer_name].astype(np.float32)
        # Handle tuple-type outputs (e.g., (res, moving_mean) returned by series_decomp)
        if isinstance(feat_data, tuple):
            feat_data = feat_data[0]  # Take the first element (residual) for analysis
        B, L, C = feat_data.shape

        # 2. Batch SVD
        S_batch = batch_svd_get_singular_values(feat_data)

        # 3. Compute SVD-derived metrics
        s_ratios_batch, rank_var95_batch, rank_epsilon_batch = compute_batch_metrics(S_batch, tolerance)

        # 4. Process sequence dimension averaging → [N,C]
        avg_feat = feat_data.mean(axis=1)  # Average over 1D sequence dimension L (axis=2)
        layer_total_feat = avg_feat  # Directly use if input contains all batches

        # 5. Compute Matrix-Based Entropy
        matrix_entropy = compute_matrix_based_entropy(layer_total_feat, alpha=entropy_alpha)

        # 6. Store results (adapted for 1D features, replace height/width with seq_length)
        svd_results = {
            "singular_values": S_batch,
            "singular_value_ratios": s_ratios_batch,
            "matrix_entropy": matrix_entropy,
            "effective_rank_var95": rank_var95_batch,
            "effective_rank_epsilon": rank_epsilon_batch,
            "n_samples": B,
            "channels": C,
            "seq_length": L
        }
        analysis_results[layer_name]["svd_results"] = svd_results

        # Statistics (including layer type parsing results)
        stats = {
            "layer_info": layer_info
        }
        analysis_results[layer_name]["statistics"] = stats

    return analysis_results


def plot_attention_output_effective_rank_ratio(
        analysis_dict_list,
        name_list,
        save_path=None
):
    """
    Transformer Attention output effective rank ratio comparison chart: Each patch subplot shows different analysis results as lines of "ε=10^-1.5 rank / maximum rank"
    Core: Embedding as layer 0, followed by encoder layer intra-attention, multiple lines distinguish different analysis results
    Added: Save plotting data as CSV files (in the same directory as images)
    """
    # -------------------------- 1. Basic configuration --------------------------
    target_epsilon = 10 ** (-1.5)  # Keep only this ε value
    target_epsilon_label = r'$\varepsilon=10^{-1.5}$'

    # Style configuration for multiple analysis results (colors/line styles/markers)
    colors = plt.cm.Set1(np.linspace(0.2, 0.8, len(analysis_dict_list)))
    line_styles = ['-', '--', '-.', ':'] * (len(analysis_dict_list) // 4 + 1)
    line_widths = [2] * len(analysis_dict_list)
    markers = ['o', 's', '^', 'D', 'v'] * (len(analysis_dict_list) // 5 + 1)

    # Validate input consistency
    if not analysis_dict_list or len(analysis_dict_list) != len(name_list):
        raise ValueError("analysis_dict_list and name_list must have the same length")

    # Get unified patch names (based on the first analysis result)
    base_patch_names = list(analysis_dict_list[0].keys())
    for i, analysis_dict in enumerate(analysis_dict_list):
        current_patch_names = list(analysis_dict.keys())
        if current_patch_names != base_patch_names:
            raise ValueError(f"The patch structure of the {i + 1}th analysis result is inconsistent with the first one!")

    n_patches = len(base_patch_names)
    if n_patches == 0:
        raise ValueError("No valid patch data in the analysis results")

    # -------------------------- New: Initialize data storage list --------------------------
    plot_data_list = []

    # Global plot configuration
    plt.rcParams.update({
        'font.size'        : 11,
        'axes.linewidth'   : 1.2,
        'grid.linewidth'   : 0.6,
        'legend.frameon'   : True,
        'legend.framealpha': 0.9,
        'legend.edgecolor' : 'black',
        'legend.fontsize'  : 10
    })
    fig, axes = plt.subplots(1, n_patches, figsize=(6 * n_patches, 7))

    # -------------------------- 2. Process each patch and draw multi-line plots --------------------------
    for patch_idx, patch_name in enumerate(base_patch_names):
        print(f"Processing patch: {patch_name} ...")
        ax = axes[patch_idx] if n_patches > 1 else axes

        # Store ratio data for all analysis results under current patch
        all_ratio_data = []
        all_layer_labels = None  # Unified layer labels (based on the first analysis result)

        # Loop through each analysis result and calculate ratio data
        for analysis_idx, analysis_dict in enumerate(analysis_dict_list):
            patch_data = analysis_dict[patch_name]
            attn_output_data = patch_data["attention_output"]
            emb_data = patch_data.get("embedding")

            # Only retain intra-attention data for the current patch
            intra_keys = [k for k in (attn_output_data.keys() if attn_output_data else [])
                          if "intra" in k and patch_name in k]
            has_attn_data = len(intra_keys) > 0
            has_emb_data = emb_data is not None and len(emb_data.get("singular_value_ratios", [])) > 0

            if not has_attn_data and not has_emb_data:
                all_ratio_data.append(None)
                continue

            # -------------------------- Process intra-attention data --------------------------
            attention_ratios = []
            valid_attn_layers = []
            if has_attn_data:
                output_info = {}
                for key in intra_keys:
                    parts = key.split("_")
                    e_layer = int(parts[1])
                    output_info[e_layer] = attn_output_data[key]
                valid_attn_layers = sorted(output_info.keys())

                # Calculate attention layer ratios
                for e_layer in valid_attn_layers:
                    data = output_info[e_layer]
                    all_ratios = data["singular_value_ratios"]

                    # Maximum rank
                    valid_sample = next((r for r in all_ratios if isinstance(r, np.ndarray) and len(r) > 0), None)
                    max_rank_attn = len(valid_sample) if valid_sample is not None else 1

                    # Calculate the average ratio for current ε
                    sample_ranks = []
                    for ratios in all_ratios:
                        if isinstance(ratios, np.ndarray) and len(ratios) > 0:
                            rank = np.sum(ratios >= target_epsilon)
                            sample_ranks.append(rank / max_rank_attn)
                    avg_ratio = np.mean(sample_ranks) if sample_ranks else np.nan
                    attention_ratios.append(avg_ratio)

            # -------------------------- Process embedding data --------------------------
            emb_ratio = np.nan
            has_valid_emb = False
            if has_emb_data:
                all_ratios_emb = emb_data["singular_value_ratios"]
                valid_sample_emb = next((r for r in all_ratios_emb if isinstance(r, np.ndarray) and len(r) > 0), None)
                max_rank_emb = len(valid_sample_emb) if valid_sample_emb is not None else 1

                sample_ranks = []
                for ratios in all_ratios_emb:
                    if isinstance(ratios, np.ndarray) and len(ratios) > 0:
                        rank = np.sum(ratios >= target_epsilon)
                        sample_ranks.append(rank / max_rank_emb)
                if sample_ranks:
                    emb_ratio = np.mean(sample_ranks)
                    has_valid_emb = True

            # -------------------------- Merge embedding and attention layers --------------------------
            final_ratios = []
            final_layer_labels = []

            # Add embedding layer
            if has_valid_emb:
                final_ratios.append(emb_ratio)
                final_layer_labels.append("embedding")

            # Add attention layers
            final_ratios.extend(attention_ratios)
            final_layer_labels.extend([f"Layer {l + 1}" for l in valid_attn_layers])

            # Filter layers without valid data
            valid_mask = ~np.isnan(final_ratios)
            final_ratios = [final_ratios[i] for i, mask in enumerate(valid_mask) if mask]
            final_layer_labels_filtered = [final_layer_labels[i] for i, mask in enumerate(valid_mask) if mask]

            # Record unified layer labels (based on the first analysis result)
            if analysis_idx == 0:
                all_layer_labels = final_layer_labels_filtered

            # Align layer labels (ensure consistent layer order across all analysis results)
            if all_layer_labels:
                aligned_ratios = []
                for label in all_layer_labels:
                    if label in final_layer_labels_filtered:
                        idx = final_layer_labels_filtered.index(label)
                        aligned_ratios.append(final_ratios[idx])
                    else:
                        aligned_ratios.append(np.nan)
                all_ratio_data.append(aligned_ratios)
            else:
                all_ratio_data.append(None)

        # -------------------------- Draw multi-line plots in the current patch subplot --------------------------
        if all_layer_labels and any(data is not None for data in all_ratio_data):
            x = np.arange(len(all_layer_labels))

            for analysis_idx, (ratios, name) in enumerate(zip(all_ratio_data, name_list)):
                if ratios is None:
                    continue

                valid_mask = ~np.isnan(ratios)
                if not np.any(valid_mask):
                    continue

                ax.plot(
                    x[valid_mask], [ratios[i] for i, mask in enumerate(valid_mask) if mask],
                    label=name,
                    color=colors[analysis_idx],
                    linestyle=line_styles[analysis_idx],
                    linewidth=line_widths[analysis_idx],
                    marker=markers[analysis_idx],
                    markersize=4,
                    alpha=0.8
                )

            # Subplot style configuration
            ax.set_title(f'Patch: {patch_name}', fontsize=14, pad=15)
            ax.set_xlabel('Layer', fontsize=13)
            if patch_idx == 0:
                ax.set_ylabel(f'Effective Rank Ratio ({target_epsilon_label} / Max Rank)', fontsize=13)
            ax.set_xticks(x)
            ax.set_xticklabels(all_layer_labels, rotation=45, ha='right', fontsize=10)
            ax.set_ylim(0, 1.05)
            ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, linewidth=1)
            ax.grid(True, which='major', alpha=0.3, linestyle='-')
            ax.tick_params(axis='both', which='major', labelsize=11, width=1.2, length=6)

            # Legend (display only in the first patch, or in every patch)
            if patch_idx == 0:
                ax.legend(loc='upper right', frameon=True, framealpha=0.9)
            else:
                ax.legend(loc='upper right', frameon=True, framealpha=0.9)

        # -------------------------- New: Collect plotting data for current patch --------------------------
        if all_layer_labels:
            for layer_idx, layer_label in enumerate(all_layer_labels):
                data_row = {
                    "patch_name": patch_name,
                    "layer_label": layer_label
                }
                # Add ratio for each analysis result
                for analysis_idx, name in enumerate(name_list):
                    if all_ratio_data[analysis_idx] is not None and layer_idx < len(all_ratio_data[analysis_idx]):
                        ratio_val = all_ratio_data[analysis_idx][layer_idx]
                        data_row[f"{name}_ratio"] = ratio_val if not np.isnan(ratio_val) else None
                    else:
                        data_row[f"{name}_ratio"] = None
                plot_data_list.append(data_row)

    # -------------------------- Layout adjustment and output --------------------------
    plt.subplots_adjust(
        bottom=0.18,
        wspace=0.4,
        top=0.9
    )
    plt.tight_layout()

    if save_path:
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        plt.savefig(
            save_path,
            dpi=500,
            format='png',
            bbox_inches='tight',
            pad_inches=0.2,
            facecolor='white',
            edgecolor='none'
        )
        print(f"High-resolution PNG image saved to: {save_path}")

        # -------------------------- New: Save plotting data as CSV --------------------------
        if plot_data_list:
            # Convert to DataFrame
            data_df = pd.DataFrame(plot_data_list)
            # Generate data save path (same directory as image, suffix _data.csv)
            data_save_path = os.path.splitext(save_path)[0] + "_data.csv"
            # Save CSV file
            data_df.to_csv(data_save_path, index=False, encoding='utf-8')
            print(f"Plotting data saved to: {data_save_path}")

    else:
        plt.show()
    plt.close()


def plot_model_layers_effective_rank_ratio(
        analysis_dict_list,
        name_list,
        save_path=None
):
    # -------------------------- 1. Basic configuration --------------------------
    target_epsilon = 10 ** (-1.5)  # Keep only this ε value
    target_epsilon_label = r'$\varepsilon=10^{-1.5}$'

    # Color configuration: Assign different colors to different analysis results (can be extended)
    colors = plt.cm.Set1(np.linspace(0.2, 0.8, len(analysis_dict_list)))
    line_styles = ['-', '--', '-.', ':'] * (len(analysis_dict_list) // 4 + 1)
    line_widths = [2] * len(analysis_dict_list)
    markers = ['o', 's', '^', 'D', 'v'] * (len(analysis_dict_list) // 5 + 1)

    # -------------------------- 2. Get unified Layer ordering (based on the first analysis result) --------------------------
    if not analysis_dict_list:
        raise ValueError("analysis_dict_list cannot be empty")

    # Get layer order from the first analysis result (assuming consistent layer structure across all analysis results)
    base_layer_names = list(analysis_dict_list[0].keys())

    # Verify layer consistency across all analysis results
    for i, analysis_dict in enumerate(analysis_dict_list):
        current_layer_names = list(analysis_dict.keys())
        if current_layer_names != base_layer_names:
            print(f"Layer structure of the {i + 1}th analysis result: {current_layer_names}")
            print(f"Layer structure of the first analysis result: {base_layer_names}")
            raise ValueError(f"The layer structure of the {i + 1}th analysis result is inconsistent with the first one!")

    valid_layer_names = []
    # Preprocess layer information (extracted from the first analysis result)
    for layer_name in base_layer_names:
        layer_data = analysis_dict_list[0][layer_name]
        svd_results = layer_data["svd_results"]
        # Check if there is valid singular value data
        ratios_list = []
        for ratios in svd_results["singular_value_ratios"]:
            if isinstance(ratios, np.ndarray) and len(ratios) > 0:
                ratios_list.append(ratios)
        if not ratios_list:
            print(f"Warning: {layer_name} has no valid singular value data, skipping")
            continue

        valid_layer_names.append(layer_name)

    if not valid_layer_names:
        raise ValueError("No valid layer data for plotting")

    # -------------------------- 3. Global plotting configuration --------------------------
    plt.rcParams.update({
        'font.size'          : 11,
        'axes.linewidth'     : 1.2,
        'grid.linewidth'     : 0.6,
        'legend.frameon'     : True,
        'legend.framealpha'  : 0.9,
        'legend.edgecolor'   : 'black',
        'legend.fontsize'    : 10,
        'legend.labelspacing': 0.5
    })
    # Core modification: Create 2 rows × 1 column subplots, sharing X-axis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    fig.suptitle('Layer-wise Effective Rank Ratio & Matrix-Based Entropy', fontsize=16, fontweight='bold', y=0.98)

    # -------------------------- 4. Calculate ratio data for each analysis result --------------------------
    all_ratio_data = []
    all_entropy_data = []
    for analysis_dict in analysis_dict_list:
        layer_ratios = []
        layer_entropies = []
        for layer_idx, layer_name in enumerate(valid_layer_names):
            layer_data = analysis_dict[layer_name]
            svd_results = layer_data["svd_results"]
            ratios_list = []

            for ratios in svd_results["singular_value_ratios"]:
                if isinstance(ratios, np.ndarray) and len(ratios) > 0:
                    ratios_list.append(ratios.astype(np.float32))

            # Calculate average ε-rank for current layer
            sample_ranks_ratio = [np.sum(ratios >= target_epsilon) / len(ratios) for ratios in ratios_list]
            layer_ratios.append(np.mean(sample_ranks_ratio))
            # Note: If entropy is stored in statistics, change to layer_data["statistics"]["matrix_based_entropy"]
            layer_entropies.append(svd_results.get("matrix_entropy", 0.0))

        all_ratio_data.append(layer_ratios)
        all_entropy_data.append(layer_entropies)

    # -------------------------- 5. Draw upper layer: Effective Rank Ratio --------------------------
    x = np.arange(len(valid_layer_names))
    for i, (ratios, name) in enumerate(zip(all_ratio_data, name_list)):
        ax1.plot(
            x, ratios,
            label=name,
            color=colors[i],
            linestyle=line_styles[i],
            linewidth=line_widths[i],
            marker=markers[i],
            markersize=4,
            alpha=0.8
        )
    ax1.set_ylabel(f'Effective Rank Ratio ({target_epsilon_label} / Max Rank)', fontsize=13)
    ax1.set_ylim(0, 1.05)
    ax1.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, linewidth=1)
    ax1.grid(True, which='major', alpha=0.3, linestyle='-')
    ax1.tick_params(axis='both', which='major', labelsize=11, width=1.2, length=6)
    ax1.legend(
        loc='upper right',
        ncol=min(2, len(name_list)),
        frameon=True,
        framealpha=0.9,
        fancybox=False,
        shadow=False
    )
    ax1.set_title('Effective Rank Ratio', fontsize=14, pad=10)

    # -------------------------- 6. Draw lower layer: Matrix-Based Entropy --------------------------
    for i, (entropies, name) in enumerate(zip(all_entropy_data, name_list)):
        ax2.plot(
            x, entropies,
            color=colors[i],
            linestyle=line_styles[i],
            linewidth=line_widths[i],
            marker=markers[i],
            markersize=4,
            alpha=0.8
        )
    ax2.set_ylabel('Matrix-Based Entropy', fontsize=13)
    ax2.grid(True, which='major', alpha=0.3, linestyle='-')
    ax2.tick_params(axis='both', which='major', labelsize=11, width=1.2, length=6)
    ax2.set_title('Matrix-Based Entropy', fontsize=14, pad=10)

    # -------------------------- 7. Shared X-axis configuration --------------------------
    ax2.set_xticks(x)
    ax2.set_xticklabels(valid_layer_names, rotation=45, ha='right', fontsize=10)
    ax2.set_xlabel('CNN Layer', fontsize=13)

    # -------------------------- 8. Layout adjustment and output --------------------------
    plt.subplots_adjust(bottom=0.12, top=0.93, hspace=0.15)  # hspace controls subplot spacing

    if save_path:
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Save the image
        plt.savefig(
            save_path,
            dpi=500,
            format='png',
            bbox_inches='tight',
            pad_inches=0.3,
            facecolor='white',
            edgecolor='none'
        )
        print(f"高分辨率PNG图已保存至：{save_path}")

        # -------------------------- New: Save plotting data (including entropy values) --------------------------
        data_dict = {
            "layer_name": valid_layer_names,
        }
        # Add ratio and entropy data for each analysis result
        for name, ratios, entropies in zip(name_list, all_ratio_data, all_entropy_data):
            data_dict[f"{name}_ratio"] = ratios
            data_dict[f"{name}_entropy"] = entropies

        # Convert to DataFrame
        data_df = pd.DataFrame(data_dict)

        # Generate data save path (same directory as image, suffix _data.csv)
        data_save_path = os.path.splitext(save_path)[0] + "_data.csv"
        # Save CSV file
        data_df.to_csv(data_save_path, index=False, encoding='utf-8')
        print(f"绘图数据已保存至：{data_save_path}")

    else:
        plt.show()
    plt.close()


def visualize_attention_heatmap(all_attention, suffix, output_dir, figsize=(15, 10)):
    """
    Generate attention heatmaps using matplotlib (static images, suitable for papers/reports)
    Each patch generates a 2x3 subplot grid (corresponding to 6 encoder layers), saved as high-resolution images

    Args:
        all_attention (dict): Attention matrix dictionary, key=patch name, value=(e_layer, seq1, seq2)
        suffix (str): File name suffix (e.g., "MedFormer_{exp_category}_initial_attention_map")
        output_dir (str/Path): Output directory path
        figsize (tuple): Overall image size (width, height)
        dpi (int): Image resolution (default 300, suitable for printing/papers)
    """
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Define patch names and corresponding titles
    patch_info = {
        "patch_5_attention" : "Patch Size = 5",
        "patch_10_attention": "Patch Size = 10",
        "patch_20_attention": "Patch Size = 20",
    }

    # Iterate through each patch to generate heatmaps
    for patch_name, attention_matrix in all_attention.items():
        attention_matrix = np.array(attention_matrix)
        average_attention = np.mean(attention_matrix, axis=1)
        num_encoder_layers = average_attention.shape[0]
        # Verify encoder layer count (2 rows × 3 columns → requires 6 layers, adjust rows/cols if layer count differs)
        if num_encoder_layers != 6:
            print(f"警告：{patch_name} 有 {num_encoder_layers} 个 encoder 层，默认按 2行×3列 绘制（多余层将被忽略）")

        # Create 2 rows × 3 columns subplot layout
        fig, axes = plt.subplots(
            nrows=2, ncols=3,
            figsize=figsize,
            sharex=True,  # All subplots share x-axis ticks
            sharey=True,  # All subplots share y-axis ticks
            constrained_layout=True  # Automatically adjust subplot spacing to avoid overlap
        )
        # Set overall figure title
        fig.suptitle(
            f"Attention Heatmaps - {patch_info[patch_name]}",
            fontsize=16, y=1
        )
        # Flatten axes for easy iteration (2 rows × 3 columns → list of 6 elements)
        axes_flat = axes.flatten()

        # Define color mapping (unify color range for all subplots for easy comparison)
        # vmin = average_attention.min()  # Minimum attention weight across all layers
        vmax = average_attention.max()  # Maximum attention weight across all layers
        vmin = 0  # Minimum attention weight across all layers
        # Iterate through each encoder layer and draw heatmap
        for layer_idx in range(min(6, num_encoder_layers)):
            ax = axes_flat[layer_idx]
            # Get attention matrix for current layer (seq1 × seq2)
            attn_layer = average_attention[layer_idx]

            # Draw heatmap
            im = ax.imshow(
                attn_layer,
                cmap="viridis",  # Color map (can use "plasma", "cividis", "RdBu_r", etc.)
                vmin=vmin, vmax=vmax,  # Unified color range
                aspect="auto"  # Automatically adjust aspect ratio
            )

            # Set subplot title (indicating layer number)
            ax.set_title(f"Encoder Layer {layer_idx + 1}", fontsize=12, fontweight="bold")

            # Set axis labels (only show on leftmost column and bottom row to avoid duplication)
            if layer_idx % 3 == 0:  # Show y-axis label for leftmost column subplots
                ax.set_ylabel("Sequence Position (seq1)", fontsize=10)
            if layer_idx >= 3:  # Show x-axis label for bottom row subplots
                ax.set_xlabel("Sequence Position (seq2)", fontsize=10)

            # Adjust axis tick font size
            ax.tick_params(axis='both', which='major', labelsize=8)

        plt.subplots_adjust(
            bottom=0.12,
            wspace=0.4,
            top=0.9  # Core: Reduce top value to reserve more vertical space for title (default ~0.95, changed to 0.9)
        )
        # Add global colorbar (shared by all subplots to avoid duplication)
        cbar = fig.colorbar(
            im,  # Based on color range of the last heatmap
            ax=axes,  # Colorbar associated with all subplots
            shrink=0.8,  # Colorbar scaling ratio
            aspect=30,  # Colorbar aspect ratio
            pad=0.02  # Spacing between colorbar and subplots
        )
        cbar.set_label("Attention Weight", fontsize=12)
        cbar.ax.tick_params(labelsize=10)  # Colorbar tick font size

        # Save image (supports png/jpg/pdf formats, automatically recognized by extension)
        save_path = output_dir / f"{patch_name}_{suffix}.svg"
        plt.savefig(
            save_path,
            bbox_inches="tight",  # Crop excess whitespace
            facecolor="white",  # Background color white (avoid transparency)
            edgecolor="none"
        )
        plt.close(fig)  # Close figure to free memory
        print(f"已保存 {patch_name} 热力图至：{save_path}")


@hydra.main(config_path="../conf/", config_name="analyze", version_base="1.2")
def get_attention_analyze_dict(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("---------------------Start-------------------------")
    exp_category = cfg.rank.state
    if cfg.rank.data_source == "character":
        processed_dir = f"{cfg.dir.data_dir}/{cfg.dataset.id}/processed_data/character"
        test_input_dataset_path = Path(processed_dir) / f'data_{exp_category}.npy'
        test_input_dataset = np.load(test_input_dataset_path, allow_pickle=True)
        initial_test_label_dataset_path = Path(processed_dir) / f'initial_label_{exp_category}.npy'
        initial_test_label_dataset = np.load(initial_test_label_dataset_path, allow_pickle=True)
        final_test_label_dataset_path = Path(processed_dir) / f'final_label_{exp_category}.npy'
        final_test_label_dataset = np.load(final_test_label_dataset_path, allow_pickle=True)
    elif cfg.rank.data_source == "sentence":
        processed_dir = f"{cfg.dir.data_dir}/{cfg.dataset.id}/processed_data/sentence"
        test_input_dataset_path = Path(processed_dir) / f'data_{exp_category}.pkl'
        test_input_dataset = np.load(test_input_dataset_path, allow_pickle=True)
        initial_test_label_dataset_path = Path(processed_dir) / f'initial_label_{exp_category}.pkl'
        initial_test_label_dataset = np.load(initial_test_label_dataset_path, allow_pickle=True)
        final_test_label_dataset_path = Path(processed_dir) / f'final_label_{exp_category}.pkl'
        final_test_label_dataset = np.load(final_test_label_dataset_path, allow_pickle=True)
    else:
        raise ValueError(f"Unknown data source: {cfg.rank.data_source}")

    initial_base_dir = Path(
        f'{cfg.dir.analyze_result_dir}/attention_map') / f'initial_attention_map_visualizations'
    os.makedirs(initial_base_dir, exist_ok=True)
    final_base_dir = Path(
        f'{cfg.dir.analyze_result_dir}/attention_map') / f'final_attention_map_visualizations'
    os.makedirs(final_base_dir, exist_ok=True)

    if cfg.model.name == 'MedFormer':
        print("Inference Start")
        initial_embedding_results_dict, initial_attention_results_dict, initial_attention_output_results_dict = inference_model_transformer(
            device, test_input_dataset, initial_test_label_dataset, cfg.rank.initial_checkpoint, cfg.rank.inference_batch_size
        )

        final_embedding_results_dict, final_attention_results_dict, final_attention_output_results_dict = inference_model_transformer(
            device, test_input_dataset, final_test_label_dataset, cfg.rank.final_checkpoint, cfg.rank.inference_batch_size
        )
        print("Inference End")
        print("SVD Start")
        initial_analysis_dict = analyze_attention_embedding_svd(initial_embedding_results_dict, initial_attention_results_dict, initial_attention_output_results_dict, tolerance=10**-1.5)
        with open(initial_base_dir / f"{cfg.model.name}_{cfg.dataset.id}_{exp_category}_{cfg.rank.data_source}_initial_analysis_dict.pkl", 'wb') as f:
            pickle.dump(initial_analysis_dict, f)

        final_analysis_dict = analyze_attention_embedding_svd(final_embedding_results_dict, final_attention_results_dict, final_attention_output_results_dict, tolerance=10**-1.5)
        with open(final_base_dir / f"{cfg.model.name}_{cfg.dataset.id}_{exp_category}_{cfg.rank.data_source}_final_analysis_dict.pkl", 'wb') as f:
            pickle.dump(final_analysis_dict, f)

        print("SVD End")
    elif cfg.model.name == 'NeuroSketch':
        print("Inference Start")
        initial_features_dict = inference_model_cnn_2d(
            device, test_input_dataset, initial_test_label_dataset, cfg.rank.initial_checkpoint, cfg.rank.inference_batch_size
        )
        final_features_dict = inference_model_cnn_2d(
            device, test_input_dataset, final_test_label_dataset, cfg.rank.final_checkpoint, cfg.rank.inference_batch_size
        )
        print("Inference End")
        print("SVD Start")
        initial_analysis_dict = analyze_cnn2d_svd(initial_features_dict, tolerance=10 ** -1.5)
        with open(initial_base_dir / f"{cfg.model.name}_{cfg.dataset.id}_{exp_category}_{cfg.rank.data_source}_initial_analysis_dict.pkl", 'wb') as f:
            pickle.dump(initial_analysis_dict, f)

        final_analysis_dict = analyze_cnn2d_svd(final_features_dict, tolerance=10 ** -1.5)
        with open(final_base_dir / f"{cfg.model.name}_{cfg.dataset.id}_{exp_category}_{cfg.rank.data_source}_final_analysis_dict.pkl", 'wb') as f:
            pickle.dump(final_analysis_dict, f)

        print("SVD End")
    elif cfg.model.name == 'ModernTCN':
        print("Inference Start")
        initial_features_dict = inference_model_cnn_1d(
            device, test_input_dataset, initial_test_label_dataset, cfg.rank.initial_checkpoint, cfg.rank.inference_batch_size
        )
        final_features_dict = inference_model_cnn_1d(
            device, test_input_dataset, final_test_label_dataset, cfg.rank.final_checkpoint, cfg.rank.inference_batch_size
        )
        print("Inference End")
        print("SVD Start")
        initial_analysis_dict = analyze_cnn1d_svd(initial_features_dict, tolerance=10 ** -1.5)
        with open(
                initial_base_dir / f"{cfg.model.name}_{cfg.dataset.id}_{exp_category}_{cfg.rank.data_source}_initial_analysis_dict.pkl",
                'wb') as f:
            pickle.dump(initial_analysis_dict, f)

        final_analysis_dict = analyze_cnn1d_svd(final_features_dict, tolerance=10 ** -1.5)
        with open(
                final_base_dir / f"{cfg.model.name}_{cfg.dataset.id}_{exp_category}_{cfg.rank.data_source}_final_analysis_dict.pkl",
                'wb') as f:
            pickle.dump(final_analysis_dict, f)

        print("SVD End")
    elif cfg.model.name == 'MultiResGRU':
        print("Inference Start")
        initial_features_dict = inference_model_gru(
            device, test_input_dataset, initial_test_label_dataset, cfg.rank.initial_checkpoint, cfg.rank.inference_batch_size
        )
        final_features_dict = inference_model_gru(
            device, test_input_dataset, final_test_label_dataset, cfg.rank.final_checkpoint, cfg.rank.inference_batch_size
        )
        print("Inference End")
        print("SVD Start")
        initial_analysis_dict = analyze_gru_svd(initial_features_dict, tolerance=10 ** -1.5)
        with open(
                initial_base_dir / f"{cfg.model.name}_{cfg.dataset.id}_{exp_category}_{cfg.rank.data_source}_initial_analysis_dict.pkl",
                'wb') as f:
            pickle.dump(initial_analysis_dict, f)

        final_analysis_dict = analyze_gru_svd(final_features_dict, tolerance=10 ** -1.5)
        with open(
                final_base_dir / f"{cfg.model.name}_{cfg.dataset.id}_{exp_category}_{cfg.rank.data_source}_final_analysis_dict.pkl",
                'wb') as f:
            pickle.dump(final_analysis_dict, f)

        print("SVD End")


@hydra.main(config_path="../conf/", config_name="analyze", version_base="1.2")
def get_attention_visualization(cfg):
    SUBJECTS_BY_STATE = {}
    SUBJECTS_BY_STATE['speaking'] = ["S1", "S2", "S3", "S5", "S6", "S7", "S9", "S11", "S12"]
    SUBJECTS_BY_STATE['listening'] = ["S4", "S5", "S6", "S7", "S10", "S11", "S12"]
    initial_base_dir = Path(
        f'{cfg.dir.analyze_result_dir}/attention_map') / f'initial_attention_map_visualizations'
    os.makedirs(initial_base_dir, exist_ok=True)
    final_base_dir = Path(
        f'{cfg.dir.analyze_result_dir}/attention_map') / f'final_attention_map_visualizations'
    os.makedirs(final_base_dir, exist_ok=True)

    sentence_initial_analysis_list = []
    sentence_initial_name_list = []
    sentence_final_analysis_list = []
    sentence_final_name_list = []

    for exp_category in ['speaking']:
        for subject_id in SUBJECTS_BY_STATE[exp_category]:
            sentence_initial_analysis_list.append(
                np.load(initial_base_dir / f"{cfg.model.name}_{subject_id}_{exp_category}_sentence_initial_analysis_dict.pkl", allow_pickle=True))
            sentence_initial_name_list.append(f"{exp_category}_{subject_id}")

            sentence_final_analysis_list.append(
                np.load(final_base_dir / f"{cfg.model.name}_{subject_id}_{exp_category}_sentence_final_analysis_dict.pkl", allow_pickle=True)
            )
            sentence_final_name_list.append(f"{exp_category}_{subject_id}")

    if cfg.model.name == "NeuroSketch" or cfg.model.name == "ModernTCN" or cfg.model.name == "MultiResGRU":
        plot_model_layers_effective_rank_ratio(
            analysis_dict_list=sentence_initial_analysis_list,
            name_list=sentence_initial_name_list,
            save_path=initial_base_dir / f"{cfg.model.name}_sentence_initial_effective_rank_ratio.png"
        )
        plot_model_layers_effective_rank_ratio(
            analysis_dict_list=sentence_final_analysis_list,
            name_list=sentence_final_name_list,
            save_path=final_base_dir / f"{cfg.model.name}_sentence_final_effective_rank_ratio.png"
        )
    elif cfg.model.name == "MedFormer":
        plot_attention_output_effective_rank_ratio(
            analysis_dict_list=sentence_initial_analysis_list,
            name_list=sentence_initial_name_list,
            save_path=initial_base_dir / f"{cfg.model.name}_sentence_initial_effective_rank_ratio.png"
        )
        plot_attention_output_effective_rank_ratio(
            analysis_dict_list=sentence_final_analysis_list,
            name_list=sentence_final_name_list,
            save_path=final_base_dir / f"{cfg.model.name}_sentence_final_effective_rank_ratio.png"
        )


if __name__ == "__main__":
    get_attention_analyze_dict()
    get_attention_visualization()