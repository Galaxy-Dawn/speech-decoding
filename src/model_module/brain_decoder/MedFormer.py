import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math
from math import sqrt
from src.model_module.brain_decoder import register_model
from src.data_module.augmentation import add_noise, ChannelMasking, TimeMasking, random_shift, Mixup


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff, dropout, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn, x_out_history = self.attention(x, attn_mask=attn_mask, tau=tau, delta=delta)

        x = [_x + self.dropout(_nx) for _x, _nx in zip(x, new_x)]

        y = x = [self.norm1(_x) for _x in x]
        y = [self.dropout(self.activation(self.conv1(_y.transpose(-1, 1)))) for _y in y]
        y = [self.dropout(self.conv2(_y).transpose(-1, 1)) for _y in y]

        output = [self.norm2(_x + _y) for _x, _y in zip(x, y)]
        return output, attn, x_out_history  # Pass pure vector list


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [[B, L1, D], [B, L2, D], ...]
        attns = []
        all_x_out_history = []
        for attn_layer in self.attn_layers:
            x, attn, x_out_history = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
            attns.append(attn)  # attn is a pure list (weight matrix)
            all_x_out_history.append(x_out_history)  # x_out_history is a pure list (vectors)

        x = torch.cat([x[:, -1, :].unsqueeze(1) for x in x], dim=1)
        if self.norm is not None:
            x = self.norm(x)

        return x, attns, all_x_out_history


class TriangularCausalMask:
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(
                torch.ones(mask_shape, dtype=torch.bool), diagonal=1
            ).to(device)

    @property
    def mask(self):
        return self._mask


class FullAttention(nn.Module):
    def __init__(
        self,
        mask_flag=True,
        factor=5,
        scale=None,
        attention_dropout=0.1,
        output_attention=False,
    ):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1.0 / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(
            torch.softmax(scale * scores, dim=-1)
        )  # Scaled Dot-Product Attention
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)  # multi-head
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries, keys, values, attn_mask, tau=tau, delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class MedformerLayer(nn.Module):
    def __init__(
        self,
        num_blocks,
        d_model,
        n_heads,
        dropout=0.1,
        output_attention=False,
        no_inter=False,
    ):
        super().__init__()

        self.intra_attentions = nn.ModuleList(
            [
                AttentionLayer(
                    FullAttention(
                        False,
                        factor=1,
                        attention_dropout=dropout,
                        output_attention=output_attention,
                    ),
                    d_model,
                    n_heads,
                )
                for _ in range(num_blocks)
            ]
        )
        if no_inter or num_blocks <= 1:
            # print("No inter attention for time")
            self.inter_attention = None
        else:
            self.inter_attention = AttentionLayer(
                FullAttention(
                    False,
                    factor=1,
                    attention_dropout=dropout,
                    output_attention=output_attention,
                ),
                d_model,
                n_heads,
            )
    def forward(self, x, attn_mask=None, tau=None, delta=None):
        attn_mask = attn_mask or ([None] * len(x))
        x_intra = []
        attn_out = []
        x_out_history = []
        # 1. Intra-Attention: Store vectors sequentially, aligned with attn_out
        for x_in, layer, mask in zip(x, self.intra_attentions, attn_mask):
            _x_out, _attn = layer(x_in, x_in, x_in, attn_mask=mask, tau=tau, delta=delta)
            x_intra.append(_x_out)
            attn_out.append(_attn)  # Store weight matrix
            x_out_history.append(_x_out)  # Store vectors (one-to-one correspondence with attn_out indices)

        # 2. Inter-Attention: If exists, append to the end of the list, aligned with attn_out
        if self.inter_attention is not None:
            routers = torch.cat([x[:, -1:] for x in x_intra], dim=1)
            x_inter, attn_inter = self.inter_attention(
                routers, routers, routers, attn_mask=None, tau=tau, delta=delta
            )
            x_out = [
                torch.cat([x[:, :-1], x_inter[:, i : i + 1]], dim=1)
                for i, x in enumerate(x_intra)
            ]
            attn_out.append(attn_inter)  # Append inter weight matrix
            x_out_history.append(x_inter)  # Append inter vectors (one-to-one correspondence with attn_out indices)
        else:
            x_out = x_intra
            # No inter: do not append (keep length consistent with attn_out)

        # Key: Return pure lists (x_out, attn_out, x_out_history), all three structures are fully aligned
        return x_out, attn_out, x_out_history


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[:, : x.size(1)]


class CrossChannelTokenEmbedding(nn.Module):
    def __init__(self, c_in, l_patch, d_model, stride=None):
        super().__init__()
        if stride is None:
            stride = l_patch
        self.tokenConv = nn.Conv2d(
            in_channels=1,
            out_channels=d_model,
            kernel_size=(c_in, l_patch),
            stride=(1, stride),
            padding=0,
            padding_mode="circular",
            bias=False,
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )

    def forward(self, x):
        x = self.tokenConv(x)
        return x


class ListPatchEmbedding(nn.Module):
    def __init__(
        self,
        enc_in,
        d_model,
        seq_len,
        patch_len_list,
        stride_list,
        dropout=0.0,
        single_channel=False,
    ):
        super().__init__()
        self.patch_len_list = patch_len_list
        self.stride_list = stride_list
        self.paddings = [nn.ReplicationPad1d((0, stride)) for stride in stride_list]
        self.single_channel = single_channel

        linear_layers = [
            CrossChannelTokenEmbedding(
                c_in=enc_in if not single_channel else 1,
                l_patch=patch_len,
                d_model=d_model,
            )
            for patch_len in patch_len_list
        ]
        self.value_embeddings = nn.ModuleList(linear_layers)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.channel_embedding = PositionalEmbedding(d_model=seq_len)
        self.dropout = nn.Dropout(dropout)
        self.learnable_embeddings = nn.ParameterList(
            [nn.Parameter(torch.randn(1, d_model)) for _ in patch_len_list]
        )

    def forward(self, x):  # (batch_size, seq_len, enc_in)
        x = x.permute(0, 2, 1)  # (batch_size, enc_in, seq_len)
        if self.single_channel:
            B, C, L = x.shape
            x = torch.reshape(x, (B * C, 1, L))

        x_list = []
        for padding, value_embedding in zip(self.paddings, self.value_embeddings):
            x_copy = x.clone()
            x_new = x_copy
            # add positional embedding to tag each channel
            x_new = x_new + self.channel_embedding(x_new)
            x_new = padding(x_new).unsqueeze(1)  # (batch_size, 1, enc_in, seq_len+stride)
            x_new = value_embedding(x_new)  # (batch_size, d_model, 1, patch_num)
            x_new = x_new.squeeze(2).transpose(1, 2)  # (batch_size, patch_num, d_model)
            x_list.append(x_new)

        x = [
            x + cxt + self.position_embedding(x)
            for x, cxt in zip(x_list, self.learnable_embeddings)
        ]  # (batch_size, patch_num_1, d_model), (batch_size, patch_num_2, d_model), ...
        return x


@register_model('MedFormer')
class MedFormer(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2405.19363
    """
    def __init__(self, cfg):
        super(MedFormer, self).__init__()
        self.cfg = cfg
        self.task = cfg.dataset.task
        self.output_attention = cfg.model.output_attention
        self.channel_dict = np.load(cfg.dataset.channel_dict_path, allow_pickle=True)
        self.channel_name = cfg.dataset.channel_index_name

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

        self.enc_in = len(self.selected_channels)

        self.single_channel = cfg.model.single_channel
        # Embedding
        patch_len_list = list(map(int, cfg.model.patch_len_list.split(",")))
        stride_list = patch_len_list
        seq_len = cfg.dataset.seq_len[self.task]
        seq_len = seq_len if seq_len%2 == 0 else seq_len-1
        patch_num_list = [
            int((seq_len - patch_len) / stride + 2)
            for patch_len, stride in zip(patch_len_list, stride_list)
        ]
        self.enc_embedding = ListPatchEmbedding(
            self.enc_in,
            cfg.model.d_model,
            seq_len,
            patch_len_list,
            stride_list,
            cfg.model.dropout,
            self.single_channel,
        )
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    MedformerLayer(
                        len(patch_len_list),
                        cfg.model.d_model,
                        cfg.model.n_heads,
                        cfg.model.dropout,
                        cfg.model.output_attention,
                        cfg.model.no_inter_attn,
                    ),
                    cfg.model.d_model,
                    cfg.model.d_ff,
                    dropout=cfg.model.dropout,
                    activation=cfg.model.activation,
                )
                for l in range(cfg.model.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(cfg.model.d_model),
        )
        self.act = F.gelu
        self.dropout = nn.Dropout(cfg.model.dropout)
        self.target_size = cfg.dataset.target_size[self.task]
        self.projection = nn.Linear(
            cfg.model.d_model
            * len(patch_num_list)
            * (1 if not self.single_channel else self.enc_in),
            self.target_size,
        )
        self.loss_fn = nn.BCEWithLogitsLoss()

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
        if type(ieeg_raw_data) == dict:
            ieeg_raw_data, labels = ieeg_raw_data["ieeg_raw_data"], ieeg_raw_data["labels"]
        x = ieeg_raw_data[:, self.selected_channels]
        if labels is not None:
            if type(labels) != torch.LongTensor:
                labels = labels.long()
            one_hot_labels = F.one_hot(labels, num_classes=self.target_size).float()

        x = x.float()

        if x.shape[-1] % 2 == 1:
            x = x[:, :, :-1]

        if self.training:
            x = self.apply_augmentation(x)
            p = random.random()
            if p < self.cfg.augmentation.mixup_prob:
                x, one_hot_labels = Mixup(alpha=0.5)(x, one_hot_labels)

            x = x.permute(0, 2, 1)
            enc_out = self.enc_embedding(x)
            enc_out, attns, all_x_out_history = self.encoder(enc_out, attn_mask=None)
            if self.single_channel:
                enc_out = torch.reshape(enc_out, (-1, self.enc_in, *enc_out.shape[-2:]))
            # Output
            output = self.act(
                enc_out
            )  # the output transformer encoder/decoder embeddings don't include non-linearity
            output = self.dropout(output)
            output = output.reshape(
                output.shape[0], -1
            )  # (batch_size, seq_length * d_model)
            prediction = self.projection(output)  # (batch_size, num_classes)

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
                    enc_out = self.enc_embedding(x_aug)
                    enc_out, attns, all_x_out_history = self.encoder(enc_out, attn_mask=None)
                    if self.single_channel:
                        enc_out = torch.reshape(enc_out, (-1, self.enc_in, *enc_out.shape[-2:]))
                    # Output
                    output = self.act(
                        enc_out
                    )  # the output transformer encoder/decoder embeddings don't include non-linearity
                    output = self.dropout(output)
                    output = output.reshape(
                        output.shape[0], -1
                    )  # (batch_size, seq_length * d_model)
                    prediction = self.projection(output)  # (batch_size, num_classes)
                    all_logits.append(prediction)
            avg_logits = torch.mean(torch.stack(all_logits), dim=0)
            loss = self.loss_fn(avg_logits, one_hot_labels) if labels is not None else None

            return {
                "loss"  : loss,
                "labels": labels,
                "logits": avg_logits,
            }