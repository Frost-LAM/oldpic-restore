"""CodeFormer face-restoration network, inference only.

This file is a cleaned inference port of
https://github.com/sczhou/CodeFormer (basicsr/archs/codeformer_arch.py and
vqgan_arch.py). Those sources are Copyright 2022 S-Lab and are licensed
under the S-Lab License 1.0, which permits non-commercial use only.
The VQGAN blocks were adapted upstream from Unleashing Transformers.

Commercial use of this module or of the CodeFormer weights requires
permission from the CodeFormer contributors. See LICENSES/S-Lab-1.0.txt
and THIRD_PARTY_NOTICES.md.

The module names match the published ``codeformer.pth`` ``params_ema``
checkpoint. Do not rename submodules.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _group_norm(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6, affine=True)


def _swish(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


def calc_mean_std(feat: Tensor, eps: float = 1e-5):
    b, c = feat.shape[:2]
    feat_var = feat.view(b, c, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(b, c, 1, 1)
    feat_mean = feat.view(b, c, -1).mean(dim=2).view(b, c, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat: Tensor, style_feat: Tensor) -> Tensor:
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized * style_std.expand(size) + style_mean.expand(size)


class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size: int, emb_dim: int, beta: float):
        super().__init__()
        self.codebook_size = codebook_size
        self.emb_dim = emb_dim
        self.beta = beta
        self.embedding = nn.Embedding(self.codebook_size, self.emb_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.codebook_size, 1.0 / self.codebook_size)

    def get_codebook_feat(self, indices: Tensor, shape) -> Tensor:
        indices = indices.view(-1, 1)
        min_encodings = torch.zeros(indices.shape[0], self.codebook_size, device=indices.device)
        min_encodings.scatter_(1, indices, 1)
        z_q = torch.matmul(min_encodings, self.embedding.weight)
        if shape is not None:
            z_q = z_q.view(shape).permute(0, 3, 1, 2).contiguous()
        return z_q


class Downsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: Tensor) -> Tensor:
        x = F.pad(x, (0, 1, 0, 1), mode="constant", value=0)
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int | None = None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        out_channels = self.out_channels
        self.norm1 = _group_norm(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = _group_norm(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            self.conv_out = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x_in: Tensor) -> Tensor:
        x = _swish(self.norm1(x_in))
        x = self.conv1(x)
        x = _swish(self.norm2(x))
        x = self.conv2(x)
        if self.in_channels != self.out_channels:
            x_in = self.conv_out(x_in)
        return x + x_in


class AttnBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.norm = _group_norm(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, 1)
        self.k = nn.Conv2d(in_channels, in_channels, 1)
        self.v = nn.Conv2d(in_channels, in_channels, 1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        h_ = self.norm(x)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w).permute(0, 2, 1)
        k = k.reshape(b, c, h * w)
        w_ = torch.bmm(q, k) * (int(c) ** -0.5)
        w_ = F.softmax(w_, dim=2)
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)
        h_ = torch.bmm(v, w_).reshape(b, c, h, w)
        return x + self.proj_out(h_)


class Encoder(nn.Module):
    def __init__(self, in_channels, nf, emb_dim, ch_mult, num_res_blocks, resolution, attn_resolutions):
        super().__init__()
        self.nf = nf
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.attn_resolutions = attn_resolutions
        curr_res = self.resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        blocks = []
        blocks.append(nn.Conv2d(in_channels, nf, kernel_size=3, stride=1, padding=1))
        for i in range(self.num_resolutions):
            block_in_ch = nf * in_ch_mult[i]
            block_out_ch = nf * ch_mult[i]
            for _ in range(self.num_res_blocks):
                blocks.append(ResBlock(block_in_ch, block_out_ch))
                block_in_ch = block_out_ch
                if curr_res in attn_resolutions:
                    blocks.append(AttnBlock(block_in_ch))
            if i != self.num_resolutions - 1:
                blocks.append(Downsample(block_in_ch))
                curr_res = curr_res // 2
        blocks.append(ResBlock(block_in_ch, block_in_ch))
        blocks.append(AttnBlock(block_in_ch))
        blocks.append(ResBlock(block_in_ch, block_in_ch))
        blocks.append(_group_norm(block_in_ch))
        blocks.append(nn.Conv2d(block_in_ch, emb_dim, kernel_size=3, stride=1, padding=1))
        self.blocks = nn.ModuleList(blocks)


class Generator(nn.Module):
    def __init__(self, nf, emb_dim, ch_mult, res_blocks, img_size, attn_resolutions):
        super().__init__()
        self.nf = nf
        self.ch_mult = ch_mult
        self.num_resolutions = len(self.ch_mult)
        self.num_res_blocks = res_blocks
        self.resolution = img_size
        self.attn_resolutions = attn_resolutions
        self.in_channels = emb_dim
        self.out_channels = 3
        block_in_ch = self.nf * self.ch_mult[-1]
        curr_res = self.resolution // 2 ** (self.num_resolutions - 1)
        blocks = []
        blocks.append(nn.Conv2d(self.in_channels, block_in_ch, kernel_size=3, stride=1, padding=1))
        blocks.append(ResBlock(block_in_ch, block_in_ch))
        blocks.append(AttnBlock(block_in_ch))
        blocks.append(ResBlock(block_in_ch, block_in_ch))
        for i in reversed(range(self.num_resolutions)):
            block_out_ch = self.nf * self.ch_mult[i]
            for _ in range(self.num_res_blocks):
                blocks.append(ResBlock(block_in_ch, block_out_ch))
                block_in_ch = block_out_ch
                if curr_res in self.attn_resolutions:
                    blocks.append(AttnBlock(block_in_ch))
            if i != 0:
                blocks.append(Upsample(block_in_ch))
                curr_res = curr_res * 2
        blocks.append(_group_norm(block_in_ch))
        blocks.append(nn.Conv2d(block_in_ch, self.out_channels, kernel_size=3, stride=1, padding=1))
        self.blocks = nn.ModuleList(blocks)


class VQAutoEncoder(nn.Module):
    def __init__(
        self,
        img_size,
        nf,
        ch_mult,
        quantizer="nearest",
        res_blocks=2,
        attn_resolutions=None,
        codebook_size=1024,
        emb_dim=256,
        beta=0.25,
    ):
        super().__init__()
        if attn_resolutions is None:
            attn_resolutions = [16]
        self.in_channels = 3
        self.nf = nf
        self.n_blocks = res_blocks
        self.codebook_size = codebook_size
        self.embed_dim = emb_dim
        self.ch_mult = ch_mult
        self.resolution = img_size
        self.attn_resolutions = attn_resolutions
        self.quantizer_type = quantizer
        self.encoder = Encoder(
            self.in_channels, self.nf, self.embed_dim, self.ch_mult, self.n_blocks, self.resolution, self.attn_resolutions
        )
        self.beta = beta
        self.quantize = VectorQuantizer(self.codebook_size, self.embed_dim, self.beta)
        self.generator = Generator(
            self.nf, self.embed_dim, self.ch_mult, self.n_blocks, self.resolution, self.attn_resolutions
        )


class TransformerSALayer(nn.Module):
    def __init__(self, embed_dim, nhead=8, dim_mlp=2048, dropout=0.0, activation="gelu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, nhead, dropout=dropout)
        self.linear1 = nn.Linear(embed_dim, dim_mlp)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_mlp, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        if activation != "gelu":
            raise ValueError(activation)
        self.activation = F.gelu

    def forward(self, tgt, query_pos: Tensor | None = None):
        tgt2 = self.norm1(tgt)
        q = k = tgt2 if query_pos is None else tgt2 + query_pos
        tgt2 = self.self_attn(q, k, value=tgt2)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout2(tgt2)
        return tgt


class Fuse_sft_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.encode_enc = ResBlock(2 * in_ch, out_ch)
        self.scale = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        )
        self.shift = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        )

    def forward(self, enc_feat, dec_feat, w=1):
        enc_feat = self.encode_enc(torch.cat([enc_feat, dec_feat], dim=1))
        scale = self.scale(enc_feat)
        shift = self.shift(enc_feat)
        residual = w * (dec_feat * scale + shift)
        return dec_feat + residual


class CodeFormer(VQAutoEncoder):
    def __init__(
        self,
        dim_embd=512,
        n_head=8,
        n_layers=9,
        codebook_size=1024,
        latent_size=256,
        connect_list=None,
    ):
        super().__init__(512, 64, [1, 2, 2, 4, 4, 8], "nearest", 2, [16], codebook_size)
        if connect_list is None:
            connect_list = ["32", "64", "128", "256"]
        self.connect_list = connect_list
        self.n_layers = n_layers
        self.dim_embd = dim_embd
        self.dim_mlp = dim_embd * 2
        self.position_emb = nn.Parameter(torch.zeros(latent_size, self.dim_embd))
        self.feat_emb = nn.Linear(256, self.dim_embd)
        self.ft_layers = nn.Sequential(
            *[TransformerSALayer(embed_dim=dim_embd, nhead=n_head, dim_mlp=self.dim_mlp, dropout=0.0) for _ in range(self.n_layers)]
        )
        self.idx_pred_layer = nn.Sequential(nn.LayerNorm(dim_embd), nn.Linear(dim_embd, codebook_size, bias=False))
        self.channels = {"16": 512, "32": 256, "64": 256, "128": 128, "256": 128, "512": 64}
        self.fuse_encoder_block = {"512": 2, "256": 5, "128": 8, "64": 11, "32": 14, "16": 18}
        self.fuse_generator_block = {"16": 6, "32": 9, "64": 12, "128": 15, "256": 18, "512": 21}
        self.fuse_convs_dict = nn.ModuleDict()
        for f_size in self.connect_list:
            in_ch = self.channels[f_size]
            self.fuse_convs_dict[f_size] = Fuse_sft_block(in_ch, in_ch)

    def forward(self, x, w=0, detach_16=True, code_only=False, adain=False):
        enc_feat_dict = {}
        out_list = [self.fuse_encoder_block[f_size] for f_size in self.connect_list]
        for i, block in enumerate(self.encoder.blocks):
            x = block(x)
            if i in out_list:
                enc_feat_dict[str(x.shape[-1])] = x.clone()
        lq_feat = x
        pos_emb = self.position_emb.unsqueeze(1).repeat(1, x.shape[0], 1)
        feat_emb = self.feat_emb(lq_feat.flatten(2).permute(2, 0, 1))
        query_emb = feat_emb
        for layer in self.ft_layers:
            query_emb = layer(query_emb, query_pos=pos_emb)
        logits = self.idx_pred_layer(query_emb)
        logits = logits.permute(1, 0, 2)
        if code_only:
            return logits, lq_feat
        soft_one_hot = F.softmax(logits, dim=2)
        _, top_idx = torch.topk(soft_one_hot, 1, dim=2)
        quant_feat = self.quantize.get_codebook_feat(top_idx, shape=[x.shape[0], 16, 16, 256])
        if detach_16:
            quant_feat = quant_feat.detach()
        if adain:
            quant_feat = adaptive_instance_normalization(quant_feat, lq_feat)
        x = quant_feat
        fuse_list = [self.fuse_generator_block[f_size] for f_size in self.connect_list]
        for i, block in enumerate(self.generator.blocks):
            x = block(x)
            if i in fuse_list:
                f_size = str(x.shape[-1])
                if w > 0:
                    x = self.fuse_convs_dict[f_size](enc_feat_dict[f_size].detach(), x, w)
        return x, logits, lq_feat
