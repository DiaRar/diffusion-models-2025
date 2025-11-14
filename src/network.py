from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .module import DownSample, ResBlock, Swish, TimeEmbedding, UpSample
from torch.nn import init


class UNet(nn.Module):
    def __init__(self, ch=128, ch_mult=[1,2,2,2], attn=[1], num_res_blocks=4, dropout=0.1, use_additional_condition=False):
        super().__init__()
        self.use_additional_condition = use_additional_condition
        assert all([i < len(ch_mult) for i in attn]), 'attn index out of bound'
        tdim = ch * 4
        self.time_embedding = TimeEmbedding(tdim)
        
        # Condition embedding for additional conditioning
        if use_additional_condition:
            self.condition_embedding = TimeEmbedding(tdim)
            # Fusion layer to combine main timestep and condition embeddings
            self.condition_fusion = nn.Sequential(
                nn.Linear(tdim * 2, tdim),
                nn.SiLU(),
                nn.Linear(tdim, tdim)
            )


        self.head = nn.Conv2d(3, ch, kernel_size=3, stride=1, padding=1)
        self.downblocks = nn.ModuleList()
        chs = [ch]  # record output channel when dowmsample for upsample
        now_ch = ch
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(ResBlock(
                    in_ch=now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn)))
                now_ch = out_ch
                chs.append(now_ch)
            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample(now_ch))
                chs.append(now_ch)

        self.middleblocks = nn.ModuleList([
            ResBlock(now_ch, now_ch, tdim, dropout, attn=True),
            ResBlock(now_ch, now_ch, tdim, dropout, attn=False),
        ])

        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(ResBlock(
                    in_ch=chs.pop() + now_ch, out_ch=out_ch, tdim=tdim,
                    dropout=dropout, attn=(i in attn)))
                now_ch = out_ch
            if i != 0:
                self.upblocks.append(UpSample(now_ch))
        assert len(chs) == 0

        self.tail = nn.Sequential(
            nn.GroupNorm(32, now_ch),
            Swish(),
            nn.Conv2d(now_ch, 3, 3, stride=1, padding=1)
        )
        self.initialize()

    def initialize(self):
        init.xavier_uniform_(self.head.weight)
        init.zeros_(self.head.bias)
        init.xavier_uniform_(self.tail[-1].weight, gain=1e-5)
        init.zeros_(self.tail[-1].bias)

    def forward(self, x, timestep, condition=None):
        """
        Forward pass with optional condition conditioning
        
        Args:
            x: Input tensor
            timestep: Main timestep tensor
            condition: Optional condition tensor of shape [batch,] for additional conditioning,
                       e.g., a step size in Shortcut Models or 
                       end timestep `s` in consistency trajectory models.
        """
        # Main timestep embedding
        temb = self.time_embedding(timestep)
        
        # Condition conditioning (optional)
        if self.use_additional_condition and condition is not None:
            condition_temb = self.condition_embedding(condition)
            # Fuse main timestep and condition embeddings
            combined_temb = torch.cat([temb, condition_temb], dim=-1)
            temb = self.condition_fusion(combined_temb)
        elif self.use_additional_condition and condition is None:
            raise ValueError("Condition is expected but not provided")

        # Downsampling
        h = self.head(x)
        hs = [h]
        for layer in self.downblocks:
            h = layer(h, temb)
            hs.append(h)
        # Middle
        for layer in self.middleblocks:
            h = layer(h, temb)
        # Upsampling
        for layer in self.upblocks:
            if isinstance(layer, ResBlock):
                h = torch.cat([h, hs.pop()], dim=1)
            h = layer(h, temb)
        h = self.tail(h)

        assert len(hs) == 0
        return h

