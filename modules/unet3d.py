# coding=utf-8
"""
UNet3D encoder for end-to-end video feature extraction.

Replaces offline S3D feature extraction with a trainable 3D U-Net encoder
that processes raw video clips directly during training.

Input:  (B, 3, T, H, W)  – batch of video clips (3-channel RGB frames)
Output: (B, T, out_dim)  – per-frame feature vectors
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock3D(nn.Module):
    """Two consecutive 3D Conv -> BN -> ReLU layers."""

    def __init__(self, in_channels, out_channels):
        super(ConvBlock3D, self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet3DEncoder(nn.Module):
    """
    3D U-Net encoder for video feature extraction.

    Progressively downsamples spatial dimensions while preserving the temporal
    dimension, producing one feature vector per input frame.

    Architecture
    ------------
    Encoder path (4 stages):
      Stage 1: Conv3D(3         → base_channels)     – full spatial resolution
      Stage 2: Conv3D(base_channels → base_channels*2) – H/2, W/2
      Stage 3: Conv3D(base_channels*2 → base_channels*4) – H/4, W/4
      Stage 4: Conv3D(base_channels*4 → base_channels*8) – H/8, W/8

    Global spatial pooling removes H and W, leaving (B, base_ch*8, T).
    A linear projection maps to out_dim, yielding (B, T, out_dim).

    Parameters
    ----------
    in_channels : int
        Number of input channels (3 for RGB video).
    base_channels : int
        Base channel multiplier for the encoder stages (default 64).
    out_dim : int
        Output feature dimension per frame.  Should match ``video_dim`` in
        the main model config (default 1024).
    """

    def __init__(self, in_channels=3, base_channels=64, out_dim=1024):
        super(UNet3DEncoder, self).__init__()

        # Encoder blocks
        self.enc1 = ConvBlock3D(in_channels, base_channels)
        self.enc2 = ConvBlock3D(base_channels, base_channels * 2)
        self.enc3 = ConvBlock3D(base_channels * 2, base_channels * 4)
        self.enc4 = ConvBlock3D(base_channels * 4, base_channels * 8)

        # Spatial-only downsampling (temporal dimension preserved)
        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        # Global spatial pooling -> (B, base_ch*8, T, 1, 1)
        self.global_spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))

        # Project encoder features to output dimension
        self.projection = nn.Linear(base_channels * 8, out_dim)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor, shape (B, 3, T, H, W)

        Returns
        -------
        torch.Tensor, shape (B, T, out_dim)
        """
        # Encoder path (spatial resolution halved at each stage)
        x1 = self.enc1(x)                    # (B, C,   T, H,   W)
        x2 = self.enc2(self.pool(x1))         # (B, 2C,  T, H/2, W/2)
        x3 = self.enc3(self.pool(x2))         # (B, 4C,  T, H/4, W/4)
        x4 = self.enc4(self.pool(x3))         # (B, 8C,  T, H/8, W/8)

        # Remove spatial dimensions
        features = self.global_spatial_pool(x4)   # (B, 8C, T, 1, 1)
        features = features.squeeze(-1).squeeze(-1)  # (B, 8C, T)
        features = features.permute(0, 2, 1)          # (B, T, 8C)

        # Project to output dimension
        features = self.projection(features)          # (B, T, out_dim)
        return features
