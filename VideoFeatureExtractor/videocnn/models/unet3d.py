"""UNet3D: A 3D CNN architecture for video feature extraction.

Relationship with CNN and S3D:
- UNet3D is a CNN — it is built entirely from 3D convolutional layers
  (nn.Conv3d, nn.BatchNorm3d, nn.MaxPool3d), exactly like S3D and ResNeXt-3D.
- S3D uses separable spatiotemporal Inception blocks for its encoder.
  UNet3D uses paired 3×3×3 Conv3D blocks ("double conv") in both an encoder
  and a mirrored decoder connected by skip connections.
- The encoder path of UNet3D is therefore a standard 3D CNN that progressively
  downsamples spatial and temporal dimensions — the same fundamental operation
  that S3D performs with its STConv3D + MaxPool3d stages.
- For video feature extraction (as used in this project) only the encoder path
  is needed; global average pooling over the bottleneck produces a compact
  feature vector that is drop-in compatible with S3D's 1024-d output.

Original UNet reference:
  O. Ronneberger, P. Fischer, T. Brox, "U-Net: Convolutional Networks for
  Biomedical Image Segmentation", MICCAI 2015.
  https://arxiv.org/abs/1505.04597

3D extension reference:
  Ö. Çiçek et al., "3D U-Net: Learning Dense Volumetric Segmentation from
  Sparse Annotation", MICCAI 2016.
  https://arxiv.org/abs/1606.06650
"""

import torch
import torch.nn as nn


class DoubleConv3D(nn.Module):
    """Two consecutive Conv3D → BatchNorm3D → ReLU blocks.

    This is the basic building block shared by both the UNet3D encoder and
    decoder — directly analogous to the Conv → BN → ReLU pattern used inside
    every STConv3D block in S3D.
    """

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv3D, self).__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down3D(nn.Module):
    """Encoder block: MaxPool3D halving → DoubleConv3D.

    Mirrors the MaxPool3D + convolution pattern used in every stage of S3D's
    forward pass.
    """

    def __init__(self, in_channels, out_channels):
        super(Down3D, self).__init__()
        self.down = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2),
            DoubleConv3D(in_channels, out_channels),
        )

    def forward(self, x):
        return self.down(x)


class Up3D(nn.Module):
    """Decoder block: trilinear upsample → concatenate skip → DoubleConv3D.

    The skip connection is the hallmark that distinguishes UNet3D from a plain
    3D CNN encoder: it reuses feature maps from the corresponding encoder stage
    to preserve fine-grained spatial and temporal detail.
    """

    def __init__(self, in_channels, out_channels):
        super(Up3D, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.conv = DoubleConv3D(in_channels, out_channels, in_channels // 2)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet3D(nn.Module):
    """3D U-Net for video feature extraction.

    The encoder (down-sampling path) is a 3D CNN — a sequence of
    MaxPool3D + double-Conv3D stages that is conceptually equivalent to the
    encoder stages in S3D.  The decoder (up-sampling path) mirrors the encoder
    with trilinear upsampling and skip connections.

    For feature extraction (rather than dense segmentation) ``extract_features``
    should be set to ``True`` (the default).  In that mode the network
    discards the decoder, applies global average pooling to the bottleneck
    feature map, and projects it to ``num_classes`` dimensions — producing a
    vector that is directly compatible with S3D's output interface.

    Args:
        num_classes (int): Dimensionality of the output feature vector when
            ``extract_features=True``, or number of segmentation classes when
            ``extract_features=False``.  Default: 512 (matches S3D default).
        in_channels (int): Number of input channels (e.g. 3 for RGB video).
        base_channels (int): Number of feature maps in the first conv stage;
            doubled at every encoder level.
        extract_features (bool): When ``True`` (default) return a pooled
            1-D feature vector per clip; when ``False`` return the full
            spatiotemporal segmentation logits.
    """

    def __init__(
        self,
        num_classes: int = 512,
        in_channels: int = 3,
        base_channels: int = 64,
        extract_features: bool = True,
    ):
        super(UNet3D, self).__init__()
        self.extract_features = extract_features
        c = base_channels

        # ── Encoder (3D CNN downsampling path) ──────────────────────────────
        # Each stage halves spatial/temporal resolution and doubles channels,
        # the same progression used by S3D's inception + maxpool stages.
        self.enc1 = DoubleConv3D(in_channels, c)          # → c
        self.enc2 = Down3D(c, c * 2)                      # → c*2
        self.enc3 = Down3D(c * 2, c * 4)                  # → c*4
        self.enc4 = Down3D(c * 4, c * 8)                  # → c*8  (bottleneck)

        if not extract_features:
            # ── Decoder (with UNet skip connections) ────────────────────────
            # The in_channels for each Up3D is the *total* after concatenation:
            # up1: bottleneck (c*8) upsampled + skip s3 (c*4) → c*12 in, c*4 out
            # up2: up1 out   (c*4) upsampled + skip s2 (c*2) → c*6  in, c*2 out
            # up3: up2 out   (c*2) upsampled + skip s1 (c)   → c*3  in, c   out
            self.up1 = Up3D(c * 8 + c * 4, c * 4)
            self.up2 = Up3D(c * 4 + c * 2, c * 2)
            self.up3 = Up3D(c * 2 + c, c)
            self.out_conv = nn.Conv3d(c, num_classes, kernel_size=1)
        else:
            # ── Feature extraction head ──────────────────────────────────────
            # Global average pool over (T, H, W) → linear projection, just
            # like S3D's final avg-pool + fc layer.
            self.fc = nn.Linear(c * 8, num_classes)

    def forward(self, x):
        # Encoder
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        bottleneck = self.enc4(s3)

        if not self.extract_features:
            # Decoder with skip connections
            out = self.up1(bottleneck, s3)
            out = self.up2(out, s2)
            out = self.up3(out, s1)
            return self.out_conv(out)

        # Global average pool then linear projection (same interface as S3D)
        out = torch.mean(bottleneck, dim=[2, 3, 4])
        out = self.fc(out)
        return out
