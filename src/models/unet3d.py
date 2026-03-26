from __future__ import annotations

import monai


class UNet3D(monai.networks.nets.UNet):
    def __init__(self, dropout: float = 0.0):
        super().__init__(
            spatial_dims=3,
            in_channels=2,
            out_channels=2,
            channels=(8, 16, 32, 64),
            strides=(2, 2, 2),
            dropout=dropout,
        )
