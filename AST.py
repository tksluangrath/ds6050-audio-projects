import torch
import torch.nn as nn
import math


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size, fstride, tstride, embed_dim):
        super().__init__()
        self.projection = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=(patch_size, patch_size),
            stride=(fstride, tstride),
            bias=False,
        )

    def forward(self, x):
        # (B, C, F, T) -> (B, embed_dim, F', T') -> (B, N, embed_dim)
        x = self.projection(x)
        x = x.flatten(start_dim=2).transpose(1, 2)
        return x


class AST(nn.Module):
    def __init__(
        self,
        num_classes=12,
        in_channels=1,
        patch_size=16,
        fstride=10,
        tstride=10,
        embed_dim=192,
        num_heads=4,
        num_layers=4,
        mlp_dim=768,
        dropout=0.1,
        freq_bins=40,
        time_frames=101,
    ):
        super().__init__()

        self.patch_embedding = PatchEmbedding(
            in_channels, patch_size, fstride, tstride, embed_dim
        )

        n_freq = (freq_bins - patch_size) // fstride + 1
        n_time = (time_frames - patch_size) // tstride + 1
        n_patches = n_freq * n_time

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.positional_embedding = nn.Parameter(
            torch.zeros(1, n_patches + 1, embed_dim)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embedding(x)
        B = x.shape[0]

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.positional_embedding

        x = self.transformer_encoder(x)
        x = self.layer_norm(x)

        return self.head(x[:, 0])