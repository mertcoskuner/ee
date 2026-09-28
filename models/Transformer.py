"""Vision Transformer for 28x28 single-channel MNIST digits."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VisionTransformer(nn.Module):
    """Classify MNIST images with a pre-norm Transformer encoder (ViT).

    The image is cut into non-overlapping patch_size x patch_size patches,
    each linearly projected to dim features. A learned class token and
    learned position embeddings are added, depth encoder layers apply
    multi-head self-attention and GELU feed-forward blocks, and a linear
    head classifies the normalized class token.
    """

    def __init__(
        self,
        image_size=28,
        patch_size=7,
        num_classes=10,
        dim=64,
        depth=2,
        heads=4,
        mlp_dim=128,
        dropout=0.1,
    ):
        """Create patch projection, embeddings, encoder, and linear head."""
        super().__init__()
        if image_size % patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        self.patch_size = patch_size
        num_patches = (image_size // patch_size) ** 2
        self.patch_proj = nn.Linear(patch_size * patch_size, dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.dropout = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            dim,
            heads,
            mlp_dim,
            dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        """Return unnormalized class logits for a batch of MNIST images."""
        patches = F.unfold(x, self.patch_size, stride=self.patch_size)
        tokens = self.patch_proj(patches.transpose(1, 2))
        cls = self.cls_token.expand(x.size(0), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1) + self.pos_embed
        tokens = self.encoder(self.dropout(tokens))
        return self.head(self.norm(tokens[:, 0]))
