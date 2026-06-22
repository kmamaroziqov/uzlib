from __future__ import annotations


def require_torch():
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "Training and prediction require PyTorch. "
            "Install dependencies with: python -m pip install -r requirements.txt"
        ) from exc
    return torch


def build_mlp(input_dim: int, num_classes: int, hidden_dim: int = 512, dropout: float = 0.25):
    torch = require_torch()
    nn = torch.nn
    return nn.Sequential(
        nn.LayerNorm(input_dim),
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, hidden_dim // 2),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim // 2, num_classes),
    )


def build_transformer(
    frame_dim: int,
    num_classes: int,
    *,
    target_frames: int,
    hidden_dim: int = 256,
    n_layers: int = 4,
    n_heads: int = 8,
    dropout: float = 0.1,
):
    torch = require_torch()
    nn = torch.nn

    class PoseTransformer(nn.Module):
        feature_dim = hidden_dim

        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(frame_dim, hidden_dim)
            self.pos = nn.Parameter(torch.zeros(1, target_frames, hidden_dim))
            nn.init.trunc_normal_(self.pos, std=0.02)
            enc_layer = nn.TransformerEncoderLayer(
                hidden_dim,
                nhead=n_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
            )
            self.enc = nn.TransformerEncoder(enc_layer, num_layers=n_layers, enable_nested_tensor=False)
            self.norm = nn.LayerNorm(hidden_dim)
            self.head = nn.Linear(hidden_dim, num_classes)

        def forward(self, x, return_features: bool = False):
            # x: (B, T, F)
            pad_mask = x.abs().sum(dim=-1) == 0
            # Guard against fully-masked rows (e.g. an aggressive temporal mask).
            all_pad = pad_mask.all(dim=1)
            if all_pad.any():
                pad_mask = pad_mask.clone()
                pad_mask[all_pad, 0] = False
            z = self.proj(x) + self.pos[:, : x.shape[1]]
            z = self.enc(z, src_key_padding_mask=pad_mask)
            keep = (~pad_mask).unsqueeze(-1).float()
            pooled = self.norm((z * keep).sum(dim=1) / keep.sum(dim=1).clamp(min=1.0))
            logits = self.head(pooled)
            if return_features:
                return logits, pooled
            return logits

    return PoseTransformer()


def build_conv_transformer(
    frame_dim: int,
    num_classes: int,
    *,
    hidden_dim: int = 256,
    n_stages: int = 2,
    conv_blocks_per_stage: int = 3,
    n_heads: int = 4,
    kernel_size: int = 17,
    dropout: float = 0.2,
    head_dropout: float = 0.4,
    attention_pool: bool = True,
):
    """GISLR-2023-style Conv1D + Transformer hybrid over per-frame landmark vectors.

    Macro pattern (per the Kaggle ASL Signs 1st-place solution): a stack of
    inverted-residual depthwise-causal Conv1D blocks with ECA channel attention,
    punctuated by a light self-attention block, repeated `n_stages` times, then
    a widening dense layer, masked global average pooling, and a linear head.
    Convolutions encode position, so no positional embedding is needed.
    """
    torch = require_torch()
    nn = torch.nn

    class ECA(nn.Module):
        def __init__(self, kernel: int = 5) -> None:
            super().__init__()
            self.conv = nn.Conv1d(1, 1, kernel, padding=kernel // 2, bias=False)

        def forward(self, x, keep):
            # x: (B, T, C); keep: (B, T, 1) validity mask for a masked mean.
            pooled = (x * keep).sum(dim=1) / keep.sum(dim=1).clamp(min=1.0)  # (B, C)
            weight = torch.sigmoid(self.conv(pooled.unsqueeze(1)).squeeze(1))  # (B, C)
            return x * weight.unsqueeze(1)

    class CausalDWConv1d(nn.Module):
        def __init__(self, channels: int, kernel: int) -> None:
            super().__init__()
            self.pad = kernel - 1
            self.conv = nn.Conv1d(channels, channels, kernel, groups=channels, bias=False)

        def forward(self, x):
            # x: (B, T, C)
            z = x.transpose(1, 2)
            z = nn.functional.pad(z, (self.pad, 0))
            return self.conv(z).transpose(1, 2)

    class Conv1DBlock(nn.Module):
        def __init__(self, dim: int, expand: int = 2) -> None:
            super().__init__()
            inner = dim * expand
            self.expand = nn.Linear(dim, inner)
            self.dwconv = CausalDWConv1d(inner, kernel_size)
            self.bn = nn.BatchNorm1d(inner)
            self.eca = ECA()
            self.project = nn.Linear(inner, dim)
            self.drop = nn.Dropout(dropout)

        def forward(self, x, keep):
            z = nn.functional.silu(self.expand(x))
            z = self.dwconv(z)
            z = self.bn(z.transpose(1, 2)).transpose(1, 2)
            z = self.eca(z, keep)
            z = self.drop(self.project(z))
            return (x + z) * keep  # re-zero padding so masks stay consistent

    class TransformerBlock(nn.Module):
        def __init__(self, dim: int, expand: int = 2) -> None:
            super().__init__()
            self.norm1 = nn.LayerNorm(dim)
            self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
            self.norm2 = nn.LayerNorm(dim)
            self.ffn = nn.Sequential(
                nn.Linear(dim, dim * expand),
                nn.SiLU(),
                nn.Linear(dim * expand, dim),
            )
            self.drop = nn.Dropout(dropout)

        def forward(self, x, keep, pad_mask):
            z = self.norm1(x)
            attn, _ = self.attn(z, z, z, key_padding_mask=pad_mask, need_weights=False)
            x = x + self.drop(attn)
            x = x + self.drop(self.ffn(self.norm2(x)))
            return x * keep

    class TemporalAttentionPool(nn.Module):
        """Content-based attention pooling: learns which frames are discriminative."""

        def __init__(self, dim: int) -> None:
            super().__init__()
            self.score = nn.Linear(dim, 1, bias=False)

        def forward(self, x, keep):
            # x: (B, T, C); keep: (B, T, 1) float validity mask
            scores = self.score(x)  # (B, T, 1)
            scores = scores.masked_fill(keep == 0, -1e4)
            weights = torch.softmax(scores, dim=1)  # (B, T, 1)
            return (x * weights).sum(dim=1)  # (B, C)

    class ConvTransformer(nn.Module):
        feature_dim = hidden_dim * 2

        def __init__(self) -> None:
            super().__init__()
            self.stem = nn.Linear(frame_dim, hidden_dim)
            stages = []
            for _ in range(n_stages):
                stages.extend(Conv1DBlock(hidden_dim) for _ in range(conv_blocks_per_stage))
                stages.append(TransformerBlock(hidden_dim))
            self.blocks = nn.ModuleList(stages)
            self.top = nn.Linear(hidden_dim, hidden_dim * 2)
            self.pool = TemporalAttentionPool(hidden_dim * 2) if attention_pool else None
            self.head_drop = nn.Dropout(head_dropout)
            self.head = nn.Linear(hidden_dim * 2, num_classes)

        def forward(self, x, return_features: bool = False):
            # x: (B, T, F)
            pad_mask = x.abs().sum(dim=-1) == 0
            all_pad = pad_mask.all(dim=1)
            if all_pad.any():
                pad_mask = pad_mask.clone()
                pad_mask[all_pad, 0] = False
            keep = (~pad_mask).unsqueeze(-1).float()
            z = self.stem(x) * keep
            for block in self.blocks:
                if isinstance(block, TransformerBlock):
                    z = block(z, keep, pad_mask)
                else:
                    z = block(z, keep)
            z = nn.functional.silu(self.top(z))
            if self.pool is not None:
                pooled = self.pool(z, keep)
            else:
                pooled = (z * keep).sum(dim=1) / keep.sum(dim=1).clamp(min=1.0)
            logits = self.head(self.head_drop(pooled))
            if return_features:
                return logits, pooled
            return logits

    return ConvTransformer()


def load_pretrained_encoder(model, encoder_path, *, strict_encoder: bool = False) -> int:
    """Load encoder weights from a Slovo pretraining checkpoint into a UzSL model.

    Only keys that exist in both state dicts AND have matching shapes are loaded;
    the classification head (which has a different output size) is always skipped.
    Returns the number of parameter tensors successfully loaded.
    """
    torch = require_torch()
    saved = torch.load(encoder_path, map_location="cpu", weights_only=False)
    # accept either a raw state dict or a full checkpoint dict
    if isinstance(saved, dict) and "model_state" in saved:
        saved = saved["model_state"]
    target = model.state_dict()
    matched = {
        k: v for k, v in saved.items()
        if k in target and v.shape == target[k].shape and not k.startswith("head.")
    }
    if strict_encoder and len(matched) != len(saved):
        missing = set(saved) - set(matched)
        raise ValueError(f"Strict encoder load failed; unmatched keys: {missing}")
    target.update(matched)
    model.load_state_dict(target)
    return len(matched)


def build_model(
    input_dim: int,
    num_classes: int,
    hidden_dim: int = 512,
    dropout: float = 0.25,
    *,
    architecture: str = "conv_transformer",
    target_frames: int = 64,
    n_layers: int = 4,
    n_heads: int = 8,
    attention_pool: bool = True,
):
    if architecture == "mlp":
        return build_mlp(input_dim=input_dim * target_frames, num_classes=num_classes, hidden_dim=hidden_dim, dropout=dropout)
    if architecture == "transformer":
        return build_transformer(
            frame_dim=input_dim,
            num_classes=num_classes,
            target_frames=target_frames,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
        )
    return build_conv_transformer(
        frame_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        n_stages=max(1, n_layers // 2),
        n_heads=n_heads,
        dropout=dropout,
        attention_pool=attention_pool,
    )
