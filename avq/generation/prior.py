import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass


@dataclass
class PriorConfig:
    vocab_size: int = 1024
    num_classes: int = 200 # 200 in Tiny Imagenet, 100 in MiniPlaces
    seq_length: int = 256 # H * W: 16 x 16 or 32 x 32
    num_codebooks: int = 1 # 1: single-stage, >1: residual

    # Spatial transformer 
    d_model: int = 384
    n_layer: int = 6
    n_head: int = 6
    mlp_ratio: int = 4
    dropout: float = 0.1

    # Depth transformer (used when num_codebooks > 1)
    depth_d_model: int = 192
    depth_n_layer: int =  3
    depth_n_head: int = 4
    depth_mlp_ratio: int = 4
    depth_droput: float = 0.1


class MLP(nn.Module):
    def __init__(self, d_model, mlp_ratio, dropout):
        super().__init__()
        hidden_dim = d_model * mlp_ratio
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, d_model)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class CausalSelfAttention(nn.Module):
    """
    Standard pre-norm causal self-attention module
    """
    def __init__(self, d_model, n_head, dropout):
        super().__init__()
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.residual_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape

        # [B, T, C] -> [B, T, 3 * C] -> [B, T, 3, n_head, head_dim]
        qkv = self.qkv(x).reshape(B, T, 3, self.n_head, self.head_dim)

        # qkv -> permute -> [3, B, n_head, T, head_dim], unpacking over 
        # dim 0 gives q, k, v, each [B, n_head, T, head_dim]
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        dropout_p = self.attn_dropout.p if self.training else 0.0
        output = F.scaled_dot_product_attention(
            q, k, v, dropout_p=dropout_p, is_causal=True,
        )

        output = output.transpose(1, 2).reshape(B, T, C)
        return self.residual_dropout(self.proj(output))


class TransformerBlock(nn.Module):
    """
    Pre-norm transformer block
    """
    def __init__(self, d_model, n_head, mlp_ratio, dropout):
        super().__init__()
        self.pre_ln = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(
            d_model=d_model, n_head=n_head, dropout=dropout,
        )
        self.ln = nn.LayerNorm(d_model)
        self.mlp = MLP(
            d_model=d_model, mlp_ratio=mlp_ratio, dropout=dropout,
        )

    def forward(self, x):
        x = x + self.attn(self.pre_ln(x))
        x = x + self.mlp(self.ln(x))
        return x


class StageTokenEmbeddings(nn.Module):
    """
    Independent learned token embedding table per quantizer stage.

    i/p: indices: [B, L, S]
    o/p: embeddings: [B, L, S, d_model]
    """
    def __init__(self, vocab_size, d_model, num_codebooks):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_codebooks = num_codebooks

        # Embedding tables
        self.tables = nn.ModuleList([
            nn.Embedding(vocab_size, d_model) for _ in range(num_codebooks)
        ])

    def forward(self, indices):
        stage_embeddings = [
            table(indices[:, :, stage]) for stage, table in enumerate(self.tables)
        ]

        return torch.stack(stage_embeddings, dim=2)


# Moshi code: https://github.com/kyutai-labs/moshi/blob/main/moshi/moshi/models/lm.py
class SpatialTransformer(nn.Module):
    """
    Autoregressive transformer over rasterized spatial positions.
    """
    def __init__(self, config, token_embeddings):
        super().__init__()
        pass


class DepthTransformer(nn.Module):
    """
    """
    pass


class TokenPrior(nn.Module):
    """
    """
    pass


def train():
    pass







    

    



