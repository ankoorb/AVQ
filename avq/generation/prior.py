import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass

from avq.generation.utils import time_training_steps


@dataclass
class PriorConfig:
    vocab_size: int = 1024
    num_classes: int = 200 # 200 in Tiny Imagenet, 100 in MiniPlaces
    seq_length: int = 256 # H * W: 16 x 16 or 32 x 32
    num_codebooks: int = 1 # 1: single-stage, >1: residual

    # Spatial transformer 
    d_model: int = 384 # TODO: Check 256 | 320
    n_layer: int = 6
    n_head: int = 6
    mlp_ratio: int = 4
    dropout: float = 0.1

    # Depth transformer (used when num_codebooks > 1)
    depth_d_model: int = 192 # TODO: Check 128 | 160
    depth_n_layer: int =  3
    depth_n_head: int = 4
    depth_mlp_ratio: int = 4
    depth_dropout: float = 0.1


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

    i/p: indices shape: [B, L, S]
    o/p: embeddings shape: [B, L, S, d_model]
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

    For residual representation, one spatial position is represented as:
    E_1[q_1] + E_2[q_2] + ... + E_S[q_S] using independent embedding tables
    E_s for each residual stage.
    """
    def __init__(self, config, token_embeddings):
        super().__init__()
        self.config = config
        self.token_embeddings = token_embeddings
        self.dropout = nn.Dropout(config.dropout)

        # Class conditioning: [class_id] -> [d_model]
        self.class_embedding = nn.Embedding(
            config.num_classes, config.d_model,
        )

        # Learned position embedding as sequence length is fixed
        self.position_embedding = nn.Embedding(
            config.seq_length, config.d_model,
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=config.d_model,
                n_head=config.n_head,
                mlp_ratio=config.mlp_ratio,
                dropout=config.dropout,
            ) for _ in range(config.n_layer)
        ])

        self.ln = nn.LayerNorm(config.d_model)

    def embed_positions(self, indices):
        """
        indices shape: [B, L, S]
        """
        stage_embeddings = self.token_embeddings(indices) # [B, L, S, d_model]

        # E_1[q_1] + E_2[q_2] + ... + E_S[q_S]
        spatial_position_embeddings = stage_embeddings.sum(dim=2) # [B, L, d_model]
        return spatial_position_embeddings

    def forward(self, indices, class_labels):
        """
        i/p:
            indices shape: [B, L, S]
            class_labels (i.e. class ids) shape: [B] 
        o/p: [B, L, d_model]
        """
        B, L, _ = indices.shape
        device = indices.device

        spatial_position_embeddings = self.embed_positions(indices)
        class_token_embeddings = self.class_embedding(class_labels).unsqueeze(1)

        # Teacher forced right shift: input_embeddings[:, 0] = class embeddings 
        # and input_embeddings[:, 1] = embeddings of position i-1 for i > 0
        input_embeddings = torch.cat([
            class_token_embeddings, spatial_position_embeddings[:, :-1, :]
        ], dim=1)

        # Position indices: [0, 1, ..., L-1]
        position_indices = torch.arange(L, device=device).unsqueeze(0)

        # Add learned positional information
        x = input_embeddings + self.position_embedding(position_indices)

        hidden_state = self.dropout(x)
        for block in self.blocks:
            hidden_state = block(hidden_state)

        return self.ln(hidden_state)


class DepthTransformer(nn.Module):
    """
    Small causal transformer over residual depth at each spatial position.

    For stage s at spatial position i it models: P(q_{i, s} | h_i, q_{i, <s})
    where h_i is hidden state produced by spatial transformer from class label
    and all previous spatial positions.

    All spatial positions are processed in parallel during teacher forced training.
    """
    def __init__(self, config, token_embeddings):
        super().__init__()
        self.config = config
        self.token_embeddings = token_embeddings
        self.dropout = nn.Dropout(config.depth_dropout)

        # Projection for spatial transformer hidden states into the smaller
        # depth transformer representation space
        self.spatial_proj = nn.Linear(config.d_model, config.depth_d_model)

        # Projection for residual token embeddings into the depth transformer
        # representation space; identity if dimensions already match.       
        if config.d_model != config.depth_d_model:
            self.token_proj = nn.Linear(config.d_model, config.depth_d_model)
        else:
            self.token_proj = nn.Identity()

        # Learned embedding over stage axis (conceptually like position embedding)
        # Identifies which target residual stage is being predicted
        self.stage_embedding = nn.Embedding(
            config.num_codebooks, config.depth_d_model,
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=config.depth_d_model,
                n_head=config.depth_n_head,
                mlp_ratio=config.depth_mlp_ratio,
                dropout=config.depth_dropout,
            ) for _ in range(config.depth_n_layer)
        ])

        self.ln = nn.LayerNorm(config.depth_d_model)

        # Categorical output head for each stage
        self.stage_heads = nn.ModuleList([
            nn.Linear(config.depth_d_model, config.vocab_size) 
            for _ in range(config.num_codebooks)
        ])

    def forward(self, spatial_hidden_state, indices):
        """
        spatial_hidden_state shape: [B, L, d_model]
        indices shape: [B, L, S]
        """
        B, L, S = indices.shape
        d = self.config.depth_d_model
        device = indices.device

        # Spatial conditioning token for every spatial position
        spatial_condition = self.spatial_proj(spatial_hidden_state).reshape(B * L, 1, d)

        # Stage specific embeddings: [B, L, S, d_model] -> [B * L, S, depth_d_model]
        stage_embeddings = self.token_embeddings(indices)
        stage_embeddings = self.token_proj(stage_embeddings).reshape(B * L, S, d)

        # Depth teacher forcing: stage 0 i/p = spation condition, stage 1 i/p = q_0,
        # stage 2 i/p = q_1. Causal self-attention lets later depth position attend to 
        # all earlier depth inputs, i.e. q_2 can use both q_0 and q_1
        input_embeddings = torch.cat([
            spatial_condition, stage_embeddings[:, :-1, :],
        ], dim=1)

        # Stage indices: [0, 1, ..., S-1]
        stage_indices = torch.arange(S, device=device).unsqueeze(0)

        # Add learned stage position information
        x = input_embeddings + self.stage_embedding(stage_indices)

        hidden_state = self.dropout(x)
        for block in self.blocks:
            hidden_state = block(hidden_state)
            
        hidden_state = self.ln(hidden_state)

        # Stage specific output heads
        stage_logits = [
            self.stage_heads[stage](hidden_state[:, stage, :]) for stage in range(S)
        ]

        logits = torch.stack(stage_logits, dim=1)

        return logits.reshape(B, L, S, self.config.vocab_size)


class TokenPrior(nn.Module):
    """
    Class conditioned token prior.

    Single-stage: 
        Class + previous spatial tokens -> Spatial transformer -> token head

    Residual: 
        Class + previous spatial tokens -> Spatial transformer -> h_i
        h_i + previous residual stages -> Depth transformer -> q_{i, s}
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.token_embeddings = StageTokenEmbeddings(
            vocab_size=config.vocab_size, 
            d_model=config.d_model, 
            num_codebooks=config.num_codebooks,
        )

        self.spatial_transformer = SpatialTransformer(
            config=config, token_embeddings=self.token_embeddings,
        )

        if config.num_codebooks == 1:
            self.depth_transformer = None
            self.single_head = nn.Linear(config.d_model, config.vocab_size)
        else:
            self.depth_transformer = DepthTransformer(
                config=config, token_embeddings=self.token_embeddings,
            )
            self.single_head = None

    def _normalize_indices(self, indices):
        """
        Normalize token tensor to [B, L, S]
        """
        if indices.ndim == 2:
            indices = indices.unsqueeze(-1)

        if indices.dtype != torch.long:
            indices = indices.long()

        return indices

    def num_parameters(self):
        count = sum(parameter.numel() for parameter in self.parameters())
        return count

    def forward(self, indices, class_labels):
        """
        i/p shape:
            indices: 
                Single stage: [B, L] or [B, L, 1]; Residual: [B, L, S]
            class_labels: [B]
        o/p shape:
            Single stage: [B, L, vocab_size]; Residual: [B, L, S, vocab_size]
        """
        indices = self._normalize_indices(indices)
        class_labels = class_labels.long()

        spatial_hidden_state = self.spatial_transformer(indices, class_labels)

        if self.config.num_codebooks == 1:
            return self.single_head(spatial_hidden_state)

        return self.depth_transformer(spatial_hidden_state, indices)

    def compute_loss(self, indices, class_labels):
        indices = self._normalize_indices(indices)
        logits = self.forward(indices, class_labels)

        if self.config.num_codebooks == 1:
            targets = indices.squeeze(-1)
        else:
            targets = indices

        loss = F.cross_entropy(
            logits.reshape(-1, self.config.vocab_size), targets.reshape(-1),
        )

        return loss

    def compute_stage_losses(self, indices, class_labels):
        """
        Optional diagnostic helper. Returns a list of cross entropy loss per
        residual stage.
        """
        indices = self._normalize_indices(indices)
        logits = self.forward(indices, class_labels)

        if self.config.num_codebooks == 1:
            targets = indices.squeeze(-1)
            loss = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size), targets.reshape(-1),
            )
            return [loss]

        losses = []
        for stage in range(self.config.num_codebooks):
            stage_logits = logits[:, :, stage, :].reshape(-1, self.config.vocab_size)
            stage_targets = indices[:, :, stage]
            stage_loss = F.cross_entropy(stage_logits, stage_targets)
            losses.append(stage_loss)

        return losses


def train_prior(model, batch_size, device, timed_steps=300):
    pass


def main():
    pass

if __name__ == "__main__":

    main()
