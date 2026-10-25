"""
Use this script to train the Tiny-ImageNet and Mini-Places priors exactly 
as described in the AVQ paper.

Tiny-Imagenet Prior: Estimated Training Times (50 epochs | 1x RTX 2000 Ada):
    - Single-stage: ~7 hours
    - Residual: ~14 hours

Mini-Places Prior: Estimated Training Times (50 epochs | 1x RTX 2000 Ada):
    - Single-stage: ~27 hours
    - Residual: ~52 hours

NOTE: If using this script to train Mini-Places priors, expect longer training times. 
For faster training use RTX 4090 or RTX A5000 or RTX 3090 or A6000.
"""
import math
import time
import logging
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from dataclasses import asdict, dataclass
from torch.utils.data import Dataset, DataLoader

from avq.helpers import set_global_seed
from avq.generation.utils import json_dump, time_training_steps
from avq.generation.paths import EXPERIMENTS, get_token_file, get_prior_dir


# Logging setup
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
)


@dataclass
class PriorConfig:
    vocab_size: int = 1024
    num_classes: int = 200 # 200 in Tiny Imagenet, 100 in MiniPlaces
    sequence_length: int = 256 # H * W: 16 x 16 or 32 x 32
    num_codebooks: int = 1 # 1: single-stage, >1: residual

    # Spatial transformer 
    d_model: int = 384 
    n_layer: int = 6 # 12 for AVQ/ARVQ MiniPlaces ablation
    n_head: int = 6
    mlp_ratio: int = 4
    dropout: float = 0.1

    # Depth transformer (used when num_codebooks > 1)
    depth_d_model: int = 192 
    depth_n_layer: int =  3 # 6 for AVQ/ARVQ MiniPlaces ablation
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
        if d_model % n_head != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_head={n_head}")
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.residual_dropout = nn.Dropout(dropout)

    def forward(self, x, kv_cache=None, use_cache=False):
        B, T, C = x.shape

        # [B, T, C] -> [B, T, 3 * C] -> [B, T, 3, n_head, head_dim]
        qkv = self.qkv(x).reshape(B, T, 3, self.n_head, self.head_dim)

        # qkv -> permute -> [3, B, n_head, T, head_dim], unpacking over 
        # dim 0 gives q, k, v, each [B, n_head, T, head_dim]
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        # KV cache logic
        if use_cache:
            if kv_cache is not None:
                past_k, past_v = kv_cache
                k = torch.cat([past_k, k], dim=2) # T is dim 2
                v = torch.cat([past_v, v], dim=2)

            updated_kv_cache = (k, v)

            # During incremental generation T=1
            output = F.scaled_dot_product_attention(
                q, k, v, dropout_p=0.0, is_causal=False,
            )

        else:
            dropout_p = self.attn_dropout.p if self.training else 0.0
            output = F.scaled_dot_product_attention(
                q, k, v, dropout_p=dropout_p, is_causal=True,
            )
            updated_kv_cache = None

        output = output.transpose(1, 2).reshape(B, T, C)
        output = self.residual_dropout(self.proj(output))

        if use_cache:
            return output, updated_kv_cache

        return output


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

    def forward(self, x, kv_cache=None, use_cache=False):
        if use_cache:
            attn_output, updated_kv_cache = self.attn(
                self.pre_ln(x), kv_cache=kv_cache, use_cache=True,
            )
            x = x + attn_output
            x = x + self.mlp(self.ln(x))
            return x, updated_kv_cache
        
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
            config.sequence_length, config.d_model,
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

    def forward_step(self, indices, class_labels, position, kv_caches):
        """
        Incrementally compute spatial hidden state for one autoregressive
        position using cached attention keys and values.

        i/p:
            indices shape: [B, L, S]
            class_labels (i.e. class ids) shape: [B]
            position: Current spatial position
            kv_caches: List of per layer (k, v) caches
        o/p: [B, 1, d_model]
        """
        device = indices.device

        if position == 0:
            # First autoregressive input is class token for conditioning
            token_embeddings = self.class_embedding(class_labels).unsqueeze(1) # [B, 1, d_model]
        else:
            # Spatial right shift: spatial position i receives the token
            # representation from spatial position i - 1
            previous_indices = indices[:, position - 1 : position, :] # [B, 1, S]
            token_embeddings = self.embed_positions(previous_indices) # [B, 1, d_model]

        position_index = torch.tensor([[position]], dtype=torch.long, device=device)

        x = token_embeddings + self.position_embedding(position_index)
        hidden_state = self.dropout(x)

        updated_kv_caches = []
        for layer_idx, block in enumerate(self.blocks):
            hidden_state, updated_kv_cache = block(
                hidden_state, kv_cache=kv_caches[layer_idx], use_cache=True,
            )
            updated_kv_caches.append(updated_kv_cache)

        hidden_state = self.ln(hidden_state)

        return hidden_state, updated_kv_caches


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

        # Depth teacher forcing: stage 0 i/p = spatial condition, stage 1 i/p = q_0,
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

    def forward_step(self, spatial_hidden_state, indices, stage, kv_caches):
        """
        Incrementally predict one residual stage.

        i/p:
            spatial_hidden_state shape: [B, 1, d_model]
            indices shape: [B, 1, S]
            stage: current residual stage
            kv_caches: List of per layer (k, v) caches
        o/p: [B, vocab_size]

        NOTE: Reset KV cache for every spatial position i (independent depth
              sequence conditioned on h_i)
        """
        device = indices.device

        if stage == 0:
            # First depth autoregressive input is spatial conditioning
            input_embeddings = self.spatial_proj(spatial_hidden_state)
        else:
            # Depth right shift: stage s receives q_{s-1}
            previous_stage_indices = indices[:, 0, stage - 1]
            previous_stage_embeddings = self.token_embeddings.tables[stage-1](
                previous_stage_indices
            )
            input_embeddings = self.token_proj(previous_stage_embeddings).unsqueeze(1)

        stage_index = torch.tensor([[stage]], dtype=torch.long, device=device)

        x = input_embeddings + self.stage_embedding(stage_index)
        hidden_state = self.dropout(x)
        
        updated_kv_caches = []
        for layer_idx, block in enumerate(self.blocks):
            hidden_state, updated_kv_cache = block(
                hidden_state, kv_cache=kv_caches[layer_idx], use_cache=True,
            )
            updated_kv_caches.append(updated_kv_cache)

        hidden_state = self.ln(hidden_state)

        logits = self.stage_heads[stage](hidden_state[:, 0, :])

        return logits, updated_kv_caches


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
        Normalize token tensor to [B, L, S] and validate its shape.
        """
        if indices.ndim == 2:
            if self.config.num_codebooks != 1:
                raise ValueError(
                    f"Residual prior expects [B, L, S] with S={self.config.num_codebooks}, "
                    f"got {tuple(indices.shape)}"
                )
            indices = indices.unsqueeze(-1)

        elif indices.ndim != 3:
            raise ValueError(
                f"Expected [B, L] or [B, L, S], got {tuple(indices.shape)}"
            )

        if indices.shape[1] != self.config.sequence_length:
            raise ValueError(
                f"Expected sequence length {self.config.sequence_length}, "
                f"got {indices.shape[1]}"
            )

        if indices.shape[2] != self.config.num_codebooks:
            raise ValueError(
                f"Expected {self.config.num_codebooks} codebooks, "
                f"got {indices.shape[2]}"
            )

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
            stage_targets = indices[:, :, stage].reshape(-1)
            stage_loss = F.cross_entropy(stage_logits, stage_targets)
            losses.append(stage_loss)

        return losses

    def _sample_from_logits(self, logits, temperature=1.0, top_k=None):
        """
        Sample categorical token IDs from logits.

        logits shape: [B, K]
        temperature:
            > 0: stochastic sampling
            <= 0: greedy argmax
        top_k:
            None: sample from full distribution
            int: restrict sampling to top-k logits
        """
        if temperature <= 0.0:
            return torch.argmax(logits, dim=-1)

        logits = logits / temperature

        if top_k is not None:
            k = min(top_k, logits.shape[-1])
            top_k_logits, _ = torch.topk(logits, k=k, dim=-1) # Values in descending order
            threshold = top_k_logits[:, -1].unsqueeze(-1)
            logits = logits.masked_fill(logits < threshold, float("-inf"))

        probabilities = F.softmax(logits, dim=-1)
        return torch.multinomial(probabilities, num_samples=1).squeeze(-1)

    @torch.inference_mode()
    def generate_legacy(self, class_labels, temperature=1.0, top_k=None):
        """
        Autoregressively generate class conditioned token sequennces 
        (w/o KV cache).

        class_label shape: [B]; already moved to device
        temperature: sampling temperature for Prior distribution
        top_k: optional top-k sampling

        o/p: Single stage: [B, L]; Residual: [B, L, S]
        """
        self.eval()
        device = next(self.parameters()).device # self.parameters() returns iterator

        B = class_labels.shape[0]
        L = self.config.sequence_length
        S = self.config.num_codebooks

        indices = torch.zeros(B, L, S, dtype=torch.long, device=device)
        for position in range(L):
            spatial_hidden_state = self.spatial_transformer(indices, class_labels)

            # [B, 1, d_model]
            h_i = spatial_hidden_state[:, position : position + 1, :]

            # Single stage prior
            if S == 1:
                logits = self.single_head(h_i[:, 0, :]) # [B, K]
                sampled_indices = self._sample_from_logits(
                    logits=logits, temperature=temperature, top_k=top_k,
                )

                indices[:, position, 0] = sampled_indices
                continue

            # Residual prior: Generate q_{i,0}, q_{i,1}, ..., q_{i,S-1}
            # sequentially using current spatial position in depth transformer
            local_indices = indices[:, position : position + 1, :].clone()
            for stage in range(S):

                # [B, 1, S, K]
                depth_logits = self.depth_transformer(h_i, local_indices)

                # P(q_{i,s} | h_i, q_{i,<s})
                logits = depth_logits[:, 0, stage, :] # [B, K]

                sampled_indices = self._sample_from_logits(
                    logits=logits, temperature=temperature, top_k=top_k,
                )

                indices[:, position, stage] = sampled_indices
                local_indices[:, 0, stage] = sampled_indices

        if S == 1:
            return indices.squeeze(-1) # [B, L]

        return indices # [B, L, S]

    @torch.inference_mode()
    def generate(self, class_labels, temperature=1.0, top_k=None):
        """
        Autoregressive generation with KV cache

        class_label shape: [B]; already moved to device
        temperature: sampling temperature for Prior distribution
        top_k: optional top-k sampling

        o/p: Single stage: [B, L]; Residual: [B, L, S]
        """
        self.eval()
        device = next(self.parameters()).device # self.parameters() returns iterator

        B = class_labels.shape[0]
        L = self.config.sequence_length
        S = self.config.num_codebooks

        # KV cache for each spatial transformer block
        spatial_kv_caches = [
            None for _ in self.spatial_transformer.blocks
        ]

        indices = torch.zeros(B, L, S, dtype=torch.long, device=device)
        for position in range(L):
            # Generate only the new spatial hidden state
            h_i, spatial_kv_caches = self.spatial_transformer.forward_step(
                indices=indices,
                class_labels=class_labels,
                position=position,
                kv_caches=spatial_kv_caches,
            )

            # Single stage prior
            if S == 1:
                logits = self.single_head(h_i[:, 0, :]) # [B, K]
                sampled_indices = self._sample_from_logits(
                    logits=logits, temperature=temperature, top_k=top_k,
                )

                indices[:, position, 0] = sampled_indices
                continue

            # Residual prior: Generate q_{i,0}, q_{i,1}, ..., q_{i,S-1}
            # sequentially using current spatial position in depth transformer
            local_indices = indices[:, position : position + 1, :].clone()

            # Depth sequence starts fresh at every spatial position i
            # so reset KV cache for every spatial position i
            depth_kv_caches = [None for _ in self.depth_transformer.blocks]
            for stage in range(S):

                # # P(q_{i,s} | h_i, q_{i,<s}); shape: [B, K]
                depth_logits, depth_kv_caches = self.depth_transformer.forward_step(
                    spatial_hidden_state=h_i,
                    indices=local_indices,
                    stage=stage,
                    kv_caches=depth_kv_caches,
                )

                sampled_indices = self._sample_from_logits(
                    logits=depth_logits, temperature=temperature, top_k=top_k,
                )

                indices[:, position, stage] = sampled_indices
                local_indices[:, 0, stage] = sampled_indices

        if S == 1:
            return indices.squeeze(-1) # [B, L]

        return indices # [B, L, S]


class CachetTokenDataset(Dataset):

    def __init__(self, tokens, labels):
        self.tokens = tokens
        self.labels = labels

    def __len__(self):
        return self.tokens.shape[0]

    def __getitem__(self, idx):
        return {
            "tokens": self.tokens[idx].long(),
            "labels": self.labels[idx].long(),
        }


def get_token_dataloaders(token_data, generator, loader_metadata):
    train_dataset = CachetTokenDataset(
        tokens=token_data["train_tokens"], labels=token_data["train_labels"]
    )

    val_dataset = CachetTokenDataset(
        tokens=token_data["val_tokens"], labels=token_data["val_labels"]
    )

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=generator,
        drop_last=True,
        **loader_metadata,
    )

    val_dataloader = DataLoader(
        val_dataset,
        shuffle=False,
        drop_last=False,
        **loader_metadata,
    )

    return train_dataloader, val_dataloader


def compute_nll_metrics(model, loader, device):
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    stage_nll = torch.zeros(model.config.num_codebooks, dtype=torch.float64)
    stage_count = torch.zeros(model.config.num_codebooks, dtype=torch.int64)

    with torch.inference_mode():
        for batch in tqdm(loader, desc="Validation", dynamic_ncols=True):
            tokens = batch["tokens"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            # Compute overall Loss
            loss = model.compute_loss(tokens, labels)

            batch_tokens = tokens.numel()
            total_nll += loss.item() * batch_tokens
            total_tokens += batch_tokens

            # Compute per-stage loss. TODO: Optimize as loss is recomputed
            stage_losses = model.compute_stage_losses(tokens, labels)

            # Number of spatial tokens in this batch (Batch * Length)
            spatial_tokens = tokens.size(0) * tokens.size(1) 
            for i, stage_loss in enumerate(stage_losses):
                stage_nll[i] += stage_loss.item() * spatial_tokens
                stage_count[i] += spatial_tokens

        mean_nll = total_nll / total_tokens
        mean_stage_nll = (stage_nll / stage_count.clamp_min(1)).tolist()

        result = {
            "prior_nll": mean_nll,
            "prior_ppl": math.exp(mean_nll),
            "per_stage_prior_nll": mean_stage_nll,
            "per_stage_prior_ppl": [math.exp(nll) for nll in mean_stage_nll]
        }

    return result
    

def train_prior(
    epochs, 
    model, 
    optimizer, 
    scheduler, 
    train_loader, 
    val_loader, 
    output_dir, 
    device,
    config,
    args,
    metadata,
):
    best_path = output_dir / "prior_best.pt"
    last_path = output_dir / "prior_last.pt"

    history = []
    best_epoch = -1
    best_val_nll = float("inf")
    entry = EXPERIMENTS[args.experiment_id]
    experiment_name = f"{entry['experiment']}_prior"

    for epoch in range(epochs):
        model.train()
        start = time.time()
        epoch_lr = optimizer.param_groups[0]["lr"]

        running_nll = 0.0
        running_tokens = 0

        progress = tqdm(
            train_loader, desc=f"Prior epoch {epoch:03d}/{epochs:03d}", dynamic_ncols=True,
        )

        for batch in progress:
            tokens = batch["tokens"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # Run forward pass and calculate loss
            loss = model.compute_loss(tokens, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            num_tokens = tokens.numel()
            running_nll += loss.item() * num_tokens
            running_tokens += num_tokens

            progress.set_postfix(nll=f"{running_nll / running_tokens:.4f}")

        scheduler.step()

        train_nll = running_nll / running_tokens
        train_ppl = math.exp(train_nll)
        val_metrics = compute_nll_metrics(
            model=model,
            loader=val_loader,
            device=device,
        )

        epoch_record = {
            "epoch": epoch,
            "train_prior_nll": train_nll,
            "train_prior_ppl": train_ppl,
            "val_prior_nll": val_metrics["prior_nll"],
            "val_prior_ppl": val_metrics["prior_ppl"],
            "val_per_stage_prior_nll": val_metrics["per_stage_prior_nll"],
            "val_per_stage_prior_ppl": val_metrics["per_stage_prior_ppl"],
            "duration": time.time() - start,
            "learning_rate": epoch_lr,
        }

        history.append(epoch_record)

        logger.info(
            f"Epoch {epoch:03d} | train prior NLL {train_nll:.4f} prior PPL {train_ppl:.2f} | "
            f"val prior NLL {val_metrics['prior_nll']:.4f} prior PPL {val_metrics['prior_ppl']:.2f} | "
            f"{epoch_record['duration']:.1f}s"
        )

        checkpoint = {
            "experiment": experiment_name,
            "model_state_dict": model.state_dict(),
            "prior_config": asdict(config),
            "token_metadata": metadata,
            "seed": args.model_seed,
            "epoch": epoch,
            "train_prior_nll": train_nll,
            "val_metrics": val_metrics,
            "parameter_count": model.num_parameters(),
            "training_args": vars(args),
        }

        torch.save(checkpoint, last_path)

        if val_metrics["prior_nll"] < best_val_nll:
            best_val_nll = val_metrics["prior_nll"]
            best_epoch = epoch
            checkpoint["best_val_prior_nll"] = best_val_nll
            checkpoint["best_epoch"] = best_epoch
            torch.save(checkpoint, best_path)

    output = {
        "seed": args.model_seed,
        "experiment": experiment_name,
        "best_epoch": best_epoch,
        "best_val_prior_nll": best_val_nll,
        "best_val_prior_ppl": math.exp(best_val_nll),
        "prior_config": asdict(config),
        "parameter_count": model.num_parameters(),
        "token_metadata": metadata,
        "history": history,
    }

    return output


def load_prior(experiment_name, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Consistency check
    prior_experiment_name = f"{experiment_name}_prior"
    checkpoint_experiment_name = checkpoint["experiment"]
    assert prior_experiment_name == checkpoint_experiment_name

    # Get prior config and load prior
    prior_config = PriorConfig(**checkpoint["prior_config"])
    prior = TokenPrior(prior_config).to(device)
    prior.load_state_dict(checkpoint["model_state_dict"], strict=True)
    prior.eval()

    return prior, prior_config, checkpoint


def main():
    # 0. Parse CLI arguments
    parser = argparse.ArgumentParser(description="Train class-conditional causal token prior.")
    parser.add_argument("--project_root", type=str, required=True)
    parser.add_argument(
        "--experiment_id", type=int, default=0, choices=EXPERIMENTS.keys(), help="Experiment ID from PRIOR_EXPERIMENTS",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--d_model", type=int, default=384)
    parser.add_argument("--n_layer", type=int, default=6, help="Spatial Transformer number of layers")
    parser.add_argument("--depth_d_model", type=int, default=192)
    parser.add_argument("--depth_n_layer", type=int, default=3, help="Depth Transformer number of layers")
    parser.add_argument("--model_seed", type=int, default=42)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--min_learning_rate", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    args = parser.parse_args()

    # 1. Reproducibility
    DATA_SEED = 15
    model_seed = args.model_seed
    set_global_seed(model_seed)

    # 2. Setup
    project_root = args.project_root
    entry = EXPERIMENTS[args.experiment_id]
    dataset = entry["dataset"]
    experiment_name = entry["experiment"]
    token_file_path = get_token_file(
        root_dir=project_root,
        dataset=dataset, 
        model_seed=model_seed,
        experiment=experiment_name,
    )

    output_dir = get_prior_dir(
        root_dir=project_root, 
        dataset=dataset, 
        model_seed=model_seed, 
        experiment=experiment_name
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    token_data = torch.load(token_file_path, map_location="cpu")
    metadata = token_data["metadata"]

    config = PriorConfig(
        vocab_size=metadata["vocab_size"],
        num_classes=metadata["num_classes"],
        sequence_length=metadata["sequence_length"],
        num_codebooks=metadata["num_stages"],
        d_model=args.d_model,
        n_layer=args.n_layer,
        depth_d_model=args.depth_d_model,
        depth_n_layer=args.depth_n_layer,
    )

    batch_size = args.batch_size
    if batch_size is None:
        batch_size = 64 if config.sequence_length == 256 else 32

    generator = torch.Generator()
    generator.manual_seed(DATA_SEED)

    loader_metadata = {
        "batch_size": batch_size,
        "num_workers": 4,
        "pin_memory": device.type == "cuda",
        "persistent_workers": True,
    }

    # 3. Get data loaders
    train_loader, val_loader = get_token_dataloaders(
        token_data=token_data,
        generator=generator,
        loader_metadata=loader_metadata,
    )

    # 4. Initialize model, optimizer and scheduler
    model = TokenPrior(config=config).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.min_learning_rate,
    )

    # 5. Launch training
    logger.info(
        f"Prior training | method={metadata['method']} | stages={config.num_codebooks} | "
        f"L={config.sequence_length} | K={config.vocab_size} | batch={batch_size} "
    )

    train_output = train_prior(
        epochs=args.epochs,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=output_dir,
        device=device,
        config=config,
        args=args,
        metadata=metadata,
    )

    # 6. Write output
    json_dump(train_output, output_dir / "summary.json")
    logger.info("Experiment '%s' completed successfully.", experiment_name)


if __name__ == "__main__":

    main()
