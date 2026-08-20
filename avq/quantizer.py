# Copyright 2026 Ankoor Bhagat
# SPDX-License-Identifier: Apache-2.0

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from avq.helpers import restore_spatial_shape


class AttentiveVectorQuantizer(nn.Module):
    """
    Attentive Vector Quantizer.
    """
    def __init__(self, num_embeddings=64, embedding_dim=16):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        self.codebook = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        
        # Standard VQ-VAE-style uniform codebook initialization
        # nn.init.uniform_(self.codebook, -1.0 / num_embeddings, 1.0 / num_embeddings)

        # K-independent initialization with approximately unit-norm embeddings.
        # Setting std = 1/sqrt(D) gives E[||e_k||^2] ≈ 1 before L2 normalization.
        nn.init.normal_(self.codebook, mean=0.0, std=1.0 / math.sqrt(embedding_dim))

    def forward(self, z, temperature):
        """
        z shape: [B, D, H, W]
        temperature: softmax temperature
        """
        B, D, H, W = z.shape

        # Reformat the spatial feature map into a flat sequence of tokens
        # [B, D, H, W] -> [B, H, W, D] -> [B, N, D] where N = H * W
        tokens = z.permute(0, 2, 3, 1).contiguous() # [B, H, W, D]
        tokens = tokens.view(B, H * W, D) # [B, N, D]

        # L2 normalize both the input tokens and the codebook embeddings. Bounds the 
        # dot product to [-1, 1], converting it to cosine similarity, which stabilizes 
        # training and helps prevent codebook collapse
        tokens_unit = F.normalize(tokens, dim=-1, eps=1e-6)
        codebook_unit = F.normalize(self.codebook, dim=-1, eps=1e-6)

        # Cross attention: # [B, N, D] @ [D, K] = [B, N, K]
        logits = tokens_unit @ codebook_unit.T

        # Scale logits by temperature and apply softmax to get assignment probabilities.
        # Each spatial token selects among K codebook entries
        probs = F.softmax(logits / temperature, dim=-1)

        # Compute the soft continuous relaxation: Differentiable weighted 
        # codebook representation: [B, N, K] @ [K, D]
        soft_tokens = probs @ codebook_unit # [B, N, D]

        # Compute the hard discrete assignments using the most likely code for each token
        indices = torch.argmax(probs, dim=-1)
        hard_tokens = F.embedding(indices, codebook_unit)

        indices = indices.view(B, H, W)
        soft_quantized = restore_spatial_shape(soft_tokens, B, H, W, D)
        hard_quantized = restore_spatial_shape(hard_tokens, B, H, W, D)

        return {
            "soft_quantized": soft_quantized,
            "hard_quantized": hard_quantized,
            "soft_tokens": soft_tokens,
            "hard_tokens": hard_tokens,
            "indices": indices,
            "probs": probs,
            "logits": logits,
        }


class AttentiveResidualVectorQuantizer(nn.Module):
    """
    Attentive Residual Vector Quantizer.
    """
    def __init__(self, num_quantizers=3, num_embeddings=64, embedding_dim=16):
        super().__init__()
        self.num_quantizers = num_quantizers
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        self.layers = nn.ModuleList([
            AttentiveVectorQuantizer(num_embeddings, embedding_dim) for _ in range(num_quantizers)
        ])

    def forward(self, z, temperature, soft_weight=1.0, mode="mixed"):
        """
        z shape: [B, D, H, W]
        temperature: softmax temperature
        soft_weight: Soft weight used only in mixed mode
        mode: "mixed" (train-time mixture), "soft" (sequential pure-soft ARVQ)
            "hard" (sequential pure-hard ARVQ)
        """
        residual = z
        aggregated_quantized = torch.zeros_like(z)

        stage_probs = []
        stage_indices = []
        for quantizer in self.layers:
            stage_output = quantizer(residual, temperature)

            soft_q = stage_output["soft_quantized"]
            hard_q = stage_output["hard_quantized"]

            if mode == "mixed":
                # Gradients flow only through the soft representation
                stage_quantized = soft_weight * soft_q + (1.0 - soft_weight) * hard_q.detach()
            elif mode == "soft":
                stage_quantized = soft_q
            elif mode == "hard":
                # Hard mode: Used w/ torch.no_grad() during evaluation
                stage_quantized = hard_q.detach()
            else:
                raise ValueError(f"mode {mode!r} is not supported.")

            # Sum quantized contributions across residual stages
            aggregated_quantized = aggregated_quantized + stage_quantized

            # Residual passed to next stage quantizer
            residual = residual - stage_quantized.detach()

            stage_probs.append(stage_output["probs"])
            stage_indices.append(stage_output["indices"])

        stage_indices = torch.stack(stage_indices, dim=1)

        return {
            "z_quantized": aggregated_quantized,
            "residual": residual,
            "indices": stage_indices,
            "all_probs": stage_probs,
        }

