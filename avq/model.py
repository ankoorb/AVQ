import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from diveq import DIVEQ, ResidualDIVEQ

from avq.helpers import restore_spatial_shape
from avq.quantizer import AttentiveVectorQuantizer, AttentiveResidualVectorQuantizer


class ResidualBlock(nn.Module):
    """
    Residual convolutional block that preserves spatial resolution
    and channel count
    """
    def __init__(self, channels, num_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.GroupNorm(num_groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)


class AVQAutoencoder8x8(nn.Module):
    """
    Single-stage attentive vector-quantized autoencoder for
    64 x 64 Tiny ImageNet and COCO 2017 images.

    input: [B, 3, 64, 64]
    output: [B, embedding_dim, 8, 8] -> each image produces 64
        discrete latent tokens
    """
    def __init__(self, num_embeddings=512, embedding_dim=64, normalize=True):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        self.encoder = nn.Sequential(
            # [B, 3, 64, 64] -> [B, 64, 64, 64]
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            ResidualBlock(64),
            
            # [B, 64, 64, 64] -> [B, 128, 32, 32]
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
            ResidualBlock(128),

            # [B, 128, 32, 32] -> [B, 256, 16, 16]
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
            ResidualBlock(256),

            # [B, 256, 16, 16] -> [B, 256, 8, 8]
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
            ResidualBlock(256),

            # Map encoder channels to quantizer dimension
            nn.GroupNorm(8, 256),
            nn.SiLU(),
            nn.Conv2d(in_channels=256, out_channels=embedding_dim, kernel_size=1, stride=1, padding=0),
        )

        self.quantizer = AttentiveVectorQuantizer(
            num_embeddings=num_embeddings, embedding_dim=embedding_dim, normalize=normalize,
        )

        self.decoder = nn.Sequential(
            # [B, D, 8, 8] -> [B, 256, 8, 8]
            nn.Conv2d(in_channels=embedding_dim, out_channels=256, kernel_size=3, stride=1, padding=1),
            ResidualBlock(256),

            # [B, 256, 8, 8] -> [B, 256, 16, 16]
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
            ResidualBlock(256),

            # [B, 256, 8, 8] -> [B, 128, 32, 32]
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
            ResidualBlock(128),

            # [B, 128, 32, 32] -> [B, 64, 64, 64]
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
            ResidualBlock(64),

            # [B, 64, 64, 64] -> [B, 3, 64, 64]
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1),

            # Dataset is normalized to [-1, 1]
            nn.Tanh()
        )

    def forward(self, x, temperature=1.0, soft_weight=1.0, return_train_reconstruction=True, return_separate_reconstructions=False):
        """
        Run the encoder, attentive quantizer, and decoder forward pass.
        """
        # Extract continuous latent representations from the input image
        encoded = self.encoder(x)

        # Quantize the latents using the attentive vector quantizer
        quantizer_outputs = self.quantizer(
            encoded, temperature=temperature
        )

        soft_quantized = quantizer_outputs["soft_quantized"]
        hard_quantized = quantizer_outputs["hard_quantized"]

        # Detach the complete hard branch to prevent gradients from flowing backward
        hard_quantized_detached = hard_quantized.detach()

        # STE-free convex mixture. The decoder is progressively exposed to true hard 
        # code vectors, while encoder/codebook gradients flow only through the actual 
        # soft contribution and are scaled by soft_weight.
        train_quantized = (
            soft_weight * soft_quantized + (1.0 - soft_weight) * hard_quantized_detached
        )

        outputs = {
            **quantizer_outputs,
            "encoded": encoded,
            "train_quantized": train_quantized,
            "soft_weight": soft_weight,
            "hard_weight": 1 - soft_weight,
        }
        
        # Reconstruction for training
        if return_train_reconstruction:
            outputs["reconstruction_train"] = self.decoder(train_quantized)

        if return_separate_reconstructions:
            outputs["reconstruction_soft"] = self.decoder(soft_quantized)
            outputs["reconstruction_hard"] = self.decoder(hard_quantized)

        return outputs


class PatchDiscriminator8x8(nn.Module):
    """
    Patch discriminator for 64 x 64 RGB images.

    input: [B, 3, 64, 64]
    output: [B, 1, 8, 8] patch logits
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # [B, 3, 64, 64] -> [B, 64, 32, 32]
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            
            # [B, 64, 32, 32] -> [B, 128, 16, 16]
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # [B, 128, 16, 16] -> [B, 256, 8, 8]
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # Preserve 8 x 8 patch logit grid
            nn.Conv2d(in_channels=256, out_channels=1, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        return self.net(x)


class Encoder16x16(nn.Module):
    """
    Encoder for 64 x 64 RGB images.

    Input: [B, 3, 64, 64]
    Output: [B, embedding_dim, 16, 16]
    """
    def __init__(self, embedding_dim=64):
        super().__init__()

        self.net = nn.Sequential(
            # [B, 3, 64, 64] -> [B, 64, 64, 64]
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            ResidualBlock(64),

            # [B, 64, 64, 64] -> [B, 128, 32, 32]
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
            ResidualBlock(128),

            # [B, 128, 32, 32] -> [B, 256, 16, 16]
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
            ResidualBlock(256),

            # Map encoder channels to quantizer dimension
            nn.GroupNorm(8, 256),
            nn.SiLU(),
            nn.Conv2d(in_channels=256, out_channels=embedding_dim, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        return self.net(x)


class Decoder16x16(nn.Module):
    """
    Decoder for a 16 x 16 latent representation.

    Input: [B, embedding_dim, 16, 16]
    Output: [B, 3, 64, 64]
    """
    def __init__(self, embedding_dim=64):
        super().__init__()

        self.net = nn.Sequential(
            # [B, D, 16, 16] -> [B, 256, 16, 16]
            nn.Conv2d(in_channels=embedding_dim, out_channels=256, kernel_size=3, stride=1, padding=1),
            ResidualBlock(256),

            # [B, 256, 16, 16] -> [B, 128, 32, 32]
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
            ResidualBlock(128),

            # [B, 128, 32, 32] -> [B, 64, 64, 64]
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
            ResidualBlock(64),

            # [B, 64, 64, 64] -> [B, 3, 64, 64]
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1),

            # Dataset is normalized to [-1, 1]
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z)


class AVQAutoencoder16x16(nn.Module):
    """
    Single-stage attentive vector-quantized autoencoder for
    64 x 64 Tiny ImageNet and COCO 2017 images.

    input: [B, 3, 64, 64]
    output: [B, embedding_dim, 16, 16] -> each image produces 256
        discrete latent tokens
    """
    def __init__(self, num_embeddings=512, embedding_dim=64, normalize=True, learned_code_scale=False):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        self.encoder = Encoder16x16(embedding_dim=embedding_dim)
        self.quantizer = AttentiveVectorQuantizer(
            num_embeddings=num_embeddings, 
            embedding_dim=embedding_dim, 
            normalize=normalize,
            learned_code_scale=learned_code_scale,
        )
        self.decoder = Decoder16x16(embedding_dim=embedding_dim)

    def forward(
        self, 
        x, 
        temperature=1.0, 
        soft_weight=1.0, 
        return_train_reconstruction=True, 
        return_separate_reconstructions=False,
    ):
        """
        Run the encoder, attentive quantizer, and decoder forward pass.
        """
        # Extract continuous latent representations from the input image
        encoded = self.encoder(x)

        # Quantize the latents using the attentive vector quantizer
        quantizer_outputs = self.quantizer(
            encoded, temperature=temperature
        )

        soft_quantized = quantizer_outputs["soft_quantized"]
        hard_quantized = quantizer_outputs["hard_quantized"]

        # Detach the complete hard branch to prevent gradients from flowing backward
        hard_quantized_detached = hard_quantized.detach()

        # STE-free convex mixture. The decoder is progressively exposed to true hard 
        # code vectors, while encoder/codebook gradients flow only through the actual 
        # soft contribution and are scaled by soft_weight.
        train_quantized = (
            soft_weight * soft_quantized + (1.0 - soft_weight) * hard_quantized_detached
        )

        outputs = {
            **quantizer_outputs,
            "encoded": encoded,
            "train_quantized": train_quantized,
            "soft_weight": soft_weight,
            "hard_weight": 1 - soft_weight,
        }

        # Reconstruction for training
        if return_train_reconstruction:
            outputs["reconstruction_train"] = self.decoder(train_quantized)

        if return_separate_reconstructions:
            outputs["reconstruction_soft"] = self.decoder(soft_quantized)
            outputs["reconstruction_hard"] = self.decoder(hard_quantized)

        return outputs


class PatchDiscriminator16x16(nn.Module):
    """
    Patch discriminator for 64 x 64 RGB images.

    input: [B, 3, 64, 64]
    output: [B, 1, 16, 16] patch logits
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # [B, 3, 64, 64] -> [B, 64, 32, 32]
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            
            # [B, 64, 32, 32] -> [B, 128, 16, 16]
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # Preserve 16 x 16 patch logit grid
            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        return self.net(x)


#  Attentive Residual Vector Quantizer
class ARVQAutoencoder16x16(nn.Module):

    def __init__(
        self, 
        num_quantizers=3, 
        num_embeddings=512, 
        embedding_dim=64, 
        learned_code_scale=False,
        use_detach=True,
    ):
        super().__init__()
        self.num_quantizers = num_quantizers
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        self.encoder = Encoder16x16(embedding_dim=embedding_dim)
        self.quantizer = AttentiveResidualVectorQuantizer(
            num_quantizers=num_quantizers,
            num_embeddings=num_embeddings, 
            embedding_dim=embedding_dim, 
            learned_code_scale=learned_code_scale,
            use_detach=use_detach,
        )
        self.decoder = Decoder16x16(embedding_dim=embedding_dim)

    def forward(self, x, temperature=1.0, soft_weight=1.0, mode="mixed"):
        """
        Run the encoder, attentive residual quantizer, and decoder forward pass.
        """
        # Extract continuous latent representations from the input image
        encoded = self.encoder(x)

        # Quantize the latents using the attentive vector quantizer
        quantizer_outputs = self.quantizer(
            encoded, temperature=temperature, soft_weight=soft_weight, mode=mode,
        )

        quantized = quantizer_outputs["z_quantized"]
        reconstruction = self.decoder(quantized)
        
        outputs = {
            **quantizer_outputs,
            "encoded": encoded,
            "quantized": quantized,
            "reconstruction": reconstruction,
            "mode": mode,
        }

        return outputs


# Vector Quantizer based on VQ-VAE by Aäron van den Oord et al.
# Modified from: https://github.com/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb
class VectorQuantizer(nn.Module):
    """
    Standard VQ-VAE quantizer with nearest-neighbor assignment, codebook and
    commitment losses, and straight-through estimator.
    """
    def __init__(self, num_embeddings=64, embedding_dim=16, commitment_cost=0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        
        # Standard VQ-VAE-style uniform codebook initialization
        nn.init.uniform_(self.codebook.weight, -1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z):
        """
        z shape: [B, D, H, W]

        Returns dict containing STE quantized representation, hard quantized 
        representation, indices, and VQ losses
        """
        B, D, H, W = z.shape
        codebook = self.codebook.weight

        # [B, D, H, W] -> [B, H, W, D] -> [B * H * W, D]
        tokens = z.permute(0, 2, 3, 1).contiguous().view(-1, D)
        
        # Compute squared Euclidean distance: ||z - e||^2 = ||z||^2 + ||e||^2 - 2 z^T e
        distances = (
            tokens.pow(2).sum(dim=1, keepdim=True)
            + codebook.pow(2).sum(dim=1)
            - 2.0 * tokens @ codebook.t()
        )

        # Hard nearest-neighbor assignment
        indices = torch.argmin(distances, dim=1)
        hard_tokens = F.embedding(indices, codebook) # [B * H * W, D]

        # Codebook loss: update selected embeddings toward encoder outputs
        codebook_loss = F.mse_loss(hard_tokens, tokens.detach())

        # Commitment loss: encourage encoder outputs toward selected codes
        commitment_loss = F.mse_loss(tokens, hard_tokens.detach())

        vq_loss = codebook_loss + self.commitment_cost * commitment_loss

        # STE: Forward value -> hard_tokens, backward gradients to encoder == identity
        ste_tokens = tokens + (hard_tokens - tokens).detach()

        indices = indices.view(B, H, W)
        ste_quantized = restore_spatial_shape(ste_tokens, B, H, W, D)
        hard_quantized = restore_spatial_shape(hard_tokens, B, H, W, D)

        return {
            "quantized": ste_quantized,
            "hard_quantized": hard_quantized,
            "hard_tokens": hard_tokens,
            "indices": indices,
            "vq_loss": vq_loss,
            "codebook_loss": codebook_loss,
            "commitment_loss": commitment_loss,
        }


class VectorQuantizerEMA(nn.Module):
    """
    VQ-VAE quantizer with nearest-neighbor assignment, EMA codebook and
    commitment losses, and straight-through estimator.
    """
    def __init__(
        self, 
        num_embeddings=64, 
        embedding_dim=16, 
        commitment_cost=0.25,
        decay=0.99,
        epsilon=1e-5,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        self.codebook = nn.Embedding(num_embeddings, embedding_dim)

        # Standard EMA VQ-VAE codebook initialization
        nn.init.normal_(self.codebook.weight)

        # Codebook is updated by EMA (not gradient descent)
        self.codebook.weight.requires_grad_(False)

        # EMA assignment counts
        self.register_buffer("ema_cluster_size", torch.ones(num_embeddings))

        # EMA sum of encoder vectors assigned to each code
        self.register_buffer("ema_codebook_sum", self.codebook.weight.detach().clone())

    def forward(self, z):
        """
        z shape: [B, D, H, W]
        
        Returns dict containing STE quantized representation, hard quantized 
        representation, indices, and VQ losses
        """
        B, D, H, W = z.shape
        codebook = self.codebook.weight

        # [B, D, H, W] -> [B, H, W, D] -> [B * H * W, D]
        tokens = z.permute(0, 2, 3, 1).contiguous().view(-1, D)

        # Compute squared Euclidean distance: ||z - e||^2 = ||z||^2 + ||e||^2 - 2 z^T e
        distances = (
            tokens.pow(2).sum(dim=1, keepdim=True)
            + codebook.pow(2).sum(dim=1)
            - 2.0 * tokens @ codebook.t()
        )

        # Hard nearest-neighbor assignment
        indices = torch.argmin(distances, dim=1)
        hard_tokens = F.embedding(indices, codebook) # [B * H * W, D]

        # EMA codebook update
        if self.training:
            with torch.no_grad():

                # [N] -> [N, K]
                encodings = F.one_hot(indices, num_classes=self.num_embeddings).type_as(tokens)

                # Number of encoder tokens assigned to each code
                cluster_size = encodings.sum(dim=0)

                # Sum of encoder vectors assigned to each code
                codebook_sum = encodings.t() @ tokens

                # Update EMA assignment counts
                self.ema_cluster_size.mul_(self.decay).add_(cluster_size, alpha=1.0 - self.decay)

                # Update EMA vector sums
                self.ema_codebook_sum.mul_(self.decay).add_(codebook_sum, alpha=1.0 - self.decay)

                # Laplace smoothing of cluster sizes
                n = self.ema_cluster_size.sum()

                numerator = self.ema_cluster_size + self.epsilon
                denominator = n + self.num_embeddings * self.epsilon
                smoothed_cluster_size = (numerator / denominator) * n

                # Update codebook from EMA statistics
                self.codebook.weight.copy_(self.ema_codebook_sum / smoothed_cluster_size.unsqueeze(1))

        # Commitment loss: encourage encoder outputs toward selected codes
        commitment_loss = F.mse_loss(tokens, hard_tokens.detach())

        vq_loss = self.commitment_cost * commitment_loss

        # STE: Forward value -> hard_tokens, backward gradients to encoder == identity
        ste_tokens = tokens + (hard_tokens - tokens).detach()

        indices = indices.view(B, H, W)
        ste_quantized = restore_spatial_shape(ste_tokens, B, H, W, D)
        hard_quantized = restore_spatial_shape(hard_tokens, B, H, W, D)

        return {
            "quantized": ste_quantized,
            "hard_quantized": hard_quantized,
            "hard_tokens": hard_tokens,
            "indices": indices,
            "vq_loss": vq_loss,
            "commitment_loss": commitment_loss,
        }


class VQAutoencoder16x16(nn.Module):
    """
    64 x 64 autoencoder using standard VQ + STE with a 16 x 16 discrete latent grid.
    """
    def __init__(self, num_embeddings=1024, embedding_dim=64, commitment_cost=0.25):
        super().__init__()

        self.encoder = Encoder16x16(embedding_dim=embedding_dim)
        self.quantizer = VectorQuantizer(
            num_embeddings=num_embeddings, embedding_dim=embedding_dim, commitment_cost=commitment_cost,
        )
        self.decoder = Decoder16x16(embedding_dim=embedding_dim)

    def forward(self, x, return_hard_reconstruction=False):
        z = self.encoder(x)
        quantizer_outputs = self.quantizer(z)

        # Training path uses STE representation
        reconstruction = self.decoder(quantizer_outputs["quantized"])

        outputs = {
            **quantizer_outputs,
            "reconstruction": reconstruction,
        }

        if return_hard_reconstruction:
            outputs["reconstruction_hard"] = self.decoder(
                quantizer_outputs["hard_quantized"].detach()
            )

        return outputs


class VQEMAAutoencoder16x16(nn.Module):
    """
    64 x 64 autoencoder using EMA VQ + STE with a 16 x 16 discrete latent grid.
    """
    def __init__(
        self,
        num_embeddings=1024,
        embedding_dim=64,
        commitment_cost=0.25,
        decay=0.99,
        epsilon=1e-5,
    ):
        super().__init__()

        self.encoder = Encoder16x16(embedding_dim=embedding_dim)
        self.quantizer = VectorQuantizerEMA(
            num_embeddings=num_embeddings, embedding_dim=embedding_dim, 
            commitment_cost=commitment_cost, decay=decay, epsilon=epsilon,
        )
        self.decoder = Decoder16x16(embedding_dim=embedding_dim)

    def forward(self, x, return_hard_reconstruction=False):
        z = self.encoder(x)
        quantizer_outputs = self.quantizer(z)

        # Training path uses STE representation
        reconstruction = self.decoder(quantizer_outputs["quantized"])

        outputs = {
            **quantizer_outputs,
            "reconstruction": reconstruction,
        }

        if return_hard_reconstruction:
            outputs["reconstruction_hard"] = self.decoder(
                quantizer_outputs["hard_quantized"].detach()
            )

        return outputs


# FSQ 
class FiniteScalarQuantizer(nn.Module):
    """
    Finite Scalar Quantization (FSQ).

    Projects encoder features to a low-dimensional scalar space, independently 
    quantizes each dimension to a fixed number of levels, and projects the quantized 
    representation back to embedding_dim.

    No learned vector codebook or VQ auxiliary loss is used.
    """
    def __init__(self, embedding_dim=64, levels=(8, 5, 5, 5)):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.levels = tuple(levels)
        self.fsq_dim = len(levels)
        self.num_embeddings = math.prod(levels)

        # Project encoder representation to low-dimensional FSQ space
        self.project_in = nn.Linear(embedding_dim, self.fsq_dim)

        # Project quantized FSQ representation back to decoder dimension
        self.project_out = nn.Linear(self.fsq_dim, embedding_dim)

        self.register_buffer("levels_tensor", torch.tensor(levels, dtype=torch.float32))

        # Mixed radix basis for converting scalar codes to one token index
        basis = [1]
        for level in levels[:-1]:
            basis.append(basis[-1] * level)

        self.register_buffer("basis", torch.tensor(basis, dtype=torch.long))

    def forward(self, z):
        """
        z shape: [B, D, H, W]

        Returns:
            quantized: STE representation used for training [B, D, H, W]
            hard_quantized: true discrete representation [B, D, H, W]
            hard_tokens: projected hard FSQ tokens [B*H*W, D]
            fsq_tokens: low-dimensional hard FSQ values [B*H*W, fsq_dim]
            indices: mixed-radix discrete token index [B, H, W]
        """
        B, D, H, W = z.shape

        # [B, D, H, W] -> [B*H*W, D]
        tokens = z.permute(0, 2, 3, 1).contiguous().view(-1, D)

        # [N, D] -> [N, fsq_dim]
        projected_tokens = self.project_in(tokens)

        # Bound each scalar dimension [-1, 1]
        bounded_tokens = torch.tanh(projected_tokens)

        levels = self.levels_tensor.to(device=z.device, dtype=z.dtype)

        # Map [-1, 1] -> [0, L_i - 1]
        scaled_tokens = (bounded_tokens + 1.0) * 0.5 * (levels - 1.0)

        # True discrete scalar indices
        scalar_indices = torch.round(scaled_tokens)
        scalar_indices = torch.minimum(
            torch.maximum(scalar_indices, torch.zeros_like(scalar_indices)), levels - 1.0
        )

        # Convert discrete levels back to [-1, 1]
        hard_fsq_tokens = (2.0 * scalar_indices / (levels - 1.0) - 1.0)

        # STE through scalar rounding: hard values forward, bounded-token gradient backward
        ste_fsq_tokens = (bounded_tokens + (hard_fsq_tokens - bounded_tokens).detach())

        # Project back to decoder embedding dimension
        ste_tokens = self.project_out(ste_fsq_tokens)
        hard_tokens = self.project_out(hard_fsq_tokens)

        # Convert scalar indices to one mixed-radix token index
        scalar_indices_long = (scalar_indices.long())
        indices = (scalar_indices_long * self.basis.unsqueeze(0)).sum(dim=-1)

        indices = indices.view(B, H, W)
        quantized = restore_spatial_shape(ste_tokens, B, H, W, D)
        hard_quantized = restore_spatial_shape(hard_tokens, B, H, W, D)

        return {
            "quantized": quantized,
            "hard_quantized": hard_quantized,
            "hard_tokens": hard_tokens,
            "fsq_tokens": hard_fsq_tokens,
            "scalar_indices": scalar_indices_long,
            "indices": indices,
        }


class FSQAutoencoder16x16(nn.Module):
    """
    64 x 64 autoencoder using FSQ with a 16 x 16 latent grid.
    """
    def __init__(self,
        embedding_dim=64, levels=(8, 5, 5, 5)):
        super().__init__()

        self.encoder = Encoder16x16(embedding_dim=embedding_dim)
        self.quantizer = FiniteScalarQuantizer(embedding_dim=embedding_dim, levels=levels)
        self.decoder = Decoder16x16(embedding_dim=embedding_dim)

    def forward(self, x, return_hard_reconstruction=False):
        z = self.encoder(x)
        quantizer_outputs = self.quantizer(z)

        # FSQ training path uses STE through scalar rounding
        reconstruction = self.decoder(quantizer_outputs["quantized"])

        outputs = {
            **quantizer_outputs,
            "reconstruction": reconstruction,
        }

        if return_hard_reconstruction:
            outputs["reconstruction_hard"] = self.decoder(
                quantizer_outputs["hard_quantized"].detach()
            )

        return outputs


class Encoder32x32(nn.Module):
    """
    Encoder for 128 x 128 RGB images.

    Input: [B, 3, 128, 128]
    Output: [B, embedding_dim, 32, 32]
    """
    def __init__(self, embedding_dim=64):
        super().__init__()

        self.net = nn.Sequential(
            # [B, 3, 128, 128] -> [B, 64, 128, 128]
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            ResidualBlock(64),

            # [B, 64, 128, 128] -> [B, 128, 64, 64]
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
            ResidualBlock(128),

            # [B, 128, 64, 64] -> [B, 256, 32, 32]
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
            ResidualBlock(256),

            # Map encoder channels to quantizer dimension
            nn.GroupNorm(8, 256),
            nn.SiLU(),
            nn.Conv2d(in_channels=256, out_channels=embedding_dim, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        return self.net(x)


class Encoder128Sq_16x16(nn.Module):
    """
    Encoder for 128 x 128 RGB images.

    Input:  [B, 3, 128, 128]
    Output: [B, embedding_dim, 16, 16]
    """
    def __init__(self, embedding_dim=64):
        super().__init__()

        self.net = nn.Sequential(
            # [B, 3, 128, 128] -> [B, 64, 128, 128]
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            ResidualBlock(64),

            # [B, 64, 128, 128] -> [B, 128, 64, 64]
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
            ResidualBlock(128),

            # [B, 128, 64, 64] -> [B, 256, 32, 32]
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
            ResidualBlock(256),

            # [B, 256, 32, 32] -> [B, 256, 16, 16]
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
            ResidualBlock(256),

            nn.GroupNorm(8, 256),
            nn.SiLU(),
            nn.Conv2d(in_channels=256, out_channels=embedding_dim, kernel_size=1),
        )

    def forward(self, x):
        return self.net(x)


class PatchDiscriminator128Sq(nn.Module):
    """
    Patch discriminator for 128 x 128 RGB images.

    input: [B, 3, 128, 128]
    output: [B, 1, 16, 16] patch logits
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # [B, 3, 128, 128] -> [B, 64, 64, 64]
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # [B, 64, 64, 64] -> [B, 128, 32, 32]
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # [B, 128, 32, 32] -> [B, 256, 16, 16]
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # Preserve 16 x 16 patch logit grid
            nn.Conv2d(in_channels=256, out_channels=1, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        return self.net(x)


class Decoder32x32(nn.Module):
    """
    Decoder for a 32 x 32 latent representation.

    Input: [B, embedding_dim, 32, 32]
    Output: [B, 3, 128, 128]
    """
    def __init__(self, embedding_dim=64):
        super().__init__()

        self.net = nn.Sequential(
            # [B, D, 32, 32] -> [B, 256, 32, 32]
            nn.Conv2d(in_channels=embedding_dim, out_channels=256, kernel_size=3, stride=1, padding=1),
            ResidualBlock(256),

            # [B, 256, 32, 32] -> [B, 128, 64, 64]
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
            ResidualBlock(128),

            # [B, 128, 64, 64] -> [B, 64, 128, 128]
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
            ResidualBlock(64),

            # [B, 64, 128, 128] -> [B, 3, 128, 128]
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1),

            # Dataset is normalized to [-1, 1]
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z)


class Decoder128Sq_16x16(nn.Module):
    """
    Decoder for a 16 x 16 latent representation.

    Input: [B, embedding_dim, 16, 16]
    Output: [B, 3, 128, 128]
    """
    def __init__(self, embedding_dim=64):
        super().__init__()

        self.net = nn.Sequential(
            # [B, D, 16, 16] -> [B, 256, 16, 16]
            nn.Conv2d(in_channels=embedding_dim, out_channels=256, kernel_size=3, stride=1, padding=1),
            ResidualBlock(256),

            # [B, 256, 16, 16] -> [B, 256, 32, 32]
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
            ResidualBlock(256),

            # [B, 256, 32, 32] -> [B, 128, 64, 64]
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
            ResidualBlock(128),

            # [B, 128, 64, 64] -> [B, 64, 128, 128]
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
            ResidualBlock(64),

            # [B, 64, 128, 128] -> [B, 3, 128, 128]
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1),

            # Dataset is normalized to [-1, 1]
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z)


class AVQAutoencoder32x32(nn.Module):
    """
    Single-stage attentive vector-quantized autoencoder for
    128 x 128 Mini Places images.

    Default:
        input: [B, 3, 128, 128]
        output: [B, embedding_dim, 32, 32] -> each image produces 1024
        discrete latent tokens

    If latent_16x16=True:
        input: [B, 3, 128, 128]
        output: [B, embedding_dim, 16, 16] -> each image produces 256
        discrete latent tokens
    """
    def __init__(self, num_embeddings=512, embedding_dim=64, normalize=True, latent_16x16=False, learned_code_scale=False):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        self.encoder = Encoder32x32(embedding_dim=embedding_dim) if not latent_16x16 else Encoder128Sq_16x16(embedding_dim=embedding_dim)
        self.quantizer = AttentiveVectorQuantizer(
            num_embeddings=num_embeddings, 
            embedding_dim=embedding_dim, 
            normalize=normalize,
            learned_code_scale=learned_code_scale,
        )
        self.decoder = Decoder32x32(embedding_dim=embedding_dim) if not latent_16x16 else Decoder128Sq_16x16(embedding_dim=embedding_dim)

    def forward(
        self, 
        x, 
        temperature=1.0, 
        soft_weight=1.0, 
        return_train_reconstruction=True, 
        return_separate_reconstructions=False,
    ):
        """
        Run the encoder, attentive quantizer, and decoder forward pass.
        """
        # Extract continuous latent representations from the input image
        encoded = self.encoder(x)

        # Quantize the latents using the attentive vector quantizer
        quantizer_outputs = self.quantizer(
            encoded, temperature=temperature
        )

        soft_quantized = quantizer_outputs["soft_quantized"]
        hard_quantized = quantizer_outputs["hard_quantized"]

        # Detach the complete hard branch to prevent gradients from flowing backward
        hard_quantized_detached = hard_quantized.detach()

        # STE-free convex mixture. The decoder is progressively exposed to true hard 
        # code vectors, while encoder/codebook gradients flow only through the actual 
        # soft contribution and are scaled by soft_weight.
        train_quantized = (
            soft_weight * soft_quantized + (1.0 - soft_weight) * hard_quantized_detached
        )

        outputs = {
            **quantizer_outputs,
            "encoded": encoded,
            "train_quantized": train_quantized,
            "soft_weight": soft_weight,
            "hard_weight": 1 - soft_weight,
        }

        # Reconstruction for training
        if return_train_reconstruction:
            outputs["reconstruction_train"] = self.decoder(train_quantized)

        if return_separate_reconstructions:
            outputs["reconstruction_soft"] = self.decoder(soft_quantized)
            outputs["reconstruction_hard"] = self.decoder(hard_quantized)

        return outputs


class ARVQAutoencoder32x32(nn.Module):

    def __init__(
        self, 
        num_quantizers=3, 
        num_embeddings=1024, 
        embedding_dim=64, 
        latent_16x16=False, 
        learned_code_scale=False,
        use_detach=True,
    ):
        super().__init__()
        self.num_quantizers = num_quantizers
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        self.encoder = Encoder32x32(embedding_dim=embedding_dim) if not latent_16x16 else Encoder128Sq_16x16(embedding_dim=embedding_dim)
        self.quantizer = AttentiveResidualVectorQuantizer(
            num_quantizers=num_quantizers,
            num_embeddings=num_embeddings, 
            embedding_dim=embedding_dim, 
            learned_code_scale=learned_code_scale,
            use_detach=use_detach,
        )
        self.decoder = Decoder32x32(embedding_dim=embedding_dim) if not latent_16x16 else Decoder128Sq_16x16(embedding_dim=embedding_dim)

    def forward(self, x, temperature=1.0, soft_weight=1.0, mode="mixed"):
        """
        Run the encoder, attentive residual quantizer, and decoder forward pass.
        """
        # Extract continuous latent representations from the input image
        encoded = self.encoder(x)

        # Quantize the latents using the attentive residual vector quantizer
        quantizer_outputs = self.quantizer(
            encoded, temperature=temperature, soft_weight=soft_weight, mode=mode,
        )

        quantized = quantizer_outputs["z_quantized"]
        reconstruction = self.decoder(quantized)
        
        outputs = {
            **quantizer_outputs,
            "encoded": encoded,
            "quantized": quantized,
            "reconstruction": reconstruction,
            "mode": mode,
        }

        return outputs


class VQAutoencoder32x32(nn.Module):
    """
    128 x 128 autoencoder using standard VQ + STE with a 32 x 32 discrete latent grid.
    """
    def __init__(self, num_embeddings=1024, embedding_dim=64, commitment_cost=0.25):
        super().__init__()

        self.encoder = Encoder32x32(embedding_dim=embedding_dim)
        self.quantizer = VectorQuantizer(
            num_embeddings=num_embeddings, embedding_dim=embedding_dim, commitment_cost=commitment_cost,
        )
        self.decoder = Decoder32x32(embedding_dim=embedding_dim)

    def forward(self, x, return_hard_reconstruction=False):
        z = self.encoder(x)
        quantizer_outputs = self.quantizer(z)

        # Training path uses STE representation
        reconstruction = self.decoder(quantizer_outputs["quantized"])

        outputs = {
            **quantizer_outputs,
            "reconstruction": reconstruction,
        }

        if return_hard_reconstruction:
            outputs["reconstruction_hard"] = self.decoder(
                quantizer_outputs["hard_quantized"].detach()
            )

        return outputs


class VQEMAAutoencoder32x32(nn.Module):
    """
    128 x 128 autoencoder using EMA VQ + STE with a 32 x 32 discrete latent grid.
    """
    def __init__(
        self,
        num_embeddings=1024,
        embedding_dim=64,
        commitment_cost=0.25,
        decay=0.99,
        epsilon=1e-5,
    ):
        super().__init__()

        self.encoder = Encoder32x32(embedding_dim=embedding_dim)
        self.quantizer = VectorQuantizerEMA(
            num_embeddings=num_embeddings, embedding_dim=embedding_dim, 
            commitment_cost=commitment_cost, decay=decay, epsilon=epsilon,
        )
        self.decoder = Decoder32x32(embedding_dim=embedding_dim)

    def forward(self, x, return_hard_reconstruction=False):
        z = self.encoder(x)
        quantizer_outputs = self.quantizer(z)

        # Training path uses STE representation
        reconstruction = self.decoder(quantizer_outputs["quantized"])

        outputs = {
            **quantizer_outputs,
            "reconstruction": reconstruction,
        }

        if return_hard_reconstruction:
            outputs["reconstruction_hard"] = self.decoder(
                quantizer_outputs["hard_quantized"].detach()
            )

        return outputs


class FSQAutoencoder32x32(nn.Module):
    """
    128 x 128 autoencoder using FSQ with a 32 x 32 latent grid.
    """
    def __init__(self,
        embedding_dim=64, levels=(8, 5, 5, 5)):
        super().__init__()

        self.encoder = Encoder32x32(embedding_dim=embedding_dim)
        self.quantizer = FiniteScalarQuantizer(embedding_dim=embedding_dim, levels=levels)
        self.decoder = Decoder32x32(embedding_dim=embedding_dim)

    def forward(self, x, return_hard_reconstruction=False):
        z = self.encoder(x)
        quantizer_outputs = self.quantizer(z)

        # FSQ training path uses STE through scalar rounding
        reconstruction = self.decoder(quantizer_outputs["quantized"])

        outputs = {
            **quantizer_outputs,
            "reconstruction": reconstruction,
        }

        if return_hard_reconstruction:
            outputs["reconstruction_hard"] = self.decoder(
                quantizer_outputs["hard_quantized"].detach()
            )

        return outputs


class ResidualVectorQuantizerEMA(nn.Module):
    """
    Residual VQ with nearest-neighbor assignment, EMA updated codebook and
    commitment losses, and straight-through estimation.
    """
    def __init__(
        self, 
        num_quantizers=3, 
        num_embeddings=64, 
        embedding_dim=16,
        commitment_cost=0.25,
        decay=0.99,
        epsilon=1e-5,
    ):
        super().__init__()
        self.num_quantizers = num_quantizers
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        self.layers = nn.ModuleList([
            VectorQuantizerEMA(
                num_embeddings=num_embeddings,
                embedding_dim=embedding_dim,
                commitment_cost=commitment_cost,
                decay=decay,
                epsilon=epsilon,
            ) for _ in range(num_quantizers)
        ])

    def forward(self, z):
        """
        z shape: [B, D, H, W]

        Returns:
            quantized: Aggregated STE quantized representation used for training.
            hard_quantized: Aggregated hard quantized representation used for 
                evaluation and decoder calibration.
            hard_tokens: Per-stage hard code vectors with shape [B * H * W, S, D]
            residual: Final residual after all quantization stages.
            indices: Hard code indices with shape [B, S, H, W].
            vq_loss: Mean VQ loss across residual stages.
            commitment_loss: Mean commitment loss across residual stages.
        """
        residual = z
        aggregated_hard_quantized = torch.zeros_like(z)

        stage_indices = []
        stage_hard_tokens = []
        stage_vq_losses = []
        stage_commitment_losses = []
        for quantizer in self.layers:
            stage_output = quantizer(residual)

            hard_quantized = stage_output["hard_quantized"]
            aggregated_hard_quantized = aggregated_hard_quantized + hard_quantized
            
            # Residual passed to next stage quantizer. Detach to prevent 
            # later stages from backpropagating through previous codes
            residual = residual - hard_quantized.detach()

            stage_indices.append(stage_output["indices"])
            stage_hard_tokens.append(stage_output["hard_tokens"])
            stage_vq_losses.append(stage_output["vq_loss"])
            stage_commitment_losses.append(stage_output["commitment_loss"])

        # Single STE after complete residual quantization to avoid scaling by
        # number of stages
        aggregated_quantized = z + (aggregated_hard_quantized - z).detach()

        stage_indices = torch.stack(stage_indices, dim=1) # [B, S, H, W]
        stage_hard_tokens = torch.stack(stage_hard_tokens, dim=1)

        # Average so the loss scale does not grow with the number of residual quantizers
        vq_loss = torch.stack(stage_vq_losses).mean()
        commitment_loss = torch.stack(stage_commitment_losses).mean()
        
        return {
            "quantized": aggregated_quantized,
            "hard_quantized": aggregated_hard_quantized,
            "hard_tokens": stage_hard_tokens,
            "residual": residual,
            "indices": stage_indices,
            "vq_loss": vq_loss,
            "commitment_loss": commitment_loss,
        }


class RVQEMAAutoencoder16x16(nn.Module):
    """
    64 x 64 autoencoder using 3-stage residual EMA VQ + STE with 
    a 16 x 16 latent grid.
    """
    def __init__(
        self, 
        num_quantizers=3, 
        num_embeddings=1024, 
        embedding_dim=64,
        commitment_cost=0.25,
        decay=0.99,
        epsilon=1e-5,
    ):
        super().__init__()

        self.encoder = Encoder16x16(embedding_dim=embedding_dim)
        self.quantizer = ResidualVectorQuantizerEMA(
            num_quantizers=num_quantizers,
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            commitment_cost=commitment_cost,
            decay=decay,
            epsilon=epsilon,
        )
        self.decoder = Decoder16x16(embedding_dim=embedding_dim)

    def forward(self, x, return_hard_reconstruction=False):
        z = self.encoder(x)
        quantizer_outputs = self.quantizer(z)

        # Forward: Sum of hard residual codes; Backward uses single STE
        reconstruction = self.decoder(quantizer_outputs["quantized"])

        outputs = {
            **quantizer_outputs,
            "reconstruction": reconstruction,
        }

        if return_hard_reconstruction:
            outputs["reconstruction_hard"] = self.decoder(
                quantizer_outputs["hard_quantized"].detach()
            )

        return outputs


class RVQEMAAutoencoder32x32(nn.Module):
    """
    128 x 128 autoencoder using 3-stage residual EMA VQ + STE with 
    a 32 x 32 latent grid.
    """
    def __init__(
        self, 
        num_quantizers=3, 
        num_embeddings=1024, 
        embedding_dim=64,
        commitment_cost=0.25,
        decay=0.99,
        epsilon=1e-5,
    ):
        super().__init__()

        self.encoder = Encoder32x32(embedding_dim=embedding_dim)
        self.quantizer = ResidualVectorQuantizerEMA(
            num_quantizers=num_quantizers,
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            commitment_cost=commitment_cost,
            decay=decay,
            epsilon=epsilon,
        )
        self.decoder = Decoder32x32(embedding_dim=embedding_dim)

    def forward(self, x, return_hard_reconstruction=False):
        z = self.encoder(x)
        quantizer_outputs = self.quantizer(z)

        # Forward: Sum of hard residual codes; Backward uses single STE
        reconstruction = self.decoder(quantizer_outputs["quantized"])

        outputs = {
            **quantizer_outputs,
            "reconstruction": reconstruction,
        }

        if return_hard_reconstruction:
            outputs["reconstruction_hard"] = self.decoder(
                quantizer_outputs["hard_quantized"].detach()
            )

        return outputs


class DIVEQAutoencoder16x16(nn.Module):
    """
    Single-stage DIVEQ autoencoder for 64 x 64 Tiny ImageNet and COCO 2017 images.

    input: [B, 3, 64, 64]
    output: [B, embedding_dim, 16, 16] -> each image produces 256 discrete latent tokens
    """
    def __init__(self, num_embeddings=1024, embedding_dim=64):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        self.encoder = Encoder16x16(embedding_dim=embedding_dim) # i/p: [B, 3, 64, 64], o/p: [B, D, 16, 16]
        self.quantizer = DIVEQ(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        self.decoder = Decoder16x16(embedding_dim=embedding_dim) # i/p: [B, D, 16, 16], o/p: [B, 3, 64, 64]

    def forward(self, x, return_hard_reconstruction=False):
        z = self.encoder(x)

        # Reshape z for DIVEQ
        B, D, h, w = z.shape
        z_flat = z.permute(0, 2, 3, 1).reshape(-1, D) # [B * h * w, D]
        
        # Differentiable DiVeQ training representation
        z_q_flat, indices_flat, perplexity = self.quantizer(z_flat)
        z_q = z_q_flat.reshape(B, h, w, D).permute(0, 3, 1, 2).contiguous()
        indices = indices_flat.reshape(B, h, w)
        
        # Reconstruction
        reconstruction = self.decoder(z_q)

        outputs = {
            "quantized": z_q,
            "indices": indices,
            "reconstruction": reconstruction,
            "perplexity_batch": perplexity,
        }

        if return_hard_reconstruction:
            z_q_hard_flat, indices_hard_flat, perplexity_hard = self.quantizer.inference(z_flat)
            z_q_hard = z_q_hard_flat.reshape(B, h, w, D).permute(0, 3, 1, 2).contiguous()
            indices_hard = indices_hard_flat.reshape(B, h, w)
            reconstruction_hard = self.decoder(z_q_hard)

            outputs.update({
                "quantized_hard": z_q_hard,
                "indices_hard": indices_hard,
                "perplexity_hard_batch": perplexity_hard,
                "reconstruction_hard": reconstruction_hard,
            })
            
        return outputs


class ResidualDIVEQAutoencoder16x16(nn.Module):
    """
    Three-stage Residual DiVeQ autoencoder for 64x64 Tiny ImageNet / COCO.

    Input: [B, 3, 64, 64]
    Latent: [B, D, 16, 16]
    Indices: [B, Q, 16, 16]
    """
    def __init__(self, num_codebooks=3, num_embeddings=1024, embedding_dim=64):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.num_codebooks = num_codebooks

        self.encoder = Encoder16x16(embedding_dim=embedding_dim)
        self.quantizer = ResidualDIVEQ(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            num_codebooks=num_codebooks,
            verbose=False,
        )
        self.decoder = Decoder16x16(embedding_dim=embedding_dim)

    def forward(self, x, return_hard_reconstruction=False):
        z = self.encoder(x)

        # Reshape for Residual DiVeQ
        B, D, h, w = z.shape
        z_flat = z.permute(0, 2, 3, 1).reshape(-1, D)

        # Differentiable Residual DiVeQ representation
        z_q_flat, indices_list, perplexity_list = self.quantizer(z_flat)
        z_q = z_q_flat.reshape(B, h, w, D).permute(0, 3, 1, 2).contiguous()
        indices = torch.stack([idx.reshape(B, h, w) for idx in indices_list], dim=1)

        outputs = {
            "quantized": z_q,
            "indices": indices,
            "perplexity_batch": perplexity_list,
            "reconstruction": self.decoder(z_q),
        }

        if return_hard_reconstruction:
            z_q_hard_flat, indices_hard_list, perplexity_hard_list = self.quantizer.inference(z_flat)
            z_q_hard = z_q_hard_flat.reshape(B, h, w, D).permute(0, 3, 1, 2).contiguous()
            indices_hard = torch.stack([idx.reshape(B, h, w) for idx in indices_hard_list], dim=1)

            outputs.update({
                "quantized_hard": z_q_hard,
                "indices_hard": indices_hard,
                "perplexity_hard_batch": perplexity_hard_list,
                "reconstruction_hard": self.decoder(z_q_hard),
            })

        return outputs


class DIVEQAutoencoder32x32(nn.Module):
    """
    Single-stage DIVEQ autoencoder for 128 x 128 MiniPlaces images.

    input: [B, 3, 128, 128]
    output: [B, embedding_dim, 128, 128] -> each image produces 256 discrete latent tokens
    """
    def __init__(self, num_embeddings=1024, embedding_dim=64):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        self.encoder = Encoder32x32(embedding_dim=embedding_dim) 
        self.quantizer = DIVEQ(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        self.decoder = Decoder32x32(embedding_dim=embedding_dim) 

    def forward(self, x, return_hard_reconstruction=False):
        z = self.encoder(x)

        # Reshape z for DIVEQ
        B, D, h, w = z.shape
        z_flat = z.permute(0, 2, 3, 1).reshape(-1, D) # [B * h * w, D]
        
        # Differentiable DiVeQ training representation
        z_q_flat, indices_flat, perplexity = self.quantizer(z_flat)
        z_q = z_q_flat.reshape(B, h, w, D).permute(0, 3, 1, 2).contiguous()
        indices = indices_flat.reshape(B, h, w)
        
        # Reconstruction
        reconstruction = self.decoder(z_q)

        outputs = {
            "quantized": z_q,
            "indices": indices,
            "reconstruction": reconstruction,
            "perplexity_batch": perplexity,
        }

        if return_hard_reconstruction:
            z_q_hard_flat, indices_hard_flat, perplexity_hard = self.quantizer.inference(z_flat)
            z_q_hard = z_q_hard_flat.reshape(B, h, w, D).permute(0, 3, 1, 2).contiguous()
            indices_hard = indices_hard_flat.reshape(B, h, w)
            reconstruction_hard = self.decoder(z_q_hard)

            outputs.update({
                "quantized_hard": z_q_hard,
                "indices_hard": indices_hard,
                "perplexity_hard_batch": perplexity_hard,
                "reconstruction_hard": reconstruction_hard,
            })
            
        return outputs


class ResidualDIVEQAutoencoder32x32(nn.Module):
    """
    Three-stage Residual DiVeQ autoencoder for 128 x 128 MiniPlaces.

    Input: [B, 3, 128, 128]
    Latent: [B, D, 32, 32]
    Indices: [B, Q, 32, 32]
    """
    def __init__(self, num_codebooks=3, num_embeddings=1024, embedding_dim=64):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.num_codebooks = num_codebooks

        self.encoder = Encoder32x32(embedding_dim=embedding_dim)
        self.quantizer = ResidualDIVEQ(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            num_codebooks=num_codebooks,
            verbose=False,
        )
        self.decoder = Decoder32x32(embedding_dim=embedding_dim)

    def forward(self, x, return_hard_reconstruction=False):
        z = self.encoder(x)

        # Reshape for Residual DiVeQ
        B, D, h, w = z.shape
        z_flat = z.permute(0, 2, 3, 1).reshape(-1, D)

        # Differentiable Residual DiVeQ representation
        z_q_flat, indices_list, perplexity_list = self.quantizer(z_flat)
        z_q = z_q_flat.reshape(B, h, w, D).permute(0, 3, 1, 2).contiguous()
        indices = torch.stack([idx.reshape(B, h, w) for idx in indices_list], dim=1)

        outputs = {
            "quantized": z_q,
            "indices": indices,
            "perplexity_batch": perplexity_list,
            "reconstruction": self.decoder(z_q),
        }

        if return_hard_reconstruction:
            z_q_hard_flat, indices_hard_list, perplexity_hard_list = self.quantizer.inference(z_flat)
            z_q_hard = z_q_hard_flat.reshape(B, h, w, D).permute(0, 3, 1, 2).contiguous()
            indices_hard = torch.stack([idx.reshape(B, h, w) for idx in indices_hard_list], dim=1)

            outputs.update({
                "quantized_hard": z_q_hard,
                "indices_hard": indices_hard,
                "perplexity_hard_batch": perplexity_hard_list,
                "reconstruction_hard": self.decoder(z_q_hard),
            })

        return outputs


# Gumbel Softmax - STE
class GumbelSoftmaxVectorQuantizer(nn.Module):
    """
    Gumbel Softmax w/ Straight Through Estimator. Modified from DiVeQ 
    implementation (https://github.com/AaltoML/DiVeQ/blob/main/vqvae_compression/vq.py)

    Based on ST-GS baseline used in DiVeQ
        - Gumbel-Softmax sampling
        - Hard one-hot forward pass
        - Soft Gumbel-Softmax backward gradient (STE)
        - KL-to-uniform regularization
    """
    def __init__(
        self, 
        num_embeddings=1024, 
        embedding_dim=64, 
        kld_scale=5e-4,
        replacement_iters=100,
        discard_threshold=0.01,
        perturb_eps=1e-9,
        codebook_reset=True,
        allow_warning=True,
        verbose=True,      
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.kld_scale = kld_scale
        self.replacement_iters = replacement_iters
        self.discard_threshold = discard_threshold
        self.perturb_eps = perturb_eps
        self.codebook_reset = codebook_reset
        self.allow_warning = allow_warning
        self.verbose = verbose

        self._check_constraints()

        if allow_warning and codebook_reset:
            if replacement_iters < 50:
                warnings.warn(f"`replacement_iters` is set to {replacement_iters},"
                              f" which is quite small. Values < 50 may cause too"
                              f" early and frequent codebook replacements.", UserWarning)
            elif replacement_iters > 300:
                warnings.warn(f"`replacement_iters` is set to {replacement_iters},"
                              f" which is quite large. Values > 300 may cause too"
                              f" late and sporadic codebook replacements.", UserWarning)

            if discard_threshold > 0.05:
                warnings.warn(f"`discard_threshold` is set to"
                              f" {discard_threshold}, which is quite large."
                              f" Values > 0.05 may discard a portion of suitable but"
                              f" rarely used codewords.", UserWarning)

            if perturb_eps > 1e-6:
                warnings.warn(f"`perturb_eps` is set to {perturb_eps}, which"
                              f" is quite large. Values > 1e-6 may cause big"
                              f" perturbation/shift from used codewords.", UserWarning)

    
        # Codebook initialization: Uniform is used as default in DiVeQ
        codebook = (torch.rand((self.num_embeddings, self.embedding_dim))
                        * (1 / self.num_embeddings))

        # Codebook
        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        self.codebook.weight.data = codebook

        # Tensors used for codebook replacement
        self.register_buffer("codebook_usage", torch.zeros(num_embeddings, dtype=torch.int32))
        self.register_buffer("iter_counter", torch.zeros(1, dtype=torch.int32))

    def _check_constraints(self):
        if self.codebook_reset:
            if (self.replacement_iters <= 0) or (type(self.replacement_iters) is not int):
                raise ValueError("`replacement_iters` must be a positive integer value."
                                 " It is recommended that 50 < replacement_iters < 300.")
            if (self.discard_threshold < 0.0) or (self.discard_threshold > 1.0):
                raise ValueError("`discard_threshold` must be in the range of [0,1].")

    def _check_input(self, z):
        if z.ndim != 4:
            raise ValueError("GumbelSoftmax input must have the shape of (B, K, H, W),"
                             " where B: batch size, K: num_embeddings,"
                             " H,W: Latent Height and Width.")
        if z.size(1) != self.num_embeddings:
            raise ValueError(f"GumbelSoftmax input.shape[1] must match the"
                             f" num_embeddings that is {self.num_embeddings}.")

    def forward(self, z, temperature=1.0, hard=True):
        """
        z shape: [B, K, H, W], categorical logits produced by the autoencoder's K-way logit projection
        temperature: Gumbel-Softmax temperature
        hard: True for ST-GS (hard one-hot forward, soft Gumbel gradient backward)
        """
        self._check_input(z)

        # Forward: One hot assignment. Backward: Gradient of soft Gumbel softmax sample
        assignments = F.gumbel_softmax(z, tau=temperature, hard=hard, dim=1) # [B, K, H, W]

        # Quantized latent: q[b, d, h, w] = sum_k assignment[b, k, h, w] * codebook[k, d]
        quantized = torch.einsum("bkhw,kd->bdhw", assignments, self.codebook.weight)

        # Selected hard code indices
        indices = assignments.argmax(dim=1) # [B, H, W]

        # Hard code representation for evaluation and Phase-II decoder calibration
        hard_tokens = F.embedding(indices, self.codebook.weight) # [B, H, W, D]

        hard_quantized = hard_tokens.permute(0, 3, 1, 2).contiguous() # [B, D, H, W]

        # DiVeQ ST-GS KL regularizer: Encourages categorical distribution toward a uniform prior
        qy = F.softmax(z, dim=1)
        log_term = (qy * torch.log(qy * self.num_embeddings + 1e-10)).sum(dim=1).mean()
        kl_loss = self.kld_scale * log_term

        if self.training and self.codebook_reset:
            with torch.no_grad():
                flat_indices = indices.reshape(-1)
                self.codebook_usage[flat_indices] += 1
                self.iter_counter += 1
                if self.iter_counter.item() % self.replacement_iters == 0:
                    self._replace_unused_entries()

        return {
            "quantized": quantized, # Use in Phase-I 
            "hard_quantized": hard_quantized,
            "hard_tokens": hard_tokens,
            "indices": indices, # [B, H, W]
            "logits": z,
            "probs": qy,
            "assignments": assignments,
            "vq_loss": kl_loss,
            "kl_loss": kl_loss,
        }

    @torch.inference_mode()
    def inference(self, z):
        """
        Deterministic hard inference.

        We use a deterministic argmax for hard-code evaluation of the Gumbel-Softmax baseline, 
        rather than the stochastic Gumbel-max sampling in the reference implementation, for 
        consistency with our evaluation protocol across all methods.
        """
        self._check_input(z)
        indices = z.argmax(dim=1) # [B, H, W]
        hard_tokens = F.embedding(indices, self.codebook.weight)
        hard_quantized = hard_tokens.permute(0, 3, 1, 2).contiguous()
        return {
            "quantized": hard_quantized,
            "hard_quantized": hard_quantized, # Used in Phase-II
            "hard_tokens": hard_tokens,
            "indices": indices, # [B, H, W]
            "logits": z,
        }

    def _replace_unused_entries(self):
        with torch.no_grad():
            usage_ratio = self.codebook_usage / self.replacement_iters
            unused_indices = torch.where(usage_ratio < self.discard_threshold)[0]
            used_indices = torch.where(usage_ratio >= self.discard_threshold)[0]

            if unused_indices.numel() == 0 or used_indices.numel() == 0:
                self.codebook_usage.zero_()
                return

            unused_count = unused_indices.numel()
            used_probs = self.codebook_usage[used_indices] / torch.sum(
                self.codebook_usage[used_indices])
            randomly_sampled_indices = used_probs.multinomial(
                num_samples=unused_count, replacement=True)
            sampled_indices = used_indices[randomly_sampled_indices]
            used_codewords = self.codebook.weight.data[sampled_indices].clone()

            self.codebook.weight.data[unused_indices] = (
                used_codewords + self.perturb_eps * torch.randn_like(used_codewords)
            ).clone()
            self.codebook_usage.zero_()

            if self.verbose:
                print(f"\n***** Replaced {unused_count} codewords *****")

    def extra_repr(self):
        return (
            f"num_embeddings={self.num_embeddings}, "
            f"embedding_dim={self.embedding_dim}, "
            f"kld_scale={self.kld_scale}, "
            f"replacement_iters={self.replacement_iters}, "
            f"discard_threshold={self.discard_threshold}, "
            f"perturb_eps={self.perturb_eps}, "
            f"codebook_reset={self.codebook_reset}, "
            f"verbose={self.verbose}"
        )


# Residual Gumbel Softmax - STE
class ResidualGumbelSoftmaxVectorQuantizer(nn.Module):
    """
    Residual Gumbel Softmax Vector Quantizer with Straight Through Estimator.
    Implemented from scratch: DiVeQ's reference code provides residual and product variants 
    of DiVeQ but not of ST-GS; we therefore implemented a residual ST-GS ourselves 
    (independent logit projection and codebook per level, per-level KL averaged across levels, 
    same replacement schedule).

    Residual extension of the ST-GS baseline used in DiVeQ

    At each residual level l:
        - Project residual embedding [B, D, H, W] -> categorical logits [B, K, H, W]
        - Apply Gumbel-Softmax with hard=True
        - Map hard assignment to codebook embedding
        - Subtract quantized representation from residual
        - Continue to next residual level

    Each residual level has:
        - its own D -> K logit projection
        - its own K x D codebook
        - its own codebook usage statistics

    Training:
        - Hard one-hot forward assignments
        - Soft Gumbel-Softmax backward gradients
        - KL-to-uniform regularization

    Inference:
        - Deterministic argmax of learned categorical logits
        - No Gumbel noise
    """

    def __init__(
        self,
        num_embeddings=1024,
        embedding_dim=64,
        num_quantizers=3,
        kld_scale=5e-4,
        replacement_iters=100,
        discard_threshold=0.01,
        perturb_eps=1e-9,
        codebook_reset=True,
        allow_warning=True,
        verbose=True,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.num_quantizers = num_quantizers
        self.kld_scale = kld_scale

        self.replacement_iters = replacement_iters
        self.discard_threshold = discard_threshold
        self.perturb_eps = perturb_eps
        self.codebook_reset = codebook_reset
        self.allow_warning = allow_warning
        self.verbose = verbose

        self._check_constraints()

        if allow_warning and codebook_reset:
            if replacement_iters < 50:
                warnings.warn(
                    f"`replacement_iters` is set to {replacement_iters}, "
                    f"which is quite small. Values < 50 may cause too "
                    f"early and frequent codebook replacements.", UserWarning,
                )
            elif replacement_iters > 300:
                warnings.warn(
                    f"`replacement_iters` is set to {replacement_iters}, "
                    f"which is quite large. Values > 300 may cause too "
                    f"late and sporadic codebook replacements.", UserWarning,
                )

            if discard_threshold > 0.05:
                warnings.warn(
                    f"`discard_threshold` is set to {discard_threshold}, "
                    f"which is quite large. Values > 0.05 may discard a portion "
                    f"of suitable but rarely used codewords.", UserWarning,
                )

            if perturb_eps > 1e-6:
                warnings.warn(
                    f"`perturb_eps` is set to {perturb_eps}, "
                    f"which is quite large. Values > 1e-6 may cause "
                    f"large perturbations from used codewords.", UserWarning,
                )

        # Independent categorical projection for each residual level
        # [B, D, H, W] -> [B, K, H, W]
        self.logit_projs = nn.ModuleList([
            nn.Conv2d(
                in_channels=embedding_dim,
                out_channels=num_embeddings,
                kernel_size=1,
                stride=1,
                padding=0,
            )
            for _ in range(num_quantizers)
        ])

        # Independent codebook for each residual level. Each codebook has shape [K, D]
        # Uniform initialization follows the DiVeQ ST-GS baseline.
        self.codebooks = nn.ModuleList()

        for _ in range(num_quantizers):
            codebook = nn.Embedding(num_embeddings, embedding_dim)
            codebook.weight.data = torch.rand(
                num_embeddings, embedding_dim,
            ) * (1.0 / num_embeddings)
            self.codebooks.append(codebook)

        # Codebook usage: [num_quantizers, num_embeddings]
        # Each residual level tracks dead code usage independently
        self.register_buffer(
            "codebook_usage", torch.zeros(num_quantizers, num_embeddings, dtype=torch.int32),
        )
        self.register_buffer("iter_counter", torch.zeros(1, dtype=torch.int32))

    def _check_constraints(self):
        if self.num_quantizers <= 0 or type(self.num_quantizers) is not int:
            raise ValueError("`num_quantizers` must be a positive integer.")

        if self.codebook_reset:
            if self.replacement_iters <= 0 or type(self.replacement_iters) is not int:
                raise ValueError(
                    "`replacement_iters` must be a positive integer value. "
                    "It is recommended that 50 < replacement_iters < 300."
                )

            if self.discard_threshold < 0.0 or self.discard_threshold > 1.0:
                raise ValueError("`discard_threshold` must be in the range [0, 1].")

    def _check_input(self, z):
        """
        z shape: [B, D, H, W]

        NOTE: Residual Gumbel Softmax receives the encoder embedding, 
            not K-way logits.
        """
        if z.ndim != 4:
            raise ValueError(
                "ResidualGumbelSoftmax input must have shape (B, D, H, W), "
                "where B is batch size, D is embedding_dim, and H,W are latent "
                "spatial dimensions."
            )

        if z.size(1) != self.embedding_dim:
            raise ValueError(
                f"Input channel dimension must match embedding_dim="
                f"{self.embedding_dim}, but got {z.size(1)}."
            )

    def forward(self, z, temperature=1.0, hard=True):
        """
        z: Encoder latent embedding [B, D, H, W]
        temperature: Gumbel-Softmax temperature.
        hard: True for ST-GS: hard one-hot forward pass, soft Gumbel-Softmax 
            backward gradient.
        """
        self._check_input(z)

        # Initial residual
        residual = z # [B, D, H, W]

        quantized_levels = []
        hard_quantized_levels = []
        hard_tokens_levels = []

        indices_levels = []
        logits_levels = []
        probs_levels = []
        assignments_levels = []
        residual_levels = []

        kl_terms = []

        # Residual quantization
        for level in range(self.num_quantizers):

            # Learned categorical logits
            logits = self.logit_projs[level](residual) # [B, K, H, W]

            # Straight-Through Gumbel-Softmax. Forward: hard one hot assignment
            # Backward: soft Gumbel-Softmax gradient
            assignments = F.gumbel_softmax(
                logits, tau=temperature, hard=hard, dim=1,
            ) # [B, K, H, W]

            # Quantized representation for this residual level   
            quantized = torch.einsum(
                "bkhw,kd->bdhw", assignments, self.codebooks[level].weight,
            ) # [B, D, H, W]

            # Hard selected code indices
            indices = assignments.argmax(dim=1) # [B, H, W]

            # Explicit hard representation. For Phase-II decoder calibration.
            hard_tokens = F.embedding(
                indices, self.codebooks[level].weight,
            )  # [B, H, W, D]

            hard_quantized = hard_tokens.permute(0, 3, 1, 2).contiguous() # [B, D, H, W]

            # DiVeQ-style KL-to-uniform regularizer. This operates on the learned 
            # categorical logits, exactly as in the single-stage ST-GS formulation.
            qy = F.softmax(logits, dim=1)
            log_term = torch.log(qy * self.num_embeddings + 1e-10)
            kl_term = (qy * log_term).sum(dim=1).mean()
            kl_terms.append(kl_term)

            # Save per-level outputs
            quantized_levels.append(quantized)
            hard_quantized_levels.append(hard_quantized)
            hard_tokens_levels.append(hard_tokens)

            indices_levels.append(indices)
            logits_levels.append(logits)
            probs_levels.append(qy)
            assignments_levels.append(assignments)

            # Residual update: r_l = r_{l-1} - q_l,subtract the current quantized contribution 
            # using detach() so later residual stages learn only from the remaining residual 
            # and cannot backpropagate through q_l to modify earlier quantizers via the residual path.
            residual = residual - quantized.detach() # ARVQ uses detach. TODO: Non detach ARVQ ablation
            residual_levels.append(residual)

            # Codebook usage tracking for this residual level
            if self.training and self.codebook_reset:
                with torch.no_grad():
                    flat_indices = indices.reshape(-1)
                    self.codebook_usage[level, flat_indices] += 1

        # Total residual representation: z_q = q_1 + q_2 + ... + q_L
        quantized_total = torch.stack(quantized_levels, dim=0).sum(dim=0)

        hard_quantized_total = torch.stack(hard_quantized_levels, dim=0).sum(dim=0)

        # Average KL across residual levels.
        kl_loss = self.kld_scale * torch.stack(kl_terms).mean()

        # Perform code replacement periodically
        if self.training and self.codebook_reset:
            with torch.no_grad():
                self.iter_counter += 1
                if self.iter_counter.item() % self.replacement_iters == 0:
                    self._replace_unused_entries()

        return {
            "quantized": quantized_total, # Cumulative residual quantization, Phase-I
            "hard_quantized": hard_quantized_total, # Used in Phase-II
            "residual": residual,
            "vq_loss": kl_loss,
            "kl_loss": kl_loss,
            "quantized_levels": quantized_levels,
            "hard_quantized_levels": hard_quantized_levels,
            "hard_tokens_levels": hard_tokens_levels,
            "indices_levels": indices_levels,
            "logits_levels": logits_levels,
            "probs_levels": probs_levels,
            "assignments_levels": assignments_levels,
            "residual_levels": residual_levels,
        }

    @torch.inference_mode()
    def inference(self, z):
        """
        Deterministic residual Gumbel Softmax inference.

        At every residual level:
            residual -> learned categorical logits
                     -> argmax
                     -> codebook lookup
                     -> subtract quantized code

        No Gumbel noise is used.
        """
        self._check_input(z)

        residual = z

        quantized_levels = []
        hard_tokens_levels = []
        indices_levels = []
        logits_levels = []
        residual_levels = []

        for level in range(self.num_quantizers):
            logits = self.logit_projs[level](residual)
            indices = logits.argmax(dim=1) # [B, H, W]
            hard_tokens = F.embedding(indices, self.codebooks[level].weight) # [B, H, W, D]
            quantized = hard_tokens.permute(0, 3, 1, 2).contiguous() # [B, D, H, W]

            quantized_levels.append(quantized)
            hard_tokens_levels.append(hard_tokens)
            indices_levels.append(indices)
            logits_levels.append(logits)

            # Residual update
            residual = residual - quantized
            residual_levels.append(residual)

        quantized_total = torch.stack(quantized_levels, dim=0).sum(dim=0)

        return {
            "quantized": quantized_total,
            "hard_quantized": quantized_total, # Used in Phase-II
            "residual": residual,
            "quantized_levels": quantized_levels,
            "hard_quantized_levels": quantized_levels,
            "hard_tokens_levels": hard_tokens_levels,
            "indices_levels": indices_levels,
            "logits_levels": logits_levels,
            "residual_levels": residual_levels,
        }

    def _replace_unused_entries(self):
        """
        Replace underused entries independently for each residual-level codebook.
        """
        with torch.no_grad():
            for level in range(self.num_quantizers):
                usage = self.codebook_usage[level]
                usage_ratio = usage.float() / self.replacement_iters
                unused_indices = torch.where(usage_ratio < self.discard_threshold)[0]
                used_indices = torch.where(usage_ratio >= self.discard_threshold)[0]

                if unused_indices.numel() == 0 or used_indices.numel() == 0:
                    self.codebook_usage[level].zero_()
                    continue

                unused_count = unused_indices.numel()
                used_probs = usage[used_indices].float() / usage[used_indices].sum()
                randomly_sampled_indices = used_probs.multinomial(
                    num_samples=unused_count,
                    replacement=True,
                )

                sampled_indices = used_indices[randomly_sampled_indices]
                used_codewords = self.codebooks[level].weight.data[sampled_indices].clone()

                self.codebooks[level].weight.data[unused_indices] = (
                    used_codewords + self.perturb_eps * torch.randn_like(used_codewords)
                ).clone()

                self.codebook_usage[level].zero_()

                if self.verbose:
                    print(
                        f"\n***** Residual level {level + 1}: "
                        f"Replaced {unused_count} codewords *****"
                    )

    def extra_repr(self):
        return (
            f"num_embeddings={self.num_embeddings}, "
            f"embedding_dim={self.embedding_dim}, "
            f"num_quantizers={self.num_quantizers}, "
            f"kld_scale={self.kld_scale}, "
            f"replacement_iters={self.replacement_iters}, "
            f"discard_threshold={self.discard_threshold}, "
            f"perturb_eps={self.perturb_eps}, "
            f"codebook_reset={self.codebook_reset}, "
            f"verbose={self.verbose}"
        )


class GumbelSoftmaxAutoencoder16x16(nn.Module):

    def __init__(self, num_embeddings=1024, embedding_dim=64):
        super().__init__()

        self.encoder = Encoder16x16(embedding_dim=embedding_dim)

        # Convert encoder representation D to k categorical logits
        self.logit_proj = nn.Conv2d(
            in_channels=embedding_dim,
            out_channels=num_embeddings,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.quantizer = GumbelSoftmaxVectorQuantizer(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            verbose=False,
        )

        self.decoder = Decoder16x16(embedding_dim=embedding_dim)

    def forward(self, x, temperature=1.0, return_hard_reconstruction=False):
        """
        Phase-I: encoder -> logits -> ST-GS -> decoder

        Hard evaluation: encoder -> logits -> argmax -> codebook -> decoder
        """
        # Encode input: [B, 3, 64, 64] -> [B, D, 16, 16]
        z = self.encoder(x)

        # Encoder feature to categorical logits: [B, D, 16, 16] -> [B, K, 16, 16]
        logits = self.logit_proj(z)

        # Quantize
        quantizer_outputs = self.quantizer(logits, temperature=temperature)
        
        # Phase-I Reconstruction
        reconstruction = self.decoder(quantizer_outputs["quantized"])

        outputs = {
            **quantizer_outputs,
            "encoded": z,
            "reconstruction": reconstruction,
        }

        if return_hard_reconstruction:
            quantizer_inference = self.quantizer.inference(logits)
            hard_quantized = quantizer_inference["hard_quantized"]

            reconstruction_hard = self.decoder(hard_quantized.detach())

            outputs.update({
                "hard_quantized": hard_quantized,
                "hard_tokens": quantizer_inference["hard_tokens"],
                "indices": quantizer_inference["indices"],
                "reconstruction_hard": reconstruction_hard,
            })

        return outputs


class GumbelSoftmaxAutoencoder32x32(nn.Module):

    def __init__(self, num_embeddings=1024, embedding_dim=64):
        super().__init__()

        self.encoder = Encoder32x32(embedding_dim=embedding_dim)

        # Convert encoder representation D to k categorical logits
        self.logit_proj = nn.Conv2d(
            in_channels=embedding_dim,
            out_channels=num_embeddings,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.quantizer = GumbelSoftmaxVectorQuantizer(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            verbose=False,
        )

        self.decoder = Decoder32x32(embedding_dim=embedding_dim)

    def forward(self, x, temperature=1.0, return_hard_reconstruction=False):
        """
        Phase-I: encoder -> logits -> ST-GS -> decoder

        Hard evaluation: encoder -> logits -> argmax -> codebook -> decoder
        """
        # Encode input: [B, 3, 128, 128] -> [B, D, 32, 32]
        z = self.encoder(x)

        # Encoder feature to categorical logits: [B, D, 32, 32] -> [B, K, 32, 32]
        logits = self.logit_proj(z)

        # Quantize
        quantizer_outputs = self.quantizer(logits, temperature=temperature)
        
        # Phase-I Reconstruction
        reconstruction = self.decoder(quantizer_outputs["quantized"])

        outputs = {
            **quantizer_outputs,
            "encoded": z,
            "reconstruction": reconstruction,
        }

        if return_hard_reconstruction:
            quantizer_inference = self.quantizer.inference(logits)
            hard_quantized = quantizer_inference["hard_quantized"]

            reconstruction_hard = self.decoder(hard_quantized.detach())

            outputs.update({
                "hard_quantized": hard_quantized,
                "hard_tokens": quantizer_inference["hard_tokens"],
                "indices": quantizer_inference["indices"],
                "reconstruction_hard": reconstruction_hard,
            })

        return outputs


class ResidualGumbelSoftmaxAutoencoder16x16(nn.Module):

    def __init__(self, num_quantizers=3, num_embeddings=1024, embedding_dim=64):
        super().__init__()

        self.encoder = Encoder16x16(embedding_dim=embedding_dim)
        self.quantizer = ResidualGumbelSoftmaxVectorQuantizer(
            num_quantizers=num_quantizers,
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            verbose=False,
        )
        self.decoder = Decoder16x16(embedding_dim=embedding_dim)

    def forward(self, x, temperature=1.0, return_hard_reconstruction=False):
        """
        Phase-I: encoder -> residual ST-GS -> decoder
        Final quantized representation: z_q = q_1 + q_2 + ... + q_L
        Hard evaluation: deterministic per-level argmax
        """
        # Encode input: [B, 3, 64, 64] -> [B, D, 16, 16]
        z = self.encoder(x)

        # Residual ST-GS quantization
        quantizer_outputs = self.quantizer(z, temperature=temperature)

        # Phase-I reconstruction: q_1 + q_2 + ... + q_L
        reconstruction = self.decoder(quantizer_outputs["quantized"])

        outputs = {
            **quantizer_outputs,
            "encoded": z,
            "reconstruction": reconstruction,
        }

        if return_hard_reconstruction:
            quantizer_inference = self.quantizer.inference(z)
            hard_quantized = quantizer_inference["hard_quantized"]
            reconstruction_hard = self.decoder(hard_quantized.detach())

            outputs.update({
                "hard_quantized": hard_quantized,
                "hard_quantized_levels": quantizer_inference["hard_quantized_levels"],
                "hard_tokens_levels":quantizer_inference["hard_tokens_levels"],
                "indices_levels": quantizer_inference["indices_levels"],
                "logits_levels": quantizer_inference["logits_levels"],          
                "residual_levels": quantizer_inference["residual_levels"],
                "reconstruction_hard": reconstruction_hard,
            })

        return outputs
    

class ResidualGumbelSoftmaxAutoencoder32x32(nn.Module):

    def __init__(self, num_quantizers=3, num_embeddings=1024, embedding_dim=64):
        super().__init__()

        self.encoder = Encoder32x32(embedding_dim=embedding_dim)
        self.quantizer = ResidualGumbelSoftmaxVectorQuantizer(
            num_quantizers=num_quantizers,
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            verbose=False,
        )
        self.decoder = Decoder32x32(embedding_dim=embedding_dim)

    def forward(self, x, temperature=1.0, return_hard_reconstruction=False):
        """
        Phase-I: encoder -> residual ST-GS -> decoder
        Final quantized representation: z_q = q_1 + q_2 + ... + q_L
        Hard evaluation: deterministic per-level argmax
        """
        # Encode input: [B, 3, 64, 64] -> [B, D, 16, 16]
        z = self.encoder(x)

        # Residual ST-GS quantization
        quantizer_outputs = self.quantizer(z, temperature=temperature)

        # Phase-I reconstruction: q_1 + q_2 + ... + q_L
        reconstruction = self.decoder(quantizer_outputs["quantized"])

        outputs = {
            **quantizer_outputs,
            "encoded": z,
            "reconstruction": reconstruction,
        }

        if return_hard_reconstruction:
            quantizer_inference = self.quantizer.inference(z)
            hard_quantized = quantizer_inference["hard_quantized"]
            reconstruction_hard = self.decoder(hard_quantized.detach())

            outputs.update({
                "hard_quantized": hard_quantized,
                "hard_quantized_levels": quantizer_inference["hard_quantized_levels"],
                "hard_tokens_levels":quantizer_inference["hard_tokens_levels"],
                "indices_levels": quantizer_inference["indices_levels"],
                "logits_levels": quantizer_inference["logits_levels"],          
                "residual_levels": quantizer_inference["residual_levels"],
                "reconstruction_hard": reconstruction_hard,
            })

        return outputs
    
