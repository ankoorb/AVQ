import math

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, num_embeddings=512, embedding_dim=64, normalize=True):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        self.encoder = Encoder16x16(embedding_dim=embedding_dim)
        self.quantizer = AttentiveVectorQuantizer(
            num_embeddings=num_embeddings, embedding_dim=embedding_dim, normalize=normalize,
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

    def __init__(self, num_quantizers=3, num_embeddings=512, embedding_dim=64):
        super().__init__()
        self.num_quantizers = num_quantizers
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        self.encoder = Encoder16x16(embedding_dim=embedding_dim)
        self.quantizer = AttentiveResidualVectorQuantizer(
            num_quantizers=num_quantizers,
            num_embeddings=num_embeddings, 
            embedding_dim=embedding_dim, 
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
    def __init__(self, num_embeddings=512, embedding_dim=64, normalize=True, latent_16x16=False):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        self.encoder = Encoder32x32(embedding_dim=embedding_dim) if not latent_16x16 else Encoder128Sq_16x16(embedding_dim=embedding_dim)
        self.quantizer = AttentiveVectorQuantizer(
            num_embeddings=num_embeddings, embedding_dim=embedding_dim, normalize=normalize,
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

    def __init__(self, num_quantizers=3, num_embeddings=1024, embedding_dim=64, latent_16x16=False):
        super().__init__()
        self.num_quantizers = num_quantizers
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        self.encoder = Encoder32x32(embedding_dim=embedding_dim) if not latent_16x16 else Encoder128Sq_16x16(embedding_dim=embedding_dim)
        self.quantizer = AttentiveResidualVectorQuantizer(
            num_quantizers=num_quantizers,
            num_embeddings=num_embeddings, 
            embedding_dim=embedding_dim, 
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


