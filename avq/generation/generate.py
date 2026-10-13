import torch
import torch.nn.functional as F


def effective_avq_codebook(quantizer):
    codebook = quantizer.codebook

    if getattr(quantizer, "normalize", False):
        codebook = F.normalize(codebook, dim=-1, eps=1e-6)

    if getattr(quantizer, "learned_code_scale", False):
        scale = F.softplus(quantizer.raw_code_scale) + 1e-6
        codebook = scale * codebook

    return codebook


def lookup_spatial_indices(indices, codebook):
    """
    indices shape: [B, H, W]
    codebook shape: [K, D]
    """
    z_q = F.embedding(indices.long(), codebook) # [B, H, W, D]
    return z_q.permute(0, 3, 1, 2).contiguous() # [B, D, H, W]


def decode_hard_indices_to_latent(model, method, indices):
    pass

