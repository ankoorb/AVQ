import math

import torch

from avq.helpers import _extract_images


def nominal_bpp(config, image_size=64):
    """
    Computes the theoretical bits per pixel (bpp) of the quantized image.
    """
    return config.latent_grid ** 2 * math.log2(config.num_codes) / image_size ** 2


@torch.no_grad()
def assignment_concentration(probs):
    """
    Calculates the concentration of probability mass in the top assignments.

    This metric function evaluates how peaked or concentrated a soft probability 
    distribution is by measuring the average total probability mass captured by 
    the top 1, 5, and 10 predictions. It is particularly useful for monitoring 
    the sharpness of soft quantization or classification assignments during training.
    """
    sorted_probs = probs.sort(dim=-1, descending=True).values

    return {
        "top1_mass": sorted_probs[..., :1].sum(dim=-1).mean().item(),
        "top5_mass": sorted_probs[..., :5].sum(dim=-1).mean().item(),
        "top10_mass": sorted_probs[..., :10].sum(dim=-1).mean().item(),
    }


def get_codebook_norm_metrics(model, method):
    """
    Computes L2 norm statistics (mean, min, max) of the quantizer's 
    codebook vectors.
    """
    if method == "fsq":
        return {}

    codebook = model.quantizer.codebook
    if isinstance(codebook, torch.nn.Embedding):
        codebook = codebook.weight

    with torch.no_grad():
        code_norms = codebook.norm(dim=-1)

        return {
            "mean_code_norm": code_norms.mean().item(),
            "min_code_norm": code_norms.min().item(),
            "max_code_norm": code_norms.max().item(),
        }


def compute_fsq_metrics(indices, scalar_indices, levels):
    """
    Compute FSQ utilization diagnostics.
    """
    num_codes = math.prod(levels)

    # Code utilization
    flat_indices = indices.reshape(-1)
    counts = torch.bincount(flat_indices, minlength=num_codes).float()
    probs = counts / counts.sum().clamp_min(1.0)

    non_zero = probs > 0
    entropy = -(probs[non_zero] * torch.log(probs[non_zero])).sum()
    perplexity = torch.exp(entropy)

    active_codes = (counts > 0).sum()
    dead_codes = num_codes - active_codes
    utilization = active_codes.float() / num_codes

    # Per-scalar dimension entropy
    normalized_entropies = []
    for dim, num_levels in enumerate(levels):

        dim_counts = torch.bincount(scalar_indices[:, dim], minlength=num_levels).float()
        dim_probs = dim_counts / dim_counts.sum().clamp_min(1.0)

        non_zero = dim_probs > 0
        dim_entropy = -(dim_probs[non_zero] * torch.log(dim_probs[non_zero])).sum()
        normalized_entropy = dim_entropy / math.log(num_levels)
        normalized_entropies.append(normalized_entropy)

    mean_scalar_entropy = torch.stack(normalized_entropies).mean()

    return {
        "perplexity": perplexity,
        "active_codes": active_codes.float(),
        "dead_codes": dead_codes.float(),
        "utilization": utilization,
        "mean_scalar_entropy": mean_scalar_entropy,
    }


@torch.no_grad()
def inspect_latent_norms(
    model,
    loader,
    device,
    temperature=0.02,
    soft_weight=0.1,
    max_batches=None,
):
    """
    Compute soft and hard latent-norm statistics across a loader.
    """
    model.eval()

    token_count = 0
    image_count = 0
    soft_sum = torch.zeros((), device=device, dtype=torch.float64)
    hard_sum = torch.zeros((), device=device, dtype=torch.float64)
    soft_squared_sum = torch.zeros((), device=device, dtype=torch.float64)
    hard_squared_sum = torch.zeros((), device=device, dtype=torch.float64)

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images = _extract_images(batch).to(device, non_blocking=True)
        outputs = model(images, temperature=temperature, soft_weight=soft_weight)

        z_soft = outputs["soft_quantized"]
        z_hard = outputs["hard_quantized"]

        # [B, D, H, W] -> [B * H * W, D]
        soft_tokens = z_soft.permute(0, 2, 3, 1).reshape(-1, z_soft.shape[1])
        hard_tokens = z_hard.permute(0, 2, 3, 1).reshape(-1, z_hard.shape[1])

        soft_norms = soft_tokens.norm(dim=-1).double()
        hard_norms = hard_tokens.norm(dim=-1).double()

        soft_sum += soft_norms.sum()
        hard_sum += hard_norms.sum()

        soft_squared_sum += soft_norms.square().sum()
        hard_squared_sum += hard_norms.square().sum()

        token_count += soft_norms.numel()
        image_count += images.shape[0]

    count = float(token_count)
    mean_soft_norm = soft_sum / count
    mean_hard_norm = hard_sum / count

    # Population standard deviation.
    soft_variance = soft_squared_sum / count - mean_soft_norm.square()
    soft_variance = soft_variance.clamp_min(0.0)

    hard_variance = hard_squared_sum / count - mean_hard_norm.square()
    hard_variance = hard_variance.clamp_min(0.0)

    metrics = {
        "num_images": float(image_count),
        "num_tokens": float(token_count),
        "mean_soft_norm": mean_soft_norm.item(),
        "std_soft_norm": soft_variance.sqrt().item(),
        "mean_hard_norm": mean_hard_norm.item(),
        "std_hard_norm": hard_variance.sqrt().item(),
        "hard_soft_norm_ratio": (
            mean_hard_norm / mean_soft_norm.clamp_min(1e-8)
        ).item(),
    }

    print(
        f"Images evaluated: {image_count:,}\n"
        f"Tokens evaluated: {token_count:,}\n"
        f"Mean soft norm: {metrics['mean_soft_norm']:.6f}\n"
        f"Std soft norm: {metrics['std_soft_norm']:.6f}\n"
        f"Mean hard norm: {metrics['mean_hard_norm']:.6f}\n"
        f"Std hard norm: {metrics['std_hard_norm']:.6f}\n"
        f"Hard/soft norm ratio: {metrics['hard_soft_norm_ratio']:.6f}"
    )

    return metrics


@torch.no_grad()
def inspect_soft_hard_alignment(
    model,
    loader,
    device,
    temperature=0.02,
    soft_weight=0.1,
    max_batches=None,
    eps=1e-6,
):
    """
    Measure soft-versus-hard latent alignment over a DataLoader.

    Statistics are averaged over all latent tokens from all evaluated
    images, rather than averaging batch-level means.
    """
    model.eval()

    image_count = 0
    token_count = 0
    cosine_sum = torch.zeros((), dtype=torch.float64, device=device)
    cosine_squared_sum = torch.zeros((), dtype=torch.float64, device=device)
    angle_sum = torch.zeros((), dtype=torch.float64, device=device)
    angle_squared_sum = torch.zeros((), dtype=torch.float64, device=device)
    distance_sum = torch.zeros((), dtype=torch.float64, device=device)
    distance_squared_sum = torch.zeros((), dtype=torch.float64, device=device)

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images = _extract_images(batch).to(device, non_blocking=True)
        outputs = model(images, temperature=temperature, soft_weight=soft_weight)

        z_soft = outputs["soft_quantized"]
        z_hard = outputs["hard_quantized"]

        # [B, D, H, W] -> [B * H * W, D]
        soft_tokens = z_soft.permute(0, 2, 3, 1).reshape(-1, z_soft.shape[1])
        hard_tokens = z_hard.permute(0, 2, 3, 1).reshape(-1, z_hard.shape[1])
        
        cosine_similarity = F.cosine_similarity(
            soft_tokens, hard_tokens, dim=-1, eps=eps,
        ).double()

        # Clamp only for numerical stability before acos
        cos_similarity = cosine_similarity.clamp(min=-1.0, max=1.0)
        angles_degrees = torch.rad2deg(torch.acos(cos_similarity))
        
        squared_distance = (soft_tokens - hard_tokens).square().sum(dim=-1).double()

        cosine_sum += cosine_similarity.sum()
        cosine_squared_sum += cosine_similarity.square().sum()
        angle_sum += angles_degrees.sum()
        angle_squared_sum += angles_degrees.square().sum()
        distance_sum += squared_distance.sum()
        distance_squared_sum += squared_distance.square().sum()

        current_token_count = cosine_similarity.numel()
        token_count += current_token_count
        image_count += images.shape[0]

    count = float(token_count)
    mean_cosine = cosine_sum / count
    mean_angle = angle_sum / count
    mean_squared_distance = distance_sum / count

    cosine_variance = cosine_squared_sum / count - mean_cosine.square()
    cosine_variance = cosine_variance.clamp_min(0.0)
    angle_variance = angle_squared_sum / count - mean_angle.square()
    angle_variance = angle_variance.clamp_min(0.0)

    distance_variance = distance_squared_sum / count - mean_squared_distance.square()
    distance_variance = distance_variance.clamp_min(0.0)

    metrics = {
        "mean_soft_hard_cosine": mean_cosine.item(),
        "std_soft_hard_cosine": cosine_variance.sqrt().item(),
        "mean_soft_hard_angle_degrees": mean_angle.item(),
        "std_soft_hard_angle_degrees": angle_variance.sqrt().item(),
        "mean_soft_hard_squared_distance": mean_squared_distance.item(),
        "std_soft_hard_squared_distance": distance_variance.sqrt().item(),
    }

    print(
        f"Mean soft-hard cosine similarity: "
        f"{metrics['mean_soft_hard_cosine']:.6f}\n"
        f"Std soft-hard cosine similarity: "
        f"{metrics['std_soft_hard_cosine']:.6f}\n"
        f"Mean soft-hard angle: "
        f"{metrics['mean_soft_hard_angle_degrees']:.6f} degrees\n"
        f"Std soft-hard angle: "
        f"{metrics['std_soft_hard_angle_degrees']:.6f} degrees\n"
        f"Mean soft-hard squared distance: "
        f"{metrics['mean_soft_hard_squared_distance']:.6f}\n"
        f"Std soft-hard squared distance: "
        f"{metrics['std_soft_hard_squared_distance']:.6f}"
    )

    return metrics


@torch.no_grad()
def inspect_cosine_logits(model, loader, device, max_batches=None, eps=1e-8):
    """
    Inspect raw encoder-token-to-codebook cosine similarities over
    an entire DataLoader.

    Computes:
        - mean and population standard deviation of all cosine logits;
        - mean top-1 cosine similarity;
        - mean and median top-1/top-2 cosine margin.

    The margin is calculated from raw cosine similarities, before
    temperature scaling.
    """
    model.eval()

    codebook = model.quantizer.codebook

    # Normalize codebook as it remains fixed during evaluation
    codebook_unit = F.normalize(codebook, dim=-1, eps=eps)

    image_count = 0
    token_count = 0
    cosine_value_count = 0
    cosine_sum = torch.zeros((), dtype=torch.float64, device=device)
    cosine_squared_sum = torch.zeros((), dtype=torch.float64, device=device)
    top1_sum = torch.zeros((), dtype=torch.float64, device=device)
    margin_sum = torch.zeros((), dtype=torch.float64, device=device)

    # Store only one scalar margin per latent token.
    margin_chunks = []

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images = _extract_images(batch).to(device, non_blocking=True)

        z = model.encoder(images)
        batch_size, embedding_dim, height, width = z.shape

        # [B, D, H, W] -> [B, H * W, D]
        tokens = z.permute(0, 2, 3, 1).contiguous()
        tokens = tokens.reshape(batch_size, height * width, embedding_dim)
        tokens_unit = F.normalize(tokens, dim=-1, eps=eps)

        # [B, H * W, K]
        cosine_logits = tokens_unit @ codebook_unit.T

        # Get two highest cosine similarities for each latent token.
        top2 = cosine_logits.topk(k=2, dim=-1, largest=True, sorted=True).values

        top1 = top2[..., 0]
        top2_value = top2[..., 1]
        margins = top1 - top2_value

        cosine_logits_double = cosine_logits.double()
        top1_double = top1.double()
        margins_double = margins.double()

        cosine_sum += cosine_logits_double.sum()
        cosine_squared_sum += cosine_logits_double.square().sum()

        top1_sum += top1_double.sum()
        margin_sum += margins_double.sum()

        margin_chunks.append(margins.detach().reshape(-1).float().cpu())
        current_token_count = margins.numel()

        image_count += images.shape[0]
        token_count += current_token_count
        cosine_value_count += cosine_logits.numel()

    mean_cosine = cosine_sum / cosine_value_count

    cosine_variance = cosine_squared_sum / cosine_value_count - mean_cosine.square()
    cosine_variance = cosine_variance.clamp_min(0.0)

    mean_top1 = top1_sum / token_count
    mean_margin = margin_sum / token_count
    all_margins = torch.cat(margin_chunks, dim=0)

    median_margin = all_margins.median()
    margin_q10 = torch.quantile(all_margins, 0.10)
    margin_q90 = torch.quantile(all_margins, 0.90)

    metrics = {
        "num_cosine_values": cosine_value_count,
        "cosine_logit_mean": mean_cosine.item(),
        "cosine_logit_std": cosine_variance.sqrt().item(),
        "mean_top1_cosine": mean_top1.item(),
        "mean_top1_top2_margin": mean_margin.item(),
        "median_top1_top2_margin": median_margin.item(),
        "p10_top1_top2_margin": margin_q10.item(),
        "p90_top1_top2_margin": margin_q90.item(),
    }

    print(
        f"Cosine values evaluated: {cosine_value_count:,}\n"
        f"Cosine-logit mean: "
        f"{metrics['cosine_logit_mean']:.6f}\n"
        f"Cosine-logit std: "
        f"{metrics['cosine_logit_std']:.6f}\n"
        f"Mean top-1 cosine: "
        f"{metrics['mean_top1_cosine']:.6f}\n"
        f"Mean top1-top2 margin: "
        f"{metrics['mean_top1_top2_margin']:.6f}\n"
        f"Median top1-top2 margin: "
        f"{metrics['median_top1_top2_margin']:.6f}\n"
        f"10th-percentile margin: "
        f"{metrics['p10_top1_top2_margin']:.6f}\n"
        f"90th-percentile margin: "
        f"{metrics['p90_top1_top2_margin']:.6f}"
    )

    return metrics


@torch.inference_mode()
def evaluate_avq_diagnostics(
    model,
    loader,
    device,
    temperature=0.02,
    soft_weight=0.2,
):
    metrics = {}

    metrics.update(
        inspect_latent_norms(
            model=model,
            loader=loader,
            device=device,
            temperature=temperature,
            soft_weight=soft_weight,
        )
    )

    metrics.update(
        inspect_soft_hard_alignment(
            model=model,
            loader=loader,
            device=device,
            temperature=temperature,
            soft_weight=soft_weight,
        )
    )

    metrics.update(
        inspect_cosine_logits(
            model=model,
            loader=loader,
            device=device,
        )
    )

    return metrics

