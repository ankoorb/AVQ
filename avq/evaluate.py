import math

import torch
import torch.nn.functional as F

from avq.helpers import _extract_images, get_hard_latents_for_calibration


@torch.inference_mode()
def evaluate_hard_mse(model, val_loader, device, temperature=0.02):
    """
    Evaluate pure hard-code reconstruction MSE. MSE is accumulated over all 
    image elements.

    TODO: Remove this
    """
    model.eval()

    total_elements = 0
    total_squared_error = 0.0

    for _, batch in enumerate(val_loader):

        images = _extract_images(batch).to(device, non_blocking=True)

        outputs = model(
            images,
            temperature=temperature,
            soft_weight=0.0, # Decoder receives 100% hard quantized latents
            return_separate_reconstructions=False,
        )

        reconstruction_hard = outputs["reconstruction_train"]

        total_elements += images.numel()
        total_squared_error += F.mse_loss(
reconstruction_hard, images, reduction="sum"
        ).item()

    return total_squared_error / total_elements
        

@torch.no_grad()
def evaluate_avq(model, loader, device, temperature=0.1, soft_weight=0.1, max_batches=None):
    """
    AVQ final evaluation with soft/hard diagnostics.
    """
    model.eval()

    num_codes = model.quantizer.num_embeddings
    soft_usage_sum = torch.zeros(num_codes, device=device, dtype=torch.float64)
    hard_counts = torch.zeros(num_codes, device=device, dtype=torch.long)

    total_tokens = 0
    total_pixels = 0
    soft_squared_error = 0.0
    hard_squared_error = 0.0
    local_entropy_sum = 0.0
    confidence_sum = 0.0

    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break

        # Images are already normalized to [-1, 1].
        images = _extract_images(batch).to(device, non_blocking=True)

        outputs = model(
            images, temperature=temperature, soft_weight=soft_weight, 
            return_train_reconstruction=False, return_separate_reconstructions=True,
        )

        reconstruction_soft = outputs["reconstruction_soft"]
        reconstruction_hard = outputs["reconstruction_hard"]
        probs = outputs["probs"]
        indices = outputs["indices"]

        batch_size, num_tokens, _ = probs.shape
        token_count = batch_size * num_tokens

        soft_usage_sum += probs.double().sum(dim=(0, 1))

        hard_counts += torch.bincount(indices.reshape(-1), minlength=num_codes)

        local_entropy = -(probs * torch.log(probs.clamp_min(1e-10))).sum(dim=-1)
        local_entropy_sum += local_entropy.sum().item()
        
        confidence_sum += probs.max(dim=-1).values.sum().item()

        soft_squared_error += F.mse_loss(reconstruction_soft, images, reduction="sum").item()
        hard_squared_error += F.mse_loss(reconstruction_hard, images, reduction="sum").item()

        total_tokens += token_count
        total_pixels += images.numel()

    # Aggregate soft usage
    soft_usage = soft_usage_sum / total_tokens
    soft_entropy = -torch.sum(soft_usage * torch.log(soft_usage.clamp_min(1e-10)))
    soft_perplexity = torch.exp(soft_entropy)

    # Aggregate hard usage
    hard_usage = hard_counts / hard_counts.sum().clamp_min(1.0)
    hard_entropy = -torch.sum(hard_usage * torch.log(hard_usage.clamp_min(1e-10)))
    hard_perplexity = torch.exp(hard_entropy)

    unique_hard_codes = int(torch.count_nonzero(hard_counts).item())

    local_entropy = local_entropy_sum / total_tokens
    normalized_local_entropy = (local_entropy / math.log(num_codes))

    mean_max_probability = confidence_sum / total_tokens

    soft_mse = soft_squared_error / total_pixels
    hard_mse = hard_squared_error / total_pixels

    return {
        "soft_perplexity": soft_perplexity.item(),
        "hard_perplexity": hard_perplexity.item(),
        "unique_hard_codes": unique_hard_codes,
        "dead_codes": num_codes - unique_hard_codes,
        "local_entropy": local_entropy,
        "normalized_local_entropy": normalized_local_entropy,
        "mean_max_probability": mean_max_probability,
        "soft_mse": soft_mse,
        "hard_mse": hard_mse,
        "soft_hard_mse_gap": hard_mse - soft_mse,
    }


@torch.inference_mode()
def evaluate_validation(model, val_loader, method, device, temperature=0.02):
    """
    Validation for AVQ, VQ, VQ-EMA, and FSQ

    Computes:
        - hard reconstruction MSE
        - hard code perplexity
        - active/dead code counts
        - code utilization

    AVQ additionally reports:
        - normalized local assignment entropy
        - mean maximum assignment probability

    FSQ additionally reports:
        - mean normalized scalar entropy
    """
    model.eval()

    num_codes = model.quantizer.num_embeddings
    hard_counts = torch.zeros(num_codes, device=device, dtype=torch.long)

    total_tokens = 0
    total_elements = 0
    total_squared_error = 0.0
    
    # AVQ statistics
    confidence_sum = 0.0
    local_entropy_sum = 0.0
    
    # FSQ statistics
    if method == "fsq":
        scalar_levels = model.quantizer.levels

        scalar_counts = [
            torch.zeros(n_levels, device=device, dtype=torch.long) for n_levels in scalar_levels
        ]

    for batch in val_loader:

        images = _extract_images(batch).to(device, non_blocking=True)

        # Forward pass
        if method == "avq":
            outputs = model(
                images,
                temperature=temperature,
                soft_weight=0.0, # Decoder receives 100% hard quantized latents
                return_separate_reconstructions=False,
            )

            reconstruction_hard = outputs["reconstruction_train"]

        elif method == "diveq":
            outputs = model(images, return_hard_reconstruction=True)
            reconstruction_hard = outputs["reconstruction_hard"]
            indices = outputs["indices_hard"]
            
        else:
            # VQ/VQ-EMA/FSQ reconstruction is based on hard quantized values in forward pass
            outputs = model(images)
            reconstruction_hard = outputs["reconstruction"]

        indices = outputs["indices"]

        # Reconstruction MSE
        total_squared_error += F.mse_loss(reconstruction_hard, images, reduction="sum").item()
        total_elements += images.numel()

        # Dataset wide hard code usage
        flat_indices = indices.reshape(-1)
        hard_counts += torch.bincount(flat_indices, minlength=num_codes)
        total_tokens += flat_indices.numel()

        # AVQ specific diagnostics
        if method == "avq":
            probs = outputs["probs"]
            local_entropy = -(probs * torch.log(probs.clamp_min(1e-10))).sum(dim=-1)
            local_entropy_sum += local_entropy.sum().item()
            confidence_sum += probs.max(dim=-1).values.sum().item()

        # FSQ-specific diagnostics
        elif method == "fsq":
            scalar_indices = outputs["scalar_indices"]
            for dim, num_levels in enumerate(scalar_levels):
                scalar_counts[dim] += torch.bincount(scalar_indices[:, dim], minlength=num_levels)

    # Hard reconstruction
    hard_mse = total_squared_error / total_elements

    # Hard code utilization
    hard_usage = hard_counts.double() / hard_counts.sum().clamp_min(1)
    non_zero = hard_usage > 0
    hard_entropy = -(hard_usage[non_zero] * torch.log(hard_usage[non_zero])).sum()
    hard_perplexity = torch.exp(hard_entropy).item()

    active_codes = int(torch.count_nonzero(hard_counts).item())
    dead_codes = num_codes - active_codes
    code_utilization = active_codes / num_codes

    metrics = {
        "hard_mse": hard_mse,
        "hard_perplexity": hard_perplexity,
        "active_codes": active_codes,
        "dead_codes": dead_codes,
        "code_utilization": code_utilization,
    }

    # AVQ metrics
    if method == "avq":
        local_entropy = local_entropy_sum / total_tokens
        normalized_local_entropy = local_entropy / math.log(num_codes)
        mean_max_probability = confidence_sum / total_tokens

        metrics.update({
            "local_entropy": local_entropy,
            "normalized_local_entropy": normalized_local_entropy,
            "mean_max_probability": mean_max_probability,
        })

    # FSQ metrics
    elif method == "fsq":
        normalized_scalar_entropies = []
        for counts, num_levels in zip(scalar_counts, scalar_levels):
            usage = counts.double() / counts.sum().clamp_min(1)

            non_zero = usage > 0
            entropy = -(usage[non_zero] * torch.log(usage[non_zero])).sum()
            normalized_entropy = entropy / math.log(num_levels)
            normalized_scalar_entropies.append(normalized_entropy.item())

        mean_scalar_entropy = (
            sum(normalized_scalar_entropies) / len(normalized_scalar_entropies)
        )

        metrics.update({"mean_scalar_entropy": mean_scalar_entropy})

        for dim, entropy in enumerate(normalized_scalar_entropies):
            metrics[f"scalar_entropy_{dim}"] = entropy

    return metrics


@torch.inference_mode()
def evaluate_arvq_hard_mse(model, val_loader, device, temperature=0.02):
    """
    Evaluate pure hard-code reconstruction MSE. MSE is accumulated over all 
    image elements.
    """
    model.eval()

    total_elements = 0
    total_squared_error = 0.0
    for batch in val_loader:
        images = _extract_images(batch).to(device, non_blocking=True)
        outputs = model(images, temperature=temperature, mode="hard")
        reconstruction = outputs["reconstruction"]
        total_squared_error += F.mse_loss(
            reconstruction, images, reduction="sum"
        ).item()

        total_elements += images.numel()

    return total_squared_error / total_elements
    

@torch.inference_mode()
def evaluate_arvq_validation(model, val_loader, device, temperature=0.02):
    """
    Validation for ARVQ

    Computes:
        - hard reconstruction MSE
        - hard code perplexity
        - active/dead code counts
        - code utilization
        - normalized local assignment entropy
        - mean maximum assignment probability
    """
    model.eval()

    num_codes = model.quantizer.num_embeddings
    num_stages = model.quantizer.num_quantizers

    hard_counts = torch.zeros(num_stages, num_codes, device=device, dtype=torch.long)
    stage_token_counts = torch.zeros(num_stages, device=device, dtype=torch.long)
    local_entropy_sums = torch.zeros(num_stages, device=device, dtype=torch.float64)
    confidence_sums = torch.zeros(num_stages, device=device, dtype=torch.float64)

    total_elements = 0
    total_squared_error = 0.0
    
    for batch in val_loader:
        images = _extract_images(batch).to(device, non_blocking=True)

        # Forward pass
        outputs = model(images, temperature=temperature, mode="hard")
        reconstruction_hard = outputs["reconstruction"]

        # [B, Q, H, W]
        indices = outputs["indices"]

        # List of Q tensors of shape [B, N, K]
        all_probs = outputs["all_probs"]

        total_elements += images.numel()

        # Reconstruction MSE
        total_squared_error += F.mse_loss(reconstruction_hard, images, reduction="sum").item()
        
        # Stage-wise ARVQ statistics
        for stage_idx in range(num_stages):

            # Hard code usage
            stage_indices = indices[:, stage_idx].reshape(-1)
            hard_counts[stage_idx] += torch.bincount(stage_indices, minlength=num_codes)
            stage_token_counts[stage_idx] += stage_indices.numel()

            # Soft assignment diagnostics
            probs = all_probs[stage_idx]
            local_entropy = -(probs * torch.log(probs.clamp_min(1e-10))).sum(dim=-1)
            local_entropy_sums[stage_idx] += local_entropy.double().sum()

            confidence_sums[stage_idx] += probs.max(dim=-1).values.double().sum()

    # Hard reconstruction
    hard_mse = total_squared_error / total_elements

    metrics = {"hard_mse": hard_mse}

    # Per-stage quantizer metrics
    mean_confidences = []
    normalized_entropies = []
    for stage_idx in range(num_stages):
        counts = hard_counts[stage_idx]
        usage = counts.double() / counts.sum().clamp_min(1)

        non_zero = usage > 0
        hard_entropy = -(usage[non_zero] * torch.log(usage[non_zero])).sum()
        hard_perplexity = torch.exp(hard_entropy).item()

        active_codes = int(torch.count_nonzero(counts).item())
        dead_codes = num_codes - active_codes
        code_utilization = active_codes / num_codes

        token_count = stage_token_counts[stage_idx].item()
        local_entropy = local_entropy_sums[stage_idx].item() / token_count
        normalized_local_entropy = local_entropy / math.log(num_codes)
        mean_max_probability = confidence_sums[stage_idx].item() / token_count

        normalized_entropies.append(normalized_local_entropy)
        mean_confidences.append(mean_max_probability)

        metrics.update({
            f"stage_{stage_idx}_hard_perplexity": hard_perplexity,
            f"stage_{stage_idx}_active_codes": active_codes,
            f"stage_{stage_idx}_dead_codes": dead_codes,
            f"stage_{stage_idx}_code_utilization": code_utilization,
            f"stage_{stage_idx}_local_entropy": local_entropy,
            f"stage_{stage_idx}_normalized_local_entropy": normalized_local_entropy,
            f"stage_{stage_idx}_mean_max_probability": mean_max_probability,
        })

    # Optional concise aggregate diagnostics
    metrics.update({
        "mean_normalized_local_entropy": sum(normalized_entropies) / num_stages,
        "mean_max_probability": sum(mean_confidences) / num_stages,
    })

    return metrics


@torch.inference_mode()
def evaluate_decoder_calibration_mse(
    model,
    val_loader,
    method,
    device,
    temperature=0.02,
):
    """
    Generic hard MSE evaluator
    """
    model.encoder.eval()
    model.quantizer.eval()
    model.decoder.eval()

    total_elements = 0
    total_squared_error = 0.0
    for batch in val_loader:
        images = _extract_images(batch).to(device, non_blocking=True)

        hard_quantized = get_hard_latents_for_calibration(
            model=model,
            images=images,
            method=method,
            temperature=temperature,
        )

        reconstruction = model.decoder(hard_quantized)

        total_elements += images.numel()
        total_squared_error += F.mse_loss(reconstruction, images, reduction="sum").item()

    return total_squared_error / total_elements


@torch.inference_mode()
def evaluate_rvqema_validation(model, val_loader, device):
    """
    Validation for RVQ-EMA

    Computes:
        - hard reconstruction MSE
        - per-stage hard code perplexity
        - per-stage active/dead code counts
        - per-stage code utilization
    """
    model.eval()
    
    num_codes = model.quantizer.num_embeddings
    num_stages = model.quantizer.num_quantizers

    hard_counts = torch.zeros(num_stages, num_codes, device=device, dtype=torch.long)

    total_elements = 0
    total_squared_error = 0.0
    
    for batch in val_loader:
        images = _extract_images(batch).to(device, non_blocking=True)

        # Forward pass
        outputs = model(images)
        reconstruction_hard = outputs["reconstruction"]

        # [B, S, H, W]
        indices = outputs["indices"]

        total_elements += images.numel()

        # Reconstruction MSE
        total_squared_error += F.mse_loss(reconstruction_hard, images, reduction="sum").item()
        
        # Stage-wise RVQ statistics
        for stage_idx in range(num_stages):

            # Hard code usage
            stage_indices = indices[:, stage_idx].reshape(-1)
            hard_counts[stage_idx] += torch.bincount(stage_indices, minlength=num_codes)

    # Hard reconstruction
    hard_mse = total_squared_error / total_elements

    metrics = {"hard_mse": hard_mse}

    # Per-stage quantizer metrics
    for stage_idx in range(num_stages):
        counts = hard_counts[stage_idx]
        usage = counts.double() / counts.sum().clamp_min(1)

        non_zero = usage > 0
        hard_entropy = -(usage[non_zero] * torch.log(usage[non_zero])).sum()
        hard_perplexity = torch.exp(hard_entropy).item()

        active_codes = int(torch.count_nonzero(counts).item())
        dead_codes = num_codes - active_codes
        code_utilization = active_codes / num_codes

        metrics.update({
            f"stage_{stage_idx}_hard_perplexity": hard_perplexity,
            f"stage_{stage_idx}_active_codes": active_codes,
            f"stage_{stage_idx}_dead_codes": dead_codes,
            f"stage_{stage_idx}_code_utilization": code_utilization,
        })

    return metrics


@torch.inference_mode()
def evaluate_residual_diveq_validation(model, val_loader, device):
    """
    Validation for Residual DiVeQ.
    """
    model.eval()

    num_codes = model.quantizer.num_embeddings
    num_stages = model.quantizer.num_codebooks
    hard_counts = torch.zeros(num_stages, num_codes, device=device, dtype=torch.long)

    total_elements = 0
    total_squared_error = 0.0

    for batch in val_loader:
        images = _extract_images(batch).to(device, non_blocking=True)

        outputs = model(images, return_hard_reconstruction=True)
        reconstruction_hard = outputs["reconstruction_hard"]
        indices = outputs["indices_hard"] # [B, Q, H, W]

        total_squared_error += F.mse_loss(reconstruction_hard, images, reduction="sum").item()
        total_elements += images.numel()

        for stage_idx in range(num_stages):
            stage_indices = indices[:, stage_idx].reshape(-1)
            hard_counts[stage_idx] += torch.bincount(stage_indices, minlength=num_codes)

    metrics = {"hard_mse": total_squared_error / total_elements}

    for stage_idx in range(num_stages):
        counts = hard_counts[stage_idx]
        usage = counts.double() / counts.sum().clamp_min(1)
        non_zero = usage > 0
        hard_entropy = -(usage[non_zero] * torch.log(usage[non_zero])).sum()

        active_codes = int(torch.count_nonzero(counts).item())

        metrics.update({
            f"stage_{stage_idx}_hard_perplexity": torch.exp(hard_entropy).item(),
            f"stage_{stage_idx}_active_codes": active_codes,
            f"stage_{stage_idx}_dead_codes": num_codes - active_codes,
            f"stage_{stage_idx}_code_utilization": active_codes / num_codes,
        })

    return metrics
