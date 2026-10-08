import math
import torch
import random
import logging
import numpy as np


logger = logging.getLogger(__name__)


def seed_worker(worker_id):
    """
    Seed NumPy and Python RNGs separately for each DataLoader worker.
    """
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def set_global_seed(seed=42):
    """
    Set global seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def update_running_metrics(running_metrics, batch_metrics, batch_size):
    """
    Accumulates batch-level metrics into running totals
    """
    for name, value in batch_metrics.items():
        if torch.is_tensor(value):
            value = value.detach().item()

        running_metrics[name] += value * batch_size


def is_log_step(idx, interval, steps):
    """
    Return True if the current batch should be logged
    """
    return idx == 0 or (idx + 1) % interval == 0 or idx == steps - 1


def set_requires_grad(module, requires_grad):
    """
    Sets the requires_grad attribute for all parameters in a PyTorch module.
    """
    for parameter in module.parameters():
        parameter.requires_grad_(requires_grad)


def prepare_model_for_calibration(model):
    """
    Freeze encoder and quantizer. Train decoder only.
    """
    # Freeze Encoder and Quantizer
    set_requires_grad(model.encoder, False)
    set_requires_grad(model.quantizer, False)
    
    # Decoder remains trainable
    set_requires_grad(model.decoder, True)

    
def change_model_modes_for_calibration(model):
    model.encoder.eval()
    model.quantizer.eval()
    model.decoder.train()


def restore_spatial_shape(tokens, batch_size, height, width, embedding_dim):
    """
    Restores a flattened sequence of tokens into a 2D spatial feature map
    """
    tokens = tokens.view(batch_size, height, width, embedding_dim)
    tokens = tokens.permute(0, 3, 1, 2)
    return tokens.contiguous()


def log_cosine_temperature_schedule(step, total_steps, start=1.5, end=0.1):
    """
    Computes a cosine annealing schedule in log-temperature space.

    This scheduling method provides smooth derivatives near both the start 
    and end of the annealing process. By interpolating in logarithmic space, 
    temperature changes are treated multiplicatively rather than additively, 
    which provides better numerical stability and scaling dynamics for 
    temperature-dependent operations.
    """
    progress = min(max(step / max(total_steps - 1, 1), 0.0), 1.0)
    cosine_weight = 0.5 * (1.0 + math.cos(math.pi * progress))
    log_temperature = math.log(end) + (math.log(start) - math.log(end)) * cosine_weight
    return math.exp(log_temperature)


def cosine_soft_weight_schedule(
    step, 
    steps_per_epoch, 
    warmup_epochs=15, 
    transition_end_epoch=40,
    start_soft_weight=1.0,
    final_soft_weight=0.2,
):
    """
    Computes a cosine-annealed schedule for a soft-to-hard latent weighting factor.

    This function calculates a progressive blending weight (alpha), often used in 
    quantization or latent regularization to smoothly transition from soft representations 
    to hard, discrete assignments. The schedule is divided into three distinct phases:

    - Warmup: Epochs [0, warmup_epochs), where alpha = start_soft_weight
    - Transition: Epochs [warmup_epochs, transition_end_epoch). Applies cosine annealing to 
      smoothly interpolate from cosine annealing from start_soft_weight to final_soft_weight
    - Final: Epochs [transition_end_epoch, ...), where alpha = final_soft_weight
    """
    warmup_steps = warmup_epochs * steps_per_epoch
    transition_end_step = transition_end_epoch * steps_per_epoch

    if step < warmup_steps:
        return start_soft_weight

    if step >= transition_end_step:
        return final_soft_weight

    transition_steps = transition_end_step - warmup_steps
    progress = (step - warmup_steps) / transition_steps
    progress = min(max(progress, 0.0), 1.0)

    cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
    soft_weight = final_soft_weight + (start_soft_weight - final_soft_weight) * cosine_factor

    return soft_weight


def denormalize_image(image):
    """
    Convert an image from [-1, 1] back to [0, 1].
    """
    return ((image + 1.0) / 2.0).clamp(0.0, 1.0)


def _extract_images(batch):
    """
    Supports common loader outputs: (images, labels), [images, labels]
    {"image": images}, images
    """
    if isinstance(batch, (tuple, list)):
        return batch[0]

    if isinstance(batch, dict):
        if "image" in batch:
            return batch["image"]
        if "images" in batch:
            return batch["images"]

        raise KeyError("Dictionary batch must contain 'image' or 'images'.")

    if torch.is_tensor(batch):
        return batch

    raise TypeError(f"Unsupported batch type: {type(batch)}")


def get_temperature_and_soft_weight(
    global_step, 
    anneal_steps,
    config,
    steps_per_epoch,
):
    """
    Get temperature and soft weight for training AVQ autoencoder
    """
    temperature = log_cosine_temperature_schedule(
        step=global_step,
        total_steps=anneal_steps,
        start=config.temperature_start,
        end=config.temperature_end,
    )

    soft_weight = cosine_soft_weight_schedule(
        step=global_step,
        steps_per_epoch=steps_per_epoch,
        warmup_epochs=config.soft_warmup_epochs,
        transition_end_epoch=config.soft_transition_end_epoch,
        start_soft_weight=config.start_soft_weight,
        final_soft_weight=config.final_soft_weight,
    )
    
    return temperature, soft_weight


def log_training_step(
    method,
    epoch,
    global_step,
    batch_metrics,
    outputs,
    codebook_metrics,
    codebook_norm_metrics,
    fsq_metrics,
    temperature=None,
    soft_weight=None,
    config=None,
):
    if method == "avq":
        schedule_stage = (
            "warmup"
            if epoch < config.soft_warmup_epochs
            else "transition"
            if epoch < config.anneal_epochs
            else "plateau"
        )

        logger.info(
            "Epoch %02d | "
            "Step %05d | "
            "Stage %s | "
            "Tau %.4f | "
            "SoftW %.4f | "
            "HardW %.4f | "
            "G %.4f | "
            "Rec %.4f | "
            "QLoss %.4f | "
            "LocalH %.4f | "
            "BalanceKL %.4f | "
            "CodeNorm %.4f | "
            "MinNorm %.4f | "
            "MaxNorm %.4f | "
            "Adv %.4f | "
            "D %.4f",
            epoch + 1,
            global_step,
            schedule_stage,
            temperature,
            soft_weight,
            1.0 - soft_weight,
            batch_metrics["generator_loss"].item(),
            batch_metrics["reconstruction_loss"].item(),
            batch_metrics["quantizer_loss"].item(),
            codebook_metrics["normalized_local_entropy"].item(),
            codebook_metrics["balance_kl"].item(),
            codebook_norm_metrics["mean_code_norm"],
            codebook_norm_metrics["min_code_norm"],
            codebook_norm_metrics["max_code_norm"],
            batch_metrics["adversarial_loss"].item(),
            batch_metrics["discriminator_loss"].item(),
        )

    elif method == "vq":
        logger.info(
            "Epoch %02d | "
            "Step %05d | "
            "G %.4f | "
            "Rec %.4f | "
            "VQ %.4f | "
            "CB %.4f | "
            "Commit %.4f | "
            "CodeNorm %.4f | "
            "MinNorm %.4f | "
            "MaxNorm %.4f | "
            "Adv %.4f | "
            "D %.4f",
            epoch + 1,
            global_step,
            batch_metrics["generator_loss"].item(),
            batch_metrics["reconstruction_loss"].item(),
            batch_metrics["quantizer_loss"].item(),
            outputs["codebook_loss"].item(),
            outputs["commitment_loss"].item(),
            codebook_norm_metrics["mean_code_norm"],
            codebook_norm_metrics["min_code_norm"],
            codebook_norm_metrics["max_code_norm"],
            batch_metrics["adversarial_loss"].item(),
            batch_metrics["discriminator_loss"].item(),
        )

    elif method in {"vqema", "rvqema"}:
        logger.info(
            "Epoch %02d | "
            "Step %05d | "
            "G %.4f | "
            "Rec %.4f | "
            "VQ %.4f | "
            "Commit %.4f | "
            "CodeNorm %.4f | "
            "MinNorm %.4f | "
            "MaxNorm %.4f | "
            "Adv %.4f | "
            "D %.4f",
            epoch + 1,
            global_step,
            batch_metrics["generator_loss"].item(),
            batch_metrics["reconstruction_loss"].item(),
            batch_metrics["quantizer_loss"].item(),
            outputs["commitment_loss"].item(),
            codebook_norm_metrics["mean_code_norm"],
            codebook_norm_metrics["min_code_norm"],
            codebook_norm_metrics["max_code_norm"],
            batch_metrics["adversarial_loss"].item(),
            batch_metrics["discriminator_loss"].item(),
        )

    elif method == "fsq":
        logger.info(
        "Epoch %02d | "
        "Step %05d | "
        "G %.4f | "
        "Rec %.4f | "
        "PPL %.1f | "
        "Active %.0f | "
        "Util %.3f | "
        "ScalarH %.3f | "
        "Adv %.4f | "
        "D %.4f",
        epoch + 1,
        global_step,
        batch_metrics["generator_loss"].item(),
        batch_metrics["reconstruction_loss"].item(),
        fsq_metrics["perplexity"].item(),
        fsq_metrics["active_codes"].item(),
        fsq_metrics["utilization"].item(),
        fsq_metrics["mean_scalar_entropy"].item(),
        batch_metrics["adversarial_loss"].item(),
        batch_metrics["discriminator_loss"].item(),
    )

    elif method == "diveq":
        logger.info(
            "Epoch %02d | "
            "Step %05d | "
            "G %.4f | "
            "Rec %.4f | "
            "PPL %.1f | "
            "CodeNorm %.4f | "
            "MinNorm %.4f | "
            "MaxNorm %.4f | "
            "Adv %.4f | "
            "D %.4f",
            epoch + 1,
            global_step,
            batch_metrics["generator_loss"].item(),
            batch_metrics["reconstruction_loss"].item(),
            batch_metrics["perplexity_batch"],
            codebook_norm_metrics["mean_code_norm"],
            codebook_norm_metrics["min_code_norm"],
            codebook_norm_metrics["max_code_norm"],
            batch_metrics["adversarial_loss"].item(),
            batch_metrics["discriminator_loss"].item(),
        )

    elif method == "rdiveq":
        logger.info(
            "Epoch %02d | Step %05d | G %.4f | Rec %.4f | PPL1 %.1f | PPL2 %.1f | PPL3 %.1f | "
            "CodeNorm %.4f | MinNorm %.4f | MaxNorm %.4f | Adv %.4f | D %.4f",
            epoch + 1,
            global_step,
            batch_metrics["generator_loss"].item(),
            batch_metrics["reconstruction_loss"].item(),
            batch_metrics["stage_0_perplexity_batch"],
            batch_metrics["stage_1_perplexity_batch"],
            batch_metrics["stage_2_perplexity_batch"],
            codebook_norm_metrics["mean_code_norm"],
            codebook_norm_metrics["min_code_norm"],
            codebook_norm_metrics["max_code_norm"],
            batch_metrics["adversarial_loss"].item(),
            batch_metrics["discriminator_loss"].item(),
        )
    
    else:
        raise ValueError(f"Unknown method: {method}")


def setup_logging(log_file):
    """
    Configures the root logger to output messages to both the console and a file
    """
    string = "%(asctime)s [%(levelname)s] %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(string, datefmt=date_format)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    logging.basicConfig(
        level=logging.INFO, handlers=[console_handler, file_handler], force=True
    )


def log_arvq_training_step(
    model,
    epoch,
    global_step,
    batch_metrics,
    stage_metrics,
    temperature,
    soft_weight,
):
    """
    Log ARVQ training metrics for one step.
    """
    logger.info(
        "ARVQ Epoch %02d | "
        "Step %05d | "
        "G %.4f | "
        "Rec %.4f | "
        "Q %.4f | "
        "Adv %.4f | "
        "D %.4f | "
        "Temp %.4f | "
        "Soft %.3f | "
        "Hard %.3f",
        epoch + 1,
        global_step,
        batch_metrics["generator_loss"].item(),
        batch_metrics["reconstruction_loss"].item(),
        batch_metrics["quantizer_loss"].item(),
        batch_metrics["adversarial_loss"].item(),
        batch_metrics["discriminator_loss"].item(),
        temperature,
        soft_weight,
        1.0 - soft_weight,
    )

    for stage_idx in range(model.quantizer.num_quantizers):
        logger.info(
            "ARVQ Stage %d | "
            "Local entropy %.4f | "
            "Balance KL %.4f",
            stage_idx,
            stage_metrics[f"stage_{stage_idx}/local_entropy"].item(),
            stage_metrics[f"stage_{stage_idx}/balance_kl"].item(),
        )


def get_hard_latents_for_calibration(
    model,
    images,
    method,
    temperature=0.02,
):
    """
    Return detached hard quantized latents for decoder calibration.
    """
    with torch.no_grad():
        encoded = model.encoder(images)

        if method == "avq":
            quantizer_outputs = model.quantizer(encoded, temperature=temperature)
        elif method in {"vq", "vqema", "rvqema","fsq"}:
            quantizer_outputs = model.quantizer(encoded)
        elif method == "diveq":
            B, D, h, w = encoded.shape
            encoded_flat = encoded.permute(0, 2, 3, 1).reshape(-1, D)
            hard_quantized_flat, _, _ = model.quantizer.inference(encoded_flat)
            hard_quantized = hard_quantized_flat.reshape(B, h, w, D).permute(0, 3, 1, 2).contiguous()
            return hard_quantized.detach()
        elif method == "rdiveq":
            B, D, h, w = encoded.shape
            encoded_flat = encoded.permute(0, 2, 3, 1).reshape(-1, D)
            hard_quantized_flat, _, _ = model.quantizer.inference(encoded_flat)
            hard_quantized = hard_quantized_flat.reshape(B, h, w, D).permute(0, 3, 1, 2).contiguous()
            return hard_quantized.detach()
        else:
            raise ValueError(f"Unsupported calibration method: {method}")

        hard_quantized = quantizer_outputs["hard_quantized"].detach()

    return hard_quantized


def safe_normalize(x, dim=-1, eps=1e-6):
    """
    If norm is dangerously small, divide by 1.0 instead of a tiny epsilon.
    It prevents the backward pass gradient from being multiplied by 1e6.

    Only use this in learned_code_scale = True branch in AVQ
    """
    norm = torch.linalg.vector.norm(x, dim=dim, keepdim=True)
    safe_norm = torch.where(norm > eps, norm, torch.ones_like(norm))
    return x / safe_norm


def gumbel_softmax_temperature_schedule(
    epoch, 
    num_epochs=50, 
    start_temperature=1.0, 
    min_temperature=0.1
):
    """
    Sets temperature based on exponential decay from start_temperature to 
    min_temperature over num_epochs.
    """
    decay_rate = (min_temperature / start_temperature) ** (1.0 / num_epochs)
    return max(start_temperature * (decay_rate ** epoch), min_temperature)

