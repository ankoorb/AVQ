import math
import torch
import torch.nn.functional as F


def sharpen_probabilities(probs, power=4.0, eps=1e-8):
    """
    Differentiably sharpens assignment probabilities to encourage confident 
    predictions.

    This function applies a power scaling to a probability distribution and 
    renormalizes it. By raising the probabilities to a power greater than 1.0, 
    the distribution is pushed toward a one-hot (hard) representation while 
    remaining fully differentiable.

    probs shape: [B, N, K]
    """
    sharpened_probs = probs.clamp_min(eps).pow(power)
    denominator = sharpened_probs.sum(dim=-1, keepdim=True).clamp_min(eps)
    return sharpened_probs / denominator


def discriminator_hinge_loss(real_logits, fake_logits):
    """
    Computes the adversarial hinge loss for the discriminator.

    This function calculates the hinge loss to improve training stability. It 
    penalizes the discriminator  if its prediction for real samples is less than 
    1.0, and if its prediction for fake (generated, reconstructed, or quantized) 
    samples is greater than -1.0.
    """
    real_loss = F.relu(1.0 - real_logits).mean()
    fake_loss = F.relu(1.0 + fake_logits).mean()
    return 0.5 * (real_loss + fake_loss)


def generator_adversarial_loss(fake_logits):
    """
    Computes the adversarial loss for the generator network.

    It encourages the generator to produce outputs that fool the 
    discriminator by penalizing small or negative discriminator responses. 
    Minimizing the negative mean of the fake logits pushes the discriminator's 
    predictions for fake samples to be as large and positive as possible.
    """
    return -fake_logits.mean()


def codebook_regularization_loss(probs, sharpening_power=4.0, local_weight=0.1, balance_weight=1.5, eps=1e-8):
    """
    Computes a regularization loss to promote confident assignments and prevent codebook collapse.

    This function applies two competing regularization forces to a soft quantization probability 
    distribution:
        - Local Entropy Penalty: Encourages individual tokens to commit strongly to a single codebook 
          vector, minimizing the uncertainty (entropy) of each assignment.
        - Global Balance (KL) Penalty: Encourages the overall usage of all codebook vectors across the 
          batch to remain uniform, preventing the network from collapsing to use only a small subset of 
          available codes.

    Both components are normalized by the log of the codebook size to ensure the loss scales consistently 
    regardless of how many codes are in the vocabulary.
    """
    num_codes = probs.shape[-1]

    # Maximum possible entropy for a uniform distribution over K items
    # Used to normalize the loss components to a [0, 1] scale.
    log_num_codes = math.log(num_codes)

    # Clamp probabilities to avoid log(0)
    probs_safe = probs.clamp_min(eps)

    # Local Entropy Calculation: Encourage confident assignments (push 
    # individual distributions toward one-hot).
    local_entropy = -(probs_safe * probs_safe.log()).sum(dim=-1).mean()
    normalized_local_entropy = local_entropy / log_num_codes

    # Approximate Hard Assignments: Sharpen the soft probabilities to better 
    # estimate what the hard (argmax) usage will look like.
    sharpened_probs = probs_safe.pow(sharpening_power)
    denominator = sharpened_probs.sum(dim=-1, keepdim=True).clamp_min(eps)
    sharpened_probs = sharpened_probs / denominator

    # Compute the mean utilization of each code across the batch and sequence dimensions.
    mean_usage = sharpened_probs.mean(dim=(0, 1)).clamp_min(eps)
    mean_usage = mean_usage / mean_usage.sum()

    # Global Balance KL Divergence Calculation: Prevent global collapse by 
    # penalizing the KL divergence between the empirical mean usage distribution 
    # and an ideal uniform distribution (1/K)
    # KL(mean_usage || uniform) = sum(mean_usage * log(mean_usage / (1/K)))
    balance_kl = torch.sum(mean_usage * torch.log(mean_usage * num_codes))
    normalized_balance_kl = balance_kl / log_num_codes

    # Combine local certainty and global uniformity objectives
    loss = local_weight * normalized_local_entropy + balance_weight * normalized_balance_kl
    
    metrics = {
        "local_entropy": local_entropy.detach(),
        "normalized_local_entropy": normalized_local_entropy.detach(),
        "balance_kl": balance_kl.detach(),
        "normalized_balance_kl": normalized_balance_kl.detach(),
    }

    return loss, metrics

# TODO
'''
def compute_rvq_codebook_loss(
    all_probs: List[Tensor],
    codebook_loss_fn: Callable,
    stage_weights: Optional[Sequence[float]] = None,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """
    Compute codebook regularization independently at each RVQ stage.

    Args:
        all_probs:
            List of [B, N, K] probability tensors.

        codebook_loss_fn:
            Function returning:
                loss, metrics

        stage_weights:
            Optional nonnegative per-stage weights.
            Weights are normalized to sum to one.
    """
    if len(all_probs) == 0:
        raise ValueError("all_probs cannot be empty.")

    num_stages = len(all_probs)

    if stage_weights is None:
        weights = all_probs[0].new_full(
            (num_stages,),
            1.0 / num_stages,
        )
    else:
        if len(stage_weights) != num_stages:
            raise ValueError(
                "stage_weights must match the number "
                "of RVQ stages."
            )

        weights = all_probs[0].new_tensor(
            stage_weights,
            dtype=all_probs[0].dtype,
        )

        if torch.any(weights < 0):
            raise ValueError(
                "stage_weights must be nonnegative."
            )

        weight_sum = weights.sum()

        if weight_sum <= 0:
            raise ValueError(
                "stage_weights must have a positive sum."
            )

        weights = weights / weight_sum

    total_loss = all_probs[0].new_zeros(())
    metrics: Dict[str, Tensor] = {}

    for stage_idx, probs in enumerate(all_probs):
        stage_loss, stage_metrics = codebook_loss_fn(
            probs
        )

        total_loss = (
            total_loss
            + weights[stage_idx] * stage_loss
        )

        metrics[
            f"stage_{stage_idx}/local_entropy"
        ] = stage_metrics["local_entropy"].detach()

        metrics[
            f"stage_{stage_idx}/balance_kl"
        ] = stage_metrics["balance_kl"].detach()

    return total_loss, metrics
'''