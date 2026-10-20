"""
NOTE: This script is for re-running final DiVeQ validation using deterministic hard 
indices and update result.json. During training the evaluator was using wrong indices
field for code usage statistics (perplexity, active codes, dead codes and utilization).

This script loads saved calibrated DiVeQ checkpoint, evaluates the official validation
split using encoder -> quantizer.inference() -> decoder and updates only the final 
validation metrics in result.json
"""

import json
import shutil
import argparse

import torch
import torch.nn.functional as F

from tqdm import tqdm
from pathlib import Path

from avq.helpers import _extract_images
from avq.config import EXPERIMENT_REGISTRY
from avq.dataset import (
    build_coco_loaders,
    build_mini_places_loaders,
    build_tiny_imagenet_loaders,
)


DATA_SEED = 15

DEFAULT_DATA_ROOTS = {
    "tiny-imagenet": "/tmp/data/tiny-imagenet-200",
    "mini-places": "/tmp/data/miniplaces",
    "coco": "/tmp/data/coco2017",
}


def build_validation_loaders(dataset, data_root):
    """
    Build the same official validation loader used by autoencoder experiments.
    """
    if dataset == "tiny-imagenet":
        _, _, val_loader, _, _ = build_tiny_imagenet_loaders(
            root_directory=data_root,
            batch_size=64,
            num_workers=4,
            image_size=64,
            seed=DATA_SEED,
            drop_last_train=True,
        )
        return val_loader

    if dataset == "mini-places":
        _, _, val_loader, _, _ = build_mini_places_loaders(
            root_directory=data_root,
            batch_size=16,
            num_workers=4,
            image_size=128,
            seed=DATA_SEED,
            drop_last_train=True,
        )
        return val_loader

    if dataset == "coco":
        _, _, val_loader, _, _ = build_coco_loaders(
            root_directory=data_root,
            batch_size=64,
            num_workers=4,
            image_size=64,
            seed=DATA_SEED,
            drop_last_train=True,
        )
        return val_loader

    raise ValueError(f"Unsupported dataset {dataset!r}.")


def load_diveq_model(experiment_name, checkpoint_path, device):
    """
    Instantiate DiVeQ autoencoder and load the saved checkpoint.
    """
    experiment = EXPERIMENT_REGISTRY[experiment_name]
    assert experiment.method == "diveq"

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model_state_dict"]

    model = experiment.model_cls(**experiment.model_kwargs).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad = False

    return model, experiment, checkpoint


@torch.inference_mode()
def evaluate_diveq(model, val_loader, num_codes, device):
    """
    Evaluate deterministic hard DiVeQ reconstruction and dataset-wide code usage.

    Uses quantizer.inference() so hard-code statistics are based on the deterministic
    assignments.
    """
    model.eval()

    num_images = 0
    total_elements = 0
    total_squared_error = 0.0
    hard_counts = torch.zeros(num_codes, device=device, dtype=torch.long)
    for batch in tqdm(val_loader, desc="DiVeQ validation", dynamic_ncols=True):
        images = _extract_images(batch).to(device, non_blocking=True)

        z = model.encoder(images)
        B, D, H, W = z.shape

        # DiVeQ inference expects [N, D]
        z_flat = z.permute(0, 2, 3, 1).contiguous().view(-1, D) # [N, D]

        # Deterministic nearest codeword quantization
        z_q_flat, indices, _ = model.quantizer.inference(z_flat)

        # Restore latent shape: [B * H * W, D] -> [B, D, H, W]
        z_q = z_q_flat.view(B, H, W, D).permute(0, 3, 1, 2).contiguous()

        reconstruction_hard = model.decoder(z_q)

        num_images += B
        total_elements += images.numel()
        total_squared_error += F.mse_loss(
            reconstruction_hard, images, reduction="sum",
        ).item()

        indices_flat = indices.long().reshape(-1)
        hard_counts += torch.bincount(indices_flat, minlength=num_codes)

    # Compute metrics
    hard_mse = total_squared_error / total_elements

    hard_usage = hard_counts.double() / hard_counts.sum().clamp_min(1)
    non_zero = hard_usage > 0
    hard_entropy = -(hard_usage[non_zero] * torch.log(hard_usage[non_zero])).sum()
    hard_perplexity = torch.exp(hard_entropy).item()

    active_codes = int(torch.count_nonzero(hard_counts).item())
    dead_codes = num_codes - active_codes
    code_utilization = active_codes / num_codes

    return {
        "hard_mse": hard_mse,
        "hard_perplexity": hard_perplexity,
        "active_codes": active_codes,
        "dead_codes": dead_codes,
        "code_utilization": code_utilization,
        "num_images": num_images,
    }


def update_result_json(result_path, corrected_metrics):
    """
    Update only final top-level validation metrics.

    Phase-I history is intentionally left untouched because correcting historical
    per-epoch entries would require evaluating every historical checkpoint.
    """
    with result_path.open('r') as file:
        result = json.load(file)

    backup_path = result_path.with_name("result_before_index_fix.json")
    if not backup_path.exists():
        shutil.copy2(result_path, backup_path)
        print(f"Backup saved: {backup_path}")
    else:
        print(f"Backup already exists, leaving unchanged: {backup_path}")

    print("\nTop-level metrics:")
    keys_to_update = (
        "hard_perplexity", "active_codes", "dead_codes", "code_utilization",
    )
    for key in keys_to_update:
        new_value = corrected_metrics[key]
        old_value = result.get(key, "<missing>")
        print(f"  {key}: {old_value} -> {new_value}")
        result[key] = new_value

    old_hard_mse = result.get("hard_mse")
    new_hard_mse = corrected_metrics["hard_mse"]
    print(
        f"\nHard MSE consistency check: stored={old_hard_mse} | recomputed={new_hard_mse}"
    )

    with result_path.open('w') as file:
        json.dump(result, file, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Recompute final DiVeQ hard-code validation statistics using "
            "quantizer.inference() and update result.json."
        )
    )

    parser.add_argument(
        "--experiment_dir",
        type=Path,
        required=True,
        help="Experiment directory containing result.json and calibration/.",
    )
    parser.add_argument("--dataset", choices=("tiny-imagenet", "mini-places", "coco"), required=True)

    args = parser.parse_args()

    # 0. Setup
    data_root = Path(DEFAULT_DATA_ROOTS[args.dataset])
    experiment_dir = Path(args.experiment_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    experiment_name = experiment_dir.name
    result_path = experiment_dir / "result.json"
    checkpoint_path = experiment_dir / "calibration" / "best_decoder_calibration.pt"

    if not data_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {data_root}")

    if not experiment_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")

    print(f"Experiment: {experiment_name}")
    print(f"Dataset: {args.dataset}")
    print(f"Data root: {data_root}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Result file: {result_path}")
    print(f"Device: {device}")

    # 1. Build validation data loader
    val_loader = build_validation_loaders(dataset=args.dataset, data_root=str(data_root))

    # 2. Load autoencoder
    model, experiment, _ = load_diveq_model(
        experiment_name=experiment_name,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    # 3. Evaluate
    corrected_metrics = evaluate_diveq(
        model=model,
        val_loader=val_loader,
        num_codes=experiment.num_codes,
        device=device,
    )

    # 4. Update result.json
    update_result_json(result_path=result_path, corrected_metrics=corrected_metrics)

    print("\nCorrected DiVeQ final validation:")
    for key, value in corrected_metrics.items():
        print(f"  {key}: {value}")

    print(f"\nUpdated result.json: {result_path}")


if __name__ == "__main__":

    main()
