import argparse

import torch
import torch.nn.functional as F

from tqdm import tqdm
from pathlib import Path

from torchmetrics.image.fid import FrechetInceptionDistance as FID
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS


from avq.generation.extract import hard_indices_from_images
from avq.generation.generate import decode_hard_indices_to_latent
from avq.generation.paths import get_dataset_root, get_autoencoder_checkpoint
from avq.generation.utils import (
    load_autoencoder,
    json_dump,
    _extract_images_labels,
    build_dataset_loaders,
)


EXPERIMENTS = {
    # Tiny ImageNet
    0: {
        "dataset": "tiny-imagenet",
        "experiment": "rdiveq_16x16_q3_k1024_d64",
    },
    1: {
        "dataset": "tiny-imagenet",
        "experiment": "arvq_16x16_q3_k1024_d64",
    },

    # MiniPlaces
    2: {
        "dataset": "mini-places",
        "experiment": "rdiveq_32x32_q3_k1024_d64",
    },
    3: {
        "dataset": "mini-places",
        "experiment": "arvq_32x32_q3_k1024_d64",
    },
}


STAGES = {
    0: "q1",
    1: "q2",
    2: "q3",
}


def permute_stage(indices, stage, generator):
    """
    Permute token assignments within one residual stage while preserving that 
    stage's exact empirical token histogram within the batch.

    indices: [B, S, H, W]
    """
    corrupted = indices.clone()
    stage_indices = corrupted[:, stage] # [B, H, W]
    flat = stage_indices.reshape(-1)
    device = indices.device

    # Generate permutation on CPU (deterministic behavior)
    permutation = torch.randperm(flat.numel(), generator=generator).to(device)

    # Corrupt indices w/ permutation
    corrupted[:, stage] = flat[permutation].view_as(stage_indices)

    return corrupted


@torch.inference_mode()
def decode_indices(model, method, indices):
    """
    Convert residual hard indices [B, S, H, W] to reconstructed images
    """
    quantized_latent = decode_hard_indices_to_latent(
        model=model,
        method=method,
        indices=indices,
    )

    reconstruction = model.decoder(quantized_latent)

    return reconstruction.clamp(-1.0, 1.0)


@torch.inference_mode()
def evaluate_quantizer_stage_corruption(
    model,
    method,
    stage,
    loader,
    device,
    temperature=0.02,
    corruption_seed=5,
):
    """
    e.g. Corrupt stage q3 and compare:
        clean:      q1 + q2 + q3
        corrupted:  q1 + q2 + shuffled(q3)

    quantizer stage is permuted rather than replaced by uniformly random tokens.
    """
    model.eval()

    # Paired perceptual metric
    clean_lpips_metric = LPIPS(
        net_type="alex", reduction="mean", normalize=False
    ).to(device)

    corrupt_lpips_metric = LPIPS(
        net_type="alex", reduction="mean", normalize=False,
    ).to(device)

    # Reconstruction FID
    clean_rfid_metric = FID(feature=2048,normalize=True).to(device)
    corrupt_rfid_metric = FID(feature=2048, normalize=True).to(device)

    clean_rfid_metric.set_dtype(torch.float64)
    corrupt_rfid_metric.set_dtype(torch.float64)

    # Fixed corruption seed
    generator = torch.Generator()
    generator.manual_seed(corruption_seed)

    num_pixels = 0
    num_images = 0
    clean_mse_sum = 0.0
    corrupt_mse_sum = 0.0
    
    progress_bar = tqdm(loader, desc=f"{STAGES[stage]} corruption: {method}", dynamic_ncols=True)
    for batch in progress_bar:

        images, _ = _extract_images_labels(batch)
        images = images.to(device, non_blocking=True)

        # 1. Get deterministic hard tokenizer indices: [B, S, H, W]
        indices = hard_indices_from_images(
            model=model,
            method=method,
            images=images,
            temperature=temperature,
        )

        # 2. Clean reconstruction.
        clean_reconstruction = decode_indices(
            model=model,
            method=method,
            indices=indices,
        )

        # 3. Corrupt only q_s stage
        corrupted_indices = permute_stage(
            indices=indices, stage=stage, generator=generator,
        )

        corrupt_reconstruction = decode_indices(
            model=model, method=method, indices=corrupted_indices,
        )

        # 4. LPIPS
        real_lpips = images.clamp(-1.0, 1.0) # [-1, 1] because normalize=False

        clean_lpips_metric.update(clean_reconstruction, real_lpips)

        corrupt_lpips_metric.update(corrupt_reconstruction, real_lpips)

        # 5. rFID
        real_fid = (
            (images.clamp(-1.0, 1.0) + 1.0) / 2.0
        ).clamp(0.0, 1.0)

        clean_fid = (
            (clean_reconstruction + 1.0) / 2.0
        ).clamp(0.0, 1.0)

        corrupt_fid = (
            (corrupt_reconstruction + 1.0) / 2.0
        ).clamp(0.0, 1.0)

        clean_rfid_metric.update(real_fid, real=True)
        clean_rfid_metric.update(clean_fid, real=False)

        corrupt_rfid_metric.update(real_fid, real=True)
        corrupt_rfid_metric.update(corrupt_fid, real=False)

        # 6. MSE 
        clean_mse_sum += F.mse_loss(
            clean_reconstruction, images, reduction="sum",
        ).item()

        corrupt_mse_sum += F.mse_loss(
            corrupt_reconstruction, images, reduction="sum",
        ).item()

        num_pixels += images.numel()
        num_images += images.shape[0]

    clean_lpips = clean_lpips_metric.compute().item()
    corrupt_lpips = corrupt_lpips_metric.compute().item()

    clean_rfid = clean_rfid_metric.compute().item()
    corrupt_rfid = corrupt_rfid_metric.compute().item()

    clean_mse = clean_mse_sum / num_pixels
    corrupt_mse = corrupt_mse_sum / num_pixels

    results = {
        "method": method,
        "num_images": num_images,
        "corrupted_stage": STAGES[stage],
        "corruption": "within_batch_random_permutation",
        "corruption_seed": corruption_seed,
        "clean": {
            "mse": clean_mse,
            "lpips": clean_lpips,
            "rfid": clean_rfid,
        },
        f"{STAGES[stage]}_corrupted": {
            "mse": corrupt_mse,
            "lpips": corrupt_lpips,
            "rfid": corrupt_rfid,
        },
        "delta": {
            "mse": corrupt_mse - clean_mse,
            "lpips": corrupt_lpips - clean_lpips,
            "rfid": corrupt_rfid - clean_rfid,
        },
    }

    return results


def main():
    parser = argparse.ArgumentParser(
        description=("Measure sensitivity to q_s corruption for ARVQ and R-DiVeQ.")
    )
    parser.add_argument("--project_root", type=str, required=True)
    parser.add_argument(
        "--experiment_id", type=int, default=0, choices=EXPERIMENTS.keys(), help="Experiment ID from EXPERIMENTS",
    )
    parser.add_argument("--model_seed", type=int, default=42)
    parser.add_argument("--corruption_stage", type=int, default=2, choices=STAGES.keys())
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--corruption_seed", type=int, default=5)
    args = parser.parse_args()

    # 0. Setup
    project_root = args.project_root
    model_seed = args.model_seed
    entry = EXPERIMENTS[args.experiment_id]
    dataset = entry["dataset"]
    experiment_name = entry["experiment"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    corruption_stage = args.corruption_stage

    # 1. Load autoencoder
    autoencoder_checkpoint_path = get_autoencoder_checkpoint(
        root_dir=project_root,
        dataset=dataset,
        model_seed=model_seed,
        experiment=experiment_name,
    )

    model, autoencoder_config, _ = load_autoencoder(
        experiment_name=experiment_name,
        checkpoint_path=autoencoder_checkpoint_path,
        device=device,
    )

    model.eval()
    method = autoencoder_config.method

    # 2. Load Tiny ImageNet validation set
    DATA_SEED = 15
    BATCH_SIZE = 64

    data_root = get_dataset_root(dataset)
    _, _, val_loader, _, _ = build_dataset_loaders(
        dataset=dataset,
        data_root=data_root,
        batch_size=BATCH_SIZE,
        data_seed=DATA_SEED,
    )

    # 3. Run residual corruption experiment
    results = evaluate_quantizer_stage_corruption(
        model=model,
        method=method,
        stage=corruption_stage,
        loader=val_loader,
        device=device,
        temperature=0.02,
        corruption_seed=args.corruption_seed,
    )

    results["dataset"] = dataset
    results["model_seed"] = model_seed
    results["experiment"] = experiment_name

    # 4. Save result
    output_root = Path(args.output_root).expanduser()
    output_dir = output_root / dataset / f"seed-{model_seed}" / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{method}_{STAGES[corruption_stage]}_corruption.json"
    json_dump(results, output_path)

    # 5. Print summary
    print(f"\n{STAGES[corruption_stage].upper()} corruption sensitivity")
    print("-------------------------")
    print(f"Method: {method}")
    print(f"Images: {results['num_images']}")
    print("\nClean")
    print(f"MSE: {results['clean']['mse']:.6f}")
    print(f"LPIPS: {results['clean']['lpips']:.6f}")
    print(f"rFID: {results['clean']['rfid']:.4f}")
    print(f"\n{STAGES[corruption_stage].upper()} corrupted")
    key = f"{STAGES[corruption_stage]}_corrupted"
    print(f"MSE: {results[key]['mse']:.6f}")
    print(f"LPIPS: {results[key]['lpips']:.6f}")
    print(f"rFID: {results[key]['rfid']:.4f}")
    print("\nDelta")
    print(f"MSE: +{results['delta']['mse']:.6f}")
    print(f"LPIPS: +{results['delta']['lpips']:.6f}")
    print(f"rFID: +{results['delta']['rfid']:.4f}")
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":

    main()
