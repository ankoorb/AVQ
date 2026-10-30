import argparse

import torch
import torch.nn.functional as F

from tqdm import tqdm
from pathlib import Path

from avq.generation.paths import (
    get_dataset_root,
    get_generated_dir, 
    get_real_fid_cache_path,
    get_autoencoder_checkpoint,
)
from avq.generation.gfid import initialize_fid_metric
from avq.generation.extract import sequence_to_structured_indices
from avq.generation.generate import effective_avq_codebook, lookup_spatial_indices
from avq.generation.utils import load_autoencoder, json_dump, _extract_images_labels, build_dataset_loaders
   

EXPERIMENTS = {
    # Tiny ImageNet
    0: {
        "dataset": "tiny-imagenet",
        "experiment": "rvqema_16x16_q3_k1024_d64",
    },
    1: {
        "dataset": "tiny-imagenet",
        "experiment": "rdiveq_16x16_q3_k1024_d64",
    },
    2: {
        "dataset": "tiny-imagenet",
        "experiment": "rstgs_16x16_q3_k1024_d64",
    },
    3: {
        "dataset": "tiny-imagenet",
        "experiment": "arvq_16x16_q3_k1024_d64",
    },
    
    # MiniPlaces
    4: {
        "dataset": "mini-places",
        "experiment": "rvqema_32x32_q3_k1024_d64",
    },
    5: {
        "dataset": "mini-places",
        "experiment": "rdiveq_32x32_q3_k1024_d64",
    },
    6: {
        "dataset": "mini-places",
        "experiment": "rstgs_32x32_q3_k1024_d64",
    },
    7: {
        "dataset": "mini-places",
        "experiment": "arvq_32x32_q3_k1024_d64",
    },
}


def get_residual_stage_latent(model, method, indices, stage):
    """
    i/p:
        indices shape: [B, S, H, W]
    o/p:
        z_q shape: [B, D, H, W]
    """
    quantizer = model.quantizer
    stage_indices = indices[:, stage]

    if method == "rdiveq":
        # R-DiVeQ codebook shape: [S, K, D]
        stage_codebook = quantizer.codebook[stage]
        return lookup_spatial_indices(stage_indices, stage_codebook)

    if method == "arvq":
        stage_codebook = effective_avq_codebook(quantizer.layers[stage])
        return lookup_spatial_indices(stage_indices, stage_codebook)

    if method == "rvqema":
        stage_codebook = quantizer.layers[stage].codebook.weight
        return lookup_spatial_indices(stage_indices, stage_codebook)

    if method == "rstgs":
        stage_codebook = quantizer.codebooks[stage].weight
        return lookup_spatial_indices(stage_indices, stage_codebook)

    raise ValueError(f"Unsupported residual method: {method}")


def decode_residual_prefix(model, method, indices, num_stages):
    """
    Decode cumulative residual representation

    i/p:
        num_stages: 1 -> q1; 2 -> q1 + q2; 3 -> q1 + q2 + q3
    """
    stage_latents = []
    for stage in range(num_stages):
        stage_latent = get_residual_stage_latent(
            model=model, method=method, indices=indices, stage=stage,
        )
        stage_latents.append(stage_latent)

    return torch.stack(stage_latents, dim=0).sum(dim=0)


@torch.inference_mode()
def process_generated_shards(
    model,
    method,
    generated_dir,
    output_dir,
    latent_grid,
    device,
):
    """
    Decode already generated residual token sequences using: q1, 
    q1 + q2 and q1 + q2 + q3
    """
    model.eval()
    prefix_names = {
        1: "q1",
        2: "q1_q2",
        3: "q1_q2_q3",
    }

    shard_paths = sorted(generated_dir.glob("generated_*.pt"))

    prefix_dirs = {}
    for num_stages, name in prefix_names.items():
        # Create separate o/p directory for every stage prefix
        prefix_dir = output_dir / name
        prefix_dir.mkdir(parents=True, exist_ok=True)
        prefix_dirs[num_stages] = prefix_dir

    num_latents = 0
    stage_norm_sum = torch.zeros(len(prefix_names), dtype=torch.float64)
    prefix_norm_sum = torch.zeros(len(prefix_names), dtype=torch.float64)

    progress_bar = tqdm(
        shard_paths, desc=f"Residual Stage ablation: {method}", dynamic_ncols=True,
    )

    for shard_path in progress_bar:
        data = torch.load(shard_path, map_location="cpu")
        tokens = data["tokens"].long()
        labels = data["labels"]

        B, L, S = tokens.shape
        tokens = tokens.to(device) # [B, L, S]
        structured_indices = sequence_to_structured_indices(
            sequence=tokens, latent_grid=latent_grid, num_stages=S,
        ) # [B, S, H, W]

        # Decode individual residual stage
        stage_latents = []
        for stage in range(S):
            q_stage = get_residual_stage_latent(
                model=model, method=method, indices=structured_indices, stage=stage,
            ) # [B, D, H, W]

            stage_latents.append(q_stage)

            # Mean of L2 norm across the embedding dimension D for positions
            stage_norm = torch.linalg.vector_norm(q_stage, ord=2, dim=1).mean()
            stage_norm_sum[stage] += stage_norm.item() * B

        # Cumulative decoding
        cumulative_latent = torch.zeros_like(stage_latents[0])
        for stage in range(S):
            cumulative_latent = cumulative_latent + stage_latents[stage]
            num_stages = stage + 1

            # Mean of L2 norm across the embedding dimension D for positions
            prefix_norm = torch.linalg.vector_norm(cumulative_latent, ord=2, dim=1).mean()
            prefix_norm_sum[stage] += prefix_norm.item() * B

            decoded = model.decoder(cumulative_latent)
            decoded = decoded.clamp(-1.0, 1.0)
            
            # uint8 images
            images = ((decoded + 1.0) * 127.5).round().clamp(0, 255)
            uint8_images = images.to(torch.uint8).cpu()

            payload = {
                "images": uint8_images,
                "labels": labels,
                "tokens": tokens,
                "num_stages_decoded": num_stages,
            }

            torch.save(payload, prefix_dirs[num_stages] / shard_path.name)

        num_latents += B

    # Residual vector magnitude summary
    stage_norm_mean = stage_norm_sum / num_latents
    prefix_norm_mean = prefix_norm_sum / num_latents

    output = {
        "method": method,
        "num_images": num_latents,
        "stage_mean_l2_norm": {
            "q1": stage_norm_mean[0].item(),
            "q2": stage_norm_mean[1].item(),
            "q3": stage_norm_mean[2].item(),
        },
        "cumulative_mean_l2_norm": {
            "q1": prefix_norm_mean[0].item(),
            "q1_q2": prefix_norm_mean[1].item(),
            "q1_q2_q3": prefix_norm_mean[2].item(),
        },
    }

    json_dump(output, output_dir / "stage_summary.json")

    output["prefix_dirs"] = prefix_dirs
    output["prefix_names"] = prefix_names

    return output


def compute_gFID(project_root, generated_dir, real_data_loader, device, dataset):
    """
    Compute generation FID
    """
    metric, num_real = initialize_fid_metric(
        project_root=project_root,
        real_data_loader=real_data_loader,
        device=device,
        dataset=dataset,
    )

    shard_paths = sorted(generated_dir.glob("generated_*.pt"))

    with torch.inference_mode():

        # Generated images
        num_generated = 0
        for shard_path in tqdm(shard_paths, desc="gFID generated", dynamic_ncols=True):
            shard = torch.load(shard_path, map_location="cpu")
            images = shard["images"].to(device, non_blocking=True).float() / 255.0
            metric.update(images, real=False)
            num_generated += images.shape[0]

    gfid = metric.compute().item()

    results = {
        "gfid": gfid,
        "num_real": num_real,
        "num_generated": num_generated,
    }

    return results

def main():
    parser = argparse.ArgumentParser(
        description="Decode existing residual prior samples using q1, q1+q2, and q1+q2+q3."
    )
    parser.add_argument("--project_root", type=str, required=True)
    parser.add_argument(
        "--experiment_id", type=int, default=0, choices=EXPERIMENTS.keys(), help="Experiment ID from EXPERIMENTS",
    )
    parser.add_argument("--model_seed", type=int, default=42)
    parser.add_argument("--output_root", type=str, required=True)
    args = parser.parse_args()

    # 0. Setup
    model_seed = args.model_seed
    project_root = args.project_root
    entry = EXPERIMENTS[args.experiment_id]
    dataset = entry["dataset"]
    latent_grid = 16 if dataset == "tiny-imagenet" else 32
    experiment_name = entry["experiment"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Get generated directory and autoencoder checkpoint
    generated_dir = get_generated_dir(
        root_dir=project_root, 
        dataset=dataset, 
        model_seed=model_seed, 
        experiment=experiment_name
    )

    autoencoder_checkpoint_path = get_autoencoder_checkpoint(
        root_dir=project_root, 
        dataset=dataset, 
        model_seed=model_seed, 
        experiment=experiment_name,
    )

    # 2. Load autoencoder
    model, autoencoder_config, _ = load_autoencoder(
        experiment_name=experiment_name,
        checkpoint_path=autoencoder_checkpoint_path,
        device=device,
    )

    model.eval()

    # 3. Process generated tokens and save diagnostic output
    output_root = Path(args.output_root).expanduser()
    output_dir = output_root / dataset / f"seed-{model_seed}" / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    output = process_generated_shards(
        model=model,
        method=autoencoder_config.method,
        generated_dir=generated_dir,
        output_dir=output_dir,
        latent_grid=latent_grid,
        device=device,
    )

    print("\nResidual latent magnitude summary")
    for key, value in output["stage_mean_l2_norm"].items():
        print(f"{key:8s}: {value:0.6f}")

    print("\nCumulative latent magnitude summary")
    for key, value in output["cumulative_mean_l2_norm"].items():
            print(f"{key:8s}: {value:0.6f}")

    print("\nDecoded image directories:")
    for num_stages, prefix_dir in output["prefix_dirs"].items():
        prefix_name = output["prefix_names"][num_stages]
        print(f"{prefix_name:10s}: {prefix_dir}")

    # 4. Run gFID
    DATA_SEED = 15
    BATCH_SIZE = 64
    data_root = get_dataset_root(dataset)
    _, _, val_loader, _, _ = build_dataset_loaders(
        dataset=dataset,
        data_root=data_root,
        batch_size=BATCH_SIZE,
        data_seed=DATA_SEED,
    )

    for num_stages, prefix_dir in output["prefix_dirs"].items():
        decoded_dir = Path(prefix_dir)
        prefix_name = output["prefix_names"][num_stages]
        print(f"Computing {prefix_name} gFID")
        prefix_output = compute_gFID(
            project_root=project_root,
            generated_dir=decoded_dir,
            real_data_loader=val_loader,
            device=device,
            dataset=dataset,
        )

        json_dump(prefix_output, decoded_dir.parent / f"{prefix_name}_gfid.json")
        print(f"{prefix_name.upper()} gFID: {prefix_output['gfid']:.4f}\n\n")


if __name__ == "__main__":

    main()
