import torch
import argparse

from tqdm import tqdm
from torchmetrics.image.fid import FrechetInceptionDistance as FID

from avq.generation.paths import EXPERIMENTS, get_generated_dir, get_dataset_root, get_gfid_path
from avq.generation.utils import json_load, json_dump, _extract_images_labels, build_dataset_loaders


def compute_gFID(generated_dir, real_data_loader, device):
    """
    Compute generation FID
    """
    metric = FID(feature=2048, normalize=True).to(device)
    metric.set_dtype(torch.float64)

    manifest_path = generated_dir / "manifest.json"
    manifest = json_load(manifest_path)
    shards = manifest["shards"]

    with torch.inference_mode():
        # Real images
        num_real = 0
        for batch in tqdm(real_data_loader, desc="gFID real", dynamic_ncols=True):
            images, _ = _extract_images_labels(batch)
            images = images.to(device, non_blocking=True).clamp(-1.0, 1.0)
            # FID with normalize=True expects float images in [0, 1]
            images = ((images + 1.0) / 2.0).clamp(0.0, 1.0)
            metric.update(images, real=True)
            num_real += images.shape[0]

        # Generated images
        num_generated = 0
        for shard_name in tqdm(shards, desc="gFID generated", dynamic_ncols=True):
            shard = torch.load(generated_dir / shard_name, map_location="cpu")
            images = shard["images"].to(device, non_blocking=True).float() / 255.0
            metric.update(images, real=False)
            num_generated += images.shape[0]

    gfid = metric.compute().item()

    results = {
        "gfid": gfid,
        "num_real": num_real,
        "num_generated": num_generated,
        "dataset": manifest["dataset"],
        "experiment": manifest["experiment"],
        "method": manifest["method"],
        "model_seed": manifest["model_seed"],
        "sampling_seed": manifest["sampling_seed"],
        "num_per_class": manifest["num_per_class"],
        "sampling_temperature": manifest["sampling_temperature"],
        "top_k": manifest["top_k"],
        "generated_token_stats": manifest["generated_token_stats"],
        "prior_val_metrics": manifest["prior_val_metrics"],
    }

    return results


def main():
     # 0. Parse CLI arguments
    parser = argparse.ArgumentParser(description="Compute generation FID on generated shards.")
    parser.add_argument("--project_root", type=str, required=True)
    parser.add_argument(
        "--experiment_id", type=int, default=0, choices=EXPERIMENTS.keys(), help="Experiment ID from PRIOR_EXPERIMENTS",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--model_seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    # 1. Setup
    DATA_SEED = 15
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_seed = args.model_seed
    project_root = args.project_root
    entry = EXPERIMENTS[args.experiment_id]
    dataset = entry["dataset"]
    experiment_name = entry["experiment"]

    # 2. Get data and generated data dir
    data_root = get_dataset_root(dataset)
    _, _, val_loader, _, _ = build_dataset_loaders(
        dataset=dataset,
        data_root=data_root,
        batch_size=args.batch_size,
        data_seed=DATA_SEED,
    )

    generated_dir = get_generated_dir(
        root_dir=project_root, 
        dataset=dataset, 
        model_seed=model_seed, 
        experiment=experiment_name
    )

    # 3. Compute gFID
    output = compute_gFID(
        generated_dir=generated_dir,
        real_data_loader=val_loader,
        device=device,
    )

    # 4. Write result
    output_path = get_gfid_path(
        output_dir=args.output_dir, 
        dataset=dataset, 
        model_seed=model_seed, 
        experiment=experiment_name,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    json_dump(output, output_path)

    print(f"gFID: {output['gfid']:.4f}")
    print(f"Output saved: {output_path}")


if __name__ == "__main__":

    main()
