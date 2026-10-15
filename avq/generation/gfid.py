import argparse

import torch

from tqdm import tqdm
from torchmetrics.image.fid import FrechetInceptionDistance as FID

from avq.generation.paths import EXPERIMENTS, get_generated_dir

from avq.generation.utils import json_load, json_dump, _extract_images_labels



def compute_gFID(generated_dir, real_data_loader, device):
    """
    """
    metric = FID(features=2048, normalize=True).to(device)
    metric.set_dtype(torch.float64)

    manifest_path = generated_dir / "manifest.json"
    manifest = json_load(manifest_path)
    shards = manifest["shards"]

    with torch.inference_mode():
        # Real images
        num_real = 0
        for batch in tqdm(real_data_loader, desc="gFID real", dynamic_ncols=True):
            images, _ = None # TODO
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
    parser.add_argument("--output_dir", type=str, default=None)

    # 1. Setup






if __name__ == "__main__":

    main()