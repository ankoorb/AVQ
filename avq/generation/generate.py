import argparse

import torch
import torch.nn.functional as F

from tqdm import tqdm

from avq.helpers import set_global_seed
from avq.generation.paths import (
    EXPERIMENTS,
    get_token_dir,
    get_generated_dir,
    get_prior_checkpoint,
    get_autoencoder_checkpoint, 
)
from avq.generation.prior import load_prior
from avq.generation.extract import sequence_to_structured_indices
from avq.generation.utils import load_autoencoder, json_load, histogram_stats, json_dump


def effective_avq_codebook(quantizer):
    """
    Return the effective AVQ codebook used to construct quantized vectors.

    - If normalize=True, code vectors are L2-normalized before use.
    - If learned_code_scale=True, the normalized/raw code vectors are multiplied
      by the learned positive scalar softplus(raw_code_scale) + 1e-6.
    """
    codebook = quantizer.codebook # nn.Parameter of shape [K, D]

    if getattr(quantizer, "normalize", False):
        codebook = F.normalize(codebook, dim=-1, eps=1e-6)

    if getattr(quantizer, "learned_code_scale", False):
        scale = F.softplus(quantizer.raw_code_scale) + 1e-6
        codebook = scale * codebook

    return codebook


def lookup_spatial_indices(indices, codebook):
    """
    Map spatial code indice to decoder space latent vectors.

    i/p:
        indices shape: [B, H, W]
        codebook shape: [K, D]
    o/p:
        z_q shape: [B, D, H, W]
    """
    z_q = F.embedding(indices.long(), codebook) # [B, H, W, D]
    return z_q.permute(0, 3, 1, 2).contiguous() # [B, D, H, W]


def decode_fsq_indices_to_latent(indices, quantizer):
    """
    i/p:
        indices shape: [B, H, W]
        quantizer: FiniteScalarQuantizer
    o/p:
        z_q shape: [B, D, H, W]
    """
    indices = indices.long()

    levels = torch.tensor(
        quantizer.levels, device=indices.device, dtype=torch.long,
    )
    basis = quantizer.basis.to(indices.device)

    # Undo: token_index = sum_i scalar_index_i * basis_i
    scalar_indices = []
    for level in range(len(quantizer.levels)):
        scalar_indices.append(indices // basis[level] % levels[level])

    scalar_indices = torch.stack(scalar_indices, dim=-1) # [B, H, W, fsq_dim]
    scalar_indices = scalar_indices.to(dtype=quantizer.project_out.weight.dtype)

    levels_float = levels.to(dtype=quantizer.project_out.weight.dtype)

    # Convert integer scalar levels back to the exact FSQ values in [-1, 1]
    hard_fsq_tokens = (2.0 * scalar_indices / (levels_float - 1.0) - 1.0) # [B, H, W, fsq_dim]

    z_q = quantizer.project_out(hard_fsq_tokens) # [B, H, W, D]

    return z_q.permute(0, 3, 1, 2).contiguous()


def decode_hard_indices_to_latent(model, method, indices):
    """
    Convert generated hard token indices back to quantized latent
    tensor expected by the pretrained autoencoder decoder.

    i/p: 
        Singe stage indices shape: [B, H, W]
        Residual indices shape: [B, S, H, W]

    o/p:
        z_q: [B, D, H, W]
    """
    quantizer = model.quantizer

    if method == "avq":
        codebook = effective_avq_codebook(quantizer)
        return lookup_spatial_indices(indices, codebook)

    if method == "arvq":
        stage_latents = []
        for stage in range(len(quantizer.layers)):
            stage_codebook = effective_avq_codebook(quantizer.layers[stage])
            stage_latent = lookup_spatial_indices(
                indices=indices[:, stage], codebook=stage_codebook,
            )
            stage_latents.append(stage_latent)

        return torch.stack(stage_latents, dim=0).sum(dim=0) # [S, B, D, H, W] then sum

    if method in {"vq", "vqema"}:
        codebook = quantizer.codebook.weight
        return lookup_spatial_indices(indices, codebook)

    if method == "rvqema":
        stage_latents = []
        for stage in range(len(quantizer.layers)):
            stage_codebook = quantizer.layers[stage].codebook.weight
            stage_latent = lookup_spatial_indices(
                indices=indices[:, stage], codebook=stage_codebook,
            )
            stage_latents.append(stage_latent)

        return torch.stack(stage_latents, dim=0).sum(dim=0) # [S, B, D, H, W] then sum

    if method == "fsq":
        return decode_fsq_indices_to_latent(indices, quantizer)

    if method == "diveq":
        codebook = quantizer.codebook
        return lookup_spatial_indices(indices, codebook)

    if method == "rdiveq":
        # codebook shape: [S, K, D]
        stage_latents = []
        for stage in range(quantizer.codebook.shape[0]):
            stage_codebook = quantizer.codebook[stage] # [K, D]
            stage_latent = lookup_spatial_indices(
                indices=indices[:, stage], codebook=stage_codebook,
            )
            stage_latents.append(stage_latent)

        return torch.stack(stage_latents, dim=0).sum(dim=0) # [S, B, D, H, W] then sum
        
    if method == "stgs":
        codebook = quantizer.codebook.weight
        return lookup_spatial_indices(indices, codebook)

    if method == "rstgs":
        stage_latents = []
        for stage in range(len(quantizer.codebooks)):
            stage_codebook = quantizer.codebooks[stage].weight
            stage_latent = lookup_spatial_indices(
                indices=indices[:, stage], codebook=stage_codebook,
            )
            stage_latents.append(stage_latent)

        return torch.stack(stage_latents, dim=0).sum(dim=0) # [S, B, D, H, W] then sum
        
    raise ValueError(f"Unsupported quantizer method {method!r}")


@torch.inference_mode()
def generate_decode(prior, model, labels, prior_config, args, device, generate_config, output_dir):
    """
    Generet tokens and decode tokens.
    """
    shard_index = 0
    shard_names = []
    total_images = labels.numel()
    method = generate_config["method"]
    batch_size = generate_config["batch_size"]
    top_k = None if args.top_k == 0 else args.top_k

    print(
        f"Generating {total_images:,} images | {args.num_per_class}/class | "
        f"method={method} | L={prior_config.sequence_length} | batch={batch_size}"
    )

    stage_histogram = torch.zeros(
        prior_config.num_codebooks, prior_config.vocab_size, dtype=torch.int64
    )

    desc = "Generate + Decode"
    for idx in tqdm(range(0, total_images, batch_size), desc=desc, dynamic_ncols=True):
        batch_labels_cpu = labels[idx : idx + batch_size]
        batch_labels = batch_labels_cpu.to(device, non_blocking=True)

        generated_sequence = prior.generate(
            class_labels=batch_labels,
            temperature=args.sampling_temperature,
            top_k=top_k,
        ) # [B, L] or [B, L, S]

        structured_indices = sequence_to_structured_indices(
            sequence=generated_sequence,
            latent_grid=generate_config["latent_grid"],
            num_stages=prior_config.num_codebooks,
        )

        quantized_latent = decode_hard_indices_to_latent(
            model=model,
            method=method,
            indices=structured_indices,
        )

        decoded = model.decoder(quantized_latent)
        decoded = decoded.clamp(-1.0, 1.0)

        # uint8 images
        images = ((decoded + 1.0) * 127.5).round().clamp(0, 255)
        uint8_images = images.to(torch.uint8).cpu()

        # Compute stats
        generated_sequence_cpu = generated_sequence.to("cpu")
        if prior_config.num_codebooks == 1:
            # generated_sequence: [B, L]
            stage_histogram[0] += torch.bincount(
                generated_sequence_cpu.reshape(-1), minlength=prior_config.vocab_size,
            )
        else:
            # generated_sequence: [B, L, S]
            for stage in range(prior_config.num_codebooks):
                stage_histogram[stage] += torch.bincount(
                    generated_sequence_cpu[:, :, stage].reshape(-1), # [B, L]
                    minlength=prior_config.vocab_size,
                )

        # Save generated sequence and images
        shard_name = f"generated_{shard_index:03d}.pt"
        shard_names.append(shard_name)

        payload = {
            "images": uint8_images,
            "labels": batch_labels_cpu.to(torch.int16),
            "tokens": generated_sequence_cpu,
        }

        torch.save(payload, output_dir / shard_name)

        shard_index += 1

    output = {
        "shard_names": shard_names,
        "per_stage": [
            histogram_stats(stage_histogram[stage]) for stage in range(prior_config.num_codebooks)
        ],
    }

    return output


def main():
    # 0. Parse CLI arguments
    parser = argparse.ArgumentParser(description="Sample tokens and decode generated images.")
    parser.add_argument("--project_root", type=str, required=True)
    parser.add_argument(
        "--experiment_id", type=int, default=0, choices=EXPERIMENTS.keys(), help="Experiment ID from EXPERIMENTS",
    )
    parser.add_argument("--num_per_class", type=int, default=None, help="50 for Tiny Imagenet. 100 for MiniPlaces")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--sampling_temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0, help="0 disables top-k truncation.")
    parser.add_argument("--model_seed", type=int, default=42)
    parser.add_argument("--sampling_seed", type=int, default=15)
    args = parser.parse_args()
    
    # 1. Reproducibility
    model_seed = args.model_seed
    sampling_seed = args.sampling_seed
    set_global_seed(model_seed)

    # 2. Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    project_root = args.project_root
    entry = EXPERIMENTS[args.experiment_id]
    dataset = entry["dataset"]
    experiment_name = entry["experiment"]

    if args.num_per_class is None:
        args.num_per_class = 50 if dataset == "tiny-imagenet" else 100

    token_dir = get_token_dir(
        root_dir=project_root,
        dataset=dataset,
        model_seed=model_seed,
        experiment=experiment_name,
    )

    metadata = json_load(token_dir / "metadata.json")

    output_dir = get_generated_dir(
        root_dir=project_root, 
        dataset=dataset, 
        model_seed=model_seed, 
        experiment=experiment_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    autoencoder_checkpoint_path = get_autoencoder_checkpoint(
        root_dir=project_root, 
        dataset=dataset, 
        model_seed=model_seed, 
        experiment=experiment_name,
    )

    prior_checkpoint_path = get_prior_checkpoint(
        root_dir=project_root, 
        dataset=dataset, 
        model_seed=model_seed, 
        experiment=experiment_name,
    )

    # 3. Load autoencoder
    model, autoencoder_config, _ = load_autoencoder(
        experiment_name=experiment_name,
        checkpoint_path=autoencoder_checkpoint_path,
        device=device,
    )

    model.eval()
    method = autoencoder_config.method

    # 4. Load prior (alread in eval mode)
    prior, prior_config, prior_checkpoint = load_prior(
        experiment_name=experiment_name,
        checkpoint_path=prior_checkpoint_path,
        device=device,
    )

    # 5. Generate tokens and decode
    set_global_seed(sampling_seed)
    batch_size = args.batch_size
    if batch_size is None:
        batch_size = 64 if prior_config.num_codebooks == 1 else 32

    unique_class_labels = torch.arange(prior_config.num_classes, dtype=torch.long)
    class_labels = unique_class_labels.repeat_interleave(args.num_per_class)

    generate_config = {
        "method": method,
        "batch_size": batch_size,
        "latent_grid": metadata["latent_grid"],
    }

    output = generate_decode(
        prior=prior,
        model=model,
        labels=class_labels,
        prior_config=prior_config,
        args=args,
        device=device,
        generate_config=generate_config,
        output_dir=output_dir,
    )

    manifest = {
        "dataset": dataset,
        "experiment": experiment_name,
        "method": method,
        "token_dir": str(token_dir),
        "prior_checkpoint_path": str(prior_checkpoint_path),
        "autoencoder_checkpoint_path": str(autoencoder_checkpoint_path),
        "batch_size": batch_size,
        "num_classes": prior_config.num_classes,
        "num_per_class": args.num_per_class,
        "num_images": class_labels.numel(),
        "sampling_temperature": args.sampling_temperature,
        "top_k": args.top_k,
        "latent_grid": metadata["latent_grid"],
        "num_codebooks": prior_config.num_codebooks,
        "sequence_length": prior_config.sequence_length,
        "vocab_size": prior_config.vocab_size,
        "generated_token_stats": output["per_stage"],
        "shards": output["shard_names"],
        "model_seed": model_seed,
        "sampling_seed": sampling_seed,
        "prior_val_metrics": prior_checkpoint.get("val_metrics"),
    }

    json_dump(manifest, output_dir / "manifest.json")

    print(f"Saved generated shards to: {output_dir}")
    if prior_config.num_codebooks == 1:
        stats = output["per_stage"][0]
        print(
            f"Generated-token PPL: {stats['perplexity']:.3f} | "
            f"utilization: {100.0 * stats['utilization']:.2f}%"
        )
    else:
        ppls = [stats["perplexity"] for stats in output["per_stage"]]
        utilizations = [100.0 * stats["utilization"] for stats in output["per_stage"]]
        print(f"Generated per-stage PPL: {[round(x, 3) for x in ppls]}")
        print(f"Generated per-stage utilization (%): {[round(x, 2) for x in utilizations]}")


if __name__ == "__main__":

    main()
