import torch
import argparse

from tqdm import tqdm
from pathlib import Path

from avq.helpers import set_global_seed
from avq.generation.utils import (
    json_dump,
    histogram_stats, 
    load_autoencoder,
    _extract_images_labels, 
    build_deterministic_token_loaders,
)
from avq.generation.paths import (
    EXPERIMENTS,
    get_autoencoder_checkpoint, 
    get_token_dir,
    get_token_file, 
    get_dataset_root,
)


@torch.inference_mode()
def hard_indices_from_images(model, method, images, temperature):
    """
    Tokenize images by quantization method

    single-stage o/p: [B, H, W]
    multi-stage o/p: [B, S, H, W]
    """
    z = model.encoder(images)
    quantizer = model.quantizer

    if method == "avq":
        output = quantizer(z, temperature=temperature)
        return output["indices"].long()

    if method == "arvq":
        output = quantizer(z, temperature=temperature, soft_weight=0.0, mode="hard")
        return output["indices"].long()

    if method == "diveq":
        B, D, H, W = z.shape
        z_flat = z.permute(0, 2, 3, 1).contiguous().view(-1, D)
        _, indices, _ = quantizer.inference(z_flat)
        return indices.view(B, H, W).long()

    if method == "rdiveq":
        B, D, H, W = z.shape
        z_flat = z.permute(0, 2, 3, 1).contiguous().view(-1, D)
        _, indices_list, _ = quantizer.inference(z_flat)
        stages = [idx.view(B, H, W) for idx in indices_list]
        return torch.stack(stages, dim=1).long()

    if method == "stgs":
        logits = model.logit_proj(z)
        output = model.quantizer.inference(logits)
        return output["indices"].long()

    if method == "rstgs":
        B, D, H, W = z.shape
        output = model.quantizer.inference(z)
        indices_list = output["indices_levels"] # List of [B, H, W]
        stages = [idx.view(B, H, W) for idx in indices_list]
        return torch.stack(stages, dim=1).long()

    if method in {"vq", "vqema", "fsq", "rvqema"}:
        output = quantizer(z)
        return output["indices"].long()

    raise ValueError(f"Unsupported quantizer method {method!r}.")


def structured_indices_to_sequence(indices):
    """
    single-stage: [B, H, W] -> [B, H * W]
    multi-stage: [B, S, H, W] -> [B, H * W, S]
    """
    if indices.ndim == 3:
        B, H, W = indices.shape
        return indices.reshape(B, H * W)
    if indices.ndim == 4:
        B, S, H, W = indices.shape
        return indices.permute(0, 2, 3, 1).contiguous().view(B, H * W, S)
    raise ValueError(f"Unexpected indices shape: {tuple(indices.shape)}")


def sequence_to_structured_indices(sequence, latent_grid, num_stages):
    """
    Single-stage: [B, H * W] -> [B, H, W]
    multi-stage: [B, H * W, S] -> [B, S, H, W]
    """
    B = sequence.shape[0]
    H = W = latent_grid
    L = H * W
    if num_stages == 1:
        return sequence.view(B, H, W)
    return sequence.view(B, H, W, num_stages).permute(0, 3, 1, 2).contiguous()


def extract_split_tokens(model, method, loader, device, temperature, vocab_size, desc):
    all_tokens = []
    all_labels = []
    num_stages = None
    latent_grid = None
    stage_histogram = None

    for batch in tqdm(loader, desc=desc, dynamic_ncols=True):
        images, labels = _extract_images_labels(batch)
        images = images.to(device, non_blocking=True)

        indices = hard_indices_from_images(
            model=model,
            method=method,
            images=images,
            temperature=temperature,
        )

        if indices.ndim == 3:
            B, H, W = indices.shape
            S = 1
            stage_view = indices[:, None, :, :]
        else:
            B, S, H, W = indices.shape
            stage_view = indices

        if latent_grid is None:
            latent_grid = H
            num_stages = S
            stage_histogram = torch.zeros(S, vocab_size, dtype=torch.int64)

        sequence = structured_indices_to_sequence(indices)

        # TODO: Remove after validating pipeline
        reconstructed_indices = sequence_to_structured_indices(
            sequence=sequence, latent_grid=H, num_stages=S,
        )
        assert torch.equal(indices, reconstructed_indices)
        # ------------------------------------------------------

        all_tokens.append(sequence.to("cpu", dtype=torch.int16))
        all_labels.append(labels.to("cpu", dtype=torch.int16))

        for stage in range(num_stages):
            counts = torch.bincount(
                stage_view[:, stage].reshape(-1).cpu(), minlength=vocab_size,
            )
            stage_histogram[stage] += counts

    tokens = torch.cat(all_tokens, dim=0)
    labels = torch.cat(all_labels, dim=0)

    stats = {
        "per_stage": [
            histogram_stats(stage_histogram[stage]) for stage in range(stage_histogram.shape[0])
        ],
    }

    return tokens, labels, latent_grid, num_stages, stats


def main():
    # 0. Parse CLI arguments
    parser = argparse.ArgumentParser(description="Extract deterministic hard token IDs.")
    parser.add_argument("--project_root", type=str, required=True)
    parser.add_argument(
        "--experiment_id", type=int, default=0, choices=EXPERIMENTS.keys(), help="Experiment ID from PRIOR_EXPERIMENTS",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--model_seed", type=int, default=42)

    args = parser.parse_args()
    
    # 1. Reproducibility
    DATA_SEED = 15
    model_seed = args.model_seed
    set_global_seed(model_seed)

    # 2. Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 3. Load autoencoder
    project_root = args.project_root
    entry = EXPERIMENTS[args.experiment_id]
    dataset = entry["dataset"]
    experiment_name = entry["experiment"]
    checkpoint_path = get_autoencoder_checkpoint(
        root_dir=project_root, 
        dataset=dataset, 
        model_seed=model_seed, 
        experiment=experiment_name,
    )
    model, experiment, checkpoint = load_autoencoder(
        experiment_name, checkpoint_path=checkpoint_path, device=device,
    )

    model.eval()

    method = experiment.method

    # 4. Build data loaders
    data_root = get_dataset_root(dataset)
    train_loader, holdout_loader, _, class_to_idx, idx_to_label = build_deterministic_token_loaders(
        dataset=dataset,
        data_root=data_root,
        batch_size=args.batch_size,
        data_seed=DATA_SEED,
    )

    # 5. Extract hard tokens
    print(f"Extracting hard tokens | method={method} | experiment={experiment_name}")
    
    TEMPERATURE = 0.02 # Only AVQ and ARVQ use temperature during inference
    train_tokens, train_labels, train_grid, train_num_stages, train_stats = extract_split_tokens(
        model=model,
        method=method,
        loader=train_loader,
        device=device,
        temperature=TEMPERATURE,
        vocab_size=experiment.num_codes,
        desc="Extract train tokens",
    )

    val_tokens, val_labels, val_grid, val_num_stages, val_stats = extract_split_tokens(
        model=model,
        method=method,
        loader=holdout_loader,
        device=device,
        temperature=TEMPERATURE,
        vocab_size=experiment.num_codes,
        desc="Extract val tokens",
    )

    assert (train_grid == val_grid) and (train_num_stages == val_num_stages)

    # 6. Prepare data for saving
    labels_all = torch.cat([train_labels.long(), val_labels.long()])
    num_classes = int(labels_all.max().item() + 1)
    unique_classes = torch.unique(labels_all)
    if unique_classes.numel() != num_classes:
        raise ValueError("Class labels are not contiguous 0..C-1.")

    payload = {
        "train_tokens": train_tokens,
        "train_labels": train_labels,
        "val_tokens": val_tokens,
        "val_labels": val_labels,
    }

    metadata = {
        "dataset": dataset,
        "data_root": data_root,
        "data_seed": DATA_SEED,
        "model_seed": model_seed,
        "experiment": experiment_name,
        "method": method,
        "autoencoder_checkpoint": str(checkpoint_path),
        "hard_eval_temperature": TEMPERATURE if method in {"avq", "arvq"} else None,
        "vocab_size": experiment.num_codes,
        "embedding_dim": experiment.embedding_dim,
        "latent_grid": train_grid,
        "num_stages": train_num_stages,
        "sequence_length": train_tokens.shape[1],
        "num_classes": num_classes,
        "num_train_examples": train_tokens.shape[0],
        "num_val_examples": val_tokens.shape[0],
        "train_token_stats": train_stats,
        "val_token_stats": val_stats,
    }

    payload["metadata"] = metadata

    # 7. Save tokenized data and metadata
    output_path = get_token_file(
        root_dir=project_root, dataset=dataset, model_seed=model_seed, experiment=experiment_name,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)

    output_dir = get_token_dir(
        root_dir=project_root, dataset=dataset, model_seed=model_seed, experiment=experiment_name,
    )
    json_dump(metadata, output_dir / "metadata.json")

    print(f"Saved token dataset: {output_path}")
    print(f"Train tokens: {tuple(train_tokens.shape)}")
    print(f"Val tokens: {tuple(val_tokens.shape)}")
    if train_num_stages == 1:
        train_ppl = train_stats["per_stage"][0]["perplexity"]
        print(f"Hard-token train PPL: {train_ppl:.3f}")
    else:
        stage_ppl = [x["perplexity"] for x in train_stats["per_stage"]]
        print(f"Per-stage train PPL: {[round(x, 3) for x in stage_ppl]}")


if __name__ == "__main__":

    main()
