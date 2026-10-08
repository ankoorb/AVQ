import json
import math
import torch

from torch.utils.data import DataLoader, Subset

from avq.config import EXPERIMENT_REGISTRY
from avq.dataset import build_tiny_imagenet_loaders, build_mini_places_loaders


def _extract_images_labels(batch):
    return batch[0], batch[1]


def json_dump(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as f:
        json.dump(obj, f, indent=4)


def model_parameter_count(model):
    return sum(p.numel() for p in model.parameters())


def _strip_module_prefix(state_dict):
    if state_dict and all(k.startswith("module.") for k in state_dict):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


def load_autoencoder(experiment_name, checkpoint_path, device):
    if experiment_name not in EXPERIMENT_REGISTRY:
        experiment_names = EXPERIMENT_REGISTRY.keys()
        available_experiments = "\n ".join(sorted(experiment_names))
        raise KeyError(
            f"Experiment {experiment_name!r} is not in EXPERIMENT_REGISTRY.\n"
            f"Available experiments: \n {available_experiments}"
        )

    experiment = EXPERIMENT_REGISTRY[experiment_name]
    model = experiment.model_cls(**experiment.model_kwargs).to(device)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model_state_dict"]
    state_dict = _strip_module_prefix(state_dict)

    model.load_state_dict(state_dict, strict=True)
    model.eval() # TODO: Check DiVEQ eval may replace codebook vectors
    for param in model.parameters():
        param.requires_grad_(False) 

    return model, experiment, checkpoint


def build_dataset_loaders(dataset, data_root, batch_size, data_seed=15):
    if dataset == "tiny":
        return build_tiny_imagenet_loaders(
                root_directory=data_root,
                batch_size=batch_size,
                num_workers=4,
                image_size=64,
                seed=data_seed,
                drop_last_train=False,
            )

    if dataset == "miniplaces":
        return build_mini_places_loaders(
                root_directory=data_root,
                batch_size=batch_size,
                num_workers=4,
                image_size=128,
                seed=data_seed,
                drop_last_train=False,
            )

    raise ValueError(
        "Class-conditional generation supports Tiny ImageNet and MiniPlaces. "
        "COCO is intentionally excluded because the current COCO task is multi-label "
        "rather than a single class label per image."
    )


def build_deterministic_token_loaders(dataset, data_root, batch_size, data_seed):
    """
    Reuse exact stratified 95/5 split. 
    """
    train_loader, train_eval_loader, val_loader, class_to_idx, idx_to_label = build_dataset_loaders(
        dataset=dataset,
        data_root=data_root,
        batch_size=batch_size,
        num_workers=4,
        data_seed=data_seed,
    )

    train_subset = train_loader.dataset
    eval_subset = train_eval_loader.dataset

    deterministic_full_train = eval_subset.dataset
    deterministic_train = Subset(deterministic_full_train, train_subset.indices)
    deterministic_holdout = eval_subset

    num_workers = 4
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
        "shuffle": False,
        "drop_last": False,
    }

    deterministic_train_loader = DataLoader(deterministic_train, **common)
    deterministic_holdout_loader = DataLoader(deterministic_holdout, **common)

    return (
        deterministic_train_loader,
        deterministic_holdout_loader,
        val_loader,
        class_to_idx,
        idx_to_label,
    )


def histogram_stats(histogram):
    histogram = histogram.double().cpu()
    total = histogram.sum().item()
    probs = histogram / total
    non_zero = probs > 0
    entropy = -(probs[non_zero] * probs[non_zero].log()).sum().item()
    active_codes = int(non_zero.sum().item())
    return {
        "perplexity": float(math.exp(entropy)),
        "active_codes": active_codes,
        "utilization": float(active_codes / histogram.numel()),
        "num_tokens": int(total),
    }


def time_training_steps(model, batch_size, device, warmup=5, timed_steps=25):
    """
    Time forward + backward + optimizer steps on synthetic token data
    """
    import time

    config = model.config

    def random_batch():
        indices = torch.randint(
            low=0,
            high=config.vocab_size,
            size=(batch_size, config.seq_length, config.num_codebooks),
            device=device,
        )

        labels = torch.randint(
            low=0,
            high=config.num_classes,
            size=(batch_size,),
            device=device,
        )

        return indices, labels
    
    model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

    for _ in range(warmup):
        indices, labels = random_batch()
        loss = model.compute_loss(indices, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    start = time.perf_counter()
    for _ in range(timed_steps):
        indices, labels = random_batch()
        loss = model.compute_loss(indices, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    elapsed = time.perf_counter() - start
    steps_per_sec = timed_steps / elapsed

    print(f"Params: {model.num_parameters():,}")
    print(
        f"Sequence length: {config.seq_length} | "
        f"Num codebooks: {config.num_codebooks} | "
        f"Batch size: {batch_size}"
    )
    print(
        f"{steps_per_sec:.3f} steps/sec | "
        f"{elapsed / timed_steps * 1000.0:.1f} ms/step"
    )

    return steps_per_sec

