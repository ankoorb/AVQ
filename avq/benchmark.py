import json
import torch
import argparse
import statistics

from avq.config import EXPERIMENT_REGISTRY


# 16x16 quantizer configurations for Tiny ImageNet / COCO
EXPERIMENTS = {
    "vq": "vq_16x16_k1024_d64",
    "vqema": "vqema_16x16_k1024_d64",
    "fsq": "fsq_16x16_k1000_d64",
    "avq": "avq_16x16_k1024_d64",
    "diveq": "diveq_16x16_k1024_d64",
    "rvqema": "rvqema_16x16_q3_k1024_d64",
    "arvq": "arvq_16x16_q3_k1024_d64",
    "rdiveq": "rdiveq_16x16_q3_k1024_d64",
    "stgs": "stgs_16x16_k1024_d64",
    "rstgs": "rstgs_16x16_q3_k1024_d64",
}


METHODS = [
    "vq",
    "vqema",
    "fsq",
    "avq",
    "diveq",
    "stgs",
    "rvqema",
    "arvq",
    "rdiveq",
    "rstgs",
]


def build_quantizer(method, device):
    experiment_name = EXPERIMENTS[method]

    if experiment_name not in EXPERIMENT_REGISTRY:
        raise KeyError(
            f"Experiment '{experiment_name}' is not in EXPERIMENT_REGISTRY. "
            f"Add the registry entry first or update EXPERIMENT_NAMES in duration.py."
        )

    experiment = EXPERIMENT_REGISTRY[experiment_name]
    model = experiment.model_cls(**experiment.model_kwargs)
    model.eval()

    # Keep quantizer only. Exclude Encoder/Decoder time and memory
    quantizer = model.quantizer.to(device)

    # ST-GS has external D -> K categorical-logit projection
    logit_proj = model.logit_proj if method == "stgs" else None

    quantizer.eval()

    if logit_proj is not None:
        logit_proj.eval()

    del model
    
    return quantizer, logit_proj


@torch.inference_mode()
def quantize(method, quantizer, z, temperature=None, logit_proj=None):
    """
    Quantizer hard inference

    z: [B, D, H, W]
    """
    if method == "avq":
        outputs = quantizer(z, temperature=temperature)
        return outputs["hard_quantized"]

    if method == "arvq":
        outputs = quantizer(
            z, temperature=temperature, soft_weight=0.0, mode="hard",
        )
        return outputs["z_quantized"]

    if method in {"vq", "vqema", "fsq", "rvqema"}:
        outputs = quantizer(z)
        return outputs["hard_quantized"]

    if method in {"diveq", "rdiveq"}:
        B, D, H, W = z.shape
        z_flat = z.permute(0, 2, 3, 1).reshape(-1, D)
        z_q_flat, _, _ = quantizer.inference(z_flat)
        z_q = z_q_flat.reshape(B, H, W, D).permute(0, 3, 1, 2).contiguous()
        return z_q

    if method == "stgs":
        logits = logit_proj(z) # [B, D, H, W] -> [B, K, H, W]
        outputs = quantizer.inference(logits)
        return outputs["hard_quantized"]

    if method == "rstgs":
        outputs = quantizer.inference(z)
        return outputs["hard_quantized"]

    raise ValueError(f"Unsupporded method: {method}")


def benchmark_method(method, z, device, temperature=0.02, warmup=5, repeats=10):
    quantizer, logit_proj = build_quantizer(method, device)

    # Warmup CUDA kernels before timing
    for _ in range(warmup):
        output = quantize(method, quantizer, z, temperature, logit_proj=logit_proj)
    torch.cuda.synchronize(device)

    output = quantize(method, quantizer, z, temperature, logit_proj=logit_proj)
    torch.cuda.synchronize(device)      

    if tuple(output.shape) != tuple(z.shape):
        raise RuntimeError(
            f"{method}: output shape {tuple(output.shape)} != input shape {tuple(z.shape)}"
        )

    # Measure temporary workspace above the already allocated input + quantizer
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    baseline_memory = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)

    timings_ms = []
    for _ in range(repeats):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        output = quantize(method, quantizer, z, temperature, logit_proj=logit_proj)
        end_event.record()

        torch.cuda.synchronize(device)
        timings_ms.append(start_event.elapsed_time(end_event))

    peak_memory = torch.cuda.max_memory_allocated(device)
    workspace_memory = max(0, peak_memory - baseline_memory)

    batch_size = z.shape[0]
    mean_ms = statistics.mean(timings_ms)
    std_ms = statistics.stdev(timings_ms) if len(timings_ms) > 1 else 0.0
    median_ms = statistics.median(timings_ms)

    result = {
        "method": method,
        "batch_size": batch_size,
        "latent_shape": list(z.shape[1:]),
        "warmup_runs": warmup,
        "timed_runs": repeats,
        "mean_batch_ms": mean_ms,
        "std_batch_ms": std_ms,
        "median_batch_ms": median_ms,
        "mean_ms_per_image": mean_ms / batch_size,
        "images_per_second": batch_size / (mean_ms / 1000.0),
        "peak_workspace_mb": workspace_memory / (1024 ** 2),
        "timings_ms": timings_ms,
    }

    del output
    del quantizer
    if logit_proj is not None:
        del logit_proj
    torch.cuda.empty_cache()

    return result


def main():
    parser = argparse.ArgumentParser(description="Benchmark quantizer-only hard inference.")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=METHODS,
        choices=METHODS,
        help="Quantizers to benchmark.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Number of latent feature maps B; total quantized tokens are B * latent_grid^2.",
    )
    parser.add_argument(
        "--latent_grid",
        type=int,
        default=16,
        choices=[16, 32],
        help="Spatial latent grid: 16 for Tiny/COCO, 32 for MiniPlaces.",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=64,
        help="Latent embedding dimension.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.02,
        help="Hard-evaluation temperature passed to AVQ/ARVQ.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
        help="Warmup runs before timing.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=100,
        help="Number of timed repetitions.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the synthetic latent tensor.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark.json",
        help="JSON file for benchmark results.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this benchmark.")

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Shared random latent batch for every quantizer
    z = torch.randn(
        args.batch_size,
        args.embedding_dim,
        args.latent_grid,
        args.latent_grid,
        device=device,
        dtype=torch.float32,
    )

    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Input: {tuple(z.shape)} | dtype={z.dtype}")
    print(f"Warmup: {args.warmup} | Repeats: {args.repeats}")
    print()

    results = {
        "gpu": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "batch_size": args.batch_size,
        "embedding_dim": args.embedding_dim,
        "latent_grid": args.latent_grid,
        "temperature": args.temperature,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "methods": {},
    }

    for method in args.methods:
        print(f"Benchmarking {method.upper()} ...")

        try:
            result = benchmark_method(
                method=method,
                z=z,
                device=device,
                temperature=args.temperature,
                warmup=args.warmup,
                repeats=args.repeats,
            )
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"  OOM: {method}")
            results["methods"][method] = {"error": "CUDA out of memory"}
            continue

        results["methods"][method] = result

        print(
            f"  {result['mean_batch_ms']:.3f} ± {result['std_batch_ms']:.3f} ms / batch | "
            f"{result['mean_ms_per_image']:.6f} ms / image | "
            f"{result['images_per_second']:.1f} images/s | "
            f"workspace {result['peak_workspace_mb']:.1f} MB"
        )

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    """
    cd /workspace/AVQ
    python -m compileall avq
    Paper 16x16: python -m avq.benchmark --latent_grid 16 --batch_size 64 --output /workspace/benchmark_16x16.json
    paper 32x32: python -m avq.benchmark --latent_grid 32 --batch_size 16 --output /workspace/benchmark_32x32.json
    """
    main()
