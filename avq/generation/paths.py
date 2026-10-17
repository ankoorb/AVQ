from pathlib import Path

"""
experiment_root/
├── autoencoders/
│   ├── tiny-imagenet/
│   │   └── seed-42/
│   │       ├── avq_16x16_k1024_d64/
│   │       │   └── calibration/
│   │       │       └── best_decoder_calibration.pt
│   │       ├── fsq_16x16_k1000_d64/
│   │       │   └── calibration/
│   │       │       └── best_decoder_calibration.pt
│   │       └── ...
│   └── mini-places/
│       └── seed-42/
│           └── ...
│
├── tokens/
│   ├── tiny-imagenet/
│   │   └── seed-42/
│   │       ├── avq_16x16_k1024_d64/
│   │       │   ├── data.pt
│   │       │   └── metadata.json
│   │       └── ...
│   └── mini-places/
│       └── seed-42/
│           └── ...
│
├── priors/
│   ├── tiny-imagenet/
│   │   └── seed-42/
│   │       ├── avq_16x16_k1024_d64_prior/
│   │       │   ├── prior_best.pt
│   │       │   ├── prior_last.pt
│   │       │   └── summary.json
│   │       └── ...
│   └── mini-places/
│       └── seed-42/
│           └── ...
│
└── generated/
    ├── tiny-imagenet/
    │   └── seed-42/
    │       ├── avq_16x16_k1024_d64_prior/
    │       │   ├── generated_000.pt
    │       │   ├── generated_001.pt
    │       │   │   ...
    │       │   └── manifest.json
    │       │       
    │       │       
    │       │       
    │       └── ...
    └── mini-places/
        └── seed-42/
            └── ...

gFID 
----
/workspace/gfid/
├── tiny-imagenet/
│   └── seed-42/
│       └── avq_16x16_k1024_d64/
│           └── gfid.json
└── mini-places/
    └── seed-42/
        └── avq_32x32_k1024_d64/
            └── gfid.json
"""

EXPERIMENTS = {
    # Tiny ImageNet
    0: {
        "dataset": "tiny-imagenet",
        "experiment": "avq_16x16_k1024_d64",
    },
    1: {
        "dataset": "tiny-imagenet",
        "experiment": "vq_16x16_k1024_d64",
    },
    2: {
        "dataset": "tiny-imagenet",
        "experiment": "vqema_16x16_k1024_d64",
    },
    3: {
        "dataset": "tiny-imagenet",
        "experiment": "fsq_16x16_k1000_d64",
    },
    4: {
        "dataset": "tiny-imagenet",
        "experiment": "diveq_16x16_k1024_d64",
    },
    5: {
        "dataset": "tiny-imagenet",
        "experiment": "stgs_16x16_k1024_d64",
    },
    6: {
        "dataset": "tiny-imagenet",
        "experiment": "rvqema_16x16_q3_k1024_d64",
    },
    7: {
        "dataset": "tiny-imagenet",
        "experiment": "rdiveq_16x16_q3_k1024_d64",
    },
    8: {
        "dataset": "tiny-imagenet",
        "experiment": "rstgs_16x16_q3_k1024_d64",
    },
    9: {
        "dataset": "tiny-imagenet",
        "experiment": "arvq_16x16_q3_k1024_d64",
    },
    
    # MiniPlaces
    10: {
        "dataset": "mini-places",
        "experiment": "avq_32x32_k1024_d64",
    },
    11: {
        "dataset": "mini-places",
        "experiment": "vq_32x32_k1024_d64",
    },
    12: {
        "dataset": "mini-places",
        "experiment": "vqema_32x32_k1024_d64",
    },
    13: {
        "dataset": "mini-places",
        "experiment": "fsq_32x32_k1000_d64",
    },
    14: {
        "dataset": "mini-places",
        "experiment": "diveq_32x32_k1024_d64",
    },
    15: {
        "dataset": "mini-places",
        "experiment": "stgs_32x32_k1024_d64",
    },
    16: {
        "dataset": "mini-places",
        "experiment": "rvqema_32x32_q3_k1024_d64",
    },
    17: {
        "dataset": "mini-places",
        "experiment": "rdiveq_32x32_q3_k1024_d64",
    },
    18: {
        "dataset": "mini-places",
        "experiment": "rstgs_32x32_q3_k1024_d64",
    },
    19: {
        "dataset": "mini-places",
        "experiment": "arvq_32x32_q3_k1024_d64",
    },
}


def get_autoencoder_dir(root_dir, dataset, model_seed, experiment):
    """
    root_dir: /workspace/experiments
    """
    root_dir = Path(root_dir)
    return root_dir / "autoencoders" / dataset / f"seed-{model_seed}" / experiment


def get_autoencoder_checkpoint(root_dir, dataset, model_seed, experiment):
    """
    root_dir: /workspace/experiments
    """
    autoencoder_dir = get_autoencoder_dir(root_dir, dataset, model_seed, experiment)
    if experiment.startswith("arvq_"):
        checkpoint_name = "best_arvq_decoder_calibration.pt"
    else:
        checkpoint_name =  "best_decoder_calibration.pt"

    return autoencoder_dir / "calibration" / checkpoint_name


def get_token_dir(root_dir, dataset, model_seed, experiment):
    """
    root_dir: /workspace/experiments
    """
    root_dir = Path(root_dir)
    return root_dir / "tokens" / dataset / f"seed-{model_seed}" / experiment


def get_token_file(root_dir, dataset, model_seed, experiment):
    """
    root_dir: /workspace/experiments
    """
    token_dir = get_token_dir(root_dir, dataset, model_seed, experiment)
    return token_dir / "data.pt"


def get_prior_dir(root_dir, dataset, model_seed, experiment):
    """
    root_dir: /workspace/experiments
    """
    root_dir = Path(root_dir)
    prior_experiment = f"{experiment}_prior"
    return root_dir / "priors" / dataset / f"seed-{model_seed}" / prior_experiment


def get_prior_checkpoint(root_dir, dataset, model_seed, experiment):
    """
    root_dir: /workspace/experiments
    """
    prior_dir = get_prior_dir(root_dir, dataset, model_seed, experiment)
    return prior_dir / "prior_best.pt"


def get_generated_dir(root_dir, dataset, model_seed, experiment):
    """
    root_dir: /workspace/experiments
    """
    root_dir = Path(root_dir)
    prior_experiment = f"{experiment}_prior"
    return root_dir / "generated" / dataset / f"seed-{model_seed}" / prior_experiment


def get_dataset_root(dataset):
    if dataset == "tiny-imagenet":
        return Path("/tmp/data/tiny-imagenet-200")

    if dataset == "mini-places":
        return Path("/tmp/data/miniplaces")

    raise ValueError(f"Unknown dataset: {dataset}")


def get_gfid_dir(output_dir, dataset, model_seed, experiment):
    """
    output_dir: /workspace/gfid/
    """
    output_dir = Path(output_dir)
    return output_dir / dataset / f"seed-{model_seed}" / experiment


def get_gfid_path(output_dir, dataset, model_seed, experiment):
    """
    output_dir: /workspace/gfid/
    """
    gfid_dir = get_gfid_dir(output_dir, dataset, model_seed, experiment)
    return gfid_dir / "gfid.json"
