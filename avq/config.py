from typing import Any
from dataclasses import dataclass, field


from avq.model import (
    AVQAutoencoder8x8, 
    AVQAutoencoder16x16,
    PatchDiscriminator8x8, 
    PatchDiscriminator16x16,
    VQAutoencoder16x16,
    FSQAutoencoder16x16,
    VQEMAAutoencoder16x16,
    ARVQAutoencoder16x16,
    AVQAutoencoder32x32,
    PatchDiscriminator128Sq,
    VQAutoencoder32x32,
    VQEMAAutoencoder32x32,
    FSQAutoencoder32x32,
    ARVQAutoencoder32x32,
    RVQEMAAutoencoder16x16,
    RVQEMAAutoencoder32x32,
)


@dataclass(frozen=True)
class ExperimentConfig:
    method: str
    model_cls: type
    discriminator_cls: type
    embedding_dim: int
    latent_grid: int
    num_codes: int
    model_kwargs: dict[str, Any] = field(default_factory=dict)


EXPERIMENT_REGISTRY = {
    # Tiny ImageNet Experiments
    # -------------------------
    "avq_16x16_k1024_d64": ExperimentConfig(
        method="avq",
        model_cls=AVQAutoencoder16x16,
        discriminator_cls=PatchDiscriminator16x16,
        embedding_dim=64,
        latent_grid=16,
        num_codes=1024,
        model_kwargs={
            "num_embeddings": 1024,
            "embedding_dim": 64,
            "normalize": True,
        },
    ),

    "avq_16x16_k512_d64": ExperimentConfig(
        method="avq",
        model_cls=AVQAutoencoder16x16,
        discriminator_cls=PatchDiscriminator16x16,
        embedding_dim=64,
        latent_grid=16,
        num_codes=512,
        model_kwargs={
            "num_embeddings": 512,
            "embedding_dim": 64,
            "normalize": True,
        },
    ),

    "avq_16x16_k1024_d128": ExperimentConfig(
        method="avq",
        model_cls=AVQAutoencoder16x16,
        discriminator_cls=PatchDiscriminator16x16,
        embedding_dim=128,
        latent_grid=16,
        num_codes=1024,
        model_kwargs={
            "num_embeddings": 1024,
            "embedding_dim": 128,
            "normalize": True,
        },
    ),

    "avq_8x8_k1024_d64": ExperimentConfig(
        method="avq",
        model_cls=AVQAutoencoder8x8,
        discriminator_cls=PatchDiscriminator8x8,
        embedding_dim=64,
        latent_grid=8,
        num_codes=1024,
        model_kwargs={
            "num_embeddings": 1024,
            "embedding_dim": 64,
            "normalize": True,
        },
    ),

    "vq_16x16_k1024_d64": ExperimentConfig(
        method="vq",
        model_cls=VQAutoencoder16x16,
        discriminator_cls=PatchDiscriminator16x16,
        embedding_dim=64,
        latent_grid=16,
        num_codes=1024,
        model_kwargs={
            "num_embeddings": 1024,
            "embedding_dim": 64,
            "commitment_cost": 0.25,
        },
    ),

    "vqema_16x16_k1024_d64": ExperimentConfig(
        method="vqema",
        model_cls=VQEMAAutoencoder16x16,
        discriminator_cls=PatchDiscriminator16x16,
        embedding_dim=64,
        latent_grid=16,
        num_codes=1024,
        model_kwargs={
            "num_embeddings": 1024,
            "embedding_dim": 64,
            "commitment_cost": 0.25,
            "decay": 0.99,
            "epsilon": 1e-5,
        },
    ),

    "fsq_16x16_k1000_d64": ExperimentConfig(
        method="fsq",
        model_cls=FSQAutoencoder16x16,
        discriminator_cls=PatchDiscriminator16x16,
        embedding_dim=64,
        latent_grid=16,
        num_codes=1000,
        model_kwargs={
            "embedding_dim": 64,
            "levels": (8, 5, 5, 5),
        },
    ),

    "avq_16x16_k1024_d64_no_norm": ExperimentConfig(
        method="avq",
        model_cls=AVQAutoencoder16x16,
        discriminator_cls=PatchDiscriminator16x16,
        embedding_dim=64,
        latent_grid=16,
        num_codes=1024,
        model_kwargs={
            "num_embeddings": 1024,
            "embedding_dim": 64,
            "normalize": False,
        },
    ),

    "arvq_16x16_q3_k1024_d64": ExperimentConfig(
        method="arvq",
        model_cls=ARVQAutoencoder16x16,
        discriminator_cls=PatchDiscriminator16x16,
        embedding_dim=64,
        latent_grid=16,
        num_codes=1024,
        model_kwargs={
            "num_quantizers": 3,
            "num_embeddings": 1024,
            "embedding_dim": 64,
        },
    ),

    "rvqema_16x16_q3_k1024_d64": ExperimentConfig(
        method="rvqema",
        model_cls=RVQEMAAutoencoder16x16,
        discriminator_cls=PatchDiscriminator16x16,
        embedding_dim=64,
        latent_grid=16,
        num_codes=1024,
        model_kwargs={
            "num_quantizers": 3,
            "num_embeddings": 1024,
            "embedding_dim": 64,
            "commitment_cost": 0.25,
            "decay": 0.99,
            "epsilon": 1e-5,
        },
    ),

    "avq_16x16_k1024_d64_scaled": ExperimentConfig(
            method="avq",
            model_cls=AVQAutoencoder16x16,
            discriminator_cls=PatchDiscriminator16x16,
            embedding_dim=64,
            latent_grid=16,
            num_codes=1024,
            model_kwargs={
                "num_embeddings": 1024,
                "embedding_dim": 64,
                "normalize": True,
                "learned_code_scale": True,
            },
        ),

    "arvq_16x16_q3_k1024_d64_scaled": ExperimentConfig(
        method="arvq",
        model_cls=ARVQAutoencoder16x16,
        discriminator_cls=PatchDiscriminator16x16,
        embedding_dim=64,
        latent_grid=16,
        num_codes=1024,
        model_kwargs={
            "num_quantizers": 3,
            "num_embeddings": 1024,
            "embedding_dim": 64,
            "learned_code_scale": True,
        },
    ),

    # Mini Places Experiments
    # -----------------------
    "avq_32x32_k1024_d64": ExperimentConfig(
        method="avq",
        model_cls=AVQAutoencoder32x32,
        discriminator_cls=PatchDiscriminator128Sq,
        embedding_dim=64,
        latent_grid=32,
        num_codes=1024,
        model_kwargs={
            "num_embeddings": 1024,
            "embedding_dim": 64,
            "normalize": True,
            "latent_16x16": False,
        },
    ),

    "vq_32x32_k1024_d64": ExperimentConfig(
        method="vq",
        model_cls=VQAutoencoder32x32,
        discriminator_cls=PatchDiscriminator128Sq,
        embedding_dim=64,
        latent_grid=32,
        num_codes=1024,
        model_kwargs={
            "num_embeddings": 1024,
            "embedding_dim": 64,
            "commitment_cost": 0.25,
        },
    ),
    
    "vqema_32x32_k1024_d64": ExperimentConfig(
        method="vqema",
        model_cls=VQEMAAutoencoder32x32,
        discriminator_cls=PatchDiscriminator128Sq,
        embedding_dim=64,
        latent_grid=32,
        num_codes=1024,
        model_kwargs={
            "num_embeddings": 1024,
            "embedding_dim": 64,
            "commitment_cost": 0.25,
            "decay": 0.99,
            "epsilon": 1e-5,
        },
    ),
    
    "fsq_32x32_k1000_d64": ExperimentConfig(
        method="fsq",
        model_cls=FSQAutoencoder32x32,
        discriminator_cls=PatchDiscriminator128Sq,
        embedding_dim=64,
        latent_grid=32,
        num_codes=1000,
        model_kwargs={
            "embedding_dim": 64,
            "levels": (8, 5, 5, 5),
        },
    ),

    "arvq_32x32_q3_k1024_d64": ExperimentConfig(
        method="arvq",
        model_cls=ARVQAutoencoder32x32,
        discriminator_cls=PatchDiscriminator128Sq,
        embedding_dim=64,
        latent_grid=32,
        num_codes=1024,
        model_kwargs={
            "num_quantizers": 3,
            "num_embeddings": 1024,
            "embedding_dim": 64,
            "latent_16x16": False,
        },
    ),

    "avq_128sq_16x16_k1024_d64": ExperimentConfig(
        method="avq",
        model_cls=AVQAutoencoder32x32,
        discriminator_cls=PatchDiscriminator128Sq,
        embedding_dim=64,
        latent_grid=16,
        num_codes=1024,
        model_kwargs={
            "num_embeddings": 1024,
            "embedding_dim": 64,
            "normalize": True,
            "latent_16x16": True,
        },
    ),

    "arvq_128sq_16x16_q3_k1024_d64": ExperimentConfig(
        method="arvq",
        model_cls=ARVQAutoencoder32x32,
        discriminator_cls=PatchDiscriminator128Sq,
        embedding_dim=64,
        latent_grid=16,
        num_codes=1024,
        model_kwargs={
            "num_quantizers": 3,
            "num_embeddings": 1024,
            "embedding_dim": 64,
            "latent_16x16": True,
        },
    ),

    "avq_32x32_k512_d64": ExperimentConfig(
        method="avq",
        model_cls=AVQAutoencoder32x32,
        discriminator_cls=PatchDiscriminator128Sq,
        embedding_dim=64,
        latent_grid=32,
        num_codes=512,
        model_kwargs={
            "num_embeddings": 512,
            "embedding_dim": 64,
            "normalize": True,
            "latent_16x16": False,
        },
    ),

    "avq_32x32_k1024_d128": ExperimentConfig(
        method="avq",
        model_cls=AVQAutoencoder32x32,
        discriminator_cls=PatchDiscriminator128Sq,
        embedding_dim=128,
        latent_grid=32,
        num_codes=1024,
        model_kwargs={
            "num_embeddings": 1024,
            "embedding_dim": 128,
            "normalize": True,
            "latent_16x16": False,
        },
    ),

    "avq_32x32_k1024_d64_no_norm": ExperimentConfig(
        method="avq",
        model_cls=AVQAutoencoder32x32,
        discriminator_cls=PatchDiscriminator128Sq,
        embedding_dim=64,
        latent_grid=32,
        num_codes=1024,
        model_kwargs={
            "num_embeddings": 1024,
            "embedding_dim": 64,
            "normalize": False,
            "latent_16x16": False,
        },
    ),

    "rvqema_32x32_q3_k1024_d64": ExperimentConfig(
        method="rvqema",
        model_cls=RVQEMAAutoencoder32x32,
        discriminator_cls=PatchDiscriminator128Sq,
        embedding_dim=64,
        latent_grid=32,
        num_codes=1024,
        model_kwargs={
            "num_quantizers": 3,
            "num_embeddings": 1024,
            "embedding_dim": 64,
            "commitment_cost": 0.25,
            "decay": 0.99,
            "epsilon": 1e-5,
        },
    ),

    "avq_32x32_k1024_d64_scaled": ExperimentConfig(
        method="avq",
        model_cls=AVQAutoencoder32x32,
        discriminator_cls=PatchDiscriminator128Sq,
        embedding_dim=64,
        latent_grid=32,
        num_codes=1024,
        model_kwargs={
            "num_embeddings": 1024,
            "embedding_dim": 64,
            "normalize": True,
            "latent_16x16": False,
            "learned_code_scale": True,
        },
    ),

    "arvq_32x32_q3_k1024_d64_scaled": ExperimentConfig(
        method="arvq",
        model_cls=ARVQAutoencoder32x32,
        discriminator_cls=PatchDiscriminator128Sq,
        embedding_dim=64,
        latent_grid=32,
        num_codes=1024,
        model_kwargs={
            "num_quantizers": 3,
            "num_embeddings": 1024,
            "embedding_dim": 64,
            "latent_16x16": False,
            "learned_code_scale": True,
        },
    ),
    

    # COCO Experiments
    # ----------------
    "avq_coco_16x16_k1024_d64": ExperimentConfig(
        method="avq",
        model_cls=AVQAutoencoder16x16,
        discriminator_cls=PatchDiscriminator16x16,
        embedding_dim=64,
        latent_grid=16,
        num_codes=1024,
        model_kwargs={
            "num_embeddings": 1024,
            "embedding_dim": 64,
            "normalize": True,
        },
    ),

    "avq_coco_16x16_k512_d64": ExperimentConfig(
        method="avq",
        model_cls=AVQAutoencoder16x16,
        discriminator_cls=PatchDiscriminator16x16,
        embedding_dim=64,
        latent_grid=16,
        num_codes=512,
        model_kwargs={
            "num_embeddings": 512,
            "embedding_dim": 64,
            "normalize": True,
        },
    ),

    "avq_coco_16x16_k1024_d128": ExperimentConfig(
        method="avq",
        model_cls=AVQAutoencoder16x16,
        discriminator_cls=PatchDiscriminator16x16,
        embedding_dim=128,
        latent_grid=16,
        num_codes=1024,
        model_kwargs={
            "num_embeddings": 1024,
            "embedding_dim": 128,
            "normalize": True,
        },
    ),

    "avq_coco_8x8_k1024_d64": ExperimentConfig(
        method="avq",
        model_cls=AVQAutoencoder8x8,
        discriminator_cls=PatchDiscriminator8x8,
        embedding_dim=64,
        latent_grid=8,
        num_codes=1024,
        model_kwargs={
            "num_embeddings": 1024,
            "embedding_dim": 64,
            "normalize": True,
        },
    ),

    "vq_coco_16x16_k1024_d64": ExperimentConfig(
        method="vq",
        model_cls=VQAutoencoder16x16,
        discriminator_cls=PatchDiscriminator16x16,
        embedding_dim=64,
        latent_grid=16,
        num_codes=1024,
        model_kwargs={
            "num_embeddings": 1024,
            "embedding_dim": 64,
            "commitment_cost": 0.25,
        },
    ),

    "vqema_coco_16x16_k1024_d64": ExperimentConfig(
        method="vqema",
        model_cls=VQEMAAutoencoder16x16,
        discriminator_cls=PatchDiscriminator16x16,
        embedding_dim=64,
        latent_grid=16,
        num_codes=1024,
        model_kwargs={
            "num_embeddings": 1024,
            "embedding_dim": 64,
            "commitment_cost": 0.25,
            "decay": 0.99,
            "epsilon": 1e-5,
        },
    ),

    "fsq_coco_16x16_k1000_d64": ExperimentConfig(
        method="fsq",
        model_cls=FSQAutoencoder16x16,
        discriminator_cls=PatchDiscriminator16x16,
        embedding_dim=64,
        latent_grid=16,
        num_codes=1000,
        model_kwargs={
            "embedding_dim": 64,
            "levels": (8, 5, 5, 5),
        },
    ),

    "arvq_coco_16x16_q3_k1024_d64": ExperimentConfig(
        method="arvq",
        model_cls=ARVQAutoencoder16x16,
        discriminator_cls=PatchDiscriminator16x16,
        embedding_dim=64,
        latent_grid=16,
        num_codes=1024,
        model_kwargs={
            "num_quantizers": 3,
            "num_embeddings": 1024,
            "embedding_dim": 64,
        },
    ),

    "avq_coco_16x16_k1024_d64_no_norm": ExperimentConfig(
        method="avq",
        model_cls=AVQAutoencoder16x16,
        discriminator_cls=PatchDiscriminator16x16,
        embedding_dim=64,
        latent_grid=16,
        num_codes=1024,
        model_kwargs={
            "num_embeddings": 1024,
            "embedding_dim": 64,
            "normalize": False,
        },
    ),

    "rvqema_coco_16x16_q3_k1024_d64": ExperimentConfig(
        method="rvqema",
        model_cls=RVQEMAAutoencoder16x16,
        discriminator_cls=PatchDiscriminator16x16,
        embedding_dim=64,
        latent_grid=16,
        num_codes=1024,
        model_kwargs={
            "num_quantizers": 3,
            "num_embeddings": 1024,
            "embedding_dim": 64,
            "commitment_cost": 0.25,
            "decay": 0.99,
            "epsilon": 1e-5,
        },
    ),

    "avq_coco_16x16_k1024_d64_scaled": ExperimentConfig(
        method="avq",
        model_cls=AVQAutoencoder16x16,
        discriminator_cls=PatchDiscriminator16x16,
        embedding_dim=64,
        latent_grid=16,
        num_codes=1024,
        model_kwargs={
            "num_embeddings": 1024,
            "embedding_dim": 64,
            "normalize": True,
            "learned_code_scale": True,
        },
    ),

    "arvq_coco_16x16_q3_k1024_d64_scaled": ExperimentConfig(
        method="arvq",
        model_cls=ARVQAutoencoder16x16,
        discriminator_cls=PatchDiscriminator16x16,
        embedding_dim=64,
        latent_grid=16,
        num_codes=1024,
        model_kwargs={
            "num_quantizers": 3,
            "num_embeddings": 1024,
            "embedding_dim": 64,
            "learned_code_scale": True,
        },
    ),
}


@dataclass
class TrainingConfig:
    num_epochs: int = 50

    # Loss weights
    usage_weight: float = 0.01
    adversarial_weight: float = 0.01
    adversarial_start_epoch: int = 5

    # AVQ temperature schedule
    temperature_start: float = 0.15
    temperature_end: float = 0.02
    anneal_epochs: int = 40

    # AVQ codebook regularization
    local_weight: float = 0.1
    balance_weight: float = 1.5
    sharpening_power: float = 4.0

    # AVQ soft-to-hard curriculum
    start_soft_weight: float = 1.0
    final_soft_weight: float = 0.2
    soft_warmup_epochs: int = 15
    soft_transition_end_epoch: int = 40

    def total_steps(self, steps_per_epoch):
        return self.num_epochs * steps_per_epoch

    def anneal_steps(self, steps_per_epoch):
        return self.anneal_epochs * steps_per_epoch


@dataclass
class TrialConfig:
    num_epochs: int = 7

    # Loss weights
    usage_weight: float = 0.01
    adversarial_weight: float = 0.01
    adversarial_start_epoch: int = 2

    # AVQ temperature schedule
    temperature_start: float = 0.15
    temperature_end: float = 0.02
    anneal_epochs: int = 4

    # AVQ codebook regularization
    local_weight: float = 0.1
    balance_weight: float = 1.5
    sharpening_power: float = 4.0

    # AVQ soft-to-hard curriculum
    start_soft_weight: float = 1.0
    final_soft_weight: float = 0.2
    soft_warmup_epochs: int = 2
    soft_transition_end_epoch: int = 4

    def total_steps(self, steps_per_epoch):
        return self.num_epochs * steps_per_epoch

    def anneal_steps(self, steps_per_epoch):
        return self.anneal_epochs * steps_per_epoch

    