import json
import time
import logging
import argparse

import torch
import torch.nn.functional as F

from pathlib import Path
from functools import partial
from collections import defaultdict

from avq.losses import (
    discriminator_hinge_loss, 
    generator_adversarial_loss,
    codebook_regularization_loss,
)

from avq.metrics import (
    compute_fsq_metrics, 
    evaluate_lpips_and_rfid,
    evaluate_avq_diagnostics,
    get_codebook_norm_metrics, 
    compute_code_usage_metrics,
)

from avq.evaluate import (
    evaluate_validation, 
    evaluate_arvq_hard_mse,
    evaluate_arvq_validation,
    evaluate_rvqema_validation,
    evaluate_decoder_calibration_mse,
    evaluate_residual_diveq_validation,
    evaluate_residual_stgs_validation,
)

from avq.helpers import (
    is_log_step,
    setup_logging,
    _extract_images,
    set_global_seed,
    set_requires_grad,
    log_training_step,
    update_running_metrics,
    log_arvq_training_step,
    prepare_model_for_calibration,
    get_temperature_and_soft_weight,
    get_hard_latents_for_calibration,
    change_model_modes_for_calibration,
    gumbel_softmax_temperature_schedule,
)

from avq.dataset import (
    build_tiny_imagenet_loaders, 
    build_mini_places_loaders,
    build_coco_loaders,
)

from avq.config import EXPERIMENT_REGISTRY, TrainingConfig, TrialConfig
from avq.losses import compute_quantizer_loss, compute_arvq_codebook_loss


# Logging setup
logger = logging.getLogger(__name__)


def train_decoder_calibration(
    model, 
    train_loader, 
    val_loader,
    method,
    device,
    experiment_dir,
    hard_finetune_epochs=10, 
    temperature=0.02, 
    patience=3,
    min_delta=1e-4,
    learning_rate=1e-4,
    log_interval=500,
):
    """
    Phase-II AVQ decoder calibration.

    Encoder and quantizer are frozen. The decoder is trained using pure hard-code 
    latents and reconstruction MSE.

    Checkpoint selection and early stopping are based on validation hard-reconstruction 
    MSE.
    """
    # Setup checkpoint directory and paths
    checkpoint_dir = Path(experiment_dir) / "calibration"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = checkpoint_dir / "best_decoder_calibration.pt"

    last_checkpoint_path = None
    prepare_model_for_calibration(model)

    decoder_optimizer = torch.optim.Adam(
        model.decoder.parameters(),lr=learning_rate, betas=(0.5, 0.9)
    )

    # Evaluate Phase-I checkpoint before calibration
    baseline_val_mse = evaluate_decoder_calibration_mse(
        model=model,
        val_loader=val_loader,
        method=method,
        device=device,
        temperature=temperature,
    )

    logger.info("Decoder calibration baseline | Val Hard MSE %.6f", baseline_val_mse)

    best_epoch = 0
    global_step = 0
    epochs_without_improvement = 0
    best_val_mse = baseline_val_mse

    # Save uncalibrated model as initial best
    torch.save(
        {
            "phase": "decoder_calibration",
            "epoch": 0,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": decoder_optimizer.state_dict(),
            "val_hard_mse": baseline_val_mse,
            "best_val_hard_mse": baseline_val_mse,
            "best_epoch": 0,
        },
        best_checkpoint_path,
    )

    history = []
    for epoch in range(1, hard_finetune_epochs + 1):

        epoch_start_time = time.time()
        change_model_modes_for_calibration(model)

        running_elements = 0
        running_squared_error = 0.0
        for batch_idx, batch in enumerate(train_loader):

            images = _extract_images(batch).to(device, non_blocking=True)

            decoder_optimizer.zero_grad(set_to_none=True)

            hard_quantized = get_hard_latents_for_calibration(
                model=model,
                images=images,
                method=method,
                temperature=temperature,
            )

            reconstruction_hard = model.decoder(hard_quantized)
            reconstruction_loss = F.mse_loss(reconstruction_hard, images)

            reconstruction_loss.backward()
            decoder_optimizer.step()

            running_elements += images.numel()
            running_squared_error += reconstruction_loss.item() * images.numel()

            if batch_idx % log_interval == 0 or batch_idx == len(train_loader) - 1:
                logger.info(
                    "Calibration Epoch %02d/%02d | "
                    "Step %05d | "
                    "Train Hard MSE %.6f",
                    epoch,
                    hard_finetune_epochs,
                    global_step,
                    reconstruction_loss.item(),
                )

            global_step += 1

        train_hard_mse = running_squared_error / running_elements

        # Validation
        val_hard_mse = evaluate_decoder_calibration_mse(
            model=model,
            val_loader=val_loader,
            method=method,
            device=device,
            temperature=temperature,
        )

        epoch_seconds = time.time() - epoch_start_time

        history.append(
            {
                "epoch": epoch,
                "global_step": global_step,
                "train_hard_mse": train_hard_mse,
                "val_hard_mse": val_hard_mse,
                "epoch_seconds": epoch_seconds,
            }
        )

        logger.info(
            "Calibration Epoch %02d/%02d finished | "
            "Train Hard MSE %.6f | "
            "Val Hard MSE %.6f | "
            "Best %.6f | "
            "Time %.1fs",
            epoch,
            hard_finetune_epochs,
            train_hard_mse,
            val_hard_mse,
            best_val_mse,
            epoch_seconds,
        )

        # Early stopping
        improvement = best_val_mse - val_hard_mse
        if improvement > min_delta:

            best_epoch = epoch
            best_val_mse = val_hard_mse
            epochs_without_improvement = 0

            torch.save(
                {
                    "phase": "decoder_calibration",
                    "epoch": epoch,
                    "global_step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": decoder_optimizer.state_dict(),
                    "train_hard_mse": train_hard_mse,
                    "val_hard_mse": val_hard_mse,
                    "best_val_hard_mse": best_val_mse,
                    "best_epoch": best_epoch,
                    "history": history,
                },
                best_checkpoint_path,
            )

            logger.info(
                "New best calibration checkpoint | "
                "Epoch %02d | "
                "Val Hard MSE %.6f | "
                "Improvement %.6f",
                epoch,
                val_hard_mse,
                improvement,
            )

        else:
            epochs_without_improvement += 1

            logger.info(
                "No significant improvement | "
                "Patience %d/%d",
                epochs_without_improvement,
                patience,
            )

        # Save epoch checkpoint
        last_checkpoint_path = checkpoint_dir / f"epoch_{epoch:02d}_decoder_calibration.pt"

        torch.save(
            {
                "phase": "decoder_calibration",
                "epoch": epoch,
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": decoder_optimizer.state_dict(),
                "train_hard_mse": train_hard_mse,
                "val_hard_mse": val_hard_mse,
                "best_val_hard_mse": best_val_mse,
                "best_epoch": best_epoch,
                "epochs_without_improvement": epochs_without_improvement,
                "history": history,
            },
            last_checkpoint_path,
        )

        # Early stopping
        if epochs_without_improvement >= patience:
            logger.info(
                "Early stopping decoder calibration | "
                "Best epoch %02d | "
                "Best Val Hard MSE %.6f",
                best_epoch,
                best_val_mse,
            )
            break


    # Restore best decoder calirated checkpoint
    checkpoint = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    logger.info(
        "Decoder calibration complete | "
        "Baseline Val Hard MSE %.6f | "
        "Best Val Hard MSE %.6f | "
        "Best epoch %02d",
        baseline_val_mse,
        best_val_mse,
        best_epoch,
    )

    return {
        "model": model,
        "history": history,
        "baseline_val_hard_mse": baseline_val_mse,
        "best_val_hard_mse": best_val_mse,
        "best_epoch": best_epoch,
        "best_checkpoint_path": str(best_checkpoint_path),
        "last_checkpoint_path": str(last_checkpoint_path) if last_checkpoint_path is not None else None,
    }


def train_one_epoch(
    model, 
    discriminator,
    train_loader, 
    g_optimizer,
    d_optimizer,
    epoch,
    global_step,
    method,
    config,
    device,
    use_adversarial,
    log_interval=500,
):
    """
    Train one Phase-I epoch.
    """
    model.train()
    discriminator.train()

    is_avq = method == "avq"
    is_stgs = method in {"stgs", "rstgs"}

    steps_per_epoch = len(train_loader)
    anneal_steps = config.anneal_steps(steps_per_epoch) if is_avq else None

    num_samples = 0
    running_metrics = defaultdict(float)
    for batch_idx, batch in enumerate(train_loader):

        images = _extract_images(batch).to(device, non_blocking=True)
        batch_size = images.size(0)

        # Method specific schedule
        if is_avq:
            temperature, soft_weight = get_temperature_and_soft_weight(
                global_step=global_step,
                anneal_steps=anneal_steps,
                config=config,
                steps_per_epoch=steps_per_epoch,
            )

            # Plateau at minimum temperature after annealing
            if epoch >= config.anneal_epochs:
                temperature = config.temperature_end
        elif is_stgs:
            temperature = gumbel_softmax_temperature_schedule(
                epoch=epoch,
                num_epochs=config.num_epochs,
                start_temperature=config.stgs_temperature_start,
                min_temperature=config.stgs_temperature_end,
            )
            soft_weight = None
        else:
            temperature = None
            soft_weight = None

        # Forward pass
        if is_avq:
            outputs = model(
                images,
                temperature=temperature,
                soft_weight=soft_weight,
                return_separate_reconstructions=False,
            )

            reconstruction = outputs["reconstruction_train"]

        elif is_stgs:
            outputs = model(images, temperature=temperature)
            reconstruction = outputs["reconstruction"]

        else:
            outputs = model(images)
            reconstruction = outputs["reconstruction"]

        reconstruction_loss = F.mse_loss(reconstruction, images)

        # 1. Discriminator update
        if use_adversarial:
            set_requires_grad(discriminator, True)
            d_optimizer.zero_grad(set_to_none=True)

            real_logits = discriminator(images)

            # Detach reconstruction so D training does not update 
            # the encoder, quantizer, or decoder
            d_fake_logits = discriminator(reconstruction.detach())

            discriminator_loss = discriminator_hinge_loss(
                real_logits, d_fake_logits
            )

            discriminator_loss.backward()
            d_optimizer.step()
        else:
            discriminator_loss = images.new_zeros(())

        # 2. Encoder / quantizer / decoder update
        set_requires_grad(discriminator, False)
        g_optimizer.zero_grad(set_to_none=True)

        if method in {"diveq", "rdiveq"}:
            quantizer_loss = images.new_zeros(())
            codebook_metrics = {}
        else:
            quantizer_loss, codebook_metrics = compute_quantizer_loss(method, outputs, config)

        if use_adversarial:
            # No detach here as adversarial gradients must flow from D
            # through reconstructionn to the autoencoder
            g_fake_logits = discriminator(reconstruction)
            adversarial_loss = generator_adversarial_loss(g_fake_logits)
        else:
            adversarial_loss = images.new_zeros(())

        generator_loss = (
            reconstruction_loss 
            + quantizer_loss 
            + config.adversarial_weight * adversarial_loss
        )

        generator_loss.backward()
        g_optimizer.step()

        set_requires_grad(discriminator, True)

        # Accumulate metrics
        batch_metrics = {
            "reconstruction_loss": reconstruction_loss,
            "quantizer_loss": quantizer_loss,
            "adversarial_loss": adversarial_loss,
            "discriminator_loss": discriminator_loss,
            "generator_loss": generator_loss,
        }

        fsq_metrics = {}
        if is_avq:
            avq_metrics = {
                "normalized_local_entropy": codebook_metrics["normalized_local_entropy"],
                "balance_kl": codebook_metrics["balance_kl"],
            }
            batch_metrics.update(avq_metrics)
        elif method == "vq":
            vq_metrics = {
                "codebook_loss": outputs["codebook_loss"],
                "commitment_loss": outputs["commitment_loss"],
            }
            batch_metrics.update(vq_metrics)
        elif method in {"vqema", "rvqema"}:
            batch_metrics.update({"commitment_loss": outputs["commitment_loss"]})
        elif method == "fsq":
            fsq_metrics = compute_fsq_metrics(
                indices=outputs["indices"],
                scalar_indices=outputs["scalar_indices"],
                levels=model.quantizer.levels,
            )
        elif method == "diveq":
            batch_metrics.update({"perplexity_batch": outputs["perplexity_batch"]})
        elif method == "rdiveq":
            for stage_idx, perplexity in enumerate(outputs["perplexity_batch"]):
                batch_metrics[f"stage_{stage_idx}_perplexity_batch"] = perplexity
        elif method == "stgs":
            usage_metrics = compute_code_usage_metrics(
                indices=outputs["indices"], num_codes=model.quantizer.num_embeddings,
            )
            batch_metrics.update({
                "kl_loss": outputs["kl_loss"],
                "perplexity_batch": usage_metrics["perplexity"],
                "active_codes_batch": usage_metrics["active_codes"],
                "utilization_batch": usage_metrics["utilization"],
            })
        elif method == "rstgs":
            batch_metrics["kl_loss"] = outputs["kl_loss"]
            for stage_idx, indices in enumerate(outputs["indices_levels"]):
                usage_metrics = compute_code_usage_metrics(
                    indices=indices, num_codes=model.quantizer.num_embeddings,
                )

                batch_metrics.update({
                    f"stage_{stage_idx}_perplexity_batch": usage_metrics["perplexity"],
                    f"stage_{stage_idx}_active_codes_batch": usage_metrics["active_codes"],
                    f"stage_{stage_idx}_utilization_batch": usage_metrics["utilization"],
                })

        num_samples += batch_size

        update_running_metrics(
            running_metrics=running_metrics, batch_metrics=batch_metrics, batch_size=batch_size,
        )

        if is_log_step(batch_idx, log_interval, steps_per_epoch):
            codebook_norm_metrics = get_codebook_norm_metrics(model, method)
            log_training_step(
                method=method,
                epoch=epoch,
                global_step=global_step,
                batch_metrics=batch_metrics,
                outputs=outputs,
                codebook_metrics=codebook_metrics,
                codebook_norm_metrics=codebook_norm_metrics,
                fsq_metrics=fsq_metrics,
                temperature=temperature,
                soft_weight=soft_weight,
                config=config,
            )

        global_step += 1

    train_metrics = {
        name: total / num_samples for name, total in running_metrics.items()
    }

    if is_avq:
        train_metrics.update({
            "temperature": temperature,
            "soft_weight": soft_weight,
            "hard_weight": 1.0 - soft_weight,
        })
    elif is_stgs:
        train_metrics.update({"temperature": temperature})

    return train_metrics, global_step


def train(
    model,
    discriminator,
    train_loader,
    val_loader,
    g_optimizer,
    d_optimizer,
    method,
    config,
    device,
    experiment_dir,
    log_interval=500,
):
    """
    Train various autoencoders
    """
    checkpoint_dir = Path(experiment_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_checkpoint_name = "best_plateau.pt" if method == "avq" else "best.pt"
    best_checkpoint_path = checkpoint_dir / best_checkpoint_name

    global_step = 0
    best_epoch = -1
    best_val_mse = float("inf")

    history = []
    for epoch in range(config.num_epochs):

        epoch_start_time = time.time()

        # GAN starts at configured epoch
        use_adversarial = epoch >= config.adversarial_start_epoch

        # Train one epoch
        train_metrics, global_step = train_one_epoch(
            model=model,
            discriminator=discriminator,
            train_loader=train_loader,
            g_optimizer=g_optimizer,
            d_optimizer=d_optimizer,
            epoch=epoch,
            global_step=global_step,
            method=method,
            config=config,
            device=device,
            use_adversarial=use_adversarial,
            log_interval=log_interval,
        )

        # Validate
        if method == "rvqema":
            val_metrics = evaluate_rvqema_validation(
                model=model,
                val_loader=val_loader,
                device=device,
            )
        elif method == "rdiveq":
            val_metrics = evaluate_residual_diveq_validation(
                model=model,
                val_loader=val_loader,
                device=device,
            )
        elif method == "rstgs":
            val_metrics = evaluate_residual_stgs_validation(
                model=model,
                val_loader=val_loader,
                device=device,
            )
        else:
            if method == "stgs":
                method_temperature = config.stgs_temperature_end
            else:
                method_temperature = config.temperature_end
            val_metrics = evaluate_validation(
                model=model,
                val_loader=val_loader,
                method=method,
                device=device,
                temperature=method_temperature,
            )

        val_hard_mse = val_metrics["hard_mse"]

        epoch_seconds = time.time() - epoch_start_time

        epoch_metrics = {
            "epoch": epoch,
            "global_step": global_step,
            **train_metrics,
            **val_metrics,
            "epoch_seconds": epoch_seconds,
        }

        history.append(epoch_metrics)

        logger.info(
            "Epoch %02d/%02d | "
            "Train MSE %.6f | "
            "Val Hard MSE %.6f | "
            "Time %.1fs",
            epoch + 1,
            config.num_epochs,
            train_metrics["reconstruction_loss"],
            val_hard_mse,
            epoch_seconds,
        )

        # Check whether this epoch is eligible to become best
        if method == "avq":
            eligible_for_best = epoch >= config.anneal_epochs
        else:
            eligible_for_best = True

        is_new_best = eligible_for_best and val_hard_mse < best_val_mse
        if is_new_best:
            best_val_mse = val_hard_mse
            best_epoch = epoch

            logger.info(
                "New best checkpoint | "
                "Epoch %02d | "
                "Val Hard MSE %.6f",
                epoch + 1,
                best_val_mse,
            )

        # Build checkpoint
        checkpoint = {
            "epoch": epoch,
            "global_step": global_step,
            "method": method,
            "model_state_dict": model.state_dict(),
            "discriminator_state_dict": discriminator.state_dict(),
            "g_optimizer_state_dict": g_optimizer.state_dict(),
            "d_optimizer_state_dict": d_optimizer.state_dict(),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "best_epoch": best_epoch,
            "best_val_hard_mse": best_val_mse,
            "history": history,
        }

        epoch_checkpoint_path = checkpoint_dir / f"epoch_{epoch + 1:02d}.pt"
        torch.save(checkpoint, epoch_checkpoint_path)

        # Save the best eligible checkpoint
        if is_new_best:
            torch.save(checkpoint, best_checkpoint_path)

    if best_epoch < 0:
        raise RuntimeError("No eligible best checkpoint was selected.")

    # Restore best Phase-I model before returning
    best_checkpoint = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    model.eval()

    logger.info(
        "Phase-I training complete | "
        "Best epoch %02d | "
        "Best Val Hard MSE %.6f",
        best_epoch + 1,
        best_val_mse,
    )

    return {
        "model": model,
        "history": history,
        "global_step": global_step,
        "best_epoch": best_epoch,
        "best_val_hard_mse": best_val_mse,
        "best_checkpoint_path": str(best_checkpoint_path),
    }


def train_arvq_decoder_calibration(
    model, 
    train_loader, 
    val_loader,
    device,
    experiment_dir,
    hard_finetune_epochs=10, 
    temperature=0.02, 
    patience=3,
    min_delta=1e-4,
    learning_rate=1e-4,
    log_interval=500,
):
    """
    Phase-II ARVQ decoder calibration.

    Encoder and quantizer are frozen. The decoder is trained using pure hard-code 
    latents and reconstruction MSE.

    Checkpoint selection and early stopping are based on validation hard-reconstruction 
    MSE.
    """
    # Setup checkpoint directory and paths
    checkpoint_dir = Path(experiment_dir) / "calibration"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = checkpoint_dir / "best_arvq_decoder_calibration.pt"

    last_checkpoint_path = None
    prepare_model_for_calibration(model)

    decoder_optimizer = torch.optim.Adam(
        model.decoder.parameters(),lr=learning_rate, betas=(0.5, 0.9)
    )

    # Evaluate Phase-I checkpoint before calibration
    baseline_val_mse = evaluate_arvq_hard_mse(
        model=model,
        val_loader=val_loader,
        device=device,
        temperature=temperature,
    )
        
    logger.info("Decoder calibration baseline | Val Hard MSE %.6f", baseline_val_mse)

    best_epoch = 0
    global_step = 0
    epochs_without_improvement = 0
    best_val_mse = baseline_val_mse

    # Save uncalibrated model as initial best
    torch.save(
        {
            "phase": "arvq_decoder_calibration",
            "epoch": 0,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": decoder_optimizer.state_dict(),
            "val_hard_mse": baseline_val_mse,
            "best_val_hard_mse": baseline_val_mse,
            "best_epoch": 0,
        },
        best_checkpoint_path,
    )
        
    history = []
    for epoch in range(1, hard_finetune_epochs + 1):

        epoch_start_time = time.time()
        change_model_modes_for_calibration(model)

        running_elements = 0
        running_squared_error = 0.0
        for batch_idx, batch in enumerate(train_loader):

            images = _extract_images(batch).to(device, non_blocking=True)

            decoder_optimizer.zero_grad(set_to_none=True)

            outputs = model(
                images,
                temperature=temperature,
                mode="hard",
            )

            reconstruction = outputs["reconstruction"]
            reconstruction_loss = F.mse_loss(reconstruction, images)

            reconstruction_loss.backward()
            decoder_optimizer.step()
        
            running_elements += images.numel()
            running_squared_error += reconstruction_loss.item() * images.numel()

            if batch_idx % log_interval == 0 or batch_idx == len(train_loader) - 1:
                logger.info(
                    "Calibration Epoch %02d/%02d | "
                    "Step %05d | "
                    "Train Hard MSE %.6f",
                    epoch,
                    hard_finetune_epochs,
                    global_step,
                    reconstruction_loss.item(),
                )

            global_step += 1

        train_hard_mse = running_squared_error / running_elements
        
        # Validation
        val_hard_mse = evaluate_arvq_hard_mse(
            model=model,
            val_loader=val_loader,
            device=device,
            temperature=temperature,
        )

        epoch_seconds = time.time() - epoch_start_time

        history.append(
            {
                "epoch": epoch,
                "global_step": global_step,
                "train_hard_mse": train_hard_mse,
                "val_hard_mse": val_hard_mse,
                "epoch_seconds": epoch_seconds,
            }
        )
        
        logger.info(
            "Calibration Epoch %02d/%02d finished | "
            "Train Hard MSE %.6f | "
            "Val Hard MSE %.6f | "
            "Best %.6f | "
            "Time %.1fs",
            epoch,
            hard_finetune_epochs,
            train_hard_mse,
            val_hard_mse,
            best_val_mse,
            epoch_seconds,
        )
        
        # Early stopping
        improvement = best_val_mse - val_hard_mse
        if improvement > min_delta:

            best_epoch = epoch
            best_val_mse = val_hard_mse
            epochs_without_improvement = 0
        
            torch.save(
                {
                    "phase": "arvq_decoder_calibration",
                    "epoch": epoch,
                    "global_step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": decoder_optimizer.state_dict(),
                    "train_hard_mse": train_hard_mse,
                    "val_hard_mse": val_hard_mse,
                    "best_val_hard_mse": best_val_mse,
                    "best_epoch": best_epoch,
                    "history": history,
                },
                best_checkpoint_path,
            )
        
            logger.info(
                "New best calibration checkpoint | "
                "Epoch %02d | "
                "Val Hard MSE %.6f | "
                "Improvement %.6f",
                epoch,
                val_hard_mse,
                improvement,
            )

        else:
            epochs_without_improvement += 1

            logger.info(
                "No significant improvement | "
                "Patience %d/%d",
                epochs_without_improvement,
                patience,
            )
        
        # Save epoch checkpoint
        last_checkpoint_path = checkpoint_dir / f"epoch_{epoch:02d}_decoder_calibration.pt"

        torch.save(
            {
                "phase": "arvq_decoder_calibration",
                "epoch": epoch,
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": decoder_optimizer.state_dict(),
                "train_hard_mse": train_hard_mse,
                "val_hard_mse": val_hard_mse,
                "best_val_hard_mse": best_val_mse,
                "best_epoch": best_epoch,
                "epochs_without_improvement": epochs_without_improvement,
                "history": history,
            },
            last_checkpoint_path,
        )
        
        # Early stopping
        if epochs_without_improvement >= patience:
            logger.info(
                "Early stopping decoder calibration | "
                "Best epoch %02d | "
                "Best Val Hard MSE %.6f",
                best_epoch,
                best_val_mse,
            )
            break
        
        
    # Restore best decoder calirated checkpoint
    checkpoint = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    logger.info(
        "Decoder calibration complete | "
        "Baseline Val Hard MSE %.6f | "
        "Best Val Hard MSE %.6f | "
        "Best epoch %02d",
        baseline_val_mse,
        best_val_mse,
        best_epoch,
    )
        
    return {
        "model": model,
        "history": history,
        "baseline_val_hard_mse": baseline_val_mse,
        "best_val_hard_mse": best_val_mse,
        "best_epoch": best_epoch,
        "best_checkpoint_path": str(best_checkpoint_path),
        "last_checkpoint_path": str(last_checkpoint_path) if last_checkpoint_path is not None else None,
    }


def train_arvq_one_epoch(
    model,
    discriminator,
    train_loader,
    g_optimizer,
    d_optimizer,
    epoch,
    global_step,
    config,
    device,
    use_adversarial,
    log_interval=500,
):
    """
    Train ARVQ autoencoder for 1 epoch
    """
    model.train()
    discriminator.train()

    arvq_codebook_loss_fn = partial(
        codebook_regularization_loss,
        local_weight=config.local_weight,
        balance_weight=config.balance_weight,
        sharpening_power=config.sharpening_power,
    )

    steps_per_epoch = len(train_loader)
    anneal_steps = config.anneal_steps(steps_per_epoch)

    num_samples = 0
    running_metrics = defaultdict(float)
    for batch_idx, batch in enumerate(train_loader):

        images = _extract_images(batch).to(device, non_blocking=True)
        batch_size = images.size(0)

        temperature, soft_weight = get_temperature_and_soft_weight(
            global_step=global_step,
            anneal_steps=anneal_steps,
            config=config,
            steps_per_epoch=steps_per_epoch,
        )

        # Plateau at minimum temperature after annealing
        if epoch >= config.anneal_epochs:
            temperature = config.temperature_end
        

        # Forward pass
        outputs = model(
            images,
            temperature=temperature,
            soft_weight=soft_weight,
            mode="mixed",
        )

        reconstruction = outputs["reconstruction"]
        reconstruction_loss = F.mse_loss(reconstruction, images)

        # 1. Discriminator update
        if use_adversarial:
            set_requires_grad(discriminator, True)
            d_optimizer.zero_grad(set_to_none=True)

            real_logits = discriminator(images)

            # Detach reconstruction so D training does not update 
            # the encoder, quantizer, or decoder
            d_fake_logits = discriminator(reconstruction.detach())

            discriminator_loss = discriminator_hinge_loss(
                real_logits, d_fake_logits
            )

            discriminator_loss.backward()
            d_optimizer.step()
        else:
            discriminator_loss = images.new_zeros(())

        # 2. Encoder / quantizer / decoder update
        set_requires_grad(discriminator, False)
        g_optimizer.zero_grad(set_to_none=True)

        arvq_codebook_loss, stage_metrics = compute_arvq_codebook_loss(
            all_probs=outputs["all_probs"],
            codebook_loss_fn=arvq_codebook_loss_fn,
            stage_weights=None, # Equal stage weights for paper
        )

        quantizer_loss = config.usage_weight * arvq_codebook_loss

        if use_adversarial:
            # No detach here as adversarial gradients must flow from D
            # through reconstructionn to the autoencoder
            g_fake_logits = discriminator(reconstruction)
            adversarial_loss = generator_adversarial_loss(g_fake_logits)
        else:
            adversarial_loss = images.new_zeros(())

        generator_loss = (
            reconstruction_loss 
            + quantizer_loss 
            + config.adversarial_weight * adversarial_loss
        )

        generator_loss.backward()
        g_optimizer.step()

        set_requires_grad(discriminator, True)

        # Accumulate metrics
        batch_metrics = {
            "reconstruction_loss": reconstruction_loss,
            "quantizer_loss": quantizer_loss,
            "arvq_codebook_loss": arvq_codebook_loss,
            "adversarial_loss": adversarial_loss,
            "discriminator_loss": discriminator_loss,
            "generator_loss": generator_loss,
        }

        num_samples += batch_size

        update_running_metrics(
            running_metrics=running_metrics, batch_metrics=batch_metrics, batch_size=batch_size,
        )

        if is_log_step(batch_idx, log_interval, steps_per_epoch):
            log_arvq_training_step(
                model=model,
                epoch=epoch,
                global_step=global_step,
                batch_metrics=batch_metrics,
                stage_metrics=stage_metrics,
                temperature=temperature,
                soft_weight=soft_weight,
            )

        global_step += 1

    train_metrics = {
        name: total / num_samples for name, total in running_metrics.items()
    }

    train_metrics.update({
        "temperature": temperature,
        "soft_weight": soft_weight,
        "hard_weight": 1.0 - soft_weight,
    })

    return train_metrics, global_step


def train_arvq(
    model,
    discriminator,
    train_loader,
    val_loader,
    g_optimizer,
    d_optimizer,
    config,
    device,
    experiment_dir,
    log_interval=500,
):
    """
    Train ARVQ autoencoder
    """
    checkpoint_dir = Path(experiment_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_checkpoint_name = "best_plateau.pt"
    best_checkpoint_path = checkpoint_dir / best_checkpoint_name

    global_step = 0
    best_epoch = -1
    best_val_mse = float("inf")

    history = []
    for epoch in range(config.num_epochs):

        epoch_start_time = time.time()

        # GAN starts at configured epoch
        use_adversarial = epoch >= config.adversarial_start_epoch

        # Train one epoch
        train_metrics, global_step = train_arvq_one_epoch(
            model=model,
            discriminator=discriminator,
            train_loader=train_loader,
            g_optimizer=g_optimizer,
            d_optimizer=d_optimizer,
            epoch=epoch,
            global_step=global_step,
            config=config,
            device=device,
            use_adversarial=use_adversarial,
            log_interval=log_interval,
        )

        # Validate
        val_metrics = evaluate_arvq_validation(
            model=model,
            val_loader=val_loader,
            device=device,
            temperature=config.temperature_end,
        )

        val_hard_mse = val_metrics["hard_mse"]

        epoch_seconds = time.time() - epoch_start_time

        epoch_metrics = {
            "epoch": epoch,
            "global_step": global_step,
            **train_metrics,
            **val_metrics,
            "epoch_seconds": epoch_seconds,
        }

        history.append(epoch_metrics)

        logger.info(
            "Epoch %02d/%02d | "
            "Train MSE %.6f | "
            "Val Hard MSE %.6f | "
            "Time %.1fs",
            epoch + 1,
            config.num_epochs,
            train_metrics["reconstruction_loss"],
            val_hard_mse,
            epoch_seconds,
        )

        # Check whether this epoch is eligible to become best
        eligible_for_best = epoch >= config.anneal_epochs
        is_new_best = eligible_for_best and val_hard_mse < best_val_mse
        if is_new_best:
            best_val_mse = val_hard_mse
            best_epoch = epoch

            logger.info(
                "New best checkpoint | "
                "Epoch %02d | "
                "Val Hard MSE %.6f",
                epoch + 1,
                best_val_mse,
            )

        # Build checkpoint
        checkpoint = {
            "epoch": epoch,
            "global_step": global_step,
            "method": "arvq",
            "model_state_dict": model.state_dict(),
            "discriminator_state_dict": discriminator.state_dict(),
            "g_optimizer_state_dict": g_optimizer.state_dict(),
            "d_optimizer_state_dict": d_optimizer.state_dict(),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "best_epoch": best_epoch,
            "best_val_hard_mse": best_val_mse,
            "history": history,
        }

        epoch_checkpoint_path = checkpoint_dir / f"epoch_{epoch + 1:02d}.pt"
        torch.save(checkpoint, epoch_checkpoint_path)

        # Save the best eligible checkpoint
        if is_new_best:
            torch.save(checkpoint, best_checkpoint_path)

    if best_epoch < 0:
        raise RuntimeError("No eligible best checkpoint was selected.")

    # Restore best Phase-I model before returning
    best_checkpoint = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    model.eval()

    logger.info(
        "Phase-I training complete | "
        "Best epoch %02d | "
        "Best Val Hard MSE %.6f",
        best_epoch + 1,
        best_val_mse,
    )

    return {
        "model": model,
        "history": history,
        "global_step": global_step,
        "best_epoch": best_epoch,
        "best_val_hard_mse": best_val_mse,
        "best_checkpoint_path": str(best_checkpoint_path),
    }


def build_loaders_for_experiment(experiment_id, data_seed):
    """
    Build dataset loaders for the selected experiment.
    """
    TINY_EXPERIMENT_IDS = set(range(0, 9)) | {27, 30, 31, 37, 38, 43, 44}
    MINIPLACES_EXPERIMENT_IDS = set(range(9, 18)) | {28, 32, 33, 36, 41, 42, 47, 48}
    COCO_EXPERIMENT_IDS = set(range(18, 27)) | {29, 34, 35, 39, 40, 45, 46}

    if experiment_id in TINY_EXPERIMENT_IDS:
        # Tiny ImageNet
        root_directory = "/tmp/data/tiny-imagenet-200"

        return build_tiny_imagenet_loaders(
            root_directory=root_directory,
            batch_size=64,
            num_workers=4,
            image_size=64,
            seed=data_seed,
            drop_last_train=True,
        )

    if experiment_id in MINIPLACES_EXPERIMENT_IDS:
        # MiniPlaces
        root_directory = "/tmp/data/miniplaces"

        return build_mini_places_loaders(
            root_directory=root_directory,
            batch_size=16,
            num_workers=4,
            image_size=128,
            seed=data_seed,
            drop_last_train=True,
        )

    if experiment_id in COCO_EXPERIMENT_IDS:
        # COCO 2017
        root_directory = "/tmp/data/coco2017"

        return build_coco_loaders(
            root_directory=root_directory,
            batch_size=64,
            num_workers=4,
            image_size=64,
            seed=data_seed,
            drop_last_train=True,
        )

    raise ValueError(
        f"Dataset loader not configured for experiment ID {experiment_id}"
    )


EXPERIMENTS = {
    # Tiny ImageNet Experiments
    0: "avq_16x16_k1024_d64",
    1: "avq_16x16_k512_d64",
    2: "avq_16x16_k1024_d128",
    3: "avq_8x8_k1024_d64",
    4: "vq_16x16_k1024_d64",
    5: "vqema_16x16_k1024_d64",
    6: "fsq_16x16_k1000_d64",
    7: "arvq_16x16_q3_k1024_d64",
    8: "avq_16x16_k1024_d64_no_norm", 
    27: "rvqema_16x16_q3_k1024_d64",
    30: "avq_16x16_k1024_d64_scaled",
    31: "arvq_16x16_q3_k1024_d64_scaled",
    37: "diveq_16x16_k1024_d64",
    38: "rdiveq_16x16_q3_k1024_d64",
    43: "stgs_16x16_k1024_d64",
    44: "rstgs_16x16_q3_k1024_d64",

    # Mini Places Experiments
    9: "avq_32x32_k1024_d64",
    10: "vq_32x32_k1024_d64",
    11: "vqema_32x32_k1024_d64",
    12: "fsq_32x32_k1000_d64",
    13: "arvq_32x32_q3_k1024_d64",
    14: "avq_128sq_16x16_k1024_d64",
    15: "arvq_128sq_16x16_q3_k1024_d64",
    16: "avq_32x32_k512_d64",
    17: "avq_32x32_k1024_d64_no_norm",
    28: "rvqema_32x32_q3_k1024_d64",
    32: "avq_32x32_k1024_d64_scaled",
    33: "arvq_32x32_q3_k1024_d64_scaled",
    36: "avq_32x32_k1024_d128",
    41: "diveq_32x32_k1024_d64",
    42: "rdiveq_32x32_q3_k1024_d64",
    47: "stgs_32x32_k1024_d64",
    48: "rstgs_32x32_q3_k1024_d64",

    # COCO Experiments
    18: "avq_coco_16x16_k1024_d64", 
    19: "avq_coco_16x16_k512_d64",
    20: "avq_coco_16x16_k1024_d128",
    21: "avq_coco_8x8_k1024_d64",
    22: "vq_coco_16x16_k1024_d64", 
    23: "vqema_coco_16x16_k1024_d64", 
    24: "fsq_coco_16x16_k1000_d64", 
    25: "arvq_coco_16x16_q3_k1024_d64", 
    26: "avq_coco_16x16_k1024_d64_no_norm",
    29: "rvqema_coco_16x16_q3_k1024_d64", 
    34: "avq_coco_16x16_k1024_d64_scaled", 
    35: "arvq_coco_16x16_q3_k1024_d64_scaled",
    39: "diveq_coco_16x16_k1024_d64",
    40: "rdiveq_coco_16x16_q3_k1024_d64",
    45: "stgs_coco_16x16_k1024_d64",
    46: "rstgs_coco_16x16_q3_k1024_d64",
}


def main():

    # 0. Parse CLI arguments
    parser = argparse.ArgumentParser(description="Train Autoencoders")
    parser.add_argument(
        "--experiment_dir", 
        type=str, 
        required=True,
        help="Directory to save model artifacts"
    )
    parser.add_argument(
        "--experiment_id",
        type=int,
        default=0,
        choices=EXPERIMENTS.keys(),
        help="Experiment ID from EXPERIMENTS registry",
    )

    parser.add_argument(
        "--model_seed",
        type=int,
        default=42,
        help="Random seed for model initialization and training randomness",
    )

    parser.add_argument(
        "--local_weight",
        type=float,
        default=None,
        help="Override AVQ/ARVQ local entropy weight"
    )

    parser.add_argument(
        "--balance_weight",
        type=float,
        default=None,
        help="Override AVQ/ARVQ global balance weight"
    )

    args = parser.parse_args()

    # 1. Reproducibility
    DATA_SEED = 15
    model_seed = args.model_seed
    set_global_seed(model_seed)

    # 2. Select experiment
    experiment_name = EXPERIMENTS[args.experiment_id]
    experiment = EXPERIMENT_REGISTRY[experiment_name]

    # 3. Setup experiment directory and configure logging
    experiment_dir = Path(args.experiment_dir) / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(experiment_dir / "train.log")

    # 4. Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Starting training on device: {device}")

    # 5. Initialize config and build data loaders
    config = TrainingConfig()

    if args.local_weight is not None:
        config.local_weight = args.local_weight

    if args.balance_weight is not None:
        config.balance_weight = args.balance_weight

    logging.info(
        "Loss weights | usage %.4f | local %.4f | balance %.4f",
        config.usage_weight,
        config.local_weight,
        config.balance_weight,
    )

    train_loader, train_eval_loader, val_loader, _, _ = build_loaders_for_experiment(
        experiment_id=args.experiment_id,
        data_seed=DATA_SEED,
    )

    # 6. Initialize model
    model = experiment.model_cls(**experiment.model_kwargs).to(device)
    discriminator = experiment.discriminator_cls().to(device)
    method = experiment.method

    logger.info("Experiment: %s | Model: %s", experiment_name, type(model).__name__)

    # 7. Initialize optimizers
    g_optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.5, 0.9))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=4e-4, betas=(0.5, 0.9))

    # 8. Launch the training
    if method == "arvq":
        train_result = train_arvq(
            model=model,
            discriminator=discriminator,
            train_loader=train_loader,
            val_loader=train_eval_loader,
            g_optimizer=g_optimizer,
            d_optimizer=d_optimizer,
            config=config,
            device=device,
            experiment_dir=experiment_dir,
        )
    else:
        train_result = train(
            model=model,
            discriminator=discriminator,
            train_loader=train_loader,
            val_loader=train_eval_loader, 
            g_optimizer=g_optimizer,
            d_optimizer=d_optimizer,
            method=method,
            config=config,
            device=device,
            experiment_dir=experiment_dir,
        )

    trained_model = train_result["model"]

    # AVQ quantizer diagnostics
    avq_diagnostics = {}
    if method == "avq":
        avq_diagnostics = evaluate_avq_diagnostics(
        model=trained_model,
        loader=val_loader,
        device=device,
        temperature=config.temperature_end,
        soft_weight=config.final_soft_weight,
    )

    # Phase-II: Decoder calibration
    if method in {"stgs", "rstgs"}:
        method_temperature = config.stgs_temperature_end
    else:
        method_temperature = config.temperature_end
    
    calibration_result = None
    if method in {
        "avq", "vq", "vqema", "rvqema", "fsq", "diveq", "rdiveq", "stgs", "rstgs",
    }:
        logger.info("Starting Phase-II %s decoder calibration", method.upper())

        calibration_result = train_decoder_calibration(
            model=trained_model,
            train_loader=train_loader,
            val_loader=train_eval_loader,
            method=method,
            device=device,
            experiment_dir=experiment_dir,
            hard_finetune_epochs=10,
            temperature=method_temperature,
            patience=3,
            learning_rate=1e-4,
        )

        trained_model = calibration_result["model"]

    elif method == "arvq":
        logger.info("Starting Phase-II ARVQ decoder calibration")
        
        calibration_result = train_arvq_decoder_calibration(
            model=trained_model,
            train_loader=train_loader,
            val_loader=train_eval_loader,
            device=device,
            experiment_dir=experiment_dir,
            hard_finetune_epochs=10,
            temperature=config.temperature_end,
            patience=3,
            learning_rate=1e-4,
        )

        trained_model = calibration_result["model"]


    # 9. Final validation
    if method == "arvq":
        val_metrics = evaluate_arvq_validation(
            model=trained_model, 
            val_loader=val_loader,
            device=device, 
            temperature=config.temperature_end,
        )
    elif method == "rvqema":
        val_metrics = evaluate_rvqema_validation(
            model=trained_model, 
            val_loader=val_loader,
            device=device, 
        )
    elif method == "rdiveq":
        val_metrics = evaluate_residual_diveq_validation(
            model=trained_model, 
            val_loader=val_loader,
            device=device,
        )
    elif method == "rstgs":
        val_metrics = evaluate_residual_stgs_validation(
            model=trained_model, 
            val_loader=val_loader,
            device=device,
        )
    else:
        val_metrics = evaluate_validation(
            model=trained_model,
            val_loader=val_loader,
            method=method,
            device=device,
            temperature=method_temperature,
        )

    # 10. Final perceptual metrics
    perceptual_metrics = evaluate_lpips_and_rfid(
        model=trained_model,
        loader=val_loader,
        method=method,
        device=device,
        temperature=method_temperature,
    )

    code_scale_metrics = {}
    if method == "avq":
        quantizer = trained_model.quantizer
        if getattr(quantizer, "learned_code_scale", False):
            scale = (F.softplus(quantizer.raw_code_scale) + 1e-6).detach().cpu().item()
            code_scale_metrics["learned_code_scale"] = scale
    elif method == "arvq":
        quantizer = trained_model.quantizer
        if getattr(quantizer, "learned_code_scale", False):
            for stage_idx, layer in enumerate(quantizer.layers):
                scale = (F.softplus(layer.raw_code_scale) + 1e-6).detach().cpu().item()
                code_scale_metrics[f"stage_{stage_idx}_code_scale"] = scale

    result = {
        "experiment": experiment_name,
        "method": method,
        "model_seed": model_seed,
        **val_metrics,
        **avq_diagnostics,
        **perceptual_metrics,
        **code_scale_metrics,
        "phase1_history": train_result["history"],
    }

    if method in {"avq", "arvq"}:
        result["phase1_best_epoch"] = train_result["best_epoch"] + 1
        result["phase1_best_val_hard_mse"] = train_result["best_val_hard_mse"]
    else:
        result["best_epoch"] = train_result["best_epoch"] + 1
        result["best_val_hard_mse"] = train_result["best_val_hard_mse"]
    
    if calibration_result is not None:
        result.update({
            "calibration_best_epoch": calibration_result["best_epoch"],
            "calibration_baseline_hard_mse": calibration_result["baseline_val_hard_mse"],
            "calibration_best_hard_mse": calibration_result["best_val_hard_mse"],
        })

    if args.local_weight is not None:
        result["local_weight"] = config.local_weight

    if args.balance_weight is not None:
        result["balance_weight"] = config.balance_weight

    result_file = experiment_dir / "result.json"
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)

    logger.info("Experiment '%s' completed successfully.", experiment_name)
    

if __name__ == "__main__":

    main()

