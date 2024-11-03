import json
import os
import time
from pathlib import Path
import pprint
import pprint
from collections import defaultdict

import torch
from omegaconf import OmegaConf
from torch.optim import AdamW
from torchinfo import summary

from models import ReconstructionLoss, Tokenizer, EMAModel
from .lr_scheduler import get_cosine_schedule_with_warmup
from .viz_utils import make_viz_from_samples, make_viz_from_samples_generation


def get_config():
    """Reads configs from a yaml file and terminal."""
    cli_conf = OmegaConf.from_cli()

    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = OmegaConf.merge(yaml_conf, cli_conf)

    return conf


class AverageMeter(object):
    """Computes and stores the average and current value.

    This class is borrowed from
    https://github.com/pytorch/examples/blob/main/imagenet/main.py#L423
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def create_model_and_loss_module(
    config,
    logger,
    accelerator
):
    """
    Create model and loss modules for given config

    :param config str: config file path
    :param logger logger: logger object
    :param accelerator accelerator: train accelerator
    """
    logger.info("Creating model and loss module")
    model = Tokenizer(config)
    loss = ReconstructionLoss(config)

    ema_model = EMAModel(
        model.parameters(),
        decay=0.999,
        model_cls=Tokenizer,
        config=config
    )
    # Create custom saving and loading hooks so that `accelerator.save_state(...)` serializes in a nice format.

    def load_model_hook(models, input_dir):
        load_model = EMAModel.from_pretrained(os.path.join(input_dir, "ema_model"),
                                              model_cls=Tokenizer, config=config)
        ema_model.load_state_dict(load_model.state_dict())
        ema_model.to(accelerator.device)
        del load_model

    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            ema_model.save_pretrained(
                os.path.join(output_dir, "ema_model"))

    accelerator.register_load_state_pre_hook(load_model_hook)
    accelerator.register_save_state_pre_hook(save_model_hook)

    # Print Model for sanity check.
    if accelerator.is_main_process:
        input_size = (1, 3, config.dataset.preprocessing.crop_size,
                      config.dataset.preprocessing.crop_size)
        model_summary_str = summary(model, input_size=input_size, depth=5,
                                    col_names=("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds"))
        logger.info(model_summary_str)

    return model, ema_model, loss


def create_optimizer(
    config,
    logger,
    model,
    loss_module
):
    """
    Create optimizers for training

    :param config str: config file path
    :param logger logger: train logger
    :param model class: model module
    :param loss_module class: loss module
    """
    logger.info("creating oprimizers")
    optim_config = config.optimizer.params  # config variables for optimizer
    lr = optim_config.learning_rate

    optim_cls = AdamW  # use Adam optimizer

    # Exclude terms we may not want to apply weight decay.
    exclude = (lambda n, p: p.ndim < 2 or "ln" in n or "bias" in n or 'latent_tokens' in n
               or 'mask_token' in n or 'embedding' in n or 'norm' in n or 'gamma' in n)

    def include(n, p): return not exclude(n, p)
    named_parameters = list(model.named_parameters())
    gain_or_bias_params = [
        p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
    rest_params = [p for n, p in named_parameters if include(
        n, p) and p.requires_grad]
    optimizer = optim_cls(
        [
            {"params": gain_or_bias_params, "weight_decay": 0.},
            {"params": rest_params, "weight_decay": optim_config.weight_decay},
        ],
        lr=lr,
        betas=(optim_config.beta1, optim_config.beta2)
    )

    # create discriminator optimizer
    discriminator_lr = optim_config.discriminator_learning_rate
    discriminator_params = list(loss_module.named_parameters())
    discriminator_gain_or_bias_params = [
        p for n, p in discriminator_params if exclude(n, p) and p.requires_grad]
    discriminator_rest_params = [
        p for n, p in discriminator_params if include(n, p) and p.requires_grad]

    discriminator_optimizer = optim_cls(
        [
            {"params": discriminator_gain_or_bias_params, "weight_decay": 0.},
            {"params": discriminator_rest_params,
             "weight_decay": optim_config.weight_decay},
        ],
        lr=discriminator_lr,
        betas=(optim_config.beta1, optim_config.beta2)
    )

    return optimizer, discriminator_optimizer


def create_lr_scheduler(config, logger, accelerator, optimizer, discriminator_optimizer=None):
    """
    Create lr scheduler for training

    :param config Dict: config object containing params
    :param logger Logger: instance of Logger
    :param accelerator Accelerator: instance of Accelerator
    :param optimizer nn.Module: model optimizer
    :param discriminator_optimizer nn.Module: discriminator optimizer
    """
    logger.info("Creating lr_schedulers.")

    # Create main model scheduler
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_training_steps=config.training.max_train_steps * accelerator.num_processes,
        num_warmup_steps=config.lr_scheduler.params.warmup_steps * accelerator.num_processes,
        base_lr=config.lr_scheduler.params.learning_rate,
        end_lr=config.lr_scheduler.params.end_lr,
    )

    # Create discriminator scheduler if needed
    if discriminator_optimizer is not None:
        discriminator_steps = (config.training.max_train_steps * accelerator.num_processes -
                               config.losses.discriminator_start)
        discriminator_lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=discriminator_optimizer,
            num_training_steps=discriminator_steps,
            num_warmup_steps=config.lr_scheduler.params.warmup_steps * accelerator.num_processes,
            base_lr=config.lr_scheduler.params.learning_rate,
            end_lr=config.lr_scheduler.params.end_lr,
        )
    else:
        discriminator_lr_scheduler = None

    return lr_scheduler, discriminator_lr_scheduler


def train_one_epoch(
    config,
    logger,
    accelerator,
    model,
    ema_model,
    loss_module,
    optimizer,
    discriminator_optimizer,
    lr_scheduler,
    discriminator_lr_scheduler,
    train_dataloader,
    eval_dataloader,
    evaluator,
    global_step,
):
    """
    Train autoencoder for one epoch

    :param config Dict: config object
    :param logger Logger: logger instance
    :param accelerator Accelerator: accelerator instance
    :param model nn.Module: autoencoder model
    :param ema_model nn.Module: EMA model
    :param loss_module nn.Module: loss module for autoencoder
    :param optimizer nn.Module: model optimizer
    :param discriminator_optimizer nn.Module: discriminator module
    :param lr_scheduler nn.Module: lr scheduler
    :param discriminator_lr_scheduler nn.Module: discriminator scheduler
    :param train_dataloader Dataloader: train data Dataloader
    :param eval_dataloader Dataloder: evaluation dataloader
    :param evaluator nn.Module: autoencoder evaluator
    :param global_step int: current training step
    :raises ValueError: ValueError
    """
    # Initialize timing meters
    batch_time_meter = AverageMeter()
    data_time_meter = AverageMeter()
    end = time.time()

    model.train()  # set model into train mode

    # Create logs for epoch
    autoencoder_logs = defaultdict(float)
    discriminator_logs = defaultdict(float)

    for i, batch in enumerate(train_dataloader):
        model.train()
        if "image" in batch:
            images = batch["image"].to(
                accelerator.device, memory_format=torch.contiguous_format, non_blocking=True
            )
        else:
            raise ValueError(f"Not found valid keys: {batch.keys()}")

        fnames = batch["__key__"]  # frame names
        data_time_meter.update(time.time() - end)  # update meter

        # forward pass through model
        with accelerator.accumulate([model, loss_module]):
            recon_images, results = model(images)
            autoencoder_loss, loss_dict = loss_module(
                images,
                recon_images,
                results,
                global_step,
                mode="generator"
            )
            # Gather the losses across all processes for logging.
            autoencoder_logs = {}
            for k, v in loss_dict.items():
                if k in ["discriminator_factor", "d_weight"]:
                    if type(v) == torch.Tensor:
                        autoencoder_logs["train/" + k] = v.cpu().item()
                    else:
                        autoencoder_logs["train/" + k] = v
                else:
                    autoencoder_logs["train/" +
                                     k] = accelerator.gather(v).mean().item()

            accelerator.backward(autoencoder_loss)

            if config.training.max_grad_norm is not None and accelerator.sync_gradients:
                accelerator.clip_grad_norm_(
                    model.parameters(), config.training.max_grad_norm)

            optimizer.step()
            lr_scheduler.step()

            # Log gradient norm before zeroing it.
            if (
                accelerator.sync_gradients
                and (global_step + 1) % config.experiment.log_grad_norm_every == 0
                and accelerator.is_main_process
            ):
                log_grad_norm(model, accelerator, global_step + 1)

            optimizer.zero_grad(set_to_none=True)

            # Train discriminator.
            discriminator_logs = defaultdict(float)
            if config.model.vq_model.finetune_decoder and accelerator.unwrap_model(loss_module).should_discriminator_be_trained(global_step):
                discriminator_logs = defaultdict(float)
                discriminator_loss, loss_dict_discriminator = loss_module(
                    images,
                    recon_images,
                    results,
                    global_step=global_step,
                    mode="discriminator",
                )

                # Gather the losses across all processes for logging.
                for k, v in loss_dict_discriminator.items():
                    if k in ["logits_real", "logits_fake"]:
                        if type(v) == torch.Tensor:
                            discriminator_logs["train/" + k] = v.cpu().item()
                        else:
                            discriminator_logs["train/" + k] = v
                    else:
                        discriminator_logs["train/" +
                                           k] = accelerator.gather(v).mean().item()

                accelerator.backward(discriminator_loss)

                if config.training.max_grad_norm is not None and accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        loss_module.parameters(), config.training.max_grad_norm)

                discriminator_optimizer.step()
                discriminator_lr_scheduler.step()

                # Log gradient norm before zeroing it.
                if (
                    accelerator.sync_gradients
                    and (global_step + 1) % config.experiment.log_grad_norm_every == 0
                    and accelerator.is_main_process
                ):
                    log_grad_norm(loss_module, accelerator, global_step + 1)

                discriminator_optimizer.zero_grad(set_to_none=True)

        if accelerator.sync_gradients:
            if config.training.use_ema:
                ema_model.step(model.parameters())
            batch_time_meter.update(time.time() - end)
            end = time.time()

            if (global_step + 1) % config.experiment.log_every == 0:
                samples_per_second_per_gpu = (
                    config.training.gradient_accumulation_steps *
                    config.training.per_gpu_batch_size / batch_time_meter.val
                )

                lr = lr_scheduler.get_last_lr()[0]
                logger.info(
                    f"Data (t): {data_time_meter.val:0.4f}, {
                        samples_per_second_per_gpu:0.2f}/s/gpu "
                    f"Batch (t): {batch_time_meter.val:0.4f} "
                    f"LR: {lr:0.6f} "
                    f"Step: {global_step + 1} "
                    f"Total Loss: {autoencoder_logs['train/total_loss']:0.4f} "
                    f"Recon Loss: {
                        autoencoder_logs['train/reconstruction_loss']:0.4f} "
                )
                logs = {
                    "lr": lr,
                    "lr/generator": lr,
                    "samples/sec/gpu": samples_per_second_per_gpu,
                    "time/data_time": data_time_meter.val,
                    "time/batch_time": batch_time_meter.val,
                }
                logs.update(autoencoder_logs)
                logs.update(discriminator_logs)
                accelerator.log(logs, step=global_step + 1)

                # Reset batch / data time meters per log window.
                batch_time_meter.reset()
                data_time_meter.reset()

            # Save model checkpoint.
            if (global_step + 1) % config.experiment.save_every == 0:
                save_path = save_checkpoint(
                    model, config.experiment.output_dir, accelerator, global_step + 1, logger=logger)
                # Wait for everyone to save their checkpoint.
                accelerator.wait_for_everyone()

            # Generate images.
            if (global_step + 1) % config.experiment.generate_every == 0 and accelerator.is_main_process:
                # Store the model parameters temporarily and load the EMA parameters to perform inference.
                if config.training.get("use_ema", False):
                    ema_model.store(model.parameters())
                    ema_model.copy_to(model.parameters())

                reconstruct_images(
                    model,
                    images[:config.training.num_generated_images],
                    fnames[:config.training.num_generated_images],
                    accelerator,
                    global_step + 1,
                    config.experiment.output_dir,
                    logger=logger,
                    config=config,
                )

                if config.training.get("use_ema", False):
                    # Switch back to the original model parameters for training.
                    ema_model.restore(model.parameters())

            # Evaluate reconstruction.
            if eval_dataloader is not None and (global_step + 1) % config.experiment.eval_every == 0:
                logger.info(f"Computing metrics on the validation set.")
                if config.training.get("use_ema", False):
                    ema_model.store(model.parameters())
                    ema_model.copy_to(model.parameters())
                    # Eval for EMA.
                    eval_scores = eval_reconstruction(
                        model,
                        eval_dataloader,
                        accelerator,
                        evaluator,
                    )
                    logger.info(
                        f"EMA EVALUATION "
                        f"Step: {global_step + 1} "
                    )
                    logger.info(pprint.pformat(eval_scores))
                    if accelerator.is_main_process:
                        eval_log = {f'ema_eval/'+k: v for k,
                                    v in eval_scores.items()}
                        accelerator.log(eval_log, step=global_step + 1)
                    if config.training.get("use_ema", False):
                        # Switch back to the original model parameters for training.
                        ema_model.restore(model.parameters())
                else:
                    # Eval for non-EMA.
                    eval_scores = eval_reconstruction(
                        model,
                        eval_dataloader,
                        accelerator,
                        evaluator,
                    )

                    logger.info(
                        f"Non-EMA EVALUATION "
                        f"Step: {global_step + 1} "
                    )
                    logger.info(pprint.pformat(eval_scores))
                    if accelerator.is_main_process:
                        eval_log = {f'eval/'+k: v for k,
                                    v in eval_scores.items()}
                        accelerator.log(eval_log, step=global_step + 1)

                accelerator.wait_for_everyone()

            global_step += 1

            if global_step >= config.training.max_train_steps:
                accelerator.print(
                    f"Finishing training: Global step is >= Max train steps: {
                        global_step} >= {config.training.max_train_steps}"
                )
                break

    return global_step


@torch.no_grad()
def eval_reconstruction(
    model,
    eval_loader,
    accelerator,
    evaluator,
):
    model.eval()
    evaluator.reset_metrics()
    local_model = accelerator.unwrap_model(model)

    for batch in eval_loader:
        images = batch["image"].to(
            accelerator.device, memory_format=torch.contiguous_format, non_blocking=True
        )
        original_images = torch.clone(images)
        reconstructed_images, model_dict = local_model(images)
        reconstructed_images = torch.clamp(reconstructed_images, 0.0, 1.0)
        # Quantize to uint8
        reconstructed_images = torch.round(
            reconstructed_images * 255.0) / 255.0
        original_images = torch.clamp(original_images, 0.0, 1.0)
        # For VQ model.
        evaluator.update(original_images, reconstructed_images.squeeze(
            2), model_dict["min_encoding_indices"])
    model.train()
    return evaluator.result()


@torch.no_grad()
def reconstruct_images(
    model,
    original_images,
    fnames,
    accelerator,
    global_step,
    output_dir,
    logger,
    config=None,
):
    logger.info("Reconstructing images...")
    original_images = torch.clone(original_images)
    model.eval()
    dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        dtype = torch.bfloat16

    with torch.autocast("cuda", dtype=dtype, enabled=accelerator.mixed_precision != "no"):
        enc_tokens, encoder_dict = accelerator.unwrap_model(
            model).encode(original_images)
    reconstructed_images = accelerator.unwrap_model(model).decode(enc_tokens)
    images_for_saving, images_for_logging = make_viz_from_samples(
        original_images,
        reconstructed_images
    )
    # Log images.
    if config.training.enable_wandb:
        accelerator.get_tracker("wandb").log_images(
            {f"Train Reconstruction": images_for_saving},
            step=global_step
        )
    else:
        accelerator.get_tracker("tensorboard").log_images(
            {"Train Reconstruction": images_for_logging}, step=global_step
        )
    # Log locally.
    root = Path(output_dir) / "train_images"
    os.makedirs(root, exist_ok=True)
    for i, img in enumerate(images_for_saving):
        filename = f"{global_step:08}_s-{i:03}-{fnames[i]}.png"
        path = os.path.join(root, filename)
        img.save(path)

    model.train()
    return


def save_checkpoint(model, output_dir, accelerator, global_step, logger) -> Path:
    save_path = Path(output_dir) / f"checkpoint-{global_step}"

    state_dict = accelerator.get_state_dict(model)
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained_weight(
            save_path / "unwrapped_model",
            save_function=accelerator.save,
            state_dict=state_dict,
        )
        json.dump({"global_step": global_step},
                  (save_path / "metadata.json").open("w+"))
        logger.info(f"Saved state to {save_path}")

    accelerator.save_state(save_path)
    return save_path


def load_checkpoint(checkpoint_path: Path, accelerator, logger, strict=True):
    logger.info(f"Load checkpoint from {checkpoint_path}")

    accelerator.load_state(checkpoint_path, strict=strict)

    with open(checkpoint_path / "metadata.json", "r") as f:
        global_step = int(json.load(f)["global_step"])

    logger.info(f"Resuming at global_step {global_step}")
    return global_step


def log_grad_norm(model, accelerator, global_step):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads = param.grad.detach().data
            grad_norm = (grads.norm(p=2) / grads.numel()).item()
            accelerator.log({"grad_norm/" + name: grad_norm}, step=global_step)
