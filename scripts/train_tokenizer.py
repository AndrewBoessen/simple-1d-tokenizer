import math
import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from omegaconf import OmegaConf

from utils.logger import setup_logger
from utils.train_utils import (create_dataloader, create_evaluator,
                               create_lr_scheduler,
                               create_model_and_loss_module, create_optimizer,
                               get_config, save_checkpoint, train_one_epoch)


def main():
    config = get_config()

    # Enable TF32 on Ampere GPUs.
    if config.training.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    # Create directories for logging and chekpoints
    output_dir = config.experiment.output_dir
    os.makedirs(output_dir, exist_ok=True)
    config.experiment.logging_dir = os.path.join(output_dir, "logs")

    # Whether logging to Wandb or Tensorboard.
    tracker = "tensorboard"
    if config.training.enable_wandb:
        tracker = "wandb"

    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
        log_with=tracker,
        project_dir=config.experiment.logging_dir,
        split_batches=False,
    )

    logger = setup_logger(
        name="1d-tokenizer",
        log_level="INFO",
        output_file=f"{output_dir}/log{accelerator.process_index}.txt",
    )

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers(config.experiment.name)
        config_path = Path(output_dir) / "config.yaml"
        logger.info(f"Saving config to {config_path}")
        OmegaConf.save(config, config_path)
        logger.info(f"Config:\n{OmegaConf.to_yaml(config)}")

    # If passed along, set the training seed now.
    if config.training.seed is not None:
        set_seed(config.training.seed, device_specific=True)

    model, ema_model, loss_module = create_model_and_loss_module(
        config, logger, accelerator
    )

    optimizer, discriminator_optimizer = create_optimizer(
        config, logger, model, loss_module
    )

    lr_scheduler, discriminator_lr_scheduler = create_lr_scheduler(
        config, logger, accelerator, optimizer, discriminator_optimizer
    )

    # create dataloaders for training and evaluation
    train_dataloader, eval_dataloader = create_dataloader(config, logger, accelerator)

    # Set up evaluator.
    evaluator = create_evaluator(config, logger, accelerator)

    # prepare accelerator for training
    (
        model,
        loss_module,
        optimizer,
        discriminator_optimizer,
        lr_scheduler,
        discriminator_lr_scheduler,
    ) = accelerator.prepare(
        model,
        loss_module,
        optimizer,
        discriminator_optimizer,
        lr_scheduler,
        discriminator_lr_scheduler,
    )

    if config.training.use_ema:
        ema_model.to(accelerator.device)  # load EMA model to device

    total_batch_size_without_accum = (
        config.training.per_gpu_batch_size * accelerator.num_processes
    )
    num_batches = math.ceil(
        config.experiment.max_train_examples / total_batch_size_without_accum
    )
    num_update_steps_per_epoch = math.ceil(
        num_batches / config.training.gradient_accumulation_steps
    )
    num_train_epochs = math.ceil(
        config.training.max_train_steps / num_update_steps_per_epoch
    )

    # Start training.
    logger.info("***** Running training *****")
    logger.info(f"  Num training steps = {config.training.max_train_steps}")
    logger.info(
        f"  Gradient Accumulation steps = {config.training.gradient_accumulation_steps}"
    )
    logger.info(
        f"  Instantaneous batch size per gpu = {config.training.per_gpu_batch_size}"
    )
    logger.info(
        f"""  Total train batch size (w. parallel, distributed & accumulation) = {(
        config.training.per_gpu_batch_size *
        accelerator.num_processes *
        config.training.gradient_accumulation_steps)}"""
    )
    global_step = 0
    first_epoch = 0

    for current_epoch in range(first_epoch, num_train_epochs):
        accelerator.print(f"Epoch {current_epoch}/{num_train_epochs-1} started.")
        global_step = train_one_epoch(
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
        )
        # Stop training if max steps is reached.
        if global_step >= config.training.max_train_steps:
            accelerator.print(
                f"Finishing training: Global step is >= Max train steps: {global_step} >= {config.training.max_train_steps}"
            )
            break

    accelerator.wait_for_everyone()
    # Save checkpoint at the end of training.
    save_checkpoint(model, output_dir, accelerator, global_step, logger=logger)
    # Save the final trained checkpoint
    if accelerator.is_main_process:
        model = accelerator.unwrap_model(model)
        if config.training.use_ema:
            ema_model.copy_to(model.parameters())
        model.save_pretrained_weight(output_dir)
    accelerator.end_training()


if __name__ == "__main__":
    main()
