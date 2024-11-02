import json
import os
import time
import math
from pathlib import Path
import pprint
import glob
from collections import defaultdict

from data import SimpleImageDataset
import torch
from omegaconf import OmegaConf
from torch.optim import AdamW
from torchinfo import summary

from models import ReconstructionLoss, Tokenizer, EMAModel


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
