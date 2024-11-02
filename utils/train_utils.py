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
