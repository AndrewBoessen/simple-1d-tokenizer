import copy
from typing import Any, Iterable, Optional, Union

import torch
from torch import nn


class EMAModel:
    """
    Exponential Moving Average of models weights
    """

    def __init__(
        self,
        parameters: Iterable[nn.Parameter],
        decay: float = 0.9999,
        min_decay: float = 0.0,
        update_after_step: int = 0,
        update_every: int = 1,
        current_step: int = 0,
        use_ema_warmup: bool = False,
        inv_gamma: Union[float, int] = 1.0,
        power: Union[float, int] = 2 / 3,
        model_cls: Optional[Any] = None,
        **model_config_kwargs
    ):
        """Initialize EMA Model with configurable decay and warmup parameters."""
        self.shadow_params = [p.clone().detach() for p in parameters]
        self.temp_stored_params = None

        self.decay = decay
        self.min_decay = min_decay
        self.update_after_step = update_after_step
        self.update_every = update_every
        self.use_ema_warmup = use_ema_warmup
        self.inv_gamma = inv_gamma
        self.power = power
        self.optimization_step = current_step
        self.cur_decay_value = None

        self.model_cls = model_cls
        self.model_config_kwargs = model_config_kwargs

    @classmethod
    def from_pretrained(cls, checkpoint, model_cls, **model_config_kwargs):
        """
        Create an EMA model from pretrained checkpoint

        :param cls class: class
        :param checkpoint str: checkpoint path
        :param model_cls str: model class
        """
        model = model_cls(**model_config_kwargs)
        model.load_pretrained_weight(checkpoint)
        return cls(model.parameters(), model_cls=model_cls, **model_config_kwargs)

    def save_pretrained(self, path):
        """
        Save model weight to path

        :param path str: path to save to
        :raises ValueError: error
        """
        if not self.model_cls or not self.model_config_kwargs:
            raise ValueError("Model class and config must be defined to save.")

        model = self.model_cls(**self.model_config_kwargs)
        self.copy_to(model.parameters())
        model.save_pretrained_weight(path)

    def set_step(self, optimization_step: int):
        """
        Set current optimization step

        :param optimization_step: id of step
        """
        self.optimization_step = optimization_step

    def get_decay(self, optimization_step: int) -> float:
        """
        Compute decay factor for current step

        :param optimization_step: current step
        :return: decay
        """
        step = max(0, optimization_step - self.update_after_step - 1)

        if step <= 0:
            return 0.0

        cur_decay_value = (
            1 - (1 + step / self.inv_gamma) ** -self.power
            if self.use_ema_warmup
            else (1 + step) / (10 + step)
        )

        cur_decay_value = max(min(cur_decay_value, self.decay), self.min_decay)
        return cur_decay_value

    @torch.no_grad()
    def step(self, parameters: Iterable[nn.Parameter]):
        """Perform EMA update on the parameters."""
        parameters = list(parameters)
        self.optimization_step += 1

        if (self.optimization_step - 1) % self.update_every != 0:
            return

        decay = self.get_decay(self.optimization_step)
        self.cur_decay_value = decay
        one_minus_decay = 1 - decay

        for s_param, param in zip(self.shadow_params, parameters):
            s_param.sub_(one_minus_decay * (s_param - param)
                         ) if param.requires_grad else s_param.copy_(param)

    def copy_to(self, parameters: Iterable[nn.Parameter]) -> None:
        """Copy averaged parameters to the given collection."""
        for s_param, param in zip(self.shadow_params, parameters):
            param.data.copy_(s_param.to(param.device).data)

    def to(self, device=None, dtype=None) -> None:
        """Move internal buffers to specified device and dtype."""
        self.shadow_params = [
            p.to(device=device, dtype=dtype) if p.is_floating_point(
            ) else p.to(device=device)
            for p in self.shadow_params
        ]

    def state_dict(self) -> dict:
        """Return the state of the EMA model for checkpointing."""
        return {
            "decay": self.decay,
            "min_decay": self.min_decay,
            "optimization_step": self.optimization_step,
            "update_after_step": self.update_after_step,
            "use_ema_warmup": self.use_ema_warmup,
            "inv_gamma": self.inv_gamma,
            "power": self.power,
            "shadow_params": self.shadow_params,
        }

    def store(self, parameters: Iterable[nn.Parameter]) -> None:
        """Temporarily store current parameters."""
        self.temp_stored_params = [param.detach().cpu().clone()
                                   for param in parameters]

    def restore(self, parameters: Iterable[nn.Parameter]) -> None:
        """Restore previously stored parameters."""
        if self.temp_stored_params is None:
            raise RuntimeError("No stored weights to restore")

        for c_param, param in zip(self.temp_stored_params, parameters):
            param.data.copy_(c_param.data)
        self.temp_stored_params = None

    def load_state_dict(self, state_dict: dict) -> None:
        """Load the EMA model state from a state dictionary."""
        state_dict = copy.deepcopy(state_dict)

        # Validation for each attribute with explicit error messages
        validation_rules = {
            "decay": (float, 0.0, 1.0, "Decay must be between 0 and 1"),
            "min_decay": (float, None, None, "Min decay must be a float"),
            "optimization_step": (int, None, None, "Optimization step must be an integer"),
            "update_after_step": (int, None, None, "Update after step must be an integer"),
            "use_ema_warmup": (bool, None, None, "Use EMA warmup must be a boolean"),
            "inv_gamma": ((float, int), None, None, "Inverse gamma must be a number"),
            "power": ((float, int), None, None, "Power must be a number"),
        }

        for attr, (expected_type, min_val, max_val, error_msg) in validation_rules.items():
            value = state_dict.get(attr, getattr(self, attr))

            if not isinstance(value, expected_type):
                raise ValueError(error_msg)

            if min_val is not None and value < min_val:
                raise ValueError(error_msg)

            if max_val is not None and value > max_val:
                raise ValueError(error_msg)

            setattr(self, attr, value)

        shadow_params = state_dict.get("shadow_params")
        if shadow_params is not None:
            if not isinstance(shadow_params, list) or not all(isinstance(p, torch.Tensor) for p in shadow_params):
                raise ValueError("Shadow params must be a list of tensors")
            self.shadow_params = shadow_params
