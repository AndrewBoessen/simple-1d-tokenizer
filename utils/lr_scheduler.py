import math
import torch


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    base_lr: float = 1e-4,
    end_lr: float = 0.0,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Creates a cosine learning rate schedule with warmup.

    The learning rate starts at 0, warms up to base_lr, then follows a cosine decay 
    to end_lr over the remaining steps.

    Args:
        optimizer: The optimizer to schedule the learning rate for
        num_warmup_steps: Number of steps for the warmup phase
        num_training_steps: Total number of training steps
        base_lr: Peak learning rate after warmup
        end_lr: Final learning rate at the end of training
        num_cycles: Number of cosine cycles over the decay period
        last_epoch: The index of last epoch (-1 to start from beginning)

    Returns:
        A LambdaLR scheduler instance
    """
    if num_warmup_steps >= num_training_steps:
        raise ValueError(
            "num_warmup_steps must be less than num_training_steps")

    def lr_lambda(current_step: int) -> float:
        # Linear warmup
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        # Cosine decay
        progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        scale = max(0.0, 0.5 * (1.0 + math.cos(math.pi *
                    float(num_cycles) * 2.0 * progress)))
        return (end_lr + (base_lr - end_lr) * scale) / base_lr

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)
