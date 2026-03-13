import torch
import math
from torch.optim.lr_scheduler import LambdaLR
from transformers import get_constant_schedule_with_warmup, get_linear_schedule_with_warmup, \
    get_cosine_schedule_with_warmup


def get_scheduler(
        scheduler_type: str,
        optimizer: torch.optim.Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        lr_end: float = 0.0,  # Minimum learning rate, default is 0, i.e., anneal to 0
)->LambdaLR:
    """
    Create a learning rate scheduler that supports cosine annealing after linear warmup.

    Args:
        scheduler_type (str): Scheduler type, e.g., "linear_warmup_cosine_decay".
        optimizer (torch.optim.Optimizer): Optimizer instance.
        num_warmup_steps (int): Number of steps for linear warmup.
        num_training_steps (int): Total number of training steps.
        lr_end (float): Minimum value to which the learning rate anneals, default is 0.0.

    Returns:
        torch.optim.lr_scheduler._LRScheduler: Learning rate scheduler.
    """
    if num_training_steps <= 0:
        raise ValueError("num_training_steps must be a positive number.")
    if num_warmup_steps < 0 or num_warmup_steps >= num_training_steps:
        raise ValueError("num_warmup_steps must be a non-negative number and less than num_training_steps.")
    if not (0.0 <= lr_end < optimizer.param_groups[0]['lr']):
        # lr_end should be less than initial learning rate and non-negative
        raise ValueError(f"lr_end ({lr_end}) must be a non-negative number and less than the optimizer's initial learning rate ({optimizer.param_groups[0]['lr']}).")

    lr_peak = optimizer.param_groups[0]['lr']  # Get the optimizer's initial learning rate as the peak learning rate

    if scheduler_type == "linear_warmup_cosine_decay":
        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                # Linear warmup phase: increase from 0 to lr_peak
                return float(current_step) / float(max(1, num_warmup_steps))

            # Cosine annealing phase: decrease from lr_peak to lr_end
            # Calculate progress in the annealing phase [0, 1]
            progress_after_warmup = float(current_step - num_warmup_steps)
            total_decay_steps = float(num_training_steps - num_warmup_steps)

            if total_decay_steps == 0:
                # If there's no annealing phase (e.g., num_warmup_steps equals num_training_steps)
                # Keep learning rate at peak value
                return 1.0

            # Ensure progress doesn't exceed 1, in case current_step slightly exceeds num_training_steps
            cosine_progress = min(1.0, progress_after_warmup / total_decay_steps)

            # Cosine function value varies between [1, 0]
            cosine_val = 0.5 * (1.0 + math.cos(math.pi * cosine_progress))

            # Map cosine_val to a factor in [lr_end/lr_peak, 1]
            # This way, when cosine_val is 1 (start of annealing), factor is 1, LR is lr_peak
            # When cosine_val is 0 (end of annealing), factor is lr_end/lr_peak, LR is lr_end
            scaled_lr_factor = (lr_end / lr_peak) + (1 - (lr_end / lr_peak)) * cosine_val
            return scaled_lr_factor

        return LambdaLR(optimizer, lr_lambda, last_epoch=-1)  # last_epoch=-1 defaults to counting from 0

    elif scheduler_type == "cosine_with_warmup":
        # Hugging Face's built-in cosine annealing scheduler, defaults to annealing to 0
        return get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
        )
    elif scheduler_type == "linear_with_warmup":
        # Hugging Face's built-in linear scheduler, defaults to annealing to 0
        return get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
        )
    # You can add other scheduler types as needed
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

