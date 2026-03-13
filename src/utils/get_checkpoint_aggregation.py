import json
import torch
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm


def aggregate_checkpoints_swa(output_dir: str,
                              model: torch.nn.Module,
                              start_epoch: int = None,
                              end_epoch: int = None,
                              epoch_freq: int = 1,
                              device: torch.device = None) -> Dict[str, torch.Tensor]:
    """
    Aggregate checkpoints using Stochastic Weight Averaging (SWA).
    This simply averages the weights from all checkpoints within the specified epoch range.

    Args:
        output_dir: Path to the output directory containing checkpoint folders.
        model: The model to instantiate for loading.
        start_epoch: Optional starting epoch (inclusive) to include in aggregation.
        end_epoch: Optional ending epoch (inclusive) to include in aggregation.
        device: Optional device to load weights onto.

    Returns:
        Averaged state_dict.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint_dirs = [d for d in Path(output_dir).iterdir() if d.is_dir() and d.name.startswith('checkpoint-epoch-')]
    if not checkpoint_dirs:
        raise ValueError(f"No checkpoint directories found in {output_dir}")

    # Extract epoch from directory name and filter based on start/end
    filtered_checkpoints = []
    for d in checkpoint_dirs:
        try:
            epoch = int(d.name.split('-')[-1])
            if start_epoch <= epoch <=end_epoch and (epoch - start_epoch) % int(epoch_freq) == 0:
                filtered_checkpoints.append(d)
        except ValueError:
            print(f"Skipping invalid checkpoint name: {d.name}")

    if not filtered_checkpoints:
        raise ValueError(f"No checkpoints found within the specified epoch range: start={start_epoch}, end={end_epoch}")

    print(
        f"Found {len(filtered_checkpoints)} checkpoints for SWA aggregation within epoch range {start_epoch}-{end_epoch}.")

    # Sort filtered checkpoints by epoch
    filtered_checkpoints = sorted(filtered_checkpoints, key=lambda x: int(x.name.split('-')[-1]))

    # Instantiate a model to get the structure
    model.to(device)

    # Initialize averaged state_dict with zeros
    avg_state_dict = {k: torch.zeros_like(v, dtype=v.dtype, device=device) for k, v in model.state_dict().items()}

    num_checkpoints = len(filtered_checkpoints)
    for ckpt_dir in tqdm(filtered_checkpoints, desc="loading checkpoints"):
        model_path = ckpt_dir / "pytorch_model.bin"
        if not model_path.exists():
            print(f"Skipping {ckpt_dir}: pytorch_model.bin not found")
            continue
        state_dict = torch.load(model_path, map_location=device)
        for k, v in state_dict.items():
            # Ensure tensor is on correct device
            v = v.to(device)
            # Accumulate with dtype preservation
            avg_state_dict[k] += (v / num_checkpoints).to(avg_state_dict[k].dtype)
    print("SWA aggregation completed.")
    return avg_state_dict


def get_checkpoint_metric(checkpoint_dir: Path) -> float:
    """Extract eval_top1_accuracy from trainer_state.json for a checkpoint."""
    trainer_state_path = checkpoint_dir / "trainer_state.json"
    if not trainer_state_path.exists():
        print(f"Warning: trainer_state.json not found in {checkpoint_dir}")
        return 0.0

    try:
        with open(trainer_state_path, 'r') as f:
            trainer_state = json.load(f)
        log_history = trainer_state.get('log_history', [])
        log_entry = log_history[-1]
        metric = log_entry.get('eval_top1_accuracy', 0.0)
        return metric

    except (json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"Warning: Error reading trainer_state.json in {checkpoint_dir}: {e}")
        return 0.0


def aggregate_checkpoints_swa_cv(output_dirs: List[str],
                                 model: torch.nn.Module,
                                 device: torch.device = None,
                                 top_k_per_fold: int = None) -> Dict[str, torch.Tensor]:
    """
    Aggregate checkpoints using Stochastic Weight Averaging (SWA) for cross-validation.
    This function averages the weights from all checkpoints across different folds,
    with optional selection of top-k checkpoints per fold based on eval_top1_accuracy.

    Args:
        output_dirs: List of paths to the output directories (one for each fold) containing checkpoint folders.
        model: The model to instantiate for loading.
        device: Optional device to load weights onto.
        top_k_per_fold: Optional number of top checkpoints to select per fold based on eval_top1_accuracy.
                       If None, uses all available checkpoints.

    Returns:
        Averaged state_dict.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    selected_checkpoint_dirs = []

    for fold_idx, output_dir in enumerate(output_dirs):
        # Get all checkpoint directories for this fold
        checkpoint_dirs = [d for d in Path(output_dir).iterdir()
                           if d.is_dir() and d.name.startswith('checkpoint-')]

        if not checkpoint_dirs:
            print(f"Warning: No checkpoint directories found in fold {fold_idx} ({output_dir})")
            continue

        # Get metrics for each checkpoint and create (metric, checkpoint_dir) pairs
        checkpoint_metrics = []
        for ckpt_dir in checkpoint_dirs:
            metric = get_checkpoint_metric(ckpt_dir)
            checkpoint_metrics.append((metric, ckpt_dir))

        # Sort by metric in descending order (highest accuracy first)
        checkpoint_metrics.sort(key=lambda x: x[0], reverse=True)

        # Select top-k checkpoints if specified
        if top_k_per_fold is not None:
            checkpoint_metrics = checkpoint_metrics[:top_k_per_fold]

        # Extract checkpoint directories and add to selected list
        fold_selected = [ckpt_dir for _, ckpt_dir in checkpoint_metrics]
        selected_checkpoint_dirs.extend(fold_selected)

        print(f"Fold {fold_idx}: Selected {len(fold_selected)} checkpoints "
              f"(top metrics: {[f'{metric:.4f}' for metric, _ in checkpoint_metrics[:5]]})"
              f"(bottom metrics: {[f'{metric:.4f}' for metric, _ in checkpoint_metrics[-5:]]})")

    if not selected_checkpoint_dirs:
        raise ValueError(f"No valid checkpoint directories found in the provided output directories")

    print(f"Total selected {len(selected_checkpoint_dirs)} checkpoints for SWA aggregation "
          f"across {len(output_dirs)} folds.")

    # Instantiate a model to get the structure
    model.to(device)

    # Initialize averaged state_dict with zeros
    avg_state_dict = {k: torch.zeros_like(v, dtype=v.dtype, device=device)
                      for k, v in model.state_dict().items()}

    num_checkpoints = len(selected_checkpoint_dirs)
    successful_loads = 0

    for ckpt_dir in selected_checkpoint_dirs:
        model_path = ckpt_dir / "pytorch_model.bin"
        if not model_path.exists():
            print(f"Skipping {ckpt_dir}: pytorch_model.bin not found")
            continue

        try:
            state_dict = torch.load(model_path, map_location=device)
            for k, v in state_dict.items():
                # Ensure tensor is on correct device
                v = v.to(device)
                # Accumulate with dtype preservation
                avg_state_dict[k] += (v / num_checkpoints).to(avg_state_dict[k].dtype)
            successful_loads += 1
        except Exception as e:
            print(f"Error loading {model_path}: {e}")
            continue

    print(f"SWA aggregation completed. Successfully loaded {successful_loads}/{num_checkpoints} checkpoints.")
    return avg_state_dict


def aggregate_checkpoints_ema(output_dir: str, model, decay: float = 0.999, start_epoch: int = None,
                              end_epoch: int = None, device: torch.device = None) -> Dict[str, torch.Tensor]:
    """
    Aggregate checkpoints using Exponential Moving Average (EMA).
    Checkpoints are processed in chronological order (sorted by epoch) within the specified range.

    Args:
        output_dir: Path to the output directory containing checkpoint folders.
        model: The model to instantiate for loading.
        decay: EMA decay factor (closer to 1 means slower adaptation).
        start_epoch: Optional starting epoch (inclusive) to include in aggregation.
        end_epoch: Optional ending epoch (inclusive) to include in aggregation.
        device: Optional device to load weights onto.

    Returns:
        EMA-averaged state_dict.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint_dirs = [d for d in Path(output_dir).iterdir() if d.is_dir() and d.name.startswith('checkpoint-epoch-')]
    if not checkpoint_dirs:
        raise ValueError(f"No checkpoint directories found in {output_dir}")

    # Extract epoch from directory name and filter based on start/end
    filtered_checkpoints = []
    for d in checkpoint_dirs:
        try:
            epoch = int(d.name.split('-')[-1])
            if (start_epoch is None or epoch >= start_epoch) and (end_epoch is None or epoch <= end_epoch):
                filtered_checkpoints.append(d)
        except ValueError:
            print(f"Skipping invalid checkpoint name: {d.name}")

    if not filtered_checkpoints:
        raise ValueError(f"No checkpoints found within the specified epoch range: start={start_epoch}, end={end_epoch}")

    # Sort filtered checkpoints by epoch
    filtered_checkpoints = sorted(filtered_checkpoints, key=lambda x: int(x.name.split('-')[-1]))

    print(
        f"Found {len(filtered_checkpoints)} checkpoints for EMA aggregation within epoch range {start_epoch}-{end_epoch}.")

    # Instantiate a model and load the first checkpoint as starting point
    model.to(device)

    first_ckpt_path = filtered_checkpoints[0] / "pytorch_model.bin"
    if not first_ckpt_path.exists():
        raise FileNotFoundError(f"First checkpoint weight file not found: {first_ckpt_path}")

    ema_state_dict = torch.load(first_ckpt_path, map_location=device)

    for ckpt_dir in filtered_checkpoints[1:]:
        model_path = ckpt_dir / "pytorch_model.bin"
        if not model_path.exists():
            print(f"Skipping {ckpt_dir}: pytorch_model.bin not found")
            continue

        state_dict = torch.load(model_path, map_location=device)
        for k in ema_state_dict.keys():
            ema_state_dict[k] = decay * ema_state_dict[k] + (1 - decay) * state_dict[k]

    print("EMA aggregation completed.")
    return ema_state_dict