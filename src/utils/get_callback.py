import os
import torch
from torch import nn
from torch.optim.swa_utils import AveragedModel, update_bn, get_ema_avg_fn, get_swa_avg_fn
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from copy import deepcopy


class SWACallback(TrainerCallback):
    def __init__(self,
                 trainer,
                 swa_start: int = 190,
                 swa_freq: int = 1,
                 save_start_epoch: int = 50,
                 save_end_epoch: int = 500,
                 save_freq: int = 10):
        """
        Initialize the SWA callback.

        :param trainer: The trainer object passed in, used to access the model and training state.
        :param swa_start: From which epoch to start applying SWA.
        :param swa_freq: How many epochs to wait before updating SWA weights again.
        """
        super().__init__()
        self._trainer = trainer  # Store trainer object
        self.swa_start = swa_start
        self.swa_freq = swa_freq
        self.swa_model = None
        self.start_epoch = save_start_epoch
        self.end_epoch = save_end_epoch
        self.save_freq = save_freq

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Called at the beginning of training. Initialize AveragedModel and set epoch counter.
        """
        model = self._trainer.model  # Get model from trainer
        # Initialize SWA model with original model
        self.swa_model = AveragedModel(model = model, avg_fn = get_swa_avg_fn(), use_buffers = True)
        args.logging_dir and print(f"SWA callback initialized: SWA starts from epoch {self.swa_start}, updates every {self.swa_freq} epochs")

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Called at the end of each epoch to decide whether to update SWA model parameters.
        """
        epoch = state.epoch
        # Check if SWA start epoch has been reached, and update SWA weights every `swa_freq` epochs
        if self.swa_start <= epoch and (epoch - self.swa_start) % self.swa_freq == 0:
            model = self._trainer.model  # Get model from trainer
            self.swa_model.update_parameters(model)  # Update SWA model weights
            args.logging_dir and print(f"SWA update: Updated SWA model weights at epoch {epoch}")

        if self.start_epoch <= epoch <= self.end_epoch and (epoch - self.start_epoch) % self.save_freq == 0:
            output_dir = os.path.join(args.output_dir, f"checkpoint-epoch-{int(epoch)}")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            # Save checkpoint
            print(f"Saving checkpoint for epoch {epoch} to {output_dir}")
            torch.save(self.swa_model.module.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))

        return control


class EMA:
    def __init__(self, model, ema_decay=0.9):
        self.module = deepcopy(model)
        self.module.eval()
        self.ema_decay = ema_decay
        self.device = "cuda"
        self.module.to(device=self.device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(
            model,
            update_fn=lambda e, m: self.ema_decay * e + (1. - self.ema_decay) * m
        )

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


class EMACallback(TrainerCallback):
    def __init__(self, trainer, ema_decay=0.99, use_ema_weights=True) -> None:
        super().__init__()
        self._trainer = trainer
        self.ema_decay = ema_decay
        self.use_ema_weights = use_ema_weights
        self.ema = None

    def on_init_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of the initialization of the [`Trainer`].
        """
        self.ema = EMA(self._trainer.model, ema_decay=self.ema_decay)
        return control

    def store(self, parameters):
        "Save the current parameters for restoring later."
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters):
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process.
        """
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)

    def copy_to(self, shadow_parameters, parameters):
        "Copy current parameters into given collection of parameters."
        for s_param, param in zip(shadow_parameters, parameters):
            if param.requires_grad:
                param.data.copy_(s_param.data)

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of a training step. If using gradient accumulation, one training step might take
        several inputs.
        """
        self.ema.update(self._trainer.model)
        self.store(self._trainer.model.parameters())
        self.copy_to(self.ema.module.parameters(),
                     self._trainer.model.parameters())
        return control

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called after an evaluation phase.
        """
        self.restore(self._trainer.model.parameters())
        return control

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of training.
        """
        if self.use_ema_weights:
            self.copy_to(self.ema.module.parameters(),
                         self._trainer.model.parameters())
            msg = "Model weights replaced with the EMA version."
            print(msg)
        return control

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called after a checkpoint save.
        """
        checkpoint_folder = f"ema-checkpoint-{self._trainer.state.global_step}"
        run_dir = args.output_dir
        output_dir = os.path.join(run_dir, checkpoint_folder)
        self.copy_to(self.ema.module.parameters(),
                     self._trainer.model.parameters())
        self._trainer.save_model(output_dir, _internal_call=True)
        self.restore(self._trainer.model.parameters())
        return control


class EarlyStoppingCallback(TrainerCallback):
    def __init__(self, early_stopping_patience: int = 1, early_stopping_threshold: float = 0.0):
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        self.last_best_metric = None
        self.patience_counter = 0

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        metric = state.log_history[-1].get('eval_' + args.metric_for_best_model)
        if metric is None:
            return

        if self.last_best_metric is None:
            self.last_best_metric = metric
            return

        improvement = metric - self.last_best_metric if args.greater_is_better else self.last_best_metric - metric
        if improvement > self.early_stopping_threshold:
            self.patience_counter = 0
            self.last_best_metric = metric
        else:
            self.patience_counter += 1

        if self.patience_counter >= self.early_stopping_patience:
            control.should_training_stop = True


class AveragingCheckpointCallback(TrainerCallback):
    def __init__(self, checkpoint_dir: str, save_name: str = "averaged_model.pth"):
        """
        This Callback averages weights from multiple checkpoints after training ends.
        Args:
            checkpoint_dir (str): Directory where checkpoint files are stored.
            save_name (str): Filename for the averaged model, default is "averaged_model.pth".
        """
        self.checkpoint_dir = checkpoint_dir
        self.save_name = save_name
        self.checkpoints = []

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # Collect all checkpoint paths
        checkpoint_paths = [os.path.join(self.checkpoint_dir, fname) for fname in os.listdir(self.checkpoint_dir) if fname.endswith('.bin')]

        # Load each checkpoint and collect the model's state_dict
        for path in checkpoint_paths:
            checkpoint = torch.load(path)
            self.checkpoints.append(checkpoint['state_dict'])

        if not self.checkpoints:
            raise ValueError("No checkpoint files found.")

        # Average parameters from all checkpoints
        averaged_state_dict = self._average_checkpoints(self.checkpoints)

        # Save the averaged model
        save_path = os.path.join(self.checkpoint_dir, self.save_name)
        torch.save({'state_dict': averaged_state_dict}, save_path)
        print(f"Saved averaged model to {save_path}")

    def _average_checkpoints(self, checkpoints):
        # Initialize an empty state dict to store accumulated parameters
        averaged_state_dict = deepcopy(checkpoints[0])

        # Accumulate parameters from each checkpoint
        for key in averaged_state_dict.keys():
            for checkpoint in checkpoints[1:]:
                averaged_state_dict[key] += checkpoint[key]

        # Divide all accumulated parameters by the number of checkpoints to get the average
        num_checkpoints = len(checkpoints)
        for key in averaged_state_dict.keys():
            averaged_state_dict[key] /= num_checkpoints

        return averaged_state_dict

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Load averaged model weights during evaluation phase.
        """
        model = kwargs.get("model", None)
        if model is None:
            raise ValueError("Model not found during on_evaluate phase.")

        # Load the averaged model
        averaged_checkpoint_path = os.path.join(self.checkpoint_dir, self.save_name)
        if not os.path.exists(averaged_checkpoint_path):
            print(f"Warning: Averaged checkpoint not found at path {averaged_checkpoint_path}")
            return

        checkpoint = torch.load(averaged_checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        print(f"Loaded averaged model from {averaged_checkpoint_path}.")