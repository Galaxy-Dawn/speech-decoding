import os
import pickle
import torch
import hydra
from omegaconf import DictConfig, OmegaConf, ListConfig
from transformers import (
    Trainer,
    TrainingArguments,
    set_seed,
    TrainerCallback,
    TrainerState,
    TrainerControl,
)
from transformers.trainer_utils import get_last_checkpoint
from src.utils.log import setup_logging
from src.utils.aux_func import count_parameters, replace_eval_with_test, print_detailed_parameters
from src.utils.get_callback import SWACallback
from src.utils.get_checkpoint_aggregation import aggregate_checkpoints_swa
from src.utils.get_optimizer import get_optimizer
from src.utils.get_schedular import get_scheduler
from src.model_module.brain_decoder import ModelFactory
from src.data_module.data_func import DataFactory


class EvalPred:
    def __init__(self, predictions):
        self.predictions = predictions

@hydra.main(config_path="conf", config_name="train", version_base="1.2")
def main(cfg: DictConfig):
    LOGGER = setup_logging(level=20)
    training_args = TrainingArguments(
        seed=cfg.training.seed,
        do_train=cfg.training.do_train,
        do_eval=cfg.training.do_eval,
        do_predict=cfg.training.do_predict,
        overwrite_output_dir=cfg.training.overwrite_output_dir,
        output_dir=f'{cfg.dir.model_save_dir}/{cfg.swanlab.exp_name}',
        learning_rate=cfg.training.learning_rate,
        warmup_ratio=cfg.training.warmup_ratio,
        lr_scheduler_type=cfg.training.lr_scheduler_type,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.training.per_device_eval_batch_size,
        weight_decay=cfg.training.weight_decay,
        dataloader_num_workers=cfg.training.dataloader_num_workers,
        num_train_epochs=cfg.training.num_train_epochs,
        eval_strategy=cfg.training.evaluation_strategy,
        eval_steps=cfg.training.eval_steps,
        logging_dir=f'{cfg.dir.logging_dir}/{cfg.swanlab.exp_name}',
        logging_first_step=cfg.training.logging_first_step,
        logging_strategy=cfg.training.logging_strategy,
        logging_steps=cfg.training.logging_steps,
        save_strategy=cfg.training.save_strategy,
        save_steps=cfg.training.save_steps,
        save_total_limit=cfg.training.save_total_limit,
        load_best_model_at_end=cfg.training.load_best_model_at_end,
        metric_for_best_model=cfg.training.metric_for_best_model,
        greater_is_better=cfg.training.greater_is_better,
        remove_unused_columns=cfg.training.remove_unused_columns,
        run_name=cfg.swanlab.exp_name,
        dataloader_pin_memory=cfg.training.dataloader_pin_memory,
        save_safetensors=False,
        report_to='none'
    )

    os.makedirs(training_args.output_dir, exist_ok=True)
    os.makedirs(training_args.logging_dir, exist_ok=True)

    config_save_path = os.path.join(training_args.output_dir, "config.yaml")

    OmegaConf.save(cfg, config_save_path)
    LOGGER.info_high(f"Configuration saved to {config_save_path}")

    brain_model = ModelFactory(cfg.model.name)(cfg)
    data_module = DataFactory(cfg.dataset.name)(cfg)

    print(f"Train dataset size: ", len(data_module.train_dataset))
    print(f"Test dataset size: ", len(data_module.test_dataset))
    total_params = count_parameters(brain_model)
    print_detailed_parameters(brain_model)
    LOGGER.info_high(f"Total number of parameters in the model: {total_params}")

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and cfg.training.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            LOGGER.info_high(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    set_seed(training_args.seed)
    total_training_steps = int(len(data_module.train_dataset) / training_args.per_device_train_batch_size * training_args.num_train_epochs)
    optimizer = get_optimizer(
        optim_type = cfg.training.optimizer,
        model = brain_model,
        lr = cfg.training.learning_rate,
        lr_muon = cfg.training.learning_rate_muon,
        weight_decay = cfg.training.weight_decay)

    scheduler = get_scheduler(
        scheduler_type="linear_warmup_cosine_decay",
        optimizer=optimizer,
        num_warmup_steps=int(total_training_steps * cfg.training.warmup_ratio),
        num_training_steps=int(total_training_steps),
        lr_end=cfg.training.min_learning_rate,
    )

    trainer = Trainer(
        model=brain_model,
        args=training_args,
        train_dataset = data_module.train_dataset if cfg.training.do_train else None,
        eval_dataset = data_module.eval_dataset if cfg.training.do_eval else None,
        data_collator= data_module.data_collator,
        compute_metrics = data_module.compute_metrics,
        optimizers=(optimizer, scheduler),
    )

    swa_callback = SWACallback(
        trainer=trainer,
        swa_start=cfg.training.swa_start,
        swa_freq=cfg.training.swa_freq,
        save_start_epoch=cfg.training.save_start_epoch,
        save_end_epoch=cfg.training.save_end_epoch,
        save_freq=cfg.training.save_freq,
    )

    # Add SWACallback to the trainer's callback list
    trainer.add_callback(swa_callback)

    if cfg.training.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    if cfg.training.do_eval:
        metrics = trainer.evaluate()
        metrics['params'] = total_params
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if cfg.training.do_predict:
        if "_inference_" in training_args.output_dir:
            base_output_dir = training_args.output_dir.rsplit("_inference_", 1)[0]
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            state_dict = torch.load(os.path.join(base_output_dir, "pytorch_model.bin"), map_location=device)
            brain_model.load_state_dict(state_dict)
            LOGGER.info_high(f"Loaded model weights into brain_model")
            trainer.model = brain_model
        else:
            if cfg.training.agg_start_epoch == -1 and cfg.training.agg_end_epoch == -1:
                state_dict = torch.load(os.path.join(training_args.output_dir, "pytorch_model.bin"))
                brain_model.load_state_dict(state_dict)
                LOGGER.info_high(f"Loaded model weights into brain_model")
                trainer.model = brain_model
            else:
                state_dict = aggregate_checkpoints_swa(
                    output_dir=training_args.output_dir,
                    model=brain_model,
                    start_epoch=cfg.training.agg_start_epoch,
                    end_epoch=cfg.training.agg_end_epoch,
                    epoch_freq=cfg.training.agg_freq,
                )
                torch.save(state_dict, os.path.join(training_args.output_dir, "pytorch_model.bin"))
                return

        pred_output = trainer.predict(data_module.test_dataset)
        label = pred_output.label_ids
        metrics = pred_output.metrics
        pred, logits = pred_output.predictions

        with open(os.path.join(training_args.output_dir, "pred.pkl"), "wb") as f:
            pickle.dump(pred, f)
        with open(os.path.join(training_args.output_dir, "logits.pkl"), "wb") as f:
            pickle.dump(logits, f)
        with open(os.path.join(training_args.output_dir, "label.pkl"), "wb") as f:
            pickle.dump(label, f)

        LOGGER.info_high(f"Saving predictions to {os.path.join(training_args.output_dir, 'pred.pkl')}")
        LOGGER.info_high(f"Saving logits to {os.path.join(training_args.output_dir, 'logits.pkl')}")
        LOGGER.info_high(f"Saving labels to {os.path.join(training_args.output_dir, 'label.pkl')}")

        metrics = replace_eval_with_test(metrics)
        metrics['params'] = total_params
        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)

    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()