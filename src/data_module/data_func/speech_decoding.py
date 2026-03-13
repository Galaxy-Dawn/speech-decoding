from omegaconf import DictConfig
import torch
import numpy as np
from pathlib import Path
from src.data_module.data_func import DataFunction
from src.data_module.compute_metrics import MetricsFactory
from src.data_module.collate_fn import DataCollatorFactory
from src.data_module.dataset import DatasetFactory
from src.data_module.data_func import register_data


@register_data('speech_decoding')
def speech_decoding_data(cfg: DictConfig):
    processed_dir: Path = Path(cfg.dir.processed_dir) / cfg.dataset.name / cfg.dataset.task
    training_dataset_path = processed_dir / f'{cfg.dataset.name}_{cfg.dataset.id}_training_data.pt'
    test_dataset_path = processed_dir / f'{cfg.dataset.name}_{cfg.dataset.id}_test_data.pt'

    if cfg.dataset.split_method == 'simple':
        train_split_filename = processed_dir / f'{cfg.dataset.name}_{cfg.dataset.id}_train_split.npy'
        eval_split_filename = processed_dir / f'{cfg.dataset.name}_{cfg.dataset.id}_eval_split.npy'
    else:
        train_split_filename = None
        eval_split_filename = None

    train_dataset_load = torch.load(training_dataset_path, weights_only=True, map_location='cpu')
    test_dataset_load = torch.load(test_dataset_path, weights_only=True, map_location='cpu')

    if cfg.dataset.split_method == 'none':
        train_dataset = DatasetFactory('speech_decoding')(train_dataset_load)
        eval_dataset = None
    elif cfg.dataset.split_method == 'simple':
        train_split, eval_split = np.load(train_split_filename), np.load(eval_split_filename)
        train_dataset_list = [train_dataset_load[i] for i in train_split]
        eval_dataset_list = [train_dataset_load[i] for i in eval_split]
        train_dataset = DatasetFactory('speech_decoding')(train_dataset_list)
        eval_dataset = DatasetFactory('speech_decoding')(eval_dataset_list)
    else:
        train_dataset = None
        eval_dataset = None

    test_dataset = DatasetFactory('speech_decoding')(test_dataset_load)
    data_collator = DataCollatorFactory('speech_decoding')
    compute_metrics = MetricsFactory('classification')

    return DataFunction(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        test_dataset=test_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )