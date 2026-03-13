from dataclasses import dataclass
from typing import Callable, Dict, Optional, Union
from torch.utils.data import Dataset, IterableDataset
from transformers.data.data_collator import DataCollator
from transformers.trainer_utils import EvalPrediction
import os
from src.utils.aux_func import import_modules

@dataclass
class DataFunction:
    train_dataset: Optional[Union[Dataset, IterableDataset, "datasets.Dataset"]] = None
    eval_dataset: Optional[Union[Dataset, IterableDataset, "datasets.Dataset"]] = None
    test_dataset: Optional[Union[Dataset, IterableDataset, "datasets.Dataset"]] = None
    data_collator: Optional[DataCollator] = None
    compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None
    def __call__(self) -> "DataFunction":
        return self

DATA_FACTORY: Dict[str, DataFunction] = {}

def DataFactory(data_name: str) -> DataFunction:
    if len(data_name.split("_")) > 2 and "_".join(data_name.split("_")[:2]) in DATA_FACTORY:
        data = DATA_FACTORY.get("_".join(data_name.split("_")[:2]))
    else:
        data = DATA_FACTORY.get(data_name, None)
    if data is None:
        raise ValueError(f"{data_name} data func is not implmentation")
    return data


def register_data(name: str) -> Callable:
    def register_data_cls(cls):
        if name in DATA_FACTORY:
            return DATA_FACTORY[name]

        DATA_FACTORY[name] = cls
        return cls
    return register_data_cls


models_dir = os.path.dirname(__file__)
import_modules(models_dir, "src.data_module.data_func")