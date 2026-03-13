import numpy as np
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold


def check_ieeg(ieeg):
    matrix = np.max(ieeg, axis=-1) - np.min(ieeg, axis=-1)
    mask = np.all(matrix != 0, axis=1)
    res = np.where(mask)[0]
    return res.tolist()


def get_split(data_args, ieeg, label=None):
    if label is None:
        label = ieeg
    train_ratio = data_args.train_ratio
    val_ratio = data_args.eval_ratio
    test_ratio = data_args.test_ratio
    stratify_flag = data_args.stratify_flag
    random_seed = data_args.random_seed
    original_indices = np.arange(len(ieeg))
    if stratify_flag:
        train_indices, temp_indices = train_test_split(
            original_indices,
            test_size=(1 - train_ratio),
            random_state=random_seed,
            shuffle=True,
            stratify=label
        )
        if test_ratio == 0:
            val_indices, test_indices = temp_indices, temp_indices
        else:
            val_size = val_ratio / (val_ratio + test_ratio)
            val_indices, test_indices = train_test_split(
                temp_indices,
                test_size=(1 - val_size),
                random_state=random_seed,
                shuffle=True,
                stratify=label[temp_indices]
            )
    else:
        train_indices, temp_indices = train_test_split(
            original_indices,
            test_size=(1 - train_ratio),
            random_state=random_seed,
            shuffle=True,
        )
        if test_ratio == 0:
            val_indices, test_indices = temp_indices, temp_indices
        else:
            val_size = val_ratio / (val_ratio + test_ratio)
            val_indices, test_indices = train_test_split(
                temp_indices,
                test_size=(1 - val_size),
                random_state=random_seed,
                shuffle=True,
            )
    return train_indices, val_indices, test_indices