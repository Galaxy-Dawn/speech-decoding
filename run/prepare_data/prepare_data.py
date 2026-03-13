from omegaconf import DictConfig
import hydra
from tqdm import tqdm
import numpy as np
from src.data_module.utils import get_split
import torch
from pathlib import Path
from src.utils.log import setup_logging, cprint, tracking
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="torch")


def gather_data(data_args, data_path, type):
    task = data_args.dataset.task
    exp_category = task.split('_')[0]
    label_category = task.split('_')[1]
    if type == "character":
        eeg = np.load(data_path + f'/data_{exp_category}.npy', allow_pickle=True)
        eeg = np.swapaxes(eeg, 0, 1)
    elif type == "sentence":
        all_sentence_eeg = np.load(data_path + f'/data_{exp_category}.pkl', allow_pickle=True)
        eeg_list = []
        for sentence_eeg in all_sentence_eeg:
            sentence_eeg = np.swapaxes(sentence_eeg, 0, 1)
            for character_eeg in sentence_eeg:
                eeg_list.append(character_eeg)
        eeg = np.array(eeg_list)
    else:
        raise ValueError("Invalid data type")

    if type == "character":
        label = np.load(data_path + f'/{label_category}_label_{exp_category}.npy', allow_pickle=True)
    elif type == "sentence":
        all_sentence_label = np.load(data_path + f'/{label_category}_label_{exp_category}.pkl', allow_pickle=True)
        label_list = []
        for sentence_label in all_sentence_label:
            for character_label in sentence_label:
                label_list.append(character_label)
        label = np.array(label_list)
    else:
        raise ValueError("Invalid data type")

    return eeg, label


@hydra.main(config_path="../conf", config_name="prepare_data", version_base="1.2")
def prepare_data(cfg: DictConfig):
    LOGGER = setup_logging(level = 20)
    data_dir: Path = Path(cfg.dir.data_dir)
    processed_dir: Path = Path(cfg.dir.processed_dir) / cfg.dataset.name / cfg.dataset.task
    processed_dir.mkdir(parents=True, exist_ok=True)
    id = cfg.dataset.id

    with tracking("Load and gather data", LOGGER):
        training_data_path = data_dir / id / 'processed_data' / 'character'
        test_character_data_path = data_dir / id / 'processed_data' / 'sentence'

        train_ieeg, train_label = gather_data(cfg, training_data_path, 'character')
        LOGGER.info_high(f"Loaded Subject {id} training data with shape: {train_ieeg.shape}")

        test_ieeg, test_label = gather_data(cfg, test_character_data_path, 'sentence')
        LOGGER.info_high(f"Loaded Subject {id} test data with shape: {test_ieeg.shape}")

    with tracking("Get and save split", LOGGER):
        if cfg.split_method == 'simple':
            train_split, eval_split, _ = get_split(cfg, train_ieeg, label=train_label)

            train_split_filename = f'{cfg.dataset.name}_{id}_train_split.npy'
            eval_split_filename = f'{cfg.dataset.name}_{id}_eval_split.npy'

            np.save(processed_dir / train_split_filename, train_split)
            np.save(processed_dir / eval_split_filename, eval_split)
        else:
            LOGGER.info_high("No need to split validation dataset")


    with tracking("Prepare and save data", LOGGER):
        training_save_path = processed_dir / f'{cfg.dataset.name}_{id}_training_data.pt'
        test_save_path = processed_dir / f'{cfg.dataset.name}_{id}_test_data.pt'

        dataset_list = []
        for i in tqdm(range(len(train_ieeg)), desc='Preparing data'):
            data_dict = {
                'ieeg_raw_data': torch.tensor(train_ieeg[i]),
                'labels'       : torch.tensor(train_label[i]),
            }
            dataset_list.append(data_dict)
        torch.save(dataset_list, training_save_path)
        LOGGER.info_high(f"Saved Subject {id} training data with shape: {train_ieeg.shape} in {training_save_path}")

        dataset_list = []
        for i in tqdm(range(len(test_ieeg)), desc='Preparing data'):
            data_dict = {
                'ieeg_raw_data': torch.tensor(test_ieeg[i]),
                'labels'       : torch.tensor(test_label[i]),
            }
            dataset_list.append(data_dict)
        torch.save(dataset_list, test_save_path)
        LOGGER.info_high(f"Saved Subject {id} test data with shape: {test_ieeg.shape} in {test_save_path}")


if __name__ == '__main__':
    prepare_data()