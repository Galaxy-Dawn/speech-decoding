from box import Box
from pathlib import Path
import numpy as np
import hydra


def get_meta_data(data_dir, subject_id):
    base_dir = Path(data_dir) / subject_id
    data_path = base_dir / "processed_data" / "sentence" / "data_reading.pkl"
    channel_index_path = base_dir / "ch_dict.pkl"
    data = np.load(data_path, allow_pickle=True)
    num_channels = data[0].shape[0]
    channel_index_list = list(np.load(channel_index_path, allow_pickle=True).keys())
    channel_index_str = " ".join(channel_index_list)
    channel_index_name = "all"
    return num_channels, channel_index_name, channel_index_str


@hydra.main(
    config_path='../conf',
    config_name="generate_yaml",
    version_base="1.2",
)
def generate_speech_decoding_yaml(cfg):
    yaml_base_dir = Path(cfg.project_dir) / 'run' / 'conf' / 'dataset'
    num_channels, channel_index_name, channel_index_str = get_meta_data(
        cfg.dir.data_dir, cfg.subject_id
    )
    print("Number of channels:", num_channels,
          "channel_index_name:", channel_index_name,
          "channel_index_str:", channel_index_str)
    yaml_content = Box(
        {
            "defaults": ["speech_decoding_base"],
            "id": cfg.subject_id,
            "input_channels": num_channels,
            "channel_index_name": channel_index_name,
        }
    )

    file_path = f"{yaml_base_dir}/speech_decoding_{cfg.subject_id}.yaml"
    if not Path(file_path).exists():
        yaml_content.to_yaml(file_path)
    else:
        print(f"yaml file {file_path} already exists, skip")
    return channel_index_str


if __name__ == "__main__":
    generate_speech_decoding_yaml()
