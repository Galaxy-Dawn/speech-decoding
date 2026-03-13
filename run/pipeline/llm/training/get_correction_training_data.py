import json
import os

import numpy as np
from tqdm import tqdm
import hydra


def select_label(data_path, is_deberta_score, is_shuffle=False):
    if is_deberta_score:
        index_map = ["A","B","C","D","E","F","G","H","I","J"]
    else:
        index_map = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T"]
    input_list = []
    with open(data_path, 'r') as f:
        for line in f:
            line_res = []
            data = json.loads(line)
            if is_deberta_score: choice_list = data['prompt'].split("\n")[17:27]
            else: choice_list = data['prompt'].split("\n")[16:36]
            for i in range(len(choice_list)):
                split_string = choice_list[i].split("\'")
                if len(split_string) == 1:
                    choice_list[i] = ""
                else:
                    choice_list[i] = split_string[1]
            for j in data['predict']:
                if j in index_map:
                    line_res.append(choice_list[index_map.index(j)])
                else:
                    line_res.append(choice_list[0])
            if is_shuffle:
                np.random.shuffle(line_res)
            input_list.append(line_res)
    return input_list


def generate_user_prompt(perturbed_syllables_list):
    # Extract input sequences and format with numbering
    sequences = [
        f"拼音序列{i + 1}：'{seq}'"
        for i, seq in enumerate(perturbed_syllables_list)
    ]
    # Handle punctuation: first two sequences separated by semicolon, third sequence preceded by comma
    formatted_sequences = "; ".join(sequences[:-1]) + f", {sequences[-1]}"
    # Build complete prompt
    prompt = (
        f"{formatted_sequences}, 经过模型修正，正确的中文句子为："
    )
    return prompt


def generate_dataset_from_reranker(data: list, listwise_input: list, system_prompt: str) -> list:
    alpaca_dataset = []
    for i in tqdm(range(len(data))):
        alpaca_entry = {
            "instruction": system_prompt,
            "input"      : generate_user_prompt(data[i]),
            "output"     : listwise_input[i]['text']
        }
        alpaca_dataset.append(alpaca_entry)

    return alpaca_dataset


def add_dataset_info_to_llama_factory(base_dir, dataset_name, dataset_path):
    json_file_path = f"{base_dir}/dataset_info.json"
    # Dataset information to add
    new_dataset_entry = {
        f"{dataset_name}": {
            "file_name": f"{dataset_path}"
        }
    }
    # Check if file exists
    if os.path.exists(json_file_path):
        # Read existing content
        with open(json_file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                # If file is empty or incorrectly formatted, initialize as empty dictionary
                data = {}
    else:
        data = {}
    data.update(new_dataset_entry)
    # Write updated content
    with open(json_file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Successfully added {dataset_name} dataset information to {json_file_path}")


@hydra.main(config_path="../../../conf", config_name="llm_training", version_base="1.2")
def get_correction_dataset(cfg):
    listwise_model_name = "Qwen7B_rank"
    for stage in ["train", "eval"]:
        listwise_prediction_path = f"{cfg.dir.llm_save_dir}/{listwise_model_name}/{stage}/generated_predictions.jsonl"
        listwise_data_path = f"{cfg.dir.llm_data_dir}/listwise/{stage}_set.json"
        correction_data_path = f"{cfg.dir.llm_data_dir}/syllable_correction_uni_{stage}.json"
        with open(listwise_data_path, "r", encoding="utf-8") as f:
            listwise_data = json.load(f)
        listwise_prediction = select_label(listwise_prediction_path, is_deberta_score=False, is_shuffle=True)
        dataset = generate_dataset_from_reranker(listwise_prediction, listwise_input=listwise_data, system_prompt=cfg.system_prompt_correction)
        with open(correction_data_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=4)
        print(f"已保存处理后的数据到 {correction_data_path}")
        add_dataset_info_to_llama_factory(cfg.dir.llm_data_dir, f"syllable_correction_uni_{stage}", correction_data_path)


if __name__ == "__main__":
    get_correction_dataset()
