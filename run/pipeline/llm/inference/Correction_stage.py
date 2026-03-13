import json
import os
import pickle
from pathlib import Path
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm


SYSTEM_PROMPT="""
## 角色											
你是一位专业严谨的拼音纠错推理大师，擅长通过无声调拼音序列推理出正确的中文语句。你的任务是：1.分析输入的拼音序列（可能包含错误）；2.通过语境、语法和常见表达习惯进行逻辑推理；3.输出最合理通顺的中文句子。															
## 限制:											
- 不要显式输出详细的推理过程。最后直接输出正确句子。											
- 输出的内容需符合中文语言表达习惯和逻辑。																			
"""


def select_label(data_path):
    index_map = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T"]
    input_list = []
    with open(data_path, 'r') as f:
        for line in f:
            line_res = []
            data = json.loads(line)
            choice_list = data['prompt'].split("\n")[16:36]
            for i in range(len(choice_list)):
                split_string = choice_list[i].split("\'")
                if len(split_string) == 1:
                    choice_list[i] = ""
                else:
                    choice_list[i] = split_string[1]
            for j in data['predict']:
                line_res.append(choice_list[index_map.index(j)])
            input_list.append(line_res)
    return input_list


def generate_user_prompt(perturbed_syllables_list, is_permutation=False):
    if is_permutation:
        # Generate all permutations of 3 elements (total 6 permutations)
        import itertools
        permutations = list(itertools.permutations(perturbed_syllables_list))
        prompts = []
        for perm in permutations:
            sequences = [
                f"拼音序列{i + 1}：'{seq}'"
                for i, seq in enumerate(perm)
            ]
            formatted_sequences = "; ".join(sequences[:-1]) + f", {sequences[-1]}"
            prompt = f"{formatted_sequences}, 经过模型修正，正确的中文句子为："
            prompts.append(prompt)
        return prompts
    else:
        # Original functionality remains unchanged
        sequences = [
            f"拼音序列{i + 1}：'{seq}'"
            for i, seq in enumerate(perturbed_syllables_list)
        ]
        formatted_sequences = "; ".join(sequences[:-1]) + f", {sequences[-1]}"
        prompt = f"{formatted_sequences}, 经过模型修正，正确的中文句子为："
        return prompt


def generate_dataset_from_reranker(data: list, label_list: list, is_permutation=False) -> list:
    alpaca_dataset = []
    for i in tqdm(range(len(data))):
        alpaca_entry = {
            "instruction": SYSTEM_PROMPT,
            "input"      : generate_user_prompt(data[i], is_permutation),
            "output"     : label_list[i%44][0]
        }
        alpaca_dataset.append(alpaca_entry)
    return alpaca_dataset


@hydra.main(config_path="../../../conf", config_name="llm_inference", version_base="1.2")
def main(cfg: DictConfig):
    label_list = np.load(cfg.test.save_label_path, allow_pickle=True)
    if cfg.inference_module.name == "all_in_two":
        data = select_label(Path(cfg.inference_module.listwise.output_path) / "generated_predictions.jsonl")
        dataset = generate_dataset_from_reranker(data, label_list, is_permutation=cfg.inference_module.correction.is_permutation)

    with open(cfg.inference_module.correction.input_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)

    json_file_path = f"{cfg.dir.llm_data_dir}/dataset_info.json"
    # Dataset information to add
    new_dataset_entry = {
        f"speech_decoding_inference_{cfg.inference_module.name}_correction_{cfg.test.subject}_{cfg.test.state}_input": {
            "file_name": f"{cfg.inference_module.correction.input_path}"
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
    print(f"Dataset information successfully added to {json_file_path}")


if __name__ == "__main__":
    main()