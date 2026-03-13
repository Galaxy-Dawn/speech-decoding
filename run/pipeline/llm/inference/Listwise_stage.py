import json
import os
import numpy as np
from omegaconf import DictConfig, OmegaConf
import hydra


SYSTEM_PROMPT="""
## 角色
你是一个专业高效的拼音序列排序专家，专注于从20个选项(A-T)中快速准确地识别出最通顺的三个拼音序列组合。
## 技能
1. 当接收到20个不同的拼音序列选项（选项从A到T）时，运用丰富的语言经验和专业知识，全面分析每个选项通过拼音到汉字的翻译，组成通顺语句的可能性。
2. 从这些选项中精心挑选出最可能组成通顺语句的三个选项，根据通顺度对选项进行优先级排序，对同等通顺度的选项按字母顺序排列。
## 限制:
- 直接输出排序题的答案，答案必须为3个英文字母。
- 所输出的内容必须按照给定的格式进行组织，绝不能偏离框架要求。							
"""

def generate_user_prompt(syllables_list):
    index_map = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T"]
    sequences = [
        f"{index_map[i]}：'{seq}'"
        for i, seq in enumerate(syllables_list)
    ]
    formatted_sequences = "\n".join(sequences)
    prompt = (
        f"{formatted_sequences}, 经过各个拼音序列选项的比较，正确的三个选项为："
    )
    return prompt


def direct_to_listwise_input(beam_search_results, output_file_path=None):
    alpaca_dataset = []
    for item in beam_search_results:
        alpaca_entry = {
            "instruction": SYSTEM_PROMPT,
            "input"      : generate_user_prompt(item['input']),
            "output"     : item['output']
        }
        alpaca_dataset.append(alpaca_entry)

    if output_file_path is not None:
        with open(output_file_path, "w", encoding="utf-8") as f:
            json.dump(alpaca_dataset, f, ensure_ascii=False, indent=4)
        print(f"Processed data saved to {output_file_path}")

    return alpaca_dataset


@hydra.main(config_path="../../../conf", config_name="llm_inference", version_base="1.2")
def main(cfg: DictConfig):
    if cfg.inference_module.name == "all_in_two":
        if cfg.beam_search.output_path.endswith('.pkl'):
            beam_search_results = np.load(cfg.beam_search.output_path, allow_pickle=True)
            _ = direct_to_listwise_input(beam_search_results, output_file_path=cfg.inference_module.listwise.input_path)
        else:
            raise ValueError("Invalid beam_search_path")
    else:
        print("Skip Listwise Stage")
    print(f"Save Listwise Input to {cfg.inference_module.listwise.input_path}")
    json_file_path = f"{cfg.dir.llm_data_dir}/dataset_info.json"
    # Dataset information to add
    new_dataset_entry = {
        f"speech_decoding_inference_{cfg.inference_module.name}_listwise_{cfg.test.subject}_{cfg.test.state}_input": {
            "file_name": f"{cfg.inference_module.listwise.input_path}"
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