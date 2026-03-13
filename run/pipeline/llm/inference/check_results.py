import json
import pickle
import hydra
from omegaconf import DictConfig, OmegaConf
import re
from src.utils.get_sentence_inference_results import get_sentence_results


@hydra.main(config_path="../../../conf", config_name="llm_inference", version_base="1.2")
def main(cfg: DictConfig):
    initial_final_results_path = cfg.results.initial_final_results_path
    beam_search_path = cfg.beam_search.output_path #20  pkl
    listwise_input_path = cfg.inference_module.listwise.input_path if cfg.inference_module.name == "all_in_two" else None #10  json
    listwise_output_path = cfg.inference_module.listwise.output_path + "/generated_predictions.jsonl" if cfg.inference_module.name == "all_in_two" else None
    correction_is_permutation = cfg.inference_module.correction.is_permutation
    correction_input_path = cfg.inference_module.correction.input_path #1 json
    correction_output_path = cfg.inference_module.correction.output_path + "/generated_predictions.jsonl" #jsonl
    correction_metric_path = cfg.inference_module.correction.output_path + "/predict_results.json"

    with open(initial_final_results_path, 'r', encoding='utf-8') as f:
        initial_final_results = json.load(f)

    # 1. Read .pkl file (beam_search results)
    with open(beam_search_path, 'rb') as f:
        beam_search_data = pickle.load(f)

    # 2. Read .json file (listwise input)
    if listwise_input_path is not None:
        with open(listwise_input_path, 'r', encoding='utf-8') as f:
            listwise_input_data = json.load(f)
    else:
        listwise_input_data = None

    # 3. Read .jsonl file (listwise output)
    if listwise_output_path is not None:
        listwise_output_data = []
        with open(listwise_output_path, 'r', encoding='utf-8') as f:
            for line in f:
                listwise_output_data.append(json.loads(line.strip()))
        print("Loaded listwise output data from", listwise_output_path)
    else:
        listwise_output_data = None

    # 4. Read .json file (correction input)
    with open(correction_input_path, 'r', encoding='utf-8') as f:
        correction_input_data = json.load(f)

    # 5. (Optional) If correction_output_path exists, read it
    correction_output_data = []
    with open(correction_output_path, 'r', encoding='utf-8') as f:
        for line in f:
            correction_output_data.append(json.loads(line.strip()))

    with open(correction_metric_path, 'r', encoding='utf-8') as f:
        correction_metric_data = json.load(f)

    separate_results = []

    for idx in range(len(beam_search_data)):
        if beam_search_path.endswith('.pkl'):
            beam_search_output = beam_search_data[idx]['input']
            label = beam_search_data[idx]['output']
        elif beam_search_path.endswith('.json'):
            beam_search_output = beam_search_data[idx]['beam_search_output']
            label = beam_search_data[idx]['label']
        else:
            raise ValueError("Invalid beam_search_path")
        pattern = r"['\"](.*?)['\"]"
        if listwise_input_data is not None:
            listwise_input = re.findall(pattern, listwise_input_data[idx]['input'])
        else:
            listwise_input = None
        pattern = r"拼音序列\d+：'(.*?)'"

        correction_input = re.findall(pattern, correction_input_data[idx]['input'])
        correction_output = correction_output_data[idx]['predict']
        results = {
            "label": label,
            "beam search output": beam_search_output,
            "listwise input": listwise_input,
            "listwise output_choice": listwise_output_data[idx] if listwise_output_data is not None else None,
            "listwise output": correction_input,
            "correction output": correction_output,
        }
        separate_results.append(results)

    total_results = []

    total_results.append(initial_final_results)

    metric_dict = {
        "bleu-4": correction_metric_data['predict_bleu-4'],
        "rouge-1": correction_metric_data['predict_rouge-1'],
        "rouge-2": correction_metric_data['predict_rouge-2'],
        "rouge-l": correction_metric_data['predict_rouge-l'],
    }

    total_results.append(metric_dict)
    total_results.append(separate_results)

    metrics, beam_search_results= get_sentence_results(total_results, correction_is_permutation=correction_is_permutation)
    total_results.append(metrics)

    json.dump(total_results, open(cfg.results.total_results_path, 'w'), ensure_ascii=False, indent=4)
    json.dump(beam_search_results, open(cfg.results.beam_search_results_path, 'w'), ensure_ascii=False, indent=4)

    print("Results saved to", cfg.results.total_results_path)
    print("Beam search results saved to", cfg.results.beam_search_results_path)


if __name__ == "__main__":
    main()