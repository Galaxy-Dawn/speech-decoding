import json
import os
import pickle
import random
import time
from numpy import average
from pathlib import Path
import torch
import hydra
import numpy as np
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from src.model_module.brain_decoder import MODEL_FACTORY
from src.data_module.dataset.simple_dataset import SimpleDataset
from sklearn.metrics import top_k_accuracy_score
from sklearn.metrics import accuracy_score, f1_score, precision_score
from speechbrain.decoders import TorchAudioCTCPrefixBeamSearcher


INITIAL_MAP = ['b', 'f', 'g', 'x', 'd', 'l', 'j', 'sh', 't', 's', '-', ]

FINAL_MAP = ['u', 'ang', 'ei', 'iang', 've', 'iao', 'e', 'ai', 'ia', 'in', 'i', 'a', 'ian', 'uo', 'an']

TONE_MAP = ['1', '2', '3', '4']

SYLLABLE_MAP = ['ba', 'bu', 'de', 'di', 'fang', 'ge', 'gei', 'jia', 'ji', 'jin', 'le', 'lai', 'shi', 'san', 'tian', 'ta',
                 'wo', 'xiang', 'xiao', 'xue', 'xin', 'yi']

NORMAL_SYLLABLE_MAP = {
    "-a": "a",
    "-ai": "ai",
    "-an": "an",
    "-ang": "ang",
    "-ao": "ao",
    "-e": "e",
    "-ei": "ei",
    "-en": "en",
    "-er": "er",
    "-o": "o",
    "-ou": "ou",
    "-ua": "wa",
    "-uai": "wai",
    "-uan": "wan",
    "-uang": "wang",
    "-uei": "wei",
    "-uen": "wen",
    "-ueng": "weng",
    "-uo": "wo",
    "-u": "wu",
    "-ia": "ya",
    "-ian": "yan",
    "-iang": "yang",
    "-iao": "yao",
    "-ie": "ye",
    "-i": "yi",
    "-in": "yin",
    "-ing": "ying",
    "-iong": "yong",
    "-iou": "you",
    "-v": "yu",
    "-van": "yuan",
    "-ve": "yue",
    "-vn": "yun",
    "jv": "ju",
    "jvan": "juan",
    "jve": "jue",
    "jvn": "jun",
    "qv": "qu",
    "qvan": "quan",
    "qve": "que",
    "qvn": "qun",
    "xv": "xu",
    "xvan": "xuan",
    "xve": "xue",
    "xvn": "xun",
}


def set_seed(seed: int):
    """
    Set random seed to ensure reproducible experimental results

    Args:
        seed (int): Random seed value
    """
    # Set seed for Python's built-in random module
    random.seed(seed)
    # Set seed for NumPy
    np.random.seed(seed)
    # Set seed for PyTorch
    torch.manual_seed(seed)
    # If using CUDA, also set CUDA's random seed
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # If using multiple GPUs
    # To ensure fully reproducible results, you can also set the following options
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ordered_deduplicate(str_list):
    seen = set()
    return [x for x in str_list if not (x in seen or seen.add(x))]


class EvalPred:
    def __init__(self, predictions):
        self.predictions = predictions


def calculate_char_accuracy(reference, candidates):
    # Convert reference string to character list (ignoring spaces)
    ref_chars = reference.split(" ")
    total_chars = len(ref_chars)
    candidate_list = [cand.split(" ") for cand in candidates if cand != ' ']
    accuracy_list = []
    exact_match = 0
    error_mapping = {}

    for candidate_idx in range(len(candidate_list)):
        candidate_str = candidate_list[candidate_idx]
        if candidate_str:
            error = 0
            if len(candidate_str) != len(ref_chars):
                print("Length mismatch!")
            for str_idx in range(len(candidate_str)):
                if candidate_str[str_idx] != ref_chars[str_idx]:
                    error += 1
                    key = (ref_chars[str_idx], candidate_str[str_idx])
                    error_mapping[key] = error_mapping.get(key, 0) + 1

            if error == 0 and exact_match == 0:
                exact_match += 1
            accuracy_list.append(error / total_chars)

    return accuracy_list, exact_match, error_mapping


def merge_error_mappings(error_mappings):
    merged = {}
    for mapping in error_mappings:
        for (correct, wrong), count in mapping.items():
            merged[(correct, wrong)] = merged.get((correct, wrong), 0) + count
    return merged


def calculate_value_distribution(data, num_bins=10):
    """
    Calculate the distribution frequency of values in specified intervals
    :param data: List[List] 2D numerical list
    :param num_bins: Number of bins (default 10 intervals)
    :return: Dictionary containing interval labels and frequencies
    """
    # Flatten the 2D list
    flattened = [item for sublist in data for item in sublist]

    # Generate interval labels
    bin_edges = [round(i / num_bins, 1) for i in range(num_bins + 1)]
    labels = [f"{bin_edges[i]}-{bin_edges[i + 1]}" for i in range(num_bins)]

    # Initialize count dictionary
    distribution = {label: 0 for label in labels}

    # Count total valid data
    valid_count = 0

    # Count frequencies
    for value in flattened:
        if not (0 <= value <= 1):
            continue

        valid_count += 1
        bin_index = int(value * num_bins)
        if bin_index == num_bins:
            bin_index -= 1

        distribution[labels[bin_index]] += 1

    # Convert to frequencies
    if valid_count > 0:
        for label in labels:
            distribution[label] = distribution[label] / valid_count

    return distribution


def process_silence_str(hyps, silence_str = ".", padding_str = "+"):
    res = []
    for i in range(len(hyps)):
        if len(hyps[i])!=0:
            for j in range(len(hyps[i])):
                total_syllable_str = hyps[i][j].text.strip(silence_str)
                legal_syllable_str = total_syllable_str.replace(padding_str, " ").strip()
                legal_syllable_list = legal_syllable_str.split(" ")
                for k in range(len(legal_syllable_list)):
                    sub_str = legal_syllable_list[k]
                    if sub_str in NORMAL_SYLLABLE_MAP:
                        legal_syllable_list[k] = NORMAL_SYLLABLE_MAP[sub_str]
                res.append(" ".join(legal_syllable_list))
        else:
            res.append([""])
    return res


def mapping_probability(logits, label, vocab_list):
    seq_len = logits.shape[0]
    candidates_num = logits.shape[1]
    total_phoneme = len(vocab_list)
    probs_matrix = np.zeros((seq_len, total_phoneme))
    for i in range(seq_len):
        for j in range(candidates_num):
            candidate = label[j]
            candidate_index = vocab_list.index(candidate)
            probs_matrix[i, candidate_index] = logits[i, j].item()
    return probs_matrix


def simple_compute_metrics(eval_pred, label_name_list):
    logits, labels = eval_pred.predictions[1], eval_pred.predictions[0]
    logits = torch.concat(logits, dim=0)
    labels = torch.concat(labels, dim=0)
    preds = torch.argmax(logits, dim=1)
    top1_accuracy = accuracy_score(labels, preds)
    top3_accuracy = top_k_accuracy_score(labels, logits, k=3, labels=np.arange(logits.shape[1]))
    top5_accuracy = top_k_accuracy_score(labels, logits, k=5, labels=np.arange(logits.shape[1]))
    precision = precision_score(labels, preds, average='weighted')
    f1 = f1_score(labels, preds, average='weighted')
    if isinstance(labels, np.ndarray):  # Check if it's a numpy.ndarray
        labels = torch.tensor(labels)  # Convert to torch.Tensor

    unique_labels = torch.unique(labels)
    class_accuracies = {}

    for label in unique_labels:
        label_name = label_name_list[int(label)]
        true_positive = torch.sum((preds == label) & (labels == label)).item()
        total_samples = torch.sum(labels == label).item()
        if total_samples > 0:
            class_accuracies[f'acc_class_{label_name}'] = true_positive / total_samples
        else:
            class_accuracies[f'acc_class_{label_name}'] = 0.0

    return {
        # Model metrics
        'top1_accuracy'       : top1_accuracy,
        'top3_accuracy'       : top3_accuracy,
        'top5_accuracy'       : top5_accuracy,
        'precision'           : precision,
        'f1'                  : f1,
        # Class-specific metrics
        **class_accuracies
    }


def beam_search_if_with_lexicon(vocab_list, lexicon, initial_map, final_map, prob_initial, prob_final, beam_size=5, top_k=3):
    if "+" not in vocab_list:
        vocab_list.insert(-1, "+")
    initial_prob_matrix = mapping_probability(prob_initial, initial_map, vocab_list)
    final_prob_matrix = mapping_probability(prob_final, final_map, vocab_list)
    num_rows, num_cols = initial_prob_matrix.shape
    combined_matrix = np.zeros((3 * num_rows, num_cols))
    for i in range(num_rows):
        combined_matrix[3 * i, :] = initial_prob_matrix[i, :]
        combined_matrix[3 * i + 1, :] = final_prob_matrix[i, :]
        combined_matrix[3 * i + 2, :] = np.zeros(num_cols)
        combined_matrix[3 * i + 2, -2] = 1.0
    combined_matrix = combined_matrix[np.newaxis, :, :]
    combined_matrix = torch.log(torch.tensor(combined_matrix))
    combined_matrix = combined_matrix.float()
    searcher = TorchAudioCTCPrefixBeamSearcher(
        tokens=vocab_list,
        lexicon=lexicon,
        blank_index=0,
        sil_index=len(vocab_list) - 1,
        beam_size=beam_size,
        topk=top_k,
        using_cpu_decoder=True
    )
    hyps = searcher(combined_matrix)
    pred_text = process_silence_str(hyps)
    return pred_text


def beam_search_if_with_tone_and_lexicon(vocab_list, lexicon, initial_map, final_map, tone_map, prob_initial, prob_final, prob_tone, beam_size=5, top_k=3):
    if "+" not in vocab_list:
        vocab_list.insert(-1, "+")
    initial_prob_matrix = mapping_probability(prob_initial, initial_map, vocab_list)
    final_prob_matrix = mapping_probability(prob_final, final_map, vocab_list)
    tone_prob_matrix = mapping_probability(prob_tone, tone_map, vocab_list)
    num_rows, num_cols = initial_prob_matrix.shape
    combined_matrix = np.zeros((4 * num_rows, num_cols))
    for i in range(num_rows):
        combined_matrix[4 * i, :] = initial_prob_matrix[i, :]
        combined_matrix[4 * i + 1, :] = final_prob_matrix[i, :]
        combined_matrix[4 * i + 2, :] = tone_prob_matrix[i, :]
        combined_matrix[4 * i + 3, :] = np.zeros(num_cols)
        combined_matrix[4 * i + 3, -2] = 1.0
    combined_matrix = combined_matrix[np.newaxis, :, :]
    combined_matrix = torch.log(torch.tensor(combined_matrix))
    combined_matrix = combined_matrix.float()
    searcher = TorchAudioCTCPrefixBeamSearcher(
        tokens=vocab_list,
        lexicon=lexicon,
        blank_index=0,
        sil_index=len(vocab_list) - 1,
        beam_size=beam_size,
        topk=top_k,
        using_cpu_decoder=True
    )
    hyps = searcher(combined_matrix)
    pred_text = process_silence_str(hyps)
    return pred_text


def inference_multi_model(cfg, device, ieeg, label, type):
    all_labels = None
    all_pred = None
    run_flag = 0
    if type == "syllable":
        checkpoint_paths = cfg.test.syllable_checkpoints
    elif type == "initial":
        checkpoint_paths = cfg.test.initial_checkpoints
    elif type == "final":
        checkpoint_paths = cfg.test.final_checkpoints
    elif type == "tone":
        checkpoint_paths = cfg.test.tone_checkpoints

    for checkpoint_path in checkpoint_paths:
        checkpoint_dir = Path(checkpoint_path).parent
        model_cfg_path = checkpoint_dir / "config.yaml"
        if not os.path.exists(model_cfg_path):
            raise FileNotFoundError(f"Model configuration file {model_cfg_path} not found")
        model_cfg = OmegaConf.load(model_cfg_path)
        cfg["model"] = model_cfg["model"]
        cfg["encoder"] = model_cfg["encoder"]
        cfg["dataset"] = model_cfg["dataset"]
        model = MODEL_FACTORY.get(cfg.model.name)(cfg).to(device)

        if checkpoint_path.endswith(('.pth', '.bin')):
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(state_dict)
        else:
            raise ValueError("Supports .pth or .bin format model files")
        model.eval()
        run_flag+=1

        preds_list = []
        label_list = [] if run_flag == 1 else None

        with torch.no_grad():
            for sentence_index in tqdm(range(len(ieeg))):
                sentence_dataset = []
                sentence_ieeg = ieeg[sentence_index]
                sentence_label = label[sentence_index]
                sentence_ieeg = np.swapaxes(sentence_ieeg, 0, 1)

                for i in range(len(sentence_ieeg)):
                    data_dict = {
                        'ieeg_raw_data': torch.tensor(sentence_ieeg[i]),
                        'labels'       : torch.tensor(sentence_label[i]),
                    }
                    sentence_dataset.append(data_dict)

                sentence_data_loader = torch.utils.data.DataLoader(
                    SimpleDataset(sentence_dataset),
                    batch_size=cfg.test.batch_size,
                    collate_fn=speech_decoding_collate_fn,
                    shuffle=False,
                )

                for batch in sentence_data_loader:
                    inputs = batch['ieeg_raw_data'].to(device)
                    labels = batch['labels'].to(device)
                    outputs = model(inputs, labels)
                    probs = torch.softmax(outputs['logits'], dim=1).cpu()
                    label_list.append(labels.cpu()) if run_flag == 1 else None
                    preds_list.append(probs)

        if run_flag == 1:
            all_labels = label_list

        model_weight = cfg.test.model_weight[run_flag - 1]

        if all_pred is None:
            all_pred = preds_list
        else:
            for i in range(len(preds_list)):
                all_pred[i] += preds_list[i] * model_weight
    return all_labels, all_pred


def translate_label2text_sen(cfg, subject, state, syllable_pred_label=None, initial_pred_label=None, final_pred_label=None, tone_pred_label=None, actual_initial_label=None, actual_final_label=None, actual_tone_label=None, beam_size=10, sentence_candidate=5, is_extra_lexicon=False, decode_with_tone=False):
    prediction_list = []
    label_sentence_list = []
    label_pinyin_list = []
    vocab_list = np.load(f"{cfg.dir.llm_utils_dir}/vocab_list.pkl", allow_pickle=True)
    vocab_list_with_tone = np.load(f"{cfg.dir.llm_utils_dir}/vocab_list_with_tone.pkl", allow_pickle=True)
    sentence_list = np.load(f'{cfg.dir.data_dir}/{subject}/processed_data/sentence/sentence_label_{state}.pkl', allow_pickle=True)
    if syllable_pred_label == None and decode_with_tone == False:
        for i in range(len(initial_pred_label)):
            initial_pred = initial_pred_label[i]
            final_pred = final_pred_label[i]
            initial_label = actual_initial_label[i]
            initial_text = [INITIAL_MAP[k] for k in initial_label]
            final_label = actual_final_label[i]
            final_text = [FINAL_MAP[k] for k in final_label]
            label_pinyin_str = " ".join([initial_text[k]+final_text[k] for k in range(len(initial_text))])
            best_prediction_str = beam_search_if_with_lexicon(vocab_list,
                                                      lexicon=f"{cfg.dir.data_dir}/initial_final_lexicon.txt" if is_extra_lexicon ==  False else f"{cfg.dir.data_dir}/initial_final_extra_lexicon.txt",
                                                      initial_map=INITIAL_MAP,
                                                      final_map=FINAL_MAP,
                                                      prob_initial=initial_pred,
                                                      prob_final=final_pred,
                                                      beam_size=beam_size,
                                                      top_k=sentence_candidate)
            if len(best_prediction_str) != sentence_candidate:
                for j in range(sentence_candidate - len(best_prediction_str)):
                    best_prediction_str.append(" ")

            prediction_list.extend(best_prediction_str)
            label_sentence_list.append(sentence_list[i][0])

            legal_syllable_list = label_pinyin_str.split(" ")
            for k in range(len(legal_syllable_list)):
                sub_str = legal_syllable_list[k]
                if sub_str in NORMAL_SYLLABLE_MAP:
                    legal_syllable_list[k] = NORMAL_SYLLABLE_MAP[sub_str]
            label_pinyin_list.append(" ".join(legal_syllable_list))

    elif syllable_pred_label == None and decode_with_tone == True:
        for i in range(len(initial_pred_label)):
            initial_pred = initial_pred_label[i]
            final_pred = final_pred_label[i]
            tone_pred = tone_pred_label[i]
            initial_label = actual_initial_label[i]
            initial_text = [INITIAL_MAP[k] for k in initial_label]
            final_label = actual_final_label[i]
            final_text = [FINAL_MAP[k] for k in final_label]
            tone_label = actual_tone_label[i]
            tone_text = [TONE_MAP[k] for k in tone_label]
            label_pinyin_str = " ".join([initial_text[k]+final_text[k]+tone_text[k] for k in range(len(initial_text))])

            best_prediction_str = beam_search_if_with_tone_and_lexicon(vocab_list_with_tone,
                                                      lexicon=f"{cfg.dir.data_dir}/initial_final_tone_lexicon.txt" if is_extra_lexicon ==  False else f"{cfg.dir.data_dir}/initial_final_tone_extra_lexicon.txt",
                                                      initial_map=INITIAL_MAP,
                                                      final_map=FINAL_MAP,
                                                      tone_map=TONE_MAP,
                                                      prob_initial=initial_pred,
                                                      prob_final=final_pred,
                                                      prob_tone=tone_pred,
                                                      beam_size=beam_size,
                                                      top_k=sentence_candidate)

            if len(best_prediction_str) != sentence_candidate:
                for j in range(sentence_candidate - len(best_prediction_str)):
                    best_prediction_str.append(" ")

            prediction_list.extend(best_prediction_str)
            label_sentence_list.append(sentence_list[i][0])

            legal_syllable_list = label_pinyin_str.split(" ")
            for k in range(len(legal_syllable_list)):
                sub_str = legal_syllable_list[k][:-1]
                if sub_str in NORMAL_SYLLABLE_MAP:
                    legal_syllable_list[k] = NORMAL_SYLLABLE_MAP[sub_str]+legal_syllable_list[k][-1]
            label_pinyin_list.append(" ".join(legal_syllable_list))


    return prediction_list, label_pinyin_list, label_sentence_list


def speech_decoding_collate_fn(instances):
    ieeg_raw_data, labels = tuple([instance[key] for instance in instances]
                                 for key in ("ieeg_raw_data", "labels"))
    ieeg_raw_data = torch.stack(ieeg_raw_data)
    labels = torch.tensor(labels)

    batch = {
        "ieeg_raw_data": ieeg_raw_data,
        "labels"       : labels,
        "return_loss"  : True,
    }
    return batch


@hydra.main(config_path="../../../conf", config_name="llm_inference", version_base="1.2")
def main(cfg: DictConfig):
    # Initial settings
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{cfg.test.gpu_id}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.test.seed)
    print("---------------------Start-------------------------")
    start_time = time.time()
    processed_dir = f"{cfg.dataset.data_dir}/{cfg.dataset.id}/processed_data/sentence"
    initial_final_results_save_path = cfg.results.initial_final_results_path
    beam_search_output_path = cfg.beam_search.output_path
    initial_final_results = {}
    if cfg.test.method == 'if':
        exp_category = cfg.test.state
        test_input_dataset_path = Path(processed_dir) / f'data_{exp_category}.pkl'
        test_input_dataset = np.load(test_input_dataset_path, allow_pickle=True)

        initial_test_label_dataset_path = Path(processed_dir) / f'initial_label_{exp_category}.pkl'
        initial_test_label_dataset = np.load(initial_test_label_dataset_path, allow_pickle=True)

        final_test_label_dataset_path = Path(processed_dir) / f'final_label_{exp_category}.pkl'
        final_test_label_dataset = np.load(final_test_label_dataset_path, allow_pickle=True)

        if cfg.test.decode_with_tone:
            tone_test_label_dataset_path = Path(processed_dir) / f'tone_label_{exp_category}.pkl'
            tone_test_label_dataset = np.load(tone_test_label_dataset_path, allow_pickle=True)

        cfg.dataset.task = f"{exp_category}_initial"
        all_labels_initial, all_pred_initial = inference_multi_model(cfg, device, test_input_dataset, initial_test_label_dataset, "initial")
        os.makedirs(Path(f'{cfg.dir.data_dir}/{cfg.dataset.id}/prediction_logits'), exist_ok=True)

        cfg.dataset.task = f"{exp_category}_final"
        all_labels_final, all_pred_final = inference_multi_model(cfg, device, test_input_dataset, final_test_label_dataset, "final")

        if cfg.test.decode_with_tone:
            cfg.dataset.task = f"{exp_category}_tone"
            all_labels_tone, all_pred_tone = inference_multi_model(cfg, device, test_input_dataset,
                                                                     tone_test_label_dataset, "tone")

        initial_final_results["initial_logits"] = [tensor.tolist() for tensor in all_pred_initial]
        initial_final_results["initial_predictions"] = [tensor.tolist() for tensor in all_labels_initial]
        initial_final_results["final_logits"] = [tensor.tolist() for tensor in all_pred_final]
        initial_final_results["final_predictions"] = [tensor.tolist() for tensor in all_labels_final]

        if cfg.test.decode_with_tone:
            initial_final_results["tone_logits"] = [tensor.tolist() for tensor in all_pred_tone]
            initial_final_results["tone_predictions"] = [tensor.tolist() for tensor in all_labels_tone]

        end_time = time.time()
        print(f"Inference time: {end_time - start_time}")

        eval_pred_initial = EvalPred(predictions=(all_labels_initial, all_pred_initial))
        eval_pred_final = EvalPred(predictions=(all_labels_final, all_pred_final))
        eval_pred_tone = EvalPred(
            predictions=(all_labels_tone, all_pred_tone)) if cfg.test.decode_with_tone else None

        initial_results = simple_compute_metrics(eval_pred_initial, INITIAL_MAP)
        final_results = simple_compute_metrics(eval_pred_final, FINAL_MAP)
        tone_results = simple_compute_metrics(eval_pred_tone, TONE_MAP) if cfg.test.decode_with_tone else None

        for metric in initial_results.keys():
            initial_final_results[f'initial_{metric}'] = round(initial_results[metric], 4)
            print(f"Initial {metric}: {initial_results[metric]:.4f}")
        for metric in final_results.keys():
            initial_final_results[f'final_{metric}'] = round(final_results[metric], 4)
            print(f"Final {metric}: {final_results[metric]:.4f}")

        if cfg.test.decode_with_tone:
            for metric in tone_results.keys():
                initial_final_results[f'tone_{metric}'] = round(tone_results[metric], 4)
                print(f"Tone {metric}: {tone_results[metric]:.4f}")

        print("Translating label to text...")
        start_time = time.time()
        prediction_list, label_pinyin_list, label_sentence_list = translate_label2text_sen(cfg,
                                                                                           cfg.test.subject,
                                                                                           cfg.test.state,
                                                                                           initial_pred_label=all_pred_initial,
                                                                                           final_pred_label=all_pred_final,
                                                                                           tone_pred_label=all_pred_tone if cfg.test.decode_with_tone else None,
                                                                                           actual_initial_label=all_labels_initial,
                                                                                           actual_final_label=all_labels_final,
                                                                                           actual_tone_label=all_labels_tone if cfg.test.decode_with_tone else None,
                                                                                           beam_size=cfg.test.beam_size,
                                                                                           sentence_candidate=cfg.test.sentence_candidate,
                                                                                           is_extra_lexicon=cfg.test.is_extra_lexicon,
                                                                                           decode_with_tone=cfg.test.decode_with_tone)
        end_time = time.time()
        print(f"Translating time: {end_time - start_time}")

    beam_search_results = []
    for i in range(len(label_sentence_list)):
        beam_search_results_single = {}
        beam_search_results_single['output'] = label_sentence_list[i]
        beam_search_results_single['output_syllable'] = label_pinyin_list[i]
        beam_search_results_single['input'] = prediction_list[i * cfg.test.sentence_candidate:(i + 1) * cfg.test.sentence_candidate]
        beam_search_results.append(beam_search_results_single)

    total_exact_match = 0
    total_min_accuracy_list = []
    total_mean_accuracy_list = []
    total_accuracy_list = []
    all_error_mappings = []
    for i in range(len(beam_search_results)):
        accuracy_list, exact_match, error_mapping = calculate_char_accuracy(beam_search_results[i]['output_syllable'], beam_search_results[i]['input'])
        total_min_accuracy_list.append(min(accuracy_list))
        total_mean_accuracy_list.append(average(accuracy_list))
        total_exact_match += exact_match
        total_accuracy_list.append(accuracy_list)
        all_error_mappings.append(error_mapping)

    distribution = calculate_value_distribution(total_accuracy_list)
    for k, v in distribution.items():
        initial_final_results[f'error rate {k}'] = f"{v:.2%}"  # Format as percentage
        print(f"{k}: {v:.2%}")

    exact_match_rate = total_exact_match / len(beam_search_results)
    min_error_rate = average(total_min_accuracy_list)
    mean_error_rate = average(total_mean_accuracy_list)

    # Record to initial_final_results and format as percentage
    initial_final_results['syllable exact match'] = f"{exact_match_rate:.2%}"
    initial_final_results['syllable min error'] = f"{min_error_rate:.2%}"
    initial_final_results['syllable mean error'] = f"{mean_error_rate:.2%}"

    print("Exact match: {:.2%}".format(exact_match_rate))
    print("Min error: {:.2%}".format(min_error_rate))
    print("Mean error: {:.2%}".format(mean_error_rate))

    if cfg.test.decode_with_tone == True:
        beam_search_output_path = beam_search_output_path.replace('.pkl', '_with_tone.pkl')

    Path(beam_search_output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(beam_search_output_path, 'wb') as f:
        pickle.dump(beam_search_results, f)

    print("Saved beam search results in", beam_search_output_path)

    if cfg.test.decode_with_tone == True:
        initial_final_results_save_path = initial_final_results_save_path.replace('.json', '_with_tone.json')

    Path(initial_final_results_save_path).parent.mkdir(parents=True, exist_ok=True)

    with open(initial_final_results_save_path, 'w') as f:
        json.dump(initial_final_results, f)

    print("Saved initial_final_results in", initial_final_results_save_path)

if __name__ == "__main__":
    main()