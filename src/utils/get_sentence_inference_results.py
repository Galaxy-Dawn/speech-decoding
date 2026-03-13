import numpy as np
from pypinyin import lazy_pinyin
import jieba
import regex
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_chinese import Rouge
import Levenshtein


def normalize_pronouns(text: str) -> str:
    """
    Replace "她" (she) and "它" (it) with "他" (he) in the text
    Args:
        text: Input text
    Returns:
        Text with pronouns normalized
    """
    return text.replace('她', '他').replace('它', '他')


def tokenize(text: str):
    return list(jieba.cut(text))


def compute_bleu(reference: str, hypothesis: str) -> float:
    reference_tokens = list(reference) # tokenize(reference)
    hypothesis_tokens = list(hypothesis) # tokenize(hypothesis)
    smooth_fn = SmoothingFunction().method3
    score = sentence_bleu([reference_tokens], hypothesis_tokens, smoothing_function=smooth_fn)
    return score


def compute_rouge(reference: str, hypothesis: str) -> dict:
    if hypothesis == '':
        return {'rouge-1': {'f': 0.0}, 'rouge-2': {'f': 0.0}, 'rouge-l': {'f': 0.0}}
    rouge = Rouge()
    reference_tokens = ' '.join(tokenize(reference))
    hypothesis_tokens = ' '.join(tokenize(hypothesis))
    scores = rouge.get_scores(hypothesis_tokens, reference_tokens)
    return scores[0]  # Only one sentence pair, take the first item


def compute_cer(reference: str, hypothesis: str) -> float:
    distance = Levenshtein.distance(reference, hypothesis)
    return distance / len(reference)


def get_result_dict(results, correction_is_permutation=False):
    result_dict = {}
    sentence_idx = range(len(results[2]))
    for index, sentence_id in enumerate(sentence_idx):
        initial_logits = np.array(results[0]['initial_logits'][sentence_id]) if not correction_is_permutation else np.array(results[0]['initial_logits'][sentence_id // 6])
        initial_pred = np.argmax(initial_logits, axis=1)
        initial_label = np.array(results[0]['initial_predictions'][sentence_id]) if not correction_is_permutation else np.array(results[0]['initial_predictions'][sentence_id // 6])
        final_logits = np.array(results[0]['final_logits'][sentence_id]) if not correction_is_permutation else np.array(results[0]['final_logits'][sentence_id // 6])
        final_pred = np.argmax(final_logits, axis=1)
        final_label = np.array(results[0]['final_predictions'][sentence_id]) if not correction_is_permutation else np.array(results[0]['final_predictions'][sentence_id // 6])

        syllable_to_sentence_result = results[2][index]
        label = regex.sub(r'[\p{P}\p{S}]', '', syllable_to_sentence_result['label'])
        assert len(label) == len(initial_pred) == len(final_pred) == len(initial_label) == len(final_label)
        pinyin_label = ' '.join(lazy_pinyin(label))
        top20 = syllable_to_sentence_result['beam search output']
        top20_match, top3_match = 0, 0
        for item in top20:
            if item == pinyin_label:
                top20_match += 1
        top3 = syllable_to_sentence_result['listwise output']
        for item in top3:
            if item == pinyin_label:
                top3_match += 1
        output = regex.sub(r'[\p{P}\p{S}]', '', syllable_to_sentence_result['correction output'])
        cer = compute_cer(normalize_pronouns(label), normalize_pronouns(output))
        item = {
                'idx': index,
                'label': label,
                'top20_match': top20_match,
                'top3_match': top3_match,
                'output': output,
                'cer': cer,
                'initial_pred': initial_pred,
                'final_pred': final_pred,
                'initial_label': initial_label,
                'final_label': final_label,
                'beam_search_output': top20,
                'listwise output': top3,
        }
        if label not in result_dict:
            result_dict[label] = [item]
        else:
            result_dict[label].append(item)

    return result_dict


def get_predicts(result_dict):
    all_predicts = []
    for label, predicts in result_dict.items():
        all_predicts += predicts
    return all_predicts


def get_sentence_results(results, correction_is_permutation=False):
    result_dict = get_result_dict(results, correction_is_permutation=correction_is_permutation)
    all_predicts = get_predicts(result_dict)
    cer, bleu, rouge1, rouge2, rougel = 0, 0, 0, 0, 0
    initial_pred, initial_label, final_pred, final_label = [], [], [], []

    beam_search_results = []
    for item in all_predicts:
        beam_search_results.append({
            'beam_search_output': item['beam_search_output'],
            'label': item['label'],
        })
        cer += item['cer']
        _bleu = compute_bleu(normalize_pronouns(item['label']), normalize_pronouns(item['output']))
        _rouge = compute_rouge(normalize_pronouns(item['label']), normalize_pronouns(item['output']))
        bleu += _bleu
        rouge1 += _rouge['rouge-1']['f']
        rouge2 += _rouge['rouge-2']['f']
        rougel += _rouge['rouge-l']['f']
        initial_pred.append(item['initial_pred'])
        initial_label.append(item['initial_label'])
        final_pred.append(item['final_pred'])
        final_label.append(item['final_label'])

    initial_pred = np.concatenate(initial_pred)
    initial_label = np.concatenate(initial_label)
    final_pred = np.concatenate(final_pred)
    final_label = np.concatenate(final_label)
    initial_acc = np.mean(initial_pred == initial_label)
    final_acc = np.mean(final_pred == final_label)
    metrics = {
        'initial_acc': initial_acc,
        'final_acc'  : final_acc,
        'cer'        : cer / len(all_predicts),
        'bleu'       : bleu / len(all_predicts),
        'rouge1'     : rouge1 / len(all_predicts),
        'rouge2'     : rouge2 / len(all_predicts),
        'rougel'     : rougel / len(all_predicts)
    }

    # Calculate maximum key length for alignment
    max_key_length = max(len(key) for key in metrics.keys())

    # Print with fixed width alignment
    for key, value in metrics.items():
        spaces = ' ' * (max_key_length - len(key) + 5)
        print(f"{key}{spaces}:{value:.4f}")

    return metrics, beam_search_results