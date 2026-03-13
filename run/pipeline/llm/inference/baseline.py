from Pinyin2Hanzi import DefaultDagParams, dag, simplify_pinyin
import regex
import Levenshtein
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import pandas as pd


def normalize_pronouns(text: str) -> str:
    return text.replace('她', '他').replace('它', '他')


def compute_cer(reference: str, hypothesis: str) -> float:
    reference = regex.sub(r'[\p{P}\p{S}]', '', reference)
    hypothesis = regex.sub(r'[\p{P}\p{S}]', '', hypothesis)
    distance = Levenshtein.distance(normalize_pronouns(reference), normalize_pronouns(hypothesis))
    return distance / len(reference)


def baseline(candidate_list, label_list):
    assert len(candidate_list) == len(label_list)
    params = DefaultDagParams()
    cer_list = []
    for (candidate, label) in zip(candidate_list, label_list):
        ori_pinyin = candidate.split(' ')
        pinyin = [simplify_pinyin(p) for p in ori_pinyin]
        pred = dag(dag_params=params, pinyin_list=pinyin, path_num=1)
        pred = ''.join(pred[0].path)
        pred_wo_punc = regex.sub(r'[\p{P}\p{S}]', '', pred)
        label_wo_punc = regex.sub(r'[\p{P}\p{S}]', '', label)
        cer = compute_cer(label_wo_punc, pred_wo_punc)
        cer_list.append(cer)
    return cer_list


@hydra.main(config_path="../../../conf", config_name="llm_inference", version_base="1.2")
def main(cfg: DictConfig):
    beam_search_results = np.load(cfg.beam_search.output_path, allow_pickle=True)
    syllable_list = []
    sentence_list = []

    for sentence_dict in beam_search_results:
        top1_candidate_syllable = sentence_dict['input'][0]
        true_sentence = str(sentence_dict['output'])
        syllable_list.append(top1_candidate_syllable)
        sentence_list.append(true_sentence)
        print(f"Top 1 candidate syllable: {top1_candidate_syllable}")
        print(f"True sentence: {true_sentence}")
        print("-" * 50)

    cer_list = baseline(syllable_list, sentence_list)
    print(f'Subject: {cfg.test.subject}, State: {cfg.test.state}, Seed: {cfg.test.seed}')
    print(f"CER: {np.average(cer_list)}")
    return cer_list


if __name__ == "__main__":
    main()
