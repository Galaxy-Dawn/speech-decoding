import os
import re
import json
import regex
from tqdm import tqdm
from pypinyin import lazy_pinyin, Style, pinyin
from opencc import OpenCC
import hydra


def is_simplified_chinese_only(text):
    pattern = r'^[\u4e00-\u9fa5]+$'
    return re.fullmatch(pattern, text) is not None


def contains_english_or_number(text):
    en_or_num = bool(re.search(r'[a-zA-Z0-9\uFF10-\uFF19]', text))
    syb = bool(re.search(r'[\u2460-\u24FF]', text))
    return en_or_num or syb


def preprocess_sentence(text: str):
    text_w_punc = re.sub(r'[\s_]+', '', text)  # remove space and '_
    text_wo_punc = regex.sub(r'[\p{P}\p{S}]', '', text_w_punc)  # remove punctuation
    return text_w_punc, text_wo_punc


def preprocess_text(text_list: list):
    process_text_list = []
    cc = OpenCC('t2s')
    for sentence in tqdm(text_list):
        sentence = cc.convert(sentence)
        text_w_punc, text_wo_punc = preprocess_sentence(sentence)
        if not is_simplified_chinese_only(text_wo_punc):
            continue
        initials = lazy_pinyin(text_wo_punc, style=Style.INITIALS)
        initials = ['-' if i == '' else i for i in initials]
        finals = lazy_pinyin(text_wo_punc, style=Style.FINALS)
        pinyin_list = lazy_pinyin(text_wo_punc)
        tone_pinyin_list = [item[0] for item in pinyin(text_wo_punc, style=Style.TONE)]

        content = {
            'text': text_w_punc,
            'text_wo_punc': text_wo_punc,
            'initials': ' '.join(initials),
            'finals': ' '.join(finals),
            'syllables': ' '.join(pinyin_list),
            'tone_syllables': ' '.join(tone_pinyin_list)
        }
        process_text_list.append(content)
    return process_text_list


@hydra.main(config_path="../../../conf", config_name="llm_training", version_base="1.2")
def nlpcc(cfg):
    train_data = open(os.path.join(cfg.dir.llm_data_dir, 'nlpcc/train/data.train'), 'r').readlines()
    test_data = open(os.path.join(cfg.dir.llm_data_dir, 'nlpcc/test/source.txt'), 'r').readlines()
    test_sentence = [line.strip() for line in test_data]
    train_sentence = []
    for line in train_data:
        line = line.strip().split('\t')
        num_correct = int(line[1])
        if num_correct == 0:
            train_sentence.append(line[2])
        else:
            train_sentence += line[3:]
    train_sentence = preprocess_text(train_sentence)
    test_sentence = preprocess_text(test_sentence)
    if not os.path.exists(os.path.join(cfg.dir.llm_data_dir, 'nlpcc')):
        os.makedirs(os.path.join(cfg.dir.llm_data_dir, 'nlpcc'))
    json.dump(train_sentence, open(os.path.join(cfg.dir.llm_data_dir, 'nlpcc/train_sentence.json'), 'w'))
    json.dump(test_sentence, open(os.path.join(cfg.dir.llm_data_dir, 'nlpcc/test_sentence.json'), 'w'))


@hydra.main(config_path="../../../conf", config_name="llm_training", version_base="1.2")
def sighan2015(cfg):
    train_data = open(os.path.join(cfg.dir.llm_data_dir, 'sighan2015/train.jsonl'), 'r')
    test_data = open(os.path.join(cfg.dir.llm_data_dir, 'sighan2015/test.jsonl'), 'r')
    train_sentence = [json.loads(line)['correct_text'] for line in train_data]
    test_sentence = [json.loads(line)['correct_text'] for line in test_data]
    train_sentence = preprocess_text(train_sentence)
    test_sentence = preprocess_text(test_sentence)
    if not os.path.exists(os.path.join(cfg.dir.llm_data_dir, 'sighan2015')):
        os.makedirs(os.path.join(cfg.dir.llm_data_dir, 'sighan2015'))
    json.dump(train_sentence, open(os.path.join(cfg.dir.llm_data_dir, 'sighan2015/train_sentence.json'), 'w'))
    json.dump(test_sentence, open(os.path.join(cfg.dir.llm_data_dir, 'sighan2015/test_sentence.json'), 'w'))


if __name__ == '__main__':
    # process the raw data downloaded from the internet
    nlpcc()
    sighan2015()
