import json
import os
import numpy as np
import pickle
import jieba
from pypinyin import lazy_pinyin, Style, pinyin
import random
from tqdm import tqdm
from collections import defaultdict
from operator import itemgetter
from itertools import groupby
import string
import hydra


def generate_user_prompt(syllable):
    # Build complete prompt
    prompt = (
        f"拼音序列：{syllable}，中文语句："
    )
    return prompt


def generate_translation_dataset_from_json(data, system_prompt) -> list:
    alpaca_dataset = []
    for item in data:
        syllables = item.get('syllables', '')
        alpaca_entry = {
            "instruction": system_prompt,
            "input"      : generate_user_prompt(syllables),
            "output"     : item.get('text', '')
        }
        alpaca_dataset.append(alpaca_entry)
    return alpaca_dataset


def get_two_char_pinyin_words(data_root, dataset_name, train_or_test):
    two_char_pinyin_words = {}
    raw_sentence = json.load(open(os.path.join(data_root, f'{dataset_name}/{train_or_test}_sentence.json'), 'r'))
    for i, content in enumerate(tqdm(raw_sentence)):
        cut_words = jieba.lcut(content['text_wo_punc'])
        cut_words_pinyin = []
        for word in cut_words:
            word_pinyin = ' '.join(lazy_pinyin(word))
            cut_words_pinyin.append(word_pinyin)
            if len(word) == 2:
                if word_pinyin not in two_char_pinyin_words:
                    two_char_pinyin_words[word_pinyin] = {word}
                else:
                    two_char_pinyin_words[word_pinyin].add(word)
        raw_sentence[i]['cut_words'] = cut_words
        raw_sentence[i]['cut_words_pinyin'] = cut_words_pinyin
    json.dump(raw_sentence, open(os.path.join(data_root, f'{dataset_name}/{train_or_test}_sentence.json'), 'w'))
    pickle.dump(two_char_pinyin_words, open(os.path.join(data_root, f'{dataset_name}/two_char_pinyin_words.pkl'), 'wb'))


def group_by_pinyin(data_root, dataset_name):
    two_char_pinyin_words = pickle.load(
        open(os.path.join(data_root, f'{dataset_name}/two_char_pinyin_words.pkl'), 'rb'))
    two_char_pinyin = [p.split(' ') for p in two_char_pinyin_words.keys()]
    first_word_group, second_word_group = defaultdict(list), defaultdict(list)
    for s1, s2 in two_char_pinyin:
        first_word_group[s1].append(s1 + ' ' + s2)
        second_word_group[s2].append(s1 + ' ' + s2)
    first_word_group = dict(first_word_group)
    second_word_group = dict(second_word_group)
    pickle.dump(first_word_group, open(os.path.join(data_root, f'{dataset_name}/first_word_group.pkl'), 'wb'))
    pickle.dump(second_word_group, open(os.path.join(data_root, f'{dataset_name}/second_word_group.pkl'), 'wb'))


def _get_level(ratio):
    assert 0 <= ratio <= 1
    if ratio <= 0.3:
        level = 0
    elif ratio <= 0.6:
        level = 1
    else:
        level = 2

    return level


def _semantic_replace(syllables, first_words_group, second_words_group, min_ratio, max_ratio, min_replace_num,
                      all_syllables, no_replace_prob):
    # Convert all_syllables to numpy array if it's not already
    if not isinstance(all_syllables, np.ndarray):
        all_syllables = np.array(all_syllables)
    
    syllables_copy = syllables.copy()
    replaced_syllables = syllables.copy()
    ratio = np.random.uniform(min_ratio, max_ratio)
    replace_num = max(int(len(replaced_syllables) * ratio), min_replace_num)
    if min_ratio == 0 and np.random.uniform() < no_replace_prob:
        replace_num = 0
    for i in range(replace_num):
        replace_index = np.random.choice(len(replaced_syllables))
        original_syllables_word = replaced_syllables[replace_index]
        original_syllables_list = original_syllables_word.split(' ')
        # use whole-word replacement for 2-syllable words, otherwise single replacement
        if len(original_syllables_list) == 2:
            replace_syllables = None
            # randomly choose first or second word match
            if np.random.uniform() < 0.5:
                first_word = original_syllables_list[0]
                # if we have choices to replace, we choose a 2-syllable word to replace, else we randomly replace the first syllable
                if first_word in first_words_group:
                    replace_pool = first_words_group[first_word]
                    if original_syllables_word in replace_pool:
                        replace_pool.remove(original_syllables_word)
                    if len(replace_pool) > 0 and np.random.uniform() < 0.5:
                        replace_syllables = np.random.choice(list(replace_pool))
                if replace_syllables is None:
                    replace_syllables = np.random.choice(all_syllables).item() + ' ' + original_syllables_list[1]
            else:
                second_word = original_syllables_list[1]
                # if we have choices to replace, we choose a 2-syllable word to replace, else we randomly replace the second syllable
                if second_word in second_words_group:
                    replace_pool = second_words_group[second_word]
                    if original_syllables_word in replace_pool:
                        replace_pool.remove(original_syllables_word)
                    if len(replace_pool) > 0 and np.random.uniform() < 0.5:
                        replace_syllables = np.random.choice(list(replace_pool))
                if replace_syllables is None:
                    replace_syllables = original_syllables_list[0] + ' ' + np.random.choice(all_syllables).item()
        else:
            # we randomly replace one of the syllables
            inner_replace_index = np.random.choice(len(original_syllables_list))
            original_syllable = original_syllables_list[inner_replace_index]
            replace_pool = all_syllables[all_syllables != original_syllable]
            if len(replace_pool) == 0:
                # Fallback if no other syllables available
                new_syllable = original_syllable
            else:
                new_syllable = np.random.choice(replace_pool).item()
            original_syllables_list[inner_replace_index] = new_syllable
            replace_syllables = ' '.join(original_syllables_list)
        replaced_syllables[replace_index] = str(replace_syllables)

    syllables_copy = ' '.join(syllables_copy).split(' ')
    replaced_syllables = ' '.join(replaced_syllables).split(' ')
    replace_num = 0
    for i in range(len(replaced_syllables)):
        if replaced_syllables[i] != syllables_copy[i]:
            replace_num += 1
    return replaced_syllables, replace_num / len(syllables)


def _random_replace(syllables, min_ratio, max_ratio, min_replace_num, all_syllables, no_replace_prob):
    # Convert all_syllables to numpy array if it's not already
    if not isinstance(all_syllables, np.ndarray):
        all_syllables = np.array(all_syllables)
    
    ratio = np.random.uniform(min_ratio, max_ratio)
    replace_syllables = syllables.copy()
    replace_num = max(int(len(syllables) * ratio), min_replace_num)
    if min_ratio == 0 and np.random.uniform() < no_replace_prob:
        replace_num = 0

    if replace_num > 0:
        replace_index = random.sample(range(len(syllables)), replace_num)
        ready_to_replace_syllables = syllables[replace_index]
        mask = ~np.isin(all_syllables, ready_to_replace_syllables)
        replace_pool = all_syllables[mask]
        replace_syllables[replace_index] = np.random.choice(replace_pool, size=replace_num, replace=True)

    return replace_syllables, replace_num / len(syllables)


def replace_and_rank(data_list, first_word_group, second_word_group, ranges, levels, candidates_per_sample,
                     replace_method='semantic', all_syllables=None):
    all_candidates = []
    for content in tqdm(data_list):
        if replace_method == 'semantic':
            syllables = content['cut_words_pinyin']
        elif replace_method == 'random':
            syllables = np.array(content['syllables'].split(' '))
        else:
            raise ValueError('replace_method must be semantic or random')
        candidates = []
        weights = np.random.dirichlet([2, 2, 1])
        for _ in range(candidates_per_sample):
            level = random.choices(levels, weights=weights, k=1)[0]
            ran = ranges[level]
            if replace_method == 'semantic':
                replaced_syllables, ratio = _semantic_replace(syllables, first_word_group, second_word_group, ran[0],
                                                              ran[1], 1, all_syllables, 0.1)
            elif replace_method == 'random':
                replaced_syllables, ratio = _random_replace(syllables, ran[0], ran[1], 1, all_syllables, 0.1)
            candidates.append((' '.join(replaced_syllables), _get_level(ratio), ratio))
        unique_candidates_num = random.randint(candidates_per_sample // 2, candidates_per_sample)
        indices = random.sample(range(candidates_per_sample), unique_candidates_num)
        candidates = [candidates[i] for i in indices]
        while len(candidates) < candidates_per_sample:
            repeat_index = random.randint(0, unique_candidates_num - 1)
            elem = candidates[repeat_index]
            candidates.append(elem)
        sorted_candidates = sorted(candidates, key=itemgetter(2), reverse=False)
        ranked = []
        rank = 1
        for key, group in groupby(sorted_candidates, key=itemgetter(2)):
            group_items = list(group)
            for item in group_items:
                ranked.append(item + (rank,))
            rank += len(group_items)
        random.shuffle(ranked)
        all_candidates.append({'candidates': ranked, 'sentence': content['text']})

    return all_candidates


def candidates_generation(data_root, train_ratio, candidates_per_sample, train_set_name, replace_method, all_syllables):
    ranges = [[0, 0.3], [0.3, 0.6], [0.6, 0.9]]
    levels = [0, 1, 2]

    all_data = json.load(open(os.path.join(data_root, train_set_name, f'train_sentence.json'), 'r'))
    all_data_first_word_group = pickle.load(
        open(os.path.join(data_root, train_set_name, f'first_word_group.pkl'), 'rb'))
    all_data_second_word_group = pickle.load(
        open(os.path.join(data_root, train_set_name, f'second_word_group.pkl'), 'rb'))
    all_candidates = replace_and_rank(all_data, all_data_first_word_group, all_data_second_word_group, ranges, levels,
                                      candidates_per_sample, replace_method, all_syllables)
    random.shuffle(all_candidates)
    train_set, eval_set = all_candidates[:int(len(all_data) * train_ratio)], all_candidates[int(len(all_data) * train_ratio):]
    if not os.path.exists(os.path.join(data_root, 'candidates')):
        os.makedirs(os.path.join(data_root, 'candidates'))
    pickle.dump(train_set, open(os.path.join(data_root, f'candidates/train_set.pkl'), 'wb'))
    pickle.dump(eval_set, open(os.path.join(data_root, f'candidates/eval_set.pkl'), 'wb'))
    print('finish generating training and evaluation sets for rerank')


def get_rerank_data(data_set, system_prompt_listwise, topk):
    dataset_for_training = []
    for content in tqdm(data_set):
        candidates, sentence = content['candidates'], content['sentence']
        index_map = np.array(list(string.ascii_uppercase))[:len(candidates)]
        item = {}
        label = ''
        for k in range(1, topk + 1):
            for i, tup in enumerate(candidates):
                if index_map[i] not in label and tup[3] == k and len(label) < topk:
                    label += index_map[i]
        item['instruction'] = system_prompt_listwise
        item['input'] = generate_user_prompt_listwise([i[0] for i in candidates])
        item['output'] = label
        item['text'] = sentence
        dataset_for_training.append(item)
    return dataset_for_training


def generate_user_prompt_listwise(syllables_list):
    index_map = np.array(list(string.ascii_uppercase))[:len(syllables_list)]
    sequences = [
        f"{index_map[i]}：'{seq}'"
        for i, seq in enumerate(syllables_list)
    ]
    formatted_sequences = "\n".join(sequences)
    prompt = (
        f"{formatted_sequences}, 经过各个拼音序列选项的比较，正确的三个选项为："
    )
    return prompt


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


def get_demo_translation_data(demo_file_path, system_prompt):
    """
    Process demo dataset for translation task
    Args:
        demo_file_path: Path to demo corpus file
        system_prompt: System prompt for translation task
    Returns:
        translation_dataset: Generated translation dataset
        processed_data: Processed data for listwise use
    """
    # Read demo dataset
    demo_data = json.load(open(demo_file_path, 'r'))
    
    # Clean and process data
    processed_data = []
    for item in demo_data:
        # Extract required fields
        syllables = item.get('syllables', '')
        text = item.get('text', '')
        text_wo_punc = item.get('text_wo_punc', text)
        
        # Basic validation
        if syllables and text:
            processed_item = {
                'syllables': syllables,
                'text': text,
                'text_wo_punc': text_wo_punc
            }
            processed_data.append(processed_item)
    
    # Generate translation dataset
    translation_dataset = generate_translation_dataset_from_json(processed_data, system_prompt)
    return translation_dataset, processed_data


@hydra.main(config_path="../../../conf", config_name="llm_training", version_base="1.2")
def main(cfg):
    # Check if demo mode is enabled (default to False if not specified)
    demo_mode = getattr(cfg, 'demo_mode', False)
    
    if demo_mode:
        # Demo mode: use demo corpus for translation and listwise
        print("Running in demo mode, using demo corpus for translation and listwise...")
        
        # Path to demo corpus file (actual location)
        # Get project root directory by going up 5 levels from current file
        current_file = os.path.abspath(__file__)
        training_dir = os.path.dirname(current_file)
        llm_dir = os.path.dirname(training_dir)
        pipeline_dir = os.path.dirname(llm_dir)
        run_dir = os.path.dirname(pipeline_dir)
        project_root = os.path.dirname(run_dir)
        demo_file_path = os.path.join(project_root, 'data', 'demo_corpus_for_post_training.json')
        
        # Generate translation dataset from demo corpus
        translation_dataset, processed_data = get_demo_translation_data(demo_file_path, cfg.system_prompt_translation)
        
        # Create translation directory if it doesn't exist
        translation_dir = os.path.join(cfg.dir.llm_data_dir, 'translation')
        os.makedirs(translation_dir, exist_ok=True)
        
        # Save translation dataset
        translation_train_path = os.path.join(translation_dir, 'train_set.json')
        json.dump(translation_dataset, open(translation_train_path, 'w'))
        print(f"Generated translation dataset with {len(translation_dataset)} samples")
        
        # Add dataset info to llama factory
        add_dataset_info_to_llama_factory(cfg.dir.llm_data_dir, 'syllable_translation_uni', translation_train_path)
        print(f"Added demo translation dataset to llama factory")
        
        # Prepare for listwise ranking data generation (same process as normal mode)
        print("Generating listwise ranking data in demo mode...")
        
        # Create demo corpus directory for listwise processing
        demo_corpus_dir = os.path.join(cfg.dir.llm_data_dir, 'demo_corpus')
        os.makedirs(demo_corpus_dir, exist_ok=True)
        
        # Save demo data as train_sentence.json (required by get_two_char_pinyin_words)
        train_sentence_path = os.path.join(demo_corpus_dir, 'train_sentence.json')
        json.dump(processed_data, open(train_sentence_path, 'w'))
        
        # get data for listwise ranking - same process as normal mode
        get_two_char_pinyin_words(cfg.dir.llm_data_dir, 'demo_corpus', 'train')
        group_by_pinyin(cfg.dir.llm_data_dir, 'demo_corpus')
        candidates_generation(data_root=cfg.dir.llm_data_dir,
                              train_ratio=0.9,
                              candidates_per_sample=20,
                              train_set_name='demo_corpus',
                              replace_method='semantic',
                              all_syllables=cfg.all_syllables)

        # Load generated candidates
        train_set = pickle.load(open(os.path.join(cfg.dir.llm_data_dir, f'candidates/train_set.pkl'), 'rb'))
        eval_set = pickle.load(open(os.path.join(cfg.dir.llm_data_dir, f'candidates/eval_set.pkl'), 'rb'))

        # Generate rerank data
        listwise_train_set = get_rerank_data(train_set, cfg.system_prompt_listwise, 3)
        listwise_eval_set = get_rerank_data(eval_set, cfg.system_prompt_listwise, 3)

        # Create listwise directory if it doesn't exist
        listwise_dir = os.path.join(cfg.dir.llm_data_dir, 'listwise')
        os.makedirs(listwise_dir, exist_ok=True)
        
        # Save listwise data
        json.dump(listwise_train_set, open(os.path.join(listwise_dir, f'train_set.json'), 'w'))
        json.dump(listwise_eval_set, open(os.path.join(listwise_dir, f'eval_set.json'), 'w'))

        # Add dataset info to llama factory
        add_dataset_info_to_llama_factory(cfg.dir.llm_data_dir, 'syllable_direct_rerank_uni_train', f'{listwise_dir}/train_set.json')
        add_dataset_info_to_llama_factory(cfg.dir.llm_data_dir, 'syllable_direct_rerank_uni_eval', f'{listwise_dir}/eval_set.json')
        print("Listwise ranking data generation completed")
    else:
        # Normal mode: use original datasets (nlpcc and sighan)
        print("Running in normal mode, using original datasets...")
        
        # get data for translation
        all_sentence = []
        for dataset_name in ['nlpcc', 'sighan2015']:
            train_path = os.path.join(cfg.dir.llm_data_dir, f'{dataset_name}/train_sentence.json')
            if os.path.exists(train_path):
                train_sentence = json.load(open(train_path, 'rb'))
                for item in train_sentence:
                    sentence_length = len(item['text_wo_punc'])
                    if sentence_length <= 30:
                        all_sentence.append(item)
        translation_dataset = generate_translation_dataset_from_json(all_sentence, cfg.system_prompt_translation)
        
        # Create translation directory if it doesn't exist
        translation_dir = os.path.join(cfg.dir.llm_data_dir, 'translation')
        os.makedirs(translation_dir, exist_ok=True)
        
        # Save translation dataset
        json.dump(translation_dataset, open(os.path.join(translation_dir, f'train_set.json'), 'w'))
        add_dataset_info_to_llama_factory(cfg.dir.llm_data_dir, 'syllable_translation_uni', f'{translation_dir}/train_set.json')
        
        # Generate listwise ranking data
        print("Generating listwise ranking data...")
        
        # get data for listwise ranking
        get_two_char_pinyin_words(cfg.dir.llm_data_dir, 'unified_corpus', 'train')
        group_by_pinyin(cfg.dir.llm_data_dir, 'unified_corpus')
        candidates_generation(data_root=cfg.dir.llm_data_dir,
                              train_ratio=0.9,
                              candidates_per_sample=20,
                              train_set_name='unified_corpus',
                              replace_method='semantic',
                              all_syllables=cfg.all_syllables)

        train_set = pickle.load(open(os.path.join(cfg.dir.llm_data_dir, f'candidates/train_set.pkl'), 'rb'))
        eval_set = pickle.load(open(os.path.join(cfg.dir.llm_data_dir, f'candidates/eval_set.pkl'), 'rb'))

        listwise_train_set = get_rerank_data(train_set, cfg.system_prompt_listwise, 3)
        listwise_eval_set = get_rerank_data(eval_set, cfg.system_prompt_listwise, 3)

        # Create listwise directory if it doesn't exist
        listwise_dir = os.path.join(cfg.dir.llm_data_dir, 'listwise')
        os.makedirs(listwise_dir, exist_ok=True)
        
        json.dump(listwise_train_set, open(os.path.join(listwise_dir, f'train_set.json'), 'w'))
        json.dump(listwise_eval_set, open(os.path.join(listwise_dir, f'eval_set.json'), 'w'))

        add_dataset_info_to_llama_factory(cfg.dir.llm_data_dir, 'syllable_direct_rerank_uni_train', f'{listwise_dir}/train_set.json')
        add_dataset_info_to_llama_factory(cfg.dir.llm_data_dir, 'syllable_direct_rerank_uni_eval', f'{listwise_dir}/eval_set.json')
        print("Listwise ranking data generation completed")


if __name__ == '__main__':
    main()