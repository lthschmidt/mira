import torch
import torch.utils.data as data
import os
import logging 
from utils import config
import pickle
from tqdm import tqdm
import numpy as np
import pprint
import nltk
from typing import Dict, List, Tuple, Any
import pathlib

pp = pprint.PrettyPrinter(indent=1)

class Lang:
    def __init__(self, init_index2word: Dict[int, str]):
        self.word2index = {str(v): int(k) for k, v in init_index2word.items()}
        self.word2count = {str(v): 1 for k, v in init_index2word.items()}
        self.index2word = init_index2word 
        self.n_words = len(init_index2word)
      
    def index_words(self, sentence: List[str]) -> None:
        for word in sentence:
            self.index_word(word.strip())

    def index_word(self, word: str) -> None:
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def clean(sentence: str) -> List[str]:
    """Tokenize and clean sentence using nltk"""
    sentence = sentence.lower()
    sentence = nltk.word_tokenize(sentence)
    return sentence

def load_npy_file(path: str) -> np.ndarray:
    """Safe loading of numpy files with version checking"""
    try:
        return np.load(path, allow_pickle=True)
    except ValueError as e:
        raise RuntimeError(f"Error loading {path}: {str(e)}")

def read_langs(vocab: Lang) -> Tuple[Dict[str, List], ...]:
    """Read and process dataset files with modern path handling"""
    dataset_dir = pathlib.Path('dataset')
    
    # Load all data files with error handling
    files = {
        'train': ['sys_dialog_texts', 'sys_target_texts', 'sys_emotion_texts', 'sys_situation_texts'],
        'dev': ['sys_dialog_texts', 'sys_target_texts', 'sys_emotion_texts', 'sys_situation_texts'],
        'test': ['sys_dialog_texts', 'sys_target_texts', 'sys_emotion_texts', 'sys_situation_texts']
    }
    
    loaded_data = {}
    for split in files:
        loaded_data[split] = {
            'context': load_npy_file(dataset_dir/f'sys_dialog_texts.{split}.npy'),
            'target': load_npy_file(dataset_dir/f'sys_target_texts.{split}.npy'),
            'emotion': load_npy_file(dataset_dir/f'sys_emotion_texts.{split}.npy'),
            'situation': load_npy_file(dataset_dir/f'sys_situation_texts.{split}.npy')
        }

    # Process data
    splits = {}
    for split in ['train', 'dev', 'test']:
        data_dict = {
            'context': [],
            'target': [],
            'emotion': [],
            'situation': []
        }
        
        # Process context (multi-turn dialogues)
        for context in loaded_data[split]['context']:
            u_list = []
            for u in context:
                cleaned = clean(u)
                u_list.append(cleaned)
                vocab.index_words(cleaned)
            data_dict['context'].append(u_list)
        
        # Process target
        for target in loaded_data[split]['target']:
            cleaned = clean(target)
            data_dict['target'].append(cleaned)
            vocab.index_words(cleaned)
        
        # Process situation
        for situation in loaded_data[split]['situation']:
            cleaned = clean(situation)
            data_dict['situation'].append(cleaned)
            vocab.index_words(cleaned)
        
        # Copy emotion as-is
        data_dict['emotion'] = loaded_data[split]['emotion'].tolist()
        
        # Validate lengths
        assert all(len(data_dict[k]) == len(data_dict['context']) for k in data_dict)
        splits[split] = data_dict
    
    return splits['train'], splits['dev'], splits['test'], vocab

def load_dataset() -> Tuple[Dict[str, List], ...]:
    """Load or build dataset with modern path handling"""
    cache_path = pathlib.Path('dataset/dataset_preproc.p')
    
    if cache_path.exists():
        logging.info("Loading preprocessed dataset")
        with open(cache_path, "rb") as f:
            data_tra, data_val, data_tst, vocab = pickle.load(f)
    else:
        logging.info("Building dataset...")
        vocab = Lang({
            config.UNK_idx: "UNK",
            config.PAD_idx: "PAD",
            config.EOS_idx: "EOS", 
            config.SOS_idx: "SOS",
            config.USR_idx: "USR",
            config.SYS_idx: "SYS",
            config.CLS_idx: "CLS"
        })
        
        data_tra, data_val, data_tst, vocab = read_langs(vocab)
        
        with open(cache_path, "wb") as f:
            pickle.dump((data_tra, data_val, data_tst, vocab), f)
            logging.info("Saved preprocessed dataset")
    
    # Print samples
    for i in range(min(1, len(data_tra['situation']))):
        print('[situation]:', ' '.join(data_tra['situation'][i]))
        print('[emotion]:', data_tra['emotion'][i])
        print('[context]:', [' '.join(u) for u in data_tra['context'][i]])
        print('[target]:', ' '.join(data_tra['target'][i]))
        print()
    
    return data_tra, data_val, data_tst, vocab