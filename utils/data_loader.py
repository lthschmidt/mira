import torch
import torch.utils.data as data
from torch.utils.data.sampler import SubsetRandomSampler
import random
import math
import os
import logging 
from utils import config
import pickle
from tqdm import tqdm
import pprint
pp = pprint.PrettyPrinter(indent=1)
import re
import ast
import time
#import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from model.common_layer import write_config
from utils.data_reader import load_dataset
from typing import Dict, List, Tuple

model_checkpoint = 'cointegrated/rubert-tiny-sentiment-balanced'
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)
if torch.cuda.is_available():
    model.cuda()

def get_sentiment(text):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True).to(model.device)
        proba = torch.sigmoid(model(**inputs).logits).cpu().numpy()[0]
        score = proba.dot([-1, 0, 1])  # аналог VADER compound
        return {
            'neg': float(proba[0]),
            'neu': float(proba[1]),
            'pos': float(proba[2]),
            'compound': float(score)
        }

class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, data: Dict, vocab):
        """Reads source and target sequences from txt files."""
        self.vocab = vocab
        self.data = data 
        self.emo_map = {
            'surprised': 0, 'excited': 1, 'annoyed': 2, 'proud': 3, 'angry': 4, 
            'sad': 5, 'grateful': 6, 'lonely': 7, 'impressed': 8, 'afraid': 9,
            'disgusted': 10, 'confident': 11, 'terrified': 12, 'hopeful': 13,
            'anxious': 14, 'disappointed': 15, 'joyful': 16, 'prepared': 17,
            'guilty': 18, 'furious': 19, 'nostalgic': 20, 'jealous': 21,
            'anticipating': 22, 'embarrassed': 23, 'content': 24, 'devastated': 25,
            'sentimental': 26, 'caring': 27, 'trusting': 28, 'ashamed': 29,
            'apprehensive': 30, 'faithful': 31
        }
        self.analyzer = get_sentiment


    def __len__(self) -> int:
        return len(self.data["target"])

    def __getitem__(self, index: int) -> Dict:
        """Returns one data pair (source and target)."""
        item = {}
        item["context_text"] = self.data["context"][index]
        item["target_text"] = self.data["target"][index]
        item["emotion_text"] = self.data["emotion"][index]

        item["context_emotion_scores"] = self.analyzer(' '.join(self.data["context"][index][0]))

        item["context"], item["context_mask"] = self.preprocess(item["context_text"])
        item["target"] = self.preprocess(item["target_text"], anw=True)
        item["emotion"], item["emotion_label"] = self.preprocess_emo(item["emotion_text"], self.emo_map)

        return item

    def preprocess(self, arr: List[str], anw: bool = False) -> Tuple[torch.Tensor, ...]:
        """Converts words to ids."""
        if anw:
            sequence = [
                self.vocab.word2index[word] if word in self.vocab.word2index else config.UNK_idx 
                for word in arr
            ] + [config.EOS_idx]
            return torch.tensor(sequence, dtype=torch.long)  # Changed from LongTensor
        else:
            X_dial = [config.CLS_idx]
            X_mask = [config.CLS_idx]
            for i, sentence in enumerate(arr):
                X_dial += [
                    self.vocab.word2index[word] if word in self.vocab.word2index else config.UNK_idx 
                    for word in sentence
                ]
                spk = self.vocab.word2index["USR"] if i % 2 == 0 else self.vocab.word2index["SYS"]
                X_mask += [spk for _ in range(len(sentence))]
            assert len(X_dial) == len(X_mask)

            return (
                torch.tensor(X_dial, dtype=torch.long),  # Changed from LongTensor
                torch.tensor(X_mask, dtype=torch.long)   # Changed from LongTensor
            )

    def preprocess_emo(self, emotion: str, emo_map: Dict[str, int]) -> Tuple[List[int], int]:
        program = [0]*len(emo_map)
        program[emo_map[emotion]] = 1
        return program, emo_map[emotion]

def collate_fn(data: List[Dict]) -> Dict:
    def merge(sequences: List[torch.Tensor]) -> Tuple[torch.Tensor, List[int]]:
        lengths = [len(seq) for seq in sequences]
        padded_seqs = torch.full((len(sequences), max(lengths)), fill_value=1, dtype=torch.long)  # Changed from ones
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths 

    data.sort(key=lambda x: len(x["context"]), reverse=True)
    item_info = {}
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]

    ## input
    input_batch, input_lengths = merge(item_info['context'])
    mask_input, _ = merge(item_info['context_mask'])

    ## Target
    target_batch, target_lengths = merge(item_info['target'])

    if torch.cuda.is_available() and config.USE_CUDA:  # More modern CUDA check
        input_batch = input_batch.cuda()
        mask_input = mask_input.cuda()
        target_batch = target_batch.cuda()
 
    return {
        "input_batch": input_batch,
        "input_lengths": torch.tensor(input_lengths, dtype=torch.long),  # Changed from LongTensor
        "mask_input": mask_input,
        "target_batch": target_batch,
        "target_lengths": torch.tensor(target_lengths, dtype=torch.long),  # Changed from LongTensor
        "target_program": item_info['emotion'],
        "program_label": item_info['emotion_label'],
        "input_txt": item_info['context_text'],
        "target_txt": item_info['target_text'],
        "program_txt": item_info['emotion_text'],
        "context_emotion_scores": item_info["context_emotion_scores"]
    }

def prepare_data_seq(batch_size: int = 8) -> Tuple[data.DataLoader, ...]:  
    pairs_tra, pairs_val, pairs_tst, vocab = load_dataset()

    logging.info(f"Vocab size: {vocab.n_words}")  # Modern f-string

    dataset_train = Dataset(pairs_tra, vocab)
    data_loader_tra = data.DataLoader(
        dataset=dataset_train,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn, 
        pin_memory=False
    )

    dataset_valid = Dataset(pairs_val, vocab)
    data_loader_val = data.DataLoader(
        dataset=dataset_valid,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=False
    )
    
    dataset_test = Dataset(pairs_tst, vocab)
    data_loader_tst = data.DataLoader(
        dataset=dataset_test,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=False
    )
    
    write_config()
    return data_loader_tra, data_loader_val, data_loader_tst, vocab, len(dataset_train.emo_map)