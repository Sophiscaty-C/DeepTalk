# -*- coding:utf-8 -*-

import os
import torch
import itertools
from torch.utils import data as data_
from config import getConfig


gConfig = {}
gConfig = getConfig.get_config()


def zeroPadding(l, fillvalue):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))


def binaryMatrix(l, value):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == value:
                m[i].append(0)
            else:
                m[i].append(1)
    return m


def create_collate_fn(padding):
    def collate_fn(corpus_item):

        corpus_item.sort(key=lambda p: len(p[0]), reverse=True) 
        inputs, targets, indexes = zip(*corpus_item)
        input_lengths = torch.tensor([len(question) for question in inputs])
        inputs = zeroPadding(inputs, padding)
        inputs = torch.LongTensor(inputs)
        
        max_target_length = max([len(answer) for answer in targets])
        targets = zeroPadding(targets, padding)
        mask = binaryMatrix(targets, padding)
        mask = torch.ByteTensor(mask)
        targets = torch.LongTensor(targets)
        
        return inputs, targets, mask, input_lengths, max_target_length, indexes

    return collate_fn


class CorpusDataset(data_.Dataset):

    def __init__(self):
        self._data = torch.load(gConfig["chatbot_cooked_data"])
        self.word2idx = self._data['word2idx']
        self.idx2word = self._data['idx2word']
        self.idx_corpus = self._data['idx_corpus']
        self.sos = self.word2idx.get(gConfig["start_of_string"])
        self.eos = self.word2idx.get(gConfig["end_of_string"])
        self.padding = self.word2idx.get(gConfig["padding"])
        self.unknown = self.word2idx.get(gConfig["unknown"])
        
    def __getitem__(self, index):
        question = self.idx_corpus[index][0]
        answer = self.idx_corpus[index][1]
        return question, answer, index

    def __len__(self):
        return len(self.idx_corpus)


def get_dataloader():
    dataset = CorpusDataset()
    dataloader = data_.DataLoader(dataset,
                                batch_size=gConfig["batch_size"],
                                shuffle=gConfig["dataloader_shuffle"],
                                num_workers=gConfig["dataloader_num_workers"],
                                drop_last=True, 
                                collate_fn=create_collate_fn(dataset.padding))
    return dataloader


