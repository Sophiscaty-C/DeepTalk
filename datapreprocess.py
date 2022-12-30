# -*- coding:utf-8 -*-

import os
import logging
import jieba
from zhon.hanzi import punctuation
import re
import json
from tqdm import tqdm
import torch
from config import getConfig
jieba.setLogLevel(logging.INFO)

cop = re.compile("[^\u4e00-\u9fa5^a-z^A-Z^0-9]") # 分词处理正则

gConfig = {}
gConfig = getConfig.get_config()

SOS = gConfig["start_of_string"]
EOS = gConfig["end_of_string"]
UNK = gConfig["unknown"]
PAD = gConfig["padding"]

print("-------------Step 1/3-------------")
"""读取原始数据集并合并"""
train_data_path_prefix = gConfig["chatbot_resource_data"]
dataset_type_list = os.listdir(train_data_path_prefix)
convs = []
for dataset_type in dataset_type_list:
	dataset_path = os.path.join(train_data_path_prefix, dataset_type)
	if not os.path.isdir(dataset_path):
		continue
	suffix_list = os.listdir(dataset_path)
	for suffix in suffix_list:
		if not suffix.endswith('.json'):
			continue
		data_path = os.path.join(dataset_path, suffix)
		print("Processing dataset {}".format(data_path))
		with open(data_path, encoding='utf-8') as f:
			
			contents = json.loads(f.read())

			if type(contents) == dict:
				for i, (k, v) in enumerate(contents.items()):
					with tqdm(range(len(v))) as pbar:
						for l, conv in enumerate(v):
							one_conv = []
							for line in conv:
								# line = re.sub(r"[%s]+" %punctuation, "", line)
								one_conv.append(line)
							convs.append(one_conv)
							pbar.set_description("Processing {:.4%}(total {}) data of ({}/{})".format((l + 1)/len(v), len(v), i + 1, len(contents)))
							pbar.update(1)

			elif type(contents) == list:
				with tqdm(range(len(contents))) as pbar:
					for i, conv in enumerate(contents):
						one_conv = []
						for line in conv:
							# line = re.sub(r"[%s]+" %punctuation, "", line)
							one_conv.append(line)
						convs.append(one_conv)
						pbar.set_description("Processing {:.4%}(total {}) data".format((i + 1)/len(contents), len(contents)))
						pbar.update(1)
			else:
				continue

print("-------------Step 2/3-------------")
"""处理合并后的对话数据集"""
seq = []
max_s_len = gConfig["max_sentence_length"]

with tqdm(range(len(convs))) as pbar:
	for j, conv in enumerate(convs):
		if len(conv) == 1:
			continue
		if len(conv) % 2 != 0:
			conv = conv[:-1]
		for i in range(len(conv)):
			if i % 2 == 0:
				question = jieba.lcut(cop.sub("", conv[i]))[:max_s_len] + [EOS]
				answer = jieba.lcut(cop.sub("", conv[i]))[:max_s_len] + [EOS]
				seq.append([question, answer])
		pbar.set_description("Processing {:.4%} now {} total {}".format((j + 1)/len(convs), j + 1, len(convs)))
		pbar.update(1)

print("-------------Step 3/3-------------")
"""生成字典和句子索引"""
max_voc_length = gConfig["max_voc_length"]
min_word_freq = gConfig["min_word_frequency"]
word_frequency = {}

def update(word_freq):
    def fun(word):
        word_freq[word] = word_freq.get(word, 0) + 1
        return None
    return fun
lambda_ = update(word_frequency)

print("统计词频并排序...")
_ = {lambda_(word) for sentences in seq for sentence in sentences for word in sentence}
word_freq_list = sorted([(num, word) for word, num in word_frequency.items()], reverse=True)

print("生成词典...")
max_voc_length = max_voc_length if max_voc_length != -1 else len(word_freq_list)
words = [word[1] for word in word_freq_list[:max_voc_length] if word[0] >= min_word_freq]
words = [UNK, PAD, SOS] + words
word2idx = {word: ix for ix, word in enumerate(words)}
idx2word = {ix: word for word, ix in word2idx.items()}
idx_corpus = [[[word2idx.get(word, word2idx.get(UNK)) for word in sentence]
                    for sentence in item]
                    for item in seq]


cooked_data = {
	'idx_corpus': idx_corpus,
	'word2idx': word2idx,
	'idx2word': idx2word
}

cookeddata_path = gConfig["chatbot_cooked_data"]

torch.save(cooked_data, cookeddata_path)
print('处理完成的数据保存在路径 " {} " 下'.format(cookeddata_path))
