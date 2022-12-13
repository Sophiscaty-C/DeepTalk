# -*- coding:utf-8 -*-
import os
import io
import sys
import time
import datetime
from tqdm import tqdm
import pickle
import jieba
import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import seq2seqModel
from config import getConfig


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gConfig = getConfig.get_config()
units = gConfig['layer_size']
hidden_size = gConfig['hidden_size']
BATCH_SIZE = gConfig['batch_size']
EOS_token = 1
SOS_token = 0
MAX_LENGTH = gConfig['max_length']


def get_timestamp():
    return "{0:%Y-%m-%dT%H-%M-%SW}".format(datetime.datetime.now())


def preprocess_sentence(w):
    w = 'start ' + w + ' end'
    # print(w)
    return w


def create_dataset(path, num_examples=None): 
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
    _num_examples = num_examples if num_examples else len(lines)
    pairs = [[preprocess_sentence(w) for w in l.split('\t')] for l in lines[:_num_examples]]
    input_lang = Lang("ans")
    output_lang = Lang("ask")
    pairs = [list(reversed(p)) for p in pairs]
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])

    return input_lang, output_lang, pairs, _num_examples


def max_length(tensor):
    return max(len(t) for t in tensor)


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "start", 1: "end"}
        self.n_words = 2  # Count start and end

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def read_data(path, num_examples=None):
    input_tensors = []
    target_tensors = []
    input_lang, target_lang, pairs, _num_examples = create_dataset(path, num_examples)
    print("Reading train dataset")
    with tqdm(range(_num_examples)) as pbar:
        for i in range(_num_examples):
            input_tensor = tensorFromSentence(input_lang, pairs[i][0])
            target_tensor = tensorFromSentence(target_lang, pairs[i][1])
            input_tensors.append(input_tensor)
            target_tensors.append(target_tensor)

            pbar.set_description(desc="{} / {}".format(i + 1, _num_examples))
            pbar.update(1)

    return input_tensors, input_lang, target_tensors, target_lang, _num_examples


def preprocess(mode='train'):
    preprocess_data_path = gConfig['preprocess_data_path']
    name_prefix = ['input_tensor', 'input_lang', 'target_tensor', 'target_lang', 'tensor_num']
    if len(os.listdir(preprocess_data_path)) != len(name_prefix):
        input_tensor, input_lang, target_tensor, target_lang, tensor_num = read_data(gConfig['chatbot_seq_data'], gConfig['max_train_data_size'])
        preprocess_data_list = [input_tensor, input_lang, target_tensor, target_lang, tensor_num]
        for i, data in enumerate(preprocess_data_list):
            prefix = name_prefix[i]
            if prefix.endswith('_tensor'):
                torch.save(data, os.path.join(preprocess_data_path, prefix + '.pt'))
            elif prefix.endswith('_lang'):
                data_file = open(os.path.join(preprocess_data_path, prefix + '.pkl'), 'wb')
                data_file.write(pickle.dumps(data))
                data_file.close()
            else:
                with open(os.path.join(preprocess_data_path, prefix + '.txt'), 'w') as f:
                    f.write(str(data))

    else:
        if mode == 'train':
            input_tensor = torch.load(os.path.join(preprocess_data_path, 'input_tensor.pt'))
            target_tensor = torch.load(os.path.join(preprocess_data_path, 'target_tensor.pt'))
        else: 
            assert mode == 'predict'
            input_tensor = 0
            target_tensor = 0

        input_lang = Lang('default')
        with open(os.path.join(preprocess_data_path, 'input_lang.pkl'), 'rb') as f:
            input_lang = pickle.loads(f.read())

        target_lang = Lang('default')
        with open(os.path.join(preprocess_data_path, 'target_lang.pkl'), 'rb') as f:
            target_lang = pickle.loads(f.read())

        with open(os.path.join(preprocess_data_path, 'tensor_num.txt'), 'r') as f:
            tensor_num = int(f.read())
    
    return input_tensor, input_lang, target_tensor, target_lang, tensor_num


def train():
    print("Preparing data in %s" % gConfig['train_data'])  
    
    input_tensor, input_lang, target_tensor, target_lang, tensor_num = preprocess('train')

    steps_per_epoch = len(input_tensor) // BATCH_SIZE
    print("Each epoch has {} steps, each step process {} conversion pairs".format(steps_per_epoch, BATCH_SIZE))
    
    checkpoint_dir = gConfig['model_data']
    checkpoint_prefix = os.path.join(checkpoint_dir, "model2.pt")
    
    start_time = time.time()
    encoder = seq2seqModel.Encoder(input_lang.n_words, hidden_size).to(device)
    decoder = seq2seqModel.AttentionDencoder(hidden_size, target_lang.n_words, dropout_p=0.1).to(device)
    if os.path.exists(checkpoint_prefix):
        checkpoint = torch.load(checkpoint_prefix)
        encoder.load_state_dict(checkpoint['modelA_state_dict'])
        decoder.load_state_dict(checkpoint['modelB_state_dict'])
    
    max_data = tensor_num
    print("Total {} conversion pairs".format(max_data))
    

    batch_loss = 1.
    local_average_loss = 1
    local_steps = 1000
    tmp_local_average_loss = 0.
    print("")

    writer = SummaryWriter(f"./log/chatbot_{get_timestamp()}/")
    global_step = 0
    epoch_num = 0
    epoch_limit = gConfig['epoch_limit']
    while batch_loss > gConfig['min_loss'] and epoch_num < epoch_limit:
        with tqdm(range(steps_per_epoch)) as t:
            
            start_time_epoch = time.time()
            total_loss = 0
            epoch_num += 1

            for i in range(0, steps_per_epoch):
                inp = input_tensor[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
                targ = target_tensor[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
                batch_loss = seq2seqModel.train_step(inp, targ, 
                                                    encoder, decoder, 
                                                    optim.SGD(encoder.parameters(), lr=0.001),
                                                    optim.SGD(decoder.parameters(), lr=0.01))
                
                global_step += 1
                total_loss += batch_loss
                local_average_loss += batch_loss
                if global_step % local_steps == 0:
                    tmp_local_average_loss = local_average_loss / local_steps
                    local_average_loss = 0
                
                t.set_description(desc="Epoch {} training".format(epoch_num))
                t.set_postfix(desc="训练总步数: {} 最新每步loss: {:.4f} 上{}步的平均loss: {:.4f}".format(i + 1, batch_loss, local_steps, tmp_local_average_loss))
                t.update(1)
                # print('训练总步数:{} 最新每步loss {:.4f}'.format(i, batch_loss))

                with torch.no_grad():
                    writer.add_scalar("train/loss", batch_loss, global_step=global_step)

            step_time_epoch = (time.time() - start_time_epoch) / steps_per_epoch
            step_time_total = (time.time() - start_time) / global_step
            
            epoch_loss = total_loss / steps_per_epoch
            
            print('训练总步数: {} 每步耗时: {}  最新每步耗时: {} 最新每步loss: {:.4f} 上一个epoch的loss: {:.4f}'.
                format(global_step, step_time_total, step_time_epoch, batch_loss, epoch_loss))
            torch.save({'modelA_state_dict': encoder.state_dict(),
                        'modelB_state_dict': decoder.state_dict()}, checkpoint_prefix)
            sys.stdout.flush()


def predict(sentence):
    max_length_tar = MAX_LENGTH

    _, input_lang, _, target_lang, _ = preprocess('predict')

    # print(input_lang.n_words, target_lang.n_words)

    encoder = seq2seqModel.Encoder(54089, hidden_size).to(device)
    decoder = seq2seqModel.AttentionDencoder(hidden_size, 62412, dropout_p=0.1).to(device)
    checkpoint_dir = gConfig['model_data']
    checkpoint_prefix = os.path.join(checkpoint_dir, "model2.pt")
    checkpoint = torch.load(checkpoint_prefix)
    encoder.load_state_dict(checkpoint['modelA_state_dict'])
    decoder.load_state_dict(checkpoint['modelB_state_dict'])

    sentence = preprocess_sentence(sentence)
    input_tensor = tensorFromSentence(input_lang, sentence)

    input_length = input_tensor.size()[0]
    result = ''
    max_length = MAX_LENGTH
    encoder_hidden = encoder.initHidden()
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                 encoder_hidden)
        encoder_outputs[ei] += encoder_output[0, 0]

    dec_input = torch.tensor([[SOS_token]], device=device)  # SOS

    dec_hidden = encoder_hidden
    # decoder_attentions = torch.zeros(max_length, max_length)
    for t in range(max_length_tar):
        predictions, dec_hidden, decoder_attentions = decoder(dec_input, dec_hidden, encoder_outputs)
        _, topi = predictions.data.topk(1)

        if topi.item() == EOS_token:
            break
        else:
            result += target_lang.index2word[topi.item()] + ' '
        dec_input = topi.squeeze().detach()
    return result


def main_predict(sentence):
    sentence = " ".join(jieba.cut(sentence))
    answer = predict(sentence)
    answer = answer.replace('_UNK', '^_^')
    answer_list = answer.strip().split()
    if answer_list[0] == 'start':
        del answer_list[0]
    if answer_list[-1] == 'end':
        del answer_list[-1]
    answer = ''.join(answer_list)

    print(answer)
    return answer

# if __name__ == '__main__':
#
#
#     s = '牛头不对马嘴'
#     main_predict(s)
