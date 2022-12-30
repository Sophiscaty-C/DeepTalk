# -*- coding: utf-8 -*-

import os
import sys
import re
import time
import datetime
import random
import jieba
import torch
import logging
import configparser
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from data_load import get_dataloader
from model import EncoderRNN, DecoderRNN
from greedysearch import GreedySearchDecoder
from config import getConfig
jieba.setLogLevel(logging.INFO)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

gConfig = {}
gConfig = getConfig.get_config()


def get_timestamp():
    return "{0:%Y-%m-%dT%H-%M-%SW}".format(datetime.datetime.now())


def maskNLLLoss(inp, target, mask):
    '''
    inp: shape [batch_size,voc_length]
    target: shape [batch_size] view ==> [batch_size, 1]
    mask: shape [batch_size]
    loss: 平均一个句子在t位置上的损失值
    '''
    nTotal = mask.sum()
    print(inp)
    print(target)
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    return loss, nTotal.item()


def batch_train(sos, data, encoder_optimizer, decoder_optimizer, encoder, decoder):
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    inputs, targets, mask, input_lengths, max_target_length, _ = data
    inputs = inputs.to(device)
    targets = targets.to(device)
    mask = mask.to(device)
    input_lengths =  input_lengths.to('cpu')

    loss = 0
    print_losses = []
    n_totals = 0

    '''
    inputs: shape [max_seq_len, batch_size]
    input_lengths: shape [batch_size]
    encoder_outputs: shape [max_seq_len, batch_size, hidden_size]
    encoder_hidden: shape [num_layers*num_directions, batch_size, hidden_size]
    decoder_input: shape [1, batch_size]
    decoder_hidden: decoder的初始hidden输入,是encoder_hidden取正方向
    '''
    encoder_outputs, encoder_hidden = encoder(inputs, input_lengths)
    decoder_input = torch.LongTensor([[sos for _ in range(gConfig["batch_size"])]])
    decoder_input = decoder_input.to(device)
    decoder_hidden = encoder_hidden[:decoder.num_layers]

    use_teacher_forcing = True if random.random() < gConfig["teacher_forcing_ratio"] else False

    if use_teacher_forcing:
        for t in range(max_target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
            decoder_input = targets[t].view(1, -1)

            mask_loss, nTotal = maskNLLLoss(decoder_output, targets[t], mask[t])
            mask_loss = mask_loss.to(device)
            loss += mask_loss

            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal
    else:
        for t in range(max_target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(gConfig["batch_size"])]])
            decoder_input = decoder_input.to(device)

            mask_loss, nTotal = maskNLLLoss(decoder_output, targets[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal

    loss.backward()

    _ = torch.nn.utils.clip_grad_norm_(encoder.parameters(), gConfig["grad_clip"])
    _ = torch.nn.utils.clip_grad_norm_(decoder.parameters(), gConfig["grad_clip"])

    encoder_optimizer.step()
    decoder_optimizer.step()

    return sum(print_losses) / n_totals 


def train():  
    dataloader = get_dataloader() 
    _data = dataloader.dataset._data 
    word2idx = _data['word2idx']
    sos = dataloader.dataset.sos
    voc_length = len(word2idx)
        
    encoder = EncoderRNN(voc_length)
    decoder = DecoderRNN(voc_length)

    if os.path.exists(gConfig["model_checkpoint"]):
        checkpoint = torch.load(gConfig["model_checkpoint"])
        encoder.load_state_dict(checkpoint['en'])
        decoder.load_state_dict(checkpoint['de'])
        
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    encoder.train()
    decoder.train()

    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=gConfig["learning_rate"])
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=gConfig["learning_rate"] * gConfig["decoder_learning_ratio"])
    if os.path.exists(gConfig["model_checkpoint"]):
        encoder_optimizer.load_state_dict(checkpoint['en_opt'])
        decoder_optimizer.load_state_dict(checkpoint['de_opt']) 

    conf = configparser.ConfigParser()
    config_path = gConfig["config_path"]
    conf.read(config_path)

    # 总训练 epoch 数
    total_epoch = gConfig["epoch"]
    
    # 每隔 local_iters 个 Iteration 计算一次平均 loss
    local_iters = 10
    # local_iters 个 Iteration 内 loss 的总值和平均值
    iters_loss = 0.
    iters_loss_avg = 0.
    
    # 一个 epoch 内 loss 的总值和平均值
    epoch_loss = 0.
    epoch_loss_avg = 0.
    
    # 最优 epoch 的平均 loss 值
    best_epoch_loss_avg = 0.

    # 每隔 save_epoch 个 epoch 保存一次模型
    save_epoch = 1

    writer = SummaryWriter(f"./log/chatbot_{get_timestamp()}/")
    global_step = 0

    for epoch in range(total_epoch):
        with tqdm(total=len(dataloader)) as t:
            for i, data in enumerate(dataloader):
                
                loss = batch_train(sos, data, encoder_optimizer, decoder_optimizer, encoder, decoder)
                iters_loss += loss
                epoch_loss += loss
                global_step += 1
  
                if (i+1) % local_iters == 0:
                    iters_loss_avg = iters_loss / local_iters
                    iters_loss = 0

                t.set_description(desc="Epoch {} training".format(epoch + 1))
                t.set_postfix(desc="已训练步数: {} 最新每步loss: {:.4f} 上{}步的平均loss: {:.4f}".format(i + 1, loss, local_iters, iters_loss_avg))
                t.update(1)

                with torch.no_grad():
                    writer.add_scalar("train/loss", loss, global_step=global_step)
            
        epoch_loss_avg = epoch_loss / len(dataloader)
        epoch_loss = 0.

        # 保存最佳模型
        if (best_epoch_loss_avg < epoch_loss_avg and epoch != 0) or epoch == 0:
            best_epoch_loss_avg = epoch_loss_avg
            print("目前最佳模型在第 {} 个 epoch, epoch_loss_average = {:.4f}".format(epoch + 1, best_epoch_loss_avg))
            best_model_path = gConfig["best_model_path"]
            torch.save({
                'en': encoder.state_dict(),
                'de': decoder.state_dict(),
                'en_opt': encoder_optimizer.state_dict(),
                'de_opt': decoder_optimizer.state_dict(),
            }, best_model_path)

        # 保存checkpoint
        if (epoch + 1) % save_epoch == 0:
            checkpoint_path = '{prefix}_{time}.pth'.format(prefix=gConfig["model_checkpoint"], time=time.strftime('%m%d_%H%M'))
            torch.save({
                'en': encoder.state_dict(),
                'de': decoder.state_dict(),
                'en_opt': encoder_optimizer.state_dict(),
                'de_opt': decoder_optimizer.state_dict(),
            }, checkpoint_path)
            conf.set("strings", "model_checkpoint", checkpoint_path)
        
        sys.stdout.flush()


def generate(input_seq, searcher, sos, eos):
    input_batch = [input_seq]
    input_lengths = torch.tensor([len(seq) for seq in input_batch])
    input_batch = torch.LongTensor([input_seq]).transpose(0,1)
    input_batch = input_batch.to(device)
    input_lengths = input_lengths.to('cpu')
    tokens, scores = searcher(sos, eos, input_batch, input_lengths, gConfig["max_generate_length"], device)
    return tokens


def eval(input_sentence):
    dataloader = get_dataloader() 
    _data = dataloader.dataset._data
    word2idx, idx2word = _data['word2idx'], _data['idx2word']
    sos = dataloader.dataset.sos
    eos = dataloader.dataset.eos
    unknown = dataloader.dataset.unknown
    voc_length = len(word2idx)

    encoder = EncoderRNN(voc_length)
    decoder = DecoderRNN(voc_length)

    if os.path.exists(gConfig["model_checkpoint"]):
        raise ValueError('model_ckpt is None.')
        return False
    checkpoint = torch.load(gConfig["model_checkpoint"], map_location=lambda s, l: s)
    encoder.load_state_dict(checkpoint['en'])
    decoder.load_state_dict(checkpoint['de'])

    with torch.no_grad():
        encoder = encoder.to(device)
        decoder = decoder.to(device)
        encoder.eval()
        decoder.eval()

        searcher = GreedySearchDecoder(encoder, decoder)

        cop = re.compile("[^\u4e00-\u9fa5^a-z^A-Z^0-9]")
        input_seq = jieba.lcut(cop.sub("", input_sentence)) 
        input_seq = input_seq[:gConfig["max_sentence_length"]] + [gConfig["end_of_string"]]
        input_seq = [word2idx.get(word, unknown) for word in input_seq]
        tokens = generate(input_seq, searcher, sos, eos)
        output_words = ''.join([idx2word[token.item()] for token in tokens])
        print('BOT: ', output_words)


def test():
    dataloader = get_dataloader() 
    _data = dataloader.dataset._data
    word2idx, idx2word = _data['word2idx'], _data['idx2word']
    sos = dataloader.dataset.sos
    eos = dataloader.dataset.eos
    unknown = dataloader.dataset.unknown
    voc_length = len(word2idx)

    encoder = EncoderRNN(voc_length)
    decoder = DecoderRNN(voc_length)

    if not os.path.exists(gConfig["model_checkpoint"]):
        raise ValueError('model_ckpt is None.')
        return False
    checkpoint = torch.load(gConfig["model_checkpoint"], map_location=lambda s, l: s)
    encoder.load_state_dict(checkpoint['en'])
    decoder.load_state_dict(checkpoint['de'])

    with torch.no_grad():
        encoder = encoder.to(device)
        decoder = decoder.to(device)
        encoder.eval()
        decoder.eval()

        searcher = GreedySearchDecoder(encoder, decoder)
        return searcher, sos, eos, unknown, word2idx, idx2word


def output_answer(input_sentence, searcher, sos, eos, unknown, word2idx, idx2word):
    cop = re.compile("[^\u4e00-\u9fa5^a-z^A-Z^0-9]")
    input_seq = jieba.lcut(cop.sub("", input_sentence))
    input_seq = input_seq[:gConfig["max_sentence_length"]] + [gConfig["end_of_string"]]
    input_seq = [word2idx.get(word, unknown) for word in input_seq]
    tokens = generate(input_seq, searcher, sos, eos)
    output_words = ''.join([idx2word[token.item()] for token in tokens if token.item() != eos])
    return output_words


def predict(sentence):
    searcher, sos, eos, unknown, word2idx, idx2word = test()
    output_sentence = output_answer(sentence, searcher, sos, eos, unknown, word2idx, idx2word)
    return output_sentence


# if __name__ == "__main__":
#     # s = '别笑了'
#     # print(predict(s))
#     train()
