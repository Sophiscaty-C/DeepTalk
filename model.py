# -*- coding: utf-8 -*- 
import torch
import logging
import torch.nn as nn
import torch.nn.functional as F
from config import getConfig


gConfig = {}
gConfig = getConfig.get_config()


class EncoderRNN(nn.Module):
    def __init__(self, voc_length):
        super(EncoderRNN, self).__init__()
        self.num_layers = gConfig["encoder_num_layers"]
        self.hidden_size = gConfig["hidden_size"]

        self.embedding = nn.Embedding(voc_length, gConfig["embedding_dim"])

        self.gru = nn.GRU(gConfig["embedding_dim"], self.hidden_size, self.num_layers,
                          dropout=(0 if self.num_layers == 1 else gConfig["dropout"]), bidirectional=gConfig["encoder_bidirectional"])

    def forward(self, input_seq, input_lengths, hidden=None):
        '''
        input_seq: 
            shape: [max_seq_len, batch_size]
        input_lengths: 
            shape: [batch_size]
        hidden:
            shape: [num_layers*num_directions, batch_size, hidden_size]
        embedded:
            shape: [max_seq_len, batch_size, embedding_dim]
        outputs:
            shape: [max_seq_len, batch_size, hidden_size]
        hidden:
            shape: [num_layers*num_directions, batch_size, hidden_size]
        '''
        
        embedded = self.embedding(input_seq) 
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        outputs, hidden = self.gru(packed, hidden)
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]
        return outputs, hidden


class Attn(torch.nn.Module):
    def __init__(self, attn_method, hidden_size):
        super(Attn, self).__init__()
        self.method = attn_method #attention方法
        self.hidden_size = hidden_size
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        if self.method == 'general':
            self.attn = torch.nn.Linear(self.hidden_size, self.hidden_size)
        elif self.method == 'concat':
            self.attn = torch.nn.Linear(self.hidden_size * 2, self.hidden_size)
            self.v = torch.nn.Parameter(torch.FloatTensor(self.hidden_size))

    def dot_score(self, hidden, encoder_outputs):
        '''
        encoder_outputs:
            encoder(双向GRU)的所有时刻的最后一层的hidden输出
            shape: [max_seq_len, batch_size, hidden_size]
            数学符号表示: h_s
        hidden:
            decoder(单向GRU)的所有时刻的最后一层的hidden输出,即decoder_ouputs
            shape: [max_seq_len, batch_size, hidden_size]
            数学符号表示: h_t
        注意: attention method: 'dot', Hadamard乘法,对应元素相乘，用*就好了
            torch.matmul是矩阵乘法, 所以最后的结果是h_s * h_t
            h_s的元素是一个hidden_size向量, 要得到score值,需要在dim=2上求和
            相当于先不看batch_size,h_s * h_t 要得到的是 [max_seq_len]
            即每个时刻都要得到一个分数值, 最后把batch_size加进来,
            最终shape为: [max_seq_len, batch_size]   
        '''

        return torch.sum(hidden * encoder_outputs, dim=2)

    def general_score(self, hidden, encoder_outputs):
        energy = self.attn(encoder_outputs)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_outputs):
        energy = self.attn(torch.cat((hidden.expand(encoder_outputs.size(0), -1, -1), 
				      encoder_outputs), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)
    
    def forward(self, hidden, encoder_outputs):
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        attn_energies = attn_energies.t()

        return F.softmax(attn_energies, dim=1).unsqueeze(1)


class DecoderRNN(nn.Module):
    def __init__(self, voc_length):
        super(DecoderRNN, self).__init__()

        self.attn_method = gConfig["attention_method"]
        self.hidden_size = gConfig["hidden_size"]
        self.output_size = voc_length
        self.num_layers = gConfig["decoder_num_layers"]
        self.dropout = gConfig["dropout"]
        self.embedding = nn.Embedding(voc_length, gConfig["embedding_dim"])
        self.embedding_dropout = nn.Dropout(self.dropout)
        self.gru = nn.GRU(gConfig["embedding_dim"], self.hidden_size, self.num_layers, dropout=(0 if self.num_layers == 1 else self.dropout))
        self.concat = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.attn = Attn(self.attn_method, self.hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        '''
        input_step: 
            [1, batch_size]
        last_hidden:
            [num_layers, batch_size, hidden_size]
        encoder_outputs:
            用于计算attention
        '''
        # [1, batch_size, embedding_dim]
        embedded = self.embedding(input_step) 
        embedded = self.embedding_dropout(embedded)
        # rnn_output: [1, batch_size, hidden_size]
        # hidden: [num_layers, batch_size, hidden_size]
        rnn_output, hidden = self.gru(embedded, last_hidden)
        # attn_weights: [batch_size, 1, max_seq_len]
        attn_weights = self.attn(rnn_output, encoder_outputs)
        # context: [batch_size, 1, hidden_size]
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)

        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))  
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)

        return output, hidden
