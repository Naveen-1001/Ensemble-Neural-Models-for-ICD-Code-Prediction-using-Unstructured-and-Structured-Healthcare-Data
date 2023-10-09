import torch
from vocab import Vocab
from constants import *
import torch.nn as nn
from torch.autograd import Variable
from embedding_layer import EmbeddingLayer
from attention_layer import AttentionLayer
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence



class RNN(nn.Module):
    def __init__(self, vocab,device):
        super(RNN, self).__init__()
        self.vocab_size = vocab.n_words()
        self.vocab = vocab
        self.use_last_hidden_state = use_last_hidden_state
        self.mode = mode
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.bidirectional = bool(bidirectional)
        self.n_directions = int(self.bidirectional) + 1
        self.attention_mode = attention_mode
        self.output_size = self.hidden_size * self.n_directions
        self.rnn_model = rnn_model
        self.device = device
        self.dropout = dropout
        self.embedding = EmbeddingLayer(embedding_mode=mode,
                                     embedding_size=embedding_size,
                                     pretrained_word_embeddings=vocab.word_embeddings,
                                     vocab_size=vocab.n_words())
        self.label_linear = nn.Linear(self.embedding.output_size, d_b)
        
        self.rnn = nn.LSTM(self.embedding.output_size, self.hidden_size, num_layers=self.n_layers,
                           bidirectional=self.bidirectional, dropout=self.dropout if self.n_layers > 1 else 0)
        
        self.use_dropout = dropout > 0
        self.dropout = nn.Dropout(dropout)
        self.attention = AttentionLayer(size=self.output_size,
                                         n_labels=self.vocab.n_labels(),attention_mode=attention_mode)
        self.classification_layer = nn.Linear(d_b , n_labels, bias=True)
        torch.nn.init.normal_(self.classification_layer.weight, 0.0, 0.03)
        torch.nn.init.normal_(self.label_linear.weight, 0.0, 0.3)

    def init_hidden(self,batch_size: int = 1) -> Variable:
        h = Variable(torch.zeros(self.n_layers * self.n_directions, batch_size, self.hidden_size)).to(self.device)
        c = Variable(torch.zeros(self.n_layers * self.n_directions, batch_size, self.hidden_size)).to(self.device)
        return h, c

    def forward(self,batch_data: torch.LongTensor,lengths: torch.LongTensor,desc_data: torch.LongTensor) -> tuple:
        batch_size = batch_data.size()[0]
        hidden = self.init_hidden(batch_size)
        embeds = self.embedding(batch_data)
        if self.use_dropout:
            embeds = self.dropout(embeds)
            
        embeds = pack_padded_sequence(embeds, lengths.cpu(), batch_first=True)
        desc_embeds = self.embedding(desc_data)
        label_weights = self.label_linear(desc_embeds)
        label_weights = [label_weights for i in range(batch_size)]
        label_weights = torch.stack(label_weights)
        rnn_output, hidden = self.rnn(embeds, hidden)
    
        if self.rnn_model.lower() == "lstm":
            hidden = hidden[0]
            
        rnn_output = pad_packed_sequence(rnn_output)[0]
        rnn_output = rnn_output.permute(1, 0, 2)
        
        weighted_output = self.attention(rnn_output,label_weights)
        weighted_output = self.classification_layer.weight.mul(weighted_output).sum(dim=2).add(self.classification_layer.bias)
        return weighted_output