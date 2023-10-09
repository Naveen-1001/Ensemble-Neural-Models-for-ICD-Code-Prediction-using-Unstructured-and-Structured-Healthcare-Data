import torch.nn as nn
import torch
import copy
class EmbeddingLayer(nn.Module):
    def __init__(self,embedding_mode,pretrained_word_embeddings,vocab_size,embedding_size):

        self.embedding_mode = embedding_mode
        super(EmbeddingLayer, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_size)
        self.embeddings.weight = nn.Parameter(copy.deepcopy(pretrained_word_embeddings), requires_grad=False)
        self.output_size = embedding_size

    def forward(self,batch_data):
        embeds = self.embeddings(batch_data)  # [batch_size x max_seq_size x embedding_size]
        return embeds