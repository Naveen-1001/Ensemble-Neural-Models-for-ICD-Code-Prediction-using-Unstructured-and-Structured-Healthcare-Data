import os
import torch
import numpy as np
import pandas as pd
from collections import Counter
from gensim.models.word2vec import Word2Vec
import pandas as pd
class Vocab(object):
    def __init__(self,training_data,
                 training_labels,label_desc,
                 min_word_frequency = -1,
                 max_vocab_size = -1,
                 word_embedding_mode = "word2vec",
                 word_embedding_file = None):
        
        self.word_embedding_mode = word_embedding_mode
        self.word_embedding_file = word_embedding_file
        self.word_embedding_size = 100
        self.word_embeddings = None
        
        self.training_data = training_data
        
        self.label_desc = label_desc
        self.PAD_TOKEN = '_PAD'
        self.UNK_TOKEN = '_UNK'
        self.word2index = None
        self.index2word = None
        
        self.label2index = None
        self.index2label = None
        
        self.vocab_words = [self.PAD_TOKEN, self.UNK_TOKEN]
        self.all_labels = []
        
        self.min_word_frequency = min_word_frequency
        self.max_vocab_size = max_vocab_size
        
        self.update_labels(training_labels)

    def index_of_word(self,word: str) -> int:
        try:
            return self.word2index[word]
        except:
            return self.word2index[self.UNK_TOKEN]

    def index_of_label(self,label: str) -> int:
        try:
            return self.label2index[label]
        except:
            return 0
    def update_labels(self, labels):
        self.all_labels = []
        self.index2label = []
        self.label2index = []
        all_labels = list(sorted(labels))
        self.label2index = {label: idx for idx, label in enumerate(all_labels)}
        self.index2label = {idx: label for idx, label in enumerate(all_labels)}
        self.all_labels = all_labels
            
    def prepare_vocab(self):
        self._build_vocab()
        if self.word_embedding_file is not None:
            self.word_embeddings = torch.FloatTensor(self._load_embeddings())

    def _build_vocab(self):
        all_words_df = pd.read_csv('../data/vocab.csv',header=None, on_bad_lines='skip')
        all_words = all_words_df.iloc[:,0].tolist()
        all_words.sort()
        
        self.vocab_words += all_words
        
        self.word2index = {word: idx for idx, word in enumerate(self.vocab_words)}
        self.index2word = {idx: word for idx, word in enumerate(self.vocab_words)}


    def _load_embeddings(self):
        if self.word_embedding_file is None:
            return None
        return self._load_word_embeddings()

    def _load_word_embeddings(self):
        unknown_vec = np.random.uniform(-0.25, 0.25, self.word_embedding_size)
        embeddings = [unknown_vec] * (len(self.vocab_words))
        embeddings[0] = np.zeros(self.word_embedding_size)
        for line in open(self.word_embedding_file, "rt"):
            split = line.rstrip().split(" ")
            word = split[0]
            vector = np.array([float(num) for num in split[1:]]).astype(np.float32)
            if len(vector) > 0:
                if word in self.word2index:
                    embeddings[self.word2index[word]] = vector
        embeddings = np.array(embeddings, dtype=np.float32)
        return embeddings
    
    def n_words(self):
        return len(self.vocab_words)

    def n_labels(self):
        return len(self.all_labels)