import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import random
from tqdm import tqdm
import numpy as np

class TextDataset(Dataset):

    def __init__(self,text_data,vocab, label_desc,sort = False,max_seq_length = -1,min_seq_length = -1, multilabel= False):
        super(TextDataset, self).__init__()
        self.vocab = vocab
        self.multilabel = multilabel
        self.max_seq_length = max_seq_length
        self.min_seq_length = min_seq_length
        self.PAD_ID = self.vocab.index_of_word(self.vocab.PAD_TOKEN)
        indexed_data = []
        self.n_instances = len(text_data)
        self.n_labels = len(label_desc)
        self.n_total_tokens = 0
        self.label_count = dict()
        self.labels = set()
        
        for text, labels, _id in tqdm(text_data, unit="samples", desc="Processing data"):
            label_list = []
            for label in labels:
                if label in self.vocab.label2index:
                    label = self.vocab.index_of_label(label)
                    if label not in self.label_count:
                        self.label_count[label] = 1
                    else:
                        self.label_count[label] += 1
                    self.labels.add(label)
                    label_list.append(label)
                else:
                    continue

            if len(label_list) == 0:
                continue

            word_seq = []
            sent_words = text.strip().split()
            for word in sent_words:
                word_idx = vocab.index_of_word(word)
                word_seq.append(word_idx)
                self.n_total_tokens += 1
                if len(word_seq) >= self.max_seq_length > 0:
                    break
                    
            indexed_label_desc = []
            for lbl, desc in label_desc:
                desc_seq = []
                desc_words = desc.strip().split()
                for word in desc_words:
                    word_idx = vocab.index_of_word(word)
                    desc_seq.append(word_idx)
                    self.n_total_tokens += 1
                indexed_label_desc.append((desc_seq,self.vocab.index_of_label(lbl)))
                 
            indexed_label_desc = sorted(indexed_label_desc, key=lambda t: t[1])
            
            if len(word_seq) > 0:
                indexed_data.append((word_seq, label_list, _id, indexed_label_desc))

        if sort:
            self.indexed_data = sorted(indexed_data, key=lambda x: -len(x[0]))

        else:
            self.indexed_data = indexed_data
            self.shuffle_data()

        self.labels = sorted(list(self.labels))
        self.size = len(self.indexed_data)

    def shuffle_data(self):
        random.shuffle(self.indexed_data)

    def __getitem__(self, index):
        word_seq, label_list, _id, indexed_label_desc = self.indexed_data[index]

        if len(word_seq) > self.max_seq_length > 0:
            word_seq = word_seq[:self.max_seq_length]

        one_hot_label_list = [0] * self.vocab.n_labels()
        for label in label_list:
            one_hot_label_list[label] = 1
        return word_seq, one_hot_label_list, _id, indexed_label_desc

    def __len__(self):
        return len(self.indexed_data)
    
    
class TextDataLoader(DataLoader):
    def __init__(self,vocab,**kwargs):
        super(TextDataLoader, self).__init__( **kwargs)
        self.collate_fn = self._collate_fn
        self.PAD_ID = vocab.index_of_word(vocab.PAD_TOKEN)
        self.vocab = vocab

    def _collate_fn(self, batch):
        length_batch = []
        feature_batch = []
        label_batch = []
        id_batch = []
        desc_batch = []
        multilabel = True
        for features, labels, _id, label_desc in batch:
            desc = [ld[0] for ld in label_desc]
            desc_torched = []
            for d in desc:
                desc_torched.append(torch.LongTensor(d))
            desc = desc_torched
            for i in range(len(desc)):
                desc[i] = nn.ConstantPad1d((0, 6 - desc[i].shape[0]), 0)(desc[i]).tolist()
            desc_batch.append(desc)
            
            feature_length = len(features)
            feature_batch.append(torch.LongTensor(features))

            length_batch.append(feature_length)
            label_batch.append(labels)
            id_batch.append(_id)

        feature_batch, label_batch, length_batch, id_batch = \
            self.sort_batch(feature_batch, label_batch, length_batch, id_batch)
        
        padded_batch = pad_sequence(feature_batch, batch_first=True)
        feature_batch = torch.LongTensor(padded_batch)
        desc_batch = torch.LongTensor(desc_batch)
        label_batch = np.stack(label_batch, axis=0)
        label_batch = torch.FloatTensor(label_batch.tolist())
        length_batch = torch.LongTensor(length_batch)
        return feature_batch, label_batch, length_batch, id_batch, desc_batch

    @staticmethod
    def sort_batch(features, labels, lengths, id_batch):
        sorted_indices = sorted(range(len(features)), key=lambda i: features[i].size(0), reverse=True)
        sorted_features = []
        sorted_labels = []
        sorted_lengths = []
        sorted_ids = []

        for index in sorted_indices:
            sorted_features.append(features[index])
            sorted_labels.append(labels[index])
            sorted_lengths.append(lengths[index])
            sorted_ids.append(id_batch[index])

        return sorted_features, sorted_labels, sorted_lengths, sorted_ids