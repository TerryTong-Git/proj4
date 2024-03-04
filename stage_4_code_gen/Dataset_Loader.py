'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.dataset import dataset
import pickle
import os
import numpy as np
import torchtext
import torch
import torch
import pandas as pd
from collections import Counter

class Dataset_Loader(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_source_folder_path=None,
        dataset_source_file_name=None
    ):
        # self.args = args
        self.dataset_source_folder_path = dataset_source_folder_path
        self.dataset_source_file_name = dataset_source_file_name
        self.words = self.load_words()
        self.uniq_words = self.get_uniq_words()

        self.index_to_word = {index: word for index, word in enumerate(self.uniq_words)}
        self.word_to_index = {word: index for index, word in enumerate(self.uniq_words)}

        self.words_indexes = [self.word_to_index[w] for w in self.words]

    def load_words(self):
        train_df = pd.read_csv(self.dataset_source_folder_path+self.dataset_source_file_name)
        text = train_df['Joke'].str.cat(sep=' ')
        return text.split(' ')

    def get_uniq_words(self):
        word_counts = Counter(self.words)
        return sorted(word_counts, key=word_counts.get, reverse=True)

    def __len__(self):
        return len(self.words_indexes) - 4

    def __getitem__(self, index):
        return (
            torch.tensor(self.words_indexes[index:index+4]),
            torch.tensor(self.words_indexes[index+1:index+4+1]),
        )
    #create testing set and add bleu and rouge 1 score