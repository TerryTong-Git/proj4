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

class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None
    
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)
        glove_embeddings_path = 'data/stage_4_data/glove.6B.200d.txt'  # Update with your file path
        self.glove_embeddings = self.load_glove_embeddings(glove_embeddings_path)

    def load_glove_embeddings(self,file_path):
        embeddings_index = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.array(values[1:], dtype='float32')
                embeddings_index[word] = vector
        return embeddings_index
    
    def sentence_to_matrix(self,sentence, embeddings_index, max_length, embedding_dim):
        matrix = np.zeros((max_length, embedding_dim))
        # sentence = sentence.split()
        for i, word in enumerate(sentence):
            # print(word)
            if i >= max_length:
                break
            embedding_vector = embeddings_index.get(word)
            # print(embedding_vector)
            if embedding_vector is not None:
                matrix[i] = embedding_vector
        return matrix
    
    def load(self):
        print('loading data...')
        # Example usage
        # print(self.glove_embeddings.get("hello"))
        Xtrain, ytrain, Xtest, ytest = [], [], [], []

        # Function to read files from a folder and append text and labels to X and y lists
        def read_files(folder_path, X, y, label):
            for file_name in os.listdir(folder_path):
                if file_name.endswith('.txt'):
                    file_path = os.path.join(folder_path, file_name)
                    with open(file_path, 'r') as file:
                        text = file.read()
                        # Get the vector for a specific word
                        text = self.sentence_to_matrix(text, self.glove_embeddings,500, 200)
                        X.append(text)
                        y.append(label)
                # break
        # Read training data
        read_files(os.path.join(self.dataset_source_folder_path, 'train', 'pos'), Xtrain, ytrain, 1)
        read_files(os.path.join(self.dataset_source_folder_path, 'train', 'neg'), Xtrain, ytrain, 0)
        
        # Read testing data
        read_files(os.path.join(self.dataset_source_folder_path, 'test', 'pos'), Xtest, ytest, 1)
        read_files(os.path.join(self.dataset_source_folder_path, 'test', 'neg'), Xtest, ytest, 0)
    
        return {'X': Xtrain, 'y': ytrain, 'X1': Xtest, 'y1': ytest}