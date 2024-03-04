'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.method import method
from code.stage_4_code_gen.Evaluate_Accuracy import Evaluate_Accuracy
import torch.nn.functional as F
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torchtext
import numpy as np
import wandb
import random
import pandas as pd


class Method_RNN(method, nn.Module):
    data = None
    # it defines the max rounds to train the model
    max_epoch = 100
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-4

    # it defines the the MLP model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    # the size of the input/output portal of the model architecture should be consistent with our data input and desired output
    def init_state(self, sequence_length):
        return (torch.zeros(self.num_layers, sequence_length, self.lstm_size),
                torch.zeros(self.num_layers, sequence_length, self.lstm_size))
    def __init__(self, input_size, hidden_size, output_size, num_layers,dataset):
        self.dataset = dataset
        method.__init__(self)
        nn.Module.__init__(self)
        self.lstm_size = 256
        self.embedding_dim = 256
        self.num_layers = 3

        n_vocab = len(dataset.uniq_words)

        #does this create own embedding
        self.embedding = nn.Embedding(
            num_embeddings=n_vocab,
            embedding_dim=self.embedding_dim,
        )
        self.lstm = nn.LSTM(
            input_size=self.lstm_size,
            hidden_size=self.lstm_size,
            num_layers=self.num_layers,
            dropout=0.2,
        )
        self.fc = nn.Linear(self.lstm_size, n_vocab)
    # it defines the forward propagation function for input x
    # this function will calculate the output layer by layer

    def forward(self, x, prev_state):
        embed = self.embedding(x)#hmm
        output, state = self.lstm(embed, prev_state)
        logits = self.fc(output)
        return logits, state
    def train(self):
        train_dataloader = DataLoader(self.dataset, batch_size=256, shuffle=True)
        wandb.login()
        run = wandb.init(
            # Set the project where this run will be logged
            project="RNN_Gen1",
            # Track hyperparameters and run metadata
            config={
                "learning_rate": 0.001,
                "epochs": self.max_epoch,
            },
        )

        # check here for the torch.optim doc: https://pytorch.org/docs/stable/optim.html
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # check here for the nn.CrossEntropyLoss doc: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        loss_function = nn.CrossEntropyLoss()
        # for training accuracy investigation purpose
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')
        
        avg_accuracy = []
        avg_loss = []

        for epoch in range(self.max_epoch): # you can do an early stop if self.max_epoch is too much...
            state_h, state_c = self.init_state(4)
            state_h, state_c = state_h.cuda(), state_c.cuda()

            for batch_X, batch_y in train_dataloader:
                optimizer.zero_grad()

                batch_X, batch_y = batch_X.cuda(), batch_y.cuda()

                # convert y to torch.tensor as well
                y_true = batch_y
                # calculate the training loss
                y_pred, (state_h, state_c) = self.forward(batch_X, (state_h, state_c))

                state_h = state_h.detach()
                state_c = state_c.detach()
                train_loss = loss_function(y_pred.transpose(1, 2), y_true)

                #keep track of averages to print
                # accuracy_evaluator.data = {'true_y': y_true.cpu(), 'pred_y': y_pred.max(1)[1].cpu()}
                # acc = accuracy_evaluator.evaluate()
                loss = train_loss.item()
                # print("eval")
                # print(acc, loss)
                # avg_accuracy.append(acc)
                avg_loss.append(loss)

                # check here for the gradient init doc: https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html
                optimizer.zero_grad()
                # check here for the loss.backward doc: https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html
                # do the error backpropagation to calculate the gradients
                train_loss.backward()
                # check here for the opti.step doc: https://pytorch.org/docs/stable/optim.html
                # update the variables according to the optimizer and the gradients calculated by the above loss.backward function
                optimizer.step()

            if epoch%10 == 0:
                # print(avg_accuracy)
                # print(avg_loss)
                print('Epoch:', epoch,  'Loss:', np.average(avg_loss)) #'Accuracy:', np.average(avg_accuracy),
                wandb.log({"accuracy": 0, "loss": loss})

    def test(self):
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')
        train_df = pd.read_csv('./data/stage_4_data/text_generation/data')
        avg_bleu = []
        avg_rouge = []
        for i in train_df['Joke'][:50]:
            # print(i[:25])
            # break
            i = i.split(" ")
            y_pred = self.predict(" ".join(i[:5]))
            # print(y_pred)
            accuracy_evaluator.data = {'true_y': i, 'pred_y': y_pred[:len(i)]}
            bleu, rouge_n = accuracy_evaluator.evaluate()
            avg_bleu.append(bleu)
            avg_rouge.append(rouge_n)
        # print(avg_bleu)
        print("bleu:",np.average(avg_bleu))
        print("rouge_n:",np.average(avg_rouge))
        # convert the probability distributions to the corresponding labels
        # instances will get the labels corresponding to the largest probability
        return 
    
    def predict(self, text, next_words=100):

        words = text.split(' ')
        state_h, state_c = self.init_state(len(words))
        state_h, state_c = state_h.cuda(),state_c.cuda()

        for i in range(0, next_words):
            x = torch.tensor([[self.dataset.word_to_index[w] for w in words[i:]]]).cuda()
            y_pred, (state_h, state_c) = self.forward(x, (state_h, state_c))

            last_word_logits = y_pred[0][-1]
            p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().cpu().numpy()
            word_index = np.random.choice(len(last_word_logits), p=p)
            words.append(self.dataset.index_to_word[word_index])
        return ' '.join(words)
    
    def run(self):
        print('method running...')
        print('--start training...')
        # train_set, val_set = torch.utils.data.random_split(self.dataset, [int(len(self.dataset)*0.8), int(len(self.dataset)*0.2)])
        # self.train_set = train_set
        # self.test_set = val_set
        self.test()
        self.train()

        print('--start testing...')
        # pred_y = self.test(self.data['test']['X'])
        # return {'pred_y': pred_y, 'true_y': self.data['test']['y']}
        return