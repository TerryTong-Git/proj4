'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.method import method
from code.stage_4_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch.nn.functional as F
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torchtext
import numpy as np
import wandb

import random
from torch.utils.data.distributed import DistributedSampler

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class Method_RNN(method, nn.Module):
    data = None
    # it defines the max rounds to train the model
    max_epoch = 500
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-4

    # it defines the the MLP model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    # the size of the input/output portal of the model architecture should be consistent with our data input and desired output
    def __init__(self, input_size, hidden_size, output_size, num_layers, device):
        self.device = device
        method.__init__(self)
        nn.Module.__init__(self)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn = nn.GRU(input_size, hidden_size, batch_first=True, num_layers=10) ##try with LSTM and GRU
        # self.rnn_layers = nn.ModuleList([nn.RNN(input_size, hidden_size, batch_first=True) for _ in range(num_layers)])

        self.fc = nn.Linear(hidden_size, output_size) 
        

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(10, batch_size, self.hidden_size).to(self.device).detach()
        # print(h0.size())
        # Forward propagate RNN
        out, _ = self.rnn(x, h0)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        # print(out)
        # Apply thresholding to convert to binary output
        return out

    def train(self, X, y):
        print("here1")
        train_dataset = CustomDataset(np.array(X),np.array(y))
        print('here2')
        train_sampler = DistributedSampler(dataset=train_dataset)
        print("here 3")
        train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True,sampler=train_sampler, num_workers=4)
        print("here 4")
        # train_dataloader = TweetBatcher(X, batch_size = 64)

        # wandb.login()
        # run = wandb.init(
        #     # Set the project where this run will be logged
        #     project="Rnn_Classification_GRU",
        #     # Track hyperparameters and run metadata
        #     config={
        #         "learning_rate": 0.001,
        #         "epochs": self.max_epoch,
        #     },
        # )
        
        # check here for the torch.optim doc: https://pytorch.org/docs/stable/optim.html
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # check here for the nn.CrossEntropyLoss doc: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        loss_function = nn.CrossEntropyLoss()
        # for training accuracy investigation purpose
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')
        
        avg_accuracy = []
        avg_loss = []

        # it will be an iterative gradient updating process
        # we don't do mini-batch, we use the whole input as one batch
        # you can try to split X and y into smaller-sized batches by yourself
        for epoch in range(self.max_epoch): # you can do an early stop if self.max_epoch is too much...
            # get the output, we need to covert X into torch.tensor so pytorch algorithm can operate on it
            for batch_X, batch_y in train_dataloader:
                # batch_X = torch.stack(batch_X, dim = 0)
                # batch_y = torch.stack(batch_y, dim = 0)
                print("A")
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                print("b")

                y_pred = self.forward(batch_X)
                # convert y to torch.tensor as well
                print("c")

                y_true = batch_y
                # calculate the training loss
                train_loss = loss_function(y_pred, y_true)
                print("d")

                #keep track of averages to print
                accuracy_evaluator.data = {'true_y': y_true.cpu(), 'pred_y': y_pred.max(1)[1].cpu()}
                acc = accuracy_evaluator.evaluate().detach()
                loss = train_loss.item().detach()
                print("e")

                # print("eval")
                # print(acc, loss)
                avg_accuracy.append(acc)
                avg_loss.append(loss)
                print("f")

                # check here for the gradient init doc: https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html
                optimizer.zero_grad()
                print("g")

                # check here for the loss.backward doc: https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html
                # do the error backpropagation to calculate the gradients
                train_loss.backward()
                print("h")

                # check here for the opti.step doc: https://pytorch.org/docs/stable/optim.html
                # update the variables according to the optimizer and the gradients calculated by the above loss.backward function
                optimizer.step()
                print("i")

            if epoch%10 == 0:
                # print(avg_accuracy)
                # print(avg_loss)
                print('Epoch:', epoch, 'Accuracy:', np.average(avg_accuracy), 'Loss:', np.average(avg_loss))
                wandb.log({"accuracy": acc, "loss": loss})

    def test(self, X):
        X = X.to(self.device)
        test_dataset = CustomDataset(np.array(X),np.array(y))
        test_sampler = DistributedSampler(dataset=test_dataset)

        test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=True,sampler=test_sampler, num_workers=4)
        # do the testing, and result the result
        y_pred = self.forward(torch.FloatTensor(X))
        # convert the probability distributions to the corresponding labels
        # instances will get the labels corresponding to the largest probability
        return y_pred.max(1)[1]
    
    def run(self):
        print('method running...')
        print('--start training...')
        self.train(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'], self.data['test']['y'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}
            