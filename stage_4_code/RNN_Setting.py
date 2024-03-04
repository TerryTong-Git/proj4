'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.setting import setting
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
import numpy as np
import torch

class RNN_Setting(setting):
    
    def load_run_save_evaluate(self):
        
        # load dataset
        loaddata= self.dataset.load()
        # X_train = torch.tensor(np.array(loaddata['X']),device='cuda').float()
        # y_train = torch.tensor(np.array(loaddata['y']),device='cuda')
        # X_test = torch.tensor(np.array(loaddata['X1']),device='cuda').float()
        # y_test = torch.tensor(np.array(loaddata['y1']),device='cuda')
        X_train = loaddata['X']
        y_train = loaddata['y']
        X_test = loaddata['X1']
        y_test = loaddata['y1']

        # run MethodModule
        self.method.data = {'train': {'X': X_train, 'y': y_train}, 'test': {'X': X_test, 'y': y_test}}
        learned_result = self.method.run()
        self.result.data = learned_result
        self.result.save()
            
        self.evaluate.data = learned_result
        
        # return self.evaluate.evaluate(), None

        