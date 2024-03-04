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
        # run MethodModule
        learned_result = self.method.module.run()
        return
        # self.result.data = learned_result
        # self.result.save()
            
        # self.evaluate.data = learned_result
        
        # return self.evaluate.evaluate(), None

        