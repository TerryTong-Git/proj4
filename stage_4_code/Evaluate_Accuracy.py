'''
Concrete Evaluate class for a specific evaluation metrics
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score, f1_score


class Evaluate_Accuracy(evaluate):
    data = None
    
    def evaluate(self):
        # print('evaluating performance...')
        pred_y = self.data['pred_y']
        # print(pred_y)
        true_y = self.data['true_y']
        # print(true_y)
        return accuracy_score(pred_y, true_y )
        