from code.stage_4_code.Dataset_Loader import Dataset_Loader
from code.stage_4_code.Method_LSTM import Method_RNN
from code.stage_1_code.Result_Saver import Result_Saver
from code.stage_4_code.RNN_Setting import RNN_Setting
from code.stage_4_code.Evaluate_Accuracy import Evaluate_Accuracy

import numpy as np
import torch
from torch.nn import DataParallel

#---- Multi-Layer Perceptron script ----
if 1:
    #---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    #------------------------------------------------------

    # ---- objection initialization setction ---------------
    data_obj = Dataset_Loader('toy', '')
    # data_obj.dataset_source_folder_path = '../../data/stage_1_data/'
    # data_obj.dataset_source_file_name = 'toy_data_file.txt'
    data_obj.dataset_source_folder_path = './data/stage_4_data/text_classification'
    # data_obj.dataset_source_file_name = 'pos'

    input_size = 200  # Dimensionality of GloVe embeddings
    hidden_size = 256  # Number of units in the RNN layer
    num_classes = 2  # Number of output classes

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        devices = list(range(torch.cuda.device_count()))
        method_obj = Method_RNN(input_size, hidden_size, num_classes,10)
        method_obj = method_obj.to("cuda")
    else:
        method_obj = Method_RNN(input_size, hidden_size, num_classes)

    setting_obj = RNN_Setting('RNN', '')
    
    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = './result/stage_4_result/RNN'
    result_obj.result_destination_file_name = 'prediction_result'
    evaluate_obj = Evaluate_Accuracy('accuracy', '')

    #setting_obj = Setting_Tra
    # in_Test_Split('train test split', '')

    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    # setting_obj.print_setup_summary()
    f1= setting_obj.load_run_save_evaluate()
    print('************ Overall Performance ************')
    print('RNN f1: ' + str(f1))
    print('************ Finish ************')
    # ------------------------------------------------------
    

    