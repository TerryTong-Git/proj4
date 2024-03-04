from code.stage_4_code.Dataset_Loader import Dataset_Loader
from code.stage_4_code.Method_GRU import Method_RNN
from code.stage_1_code.Result_Saver import Result_Saver
from code.stage_4_code.RNN_Setting import RNN_Setting
from code.stage_4_code.Evaluate_Accuracy import Evaluate_Accuracy

import numpy as np
import torch
from torch.nn import DataParallel
import argparse
import random
import os
# local_rank = int(os.environ["LOCAL_RANK"]) 
# def set_random_seeds(random_seed=0):

#     torch.manual_seed(random_seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#     np.random.seed(random_seed)
#     random.seed(random_seed)


# if 1:
#     #parser for distributed training 
#     parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#     parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#     parser.add_argument("--local-rank", type=int, help="Local rank. Necessary for using the torch.distributed.launch utility.")
#     parser.add_argument("--num_epochs", type=int, help="Number of training epochs.", default=200)
#     parser.add_argument("--batch_size", type=int, help="Training batch size for one process.", default=64)
#     parser.add_argument("--learning_rate", type=float, help="Learning rate.", default=1e3)
#     parser.add_argument("--random_seed", type=int, help="Random seed.", default=1)
#     parser.add_argument("--model_dir", type=str, help="Directory for saving models.", default="./")
#     parser.add_argument("--model_filename", type=str, help="Model filename.", default="./")
#     parser.add_argument("--resume", action="store_true", help="Resume training from saved checkpoint.")
#     argv = parser.parse_args()

#     local_rank = argv.local_rank
#     num_epochs = argv.num_epochs
#     batch_size = argv.batch_size
#     learning_rate = argv.learning_rate
#     random_seed = argv.random_seed
#     model_dir = argv.model_dir
#     model_filename = argv.model_filename
#     resume = argv.resume

#     model_filepath = os.path.join(model_dir, model_filename)
#     torch.distributed.init_process_group(backend="nccl")
#     device = torch.device("cuda:{}".format(local_rank))

    # We need to use seeds to make sure that the models initialized in different processes are the same
    # set_random_seeds(random_seed=random_seed)

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
        model = Method_RNN(input_size, hidden_size, num_classes,10)
        method_obj = model
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
    

    