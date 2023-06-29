# This code reads all the results stored

from sklearn.metrics import confusion_matrix as cm
import numpy as np
import pickle
import os
import argparse

if __name__ == '__main__':

    __file__ = os.path.abspath('')
    parser = argparse.ArgumentParser()
    parser.add_argument('--expno', default="test", type=str, help='The experiment number. Default "test"')
    parser.add_argument('--data_name', default="MIMIC", type=str, help='The data name. Choose from these options: "MIMIC", "P12", "P19". Default MIMIC')
    args = parser.parse_args()

    experiment_no = args.expno
    if args.data_name not in ["MIMIC", "P12", "P19"]:
        raise NameError("Wrong data_name chosen. Pass only from these three options: 'MIMIC', 'P12', 'P19'")
    data_name = args.data_name

    path_to_result = __file__ + '/Results/' + data_name + '/'
    if data_name == "MIMIC":
        result_addr = path_to_result + "MIMIC_IHM" + "_Exp" + experiment_no + ".data"
    else:
        result_addr = path_to_result + data_name + "_Exp" + experiment_no + ".data"
    file = open(result_addr, 'rb')
    data = pickle.load(file)
    print(data.keys())
    print("data_name", '\t',  data['data_name'])
    print("experiment_no", '\t',  data['experiment_no'])
    print("epochs", '\t', data['epochs'])
    print("learning_rate", '\t', data['learning_rate'])
    print("number_features", '\t', data['number_features'])
    print("hidden_size", '\t', data['hidden_size'])
    print("n_class", '\t', data['n_class'])
    print("batch_size", '\t', data['batch_size'])
    print("padding", '\t',  data['padding'])
    print("load_all", '\t', data['load_all'])
    print("num_of_instances", '\t', data['num_of_instances'])
    print("cuda", '\t', data['cuda'])
    print("gpu_no", '\t', data['gpu_no'])
    print("seed", '\t', data['seed'])
    print("normalize_data \t", data['normalize_data'])
    print("number of epochs ran", '\t', len(data['Train_Loss']), '\n')

    print("Train Loss", '\t',  np.round(np.mean(data['Train_Loss'])*100,2))
    print("Train Acc", '\t', np.round(np.mean(data["Train_Acc"])*100,2))
    print("Train AUROC", '\t', np.round(np.mean(data['Train_AUROC'])*100,2))
    print("Train AUPRC", '\t', np.round(np.mean(data['Train_AUPRC'])*100,2))
    if data_name == "P19":
        print("Train balance accuracy", '\t', np.round(np.mean(data['Train_balance_acc'])*100,2))
        print("Train utility", '\t', np.round(np.mean(data['Train_utility'])*100,2), '\n')

    print("Val Loss", '\t',  np.round(np.mean(data['Val_Loss'])*100,2))
    print("Val Acc", '\t', np.round(np.mean(data["Val_Acc"])*100,2))
    print("Val AUROC Macro", '\t', np.round(np.mean(data['Val_AUROC_Macro'])*100,2))
    print("Val AUPRC", '\t', np.round(np.mean(data['Val_AUPRC'])*100,2))
    if data_name == "P19":
        print("Val balance accuracy", '\t', np.round(np.mean(data['Val_balance_acc'])*100,2))
        print("Val utility", '\t', np.round(np.mean(data['Val_utility'])*100,2), '\n')

    print("Test result on all the earlystop on the AUPRC")
    print("Val AUPRC: Test result is reported on the epoch number", np.argmax(data['Val_AUPRC'])+1)
    print("Test Loss AUPRC", '\t',  np.round(data['test_loss_auprc']*100,2))
    print("Test Acc AUPRC", '\t', np.round(data["test_acc_auprc"]*100,2))
    print("Test AUROC Macro AUPRC", '\t', np.round(data['test_auroc_macro_auprc']*100,2))
    print("Test AUPRC AUPRC", '\t', np.round(data['test_auprc_auprc']*100,2))
    if data_name == "P19":
        print("Test balanced acc AUPRC", '\t',  np.round(data['test_balance_acc_auprc']*100,2))
        print("Test utility score AUPRC", '\t', np.round(data["test_utility_auprc"]*100,2))
    print(cm(data['y_test_list_auprc'], data['test_pred_auprc']), "\n\n")