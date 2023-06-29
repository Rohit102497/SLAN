# ------------------------------Libraries------------------------------
import torch
import torch.nn as nn
from tqdm import tqdm
import os
import numpy as np
import torch.optim as optim
import pickle
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score
from torch.utils.data import DataLoader, WeightedRandomSampler
import collections
import time
import os
import argparse

# ------------------------------Supporting .py Files------------------------------
from data import MIMICdata
from LSTMCell_MIMIC import LSTMCell
from utils import EarlyStopping_AUPRC, seed_everything

if __name__ == '__main__':
    
    __file__ = os.path.abspath('')

    # ------------------------------Variables------------------------------
    # get args
    parser = argparse.ArgumentParser()
    parser.add_argument('--expno', default="test", type=str, help='The experiment number. Default "test"')
    parser.add_argument('--seed', default=2020, type=int, help='The see value. Default 2020')

    parser.add_argument('--hidsz', default=32, type=int, help='The hidden size of LSTM. Default 32')
    parser.add_argument('--lr', default=0.0005, type=float, help='The learning rate. Default 0.0005')
    parser.add_argument('--bs', default=32, type=int, help='The batch size. Default 32')
    parser.add_argument('--epochs', default=1, type=int, help='Number of epochs. Default 1')
    parser.add_argument('--load_all', default=True, action=argparse.BooleanOptionalAction, help='Load all the training data. Default True')
    parser.add_argument('--ninst', default=64, type=int, help='The number of training instance to load. Default 32')
    parser.add_argument('--cuda', default=False, action=argparse.BooleanOptionalAction, help='True if using gpu. Default False')
    parser.add_argument('--gpu_no', default=0, type=int, help='The number of gpu to use. Default 0')
    parser.add_argument('--if_es', default=True, action=argparse.BooleanOptionalAction, help='Use early stopping. Default True')
    parser.add_argument('--patience_es', default=5, type=int, help='Patience of early stopping. Default 5')
    parser.add_argument('--delta_es', default=0.0, type=float, help='Delta of early stopping. Default 0.0')
    parser.add_argument('--if_scheduler', default=True, action=argparse.BooleanOptionalAction, help='learning rate scheduler is applied. Default True')
    parser.add_argument('--scheduler', default=0.5, type=float, help='Threshold for early stopping. Default 0.5')
    parser.add_argument('--if_dropout', default=True, action=argparse.BooleanOptionalAction, help='Dropout hidden nodes. Default True')
    parser.add_argument('--dropout', default=0.3, type=float, help='Dropout value. Default 0.3')
    parser.add_argument('--if_decay', default=True, action=argparse.BooleanOptionalAction, help='Decay the hidden states. Default True')
    
    parser.add_argument('--norm', default=False, action=argparse.BooleanOptionalAction, help='Normalize the data. Default False')
    parser.add_argument('--standard', default=True, action=argparse.BooleanOptionalAction, help='Standardize the data. Default True')
    parser.add_argument('--f_decay', default=True, action=argparse.BooleanOptionalAction, help='Decay he final hidden state of each feature. Default True')
    parser.add_argument('--f_only_ct', default=False, action=argparse.BooleanOptionalAction, help='Use only summary state for final prediction. Default False')
    parser.add_argument('--f_no_ct', default=False, action=argparse.BooleanOptionalAction, help='Use only hidden states for final prediction. Default False')
    parser.add_argument('--if_sr', default=True, action=argparse.BooleanOptionalAction, help='Use smapling rate for decay. Default True')
    parser.add_argument('--if_relu', default=False, action=argparse.BooleanOptionalAction, help='Apply relu layer to the concat of ht and ct before final prediction. Default False')
    parser.add_argument('--agg_by', default='mean', type=str, help='aggregate function to use for summary state. Aggregate options are - "mean",  "max", "attention". Default "mean"')
    parser.add_argument('--subsample', default=False, action=argparse.BooleanOptionalAction, help='A subset of features are used for training. Default False')
    parser.add_argument('--top', default=True, action=argparse.BooleanOptionalAction, help='If subsample is true, then top = True implies top 6 high sampling rate feature is used\
                        for all the model training and testing otherwise bottom 6 high sampling rate feature is used. Default True')

    parser.add_argument('--abl25', default=False, action=argparse.BooleanOptionalAction, help='Ablation study with only 25 percent data. Default False')
    parser.add_argument('--abl50', default=False, action=argparse.BooleanOptionalAction, help='Ablation study with only 50 percent data. Default False')
    parser.add_argument('--abl75', default=False, action=argparse.BooleanOptionalAction, help='Ablation study with only 75 percent data. Default False')
    
    args = parser.parse_args()


    experiment_no = args.expno # Assign an experiment number so that each saved result is unique
    seed = args.seed # Seeding parametes
    hidden_size = args.hidsz # The hidden size of all the LSTM used in the models
    lr = args.lr # Learning rate
    batch_size = args.bs
    epochs = args.epochs

    load_all = args.load_all # True, if all the training data is to be used.
    num_of_instances = args.ninst # Number of instances to train the model
    stat_file = __file__ + '/Data/MIMIC/mimic_stat.json'  # The location of the file containing statistics of training data
    sr_file = __file__ + '/Data/MIMIC/sampling_rate_mimic.npy' # It contains sampling rate value of each feature based on the training data
    
    # Ablation study. Using a fraction of training data
    if args.abl25: # 25% data used
        load_all = False
        num_of_instances = 3665
        stat_file = __file__ + '/Data/MIMIC/ablation_mimic_stat_25.json'
        sr_file = __file__ + '/Data/MIMIC/ablation_sampling_rate_mimic_25.npy'
    if args.abl50: # 50% data used
        load_all = False
        num_of_instances = 7331
        stat_file = __file__ + '/Data/MIMIC/ablation_mimic_stat_50.json'
        sr_file = __file__ + '/Data/MIMIC/ablation_sampling_rate_mimic_50.npy'
    if args.abl75: # 75% data used
        load_all = False
        num_of_instances = 10996
        stat_file = __file__ + '/Data/MIMIC/ablation_mimic_stat_75.json'
        sr_file = __file__ + '/Data/MIMIC/ablation_sampling_rate_mimic_75.npy'

    data_name = "MIMIC_IHM" # Keep this as data name for mimic. IHM stands for in hospital mortality
    number_features = 17 # Number of features in the IHM dataset
    n_class = 2 # Number of output class
    padding = True # Padding the inputs for fixed length in each instance. During training we unpad them.

    cuda = args.cuda # Use cpu or gpu
    gpu_no = args.gpu_no # IF gpu, specify the number of gpus
    es_flag_auprc = args.if_es # If use earlystopping
    patience_auprc = args.patience_es # Patience for early stopping
    delta_auprc = args.delta_es # Threshold for early stopping
    scheduler_factor = args.scheduler # Learning rate decay value
    dropout = args.dropout # Dropout value

    normalize_data = args.norm # Normalize the data
    standardize_data = args.standard # Standardize the data
    final_decay = args.f_decay # Decay of the final hidden state of each feature
    final_only_ct = args.f_only_ct # For final prediciton only use summary state and not the hidden state
    final_no_ct = args.f_no_ct # Only use final hidden state (ht) for prediction
    use_sr = args.if_sr # Use smapling rate for decay
    if_scheduler = args.if_scheduler # if learning rate scheduler is applied
    if_relu_at_end = args.if_relu # Apply relu layer to the concat of ht and ct before final prediction
    if_dropout = args.if_dropout # Dropout hidden nodes
    if_decay = args.if_decay # If decay the hidden states
    # Aggregate options are - #'mean' # 'max', 'attention'
    if args.agg_by not in ["mean", "max", "attention"]:
        # print("Wrong aggregate option chosen. Pass only from these three options: 'mean', 'max', 'attention'")
        raise NameError("Wrong aggregate option chosen. Pass only from these three options: 'mean', 'max', 'attention'")
    aggregate_by = args.agg_by # aggregate function to use for summary state
    # The last two parameters are for ablation study
    subsample = args.subsample # If a subset of features are used for training
    top = args.top # If subsample is true, then top = True implies top 6 high sampling rate feature is used
    # for all the model training and testing otherwise bottom 6 high sampling rate feature is used

    # ------------------------------Result Address------------------------------
    path_to_result = __file__ + '/Results/MIMIC/'
    result_addr = path_to_result + data_name + "_Exp"  +  experiment_no + ".data"
    auprc_model_addr = path_to_result + data_name + "_Exp"  +  experiment_no + "_auprcmodel.pth"
    datadir = __file__ + '/Data/MIMIC/in-hospital-mortality/'
    configdir = __file__ + '/Data/MIMIC/resources/'

    # ------------------------------Parameters to save------------------------------
    train_epoch_loss = []
    train_epoch_acc = []
    train_epoch_auprc = []
    train_epoch_auroc = []

    val_epoch_loss = []
    val_epoch_acc = []
    val_epoch_auprc = []
    val_epoch_auroc_macro = []

    epoch_time = []

    # ------------------------------Prepare DataLoaders------------------------------
    seed_everything(seed)
    if cuda:
        device = torch.device('cuda:' + str(gpu_no)) if torch.cuda.is_available() else torch.device('cpu')
    else:
        device = torch.device('cpu')
    print("device: ", device)
    trainset = MIMICdata(datadir = datadir,
                            configdir = configdir,
                                name = 'train', 
                                device = device,
                                num_of_instances = num_of_instances,
                                normalize_data = normalize_data,
                                standardize_data = standardize_data,
                                padding = padding, 
                                load_all = load_all,
                                subsample = subsample,
                                top = top,
                                stat_file = stat_file,
                                sr_file = sr_file)
    valset = MIMICdata(datadir = datadir, 
                            configdir = configdir,
                                name = 'val', 
                                device = device,
                                num_of_instances=num_of_instances,
                                normalize_data = normalize_data,
                                standardize_data = standardize_data,
                                padding = padding, 
                                load_all = True,
                                subsample = subsample,
                                top = top,
                                stat_file = stat_file,
                                sr_file = sr_file) 
    testset = MIMICdata(datadir = datadir, 
                            configdir = configdir,
                                name = 'test', 
                                device = device,
                                num_of_instances = num_of_instances,
                                normalize_data = normalize_data, 
                                standardize_data = standardize_data,
                                padding = padding, 
                                load_all = True,
                                subsample = subsample,
                                top = top,
                                stat_file = stat_file,
                                sr_file = sr_file) 

    # ------------------------------Prepare weighted sampler------------------------------
    target_list = trainset.__labels__()
    ld = collections.Counter(target_list)
    print(ld)
    class_count = [v for i,v in sorted(ld.items())]
    class_weights = 1./torch.tensor(class_count, dtype=torch.float) 
    class_weights_all = class_weights[target_list]
    weighted_sampler = WeightedRandomSampler(
        weights=class_weights_all,
        num_samples=len(class_weights_all),
        replacement=True
    )
    trainloader = DataLoader(trainset, batch_size = batch_size, sampler=weighted_sampler)
    valloader = DataLoader(valset, batch_size = batch_size, shuffle=False)
    testloader = DataLoader(testset, batch_size = batch_size, shuffle=False)

    # ------------------------------Initialize early stopping------------------------------
    early_stopping_auprc = EarlyStopping_AUPRC(patience=patience_auprc, 
                verbose=True, delta = delta_auprc, save_path =auprc_model_addr)

    # ------------------------------Model Creation------------------------------
    model = LSTMCell(hidden_size, number_features, n_class, device, sr = trainset.sampling_rate, use_sr = use_sr,
                final_decay = final_decay, final_only_ct = final_only_ct, if_relu_at_end = if_relu_at_end, 
                if_dropout = if_dropout, dropout = dropout, aggregate_by = aggregate_by, if_decay = if_decay, 
                final_no_ct = final_no_ct)
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=scheduler_factor,
                                                            patience=1, threshold=0.0001, threshold_mode='rel',
                                                            cooldown=0, min_lr=1e-8, eps=1e-08, verbose=True)

    # ------------------------------Model Training------------------------------
    for e in range(epochs):
        print("Epoch: ", e+1)
        train_loss = 0
        train_pred = []
        train_logits = []
        y_train_list = []
        y_val_list = []
        val_loss = 0
        val_pred = []
        val_logits = []

        train_no_of_batch = trainloader.__len__()
        model.train()
        st_time = time.time()
        for sample in tqdm(trainloader):
            optimizer.zero_grad()
            y_pred = model.forward(sample['x'].to(device), 
                sample['lx'].to(device), training = True)
            print(sample['y'])
            loss = loss_fn(y_pred, sample['y'].to(device))
            y_train_list += sample['y'].tolist()
            train_loss = train_loss + loss.item()
            loss.backward()
            optimizer.step()
            train_pred += torch.argmax(y_pred, 1).tolist()
            train_logits += y_pred[:,1].tolist()
        end_time = time.time()
        epoch_time.append(end_time - st_time)
        print("Epoch time: ", epoch_time[e])

        val_no_of_batch = valloader.__len__()
        model.eval()
        for sample in tqdm(valloader):
            y_pred = model.forward(sample['x'].to(device), sample['lx'].to(device))
            loss = loss_fn(y_pred, sample['y'].to(device))
            y_val_list += sample['y'].tolist()
            val_loss = val_loss + loss.item()
            val_pred += torch.argmax(y_pred, 1).tolist()
            val_logits += y_pred[:,1].tolist()

        # Calculate mean loss of training data and validation data
        train_epoch_loss.append(train_loss/train_no_of_batch)
        val_epoch_loss.append(val_loss/val_no_of_batch)

        # Calculate Accuracy of training data and validation data
        train_epoch_acc.append(accuracy_score(y_train_list, train_pred))
        val_epoch_acc.append(accuracy_score(y_val_list, val_pred))

        # Calculate auprc of training data and validation data
        train_epoch_auprc.append(average_precision_score(y_train_list, train_logits))
        val_epoch_auprc.append(average_precision_score(y_val_list, val_logits))

        # Calculate auroc of training data and validation data
        train_epoch_auroc.append(roc_auc_score(y_train_list, train_logits, average = "macro"))
        val_epoch_auroc_macro.append(roc_auc_score(y_val_list, val_logits, average = "macro"))

        print("Train Loss: ", train_epoch_loss[e], "\t Val Loss: ", val_epoch_loss[e])

        # early stopping check
        if es_flag_auprc:
            early_stopping_auprc(val_epoch_auprc[e], model)
            if early_stopping_auprc.early_stop:
                print(f"Early Stopping for AUPRC after {e+1} epochs")
                break    
        if if_scheduler:
            scheduler.step(val_epoch_auprc[e])

    # ------------------------------Save result on train and val data------------------------------
    result = {
        "data_name": data_name,
        "experiment_no": experiment_no,
        "epochs": epochs,
        "learning_rate": lr,
        "number_features": number_features,
        "hidden_size": hidden_size,
        "n_class": n_class,
        "batch_size": batch_size,
        "padding": padding,
        "load_all": load_all,
        "num_of_instances": num_of_instances,

        "cuda": cuda,
        "gpu_no": gpu_no,
        "patience_auprc": patience_auprc,
        "delta_auprc": delta_auprc,
        "es_flag_auprc": es_flag_auprc,
        "seed": seed,
        "normalize_data": normalize_data,
        "final_decay": final_decay,

        "final_only_ct": final_only_ct,
        "use_sr": use_sr,
        "if_scheduler": if_scheduler,
        "if_relu_at_end": if_relu_at_end, 
        "if_dropout": if_dropout,

        "scheduler_factor": scheduler_factor,
        "dropout": dropout,

        "Train_Loss" :  train_epoch_loss,
        "Train_Acc" :   train_epoch_acc,
        "Train_F1" :    train_epoch_f1,
        "Train_AUROC" : train_epoch_auroc,
        "Train_AUPRC" : train_epoch_auprc,
        
        "Val_Loss" :    val_epoch_loss,
        "Val_Acc" :     val_epoch_acc,
        "Val_F1" :      val_epoch_f1,
        "Val_AUROC_Macro" :   val_epoch_auroc_macro,
        "Val_AUPRC" :   val_epoch_auprc,

        "epoch time": epoch_time
    }

    # ------------------------------Load model for testing------------------------------
    best_model_auprc = LSTMCell(hidden_size, number_features, n_class, device, sr = trainset.sampling_rate, use_sr = use_sr,
                final_decay = final_decay, final_only_ct = final_only_ct, if_relu_at_end = if_relu_at_end, 
                if_dropout = if_dropout, dropout = dropout, aggregate_by = aggregate_by, if_decay = if_decay, final_no_ct = final_no_ct)
    best_model_auprc.load_state_dict(torch.load(auprc_model_addr))
    print("Model based on AUPRC loaded for testing.")

    test_loss = 0
    test_pred = []
    test_logits = []
    y_test_list = []
    test_no_of_batch = testloader.__len__()
    best_model_auprc.eval()
    for sample in tqdm(testloader):
        y_pred = best_model_auprc.forward(sample['x'].to(device), sample['lx'].to(device))
        loss = loss_fn(y_pred, sample['y'].to(device))
        y_test_list += sample['y'].tolist()
        test_loss = test_loss + loss.item()
        test_pred += torch.argmax(y_pred, 1).tolist()
        test_logits += y_pred[:,1].tolist()

    # Calculate mean loss of test data
    result["test_loss_auprc"] = test_loss/test_no_of_batch
    # Calculate Accuracy of test data
    result["test_acc_auprc"] = accuracy_score(y_test_list, test_pred)
    # Calculate AUPRC
    result["test_auprc_auprc"] = average_precision_score(y_test_list, test_logits)
    # Calculate auroc test data
    result["test_auroc_macro_auprc"] = roc_auc_score(y_test_list, test_logits, average = "macro")   
    result["test_pred_auprc"] = test_pred
    result["test_logits_auprc"] = test_logits
    result["y_test_list_auprc"] = y_test_list

    # ------------------------------Save results to a file------------------------------
    with open(result_addr, 'wb') as file:
            pickle.dump(result, file)

    print("Model succefully run and results and model are stored")