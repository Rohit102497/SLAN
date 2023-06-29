#data.py

import pandas as pd
import numpy as np
import json
import collections
import pickle
import torch
from torch.utils.data import Dataset
import os
__file__ = os.path.abspath('')


class mimicforsetlstm(Dataset):
    def __init__(self, datadir, configdir, name, device, num_of_instances=2560,
    normalize_data =False, standardize_data = False, padding=False, load_all=True,
    subsample=False, top=False, stat_file = './mimic_stat.json', sr_file = './sampling_rate_mimic.npy'):

        self.datadir = datadir
        self.configdir = configdir
        self.name = name
        self.device  = device
        self.do_padding = padding
        # normalization flag
        self.normalize_data = normalize_data
        self.standardize_data = standardize_data
        self.load_all = load_all
        # The max length for IHM is 1569 after removing the 32 patients as done in SeFT
        self.MAX_LEN = 1569 
        self.subsample = subsample
        self.top = top
        #shifted one index
        #Sort in ascending
        # Pick last six and reverse them
        self.topsix_idx = [12,13,9,14,2,11]
        #pick first six
        self.bottom_idx = [1, 10, 16, 3, 17, 6]
        with open(self.configdir+ 'discretizer_config.json','r') as json_file:
            self.discretizer_config = json.loads(json_file.read())
        
        with open(self.configdir + 'channel_info.json','r') as json_file:
            self.channelinfo = json.loads(json_file.read())
        ignore_files = []
        with open(__file__ + '/Data/MIMIC/ignore_file.txt','r') as fp:
            for line in fp:
                ignore_files.append(line.strip())
        with open(stat_file, 'r') as fp:
            self.loaded_stat = json.loads(fp.read())  
        
        with open(sr_file, 'rb') as f:
            self.sampling_rate = np.load(f)
        
        if self.subsample:
            if self.top:
                self.sampling_rate = self.sampling_rate[np.array(self.topsix_idx) - 1]
            else:
                self.sampling_rate = self.sampling_rate[np.array(self.bottom_idx) - 1]

        self.df = pd.read_csv(self.datadir + f'{name}_listfile.csv')
        self.df = self.df[~self.df.stay.isin(ignore_files)]

        if self.load_all:
            self.df = self.df[:]
        else:
            self.df = self.df[:num_of_instances]

        self.df = self.df[~self.df.stay.isin(ignore_files)]

        self.id_to_channel = self.discretizer_config['id_to_channel']
        self.is_categorical_channel = self.discretizer_config['is_categorical_channel']

        self.possible_values = self.channelinfo
        
        self.normal_values = self.discretizer_config['normal_values']

        #self.configdir = configdir

    def __len__(self):
        return len(self.df)

    def __labels__(self):
        return list(self.df.y_true)

    def get_labels(self):
        return list(self.df.y_true)

    def __getitem__(self, idx):

        info = self.df.iloc[idx,:].values
        filename = info[0]
        y = info[1]
        if self.name != 'test':
            patient_df = pd.read_csv(self.datadir + f'train/{filename}')
        else:
            patient_df = pd.read_csv(self.datadir + f'test/{filename}')

        if self.subsample:
            if self.top:
                patient_df = patient_df.iloc[:,[0] +self.topsix_idx]
            else:
                patient_df = patient_df.iloc[:,[0] +self.bottom_idx]

        info = filename.split('_')
        patient_id = float(info[0] + '.' +  info[1].split('episode')[-1])
        feature_dict = {fname:idx for idx, fname in enumerate(patient_df.columns[1:])}
        
        feature_ftime = {idx:[] for idx, fname in enumerate(patient_df.columns[1:])}
        time_global = []
        type_global = []
        z_global = []
        delt_global = []
        

        for i,rowidx in enumerate(patient_df.iterrows()):
            info = rowidx[1]
            
            timestamp = info.values[0]
            features = info.values[1:] # feature values
            
            rowmask = ~info.isna().values[1:] # which are available
         
            available_feature_keys = np.array(list(feature_dict.keys()))[rowmask]
            available_feature_values = features[rowmask]
            
            for af,av in zip(available_feature_keys, available_feature_values):
                feat_key = feature_dict[af]
                
                time_global.append(timestamp)
                type_global.append(feat_key)
                
                if self.is_categorical_channel[af]:
                    "fixing categorical values"
                    vdict = self.possible_values[af]['values']
                    
                    if str(av) not in vdict:
                        av = str(av).split('.')[0]
                        z_global.append(vdict[str(av)])
                    else:
                        z_global.append(vdict[str(av)])
                else:
                    if self.normalize_data:
                        mn,mx = self.loaded_stat[af]['min'],self.loaded_stat[af]['max']
                        if mx == mn:
                            normalized_av =  av/mx
                        else:
                            normalized_av =  (av - mn) / (mx-mn)
                        z_global.append(normalized_av)
                    elif self.standardize_data:
                        mean,std = self.loaded_stat[af]['mean'],self.loaded_stat[af]['std']
                        if std == 0:
                            standardize_av =  1
                        else:
                            standardize_av =  (av - mean) / std
                        z_global.append(standardize_av)
                    else:
                        z_global.append(av)
                if len(feature_ftime[feat_key]) == 0:
                    # meaning: feature af is appearing for the first time
                    feature_ftime[feat_key].append(timestamp)
                    delt_global.append(0)
                else:
                    feature_ftime[feat_key].append(timestamp)
                    delt_global.append(timestamp - feature_ftime[feat_key][-2])
                    
        curr_len = len(z_global)
        if self.do_padding:
            diff = self.MAX_LEN - curr_len
            if diff > 0:
                time_global += [-1]*diff
                type_global += [-1]*diff
                z_global += [-1]*diff
                delt_global += [-1]*diff
            else:
                time_global = time_global[:self.MAX_LEN]
                type_global = type_global[:self.MAX_LEN]
                z_global = z_global[:self.MAX_LEN]
                delt_global = delt_global[:self.MAX_LEN]
        
        stacked_x = torch.vstack((torch.tensor(time_global),
                                    torch.tensor(type_global),
                                    torch.tensor(z_global), 
                                torch.tensor(delt_global)))

        return {'x': stacked_x,
                'y':torch.tensor(y), 
                 'lx':torch.tensor(curr_len),
                'pid':torch.tensor(patient_id)}

class P12data(Dataset):
    def __init__(self, config_dir, name, device, num_of_instances=2560, 
                normalize_data=False, standardize_data =False, 
                padding=True, load_all=True, to_predict='mortality'):
        self.config_dir = config_dir
        self.name = name
        self.load_all = load_all
        self.device=device
        self.normalize_data = normalize_data
        self.standardize_data = standardize_data
        with open(self.config_dir + f'p12_{name}.pickle', 'rb') as fp:
            self.data = pickle.load(fp)
        
        with open(self.config_dir + 'p12_stat.pickle', 'rb') as fp:
            self.loaded_stat = pickle.load(fp) 
        
        self.labels = [d['target'] for d in self.data]
        
        if not self.load_all:
            self.data = self.data[:num_of_instances]
            self.labels = self.labels[:num_of_instances]
        
        print(f"{self.name} loaded , num of instances:", len(self.data))#, self.sample_idx)
        print(f'{self.name} distribution:', collections.Counter(self.labels))
        self.MAX_LEN = 1400
        self.do_padding = padding

        with open(__file__ + '/Data/P12/sampling_rate_p12.npy', 'rb') as f:
            self.sampling_rate = np.load(f)
        
        
    def __len__(self):
        return len(self.labels)
    
    def __labels__(self):
        # Counter({0: 10281, 1: 1707})
        return np.array(self.labels)
    
    def __getitem__(self, idx):
        arr = self.data[idx]
        nfeatures = arr['diagnosis'].shape[1]
        y = self.labels[idx]
        curr_len = arr['length']
        t = arr['timestamps'].reshape(-1)
        #pid = int(arr['id'])
        static = arr['static']
        # all are padded to 215
        time_global = []
        type_global = []
        z_global = []
        delt_global = []
        
        lasttime_feat = {idx:[] for idx in range(37)}

        for ii, (row ,mask, tmp) in enumerate(zip(arr['diagnosis'],arr['masks'], t)):
            # iterating each row
            if ii < curr_len:
                
                nonzero_idx = [i for i,r in enumerate(mask) if r]
                #nonzero_idx non zero feature ids
                for nidx in nonzero_idx:
                    time_global.append(tmp)
                    type_global.append(nidx)
                    av = row[nidx]
                    if self.normalize_data:
                        mn,mx = self.loaded_stat[f'idx{nidx}']['min'],self.loaded_stat[f'idx{nidx}']['max']
                        if mn != mx:
                            normalized_av =  (av - mn) / (mx-mn)
                        else:
                            normalized_av = av
                        z_global.append(normalized_av)
                    elif self.standardize_data:
                        mean,std = self.loaded_stat[f'idx{nidx}']['mean'],self.loaded_stat[f'idx{nidx}']['std']
                        if std != 0:
                            normalized_av = (av - mean) / std
                        else:
                            normalized_av = av
                        z_global.append(normalized_av)
                    else:
                        z_global.append(av)
                    if len(lasttime_feat[nidx]) == 0:
                        delt_global.append(0)
                        lasttime_feat[nidx].append(tmp)
                    else:
                        lasttime_feat[nidx].append(tmp)
                        delt_global.append(tmp-lasttime_feat[nidx][-2])
                    
        curr_len = len(z_global)
        if self.do_padding:
            diff = self.MAX_LEN - curr_len
            if diff > 0:
                time_global += [-1]*diff
                type_global += [-1]*diff
                z_global += [-1]*diff
                delt_global += [-1]*diff
            else:
                time_global = time_global[:self.MAX_LEN]
                type_global = type_global[:self.MAX_LEN]
                z_global = z_global[:self.MAX_LEN]
                delt_global = delt_global[:self.MAX_LEN]
        
        stacked_x = torch.vstack((torch.tensor(time_global),
                                    torch.tensor(type_global), # indices of features (0,1, 2.... so on)
                                    torch.tensor(z_global), 
                                torch.tensor(delt_global)))
        
        return {
            'x' : stacked_x,
            'y' : torch.tensor(y),
            'lx': torch.tensor(curr_len),
            #'pid': torch.tensor(pid),
            'static': torch.tensor(static)
        }

class P19data(Dataset):
    def __init__(self, config_dir, name, device, num_of_instances=2560, 
                normalize_data=False, standardize_data =False, 
                padding=True, load_all=True, to_predict='sepsis'):
        self.config_dir = config_dir
        self.name = name
        self.load_all = load_all
        self.device=device
        self.normalize_data = normalize_data
        self.standardize_data = standardize_data
        self.num_features = 336
        with open(self.config_dir + f'p19_{name}.pickle', 'rb') as fp:
            self.data = pickle.load(fp)
        
        with open(self.config_dir + 'p19_stat.pickle', 'rb') as fp:
            self.loaded_stat = pickle.load(fp) 
            
        
        self.labels = [d['target'] for d in self.data]
        
        if not self.load_all:
            self.data = self.data[:num_of_instances]
            self.labels = self.labels[:num_of_instances]
        
        print(f"{self.name} loaded , num of instances:", len(self.data))#, self.sample_idx)
        self.MAX_LEN = 4300
        self.MAX_timesteps = 350
        self.do_padding = padding

        with open(__file__ + '/Data/P19/sampling_rate_p19.npy', 'rb') as f:
            self.sampling_rate = np.load(f)

    def __len__(self):
        return len(self.data)
    
    def __labels__(self):
        # Counter({0: 10281, 1: 1707})
        return self.labels
    
    def __getitem__(self, idx):
        arr = self.data[idx]
        nfeatures = arr['diagnosis'].shape[1]
        y = self.labels[idx].tolist()
        y_modified = []
        curr_len = arr['length']
        t = arr['timestamps'].reshape(-1)
        static = arr['static']
        
        time_global = []
        type_global = []
        z_global = []
        delt_global = []
        
        lasttime_feat = {idx:[] for idx in range(self.num_features)}
        for ii, (row ,mask, tmp) in enumerate(zip(arr['diagnosis'],arr['masks'], t)):
            # iterating each row
            if ii < curr_len:
                
                nonzero_idx = [i for i,r in enumerate(mask) if r]
                #nonzero_idx non zero feature ids
                if nonzero_idx:
                    y_modified.append(y[ii])
                for nidx in nonzero_idx:
                    time_global.append(tmp)
                    type_global.append(nidx)
                    av = row[nidx]
                    if self.normalize_data:
                        mn,mx = self.loaded_stat[f'idx{nidx}']['min'],self.loaded_stat[f'idx{nidx}']['max']
                        if mn != mx:
                            normalized_av =  (av - mn) / (mx-mn)
                        else:
                            normalized_av = av
                        z_global.append(normalized_av)
                    elif self.standardize_data:
                        mean,std = self.loaded_stat[f'idx{nidx}']['mean'],self.loaded_stat[f'idx{nidx}']['std']
                        if std !=0:
                            normalized_av = (av - mean) / std
                        else:
                            normalized_av = av
                        z_global.append(normalized_av)
                    else:
                        z_global.append(av)

                    if len(lasttime_feat[nidx]) == 0:
                        delt_global.append(0)
                        lasttime_feat[nidx].append(tmp)
                    else:
                        lasttime_feat[nidx].append(tmp)
                        delt_global.append(tmp-lasttime_feat[nidx][-2])
                    
        y = y_modified
        current_y_len = len(y)
        if self.do_padding:
            diff_y_len = self.MAX_timesteps - current_y_len
            if diff_y_len > 0:
                y +=[-1]*diff_y_len
            else:
                y = y[:self.MAX_timesteps]

        curr_len = len(z_global)
        if self.do_padding:
            diff = self.MAX_LEN - curr_len
            if diff > 0:
                time_global += [-1]*diff
                type_global += [-1]*diff
                z_global += [-1]*diff
                delt_global += [-1]*diff
            else:
                time_global = time_global[:self.MAX_LEN]
                type_global = type_global[:self.MAX_LEN]
                z_global = z_global[:self.MAX_LEN]
                delt_global = delt_global[:self.MAX_LEN]
        
        stacked_x = torch.vstack((torch.tensor(time_global),
                                    torch.tensor(type_global), # indices of features (0,1, 2.... so on)
                                    torch.tensor(z_global), 
                                torch.tensor(delt_global)))
        
        
        return {
            'x' : stacked_x,
            'y' : torch.tensor(y),
            'lx': torch.tensor(curr_len),
            'ly': torch.tensor(current_y_len),
            #'pid': torch.tensor(pid),
            'static': torch.tensor(static)
        }