# MIMIC Code
1. Running the Code: `python code/main_MIMIC.py`
2. usage: main_MIMIC.py [-h] [--expno EXPNO] [--seed SEED] [--hidsz HIDSZ]
                    [--lr LR] [--bs BS] [--epochs EPOCHS]
                    [--load_all | --no-load_all] [--ninst NINST] 
                    [--cuda | --no-cuda] [--gpu_no GPU_NO]
                    [--if_es | --no-if_es] [--patience_es PATIENCE_ES] [--delta_es DELTA_ES]
                    [--if_scheduler | --no-if_scheduler] 
                    [--scheduler SCHEDULER] [--if_dropout | --no-if_dropout]
                    [--dropout DROPOUT] [--if_decay | --no-if_decay] 
                    [--norm | --no-norm] [--standard | --no-standard]
                    [--f_decay | --no-f_decay] [--f_only_ct | --no-f_only_ct] [--f_no_ct | --no-f_no_ct]
                    [--if_sr | --no-if_sr] [--if_relu | --no-if_relu] 
                    [--agg_by AGG_BY] [--subsample | --no-subsample]
                    [--top | --no-top] [--abl25 | --no-abl25] 
                    [--abl50 | --no-abl50] [--abl75 | --no-abl75]
3. optional arguments:
    - -h, --help            show this help message and exit
    - --expno EXPNO         The experiment number. Default "test"
    - --seed SEED           The see value. Default 2020
    - --hidsz HIDSZ         The hidden size of LSTM. Default 32
    - --lr LR               The learning rate. Default 0.0005
    - --bs BS               The batch size. Default 32
    - --epochs EPOCHS       Number of epochs. Default 1
    - --load_all, --no-load_all
                            Load all the training data. Default True (default: True)
    - --ninst NINST         The number of training instance to load. Default 32
    - --cuda, --no-cuda     True if using gpu. Default False (default: False)
    - --gpu_no GPU_NO       The number of gpu to use. Default 0
    - --if_es, --no-if_es   Use early stopping. Default True (default: True)
    - --patience_es PATIENCE_ES
                            Patience of early stopping. Default 5
    - --delta_es DELTA_ES   Delta of early stopping. Default 0.0
    - --if_scheduler, --no-if_scheduler
                        learning rate scheduler is applied. Default True (default: True)
    - --scheduler SCHEDULER
                        Threshold for early stopping. Default 0.5
    - --if_dropout, --no-if_dropout
                        Dropout hidden nodes. Default True (default: True)
    - --dropout DROPOUT     Dropout value. Default 0.3
    - --if_decay, --no-if_decay
                        Decay the hidden states. Default True (default: True)
    - --norm, --no-norm     Normalize the data. Default False (default: False)
    - --standard, --no-standard
                        Standardize the data. Default True (default: True)
    - --f_decay, --no-f_decay
                        Decay he final hidden state of each feature. Default True (default: True)
    - --f_only_ct, --no-f_only_ct
                        Use only summary state for final prediction. Default False (default: False)
    - --f_no_ct, --no-f_no_ct
                        Use only hidden states for final prediction. Default False (default: False)
    - --if_sr, --no-if_sr   Use smapling rate for decay. Default True (default: True)
    - --if_relu, --no-if_relu
                        Apply relu layer to the concat of ht and ct before final prediction. Default False (default: False)
    - --agg_by AGG_BY       aggregate function to use for summary state. Aggregate options are - "mean", "max", "attention".
                        Default "mean"
    - --subsample, --no-subsample
                        A subset of features are used for training. Default False (default: False)
    - --top, --no-top       If subsample is true, then top = True implies top 6 high sampling rate feature is used for all the
                        model training and testing otherwise bottom 6 high sampling rate feature is used. Default True
                        (default: True)
    - --abl25, --no-abl25   Ablation study with only 25 percent data. Default False (default: False)
    - --abl50, --no-abl50   Ablation study with only 50 percent data. Default False (default: False)
    - --abl75, --no-abl75   Ablation study with only 75 percent data. Default False (default: False)

# P12 or P19 Code
1. Running the P12 Code: `python code/main_P12.py`
2. Running the P19 Code: `python code/main_P19.py`
3. usage: main_P12.py [-h] [--expno EXPNO] [--seed SEED] [--hidsz HIDSZ] [--lr LR] [--bs BS] 
                   [--epochs EPOCHS]
                   [--load_all | --no-load_all] [--ninst NINST] [--cuda | --no-cuda] [--gpu_no GPU_NO] [--if_es | --no-if_es]
                   [--patience_es PATIENCE_ES] [--delta_es DELTA_ES] [--if_scheduler | --no-if_scheduler]
                   [--scheduler SCHEDULER] [--if_dropout | --no-if_dropout] [--dropout DROPOUT] [--if_decay | --no-if_decay]
                   [--if_static | --no-if_static] [--norm | --no-norm] [--standard | --no-standard] [--f_decay | --no-f_decay]
                   [--f_only_ct | --no-f_only_ct] [--f_no_ct | --no-f_no_ct] [--if_sr | --no-if_sr] [--if_relu | --no-if_relu]
                   [--agg_by AGG_BY]
4. optional arguments:
    -h, --help            show this help message and exit
    --expno EXPNO         The experiment number. Default "test"
    --seed SEED           The seed value. Default 2020
    --hidsz HIDSZ         The hidden size of LSTM. Default 32
    --lr LR               The learning rate. Default 0.0005
    --bs BS               The batch size. Default 32
    --epochs EPOCHS       Number of epochs. Default 1
    --load_all, --no-load_all
                        Load all the training data. Default True (default: True)
    --ninst NINST         The number of training instance to load. Default 32
    --cuda, --no-cuda     True if using gpu. Default False (default: False)
    --gpu_no GPU_NO       The number of gpu to use. Default 0
    --if_es, --no-if_es   Use early stopping. Default True (default: True)
    --patience_es PATIENCE_ES
                        Patience of early stopping. Default 5
    --delta_es DELTA_ES   Delta of early stopping. Default 0.0
    --if_scheduler, --no-if_scheduler
                        learning rate scheduler is applied. Default True (default: True)
    --scheduler SCHEDULER
                        Threshold for early stopping. Default 0.5
    --if_dropout, --no-if_dropout
                        Dropout hidden nodes. Default True (default: True)
    --dropout DROPOUT     Dropout value. Default 0.3
    --if_decay, --no-if_decay
                        Decay the hidden states. Default True (default: True)
    --if_static, --no-if_static
                        Use the static data. Default True (default: True)
    --norm, --no-norm     Normalize the data. Default False (default: False)
    --standard, --no-standard
                        Standardize the data. Default True (default: True)
    --f_decay, --no-f_decay
                        Decay he final hidden state of each feature. Default True (default: True)
    --f_only_ct, --no-f_only_ct
                        Use only summary state for final prediction. Default False (default: False)
    --f_no_ct, --no-f_no_ct
                        Use only hidden states for final prediction. Default False (default: False)
    --if_sr, --no-if_sr   Use smapling rate for decay. Default True (default: True)
    --if_relu, --no-if_relu
                        Apply relu layer to the concat of ht and ct before final prediction. Default False (default: False)
    --agg_by AGG_BY       aggregate function to use for summary state. Aggregate options are "mean", "max", "attention". Default "mean"