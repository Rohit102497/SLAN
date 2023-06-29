import torch
import os
import numpy as np
import pathlib
import random

class EarlyStopping_AUPRC:
    """Early stops the training if AUPRC metric doesn't improve after a given patience."""
    def __init__(self, patience, verbose=False, delta=0, save_path='checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time AUPRC improved.
                            Default: 7
            verbose (bool): If True, prints a message for each auprc improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.auprc_max = 0
        self.delta = delta
        self.save_path = save_path
        os.makedirs(pathlib.Path(self.save_path).parent, exist_ok=True)

    def __call__(self, auprc, model):

        score = auprc

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(auprc, model)
        elif auprc < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping AUPRC counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(auprc, model)
            self.counter = 0

    def save_checkpoint(self, auprc, model):
        """Saves model when auprc value increases."""
        if self.verbose:
            print(f'AUPRC increased ({self.auprc_max:.6f} --> {auprc:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.save_path)
        self.auprc_max = auprc

# ------------------------------Seeding------------------------------
# Seeding for reproducibility
def seed_everything(seed: int):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = True