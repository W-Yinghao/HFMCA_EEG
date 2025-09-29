import os
import time
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from collections import Counter

# import wandb
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression as LR

from model import *
from loss import  HFMCA
from datasets.seed import SEED_ss, SEED_ssH, prepare_seed
from utils import SEEDLoaderH, SEEDLoader,get_bci4_cross, get_bci4_within

" Training codes will be released. "


if __name__ == '__main__':    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=60, help="number of epochs")
    parser.add_argument('--lr', type=float, default=3e-5, help="learning rate")
    parser.add_argument('--n_dim', type=int, default=64, help="hidden units")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="weight decay")
    parser.add_argument('--pretext', type=int, default=10, help="pretext subject")
    parser.add_argument('--training', type=int, default=10, help="training subject")
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--m', type=float, default=0.9995, help="moving coefficient")
    parser.add_argument('--model', type=str, default='HFMCA', help="which model")
    parser.add_argument('--dataset', type=str, default='SEED', help="dataset")
    parser.add_argument('--testsub', type=str, default='1', help="dataset")
    parser.add_argument('--steplr_point', type=int, default=20, help="seed 20, bci 200")
    parser.add_argument('--temperature',type=float,default=0.07)
    parser.add_argument('--alpha',type=float,default=1.0)
    parser.add_argument('--beta',type=float,default=0.5)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print ('device:', device)

    # set random seed
    seed = 2025
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
