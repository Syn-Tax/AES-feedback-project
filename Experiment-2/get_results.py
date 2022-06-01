#!/usr/bin/env python3

import pandas as pd
import sklearn
from sklearn import metrics
import math
import numpy as np
import torch
import torchtext
import wandb
import os
import tqdm


class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {"IDs": torch.tensor(self.encodings[idx], dtype=torch.long), "labels": torch.tensor(self.labels[idx], dtype=torch.float32)}
        return item

    def __len__(self):
        return len(self.labels)

def mae_loss(output, target):
    mae = torch.mean(torch.abs(output - target))
    return mae

def mse_loss(output, target):
    mse = torch.mean((output - target)**2)
    return mse

def max_loss(output, target):
    max_error = torch.max(torch.abs(output - target))
    return max_error

def rmse_loss(output, target):
    rmse = torch.sqrt(mse_loss(output, target))
    return rmse

def load_data(path, eval_frac=0.1):
    df = pd.read_csv(path)

    train_df = df.iloc[int(df.shape[0]*eval_frac):]
    train_df.columns = ["text", "labels"]

    eval_df = df.iloc[:int(df.shape[0]*eval_frac)]
    eval_df.columns = ["text", "labels"]

    train_df = train_df.reset_index(drop=True)
    eval_df = eval_df.reset_index(drop=True)

    return train_df, eval_df
