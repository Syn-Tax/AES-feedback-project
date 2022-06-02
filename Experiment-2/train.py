#!/usr/bin/env python3

import pandas as pd
import sklearn
from sklearn import metrics
import math
import numpy as np
import torch
import torchtext
import transformers
import wandb
import os
import tqdm
from model import SelfAttention

name = sys.argv[1]

wandb.init(project="AES-Experiment-2", name=name)

wandb.config = {
    "batch_size": 32,
    "epochs": 100,
    "lr": 1e-4,
    "hidden_size": 1024,
    "embedding_length": 300,
    "name": name
}


class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        # item = {"IDs": torch.tensor(self.encodings[idx], dtype=torch.long), "labels": torch.tensor(self.labels[idx], dtype=torch.float32)}
        item = {key: torch.tensor(val[idx], dtype=torch.long) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float32)
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

def process_data(df, tokenizer):
    texts = df["text"]
    labels = df["labels"]

    # tokenizer = torch.load("data/tokenizer.pt")
    # vocab = torch.load("data/vocab.pt")

    # encodings = [vocab(tokenizer(x)) for x in texts]
    #

    encodings = tokenizer(list(texts), padding=True, truncation=True)

    dataset = Dataset(encodings, labels)

    return dataset

def compute_metrics(model_outputs, correct):
    max_error = metrics.max_error(correct, model_outputs)
    mse = metrics.mean_squared_error(correct, model_outputs)
    mae = metrics.mean_absolute_error(correct, model_outputs)
    r2 = metrics.r2_score(correct, model_outputs)
    rmse = math.sqrt(mse)
    stddev = np.std(model_outputs)

    return {
        "eval_max": max_error,
        "eval_mse": mse,
        "eval_mae": mae,
        "eval_rmse": rmse,
        "eval_r2": r2,
        "eval_stddev": stddev
    }

def train():
    train_df, eval_df = load_data(f"/content/AES-feedback-project/Experiment-2/datasets/{name}/data.csv")

    tokenizer = transformers.BertTokenizerFast.from_pretrained("bert-base-cased")

    train_dataset = process_data(train_df, tokenizer)
    eval_dataset = process_data(eval_df, tokenizer)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, drop_last=True, batch_size=wandb.config["batch_size"])
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, drop_last=True, batch_size=wandb.config["batch_size"])

    model = SelfAttention(wandb.config["batch_size"], 1, wandb.config["hidden_size"], tokenizer.vocab_size, wandb.config["embedding_length"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=wandb.config["lr"])

    num_training_steps = len(train_dataloader)*wandb.config["epochs"]

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)


    for epoch in range(wandb.config["epochs"]):
        model.train()
        progress_bar = tqdm.auto.tqdm(range(len(train_dataloader)))
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(batch["input_ids"])

            loss = mse_loss(outputs, batch["labels"])
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            progress_bar.update(1)

        model.eval()
        output_logits = []
        output_labels = []
        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.no_grad():
                outputs = model(batch["input_ids"])

            logits = [float(logit) for logit in outputs]
            [output_logits.append(logit) for logit in logits]
            [output_labels.append(float(label)) for label in batch["labels"]]

        metrics = compute_metrics(output_logits, output_labels)
        wandb.log(metrics)

    torch.save(model, f"/content/drive/AES-feedback-project/Experiment-2/Models/model-{name}.pt")


if __name__ == "__main__":
    train()
