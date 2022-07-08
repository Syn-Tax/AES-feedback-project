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
import sys
import argparse
from model import Model

name = "pre-train"

argparser = argparse.ArgumentParser()
argparser.add_argument("--pre-batch-size", help="Training batch size", default=32, type=int)
argparser.add_argument("--pre-epochs", help="Number of training epochs", default=1, type=int)
argparser.add_argument("--final-batch-size", help="Training batch size", default=32, type=int)
argparser.add_argument("--final-epochs", help="Number of training epochs", default=1, type=int)
argparser.add_argument("--lr", help="Learning Rate", default=1e-4, type=float)
argparser.add_argument("--hidden", help="Transformer hidden size", default=512, type=int)
argparser.add_argument("--embedding", help="Transformer embedding length", default=128, type=int)
argparser.add_argument("--stdev-coeff", help="stardard deviation coefficient", default=0.6, type=float)
argparser.add_argument("--stdev-start", help="standard deviation starting fraction", default=0.2, type=float)
argparser.add_argument("--stdev-start-coeff", help="starting coefficient of standard deviation", default=1.0, type=float)
argparser.add_argument("--r2-coeff", help="coefficient of r2", default=0.0007, type=float)
argparser.add_argument("--path", help="path to model file", type=str)

from prettytable import PrettyTable

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    print(table)
    print(f"Total Trainable Params: {round(total_params/1000000, 1)}M")
    return total_params

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

def r2_loss(output, target):
    target_mean = torch.mean(target)
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - output) ** 2)
    r2 = 1 - ss_res / ss_tot
    return -r2

def stdev_error(output, target, unbiased=False):
    target_std = torch.std(target, unbiased=unbiased)
    output_std = torch.std(output, unbiased=unbiased)

    return torch.abs(target_std - output_std)

def load_data(path, eval_frac=0.15):
    aes_df = pd.read_csv("datasets/aes/data.csv")
    sas_df = pd.read_csv("datasets/sas/data.csv")
    fine_df = pd.read_csv("datasets/fine-tune/data.csv")

    df = pd.concat([aes_df, sas_df], ignore_index=True)

    df = df.sample(frac=1).reset_index(drop=True)

    pre_train_df = df.iloc[int(df.shape[0]*eval_frac):]
    pre_train_df.columns = ["text", "labels"]

    pre_eval_df = df.iloc[:int(df.shape[0]*eval_frac)]
    pre_eval_df.columns = ["text", "labels"]

    pre_train_df = pre_train_df.reset_index(drop=True)
    pre_eval_df = pre_eval_df.reset_index(drop=True)

    final_train_df = fine_df.iloc[int(fine_df.shape[0]*eval_frac):]
    final_train_df.columns = ["text", "labels"]

    final_eval_df = fine_df.iloc[:int(fine_df.shape[0]*eval_frac)]
    final_eval_df.columns = ["text", "labels"]

    final_train_df = final_train_df.reset_index(drop=True)
    final_eval_df = final_eval_df.reset_index(drop=True)


    return pre_train_df, pre_eval_df, final_train_df, final_eval_df

def process_data(df, tokenizer):
    texts = df["text"]
    labels = df["labels"]

    # tokenizer = torch.load("data/tokenizer.pt")
    # vocab = torch.load("data/vocab.pt")

    # encodings = [vocab(tokenizer(x)) for x in texts]
    #

    encodings = tokenizer(list(texts), padding=True, truncation=True, max_length=512)

    dataset = Dataset(encodings, labels)

    return dataset

def compute_metrics(model_outputs, correct):
    max_error = metrics.max_error(correct, model_outputs)
    mse = metrics.mean_squared_error(correct, model_outputs)
    mae = metrics.mean_absolute_error(correct, model_outputs)
    r2 = metrics.r2_score(correct, model_outputs)
    rmse = math.sqrt(mse)
    stddev = np.std(model_outputs)
    stdev_err = abs(np.std(model_outputs)-np.std(correct))

    return {
        "eval_max": max_error,
        "eval_mse": mse,
        "eval_mae": mae,
        "eval_rmse": rmse,
        "eval_r2": r2,
        "eval_stdev": stddev,
        "eval_stdev_error": stdev_err
    }

def calculate_loss(curr_frac, outputs, labels, start, start_coeff, stdev_coeff, r2_coeff):
    rmse = rmse_loss(outputs, labels)
    stdev = stdev_error(outputs, labels)
    r2 = r2_loss(outputs, labels)

    if curr_frac < start:
        stdev_factor = start_coeff
    else:
        stdev_factor = start_coeff * math.exp(-stdev_coeff*(curr_frac - start))

    loss = ((1-stdev_factor)*(rmse+r2*r2_coeff)) + (stdev_factor * stdev)

    return loss, stdev, rmse, r2, stdev_factor

def train(model, epochs, train_df, device, batch_size, optimizer, tokenizer, eval_df=None, eval_during_training=True, log_wandb=True, is_transformer=False):
    if eval_during_training and eval_df.empty:
        raise ValueError("No Eval dataloader supplied: disable 'eval_during_training' or supply 'eval_dataloader'")

    train_dataset = process_data(train_df, tokenizer)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, drop_last=False, batch_size=batch_size)

    num_training_steps = len(train_dataloader)*epochs
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer)

    for epoch in range(epochs):
        print(f"############## EPOCH: {epoch} ################")
        model.train()
        progress_bar = tqdm.auto.tqdm(range(len(train_dataloader)))
        for i, batch in enumerate(train_dataloader):

            batch = {k: v.to(device) for k, v in batch.items()}

            output = model(batch["input_ids"], len(batch["input_ids"]))


            if is_transformer:
                outputs = output.logits
            else:
                outputs = output

            curr_step = epoch * len(train_dataloader) + i
            curr_frac = curr_step / num_training_steps

            loss, stdev, rmse, r2, stdev_factor = calculate_loss(curr_frac,
                                  outputs, batch["labels"],
                                  wandb.config["stdev_start"],
                                  wandb.config["stdev_start_coeff"],
                                  wandb.config["stdev_coeff"],
                                  wandb.config["r2_coeff"])
            loss.backward()

            wandb.log({"train_loss": loss, "train_stdev": stdev, "train_rmse": rmse, "train_r2": r2, "stdev_factor": stdev_factor})

            optimizer.zero_grad()
            optimizer.step()
            lr_scheduler.step()
            progress_bar.update(1)

        if eval_during_training:
            evaluate(model, eval_df, tokenizer, device, batch_size, log_wandb=log_wandb, is_transformer=is_transformer)

    return model


def evaluate(model, eval_df, tokenizer, device, batch_size, log_wandb=True, is_transformer=False):
    eval_dataset = process_data(eval_df, tokenizer)
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, drop_last=False, batch_size=batch_size)

    model.eval()
    output_logits = []
    output_labels = []
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad():
            output = model(batch["input_ids"], len(batch["input_ids"]))

        if is_transformer:
            outputs = output.logits
        else:
            outputs = output

        logits = [float(logit) for logit in outputs]
        [output_logits.append(logit) for logit in logits]
        [output_labels.append(float(label)) for label in batch["labels"]]

    metrics = compute_metrics(output_logits, output_labels)
    print(metrics)
    wandb.log(metrics)

    output_df = pd.DataFrame(list(zip(list(eval_df["text"]), output_logits, output_labels)))
    output_df.columns = ["text", "prediction", "true"]

    return output_df

def train_model(args,technique=None):
    pre_train_df, pre_eval_df, final_train_df, final_eval_df = load_data(f"datasets/{name}/data.csv")

    tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")

    process_data(pre_train_df, tokenizer)
    process_data(final_train_df, tokenizer)
    process_data(pre_eval_df, tokenizer)
    process_data(final_eval_df, tokenizer)

    if args["path"]:
        model = torch.load(args["path"])
    else:
        model = Model(tokenizer.vocab_size, wandb.config["embedding_length"], wandb.config["hidden_size"])

    count_parameters(model)

    #sys.exit(0)

    is_transformer = False

    optimizer = torch.optim.AdamW(model.parameters(), lr=wandb.config["lr"])

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    model = train(model, wandb.config["pre-epochs"], pre_train_df, device, wandb.config["pre-batch_size"], optimizer, tokenizer, eval_df=pre_eval_df)

    torch.save(model, f"models/model-{name}-pretrained.pt")

    model = train(model, wandb.config["final-epochs"], final_train_df, device, wandb.config["final-batch_size"], optimizer, tokenizer, eval_df=final_eval_df)


    print("Final Evaluation")

    output_df = evaluate(model, final_eval_df, tokenizer, device, wandb.config["final-batch_size"])

    output_df.to_csv(f"results-aes-self_attention.csv", index=False)


if __name__ == "__main__":
    args = vars(argparser.parse_args())
    config = {
        "pre-batch_size": args["pre_batch_size"],
        "pre-epochs": args["pre_epochs"],
        "final-batch_size": args["final_batch_size"],
        "final-epochs": args["final_epochs"],
        "lr": args["lr"],
        "hidden_size": args["hidden"],
        "embedding_length": args["embedding"],
        "name": name,
        "stdev_coeff": args["stdev_coeff"],
        "stdev_start": args["stdev_start"],
        "stdev_start_coeff": args["stdev_start_coeff"],
        "r2_coeff": args["r2_coeff"]
    }

    technique = "min_max"
    run = wandb.init(project="AES-Experiment-7", config=config)
    train_model(args, technique=technique)
    run.finish()
