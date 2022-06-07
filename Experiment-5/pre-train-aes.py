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
from model import SelfAttention

name = "aes"

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

def load_data(path, eval_frac=0.1):
    df = pd.read_csv(path)

    df = df.sample(frac=1).reset_index(drop=True)

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

def train(technique=None):
    if technique:
        train_df, eval_df = load_data(f"datasets/{name}/data_prompt_3_{technique}.csv")
    else:
        train_df, eval_df = load_data(f"datasets/{name}/data.csv")

    tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")

    train_dataset = process_data(train_df, tokenizer)
    eval_dataset = process_data(eval_df, tokenizer)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, drop_last=True, batch_size=wandb.config["batch_size"])
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, drop_last=True, batch_size=wandb.config["batch_size"])

    # bert_config = transformers.BertConfig.from_pretrained(
    #     "bert-base-uncased",
    #     vocab_size=tokenizer.vocab_size,
    #     hidden_size=wandb.config["hidden_size"],
    #     num_hidden_layers=wandb.config["num_hidden_layers"],
    #     num_attention_heads=wandb.config["num_attention_heads"],
    #     intermediate_size=wandb.config["intermediate_size"],
    #     hidden_act=wandb.config["hidden_act"],
    #     hidden_dropout_prob=wandb.config["hidden_dropout_prob"],
    #     attention_probs_dropout_prob=wandb.config["attention_probs_dropout_prob"],
    #     classifier_dropout=wandb.config["classifier_dropout"]
    # )

    model = SelfAttention(wandb.config["batch_size"], 1, wandb.config["hidden_size"], tokenizer.vocab_size, wandb.config["embedding_length"])
    #model = transformers.AutoModelForSequenceClassification.from_pretrained("prajjwal1/bert-tiny", num_labels=1)
    #model = transformers.BertForSequenceClassification.from_pretrained("bert-base-uncased", config=bert_config)

    is_transformer = False

    optimizer = torch.optim.AdamW(model.parameters(), lr=wandb.config["lr"])

    num_training_steps = len(train_dataloader)*wandb.config["epochs"]
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)


    for epoch in range(wandb.config["epochs"]):
        print(f"############## EPOCH: {epoch} ################")
        model.train()
        progress_bar = tqdm.auto.tqdm(range(len(train_dataloader)))
        for i, batch in enumerate(train_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            output = model(batch["input_ids"])
            if is_transformer:
                outputs = output.logits
            else:
                outputs = output

            rmse = rmse_loss(outputs, batch["labels"])
            stdev = stdev_error(outputs, batch["labels"])
            r2 = r2_loss(outputs, batch["labels"])

            curr_step = epoch * len(train_dataloader) + i
            curr_frac = curr_step / num_training_steps

            if curr_frac < wandb.config["stdev_start"]:
                stdev_factor = wandb.config["stdev_start_coeff"]
            else:
                stdev_factor = math.exp(-wandb.config["stdev_coeff"]*(curr_frac - wandb.config["stdev_start"]))


            #loss = mse + ((wandb.config["epochs"]/(epoch+1))*stdev)
            loss = ((1-stdev_factor)*(rmse-r2)) + (stdev_factor * stdev)
            loss.backward()

            wandb.log({"train_loss": loss, "train_stdev": stdev, "train_rmse": rmse, "train_r2": r2, "stdev_factor": stdev_factor})

            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()
            progress_bar.update(1)

        model.eval()
        output_logits = []
        output_labels = []
        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.no_grad():
                output = model(batch["input_ids"])

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

    torch.save(model, f"model/model-{name}.pt")


    print("Final Evaluation")

    model.eval()
    output_logits = []
    output_labels = []
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad():
            output = model(batch["input_ids"])

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

    output_df.to_csv(f"results-aes-self_attention.csv", index=False)


if __name__ == "__main__":
    config = {
        "batch_size": 16,
        "epochs": 200,
        "lr":1e-4,
        "hidden_size": 512,
        "embedding_length": 128,
        # "num_hidden_layers": 8,
        # "num_attention_heads": 8,
        # "intermediate_size": 2048,
        # "hidden_act": "gelu",
        # "hidden_dropout_prob": 0.1,
        # "attention_probs_dropout_prob": 0.1,
        # "classifier_dropout": None,
        "name": name,
        "stdev_coeff": 1,
        "stdev_start": 0.2,
        "stdev_start_coeff": 1,
    }

    try:
        technique = sys.argv[1]
        run = wandb.init(project="AES-Experiment-5", name=f"{name}-{technique}-small-prompt3-msestd", config=config)
        train(technique=technique)
        run.finish()
    except:
        raise Exception("Provide a valid normalisation technique")
