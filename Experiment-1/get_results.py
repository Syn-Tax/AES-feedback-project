#!/usr/bin/env python3

import pandas as pd
import sklearn
import numpy as np
import torch
import transformers
import wandb
import os

wandb.login()

os.environ["WANDB_ENTITY"] = "syntax483"
os.environ["WANDB_PROJECT"] = "AES-Experiment-1"

wandb_config = {
    "epochs": 1,
    "train_batch_size": 8,
    "eval_batch_size": 4,
    "lr": 1e-4,
    "adam_beta_1": 0.9,
    "adam_beta_2": 0.999,
    "adam_epsilon": 1e-8,
    "lr_scheduler": "linear",
    "model": "bert",
    "save": "bert-base-cased"
}

def load_data(train_size=50, eval_size=50):
    df = pd.read_csv("/content/AES-feedback-project/Experiment-1/data.csv")
    df = df.sample(frac=1).reset_index(drop=True)

    train_df = df.iloc[:train_size]
    train_df.columns = ["text", "labels"]

    eval_df = df.iloc[eval_size:]
    eval_df.columns = ["text", "labels"]

    return train_df, eval_df

def configure_model(config):
    return transformers.TrainingArguments(
        output_dir="/content/output",
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        learning_rate=config["lr"],
        adam_beta1=config["adam_beta_1"],
        adam_beta2=config["adam_beta_2"],
        adam_epsilon=config["adam_epsilon"],
        lr_scheduler_type=config["lr_scheduler"],
        save_strategy="no"
    )

def tokenize(example, tokenizer):
    return tokenizer(example["text"], padding="max_length", truncation=True)

def train():
    tokenizer = transformers.AutoTokenizer.from_pretrained(wandb_config["save"])

    train_df, eval_df = load_data()
    train_tokenized = tokenize(train_df, tokenizer)
    eval_tokenized = tokenize(eval_df, tokenizer)

    training_args = configure_model(wandb_config)

    model = transformers.AutoModelForSequenceClassification.from_pretrained(wandb_config["save"], num_labels=1)

    train_dataloader = torch.utils.data.DataLoader(train_tokenized, batch_size=wandb_config["train_batch_size"])
    eval_dataloader = torch.utils.data.DataLoader(eval_tokenized, batch_size=wandb_config["eval_batch_size"])

    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataloader,
        eval_dataset=eval_dataloader
    )

    trainer.train()

if __name__ == "__main__":
    train()
