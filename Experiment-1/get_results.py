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

class Report_Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx], dtype=torch.long) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float32)
        return item

    def __len__(self):
        return len(self.labels)


def load_data(train_size=50, eval_size=50):
    df = pd.read_csv("/content/AES-feedback-project/Experiment-1/data.csv")
    df = df.sample(frac=1).reset_index(drop=True)

    train_df = df.iloc[:train_size]
    train_df.columns = ["text", "labels"]

    eval_df = df.iloc[df.shape[0] - eval_size:]
    eval_df.columns = ["text", "labels"]
    eval_df = eval_df.reset_index()

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

def process_data(train_df, eval_df):
    train_texts = train_df["text"]
    train_labels = train_df["labels"]

    eval_texts = eval_df["text"]
    eval_labels = eval_df["labels"]

    tokenizer = transformers.AutoTokenizer.from_pretrained(wandb_config["save"])

    train_texts_encodings = tokenizer(list(train_texts), padding=True, truncation=True)
    eval_texts_encodings = tokenizer(list(eval_texts), padding=True, truncation=True)

    train_dataset = Report_Dataset(train_texts_encodings, train_labels)
    eval_dataset = Report_Dataset(eval_texts_encodings, eval_labels)

    return train_dataset, eval_dataset

def compute_metrics(eval_pred):
    logits, labels = eval_pred

    print(logits)
    print(labels)

def train():
    tokenizer = transformers.AutoTokenizer.from_pretrained(wandb_config["save"])

    train_df, eval_df = load_data()

    training_args = configure_model(wandb_config)

    model = transformers.AutoModelForSequenceClassification.from_pretrained(wandb_config["save"], num_labels=1)

    train_dataset, eval_dataset = process_data(train_df, eval_df)

    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()

if __name__ == "__main__":
    train()
