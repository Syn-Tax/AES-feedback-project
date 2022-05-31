#!/usr/bin/env python3

import pandas as pd
import sklearn
from sklearn import metrics
import numpy as np
import torch
import transformers
import wandb
import os
import tqdm

wandb.login()
wandb.init(project="AES-Experiment-1")

wandb_config = {
    "epochs": 100,
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

wandb.config = wandb_config

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

def compute_metrics(model_outputs, correct):
    max_error = metrics.max_error(correct, model_outputs)
    mse = metrics.mean_squared_error(correct, model_outputs)
    mae = metrics.mean_absolute_error(correct, model_outputs)
    r2 = metrics.r2_score(correct, model_outputs)

    return {
        "eval_max": max_error,
        "eval_mse": mse,
        "eval_mae": mae,
        "eval_r2": r2
    }

def train():
    tokenizer = transformers.AutoTokenizer.from_pretrained(wandb_config["save"])

    train_df, eval_df = load_data()


    model = transformers.AutoModelForSequenceClassification.from_pretrained(wandb_config["save"], num_labels=1)

    train_dataset, eval_dataset = process_data(train_df, eval_df)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=wandb_config["train_batch_size"])
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=wandb_config["eval_batch_size"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=wandb_config["lr"])

    num_training_steps = len(train_dataloader)*wandb_config["epochs"]
    lr_scheduler = transformers.get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    for epoch in range(wandb_config["epochs"]):
        print(f"Epoch number {epoch}")
        model.train()
        progress_bar = tqdm.auto.tqdm(range(num_training_steps))
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = torch.nn.MSELoss()(outputs.logits, batch["labels"])
            loss.backward()

            wandb.log({"train_loss": loss})

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

        # Eval at the end of every epoch
        print(f"\nEvaluating after epoch {epoch}")
        model.eval()
        progress_bar = tqdm.auto.tqdm(range(len(eval_dataloader)))
        output_logits = []
        output_labels = []
        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)

            logits = [float(logit) for logit in outputs.logits]
            [output_logits.append(logit) for logit in logits]

            [output_labels.append(float(label)) for label in batch["labels"]]

            progress_bar.update(1)

        metrics = compute_metrics(output_logits, output_labels)
        wandb.log(metrics)


    # Eval at the end of every epoch
    print(f"Final Evaluation")
    model.eval()
    progress_bar = tqdm.auto.tqdm(range(len(eval_dataloader)))
    output_logits = []
    output_labels = []
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = [float(logit) for logit in outputs.logits]
        [output_logits.append(logit) for logit in logits]

        [output_labels.append(float(label)) for label in batch["labels"]]

        progress_bar.update(1)

    metrics = compute_metrics(output_logits, output_labels)
    print()
    print(metrics)

    output_df = pd.DataFrame(list(zip(list(eval_df["text"]), output_logits, output_labels)))
    output_df.columns = ["text", "prediction", "truth"]

    output_df.to_csv(f"/content/drive/MyDrive/AES-feedback-project/Experiment-1/results/{wandb_config['model']}.csv")


if __name__ == "__main__":
    train()
