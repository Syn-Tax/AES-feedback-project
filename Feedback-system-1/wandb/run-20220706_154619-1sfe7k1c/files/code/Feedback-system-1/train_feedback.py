#!/usr/bin/env python3

from simpletransformers.classification import ClassificationArgs, ClassificationModel
import pandas as pd
import sklearn.metrics
import wandb

wandb_config = {"epochs": 5, "train_batch_size": 64, "eval_batch_size": 64, "lr": 5e-5}

def load_data(eval_frac=0.1):
    df = pd.read_csv("../datasets/pubMed-20k/data.csv")

    df = df.sample(frac=1).reset_index(drop=True)

    train_df = df.iloc[int(df.shape[0]*eval_frac):]
    train_df.columns = ["text", "labels"]

    eval_df = df.iloc[:int(df.shape[0]*eval_frac)]
    eval_df.columns = ["text", "labels"]

    train_df = train_df.reset_index(drop=True)
    eval_df = eval_df.reset_index(drop=True)

    return train_df, eval_df

if __name__ == "__main__":
    model_args = ClassificationArgs()
    model_args.num_train_epochs = wandb_config["epochs"]
    model_args.train_batch_size = wandb_config["train_batch_size"]
    model_args.eval_batch_size = wandb_config["eval_batch_size"]
    model_args.wandb_project = "Feedback-System-1"
    model_args.learning_rate = wandb_config["lr"]
    model_args.no_save = True
    model_args.overwrite_output_dir = True
    model_args.logging_steps = 1
    model_args.evaluate_during_training = True
    model_args.use_eval_cached_features = True
    model_args.evaluate_during_training_verbose = True

    model = ClassificationModel("bert", "prajjwal1/bert-mini", num_labels=3, args=model_args)

    train_df, eval_df = load_data()

    model.train_model(train_df, eval_df=eval_df, acc=sklearn.metrics.accuracy_score)
