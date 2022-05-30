import pandas as pd
from simpletransformers.classification import ClassificationArgs, ClassificationModel
from IPython.display import clear_output
import torch
import logging
import sklearn
import numpy as np
import os
import sys
import json
from google.colab import drive
drive.mount('/content/drive')

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# model_types = ["bert", "albert", "xlmroberta", "longformer"]
model_types = ["bert"]
# model_saves = ["roberta-base", "bert-base-cased", "albert-base-v2", "xlm-roberta-base", "allenai/longformer-base-4096"]
model_saves = ["bert-base-cased"]
# model_types = ["longformer"]
# model_saves = ["allenai/longformer-base-4096"]
# sample_sizes = [10, 20, 30, 40, 50]
sample_sizes = [50]
num_repeats = 5
eval_size = 50

def train(model_type, model_save, sample_size, df, repeat):
    wandb_config = {
        "epochs": 50,
        "train_batch_size": 8,
        "eval_batch_size": 4,
        "lr": 5e-5,
        "samples": sample_size,
        "max_seq_len": 512,
        "model": model_type,
        "save": model_save
    }


    train_df = df.iloc[:wandb_config["samples"], :]

    train_df.columns = ["text", "labels"]

    eval_df = df.iloc[50:50+eval_size, :]

    eval_df.columns = ["text", "labels"]

    model_args = ClassificationArgs()
    model_args.num_train_epochs = wandb_config["epochs"]
    model_args.eval_batch_size = wandb_config["eval_batch_size"]
    model_args.train_batch_size = wandb_config["train_batch_size"]
    model_args.wandb_project = "transformer-aes"
    model_args.wandb_kwargs = {"name": "{}-{}-{}".format(wandb_config["model"], wandb_config["samples"], repeat+1) }
    model_args.learning_rate = wandb_config["lr"]
    model_args.model = wandb_config["model"]
    model_args.samples = wandb_config["samples"]
    model_args.max_seq_length = wandb_config["max_seq_len"]
    model_args.regression = True
    model_args.no_save = True
    model_args.overwrite_output_dir = True
    model_args.logging_steps = np.ceil((wandb_config["samples"]/wandb_config["train_batch_size"]))
    model_args.evaluate_during_training = True
    # model_args.evaluate_during_training_verbose = True
    model_args.evaluate_during_training_steps = np.ceil((wandb_config["samples"]/wandb_config["train_batch_size"])*10)
    model_args.use_eval_cached_features = True

    model = ClassificationModel(
        wandb_config["model"],
        wandb_config["save"],
        num_labels=1,
        args=model_args
    )

    model.train_model(
        train_df,
        eval_df=eval_df,
        mse=sklearn.metrics.mean_squared_error,
        mae=sklearn.metrics.mean_absolute_error,
        r2=sklearn.metrics.r2_score,
        max=sklearn.metrics.max_error
    )

    result, model_outputs, wrong_predictions = model.eval_model(
        eval_df,
        mse=sklearn.metrics.mean_squared_error,
        mae = sklearn.metrics.mean_absolute_error,
        r2=sklearn.metrics.r2_score,
        max=sklearn.metrics.max_error
    )

    return result, wrong_predictions, model_outputs

def run():
    results = []
    for j in range(num_repeats):
        df = pd.read_csv("data.csv")
        df = df.sample(frac=1).reset_index(drop=True)
        result, wrong_predictions, model_outputs = train(model, model_saves[1], 50, df, j)
        result["model"] = model
        result["sample size"] = sample
        results.append(results)
        torch.cuda.empty_cache()
        clear_output()

        errors = [[wrong_predictions[0][i], float(pred), float(model_outputs[i])] for i, pred in enumerate(wrong_predictions[1])]
        errors_df = pd.DataFrame(errors).sort_values(1, ascending=False)
        errors_df.to_csv(f"/content/drive/MyDrive/AES-feedback-project/Experiment-1/results/{model}-{sample}-{j}.csv")

    return results

def save_results(results):
    with open("/content/drive/MyDrive/transformer-aem/results.json", "w") as f:
        f.write(json.dumps(results))

if __name__ == "__main__":
    results = run()
    save_results(results)
