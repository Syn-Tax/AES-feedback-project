import pandas as pd
from simpletransformers.classification import ClassificationArgs, ClassificationModel
import logging
import sklearn
import numpy as np
import os
import sys

logging.basicConfig(level=logging.ERROR)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

model_types = ["roberta", "bert", "albert", "xlmroberta", "longformer"]
model_saves = ["roberta-base", "bert-base-cased", "albert-base-v2", "xlm-roberta-base", "allenai/longformer-base-4096"]
sample_sizes = [10, 20, 30, 40, 50]

wandb_config = {"epochs": 100, "train_batch_size": 4, "eval_batch_size": 4, "lr": 5e-5, "samples": sample_sizes[int(sys.argv[1])], "max_seq_len": 512, "model": model_types[int(sys.argv[2])], "save": model_saves[int(sys.argv[2])]}

df = pd.read_csv("data.csv")

train_df = df.iloc[:wandb_config["samples"], :]

train_df.columns = ["text", "labels"]

eval_df = df.iloc[wandb_config["samples"]:, :]

eval_df.columns = ["text", "labels"]

model_args = ClassificationArgs()
model_args.num_train_epochs = wandb_config["epochs"]
model_args.eval_batch_size = wandb_config["eval_batch_size"]
model_args.train_batch_size = wandb_config["train_batch_size"]
model_args.wandb_project = "transformer-aes"
model_args.wandb_kwargs = {"name": "{}-{}".format(wandb_config["model"], wandb_config["samples"]) }
model_args.learning_rate = wandb_config["lr"]
model_args.model = wandb_config["model"]
model_args.samples = wandb_config["samples"]
# model_args.max_seq_length = wandb_config["max_seq_length"]
model_args.regression = True
model_args.no_save = True
model_args.overwrite_output_dir = True
model_args.logging_steps = 1
model_args.evaluate_during_training = True
model_args.evaluate_during_training_verbose = True
model_args.evaluate_during_training_steps = np.ceil((wandb_config["samples"]/wandb_config["train_batch_size"])*10)
model_args.use_eval_cached_features = True

model = ClassificationModel(wandb_config["model"], wandb_config["save"], num_labels=1, args=model_args)

model.train_model(train_df, eval_df=eval_df, mse=sklearn.metrics.mean_squared_error, mae=sklearn.metrics.mean_absolute_error, r2=sklearn.metrics.r2_score, max=sklearn.metrics.max_error)

result, model_outputs, wrong_predictions = model.eval_model(eval_df, mse=sklearn.metrics.mean_squared_error, mae = sklearn.metrics.mean_absolute_error, r2=sklearn.metrics.r2_score, max=sklearn.metrics.max_error)


file_exists = os.path.exists("results.txt")
with open("results.csv", "a") as f:
    if not file_exists:
        f.write("model,samples,"+",".join(result.keys()))

    output = "{},{}".format(wandb_config["model"], wandb_config["samples"])
    for key in result.keys():
        output += ",{}".format(result[key])
    f.write(output)
