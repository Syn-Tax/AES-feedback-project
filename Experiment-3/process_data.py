#!/usr/bin/env python3

import pandas as pd
import numpy as np

def SAS_dataset():
    sas_df = pd.read_csv("datasets/sas/train.tsv", sep='\t', header=0)

    sas_df = sas_df.drop("Id", axis=1).drop("Score2", axis=1)

    valid_prompts = [1, 2, 5, 6, 10]
    sas_df = sas_df[sas_df["EssaySet"].isin(valid_prompts) == True]
    sas_df = sas_df.reset_index(drop=True)

    sas_df["Score1"] = sas_df["Score1"].apply(lambda x: x/3)
    sas_df = sas_df.drop("EssaySet", axis=1)

    sas_df = sas_df[["EssayText", "Score1"]]
    sas_df.columns = ["text", "labels"]

    print(sas_df.shape[0])

    sas_df.to_csv("datasets/sas/data.csv", index=False, encoding="utf-8")

def AES_dataset():
    aes_df = pd.read_csv("datasets/aes/train.tsv", sep='\t', header=0, encoding="ISO-8859-1")

    max_grades = [6, 1, 3, 3, 4, 4, 15, 1]
    valid_prompts = [1, 3, 4, 5, 6, 7]

    def apply_func(row):
        row["rater1_domain1"] = row["rater1_domain1"] / max_grades[row["essay_set"] - 1]
        return row

    aes_df = aes_df.apply(apply_func, axis=1)

    aes_df = aes_df[aes_df["essay_set"].isin(valid_prompts) == True]

    aes_df = aes_df[["essay", "rater1_domain1"]]

    aes_df.columns = ["text", "labels"]

    print(aes_df.shape[0])

    aes_df.to_csv("datasets/aes/data.csv", index=False, encoding="utf-8")

if __name__ == "__main__":
    SAS_dataset()
    AES_dataset()
