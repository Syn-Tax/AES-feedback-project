#!/usr/bin/env python3

import pandas as pd
import numpy as np
import math
import sys

def Zscore(value, mean, stdev):
    return (value - mean) / stdev

def min_max(value, maximum, minimum):
    return (value-minimum) / (maximum-minimum)

def median_MAD(value, median, MAD):
    return (value - median) / MAD

def tanh(value, mean, stdev):
    return 0.5 * (math.tanh(0.01 * ((value - mean) / stdev)) + 1)

def SAS_dataset(technique):


    valid_prompts = [1, 2, 5, 6, 10]

    for prompt in valid_prompts:
        sas_df = pd.read_csv("datasets/sas/train.tsv", sep='\t', header=0)
        sas_df = sas_df.drop("Id", axis=1).drop("Score2", axis=1)
        sas_df = sas_df[sas_df["EssaySet"].isin([prompt]) == True]
        sas_df = sas_df.reset_index(drop=True)

        sas_df["Score1"] = sas_df["Score1"].apply(lambda x: min_max(x, 3, 0))

        sas_df = sas_df.drop("EssaySet", axis=1)

        sas_df = sas_df[["EssayText", "Score1"]]
        sas_df.columns = ["text", "labels"]

        print(f"Prompt {prompt}: ", sas_df.shape[0])

        sas_df.to_csv(f"datasets/sas/data_prompt_{prompt}.csv", index=False, encoding="utf-8")

def AES_dataset(technique):
    score_column = "domain1_score"

    max_grades = [12, 6, 3, 3, 4, 4, 24, 60]
    min_grades = [2, 1, 0, 0, 0, 0, 2, 10]
    valid_prompts = [1, 2, 3, 4, 5, 6, 7, 8]

    for prompt in valid_prompts:
        aes_df = pd.read_csv("datasets/aes/train.tsv", sep='\t', header=0, encoding="ISO-8859-1")
        aes_df = aes_df[aes_df["essay_set"].isin([prompt]) == True]

        print(f"Prompt {prompt}: {min(list(aes_df[score_column]))}-{max(list(aes_df[score_column]))}")

        minimum = min_grades[prompt-1]
        maximum = max_grades[prompt-1]

        aes_df.loc[:, score_column] = aes_df.loc[:, score_column].apply(lambda x: min_max(x, maximum, minimum))

        aes_df = aes_df[["essay", score_column]]

        aes_df.columns = ["text", "labels"]

        print("Prompt {prompt} length:", aes_df.shape[0])

        aes_df.to_csv(f"datasets/aes/data_prompt_{prompt}.csv", index=False, encoding="utf-8")

if __name__ == "__main__":
    SAS_dataset("min_max")
    #AES_dataset("min_max")
