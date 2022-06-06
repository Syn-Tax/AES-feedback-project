#!/usr/bin/env python3

import pandas as pd
import numpy as np
import math
import sys

def Zscore(value, mean, stdev):
    return (value - mean) / stdev

def min_max(value, maximum):
    return value / maximum

def median_MAD(value, median, MAD):
    return (value - median) / MAD

def tanh(value, mean, stdev):
    return 0.5 * (math.tanh(0.01 * ((value - mean) / stdev)) + 1)

def SAS_dataset(technique):
    sas_df = pd.read_csv("datasets/sas/train.tsv", sep='\t', header=0)

    sas_df = sas_df.drop("Id", axis=1).drop("Score2", axis=1)

    valid_prompts = [1, 2, 5, 6, 10]
    sas_df = sas_df[sas_df["EssaySet"].isin(valid_prompts) == True]
    sas_df = sas_df.reset_index(drop=True)

    mean = sas_df["Score1"].mean()
    stdev = sas_df["Score1"].std()
    median = sas_df["Score1"].median()
    mad = sas_df["Score1"].mad()

    if technique == "Zscore":
        sas_df.loc["Score1"] = sas_df["Score1"].apply(lambda x: Zscore(x, mean, stdev))
    elif technique == "min_max":
        sas_df.loc["Score1"] = sas_df["Score1"].apply(lambda x: min_max(x, 3))
    elif technique == "median_MAD":
        sas_df.loc["Score1"] = sas_df["Score1"].apply(lambda x: median_MAD(x, median, mad))
    elif technique == "tanh":
        sas_df.loc["Score1"] = sas_df["Score1"].apply(lambda x: tanh(x, mean, stdev))
    else:
        raise ValueError("that is not a valid normalisation method")


    sas_df = sas_df.drop("EssaySet", axis=1)

    sas_df = sas_df[["EssayText", "Score1"]]
    sas_df.columns = ["text", "labels"]

    print(sas_df.shape[0])

    sas_df.to_csv(f"datasets/sas/data_{technique}.csv", index=False, encoding="utf-8")

def AES_dataset(technique):
    aes_df = pd.read_csv("datasets/aes/train.tsv", sep='\t', header=0, encoding="ISO-8859-1")
    score_column = "domain1_score"

    #max_grades = [6, 3, 3, 4, 4, 15]
    #valid_prompts = [1, 3, 4, 5, 6, 7]
    valid_prompts = [3]
    max_grades = [3]
    aes_df = aes_df[aes_df["essay_set"].isin(valid_prompts) == True]

    aes_list = [aes_df[aes_df["essay_set"].isin([x]) == True] for x in valid_prompts]

    for i, df in enumerate(aes_list):
        #df.loc[:, score_column] = df.loc[:, score_column].apply(lambda x: x-1)
        maximum = max_grades[i]
        mean = df[score_column].mean()
        stdev = df[score_column].std()
        median = df[score_column].median()
        mad = df[score_column].mad()

        print(mean, stdev)

        if technique == "Zscore":
            df.loc[:, score_column] = df.loc[:, score_column].apply(lambda x: Zscore(x, mean, stdev))
        elif technique == "min_max":
            df.loc[:, score_column] = df.loc[:, score_column].apply(lambda x: min_max(x, maximum))
        elif technique == "median_MAD":
            df.loc[:, score_column] = df.loc[:, score_column].apply(lambda x: median_MAD(x, median, mad))
        elif technique == "tanh":
            df.loc[:, score_column] = df.loc[:, score_column].apply(lambda x: tanh(x, mean, stdev))
        else:
            raise ValueError("that is not a valid normalisation method")

    aes_df = pd.concat(aes_list, ignore_index=True)

    aes_df = aes_df[["essay", score_column]]

    aes_df.columns = ["text", "labels"]

    print(aes_df.shape[0])


    aes_df.to_csv(f"datasets/aes/data_prompt_3_{technique}.csv", index=False, encoding="utf-8")

if __name__ == "__main__":
    try:
        technique = sys.argv[1]
        SAS_dataset(technique)
        AES_dataset(technique)
    except:
        techniques = ["Zscore", "min_max", "median_MAD", "tanh"]
        for technique in techniques:
            SAS_dataset(technique)
            AES_dataset(technique)
