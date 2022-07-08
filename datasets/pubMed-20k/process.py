#!/usr/bin/env python3

import pandas as pd
from tqdm import tqdm

def load_data():
    with open("train.txt", "r") as f:
        file = f.read()

    dataset = []

    for line in tqdm(file.split("\n")):
        if line.startswith("#") or line == "":
            continue
        line = line.split("\t")
        sample = {"text": line[1]}
        if line[0] == "BACKGROUND":
            sample["label"] = 0
        elif line[0] == "OBJECTIVE":
            sample["label"] = 0
        elif line[0] == "METHODS":
            sample["label"] = 1
        elif line[0] == "RESULTS":
            sample["label"] = 2
        elif line[0] == "CONCLUSIONS":
            sample["label"] = 2
        else: print(line[0])

        dataset.append(sample)

    df = pd.DataFrame(dataset)
    print(df)
    df.to_csv("data.csv", index=False)

if __name__ == "__main__":
    load_data()
