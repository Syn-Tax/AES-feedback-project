#!/usr/bin/env python3

import re
import sys
import os
import pandas as pd
import spacy
from tqdm import tqdm


def detect_correct(df):
    correct_answers = {}

    for doi in list(df["DOI"].unique()):
        correct_answers[doi] = {
            "Impact Factor": float(df[df["DOI"] == doi]["Impact Factor"].value_counts().idxmax()),
            "cited": int(df[df["DOI"] == doi]["cited"].value_counts().idxmax()),
            "RSC": str(df[df["DOI"] == doi]["RSC"].value_counts().idxmax()),
            "ACS": str(df[df["DOI"] == doi]["ACS"].value_counts().idxmax())
        }

    return correct_answers

def mark_numerical(value, correct, lenience_1=0.15, lenience_half=0.25):
    if value == correct:
        return 1

    perc_diff = abs(value - correct)/((value + correct)/2)

    if perc_diff < lenience_1:
        return 1
    elif perc_diff < lenience_half:
        return 0.5
    else:
        return 0

def mark_text(value, correct, nlp, lenience_1=0.85, lenience_half=0.6):
    value_nlp = nlp(value)
    correct_nlp = nlp(correct)

    similarity = value_nlp.similarity(correct_nlp)

    if similarity > lenience_1:
        return 1
    elif similarity > lenience_half:
        return 0.5
    else:
        return 0

def mark(sample, correct_answers, nlp):
    correct = correct_answers[sample["DOI"]]

    impact_factor = mark_numerical(sample["Impact Factor"], correct["Impact Factor"])
    cited = mark_numerical(sample["cited"], correct["cited"])
    rsc = mark_text(sample["RSC"], correct["RSC"], nlp)
    acs = mark_text(sample["ACS"], correct["ACS"], nlp)

    return impact_factor + cited + rsc + acs

if __name__ == "__main__":
    df = pd.read_csv("../datasets/Abstract-split/data.csv")
    correct_answers = detect_correct(df)

    nlp = spacy.load("en_core_web_lg")

    marks = []
    abstract_marks = []

    for i in tqdm(range(df.shape[0])):
        sample_mark = mark(df.loc[i], correct_answers, nlp)/10
        marks.append(sample_mark)
        abstract_marks.append(df.loc[i]["Total mark"] - sample_mark)

    abstract = [i/max(abstract_marks) for i in abstract_marks]
    marks_normalised = [i/max(marks) for i in marks]

    df["Info mark"] = marks_normalised
    df["Abstract mark"] = abstract

    df.to_csv("../datasets/Abstract-split/data.csv", index=False)
    print(df)
    print(sum(marks)/len(marks))
    print(sum(abstract_marks)/len(abstract_marks))
    print(max(abstract_marks))
