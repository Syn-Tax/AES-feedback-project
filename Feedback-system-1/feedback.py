#!/usr/bin/env python3

import os
import pdfplumber
import mark
import sys
from simpletransformers.classification import ClassificationModel
from process_pdf import process_pdf
import spacy
from collections import Counter


def main(model, nlp):
    path = sys.argv[1]
    sample = process_pdf(path)

    tokens = nlp(sample["Abstract"])
    sentences = [sent.text.strip() for sent in tokens.sents]
    print()
    print()
    print(sentences)

    predictions, raw_outputs = model.predict(sentences)

    print(predictions)
    labels = ["BACKGROUND", "TECHNIQUE", "RESULT"]
    print([{sentences[i]: labels[predictions[i]]} for i in range(len(predictions))])

    percs = {}
    for key in Counter(predictions).keys():
        percs[labels[key]] = dict(Counter(predictions))[key]/len(predictions)

    print(percs)

if __name__ == "__main__":
    model_path = "outputs/best_model"
    model = ClassificationModel("bert", model_path)

    nlp = spacy.load("en_core_web_lg")

    main(model, nlp)
