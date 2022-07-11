#!/usr/bin/env python3

import os
import pdfplumber
import mark
import sys
from simpletransformers.classification import ClassificationModel
from process_pdf import process_pdf
import spacy
from collections import Counter


def main(model, nlp, path):
    sample = process_pdf(path)

    if not sample["Abstract"]:
        raise ValueError()

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

    if "BACKGROUND" not in percs.keys:
        percs["BACKGROUND"] = 0
    if "TECHNIQUE" not in percs.keys:
        percs["TECHNIQUE"] = 0
    if "RESULT" not in percs.keys:
        percs["RESULT"] = 0

    print(percs)
    return percs

if __name__ == "__main__":
    model_path = "outputs/best_model"
    model = ClassificationModel("bert", model_path)

    nlp = spacy.load("en_core_web_lg")

    if not os.path.isdir(sys.argv[1]):
        main(model, nlp, sys.argv[1])
    else:
        files = []
        for f in os.listdir(sys.argv[1]):
            if f.endswith(".pdf"):
                files.append(os.path.join(sys.argv[1], f))

        percs = []

        for f in files:
            try:
                percs.append(main(model, nlp, f))
            except: continue

        print(f"BACKGROUND: {sum([perc['BACKGROUND'] for perc in percs])/len(files)}")
        print(f"TECHNIQUE: {sum([perc['TECHNIQUE'] for perc in percs])/len(files)}")
        print(f"RESULT: {sum([perc['RESULT'] for perc in percs])/len(files)}")
