#!/usr/bin/env python3

import os
import pdfplumber
import mark
import sys
from simpletransformers.classification import ClassificationModel
from process_pdf import process_pdf
import spacy
from collections import Counter


def main(model, nlp, sample):

    if not sample["Abstract"]:
        raise ValueError()

    tokens = nlp(sample["Abstract"])
    sentences = [sent.text.strip() for sent in tokens.sents]

    predictions, raw_outputs = model.predict(sentences)

    labels = ["BACKGROUND", "TECHNIQUE", "RESULT"]

    percs = {}
    for key in Counter(predictions).keys():
        percs[labels[key]] = dict(Counter(predictions))[key]/len(predictions)

    if "BACKGROUND" not in percs.keys():
        percs["BACKGROUND"] = 0
    if "TECHNIQUE" not in percs.keys():
        percs["TECHNIQUE"] = 0
    if "RESULT" not in percs.keys():
        percs["RESULT"] = 0

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
                sample = process_pdf(f)
                percs.append(main(model, nlp, sample))
                sys.exit(0)
            except KeyboardInterrupt:
                break
            except: continue

        print(f"BACKGROUND: {sum([perc['BACKGROUND'] for perc in percs])/len(percs)}")
        print(f"TECHNIQUE: {sum([perc['TECHNIQUE'] for perc in percs])/len(percs)}")
        print(f"RESULT: {sum([perc['RESULT'] for perc in percs])/len(percs)}")
        print(f"COUNT 2: {sum([1 if 0 in perc.values() else 0 for perc in percs])}")
        print(f"PERC 2: {sum([1 if 0 in perc.values() else 0 for perc in percs])/len(percs)}")
