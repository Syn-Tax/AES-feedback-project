#!/usr/bin/env python3

import os
import pdfplumber
import mark
import sys
from simpletransformers.classification import ClassificationModel
from process_pdf import process_pdf
import spacy
import json


def main():
    path = sys.argv[1]
    model_path = "outputs/best_model"

    nlp = spacy.load("en_core_web_lg")

    model = ClassificationModel("bert", model_path)

    sample = process_pdf(path)

    tokens = nlp(sample["Abstract"])

    sentences = [sent.string.strip() for sent in tokens.sents]
    print(sentences)
    sys.exit(0)

    predictions, raw_outputs = model.predict([sample["Abstract"]])

if __name__ == "__main__":
    main()
