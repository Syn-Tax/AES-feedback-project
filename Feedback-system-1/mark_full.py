#!/usr/bin/env python3

import transformers
import torch
import pandas as pd
import json
import mark
import spacy
import pdfplumber
import sys
from process_pdf import process_pdf

def mark_abstract(sample, model):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model.to(device)
    tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")

    tokenized = tokenizer(str(sample), padding=True, truncation=True, max_length=512)
    tokenized = {key: torch.tensor([val], dtype=torch.long) for key, val in tokenized.items()}

    tokenized = {k: v.to(device) for k, v in tokenized.items()}

    with torch.no_grad():
        output = model(tokenized["input_ids"], 1)

    return int(output[0]*6)

def mark_info(doi, impact, cited, rsc, acs):
    with open("../datasets/Abstract-split/correct.json", "r") as f:
        correct_answers = json.loads(f.read())

    nlp = spacy.load("en_core_web_lg")

    return mark.mark({
        "DOI": doi,
        "Impact Factor": impact,
        "cited": cited,
        "RSC": rsc,
        "ACS": acs
    }, correct_answers, nlp)

if __name__ == "__main__":
    model = torch.load("models/model-Abstract-mark.pt")
    if not os.isdir(sys.argv[1]):
        sample = process_pdf(sys.argv[1])

        abstract_mark = mark_abstract(sample["Abstract"], model)
        info_mark = mark_info(sample["DOI"], sample["Impact Factor"], sample["cited"], sample["RSC"], sample["ACS"])
        print(info_mark+abstract_mark)
    else:
        files = []
        for f in os.listdir(sys.argv[1]):
            if f.endswith(".pdf"):
                files.append(f)

        for f in files:
            sample = process_pdf(f)

            abstract_mark = mark_abstract(sample["Abstract"], model)
            info_mark = mark_info(sample["DOI"], sample["Impact Factor"], sample["cited"], sample["RSC"], sample["ACS"])
            print(info_mark+abstract_mark)
