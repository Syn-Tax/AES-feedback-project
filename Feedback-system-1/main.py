#!/usr/bin/env python3

import feedback
import process_pdf
import mark_full
import spacy
from simpletransformers.classification import ClassificationModel
import sys
import torch

def main(path):
    sample = process_pdf.process_pdf(path)

    nlp = spacy.load("en_core_web_lg")

    model_feedback = ClassificationModel("bert", "outputs/best_model")
    model_mark = torch.load("models/model-Abstract-mark.pt")

    abstract_mark = mark_full.mark_abstract(sample, model_mark)
    info_mark, correct_answers = mark_full.mark_info(sample, nlp)

    print(info_mark, correct_answers)

    percs = feedback.main(model_feedback, nlp, sample)

    for k, v in info_mark.items():
        if not v == 1:
            print(f"{k}: {v} marks")
            print(f"{correct_answers[k]}\nis the correct answer")
        else:
            print(f"{k}: {v} mark, Perfect!")

    print(f"ABSTRACT: {abstract_mark} marks")
    print(f"Background: {round(percs['BACKGROUND'], 2)*100}")
    print(f"Technique: {round(percs['TECHNIQUE'], 2)*100}")
    print(f"Result: {round(percs['RESULT'], 2)*100}")
    print("Often when there is a disproportionate percentage of background sentences, the abstract may not sound like it 'belongs in the paper' e.g. using 'The paper' instead of 'This paper'")

if __name__ == "__main__":
    main(sys.argv[1])
