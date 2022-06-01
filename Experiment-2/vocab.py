#!/usr/bin/env python3

import torch
import torchtext
import pandas as pd

def build_vocab():
    sas_df = pd.read_csv("datasets/sas/data.csv", header=0)
    aes_df = pd.read_csv("datasets/aes/data.csv", header=0)
    fine_df = pd.read_csv("datasets/fine-tune/data.csv", header=0)

    tokenizer = torchtext.data.utils.get_tokenizer("basic_english")
    data_list = list(sas_df["text"]) + list(aes_df["text"]) + list(fine_df["text"])

    def yield_tokens(iterator):
        for text in iterator:
            yield tokenizer(text)

    vocab = torchtext.vocab.build_vocab_from_iterator(yield_tokens(data_list))

    torch.save(vocab, "data/vocab.pt")
    torch.save(tokenizer, "data/tokenizer.pt")

if __name__ == "__main__":
    build_vocab()
