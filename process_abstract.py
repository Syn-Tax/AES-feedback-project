#!/usr/bin/env python3

import pandas as pd
import pdfplumber
import os

def get_2021():
    pass

def get_2022():
    files = []

    for root, directories, files_ in os.walk("raw_data/CH2601 abstracting exercise/2022/Submissions2022"):
        for file  in files_:
            if file.endswith(".txt"):
                continue
            files.append(file)

    marks_df = pd.read_csv("raw_data/CH2601 abstracting exercise/2022/marks_2022.csv")
    print(marks_df)
    final_df = pd.DataFrame(columns=["text", "label"])

    for i, file in enumerate(files):
        try:
            with pdfplumber.open(f"raw_data/CH2601 abstracting exercise/2022/Submissions2022/{file}") as pdf:
                chars = pdf.chars

            text = ''.join([char["text"] for char in chars])

            start_index = text.find("Journal Title: ")
            end_index = text.find("Number of times cited: ")
            text = text[start_index:end_index].strip()
            if text == "":
                continue
            print(text)
            student_id = file.split("-")[0]
            print(student_id)
            mark = float(marks_df.loc[marks_df["Student ID"] == int(student_id)]["Mark"])/10
            print(mark)
            final_df.loc[i] = [text, mark]
        except:
            continue

    print(final_df)
    final_df.to_csv("2022.csv", index=False)

if __name__ == "__main__":
    get_2022()
