import docx2txt
import random
import re
import os
import codecs
import pandas as pd
import tika
tika.initVM()
from tika import parser

def get_marks(csv, name):
    name = name.split(" ")

    try:
        index = list(csv["Surname"]).index(name[-1])
    except:
        try:
            index = list(csv["Surname"]).index(name[0])
        except:
            return False
    
    return int(csv["Exp 4"][index])


def get_docx_text(file):
    try:
        text = codecs.decode(docx2txt.process(file).encode('utf-8'), 'utf-8').split("\n")
    except:
        return False

    text_without_spaces = []
    for line in text:
        if not (line.isspace() or line == "" or line.strip().isdecimal()):
            text_without_spaces.append(line.strip())
    
    # FIND NAME
    try:
        name_raw = text_without_spaces[text_without_spaces.index("Name and Group Number")+1]
        name = re.sub(r",* [gG]roup [0-9]*|| [0-9]*$", "", name_raw).strip()
    except:
        return False

    # FIND ANSWER
    try:
        start_index_raw = text_without_spaces.index("Reaction of tin metal with elemental iodine")
        if text_without_spaces[start_index_raw+2] in ["procedure", "procedure."]:
            start_index = start_index_raw+3
        else:
            start_index = start_index_raw+2

        try:
            end_index_raw = text_without_spaces.index("A) Calculate the % yield of SnI4 you obtained based on the amount of product you")
            if text_without_spaces[end_index_raw] == 'll':
                end_index_raw -= 1
            elif text_without_spaces[end_index_raw] == 'School of Chemistry':
                end_index_raw -= 2
        except:
            end_index_raw = text_without_spaces.index("A) Calculate the % yield of SnI4 you obtained based on the amount of product you isolated.")-1

    except:
        return False

    answer = " ".join(text_without_spaces[start_index:end_index_raw])

    return [name, answer]

def get_pdf_text(file):
    # text = str(textract.process(file))[3:-1].replace("\\r\\n", "\n").replace("\\n", "\n").split("\n")
    text = parser.from_file(file)["content"].split("\n")

    text_without_spaces = []
    for line in text:
        if not (line.isspace() or line == "" or line.strip().isdecimal()):
            text_without_spaces.append(line.strip())

    # FIND NAME
    try:
        name_raw = text_without_spaces[text_without_spaces.index("Name and Group Number")+1]
        name = re.sub(r",* [gG]roup [0-9]*|| [0-9]*$", "", name_raw).strip()
    except:
        return False

    # FIND ANSWER
    try:
        start_index_raw = text_without_spaces.index("Reaction of tin metal with elemental iodine")
        if text_without_spaces[start_index_raw+2] in ["procedure", "procedure."]:
            start_index = start_index_raw+3
        else:
            start_index = start_index_raw+2

        try:
            end_index_raw = text_without_spaces.index("A) Calculate the % yield of SnI4 you obtained based on the amount of product you")
            if text_without_spaces[end_index_raw] == 'll':
                end_index_raw -= 1
            elif text_without_spaces[end_index_raw] == 'School of Chemistry':
                end_index_raw -= 2
        except:
            end_index_raw = text_without_spaces.index("A) Calculate the % yield of SnI4 you obtained based on the amount of product you isolated.")-1

    except:
        return [False, file]

    answer = " ".join(text_without_spaces[start_index:end_index_raw])

    # return " ".join(text_without_spaces[6:-3]), name
    return [name, answer]

def main():
    folder = "Data/Raw Data/"

    files = os.listdir(folder)

    df = pd.read_csv("Data/Marks.csv")

    marks = []

    output = open("data.csv", "a")

    for file in files:
        if file.endswith(".pdf"):
            result = get_pdf_text(folder+file)
        else:
            result = get_docx_text(folder+file)

        if result: 
            mark = get_marks(df, result[0])
        else:
            continue

        if mark == False:
            continue
        else:
            try:
                output.write(f"\"{result[1]}\", {mark/100}\n") 
            except:
                continue
        

if __name__ == "__main__":
    main()
