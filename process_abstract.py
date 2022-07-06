#!/usr/bin/env python3

import pandas as pd
import pdfplumber
import os
import re
import requests
import json
from tqdm import tqdm

def main():
    directory = "raw_data/CH2601 abstracting exercise/2022/Sumbissions2022"

    files = []

    for root, directories, files_ in os.walk("raw_data/CH2601 abstracting exercise/2022/Submissions2022"):
        for file  in files_:
            if file.endswith(".txt"):
                continue
            files.append(file)

    marks_df = pd.read_csv("raw_data/CH2601 abstracting exercise/2022/marks_2022.csv")
    print(marks_df)
    output = []

    for i, file in enumerate(files[1::]):
            with pdfplumber.open(f"raw_data/CH2601 abstracting exercise/2022/Submissions2022/{file}") as pdf:
                chars = pdf.chars

            text = ''.join([char["text"] for char in chars]).strip()

            if text == "":
                continue

            student_id = file.split("-")[0]
            try:
                mark = float(marks_df.loc[marks_df["Student ID"] == int(student_id)]["Mark"])/10
            except: continue

            """
                "DOI": text_raw[text_raw.lower().find("doi assigned:")+14:start_index].strip(),
                "Journal": text[15:text.lower().find("impact factor", 20, len(text))-9].strip(),
                "Impact Factor": re.sub(r"\([^()]*\)", "", text[text.lower().find("impact factor", 20, len(text))+16:text.lower().find("title of paper", 30, len(text))]).strip().replace(",","").split(",")[0],
                "Title": text[text.lower().find("title of paper:")+16:text.lower().find("reference in rsc format:")].strip(),
                "RSC": text[text.lower().find("rsc format:")+12:text.lower().find("reference in acs format: ")].strip(),
                "ACS": text[text.lower().find("reference in acs format:")+25:text.lower().find("abstract")].strip(),
                "cited": re.sub(r"[^0-9]*" ,"", text[text.find("Number of times cited:")+23:text.find("Data taken")-1].strip().split(",")[0]),
                "Abstract": text[text.lower().find("abstract"):text.lower().find("number of times cited:")][text[text.lower().find("abstract:")::].find(":")::].strip(),
                "Total mark": mark
            """
            doi_index = text.lower().find("doi assigned:")+13
            journal_index = text.lower().find("journal title:")+14

            doi = text[doi_index:journal_index-14].strip()
            text = text[journal_index::]

            impact_index = text.lower().find("impact factor")+15
            journal = re.sub(r"[^a-zA-Z0-9\s]", "", text[:impact_index - 24]).strip().lower()
            text = text[impact_index::]

            title_index = text.lower().find("title of paper")+15
            impact = re.sub(r"2[0-9]{3}", "", re.sub(r"[^0-9\.]", " ",  text[:title_index-15])).strip()
            impact = impact.split()[0] if impact else "0"
            impact = re.sub(r"\.$", "", impact)
            if not impact:
                impact = 0
            else:
                impact = float(impact)

            text = text[title_index::]
            rsc_index = text.lower().find("rsc format")+10
            title = re.sub(r"[^a-zA-Z0-9-\(\)\[\]\s]", "", text[:rsc_index-25]).strip()

            text = text[rsc_index::]
            acs_index = text.lower().find("acs format")+10
            rsc = re.sub(r"^([0-9]*|\[[0-9]*\])", "", re.sub(r"^\:", "", text[:acs_index-23]).strip()).strip()

            text = text[acs_index::]
            abstract_index = text.lower().find("abstract")+7
            acs = re.sub(r"^([0-9]*|\([0-9]*\)|[0-9]*\))", "", re.sub(r"^\:", "", text[:abstract_index-7]).strip()).strip()

            text = text[abstract_index::]
            cited_index = text.lower().find("cited", text.lower().find("number of times cited") - 50, len(text))+6
            abstract = re.sub(r"^(\(.{0,20}\)|[0-9]* words|word count: [0-9]*)", "", text[:cited_index-22][text.find(":")+1::].strip()).strip()

            text = text[cited_index::]
            cited = re.sub(r"[\[\(\{].*[\]\)\}]", "", text)
            cited = re.sub(r"[^0-9]", " ", cited).strip()
            if cited:
                cited = int(cited.split()[0])
            else:
                cited = 0

            data = {
                "DOI": doi,
                "Impact Factor": impact,
                "RSC": rsc,
                "ACS": acs,
                "Abstract": abstract,
                "cited": cited,
                "Total mark": mark
            }

            if len(data["DOI"]) > 100:
                continue

            x = requests.get(f"https://doi.org/api/handles/{data['DOI']}")
            if json.loads(x.text)["responseCode"] != 1:
                continue

            if "doi.org" in data["DOI"]:
                data["DOI"] = data["DOI"][::data["DOI"].find(".org")+5]
            if data["DOI"].startswith("0"):
                data["DOI"] = "1"+data["DOI"]

            print(data)

            output.append(data)

    final_df = pd.DataFrame(output)

    print(final_df)
    final_df.to_csv("datasets/Abstract-split/data.csv", index=False)

if __name__ == "__main__":
    main()
