#!/usr/bin/env python3

import pdfplumber
import re
import requests
import json

def processing_error():
    raise ValueError("Unable to process pdf, Processing must be done manually")

def process_pdf(path):
    with pdfplumber.open(path) as pdf:
        chars = pdf.chars

    text = ''.join([char["text"] for char in chars]).strip()

    if text == "":
        processing_error()

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
    }

    if len(data["DOI"]) > 100:
        processing_error()

    x = requests.get(f"https://doi.org/api/handles/{data['DOI']}")
    if json.loads(x.text)["responseCode"] != 1:
        raise ValueError("DOI Does not Exist")

    if "doi.org" in data["DOI"]:
        data["DOI"] = data["DOI"][::data["DOI"].find(".org")+5]
    if data["DOI"].startswith("0"):
        data["DOI"] = "1"+data["DOI"]

    return data
