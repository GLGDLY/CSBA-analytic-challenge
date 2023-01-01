# !/usr/bin/env python3
# -*- coding: utf-8 -*-
from transformers import BertTokenizer, BertForSequenceClassification, pipeline, MarianMTModel, MarianTokenizer
from transformers.pipelines.pt_utils import KeyDataset
from pandas import read_excel, read_pickle
from json import dumps
from typing import Literal
from datetime import datetime
from copy import deepcopy
from tqdm import tqdm
from datasets import Dataset

# downloads:
# https://developer.nvidia.com/cuda-downloads
# https://developer.nvidia.com/rdp/cudnn-archive
import torch

print(f"==== CUDA available: {torch.cuda.is_available()} ====")
if torch.cuda.is_available():
    torch.cuda.init()
    torch.cuda.empty_cache()
    device_index = 0
else:
    device_index = -1
device = torch.device("cuda")


class translator:
    _model_name = 'Helsinki-NLP/opus-mt-zh-en'
    _model = MarianMTModel.from_pretrained(_model_name).to(device)
    _tokenizer = MarianTokenizer.from_pretrained(_model_name)
    translate = pipeline("translation_zh_to_en", model=_model, tokenizer=_tokenizer, device=0, batch_size=16)


finbert = BertForSequenceClassification.from_pretrained("yiyanghkust/finbert-esg",
                                                        num_labels=4).to(device)
tokenizer = BertTokenizer.from_pretrained("yiyanghkust/finbert-esg")
_esg = pipeline("text-classification", model=finbert, tokenizer=tokenizer, device=0, batch_size=16)
print("==== load finish ====")

finbert1 = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-esg-9-categories',
                                                         num_labels=9).to(device)
tokenizer1 = BertTokenizer.from_pretrained('yiyanghkust/finbert-esg-9-categories')
_esg_9class = pipeline("text-classification", model=finbert1, tokenizer=tokenizer1, device=0, batch_size=16)

ESG_template = {"Environmental": 0, "Social": 0, "Governance": 0, "None": 0}
ESG_9class_template = {"Climate Change": 0, "Pollution & Waste": 0, "Corporate Governance": 0,
                       "Natural Capital": 0, "Product Liability": 0, "Human Capital": 0,
                       "Business Ethics & Values": 0, "Community Relations": 0, "Non-ESG": 0}


class Datas:
    def __init__(self):
        self.ESG = deepcopy(ESG_template)
        self.ESG_9class = deepcopy(ESG_9class_template)
        self.monthly_ESG = {}
        self.monthly_9class = {}


count = 0
processed_docid = {}
headline_data = Datas()
content_data = Datas()


def esg_stats(_type: Literal["headline", "content"], _row):
    esg = _row[f"{_type}_ESG"]
    esg_class = _row[f"{_type}_ESG_9class"]

    dataset = headline_data if _type == "headline" else content_data
    dataset.ESG[esg] += 1
    dataset.ESG_9class[esg_class] += 1

    _time: datetime = _row["pubdate"].to_pydatetime()
    _month = "%d-%02d" % (_time.year, _time.month)
    if _month not in dataset.monthly_ESG:
        dataset.monthly_ESG[_month] = deepcopy(ESG_template)
    dataset.monthly_ESG[_month][esg] += 1
    if _month not in dataset.monthly_9class:
        dataset.monthly_9class[_month] = deepcopy(ESG_9class_template)
    dataset.monthly_9class[_month][esg_class] += 1


def cache():
    with open(r"cache.txt", "w", encoding="utf-8") as cache_data:
        cache_data.write(f"[headline]\n\nESG:\n{dumps(headline_data.ESG)}\n\n"
                         f"ESG 9 class:\n{dumps(headline_data.ESG_9class)}\n\n"
                         f"monthly ESG:\n{dumps(headline_data.monthly_ESG)}\n\n"
                         f"monthly ESG 9 class:\n{dumps(headline_data.monthly_9class)}\n\n"
                         f"\n===\n"
                         f"[content]\n\nESG:\n{dumps(content_data.ESG)}\n\n"
                         f"ESG 9 class:\n{dumps(content_data.ESG_9class)}\n\n"
                         f"monthly ESG:\n{dumps(content_data.monthly_ESG)}\n\n"
                         f"monthly ESG 9 class:\n{dumps(content_data.monthly_9class)}\n\n"
                         f"\n===\n"
                         f"total processed: {count}")


for i in range(2):
    print(f"==== read data set {i + 1} ====")
    df = read_excel(rf"./datas/AC2022_set{i + 1}.xlsx", "Sheet1")
    # df = read_pickle(rf"./datas/AC2022_set{i + 1}.pk1")
    print(f"==== read data set {i + 1} finish ====")
    ds = Dataset.from_dict({"c": df["headline"].astype(str).tolist()})
    df["trans_headline"] = [out[0].get("translation_text")
                            for out in tqdm(translator.translate(KeyDataset(ds, "c"),
                                                                 truncation=True, max_length=512))]
    df.to_pickle(rf"./datas/AC2022_set{i + 1}.pk1")

    ds = Dataset.from_dict({"c": df["content"].astype(str).tolist()})
    df["trans_content"] = [out[0].get("translation_text")
                           for out in tqdm(translator.translate(KeyDataset(ds, "c"),
                                                                truncation=True, max_length=512))]
    df.to_pickle(rf"./datas/AC2022_set{i + 1}.pk1")

    ds = Dataset.from_dict({"c": df["trans_headline"].astype(str).tolist()})
    df["headline_ESG"] = [out.get("label")
                          for out in tqdm(_esg(KeyDataset(ds, "c"), truncation=True, max_length=512))]
    df.to_pickle(rf"./datas/AC2022_set{i + 1}.pk1")
    df["headline_ESG_9class"] = [out.get("label")
                                 for out in tqdm(_esg_9class(KeyDataset(ds, "c"), truncation=True, max_length=512))]
    df.to_pickle(rf"./datas/AC2022_set{i + 1}.pk1")

    ds = Dataset.from_dict({"c": df["trans_content"].astype(str).tolist()})
    df["content_ESG"] = [out.get("label")
                         for out in tqdm(_esg(KeyDataset(ds, "c"), truncation=True, max_length=512))]
    df.to_pickle(rf"./datas/AC2022_set{i + 1}.pk1")
    df["content_ESG_9class"] = [out.get("label")
                                for out in tqdm(_esg_9class(KeyDataset(ds, "c"), truncation=True, max_length=512))]
    df.to_pickle(rf"./datas/AC2022_set{i + 1}.pk1")
    print(f"==== process data set {i + 1} finish ====")

    for index, row in df.iterrows():
        count += 1
        # filter repeated data
        if row["docid"] in processed_docid:
            continue

        # process data
        try:
            esg_stats("headline", row)
            esg_stats("content", row)
            cache()
        except Exception:
            pass
    df.to_pickle(rf"./datas/AC2022_set{i + 1}.pk1")

headline_data.monthly_ESG = dict(sorted(headline_data.monthly_ESG.items()))
headline_data.monthly_9class = dict(sorted(headline_data.monthly_9class.items()))
content_data.monthly_ESG = dict(sorted(content_data.monthly_ESG.items()))
content_data.monthly_9class = dict(sorted(content_data.monthly_9class.items()))
cache()
print("==== datasets process finish ====")

print(f"[headline]\n\nESG:\n{dumps(headline_data.ESG)}\n\n"
      f"ESG 9 class:\n{dumps(headline_data.ESG_9class)}\n\n"
      f"monthly ESG:\n{dumps(headline_data.monthly_ESG)}\n\n"
      f"monthly ESG 9 class:\n{dumps(headline_data.monthly_9class)}\n\n"
      f"\n===\n"
      f"[content]\n\nESG:\n{dumps(content_data.ESG)}\n\n"
      f"ESG 9 class:\n{dumps(content_data.ESG_9class)}\n\n"
      f"monthly ESG:\n{dumps(content_data.monthly_ESG)}\n\n"
      f"monthly ESG 9 class:\n{dumps(content_data.monthly_9class)}\n\n"
      f"\n===\n"
      f"total processed: {count}")
