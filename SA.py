# sentiment analysis
# objectives:
# 1. find the highest mentioned topics in each month(among all ESG classes) 
# 2 ,and find out the corresponding sentiment 
from transformers import BertTokenizer, BertForSequenceClassification
from transformers.pipelines.pt_utils import KeyDataset
from transformers import pipeline
from tqdm import tqdm
import pandas as pd
from json import loads, dumps
from datasets import Dataset
import torch
from typing import Literal
from copy import deepcopy

print(f"==== CUDA available: {torch.cuda.is_available()} ====")
if torch.cuda.is_available():
    torch.cuda.init()
    torch.cuda.empty_cache()
    device_index = 0
else:
    device_index = -1
device = torch.device("cuda")

df = pd.read_pickle("./datas/AC2022_set1.pk1")

#inspect the data
#print(df['pubdate'][:10])
#columns = list(df.columns)

#data clean and orgainzing
df = df.sort_values(by='pubdate')
df.drop(df[df["content"].str.contains("歡迎各位益友加入成為頻道會員", na=False)].index, inplace=True)

# step1 for each month
    # 1.1 find the highest mention topics from dataset1.stat
with open("./datas/dataset1_stats.txt", "r", encoding="utf-8") as f:
    datas = f.read().split("\n")
    headline_monthly_9ESG = loads(datas[12])
stat_df = pd.DataFrame(headline_monthly_9ESG).T
stat_df['max'] = stat_df.idxmax(axis="columns")
#print(stat_df)
#print(len(stat_df[stat_df['max']!='Community Relations']))

    # 1.2 locate and extract all obs (highest mentioned) in df (from pickle file), store it in temp obj maybe
df = df.sort_values(by='non_view_engagements', ascending=False)
#print(df[['non_view_engagements','content', 'author_name']].head(50))

    # 1.3 run sentiment analysis on the extracted data, store it in temp obj maybe (overall)

finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone', num_labels=3).to(device)
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
nlp = pipeline("sentiment-analysis", model=finbert, tokenizer=tokenizer, device=device_index, batch_size=16)
#print(df.columns)

ds = Dataset.from_dict({"c": df['trans_headline'].astype(str).tolist()})
df["headline_sentiment"] = [out.get("label") for out in tqdm(nlp(KeyDataset(ds, "c"), truncation=True, max_length=512))]

df.to_pickle("./datas/AC2022_set1_SA_revised.pk1")

sentiment_template = {"Positive": 0, "Neutral": 0, "Negative": 0}


class SentimentData:
    def __init__(self):
        self.sentiment = deepcopy(sentiment_template)
        self.monthly_sentiment = {}


count = 0
processed_docid = []
headline_data = SentimentData()
content_data = SentimentData()


def sentiment_stats(_type: Literal["headline", "content"], _row):
    sentiment = _row[f"{_type}_sentiment"]

    dataset = headline_data if _type == "headline" else content_data
    dataset.sentiment[sentiment] += 1

    _time = _row["pubdate"].to_pydatetime()
    _month = "%d-%02d" % (_time.year, _time.month)
    if _month not in dataset.monthly_sentiment:
        dataset.monthly_sentiment[_month] = deepcopy(sentiment_template)
    dataset.monthly_sentiment[_month][sentiment] += 1


def cache():
    with open(r"cache.txt", "w", encoding="utf-8") as cache_data:
        cache_data.write(f"[headline]\n\nsentiment:\n{dumps(headline_data.sentiment)}\n\n"
                         f"monthly sentiment:\n{dumps(headline_data.monthly_sentiment)}\n\n"
                         f"\n===\n"
                         f"total processed: {count}")


for index, row in df.iterrows():
    count += 1
    # filter repeated data
    if row["docid"] in processed_docid:
        continue
    processed_docid.append(row["docid"])

    # process data
    try:
        sentiment_stats("headline", row)
        # sentiment_stats("content", row)
        cache()
    except Exception:
        pass

headline_data.monthly_sentiment = dict(sorted(headline_data.monthly_sentiment.items()))
cache()
print("==== datasets process finish ====")
print(f"[headline]\n\nsentiment:\n{dumps(headline_data.sentiment)}\n\n"
      f"monthly sentiment:\n{dumps(headline_data.monthly_sentiment)}\n\n"
      f"\n===\n"
      f"total processed: {count}")


# step2 link w/ HSI
    # 2.1 find monthly HSI
    # 2.2 for each month
        # 2.2.1 get the rate of change in HSI
        # 2.2.2 match the correponding hottest topic and sentiment
        # 2.2.3 plot graph (HSI verse sentiment)
    # 2.3 find the most freq combination (+/-)

