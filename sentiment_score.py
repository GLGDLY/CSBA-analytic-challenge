import pandas as pd
import numpy as np
from json import loads, dumps
from difflib import SequenceMatcher
from tqdm import tqdm

#python3.10 -u sentiment_score.py

#total Weighted average for each month
with open("./datas/dataset1_sentiment_summary.txt", "r", encoding="utf-8") as f:
    datas = f.read().split("\n")
    sentiment = loads(datas[4])
sentiment_df = pd.DataFrame(sentiment).T
score_dic = {'Negative' : 0, 'Neutral' : 5, 'Positive' : 10} #we can think about weighting
#for each month, calculate average sentiment score
total_count = sentiment_df['Positive'] + sentiment_df['Negative'] + sentiment_df['Neutral']
weighted_sum = sentiment_df['Positive']*score_dic['Positive'] + sentiment_df['Negative']*score_dic['Negative'] + sentiment_df['Neutral']*score_dic['Neutral']
score_df = pd.DataFrame(weighted_sum/total_count, columns=['score'])

score_df.to_csv('./datas/dataset1_agg_ss.csv')
#Weighted average for every classes for each month
df = pd.read_pickle('./datas/AC2022_set1_SA_revised.pk1')
df = df.sort_values(by='pubdate')
conditions = [(df['headline_sentiment'] == 'Negative'), (df['headline_sentiment'] == 'Neutral'), (df['headline_sentiment'] == 'Positive')]
values = [0, 5, 10]
df['sentiment_score'] = np.select(conditions, values)
df = df.set_index('pubdate')
df = df[['headline_ESG_9class', 'sentiment_score']]
#create dataframes for each 9class_esg
grouped = list(df.groupby(['headline_ESG_9class']))
#print(grouped[0][1])
result = {}
for i in range(len(grouped)):
    current = grouped[i][1]
    result[grouped[i][0]] = current.groupby(pd.Grouper(freq='M'))['sentiment_score'].mean()
#now we have created a dictionary with key as 9class_esg and values as series of weighted average for each month
result_df = pd.DataFrame.from_dict(result)
result_df.to_csv('./datas/dataset1_per_class_ss.csv')





"""
# pickle file
# http://43.136.55.86/AC2022_set1.pk1, http://43.136.55.86/AC2022_set2.pk1
df = pd.read_pickle('./datas/AC2022_set1.pk1')
#data wash 
#cutting ESG_9class to those whom influence HSI
cut_df = df[df["headline_ESG_9class"].isin(["Climate Change", "Pollution & Waste", "Product Liability", "Community Relations"])]
#cutting non_view_engaements less than 50
cut_df = cut_df[cut_df['non_view_engagements']>=50]
cut_df = cut_df.reset_index(drop=True)
filter_indexs = []
for row_index in tqdm(range(cut_df.shape[0])):
    base_headline = str(cut_df.iloc[row_index]["headline"])
    has_repeat = False
    for row in cut_df[row_index + 1:].iterrows():
        if SequenceMatcher(None, base_headline, str(row[1]["headline"])).ratio() >= 0.7:
            has_repeat = True
            break
    if has_repeat:
        filter_indexs.append(row_index)
cut_df.drop(filter_indexs, inplace=True)

print(cut_df)
cut_df.to_pickle(r"./datas/AC2022_set1_after_filer.pk1")
"""
