import pandas as pd
import numpy as np
from json import loads, dumps

with open("./datas/dataset1_sentiment_summary.txt", "r", encoding="utf-8") as f:
    datas = f.read().split("\n")
    sentiment = loads(datas[4])
df = pd.DataFrame(sentiment).T
score_dic = {'Negative' : 10, 'Neutral' : 5, 'Positive' : 0} #we can think about weighting
print(df)
#for each month, calculate average sentiment score
total_count = df['Positive'] + df['Negative'] + df['Neutral']
weighted_sum = df['Positive']*score_dic['Positive'] + df['Negative']*score_dic['Negative'] + df['Neutral']*score_dic['Neutral']
score_df = pd.DataFrame(weighted_sum/total_count, columns=['score'])

#print(np.array(df))
#score = np.average(np.array(df), axis = 1, weights=[1, 0.01, -1])
print(score_df)
#print(score_df)
