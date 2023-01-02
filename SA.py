# sentiment analysis
# objectves: 
# 1. find the highest mentioned topics in each month(among all ESG classes) 
# 2 ,and find out the corresponding sentiment 
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
import pandas as pd
from json import loads

df = pd.read_pickle("./datas/AC2022_set1.pk1")

#inspect the data 
#print(df['pubdate'][:10])
#columns = list(df.columns)

#data clean and orgainzing
df = df.sort_values(by='pubdate')
df.drop(df[df["content"].str.contains("歡迎各位益友加入成為頻道會員", na=False)].index, inplace = True)

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
df = df.sort_values(by='non_view_engagements', ascending = False)
print(df[['non_view_engagements','content', 'author_name']].head(50))

    # 1.3 run sentiment analysis on the extracted data, store it in temp obj maybe (overall)
    
#finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone',num_labels=3)
#tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
#nlp = pipeline("sentiment-analysis", model=finbert, tokenizer=tokenizer)



# step2 link w/ HSI 
    # 2.1 find monthly HSI 
    # 2.2 for each month
        # 2.2.1 get the rate of change in HSI
        # 2.2.2 match the correponding hottest topic and sentiment 
        # 2.2.3 plot graph (HSI verse sentiment)
    # 2.3 find the most freq combination (+/-)
    
