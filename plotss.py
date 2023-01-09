import numpy as np
import pandas as pd
import statsmodels.api as sm
import plotly.graph_objects as go
import altair as alt
#python3.10 -u plotss.py

df = pd.read_csv('./datas/dataset1_per_class_ss.csv')

#changing column name
df.rename(columns = {'pubdate': 'date'}, inplace = True)
#removing day from the date column
df['date'] = pd.to_datetime(df['date'])
df['date'] = df['date'].apply(lambda x: x.strftime('%Y-%m')) #only year and month is left
df = df.melt('date')
df.rename(columns = {'variable': 'class', 'value': 'sentiment score'}, inplace = True)
print(df)

base = alt.Chart(df).mark_line().encode(
  x='date:T',
  y='sentiment score:Q',
  color='class'
).interactive(bind_y=False)

base.show()

