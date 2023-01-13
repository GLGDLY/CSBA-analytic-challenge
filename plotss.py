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

#new changes: choose statistically signficant variable only 
new_df = df.drop(['Corporate Governance', 'Natural Capital', 'Human Capital','Business Ethics & Values', 'Non-ESG'], axis=1)
#get the average of the remaining 4 data colume 
print(new_df[['Climate Change', 'Community Relations', 'Pollution & Waste',
       'Product Liability']].mean())

#turn the classes into one columns
df = df.melt('date')
df.rename(columns = {'variable': 'class', 'value': 'sentiment score'}, inplace = True)
#print(df)

base = alt.Chart(df).mark_line().encode(
  x='date:T',
  y='sentiment score:Q',
  color='class'
).interactive(bind_y=False)

base.show()


#creat chart for HSI 
HSI_df = pd.read_csv("./datas/HSI_1.csv")
#HSI_df.drop(['High','Low', 'Close', 'Adj Close', 'Volume'], axis=1)
#HSI_df = HSI_df.set_index(new_df.index)
#new_df['Open price'] = HSI_df.iloc[:,1].tolist()
open_close_color = alt.condition("datum.Open <= datum.Close",
                                 alt.value("#06982d"),
                                 alt.value("#ae1325"))

base = alt.Chart(HSI_df).encode(
    alt.X('Date:T',
          axis=alt.Axis(
              format='%m/%d',
              labelAngle=-45,
              title='Date'
          )
    ),
    color=open_close_color
)

rule = base.mark_rule().encode(
    alt.Y(
        'Low:Q',
        title='Price',
        scale=alt.Scale(zero=False),
    ),
    alt.Y2('High:Q')
)

bar = base.mark_bar().encode(
    alt.Y('Open:Q'),
    alt.Y2('Close:Q')
)

base1 = (rule + bar)

#plot line diagram, with stat sign var only 
new_df = new_df.melt('date')
new_df.rename(columns = {'variable': 'class', 'value': 'sentiment score'}, inplace = True)
#print(new_df)
base2 = alt.Chart(new_df).mark_line().encode(
  x='date:T',
  y='sentiment score:Q',
  color='class'
).interactive(bind_y=False)

#combine base 1 and 2 and show
alt.layer(base1, base2).resolve_scale(y='independent').show()
